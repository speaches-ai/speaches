from collections.abc import Hashable, Iterator
import logging
from typing import TYPE_CHECKING, Annotated, Literal, cast

from fastapi import APIRouter, Form, Response
from fastapi.responses import JSONResponse
import numpy as np
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pydantic import BaseModel
import torch

from speaches.audio import Audio, resample_audio_data
from speaches.dependencies import AudioFileDependency, ExecutorRegistryDependency
from speaches.diarization import KnownSpeaker, match_speakers_to_known
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.utils import parse_data_url_to_audio

if TYPE_CHECKING:
    from pyannote.core.segment import Segment
    from pyannote.core.utils.types import TrackName

logger = logging.getLogger(__name__)
router = APIRouter()


class DiarizationSegment(BaseModel):
    start: float
    """Start timestamp of the segment in seconds."""
    end: float
    """End timestamp of the segment in seconds."""
    speaker: str
    """Speaker label for this segment. When a known speaker reference matches this speaker above the similarity threshold, the label is the known speaker name. Otherwise speakers are labeled as SPEAKER_00, SPEAKER_01, etc."""


class DiarizationResponse(BaseModel):
    duration: float
    """Duration of the input audio in seconds."""
    segments: list[DiarizationSegment]
    """Diarization segments annotated with timestamps and speaker labels."""


def _map_to_known_speakers(
    pipeline: Pipeline,
    diarization: DiarizeOutput,
    known_speakers: list[KnownSpeaker],
    threshold: float,
) -> dict[Hashable, str]:
    # The community-1 pipeline exposes its speaker embedding model as a callable
    # PretrainedSpeakerEmbedding (cosine metric). It takes a (batch, channel, samples) tensor at
    # its own sample rate and returns a (batch, dimension) array. Note: it is NOT a pyannote Model,
    # so it cannot be wrapped in Inference().
    embedding = pipeline._embedding  # noqa: SLF001
    target_sample_rate = embedding.sample_rate

    # Embed each reference clip.
    known_embeddings: dict[str, np.ndarray] = {}
    for ks in known_speakers:
        try:
            data = ks.audio.data
            if ks.audio.sample_rate != target_sample_rate:
                data = resample_audio_data(data, ks.audio.sample_rate, target_sample_rate)
            ref_waveform = torch.from_numpy(data).float().reshape(1, 1, -1)
            emb = np.asarray(embedding(ref_waveform))[0]
        except Exception:
            logger.exception(f"Failed to compute embedding for known speaker {ks.name}")
            continue
        if np.isnan(emb).any():
            logger.warning(f"Embedding for known speaker {ks.name} contains NaN, skipping")
            continue
        known_embeddings[ks.name] = emb

    # Reuse the per-speaker centroid embeddings the pipeline already computed during clustering.
    # They are aligned with diarization.speaker_diarization.labels() and live in the same embedding
    # space as the reference embeddings, so no re-embedding of the main audio is needed.
    speaker_embeddings = diarization.speaker_embeddings
    if speaker_embeddings is None:
        return {}
    avg_embeddings: dict[Hashable, np.ndarray] = {}
    for index, label in enumerate(diarization.speaker_diarization.labels()):
        if index >= len(speaker_embeddings):
            break
        emb = np.asarray(speaker_embeddings[index])
        if not np.isnan(emb).any():
            avg_embeddings[label] = emb

    return match_speakers_to_known(avg_embeddings, known_embeddings, threshold)


@router.post(
    "/v1/audio/diarization",
    response_model=DiarizationResponse,
    responses={
        200: {
            "content": {
                "text/plain": {
                    "example": "SPEAKER uedkc 1 0.000 4.337 <NA> <NA> SPEAKER_03 <NA> <NA>\nSPEAKER uedkc 1 4.337 2.007 <NA> <NA> SPEAKER_00 <NA> <NA>\nSPEAKER uedkc 1 7.568 6.054 <NA> <NA> SPEAKER_03 <NA> <NA>",
                },
            },
        },
    },
)
def diarize_audio(
    executor_registry: ExecutorRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    known_speaker_names: Annotated[list[str] | None, Form(alias="known_speaker_names[]")] = None,
    known_speaker_references: Annotated[list[str] | None, Form(alias="known_speaker_references[]")] = None,
    known_speaker_threshold: Annotated[float, Form()] = 0.5,
    num_speakers: Annotated[int | None, Form()] = None,
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
    response_format: Annotated[Literal["json", "rttm"] | None, Form()] = "json",
) -> Response:
    known_speakers: list[KnownSpeaker] | None = None
    if known_speaker_names and known_speaker_references:
        known_speakers = [
            KnownSpeaker(
                name=name,
                audio=Audio(parse_data_url_to_audio(ref), sample_rate=16000),
            )
            for name, ref in zip(known_speaker_names, known_speaker_references, strict=True)
        ]

    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.diarization)

    with executor.model_manager.load_model(model) as pipeline:
        waveform = torch.from_numpy(audio.data).unsqueeze(0).float()
        diarization_kwargs: dict[str, int] = {}
        if num_speakers is not None:
            diarization_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers
        diarization = pipeline({"waveform": waveform, "sample_rate": audio.sample_rate}, **diarization_kwargs)
        assert isinstance(diarization, DiarizeOutput), f"Expected DiarizeOutput, got {type(diarization)}"

        speaker_mapping: dict[Hashable, str] | None = None
        if known_speakers:
            try:
                speaker_mapping = _map_to_known_speakers(pipeline, diarization, known_speakers, known_speaker_threshold)
            except Exception:
                logger.exception("Failed to map diarized speakers to known speakers, using default labels")

        speaker_track_gen = diarization.speaker_diarization.itertracks(yield_label=True)
        speaker_track_gen = cast("Iterator[tuple[Segment, TrackName, Hashable]]", speaker_track_gen)
        if response_format == "rttm":
            file_id = audio.name or "audio"
            lines: list[str] = []
            for turn, _, speaker in speaker_track_gen:
                label = speaker_mapping[speaker] if speaker_mapping else speaker
                duration = turn.end - turn.start
                lines.append(f"SPEAKER {file_id} 1 {turn.start:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>")
            return Response(content="\n".join(lines), media_type="text/plain")
        else:
            segments = []
            for turn, _, speaker in speaker_track_gen:
                label = speaker_mapping[speaker] if speaker_mapping else speaker
                segments.append(
                    DiarizationSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=label,  # pyrefly: ignore[bad-argument-type]
                    )
                )
            response = DiarizationResponse(duration=float(audio.duration), segments=segments)
            return JSONResponse(content=response.model_dump())
