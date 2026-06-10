from collections.abc import Hashable, Iterator
import logging
import os
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
from speaches.diarization import KnownSpeaker
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.utils import parse_data_url_to_audio

if TYPE_CHECKING:
    from pyannote.core.segment import Segment
    from pyannote.core.utils.types import TrackName

logger = logging.getLogger(__name__)
router = APIRouter()

# Sample rate expected by the speaker-embedding model.
EMBEDDING_SAMPLE_RATE = 16000

# Minimum cosine similarity required before a diarized speaker is mapped onto a
# known speaker. Below this value the speaker keeps its anonymous SPEAKER_XX
# label instead of being forced onto the nearest reference. Can be overridden
# per request via the `known_speaker_threshold` form field, or globally via the
# DIARIZATION_KNOWN_SPEAKER_THRESHOLD environment variable.
DEFAULT_KNOWN_SPEAKER_THRESHOLD = float(os.getenv("DIARIZATION_KNOWN_SPEAKER_THRESHOLD", "0.5"))


class DiarizationSegment(BaseModel):
    start: float
    """Start timestamp of the segment in seconds."""
    end: float
    """End timestamp of the segment in seconds."""
    speaker: str
    """Speaker label for this segment. When known speakers are provided, the label matches the known speaker name. Otherwise speakers are labeled as SPEAKER_00, SPEAKER_01, etc."""


class DiarizationResponse(BaseModel):
    duration: float
    """Duration of the input audio in seconds."""
    segments: list[DiarizationSegment]
    """Diarization segments annotated with timestamps and speaker labels."""


def _embed_waveform(embedding: object, data: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Compute a single speaker embedding for a mono waveform.

    `embedding` is the pipeline's embedding component (`pipeline._embedding`), a
    callable that expects a (batch, channel, samples) tensor at 16 kHz and
    returns a (batch, dimension) array. Returns a 1-D embedding, or None when the
    segment is empty / too short (the model yields NaN for sub-minimal segments).
    """
    if sample_rate != EMBEDDING_SAMPLE_RATE:
        data = resample_audio_data(data, sample_rate, EMBEDDING_SAMPLE_RATE)
    if data.size == 0:
        return None
    waveform = torch.from_numpy(np.ascontiguousarray(data)).float().reshape(1, 1, -1)
    result = np.asarray(embedding(waveform)).reshape(-1)  # type: ignore[operator]
    if result.size == 0 or not np.all(np.isfinite(result)):
        return None
    return result


def _map_to_known_speakers(
    pipeline: Pipeline,
    main_data: np.ndarray,
    sample_rate: int,
    diarization: DiarizeOutput,
    known_speakers: list[KnownSpeaker],
    min_similarity: float,
) -> dict[Hashable, str]:
    # In pyannote.audio 4.x the pipeline's embedding component is a directly
    # callable BaseInference object (e.g. PyannoteAudioPretrainedSpeakerEmbedding),
    # not a torch Model, so it must NOT be wrapped in Inference(...).
    embedding = pipeline._embedding  # noqa: SLF001

    # Compute embeddings for reference speakers
    known_embeddings: dict[str, np.ndarray] = {}
    for ks in known_speakers:
        emb = _embed_waveform(embedding, ks.audio.data, ks.audio.sample_rate)
        if emb is None:
            logger.warning("Reference audio for known speaker '%s' is too short/invalid; skipping", ks.name)
            continue
        known_embeddings[ks.name] = emb

    if not known_embeddings:
        logger.warning("No usable known-speaker references; keeping default speaker labels")
        return {}

    # Collect embeddings per diarized speaker across all their turns
    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    speaker_track_gen = diarization.speaker_diarization.itertracks(yield_label=True)
    speaker_track_gen = cast("Iterator[tuple[Segment, TrackName, Hashable]]", speaker_track_gen)
    for turn, _, speaker in speaker_track_gen:
        start = max(0, int(turn.start * sample_rate))
        end = min(len(main_data), int(turn.end * sample_rate))
        if end <= start:
            continue
        try:
            emb = _embed_waveform(embedding, main_data[start:end], sample_rate)
        except Exception:
            logger.exception(f"Failed to extract embedding for speaker {speaker} turn {turn}")
            continue
        if emb is not None:
            speaker_embeddings.setdefault(speaker, []).append(emb)  # pyrefly: ignore[no-matching-overload]

    avg_embeddings = {spk: np.mean(embs, axis=0) for spk, embs in speaker_embeddings.items() if embs}

    # Match each diarized speaker to the most similar known speaker via cosine similarity.
    # A match is only accepted when its similarity clears `min_similarity`; otherwise the
    # speaker keeps its anonymous SPEAKER_XX label. This prevents every voice from being
    # collapsed onto the nearest reference (e.g. labeling everyone as the single known speaker).
    mapping: dict[Hashable, str] = {}
    for diarized_spk, diarized_emb in avg_embeddings.items():
        best_candidate: Hashable = diarized_spk
        best_sim = -2.0
        for known_name, known_emb in known_embeddings.items():
            denom = float(np.linalg.norm(diarized_emb) * np.linalg.norm(known_emb))
            if denom < 1e-8:
                continue
            sim = float(np.dot(diarized_emb, known_emb) / denom)
            if sim > best_sim:
                best_sim = sim
                best_candidate = known_name

        if best_sim >= min_similarity:
            mapping[diarized_spk] = cast("str", best_candidate)
        else:
            # No reference is close enough -> keep the anonymous label.
            mapping[diarized_spk] = cast("str", diarized_spk)

        # Log the actual numbers so the threshold can be calibrated empirically.
        logger.info(
            "known-speaker match: %s -> best='%s' cosine=%.3f threshold=%.3f => labeled '%s'",
            diarized_spk,
            best_candidate,
            best_sim,
            min_similarity,
            mapping[diarized_spk],
        )

    return mapping


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
    known_speaker_threshold: Annotated[float | None, Form()] = None,
    num_speakers: Annotated[int | None, Form()] = None,
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
    response_format: Annotated[Literal["json", "rttm"] | None, Form()] = "json",
) -> Response:
    known_speakers: list[KnownSpeaker] | None = None
    if known_speaker_names and known_speaker_references:
        known_speakers = []
        for name, ref in zip(known_speaker_names, known_speaker_references, strict=True):
            ref_data, ref_sr = parse_data_url_to_audio(ref)
            known_speakers.append(KnownSpeaker(name=name, audio=Audio(ref_data, sample_rate=ref_sr)))

    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.diarization)

    with executor.model_manager.load_model(model) as pipeline:
        waveform = torch.from_numpy(audio.data).unsqueeze(0).float()

        # Optional bounds on the number of speakers. Supplying these (especially
        # min_speakers/max_speakers, or num_speakers when known) is the single most
        # effective way to stop the clustering step from merging distinct voices.
        pipeline_kwargs: dict[str, int] = {}
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            pipeline_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            pipeline_kwargs["max_speakers"] = max_speakers

        diarization = pipeline({"waveform": waveform, "sample_rate": audio.sample_rate}, **pipeline_kwargs)
        assert isinstance(diarization, DiarizeOutput), f"Expected DiarizeOutput, got {type(diarization)}"

        speaker_mapping: dict[Hashable, str] | None = None
        if known_speakers:
            threshold = (
                known_speaker_threshold
                if known_speaker_threshold is not None
                else DEFAULT_KNOWN_SPEAKER_THRESHOLD
            )
            try:
                speaker_mapping = _map_to_known_speakers(
                    pipeline, audio.data, audio.sample_rate, diarization, known_speakers, threshold
                )
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
