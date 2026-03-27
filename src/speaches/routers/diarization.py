from __future__ import annotations

from collections.abc import Hashable, Iterator
import logging
from typing import TYPE_CHECKING, Annotated, Literal, cast

from fastapi import APIRouter, Form, Response
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel

from speaches.audio import Audio
from speaches.dependencies import AudioFileDependency, ExecutorRegistryDependency
from speaches.diarization import KnownSpeaker
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.utils import parse_data_url_to_audio

if TYPE_CHECKING:
    from pyannote.audio.core.pipeline import Pipeline
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
    from pyannote.core.segment import Segment
    from pyannote.core.utils.types import TrackName
    import torch

logger = logging.getLogger(__name__)
router = APIRouter()


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


def _map_to_known_speakers(
    pipeline: Pipeline,
    waveform: torch.Tensor,
    sample_rate: int,
    diarization: DiarizeOutput,
    known_speakers: list[KnownSpeaker],
) -> dict[Hashable, str]:
    from pyannote.audio import Inference
    import torch as _torch

    inference = Inference(pipeline._embedding, window="whole")  # noqa: SLF001
    main_audio = {"waveform": waveform, "sample_rate": sample_rate}

    # Compute embeddings for reference speakers
    known_embeddings: dict[str, np.ndarray] = {}
    for ks in known_speakers:
        ref_waveform = _torch.from_numpy(ks.audio.data).unsqueeze(0).float()
        known_embeddings[ks.name] = np.asarray(
            inference({"waveform": ref_waveform, "sample_rate": ks.audio.sample_rate})
        )

    # Collect embeddings per diarized speaker across all their turns
    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    speaker_track_gen = diarization.speaker_diarization.itertracks(yield_label=True)
    speaker_track_gen = cast("Iterator[tuple[Segment, TrackName, Hashable]]", speaker_track_gen)
    for turn, _, speaker in speaker_track_gen:
        try:
            emb = np.asarray(inference.crop(main_audio, turn))
            speaker_embeddings.setdefault(speaker, []).append(emb)  # pyrefly: ignore[no-matching-overload]
        except Exception:
            logger.exception(f"Failed to extract embedding for speaker {speaker} turn {turn}")

    avg_embeddings = {spk: np.mean(embs, axis=0) for spk, embs in speaker_embeddings.items() if embs}

    # Match each diarized speaker to the most similar known speaker via cosine similarity
    mapping: dict[Hashable, str] = {}
    for diarized_spk, diarized_emb in avg_embeddings.items():
        best_name = diarized_spk
        best_sim = -2.0
        for known_name, known_emb in known_embeddings.items():
            denom = float(np.linalg.norm(diarized_emb) * np.linalg.norm(known_emb))
            if denom < 1e-8:
                continue
            sim = float(np.dot(diarized_emb, known_emb) / denom)
            if sim > best_sim:
                best_sim = sim
                best_name = known_name
        mapping[diarized_spk] = best_name

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

    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput as _DiarizeOutput
    import torch as _torch

    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.diarization)

    with executor.model_manager.load_model(model) as pipeline:
        waveform = _torch.from_numpy(audio.data).unsqueeze(0).float()
        diarization = pipeline({"waveform": waveform, "sample_rate": audio.sample_rate})
        assert isinstance(diarization, _DiarizeOutput), f"Expected DiarizeOutput, got {type(diarization)}"

        speaker_mapping: dict[Hashable, str] | None = None
        if known_speakers:
            try:
                speaker_mapping = _map_to_known_speakers(
                    pipeline, waveform, audio.sample_rate, diarization, known_speakers
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
