import logging
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    Response,
)
from fastapi.responses import JSONResponse
from onnx_diarization.clustering.plda import PLDATransform, load_xvex_and_plda_data
from onnx_diarization.embedding import WeSpeakerEmbeddingModel
from onnx_diarization.fbank import FbankExtractor
from onnx_diarization.pipeline import PyannnoteSegmentation, SpeakerDiarizationPipeline
from pydantic import BaseModel

from speaches.audio import Audio
from speaches.dependencies import AudioFileDependency, ExecutorRegistryDependency
from speaches.diarization import KnownSpeaker
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.utils import parse_data_url_to_audio

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)
router = APIRouter()

EMBEDDING_MODEL_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
SEGMENTATION_MODEL_ID = "fedirz/segmentation_community_1"


class DiarizationSegment(BaseModel):
    start: float
    """Start timestamp of the segment in seconds."""
    end: float
    """End timestamp of the segment in seconds."""
    speaker: str
    """Speaker label for this segment. When known speakers are provided, the label matches known_speaker_names[]. Otherwise speakers are labeled sequentially using capital letters (A, B, ...)."""


class DiarizationResponse(BaseModel):
    duration: float
    """Duration of the input audio in seconds."""
    segments: list[DiarizationSegment]
    """Diarization segments annotated with timestamps and speaker labels."""


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
    known_speaker_names: Annotated[list[str] | None, Form(alias="known_speaker_names[]")] = None,
    known_speaker_references: Annotated[list[str] | None, Form(alias="known_speaker_references[]")] = None,
    response_format: Annotated[Literal["json", "rttm"] | None, Form()] = "json",
) -> Response:
    known_speaker_references_audio_data = (
        [parse_data_url_to_audio(ref) for ref in known_speaker_references] if known_speaker_references else None
    )
    # HACK:: proper sample rate
    known_speaker_references_audio = (
        [Audio(audio_data, sample_rate=16000) for audio_data in known_speaker_references_audio_data]
        if known_speaker_references_audio_data
        else None
    )

    if known_speaker_references_audio and known_speaker_names:
        known_speakers = [
            KnownSpeaker(name=name, audio=ref_audio)
            for name, ref_audio in zip(known_speaker_names, known_speaker_references_audio, strict=True)
        ]
    else:
        known_speakers = None

    known_speakers_dict: dict[str, npt.NDArray] | None = None
    if known_speakers:
        known_speakers_dict = {speaker.name: speaker.audio.data for speaker in known_speakers}

    embedding_model_card_data = get_model_card_data_or_raise(EMBEDDING_MODEL_ID)
    embedding_executor = find_executor_for_model_or_raise(
        EMBEDDING_MODEL_ID, embedding_model_card_data, executor_registry.speaker_embedding
    )

    segmentation_model_card_data = get_model_card_data_or_raise(SEGMENTATION_MODEL_ID)
    segmentation_executor = find_executor_for_model_or_raise(
        SEGMENTATION_MODEL_ID, segmentation_model_card_data, executor_registry.speaker_segmentation
    )

    with (
        embedding_executor.model_manager.load_model(EMBEDDING_MODEL_ID) as embedding_model_sess,
        segmentation_executor.model_manager.load_model(SEGMENTATION_MODEL_ID) as segmentation_model_sess,
    ):
        fbank_extractor = FbankExtractor(sample_rate=16000)
        xvex_data, plda_data = load_xvex_and_plda_data()
        pipeline = SpeakerDiarizationPipeline(
            segmentation=PyannnoteSegmentation(
                session=segmentation_model_sess,
            ),
            embedding=WeSpeakerEmbeddingModel(
                session=embedding_model_sess,
                fbank_extractor=fbank_extractor,
            ),
            plda=PLDATransform(xvex_data, plda_data),
            embedding_batch_size=32,
        )

        diarization_results = pipeline(
            audio.data,
            file_id=audio.name,
            known_speakers=known_speakers_dict,
        )

        if response_format == "rttm":
            rttm_str = diarization_results.to_rttm()
            return Response(content=rttm_str, media_type="text/plain")
        else:
            segments = []
            for segment in diarization_results.itersegments():
                labels = diarization_results.get_labels(segment)
                if len(labels) == 0:
                    logger.warning(f"No label found for segment {segment}, assigning 'UNKNOWN'")
                    speaker = "UNKNOWN"
                else:
                    if len(labels) > 1:
                        logger.warning(f"Multiple labels found for segment {segment}: {labels}, using the first one")
                    speaker = next(iter(labels))
                segments.append(
                    DiarizationSegment(
                        start=segment.start,
                        end=segment.end,
                        speaker=speaker,  # pyright: ignore[reportArgumentType]
                    )
                )

            response = DiarizationResponse(
                duration=float(audio.duration),
                segments=segments,
            )
            return JSONResponse(content=response.model_dump())
