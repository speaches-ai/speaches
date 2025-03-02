# Resources:
# - https://github.com/snakers4/silero-vad

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Form,
)
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pydantic import BaseModel

from speaches.dependencies import AudioFileDependency  # noqa: TC001
from speaches.model_aliases import ModelId

# NOTE: this should match the default value in `decode_audio` which gets called by `AudioFileDependency`
SAMPLE_RATE = 16000
MS_SAMPLE_RATE = SAMPLE_RATE // 1000
MODEL_ID = "silero_vad_v5"


logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice-activity-detection"])


class SpeechTimestamp(BaseModel):
    start: int
    end: int


def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    for i in range(len(speech_timestamps)):
        speech_timestamps[i].start = speech_timestamps[i].start // MS_SAMPLE_RATE
        speech_timestamps[i].end = speech_timestamps[i].end // MS_SAMPLE_RATE
    return speech_timestamps


# TODO: adapt parameter names from here https://platform.openai.com/docs/api-reference/realtime-sessions/create#realtime-sessions-create-turn_detection
# TODO: use model manager
# TODO: use CudaExecutionProvider
@router.post("/v1/audio/speech/timestamps")
def detect_speech_timestamps(
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()] = MODEL_ID,
    threshold: Annotated[
        float,
        Form(
            ge=0,
            le=1,
            description="""Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH. It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.""",
        ),
    ] = 0.75,
    neg_threshold: Annotated[
        float | None,
        Form(
            ge=0,
            description="""Silence threshold for determining the end of speech. If a probability is lower
        than neg_threshold, it is always considered silence. Values higher than neg_threshold
        are only considered speech if the previous sample was classified as speech; otherwise,
        they are treated as silence. This parameter helps refine the detection of speech
         transitions, ensuring smoother segment boundaries.""",
        ),
    ] = None,
    min_speech_duration_ms: Annotated[
        int, Form(ge=0, description="""Final speech chunks shorter min_speech_duration_ms are thrown out.""")
    ] = 0,
    max_speech_duration_s: Annotated[
        float,
        Form(
            ge=0,
            description="""Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.""",
        ),
    ] = float("inf"),
    min_silence_duration_ms: Annotated[
        int,
        Form(
            ge=0,
            description="""In the end of each speech chunk wait for min_silence_duration_ms
        before separating it""",
        ),
    ] = 1000,
    speech_pad_ms: Annotated[
        int, Form(ge=0, description="""Final speech chunks are padded by speech_pad_ms each side""")
    ] = 0,
) -> list[SpeechTimestamp]:
    assert model == "silero_vad_v5", "Only 'silero_vad_v5' model is supported"
    vad_options = VadOptions(
        threshold=threshold,
        neg_threshold=neg_threshold,  # pyright: ignore[reportArgumentType]
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    speech_timestamps = to_ms_speech_timestamps(
        [
            SpeechTimestamp.model_validate(x)
            for x in get_speech_timestamps(audio, vad_options=vad_options, sampling_rate=SAMPLE_RATE)
        ]
    )
    return speech_timestamps
