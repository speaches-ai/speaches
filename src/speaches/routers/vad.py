# Resources:
# - https://github.com/snakers4/silero-vad


import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Form,
)

from speaches.dependencies import AudioFileDependency, ExecutorRegistryDependency
from speaches.executors.shared.handler_protocol import VadRequest
from speaches.executors.silero_vad_v5 import MODEL_ID, SAMPLE_RATE, SpeechTimestamp, VadOptions, to_ms_speech_timestamps
from speaches.model_aliases import ModelId

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice-activity-detection"])


# TODO: adapt parameter names from here https://platform.openai.com/docs/api-reference/realtime-sessions/create#realtime-sessions-create-turn_detection
@router.post("/v1/audio/speech/timestamps")
def detect_speech_timestamps(
    audio: AudioFileDependency,
    executor_registry: ExecutorRegistryDependency,
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
    assert model == MODEL_ID, f"Only '{MODEL_ID}' model is supported"

    vad_options = VadOptions(
        threshold=threshold,
        neg_threshold=neg_threshold,  # pyright: ignore[reportArgumentType]
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    vad_request = VadRequest(audio_data=audio, model_id=model, vad_options=vad_options, sampling_rate=SAMPLE_RATE)

    speech_timestamps_raw = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    speech_timestamps = to_ms_speech_timestamps(speech_timestamps_raw)
    return speech_timestamps
