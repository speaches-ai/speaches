from collections.abc import Generator
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from speaches.api_types import (
    DEFAULT_SPEECH_RESPONSE_FORMAT,
    MAX_SPEECH_SAMPLE_RATE,
    MIN_SPEECH_SAMPLE_RATE,
    SUPPORTED_STREAMABLE_SPEECH_RESPONSE_FORMATS,
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechAudioTokenUsage,
    SpeechResponseFormat,
)
from speaches.audio import Audio
from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.shared.handler_protocol import SpeechRequest
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import format_as_sse, strip_emojis, strip_markdown_emphasis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])


class CreateSpeechRequestBody(BaseModel):
    model: ModelId
    input: str
    """The text to generate audio for."""
    voice: str
    response_format: SpeechResponseFormat = DEFAULT_SPEECH_RESPONSE_FORMAT
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
    speed: float = 1.0
    """The speed of the generated audio. 1.0 is the default. Different models have different supported speed ranges."""
    stream_format: Literal["audio", "sse"] = "audio"
    """The format to stream the audio in. Supported formats are sse and audio"""
    sample_rate: int | None = Field(None, ge=MIN_SPEECH_SAMPLE_RATE, le=MAX_SPEECH_SAMPLE_RATE)
    """Desired sample rate to convert the generated audio to. If not provided, the model's default sample rate will be used."""


def audio_gen_to_speech_audio_events(
    audio_generator: Generator[Audio],
) -> Generator[SpeechAudioDeltaEvent | SpeechAudioDoneEvent]:
    for audio in audio_generator:
        yield SpeechAudioDeltaEvent(audio=audio.to_base64())
    # HACK: token usage is not tracked in any way yet
    yield SpeechAudioDoneEvent(token_usage=SpeechAudioTokenUsage(input_tokens=0, output_tokens=0, total_tokens=0))


def speech_audio_events_to_sse(
    speech_audio_events: Generator[SpeechAudioDeltaEvent | SpeechAudioDoneEvent],
) -> Generator[str]:
    for event in speech_audio_events:
        yield format_as_sse(event.model_dump_json())


# https://platform.openai.com/docs/api-reference/audio/createSpeech
# NOTE: `response_model=None` because `Response | StreamingResponse` are not serializable by Pydantic.
@router.post("/v1/audio/speech", response_model=None)
def synthesize(
    executor_registry: ExecutorRegistryDependency,
    body: CreateSpeechRequestBody,
) -> Response | StreamingResponse:
    model_card_data = get_model_card_data_or_raise(body.model)
    executor = find_executor_for_model_or_raise(body.model, model_card_data, executor_registry.text_to_speech)

    body.input = strip_emojis(body.input)
    body.input = strip_markdown_emphasis(body.input)

    speech_request = SpeechRequest(
        model=body.model,
        voice=body.voice,
        text=body.input,
        speed=body.speed,
    )
    # HACK: here we assume that the ValueError is only raised for invalid input data which may not be correct. This is a workaround for avoiding raising `HTTPException` from the executor code
    try:
        audio_generator = executor.model_manager.handle_speech_request(
            speech_request,
        )
    except ValueError as e:
        logger.exception("Value error during speech synthesis")
        raise HTTPException(status_code=422, detail=str(e)) from e
    if body.stream_format == "sse":
        return StreamingResponse(
            speech_audio_events_to_sse(audio_gen_to_speech_audio_events(audio_generator)),
            media_type="text/event-stream",
        )

    # HACK: some response formats may not directly map to soundfile formats
    # HACK: some response formats may not directly map to mime types
    if body.response_format not in SUPPORTED_STREAMABLE_SPEECH_RESPONSE_FORMATS:
        audio = Audio.concatenate(list(audio_generator))
        if body.sample_rate is not None:
            audio = audio.resample(body.sample_rate)
        return Response(
            content=audio.as_formatted_bytes(body.response_format),
            media_type=f"audio/{body.response_format}",
        )

    def resampled_audio_generator() -> Generator[bytes]:
        for audio in audio_generator:
            if body.sample_rate is not None:
                audio.resample(body.sample_rate)

            yield audio.as_formatted_bytes(body.response_format)

    return StreamingResponse(resampled_audio_generator(), media_type=f"audio/{body.response_format}")
