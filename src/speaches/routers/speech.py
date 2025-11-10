from collections.abc import Generator
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from speaches.api_types import (
    DEFAULT_SPEECH_RESPONSE_FORMAT,
    MAX_SPEECH_SAMPLE_RATE,
    MIN_SPEECH_SAMPLE_RATE,
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechAudioTokenUsage,
    SpeechResponseFormat,
)
from speaches.audio import Audio, stream_audio_as_formatted_bytes
from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.shared.handler_protocol import SpeechRequest
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import format_as_sse, strip_emojis, strip_markdown_emphasis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])

RESPONSE_FORMAT_MIME_TYPE_MAP = {
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
}


def response_format_to_mime_type(response_format: SpeechResponseFormat) -> str:
    mime_type = RESPONSE_FORMAT_MIME_TYPE_MAP[response_format]
    # Adding additional information to help client in decoding
    # Per https://voysis.readme.io/docs/audio-guidelines
    # NOTE: I'm not sure how widely supported these additional parameters are so for now they are commented out
    # if response_format == "pcm":
    #     mime_type += f";rate={audio.sample_rate}"
    #     mime_type += ";bits=16"
    #     mime_type += ";encoding=signed-int"
    #     mime_type += ";channels=1"
    #     mime_type += ";big-endian=false"
    return mime_type


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
@router.post("/v1/audio/speech")
def synthesize(
    executor_registry: ExecutorRegistryDependency,
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
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
    try:
        audio_generator = executor.model_manager.handle_speech_request(
            speech_request,
        )
        if body.stream_format == "sse":
            return StreamingResponse(
                speech_audio_events_to_sse(audio_gen_to_speech_audio_events(audio_generator)),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            stream_audio_as_formatted_bytes(
                audio_generator,
                audio_format=body.response_format,
                sample_rate=body.sample_rate,
            ),
            media_type=response_format_to_mime_type(body.response_format),
        )
    except ValueError as e:
        if "speed must be between" in str(e):
            logger.warning("Unsupported speed value requested for speech synthesis")
            raise HTTPException(status_code=422, detail=str(e)) from e
        raise
