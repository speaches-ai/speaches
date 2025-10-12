import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from speaches.audio import convert_audio_format
from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors import kokoro, piper
from speaches.executors.kokoro import KokoroModelManager
from speaches.executors.piper import PiperModelManager
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import strip_emojis, strip_markdown_emphasis

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-response_format
DEFAULT_RESPONSE_FORMAT = "mp3"

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
# https://platform.openai.com/docs/guides/text-to-speech/voice-options
OPENAI_SUPPORTED_SPEECH_VOICE_NAMES = ("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")

# https://platform.openai.com/docs/guides/text-to-speech/supported-output-formats
type ResponseFormat = Literal["mp3", "flac", "wav", "pcm"]
SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm")
SUPPORTED_NON_STREAMABLE_RESPONSE_FORMATS = ("flac", "wav")
UNSUPORTED_RESPONSE_FORMATS = ("opus", "aac")

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])


class CreateSpeechRequestBody(BaseModel):
    model: ModelId
    input: str
    """The text to generate audio for."""
    voice: str
    response_format: ResponseFormat = DEFAULT_RESPONSE_FORMAT
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
    speed: float = 1.0
    """The speed of the generated audio. 1.0 is the default. Different models have different supported speed ranges."""
    sample_rate: int | None = Field(None, ge=MIN_SAMPLE_RATE, le=MAX_SAMPLE_RATE)
    """Desired sample rate to convert the generated audio to. If not provided, the model's default sample rate will be used."""


# https://platform.openai.com/docs/api-reference/audio/createSpeech
# NOTE: `response_model=None` because `Response | StreamingResponse` are not serializable by Pydantic.
@router.post("/v1/audio/speech", response_model=None)
async def synthesize(  # noqa: C901
    executor_registry: ExecutorRegistryDependency,
    body: CreateSpeechRequestBody,
) -> Response | StreamingResponse:
    model_card_data = get_model_card_data_or_raise(body.model)

    body.input = strip_emojis(body.input)
    body.input = strip_markdown_emphasis(body.input)

    executor = find_executor_for_model_or_raise(body.model, model_card_data, executor_registry.text_to_speech)

    if isinstance(executor.model_manager, KokoroModelManager):
        if body.speed < 0.5 or body.speed > 2.0:
            raise HTTPException(
                status_code=422,
                detail=f"Speed must be between 0.5 and 2.0, got {body.speed}",
            )
        if body.voice not in [v.name for v in kokoro.VOICES]:
            if body.voice in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
                logger.warning(
                    f"Voice '{body.voice}' is not supported by the model '{body.model}'. It will be replaced with '{kokoro.VOICES[0].name}'. The behaviour of substituting OpenAI voices may be removed in the future without warning."
                )
                body.voice = kokoro.VOICES[0].name
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Voice '{body.voice}' is not supported. Supported voices: {kokoro.VOICES}",
                )
        with executor.model_manager.load_model(body.model) as tts:
            audio_generator = kokoro.generate_audio(
                tts,
                body.input,
                body.voice,
                speed=body.speed,
                sample_rate=body.sample_rate,
            )
            # these file formats can't easily be streamed because they have headers and/or metadata
            if body.response_format in SUPPORTED_NON_STREAMABLE_RESPONSE_FORMATS:
                audio_data = b"".join([audio_bytes async for audio_bytes in audio_generator])
                audio_data = convert_audio_format(
                    audio_data, body.sample_rate or kokoro.SAMPLE_RATE, body.response_format
                )
                return Response(audio_data, media_type=f"audio/{body.response_format}")
            if body.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(audio_bytes, body.sample_rate or kokoro.SAMPLE_RATE, body.response_format)
                    async for audio_bytes in audio_generator
                )
            return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")
    elif isinstance(executor.model_manager, PiperModelManager):
        if body.speed < 0.25 or body.speed > 4.0:
            raise HTTPException(
                status_code=422,
                detail=f"Speed must be between 0.25 and 4.0, got {body.speed}",
            )
        # TODO: maybe check voice
        with executor.model_manager.load_model(body.model) as piper_tts:
            # TODO: async generator
            audio_generator = piper.generate_audio(
                piper_tts, body.input, speed=body.speed, sample_rate=body.sample_rate
            )
            # these file formats can't easily be streamed because they have headers and/or metadata
            if body.response_format in SUPPORTED_NON_STREAMABLE_RESPONSE_FORMATS:
                audio_data = b"".join(audio_bytes for audio_bytes in audio_generator)
                audio_data = convert_audio_format(
                    audio_data, body.sample_rate or kokoro.SAMPLE_RATE, body.response_format
                )
                return Response(audio_data, media_type=f"audio/{body.response_format}")
            if body.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(
                        audio_bytes, body.sample_rate or piper_tts.config.sample_rate, body.response_format
                    )
                    for audio_bytes in audio_generator
                )
            return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")

    raise HTTPException(
        status_code=500,
        detail=f"Executor for model '{body.model}' exists but is not properly configured. This is a bug.",
    )
