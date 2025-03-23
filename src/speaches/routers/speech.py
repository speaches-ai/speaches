import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub.utils._cache_manager import _scan_cached_repo
from pydantic import BaseModel, Field

from speaches.audio import convert_audio_format
from speaches.dependencies import KokoroModelManagerDependency, PiperModelManagerDependency
from speaches.executors.kokoro import utils as kokoro_utils
from speaches.executors.piper import utils as piper_utils
from speaches.hf_utils import get_model_card_data_from_cached_repo_info, get_model_repo_path
from speaches.model_aliases import ModelId

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-response_format
DEFAULT_RESPONSE_FORMAT = "mp3"

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
# https://platform.openai.com/docs/guides/text-to-speech/voice-options
OPENAI_SUPPORTED_SPEECH_VOICE_NAMES = ("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")

# https://platform.openai.com/docs/guides/text-to-speech/supported-output-formats
type ResponseFormat = Literal["mp3", "flac", "wav", "pcm"]
SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm")
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
@router.post("/v1/audio/speech")
async def synthesize(
    piper_model_manager: PiperModelManagerDependency,
    kokoro_model_manager: KokoroModelManagerDependency,
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
    model_repo_path = get_model_repo_path(body.model)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    assert model_card_data is not None, cached_repo_info  # FIXME

    if kokoro_utils.hf_model_filter.passes_filter(model_card_data):
        if body.speed < 0.5 or body.speed > 2.0:
            raise HTTPException(
                status_code=422,
                detail=f"Speed must be between 0.5 and 2.0, got {body.speed}",
            )
        if body.voice not in [v.name for v in kokoro_utils.VOICES]:
            if body.voice in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
                logger.warning(
                    f"Voice '{body.voice}' is not supported by the model '{body.model}'. It will be replaced with '{kokoro_utils.VOICES[0].name}'. The behaviour of substituting OpenAI voices may be removed in the future without warning."
                )
                body.voice = kokoro_utils.VOICES[0].name
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Voice '{body.voice}' is not supported. Supported voices: {kokoro_utils.VOICES}",
                )
        with kokoro_model_manager.load_model(body.model) as tts:
            audio_generator = kokoro_utils.generate_audio(
                tts,
                body.input,
                body.voice,
                speed=body.speed,
                sample_rate=body.sample_rate,
            )
            if body.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(
                        audio_bytes, body.sample_rate or kokoro_utils.SAMPLE_RATE, body.response_format
                    )
                    async for audio_bytes in audio_generator
                )
            return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")
    elif piper_utils.hf_model_filter.passes_filter(model_card_data):
        if body.speed < 0.25 or body.speed > 4.0:
            raise HTTPException(
                status_code=422,
                detail=f"Speed must be between 0.25 and 4.0, got {body.speed}",
            )
        # TODO: maybe check voice
        with piper_model_manager.load_model(body.model) as piper_tts:
            # TODO: async generator
            audio_generator = piper_utils.generate_audio(
                piper_tts, body.input, speed=body.speed, sample_rate=body.sample_rate
            )
            if body.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(
                        audio_bytes, body.sample_rate or piper_tts.config.sample_rate, body.response_format
                    )
                    for audio_bytes in audio_generator
                )
            return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' is not supported. If you think this is a mistake, please open an issue.",
        )
