import logging
from typing import Annotated, Literal, Self

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, BeforeValidator, Field, model_validator

from speaches import kokoro_utils
from speaches.api_types import Voice
from speaches.audio import convert_audio_format
from speaches.dependencies import KokoroModelManagerDependency, PiperModelManagerDependency
from speaches.hf_utils import (
    get_kokoro_model_path,
    list_piper_models,
    read_piper_voices_config,
)

DEFAULT_MODEL_ID = "hexgrad/Kokoro-82M"
# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-response_format
DEFAULT_RESPONSE_FORMAT = "mp3"
DEFAULT_VOICE_ID = "af"  # TODO: make configurable

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-model
# https://platform.openai.com/docs/models/tts
OPENAI_SUPPORTED_SPEECH_MODEL = ("tts-1", "tts-1-hd")

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


def handle_openai_supported_model_ids(model_id: str) -> str:
    if model_id in OPENAI_SUPPORTED_SPEECH_MODEL:
        logger.warning(f"{model_id} is not a valid model name. Using '{DEFAULT_MODEL_ID}' instead.")
        return DEFAULT_MODEL_ID
    return model_id


ModelId = Annotated[
    Literal["hexgrad/Kokoro-82M", "rhasspy/piper-voices"],
    BeforeValidator(handle_openai_supported_model_ids),
    Field(
        description="The ID of the model",
        examples=["hexgrad/Kokoro-82M", "rhasspy/piper-voices"],
    ),
]


def handle_openai_supported_voices(voice_id: str) -> str:
    if voice_id in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
        logger.warning(f"{voice_id} is not a valid voice id. Using '{DEFAULT_VOICE_ID}' instead.")
        return DEFAULT_VOICE_ID
    return voice_id


VoiceId = Annotated[str, BeforeValidator(handle_openai_supported_voices)]  # TODO: description and examples


class CreateSpeechRequestBody(BaseModel):
    model: ModelId = DEFAULT_MODEL_ID
    input: str = Field(
        ...,
        description="The text to generate audio for. ",
        examples=[
            "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky. The rainbow takes the form of a multicoloured circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun. Rainbows can be caused by many forms of airborne water. These include not only rain, but also mist, spray, and airborne dew."
        ],
    )
    voice: VoiceId = DEFAULT_VOICE_ID
    """
For 'rhasspy/piper-voices' voices the last part of the voice name is the quality (x_low, low, medium, high).
Each quality has a different default sample rate:
- x_low: 16000 Hz
- low: 16000 Hz
- medium: 22050 Hz
- high: 22050 Hz
    """
    language: kokoro_utils.Language | None = None
    """
    Only used for 'hexgrad/Kokoro-82M' models. The language of the text to generate audio for.
    """
    response_format: ResponseFormat = Field(
        DEFAULT_RESPONSE_FORMAT,
        description=f"The format to audio in. Supported formats are {', '.join(SUPPORTED_RESPONSE_FORMATS)}. {', '.join(UNSUPORTED_RESPONSE_FORMATS)} are not supported",
        examples=list(SUPPORTED_RESPONSE_FORMATS),
    )
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
    speed: float = Field(1.0)
    """The speed of the generated audio. 1.0 is the default.
    For 'hexgrad/Kokoro-82M' models, the speed can be set to 0.5 to 2.0.
    For 'rhasspy/piper-voices' models, the speed can be set to 0.25 to 4.0.
    """
    sample_rate: int | None = Field(None, ge=MIN_SAMPLE_RATE, le=MAX_SAMPLE_RATE)
    """Desired sample rate to convert the generated audio to. If not provided, the model's default sample rate will be used.
    For 'hexgrad/Kokoro-82M' models, the default sample rate is 24000 Hz.
    For 'rhasspy/piper-voices' models, the sample differs based on the voice quality (see `voice`).
    """

    @model_validator(mode="after")
    def verify_voice_is_valid(self) -> Self:
        if self.model == "hexgrad/Kokoro-82M":
            assert self.voice in kokoro_utils.VOICE_IDS
        elif self.model == "rhasspy/piper-voices":
            assert self.voice in read_piper_voices_config()
        return self

    @model_validator(mode="after")
    def validate_speed(self) -> Self:
        if self.model == "hexgrad/Kokoro-82M":
            assert 0.5 <= self.speed <= 2.0
        if self.model == "rhasspy/piper-voices":
            assert 0.25 <= self.speed <= 4.0
        return self


# https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post("/v1/audio/speech")
async def synthesize(
    piper_model_manager: PiperModelManagerDependency,
    kokoro_model_manager: KokoroModelManagerDependency,
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
    match body.model:
        case "hexgrad/Kokoro-82M":
            # TODO: download the `voices.bin` file
            with kokoro_model_manager.load_model(body.voice) as tts:
                audio_generator = kokoro_utils.generate_audio(
                    tts,
                    body.input,
                    body.voice,
                    language=body.language or "en-us",
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
        case "rhasspy/piper-voices":
            from speaches import piper_utils

            with piper_model_manager.load_model(body.voice) as piper_tts:
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


@router.get("/v1/audio/speech/voices")
def list_voices(model_id: ModelId | None = None) -> list[Voice]:
    voices: list[Voice] = []
    if model_id == "hexgrad/Kokoro-82M" or model_id is None:
        kokoro_model_path = get_kokoro_model_path()
        for voice_id in kokoro_utils.VOICE_IDS:
            voice = Voice(
                created=0,
                model_path=kokoro_model_path,
                model_id="hexgrad/Kokoro-82M",
                owned_by="hexgrad",
                sample_rate=kokoro_utils.SAMPLE_RATE,
                voice_id=voice_id,
            )
            voices.append(voice)
    elif model_id == "rhasspy/piper-voices" or model_id is None:
        voices.extend(list(list_piper_models()))

    return voices
