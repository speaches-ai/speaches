from functools import lru_cache
import logging
from typing import Annotated

import av.error
from fastapi import (
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper.audio import decode_audio
from httpx import ASGITransport, AsyncClient
from numpy import float32
from numpy.typing import NDArray
from openai import AsyncOpenAI
from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from speaches.config import Config
from speaches.executors.kokoro.model_manager import KokoroModelManager
from speaches.executors.piper.model_manager import PiperModelManager
from speaches.executors.whisper.model_manager import WhisperModelManager

logger = logging.getLogger(__name__)

# NOTE: `get_config` is called directly instead of using sub-dependencies so that these functions could be used outside of `FastAPI`


# https://fastapi.tiangolo.com/advanced/settings/?h=setti#creating-the-settings-only-once-with-lru_cache
# WARN: Any new module that ends up calling this function directly (not through `FastAPI` dependency injection) should be patched in `tests/conftest.py`
@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> WhisperModelManager:
    config = get_config()
    return WhisperModelManager(config.whisper)


WhisperModelManagerDependency = Annotated[WhisperModelManager, Depends(get_model_manager)]


@lru_cache
def get_piper_model_manager() -> PiperModelManager:
    config = get_config()
    return PiperModelManager(config.whisper.ttl)  # HACK: should have its own config


PiperModelManagerDependency = Annotated[PiperModelManager, Depends(get_piper_model_manager)]


@lru_cache
def get_kokoro_model_manager() -> KokoroModelManager:
    config = get_config()
    return KokoroModelManager(config.whisper.ttl)  # HACK: should have its own config


KokoroModelManagerDependency = Annotated[KokoroModelManager, Depends(get_kokoro_model_manager)]

security = HTTPBearer()


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


ApiKeyDependency = Depends(verify_api_key)


# TODO: test async vs sync performance
def audio_file_dependency(
    file: Annotated[UploadFile, Form()],
) -> NDArray[float32]:
    try:
        audio = decode_audio(file.file)
    except av.error.InvalidDataError as e:
        raise HTTPException(
            status_code=415,
            detail="Failed to decode audio. The provided file type is not supported.",
        ) from e
    except av.error.ValueError as e:
        raise HTTPException(
            status_code=400,
            # TODO: list supported file types
            detail="Failed to decode audio. The provided file is likely empty.",
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to decode audio. This is likely a bug. Please create an issue at https://github.com/speaches-ai/speaches/issues/new."
        )
        raise HTTPException(status_code=500, detail="Failed to decode audio.") from e
    else:
        return audio  # pyright: ignore reportReturnType


AudioFileDependency = Annotated[NDArray[float32], Depends(audio_file_dependency)]


@lru_cache
def get_completion_client() -> AsyncCompletions:
    config = get_config()
    oai_client = AsyncOpenAI(
        base_url=config.chat_completion_base_url,
        api_key=config.chat_completion_api_key.get_secret_value(),
        max_retries=1,
    )
    return oai_client.chat.completions


CompletionClientDependency = Annotated[AsyncCompletions, Depends(get_completion_client)]


@lru_cache
def get_speech_client() -> AsyncSpeech:
    # this might not work as expected if `speech_router` won't have shared state (access to the same `model_manager`) with the main FastAPI `app`. TODO: verify
    from speaches.routers.speech import (
        router as speech_router,
    )

    config = get_config()
    http_client = AsyncClient(
        transport=ASGITransport(speech_router), base_url="http://test/v1"
    )  # NOTE: "test" can be replaced with any other value
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=1,
    )
    return oai_client.audio.speech


SpeechClientDependency = Annotated[AsyncSpeech, Depends(get_speech_client)]


@lru_cache
def get_transcription_client() -> AsyncTranscriptions:
    # this might not work as expected if `stt_router` won't have shared state (access to the same `model_manager`) with the main FastAPI `app`. TODO: verify
    from speaches.routers.stt import (
        router as stt_router,
    )

    config = get_config()
    http_client = AsyncClient(
        transport=ASGITransport(stt_router), base_url="http://test/v1"
    )  # NOTE: "test" can be replaced with any other value
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=1,
    )
    return oai_client.audio.transcriptions


TranscriptionClientDependency = Annotated[AsyncTranscriptions, Depends(get_transcription_client)]
