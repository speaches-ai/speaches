import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Form,
)

from speaches.api_types import (
    CreateEmbeddingResponse,
    EmbeddingObject,
    EmbeddingUsage,
)
from speaches.dependencies import (
    AudioFileDependency,
    ExecutorRegistryDependency,
)
from speaches.executors.shared.handler_protocol import SpeakerEmbeddingRequest
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speaker-embedding"])


@router.post(
    "/v1/audio/speech/embedding",
)
def create_speech_embedding(
    executor_registry: ExecutorRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
) -> CreateEmbeddingResponse:
    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.speaker_embedding)

    speaker_embedding_request = SpeakerEmbeddingRequest(
        audio=audio,
        model_id=model,
    )
    speaker_embedding = executor.model_manager.handle_speaker_embedding_request(speaker_embedding_request)
    return CreateEmbeddingResponse(
        object="list",
        data=[EmbeddingObject(embedding=speaker_embedding.tolist())],
        model=model,
        usage=EmbeddingUsage(prompt_tokens=len(audio.data), total_tokens=len(audio.data)),
    )
