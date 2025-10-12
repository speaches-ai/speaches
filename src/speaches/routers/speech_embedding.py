import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Form,
)
import numpy as np

from speaches.api_types import (
    CreateEmbeddingResponse,
    EmbeddingObject,
    EmbeddingUsage,
)
from speaches.dependencies import (
    AudioFileDependency,
    ExecutorRegistryDependency,
)
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

    with executor.model_manager.load_model(model) as inference_session:
        audio_input = audio.astype(np.float32)
        if len(audio_input.shape) == 1:
            audio_input = audio_input.reshape(1, -1)

        outputs = inference_session.run(None, {"waveform": audio_input})  # TODO: handle other input name possibilities
        embedding = outputs[0][0].tolist()

        return CreateEmbeddingResponse(
            object="list",
            data=[EmbeddingObject(embedding=embedding)],
            model=model,
            usage=EmbeddingUsage(prompt_tokens=len(audio), total_tokens=len(audio)),
        )
