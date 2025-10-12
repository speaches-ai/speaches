import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
)
from huggingface_hub.utils._cache_manager import _scan_cached_repo
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
from speaches.hf_utils import (
    MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE,
    get_model_card_data_from_cached_repo_info,
    get_model_repo_path,
)
from speaches.model_aliases import ModelId

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
    model_repo_path = get_model_repo_path(model)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    if model_card_data is None:
        raise HTTPException(
            status_code=500,
            detail=MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE.format(model_id=model),
        )

    for executor in executor_registry.speaker_embedding:
        if executor.can_handle_model(model, model_card_data):
            with executor.model_manager.load_model(model) as inference_session:
                audio_input = audio.astype(np.float32)
                if len(audio_input.shape) == 1:
                    audio_input = audio_input.reshape(1, -1)

                outputs = inference_session.run(
                    None, {"waveform": audio_input}
                )  # TODO: handle other input name possibilities
                embedding = outputs[0][0].tolist()

                return CreateEmbeddingResponse(
                    object="list",
                    data=[EmbeddingObject(embedding=embedding)],
                    model=model,
                    usage=EmbeddingUsage(prompt_tokens=len(audio), total_tokens=len(audio)),
                )

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model}' is not supported. If you think this is a mistake, please open an issue.",
    )
