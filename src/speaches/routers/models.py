import os

from fastapi import (
    APIRouter,
    HTTPException,
)

from speaches import kokoro_utils, piper_utils
from speaches.api_types import (
    ListModelsResponse,
    Model,
    ModelTask,
)
from speaches.model_aliases import ModelId
from speaches.whisper_utils import list_local_whisper_models, list_whisper_models

router = APIRouter(tags=["models"])

# TODO: should model aliases be listed?


@router.get("/v1/models")
def get_models(task: ModelTask | None = None) -> ListModelsResponse:
    models: list[Model] = []
    if task is None or task == "text-to-speech":
        models.extend(kokoro_utils.get_kokoro_models())
        models.extend(piper_utils.get_piper_models())
    if task is None or task == "automatic-speech-recognition":
        if os.getenv("HF_HUB_OFFLINE") is not None:
            models.extend(list(list_local_whisper_models()))
        else:
            models.extend(list(list_whisper_models()))
    return ListModelsResponse(data=models)


# very naive implementation
@router.get("/v1/models/{model_id:path}")
def get_model(model_id: ModelId) -> Model:
    models: list[Model] = []
    models.extend(kokoro_utils.get_kokoro_models())
    models.extend(piper_utils.get_piper_models())
    if os.getenv("HF_HUB_OFFLINE") is not None:
        models.extend(list(list_local_whisper_models()))
    else:
        models.extend(list(list_whisper_models()))
    for model in models:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
