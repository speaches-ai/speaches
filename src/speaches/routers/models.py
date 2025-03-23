import os

from fastapi import (
    APIRouter,
    HTTPException,
)

from speaches.api_types import (
    ListModelsResponse,
    Model,
    ModelTask,
)
from speaches.executors.kokoro.utils import list_kokoro_models
from speaches.executors.piper.utils import list_piper_models
from speaches.executors.whisper.utils import list_local_whisper_models, list_whisper_models
from speaches.model_aliases import ModelId

router = APIRouter(tags=["models"])

# TODO: should model aliases be listed?


@router.get("/v1/models")
def get_models(task: ModelTask | None = None) -> ListModelsResponse:
    models: list[Model] = []
    if task is None or task == "text-to-speech":
        models.extend(list_kokoro_models())
        models.extend(list_piper_models())
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
    models.extend(list_kokoro_models())
    models.extend(list_piper_models())
    if os.getenv("HF_HUB_OFFLINE") is not None:
        models.extend(list(list_local_whisper_models()))
    else:
        models.extend(list(list_whisper_models()))
    for model in models:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
