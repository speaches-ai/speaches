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
from speaches.executors.kokoro.utils import list_kokoro_local_models, list_kokoro_remote_models
from speaches.executors.piper.utils import list_piper_local_models, list_piper_remote_models
from speaches.executors.whisper.utils import list_whisper_local_models, list_whisper_remote_models
from speaches.model_aliases import ModelId

router = APIRouter(tags=["models"])

# TODO: should model aliases be listed?


@router.get("/v1/models")
def get_local_models(task: ModelTask | None = None) -> ListModelsResponse:
    models: list[Model] = []
    if task is None or task == "text-to-speech":
        models.extend(list_kokoro_local_models())
        models.extend(list_piper_local_models())
    if task is None or task == "automatic-speech-recognition":
        models.extend(list(list_whisper_local_models()))
    return ListModelsResponse(data=models)


# very naive implementation
@router.get("/v1/models/{model_id:path}")
def get_local_model(model_id: ModelId) -> Model:
    models: list[Model] = []
    models.extend(list_kokoro_remote_models())
    models.extend(list_piper_remote_models())
    if os.getenv("HF_HUB_OFFLINE") is not None:
        models.extend(list(list_whisper_local_models()))
    else:
        models.extend(list(list_whisper_remote_models()))
    for model in models:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.get("/v1/registry")
def get_remote_models(task: ModelTask | None = None) -> ListModelsResponse:
    models: list[Model] = []
    if task is None or task == "text-to-speech":
        models.extend(list_kokoro_remote_models())
        models.extend(list_piper_remote_models())
    if task is None or task == "automatic-speech-recognition":
        models.extend(list(list_whisper_remote_models()))
    return ListModelsResponse(data=models)
