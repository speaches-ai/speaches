from fastapi import (
    APIRouter,
    HTTPException,
    Response,
)
from fastapi.responses import JSONResponse

from speaches.api_types import (
    ListModelsResponse,
    Model,
    ModelTask,
)
from speaches.executors.kokoro.utils import model_registry as kokoro_model_registry
from speaches.executors.piper.utils import model_registry as piper_model_registry
from speaches.executors.whisper.utils import model_registry as whisper_model_registry
from speaches.hf_utils import delete_local_model_repo
from speaches.model_aliases import ModelId

router = APIRouter(tags=["models"])

# TODO: should model aliases be listed?


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/models", response_model=ListModelsResponse)
def list_local_models(task: ModelTask | None = None) -> JSONResponse:
    models: list[Model] = []
    if task is None or task == "text-to-speech":
        models.extend(list(kokoro_model_registry.list_local_models()))
        models.extend(list(piper_model_registry.list_local_models()))
    if task is None or task == "automatic-speech-recognition":
        models.extend(list(whisper_model_registry.list_local_models()))
    return JSONResponse(content={"data": [model.model_dump() for model in models], "object": "list"})


# TODO: this is very naive implementation. It should be improved
@router.get("/v1/models/{model_id:path}")
def get_local_model(model_id: ModelId) -> Model:
    models: list[Model] = []
    models.extend(list(kokoro_model_registry.list_local_models()))
    models.extend(list(piper_model_registry.list_local_models()))
    models.extend(list(whisper_model_registry.list_local_models()))
    for model in models:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.post("/v1/models/{model_id:path}")
def download_remote_model(model_id: ModelId) -> Response:
    if model_id in [model.id for model in kokoro_model_registry.list_remote_models()]:
        was_downloaded = kokoro_model_registry.download_model_files_if_not_exist(model_id)
    elif model_id in [model.id for model in piper_model_registry.list_remote_models()]:
        was_downloaded = piper_model_registry.download_model_files_if_not_exist(model_id)
    elif model_id in [model.id for model in whisper_model_registry.list_remote_models()]:
        was_downloaded = whisper_model_registry.download_model_files_if_not_exist(model_id)
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if was_downloaded:
        return Response(status_code=200, content=f"Model '{model_id}' downloaded")
    else:
        return Response(status_code=201, content=f"Model '{model_id}' already exists")


# TODO: document that any model will be deleted regardless if it's supported speaches or not
@router.delete("/v1/models/{model_id:path}")
def delete_model(model_id: ModelId) -> Response:
    try:
        delete_local_model_repo(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.args[0]) from e
    return JSONResponse(status_code=200, content={"detail": f"Model '{model_id}' deleted"})


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/registry", response_model=ListModelsResponse)
def get_remote_models(task: ModelTask | None = None) -> JSONResponse:
    models: list[Model] = []
    if task is None or task == "text-to-speech":
        models.extend(list(kokoro_model_registry.list_remote_models()))
        models.extend(list(piper_model_registry.list_remote_models()))
    if task is None or task == "automatic-speech-recognition":
        models.extend(list(whisper_model_registry.list_remote_models()))
    return JSONResponse(content={"data": [model.model_dump() for model in models], "object": "list"})
