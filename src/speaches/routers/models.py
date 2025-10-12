from fastapi import (
    APIRouter,
    HTTPException,
    Response,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from speaches.api_types import (
    ListModelsResponse,
    Model,
    ModelTask,
)
from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.kokoro import KokoroModel, KokoroModelVoice
from speaches.executors.piper import PiperModel
from speaches.hf_utils import delete_local_model_repo
from speaches.model_aliases import ModelId

router = APIRouter(tags=["models"])

# TODO: should model aliases be listed?


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/models", response_model=ListModelsResponse)
def list_local_models(executor_registry: ExecutorRegistryDependency, task: ModelTask | None = None) -> JSONResponse:
    models: list[Model] = []
    executors = executor_registry.all_executors()
    for executor in executors:
        if task is None or executor.task == task:
            models.extend(list(executor.model_registry.list_local_models()))
    return JSONResponse(content={"data": [model.model_dump() for model in models], "object": "list"})


class ListAudioModelsResponse(BaseModel):
    models: list[Model]
    object: str = "list"


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/audio/models", response_model=ListAudioModelsResponse)
def list_local_audio_models(executor_registry: ExecutorRegistryDependency) -> JSONResponse:
    models: list[Model] = []
    for executor in executor_registry.text_to_speech:
        models.extend(list(executor.model_registry.list_local_models()))
    return JSONResponse(content={"models": [model.model_dump() for model in models], "object": "list"})


class ListVoicesResponse(BaseModel):
    voices: list[KokoroModelVoice | PiperModel]


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/audio/voices", response_model=ListModelsResponse)
def list_local_audio_voices(executor_registry: ExecutorRegistryDependency) -> JSONResponse:
    models: list[KokoroModel | PiperModel] = []
    for executor in executor_registry.text_to_speech:
        models.extend(list(executor.model_registry.list_local_models()))
    voices = [voice for model in models for voice in model.voices]
    return JSONResponse(content={"voices": [voice.model_dump() for voice in voices], "object": "list"})


# TODO: this is very naive implementation. It should be improved
# NOTE: without `response_model` and `JSONResponse` extra fields aren't included in the response
@router.get("/v1/models/{model_id:path}", response_model=Model)
def get_local_model(executor_registry: ExecutorRegistryDependency, model_id: ModelId) -> JSONResponse:
    models: list[Model] = []
    for executor in executor_registry.all_executors():
        models.extend(list(executor.model_registry.list_local_models()))
    for model in models:
        if model.id == model_id:
            return JSONResponse(content=model.model_dump())
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# NOTE: without `response_model` and `JSONResponse` extra fields aren't included in the response
@router.post("/v1/models/{model_id:path}")
def download_remote_model(executor_registry: ExecutorRegistryDependency, model_id: ModelId) -> Response:
    for executor in executor_registry.all_executors():
        if model_id in [model.id for model in executor.model_registry.list_remote_models()]:
            was_downloaded = executor.model_registry.download_model_files_if_not_exist(model_id)
            if was_downloaded:
                return Response(status_code=200, content=f"Model '{model_id}' downloaded")
            else:
                return Response(status_code=201, content=f"Model '{model_id}' already exists")
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


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
def get_remote_models(executor_registry: ExecutorRegistryDependency, task: ModelTask | None = None) -> JSONResponse:
    models: list[Model] = []
    for executor in executor_registry.all_executors():
        if task is None or executor.task == task:
            models.extend(list(executor.model_registry.list_remote_models()))
    return JSONResponse(content={"data": [model.model_dump() for model in models], "object": "list"})
