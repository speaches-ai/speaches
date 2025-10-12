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
from speaches.executors.kokoro.utils import KokoroModel, KokoroModelVoice
from speaches.executors.kokoro.utils import model_registry as kokoro_model_registry
from speaches.executors.parakeet.utils import model_registry as parakeet_model_registry
from speaches.executors.piper.utils import PiperModel
from speaches.executors.piper.utils import model_registry as piper_model_registry
from speaches.executors.pyannote.utils import model_registry as pyannote_model_registry
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
        models.extend(list(parakeet_model_registry.list_local_models()))
    if task is None or task == "speaker-embedding":
        models.extend(list(pyannote_model_registry.list_local_models()))
    return JSONResponse(content={"data": [model.model_dump() for model in models], "object": "list"})


class ListAudioModelsResponse(BaseModel):
    models: list[Model]
    object: str = "list"


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/audio/models", response_model=ListAudioModelsResponse)
def list_local_audio_models() -> JSONResponse:
    models: list[Model] = []
    models.extend(list(kokoro_model_registry.list_local_models()))
    models.extend(list(piper_model_registry.list_local_models()))
    return JSONResponse(content={"models": [model.model_dump() for model in models], "object": "list"})


class ListVoicesResponse(BaseModel):
    voices: list[KokoroModelVoice | PiperModel]


# HACK: returning ListModelsResponse directly causes extra `Model` fields to be omitted
@router.get("/v1/audio/voices", response_model=ListModelsResponse)
def list_local_audio_voices() -> JSONResponse:
    models: list[KokoroModel | PiperModel] = []
    models.extend(list(kokoro_model_registry.list_local_models()))
    models.extend(list(piper_model_registry.list_local_models()))
    voices = [voice for model in models for voice in model.voices]
    return JSONResponse(content={"voices": [voice.model_dump() for voice in voices], "object": "list"})


# TODO: this is very naive implementation. It should be improved
# NOTE: without `response_model` and `JSONResponse` extra fields aren't included in the response
@router.get("/v1/models/{model_id:path}", response_model=Model)
def get_local_model(model_id: ModelId) -> JSONResponse:
    models: list[Model] = []
    models.extend(list(kokoro_model_registry.list_local_models()))
    models.extend(list(piper_model_registry.list_local_models()))
    models.extend(list(whisper_model_registry.list_local_models()))
    models.extend(list(parakeet_model_registry.list_local_models()))
    models.extend(list(pyannote_model_registry.list_local_models()))
    for model in models:
        if model.id == model_id:
            return JSONResponse(content=model.model_dump())
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# NOTE: without `response_model` and `JSONResponse` extra fields aren't included in the response
@router.post("/v1/models/{model_id:path}")
def download_remote_model(model_id: ModelId) -> Response:
    if model_id in [model.id for model in kokoro_model_registry.list_remote_models()]:
        was_downloaded = kokoro_model_registry.download_model_files_if_not_exist(model_id)
    elif model_id in [model.id for model in piper_model_registry.list_remote_models()]:
        was_downloaded = piper_model_registry.download_model_files_if_not_exist(model_id)
    elif model_id in [model.id for model in whisper_model_registry.list_remote_models()]:
        was_downloaded = whisper_model_registry.download_model_files_if_not_exist(model_id)
    elif model_id in [model.id for model in parakeet_model_registry.list_remote_models()]:
        was_downloaded = parakeet_model_registry.download_model_files_if_not_exist(model_id)
    elif model_id in [model.id for model in pyannote_model_registry.list_remote_models()]:
        was_downloaded = pyannote_model_registry.download_model_files_if_not_exist(model_id)
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
        models.extend(list(parakeet_model_registry.list_remote_models()))
    if task is None or task == "speaker-embedding":
        models.extend(list(pyannote_model_registry.list_remote_models()))
    return JSONResponse(content={"data": [model.model_dump() for model in models], "object": "list"})
