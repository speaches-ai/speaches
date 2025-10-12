import logging

from fastapi import (
    APIRouter,
    # Response,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub.utils._cache_manager import _scan_cached_repo
from pydantic import BaseModel, Field

from speaches.dependencies import (
    KokoroModelManagerDependency,
    ParakeetModelManagerDependency,
    PiperModelManagerDependency,
    WhisperModelManagerDependency,
)
from speaches.executors.kokoro import utils as kokoro_utils
from speaches.executors.parakeet import utils as parakeet_utils
from speaches.executors.piper import utils as piper_utils
from speaches.executors.whisper import utils as whisper_utils
from speaches.hf_utils import get_model_card_data_from_cached_repo_info, get_model_repo_path
from speaches.model_aliases import ModelId

logger = logging.getLogger(__name__)
router = APIRouter()


class MessageResponse(BaseModel):
    message: str = Field(..., description="A message describing the result of the operation.")


class RunningModelsResponse(BaseModel):
    models: list[str] = Field(..., description="List of model IDs that are currently loaded in memory.")


@router.get("/health", tags=["diagnostic"])
def health() -> JSONResponse:
    return JSONResponse(status_code=200, content={"message": "OK"})


@router.get("/api/ps", tags=["experimental"], summary="Get a list of loaded models.")
def get_running_models(
    whisper_model_manager: WhisperModelManagerDependency,
    piper_model_manager: PiperModelManagerDependency,
    kokoro_model_manager: KokoroModelManagerDependency,
    parakeet_model_manager: ParakeetModelManagerDependency,
) -> RunningModelsResponse:
    models = [
        *whisper_model_manager.loaded_models.keys(),
        *piper_model_manager.loaded_models.keys(),
        *kokoro_model_manager.loaded_models.keys(),
        *parakeet_model_manager.loaded_models.keys(),
    ]
    return RunningModelsResponse(models=models)


@router.post(
    "/api/ps/{model_id:path}",
    tags=["experimental"],
    summary="Load a model into memory.",
    responses={
        201: {"model": MessageResponse},
        409: {"model": MessageResponse},
        404: {"model": MessageResponse},
    },
)
def load_model_route(
    whisper_model_manager: WhisperModelManagerDependency,
    piper_model_manager: PiperModelManagerDependency,
    kokoro_model_manager: KokoroModelManagerDependency,
    parakeet_model_manager: ParakeetModelManagerDependency,
    model_id: ModelId,
) -> JSONResponse:
    model_managers = [whisper_model_manager, piper_model_manager, kokoro_model_manager]
    for model_manager in model_managers:
        if model_id in model_manager.loaded_models:
            return JSONResponse(
                status_code=409,
                content={
                    "message": f"Model '{model_id}' is already loaded.",
                },
            )
    model_repo_path = get_model_repo_path(model_id)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{model_id}' is not installed locally. You can download the model using `POST /v1/models`",
            },
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    assert model_card_data is not None, cached_repo_info  # FIXME

    if piper_utils.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
        with piper_model_manager.load_model(model_id):
            pass
    elif kokoro_utils.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
        with kokoro_model_manager.load_model(model_id):
            pass
    elif whisper_utils.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
        with whisper_model_manager.load_model(model_id):
            pass
    elif parakeet_utils.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
        with parakeet_model_manager.load_model(model_id):
            pass

    return JSONResponse(status_code=201, content={"message": f"Model '{model_id}' loaded."})


@router.delete(
    "/api/ps/{model_id:path}",
    tags=["experimental"],
    summary="Unload a model from memory.",
    responses={
        200: {"model": MessageResponse},
        404: {"model": MessageResponse},
    },
)
def stop_running_model(
    whisper_model_manager: WhisperModelManagerDependency,
    piper_model_manager: PiperModelManagerDependency,
    kokoro_model_manager: KokoroModelManagerDependency,
    parakeet_model_manager: ParakeetModelManagerDependency,
    model_id: str,
) -> JSONResponse:
    model_managers = [whisper_model_manager, piper_model_manager, kokoro_model_manager, parakeet_model_manager]
    for model_manager in model_managers:
        if model_id in model_manager.loaded_models:
            model_manager.unload_model(model_id)
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Model {model_id} unloaded.",
                },
            )
    return JSONResponse(
        status_code=404,
        content={
            "message": f"Model {model_id} is not loaded.",
        },
    )
