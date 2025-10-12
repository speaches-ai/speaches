import logging

from fastapi import (
    APIRouter,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from speaches.dependencies import ExecutorRegistryDependency
from speaches.model_aliases import ModelId
from speaches.routers.utils import get_model_card_data_or_raise

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
def get_running_models(executor_registry: ExecutorRegistryDependency) -> RunningModelsResponse:
    models = []
    for executor in executor_registry.all_executors():
        models.extend(executor.model_manager.loaded_models.keys())
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
def load_model_route(executor_registry: ExecutorRegistryDependency, model_id: ModelId) -> JSONResponse:
    # Check if model is already loaded
    for executor in executor_registry.all_executors():
        if model_id in executor.model_manager.loaded_models:
            return JSONResponse(
                status_code=409,
                content={
                    "message": f"Model '{model_id}' is already loaded.",
                },
            )

    model_card_data = get_model_card_data_or_raise(model_id)

    for executor in executor_registry.all_executors():
        if executor.can_handle_model(model_id, model_card_data):
            with executor.model_manager.load_model(model_id):
                pass
            return JSONResponse(status_code=201, content={"message": f"Model '{model_id}' loaded."})

    return JSONResponse(status_code=404, content={"message": f"Model '{model_id}' not supported."})


@router.delete(
    "/api/ps/{model_id:path}",
    tags=["experimental"],
    summary="Unload a model from memory.",
    responses={
        200: {"model": MessageResponse},
        404: {"model": MessageResponse},
        409: {"model": MessageResponse},
    },
)
def stop_running_model(executor_registry: ExecutorRegistryDependency, model_id: str) -> JSONResponse:
    for executor in executor_registry.all_executors():
        if model_id in executor.model_manager.loaded_models:
            try:
                executor.model_manager.unload_model(model_id)
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": f"Model {model_id} unloaded.",
                    },
                )
            except ValueError as e:
                return JSONResponse(status_code=409, content={"message": str(e)})
    return JSONResponse(
        status_code=404,
        content={
            "message": f"Model {model_id} is not loaded.",
        },
    )
