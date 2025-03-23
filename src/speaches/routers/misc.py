from fastapi import (
    APIRouter,
    Response,
)

from speaches.dependencies import WhisperModelManagerDependency
from speaches.model_aliases import ModelId

router = APIRouter()


@router.get("/health", tags=["diagnostic"])
def health() -> Response:
    return Response(status_code=200, content="OK")


# FIX: support non-whisper models
@router.get("/api/ps", tags=["experimental"], summary="Get a list of loaded models.")
def get_running_models(
    model_manager: WhisperModelManagerDependency,
) -> dict[str, list[str]]:
    return {"models": list(model_manager.loaded_models.keys())}


# FIX: support non-whisper models
@router.post("/api/ps/{model_id:path}", tags=["experimental"], summary="Load a model into memory.")
def load_model_route(model_manager: WhisperModelManagerDependency, model_id: ModelId) -> Response:
    if model_id in model_manager.loaded_models:
        return Response(status_code=409, content="Model already loaded")
    with model_manager.load_model(model_id):
        pass
    return Response(status_code=201)


# FIX: support non-whisper models
@router.delete("/api/ps/{model_id:path}", tags=["experimental"], summary="Unload a model from memory.")
def stop_running_model(model_manager: WhisperModelManagerDependency, model_id: str) -> Response:
    try:
        model_manager.unload_model(model_id)
        return Response(status_code=204)
    except (KeyError, ValueError) as e:
        match e:
            case KeyError():
                return Response(status_code=404, content="Model not found")
            case ValueError():
                return Response(status_code=409, content=str(e))
