from __future__ import annotations

import os
from typing import Annotated

from fastapi import (
    APIRouter,
    HTTPException,
    Path,
)
from pydantic import BeforeValidator

from speaches import kokoro_utils, piper_utils
from speaches.api_types import (
    ListModelsResponse,
    Model,
)
from speaches.model_aliases import resolve_model_id_alias
from speaches.whisper_utils import list_local_whisper_models, list_whisper_models

router = APIRouter(tags=["models"])

# TODO: should model aliases be listed?


@router.get("/v1/models")
def get_models() -> ListModelsResponse:
    models: list[Model] = []
    models.extend(kokoro_utils.get_kokoro_models())
    models.extend(piper_utils.get_piper_models())
    if os.getenv("HF_HUB_OFFLINE") is not None:
        models.extend(list(list_local_whisper_models()))
    else:
        models.extend(list(list_whisper_models()))
    return ListModelsResponse(data=models)


ModelId = Annotated[
    str,
    BeforeValidator(resolve_model_id_alias),
    # NOTE: `examples` doesn't work https://github.com/tiangolo/fastapi/discussions/10537
    Path(description="The ID of the model", example="Systran/faster-distil-whisper-large-v3"),
]


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
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' not found",
    )
