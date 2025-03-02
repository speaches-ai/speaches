from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field

MODEL_ID_ALIASES_PATH = Path("model_aliases.json")  # TODO: make configurable


@lru_cache
def load_model_id_aliases() -> dict[str, str]:
    return json.loads(MODEL_ID_ALIASES_PATH.read_text())


def resolve_model_id_alias(model_id: str) -> str:
    model_id_aliases = load_model_id_aliases()
    return model_id_aliases.get(model_id, model_id)


ModelId = Annotated[
    str,
    BeforeValidator(resolve_model_id_alias),
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]
