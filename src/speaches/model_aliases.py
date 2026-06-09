from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field

MODEL_ID_ALIASES_PATH = Path(__file__).resolve().parent / "model_aliases.json"

@lru_cache
def load_model_id_aliases():
    if not MODEL_ID_ALIASES_PATH.exists():
        return {}

    text = MODEL_ID_ALIASES_PATH.read_text().strip()

    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        return {}

def resolve_model_id_alias(model_id: str) -> str:
    model_id_aliases = load_model_id_aliases()

    if not isinstance(model_id, str):
        return str(model_id)

    model_id = model_id.strip()

    return model_id_aliases.get(model_id, model_id)

ModelId = Annotated[
    str,
    BeforeValidator(resolve_model_id_alias),
    Field(
        min_length=1,
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]
