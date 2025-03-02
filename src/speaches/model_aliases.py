MODEL_ID_ALIASES = {}


def resolve_model_id_alias(model_id: str) -> str:
    return MODEL_ID_ALIASES.get(model_id, model_id)
