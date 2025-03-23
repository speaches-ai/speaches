from collections.abc import Generator
import logging
from pathlib import Path
import typing

import huggingface_hub

from speaches.api_types import Model

LIBRARY_NAME = "ctranslate2"
TASK_NAME_TAG = "automatic-speech-recognition"

logger = logging.getLogger(__name__)


def list_whisper_remote_models() -> Generator[Model, None, None]:
    models = huggingface_hub.list_models(library=LIBRARY_NAME, tags=TASK_NAME_TAG, cardData=True)
    models = list(models)
    models.sort(key=lambda model: model.downloads or -1, reverse=True)
    for model in models:
        assert model.created_at is not None
        assert model.card_data is not None
        assert model.card_data.language is None or isinstance(model.card_data.language, str | list)
        if model.card_data.language is None:
            language = []
        elif isinstance(model.card_data.language, str):
            language = [model.card_data.language]
        else:
            language = model.card_data.language
        transformed_model = Model(
            id=model.id,
            created=int(model.created_at.timestamp()),
            owned_by=model.id.split("/")[0],
            language=language,
            task=TASK_NAME_TAG,
        )
        yield transformed_model


def list_whisper_local_models() -> Generator[Model, None, None]:
    hf_cache = huggingface_hub.scan_cache_dir()
    hf_models = [repo for repo in list(hf_cache.repos) if repo.repo_type == "model"]
    for model in hf_models:
        revision = next(iter(model.revisions))
        cached_readme_file = next((f for f in revision.files if f.file_name == "README.md"), None)
        if cached_readme_file:
            readme_file_path = Path(cached_readme_file.file_path)
        else:
            # NOTE: the README.md doesn't get downloaded when `WhisperModel` is called
            logger.debug(f"Model {model.repo_id} does not have a README.md file. Downloading it.")
            readme_file_path = Path(huggingface_hub.hf_hub_download(model.repo_id, "README.md"))

        model_card = huggingface_hub.ModelCard.load(readme_file_path)
        model_card_data = typing.cast(huggingface_hub.ModelCardData, model_card.data)
        if (
            model_card_data.library_name == LIBRARY_NAME
            and model_card_data.tags is not None
            and TASK_NAME_TAG in model_card_data.tags
        ):
            if model_card_data.language is None:
                language = []
            elif isinstance(model_card_data.language, str):
                language = [model_card_data.language]
            else:
                language = model_card_data.language
            transformed_model = Model(
                id=model.repo_id,
                created=int(model.last_modified),
                owned_by=model.repo_id.split("/")[0],
                language=language,
                task=TASK_NAME_TAG,
            )
            yield transformed_model
