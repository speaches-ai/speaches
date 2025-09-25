from collections.abc import Generator
import logging
from pathlib import Path

import huggingface_hub
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

LIBRARY_NAME = "ctranslate2"
TASK_NAME_TAG = "automatic-speech-recognition"

logger = logging.getLogger(__name__)

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


class WhisperModelFiles(BaseModel):
    model: Path
    config: Path
    tokenizer: Path
    preprocessor_config: Path


class WhisperModelRegistry(ModelRegistry[Model, WhisperModelFiles]):
    def list_remote_models(self) -> Generator[Model, None, None]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model, None, None]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> WhisperModelFiles:
        model_files = list(list_model_files(model_id))

        # the necessary files are specified in `faster_whisper.transcribe`
        model_file_path = next(file_path for file_path in model_files if file_path.name == "model.bin")
        config_file_path = next(
            file_path for file_path in model_files if file_path.name == "config.json"
        )  # NOTE: I don't think this file is used
        tokenizer_file_path = next(file_path for file_path in model_files if file_path.name == "tokenizer.json")
        preprocessor_config_file_path = next(
            file_path for file_path in model_files if file_path.name == "preprocessor_config.json"
        )
        return WhisperModelFiles(
            model=model_file_path,
            config=config_file_path,
            tokenizer=tokenizer_file_path,
            preprocessor_config=preprocessor_config_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        # Taken from faster_whisper/utils.py
        allow_patterns = [
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ]
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=[*allow_patterns, "README.md"]
        )


model_registry = WhisperModelRegistry(hf_model_filter=hf_model_filter)
