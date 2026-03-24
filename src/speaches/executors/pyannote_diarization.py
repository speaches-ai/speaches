from collections.abc import Generator
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.hf_utils import (
    HfModelFilter,
    get_cached_model_repos_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

if TYPE_CHECKING:
    from pyannote.audio import Pipeline

AVAILABLE_MODELS = {"pyannote/speaker-diarization-3.1"}
TASK_NAME_TAG = "speaker-diarization"

hf_model_filter = HfModelFilter(
    task=TASK_NAME_TAG,
    tags={"pyannote"},
)

logger = logging.getLogger(__name__)


class PyannoteDiarizationModelFiles(BaseModel):
    config: Path


class PyannoteDiarizationModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[Model]:
        for model_id in AVAILABLE_MODELS:
            yield Model(
                id=model_id,
                created=0,
                owned_by=model_id.split("/")[0],
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id in AVAILABLE_MODELS:
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> PyannoteDiarizationModelFiles:
        model_files = list(list_model_files(model_id))
        config_file = next((f for f in model_files if f.name == "config.yaml"), None)
        if config_file is None:
            raise FileNotFoundError(f"config.yaml not found in local cache for model '{model_id}'")
        return PyannoteDiarizationModelFiles(config=config_file)

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


pyannote_diarization_model_registry = PyannoteDiarizationModelRegistry(hf_model_filter=hf_model_filter)


class PyannoteDiarizationModelManager(BaseModelManager["Pipeline"]):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str) -> "Pipeline":
        from pyannote.audio import Pipeline
        import torch

        logger.info(f"Loading pyannote diarization pipeline: {model_id}")
        pipeline = Pipeline.from_pretrained(model_id)
        assert pipeline is not None, f"Failed to load pyannote diarization pipeline for model '{model_id}'"
        if torch.cuda.is_available():
            logger.info("CUDA available, moving diarization pipeline to GPU")
            pipeline.to(torch.device("cuda"))
        return pipeline
