from collections.abc import Generator
import logging
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
from pydantic import BaseModel
import torch

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import SpeakerEmbeddingRequest, SpeakerEmbeddingResponse
from speaches.hf_utils import (
    HfModelFilter,
    get_cached_model_repos_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced

AVAILABLE_MODELS = {"pyannote/wespeaker-voxceleb-resnet34-LM"}
TASK_NAME_TAG = "speaker-embedding"

hf_model_filter = HfModelFilter(
    model_name="wespeaker-voxceleb-resnet34-LM",
)

logger = logging.getLogger(__name__)


class WespeakerModelFiles(BaseModel):
    config: Path


class WespeakerSpeakerEmbeddingModelRegistry(ModelRegistry):
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
            if cached_repo_info.repo_id not in AVAILABLE_MODELS:
                continue
            yield Model(
                id=cached_repo_info.repo_id,
                created=int(cached_repo_info.last_modified),
                owned_by=cached_repo_info.repo_id.split("/")[0],
                task=TASK_NAME_TAG,
            )

    def get_model_files(self, model_id: str) -> WespeakerModelFiles:
        model_files = list(list_model_files(model_id))
        config_file = next((f for f in model_files if f.name == "config.yaml"), None)
        if config_file is None:
            raise FileNotFoundError(f"config.yaml not found in local cache for model '{model_id}'")
        return WespeakerModelFiles(config=config_file)

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


wespeaker_speaker_embedding_model_registry = WespeakerSpeakerEmbeddingModelRegistry(hf_model_filter=hf_model_filter)


class WespeakerSpeakerEmbeddingModelManager(BaseModelManager):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str) -> Any:  # pyannote.audio.Inference
        from pyannote.audio import Inference, Model

        logger.info(f"Loading speaker embedding model: {model_id}")
        model = Model.from_pretrained(model_id)
        assert model is not None, f"Failed to load speaker embedding model '{model_id}'"
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        return Inference(model, window="whole")

    @traced()
    def handle_speaker_embedding_request(self, request: SpeakerEmbeddingRequest, **_kwargs) -> SpeakerEmbeddingResponse:
        with self.load_model(request.model_id) as inference:
            waveform = torch.from_numpy(request.audio.data).unsqueeze(0).float()
            embedding = np.asarray(inference({"waveform": waveform, "sample_rate": request.audio.sample_rate}))
            if embedding.ndim == 2:
                embedding = embedding.squeeze()
            return embedding
