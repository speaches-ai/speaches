from __future__ import annotations

from collections import OrderedDict
import logging
import threading
from typing import TYPE_CHECKING

from pyannote.audio import Pipeline
import torch

from speaches.model_manager import SelfDisposingModel

if TYPE_CHECKING:
    from speaches.config import PyannoteConfig

logger = logging.getLogger(__name__)


class PyannoteModelManager:
    def __init__(self, pyannote_config: PyannoteConfig) -> None:
        self.pyannote_config = pyannote_config
        self.loaded_models: OrderedDict[str, SelfDisposingModel[Pipeline]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> Pipeline:
        device = self.pyannote_config.inference_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        pipeline = Pipeline.from_pretrained(model_id)
        pipeline.to(torch.device(device))
        return pipeline

    def _handle_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id} not found")
            self.loaded_models[model_id].unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[Pipeline]:
        logger.debug(f"Loading Pyannote model {model_id}")
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} Pyannote model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[Pipeline](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.pyannote_config.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
