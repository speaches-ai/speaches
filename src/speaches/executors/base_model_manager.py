from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
import threading

from speaches.model_manager import SelfDisposingModel

logger = logging.getLogger(__name__)


class BaseModelManager[T](ABC):
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[T]] = OrderedDict()
        self._lock = threading.Lock()

    @abstractmethod
    def _load_fn(self, model_id: str) -> T:
        pass

    def _handle_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id} not found")
            del self.loaded_models[model_id]
        model.unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[T]:
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[T](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
