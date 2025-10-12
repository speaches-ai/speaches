from __future__ import annotations

from collections import OrderedDict
import logging
import threading
from typing import TYPE_CHECKING

from onnxruntime import InferenceSession, get_available_providers

from speaches.executors.pyannote.utils import model_registry
from speaches.model_manager import SelfDisposingModel

if TYPE_CHECKING:
    from speaches.config import OrtOptions

logger = logging.getLogger(__name__)


class PyannoteModelManager:
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        self.ttl = ttl
        self.ort_opts = ort_opts
        self.loaded_models: OrderedDict[str, SelfDisposingModel[InferenceSession]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> InferenceSession:
        model_files = model_registry.get_model_files(model_id)
        available_providers: list[str] = get_available_providers()
        logger.debug(f"Available ONNX Runtime providers: {available_providers}")
        available_providers = [
            provider for provider in available_providers if provider not in self.ort_opts.exclude_providers
        ]
        available_providers = sorted(
            available_providers,
            key=lambda x: self.ort_opts.provider_priority.get(x, 0),
            reverse=True,
        )
        available_providers_with_opts = [
            (provider, self.ort_opts.provider_opts.get(provider, {})) for provider in available_providers
        ]
        logger.debug(f"Using ONNX Runtime providers: {available_providers_with_opts}")
        inf_sess = InferenceSession(model_files.model, providers=available_providers_with_opts)
        return inf_sess

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

    def load_model(self, model_id: str) -> SelfDisposingModel[InferenceSession]:
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[InferenceSession](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
