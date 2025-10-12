from __future__ import annotations

from collections import OrderedDict
import logging
import threading
from typing import TYPE_CHECKING

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnxruntime import get_available_providers

from speaches.model_manager import SelfDisposingModel

if TYPE_CHECKING:
    from speaches.config import OrtOptions

logger = logging.getLogger(__name__)


class ParakeetModelManager:
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        self.ttl = ttl
        self.ort_opts = ort_opts
        self.loaded_models: OrderedDict[str, SelfDisposingModel[TextResultsAsrAdapter]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> TextResultsAsrAdapter:
        # NOTE: `get_available_providers` is an unknown symbol (on MacOS at least)
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
        return onnx_asr.load_model(model_id, providers=available_providers_with_opts)

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

    def load_model(self, model_id: str) -> SelfDisposingModel[TextResultsAsrAdapter]:
        logger.debug(f"Loading model {model_id}")
        with self._lock:
            logger.debug("Acquired lock")
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[TextResultsAsrAdapter](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
