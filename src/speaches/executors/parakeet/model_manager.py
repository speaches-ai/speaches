import logging

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnxruntime import get_available_providers

from speaches.config import OrtOptions
from speaches.executors.base_model_manager import BaseModelManager

logger = logging.getLogger(__name__)


class ParakeetModelManager(BaseModelManager[TextResultsAsrAdapter]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

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
