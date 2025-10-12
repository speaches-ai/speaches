import logging

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter

from speaches.config import OrtOptions
from speaches.executors.base_model_manager import BaseModelManager, get_ort_providers_with_options

logger = logging.getLogger(__name__)


class ParakeetModelManager(BaseModelManager[TextResultsAsrAdapter]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> TextResultsAsrAdapter:
        providers = get_ort_providers_with_options(self.ort_opts)
        return onnx_asr.load_model(model_id, providers=providers)
