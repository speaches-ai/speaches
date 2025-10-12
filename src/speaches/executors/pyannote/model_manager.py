import logging

from onnxruntime import InferenceSession

from speaches.config import OrtOptions
from speaches.executors.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.pyannote.utils import model_registry

logger = logging.getLogger(__name__)


class PyannoteModelManager(BaseModelManager[InferenceSession]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> InferenceSession:
        model_files = model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        inf_sess = InferenceSession(model_files.model, providers=providers)
        return inf_sess
