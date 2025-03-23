from collections import OrderedDict
import logging
import threading

from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession

from speaches.executors.kokoro.utils import get_kokoro_model_path
from speaches.model_manager import SelfDisposingModel

logger = logging.getLogger(__name__)


ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class KokoroModelManager:
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[Kokoro]] = OrderedDict()
        self._lock = threading.Lock()

    # TODO
    def _load_fn(self, _model_id: str) -> Kokoro:
        model_path = get_kokoro_model_path()
        voices_path = model_path.parent / "voices.bin"
        inf_sess = InferenceSession(model_path, providers=ONNX_PROVIDERS)
        return Kokoro.from_session(inf_sess, str(voices_path))

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

    def load_model(self, model_id: str) -> SelfDisposingModel[Kokoro]:
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[Kokoro](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
