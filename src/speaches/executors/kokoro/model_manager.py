from collections import OrderedDict
import logging
import threading

from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, get_available_providers

from speaches.executors.kokoro.utils import model_registry
from speaches.model_manager import SelfDisposingModel

logger = logging.getLogger(__name__)

ORT_PROVIDERS_BLACKLIST = {"TensorrtExecutionProvider"}


class KokoroModelManager:
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[Kokoro]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> Kokoro:
        model_files = model_registry.get_model_files(model_id)
        available_providers: set[str] = set(
            get_available_providers()
        )  # HACK: `get_available_providers` is an unknown symbol (on MacOS at least)
        available_providers = available_providers - ORT_PROVIDERS_BLACKLIST
        if "TensorrtExecutionProvider" in available_providers:
            available_providers.remove("TensorrtExecutionProvider")
        inf_sess = InferenceSession(model_files.model, providers=list(available_providers))
        return Kokoro.from_session(inf_sess, str(model_files.voices))

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
