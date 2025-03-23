from __future__ import annotations

from collections import OrderedDict
import json
import logging
from pathlib import Path
import threading
from typing import TYPE_CHECKING

from onnxruntime import InferenceSession

from speaches.executors.piper.utils import get_piper_voice_model_file
from speaches.model_manager import SelfDisposingModel

if TYPE_CHECKING:
    from piper.voice import PiperVoice


logger = logging.getLogger(__name__)


ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class PiperModelManager:
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[PiperVoice]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> PiperVoice:
        from piper.voice import PiperConfig, PiperVoice

        model_path = get_piper_voice_model_file(model_id)
        inf_sess = InferenceSession(model_path, providers=ONNX_PROVIDERS)
        config_path = Path(str(model_path) + ".json")
        conf = PiperConfig.from_dict(json.loads(config_path.read_text()))
        return PiperVoice(session=inf_sess, config=conf)

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

    def load_model(self, model_id: str) -> SelfDisposingModel[PiperVoice]:
        from piper.voice import PiperVoice

        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[PiperVoice](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
