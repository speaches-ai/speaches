from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from onnxruntime import InferenceSession

from speaches.config import OrtOptions  # noqa: TC001
from speaches.executors.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.piper.utils import piper_model_registry

if TYPE_CHECKING:
    from piper.voice import PiperVoice


logger = logging.getLogger(__name__)


class PiperModelManager(BaseModelManager["PiperVoice"]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> PiperVoice:
        from piper.voice import PiperConfig, PiperVoice

        model_files = piper_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        inf_sess = InferenceSession(model_files.model, providers=providers)
        conf = PiperConfig.from_dict(json.loads(model_files.config.read_text()))
        return PiperVoice(session=inf_sess, config=conf)
