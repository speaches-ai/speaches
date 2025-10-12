import logging

from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, get_available_providers  # pyright: ignore[reportAttributeAccessIssue]

from speaches.config import OrtOptions
from speaches.executors.base_model_manager import BaseModelManager
from speaches.executors.kokoro.utils import model_registry

logger = logging.getLogger(__name__)


class KokoroModelManager(BaseModelManager[Kokoro]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> Kokoro:
        model_files = model_registry.get_model_files(model_id)
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
        inf_sess = InferenceSession(model_files.model, providers=available_providers_with_opts)
        return Kokoro.from_session(inf_sess, str(model_files.voices))
