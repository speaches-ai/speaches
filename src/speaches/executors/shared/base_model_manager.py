from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import gc
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from speaches.config import OrtOptions

logger = logging.getLogger(__name__)


def get_ort_providers_with_options(ort_opts: OrtOptions) -> list[tuple[str, dict]]:
    from onnxruntime import get_available_providers  # pyright: ignore[reportAttributeAccessIssue]

    available_providers: list[str] = get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")
    available_providers = [provider for provider in available_providers if provider not in ort_opts.exclude_providers]
    available_providers = sorted(
        available_providers,
        key=lambda x: ort_opts.provider_priority.get(x, 0),
        reverse=True,
    )
    available_providers_with_opts = [
        (provider, ort_opts.provider_opts.get(provider, {})) for provider in available_providers
    ]
    logger.debug(f"Using ONNX Runtime providers: {available_providers_with_opts}")
    return available_providers_with_opts


class SelfDisposingModel[T]:
    def __init__(
        self,
        model_id: str,
        load_fn: Callable[[], T],
        ttl: int,
        model_unloaded_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.load_fn = load_fn
        self.ttl = ttl
        self.model_unloaded_callback = model_unloaded_callback

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.model: T | None = None

    def unload(self) -> None:
        with self.rlock:
            if self.model is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()
            self.model = None
            gc.collect()
            logger.info(f"Model {self.model_id} unloaded")
            if self.model_unloaded_callback is not None:
                self.model_unloaded_callback(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.model is None
            logger.debug(f"Loading model {self.model_id}")
            start = time.perf_counter()
            self.model = self.load_fn()
            logger.info(f"Model {self.model_id} loaded in {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                logger.debug(f"Model was set to expire in {self.expire_timer.interval}s, cancelling")
                self.expire_timer.cancel()
            logger.debug(f"Incremented ref count for {self.model_id}, {self.ref_count=}")

    def _decrement_ref(self) -> None:
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.ttl > 0:
                    logger.info(f"Model {self.model_id} is idle, scheduling offload in {self.ttl}s")
                    self.expire_timer = threading.Timer(self.ttl, self.unload)
                    self.expire_timer.start()
                elif self.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> T:
        with self.rlock:
            if self.model is None:
                self._load()
            self._increment_ref()
            assert self.model is not None
            return self.model

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


class BaseModelManager[T](ABC):
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[T]] = OrderedDict()
        self._lock = threading.Lock()

    @abstractmethod
    def _load_fn(self, model_id: str) -> T:
        pass

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

    def load_model(self, model_id: str) -> SelfDisposingModel[T]:
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[T](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
