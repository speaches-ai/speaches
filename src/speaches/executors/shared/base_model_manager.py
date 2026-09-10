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
        model_unloaded_callback: Callable[[SelfDisposingModel[T]], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.load_fn = load_fn
        self.ttl = ttl
        self.model_unloaded_callback = model_unloaded_callback

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.model: T | None = None
        self.unloaded = False

    def unload(self) -> None:
        with self.rlock:
            if self.model is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            self._unload_locked()
        # The callback takes the manager's `_lock`, and `unload_model` holds
        # that lock across `unload()`. Running the callback inside `rlock`
        # would hold the locks in the opposite order and could deadlock, so
        # it fires only after `rlock` is released.
        if self.model_unloaded_callback is not None:
            self.model_unloaded_callback(self)

    def _unload_locked(self) -> None:
        # Caller holds `rlock` and has already checked the model is resident
        # and unused.
        if self.expire_timer:
            self.expire_timer.cancel()
        self.model = None
        # Mark the handle dead while `rlock` is still held, so a thread
        # already holding it cannot enter `__enter__` afterwards and
        # reload weights the registry no longer tracks.
        self.unloaded = True
        gc.collect()
        logger.info(f"Model {self.model_id} unloaded")

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
        unloaded = False
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.ttl > 0:
                    logger.debug(f"Model {self.model_id} is idle, scheduling offload in {self.ttl}s")
                    self.expire_timer = threading.Timer(self.ttl, self.unload)
                    self.expire_timer.start()
                elif self.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    if self.model is not None:
                        # Tear down inside this `rlock` hold. Releasing it
                        # first would let another request enter the handle and
                        # bump `ref_count`, turning this request's `__exit__`
                        # into a 500 for a call that succeeded.
                        self._unload_locked()
                        unloaded = True
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")
        # The callback takes the manager `_lock`, so it must not run while
        # `rlock` is held: `unload_model` already runs `_lock` -> `rlock`,
        # and nesting the other order can deadlock.
        if unloaded and self.model_unloaded_callback is not None:
            self.model_unloaded_callback(self)

    def __enter__(self) -> T:
        with self.rlock:
            if self.unloaded:
                raise ValueError(f"Model {self.model_id} has been unloaded")
            if self.model is None:
                self._load()
            self._increment_ref()
            assert self.model is not None
            return self.model

    def __exit__(self, *_args) -> None:
        self._decrement_ref()


class BaseModelManager[T](ABC):
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[T]] = OrderedDict()
        # RLock because `_handle_model_unloaded` re-enters it on the same
        # thread: `unload_model` holds `_lock` across `model.unload()`, and
        # `unload()` fires this callback inside that critical section.
        self._lock = threading.RLock()

    @abstractmethod
    def _load_fn(self, model_id: str) -> T:
        pass

    def _handle_model_unloaded(self, model: SelfDisposingModel[T]) -> None:
        with self._lock:
            # Remove the entry only if it still points at the handle that was
            # unloaded: `load_model` may already have replaced a dead entry
            # with a fresh one, and deleting that would orphan live weights.
            if self.loaded_models.get(model.model_id) is model:
                del self.loaded_models[model.model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id} not found")
            # `_lock` is held across `unload()` so `load_model` can neither
            # hand out the handle being torn down nor see a registry that
            # still lists it: acquisition and removal are serialized.
            # Removing the entry stays `unload()`'s job, via
            # `model_unloaded_callback` -- a refused unload (`ref_count > 0`)
            # raises before the callback runs, leaving the model registered,
            # listed by `GET /api/ps`, and reachable by a later
            # `DELETE /api/ps/{model_id}` retry.
            model.unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[T]:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is not None and model.unloaded:
                # A TTL unload can finish its teardown while the entry is
                # still present: `unload()` releases `rlock` before its
                # callback removes it. Drop the dead handle rather than
                # returning it -- its `__enter__` would only raise.
                del self.loaded_models[model_id]
                model = None
            if model is None:
                model = SelfDisposingModel[T](
                    model_id,
                    load_fn=lambda: self._load_fn(model_id),
                    ttl=self.ttl,
                    model_unloaded_callback=self._handle_model_unloaded,
                )
                self.loaded_models[model_id] = model
            else:
                logger.debug(f"{model_id} model already loaded")
            return model
