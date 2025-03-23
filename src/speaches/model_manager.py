from __future__ import annotations

from collections import OrderedDict
import gc
import json
import logging
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession

from speaches.executors.kokoro.utils import get_kokoro_model_path
from speaches.executors.piper.utils import get_piper_voice_model_file

if TYPE_CHECKING:
    from collections.abc import Callable

    from piper.voice import PiperVoice

    from speaches.config import (
        WhisperConfig,
    )

logger = logging.getLogger(__name__)


# TODO: enable concurrent model downloads


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


class WhisperModelManager:
    def __init__(self, whisper_config: WhisperConfig) -> None:
        self.whisper_config = whisper_config
        self.loaded_models: OrderedDict[str, SelfDisposingModel[WhisperModel]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> WhisperModel:
        return WhisperModel(
            model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
        )

    def _handle_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id} not found")
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992
            self.loaded_models[model_id].unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[WhisperModel]:
        logger.debug(f"Loading model {model_id}")
        with self._lock:
            logger.debug("Acquired lock")
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[WhisperModel](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.whisper_config.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]


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
