from __future__ import annotations

from typing import TYPE_CHECKING

from speaches.executors.executor import Executor
from speaches.executors.kokoro.model_manager import KokoroModelManager
from speaches.executors.kokoro.utils import kokoro_model_registry
from speaches.executors.parakeet.model_manager import ParakeetModelManager
from speaches.executors.parakeet.utils import parakeet_model_registry
from speaches.executors.piper.model_manager import PiperModelManager
from speaches.executors.piper.utils import piper_model_registry
from speaches.executors.pyannote.model_manager import PyannoteModelManager
from speaches.executors.pyannote.utils import pyannote_model_registry
from speaches.executors.whisper.model_manager import WhisperModelManager
from speaches.executors.whisper.utils import whisper_model_registry

if TYPE_CHECKING:
    from speaches.config import Config


class ExecutorRegistry:
    def __init__(self, config: Config) -> None:
        self._whisper_executor = Executor(
            name="whisper",
            model_manager=WhisperModelManager(config.stt_model_ttl, config.whisper),
            model_registry=whisper_model_registry,
            task="automatic-speech-recognition",
        )
        self._parakeet_executor = Executor(
            name="parakeet",
            model_manager=ParakeetModelManager(config.stt_model_ttl, config.unstable_ort_opts),
            model_registry=parakeet_model_registry,
            task="automatic-speech-recognition",
        )
        self._piper_executor = Executor(
            name="piper",
            model_manager=PiperModelManager(config.tts_model_ttl, config.unstable_ort_opts),
            model_registry=piper_model_registry,
            task="text-to-speech",
        )
        self._kokoro_executor = Executor(
            name="kokoro",
            model_manager=KokoroModelManager(config.tts_model_ttl, config.unstable_ort_opts),
            model_registry=kokoro_model_registry,
            task="text-to-speech",
        )
        self._pyannote_executor = Executor(
            name="pyannote",
            model_manager=PyannoteModelManager(config.stt_model_ttl, config.unstable_ort_opts),
            model_registry=pyannote_model_registry,
            task="speaker-embedding",
        )

    @property
    def transcription(self):  # noqa: ANN201
        return (self._whisper_executor, self._parakeet_executor)

    @property
    def text_to_speech(self):  # noqa: ANN201
        return (self._piper_executor, self._kokoro_executor)

    @property
    def speaker_embedding(self):  # noqa: ANN201
        return (self._pyannote_executor,)

    def all_executors(self):  # noqa: ANN201
        return (
            self._whisper_executor,
            self._parakeet_executor,
            self._piper_executor,
            self._kokoro_executor,
            self._pyannote_executor,
        )
