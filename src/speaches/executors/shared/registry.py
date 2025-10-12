from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speaches.config import Config

from speaches.executors.kokoro import KokoroModelManager, kokoro_model_registry
from speaches.executors.parakeet import ParakeetModelManager, parakeet_model_registry
from speaches.executors.piper import PiperModelManager, piper_model_registry
from speaches.executors.pyannote import PyannoteModelManager, pyannote_model_registry
from speaches.executors.shared.executor import Executor
from speaches.executors.whisper import WhisperModelManager, whisper_model_registry


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
