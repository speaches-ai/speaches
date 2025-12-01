from __future__ import annotations

from typing import TYPE_CHECKING

from speaches.executors.pyannote_speaker_segmentation import (
    PyannoteSpeakerSegmentationModelManager,
    pyannote_speaker_segmentation_model_registry,
)

if TYPE_CHECKING:
    from speaches.config import Config

from speaches.executors.kokoro import KokoroModelManager, kokoro_model_registry
from speaches.executors.parakeet import ParakeetModelManager, parakeet_model_registry
from speaches.executors.piper import PiperModelManager, piper_model_registry
from speaches.executors.shared.executor import Executor
from speaches.executors.silero_vad_v5 import SileroVADModelManager, silero_vad_model_registry
from speaches.executors.wespeaker_speaker_embedding import (
    WespeakerSpeakerEmbeddingModelManager,
    wespeaker_speaker_embedding_model_registry,
)
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
        self._wespeaker_speaker_embedding_executor = Executor(
            name="wespeaker-speaker-embedding",
            model_manager=WespeakerSpeakerEmbeddingModelManager(0, config.unstable_ort_opts),  # HACK: hardcoded ttl
            model_registry=wespeaker_speaker_embedding_model_registry,
            task="speaker-embedding",
        )
        self._pyannote_speaker_segmentation_executor = Executor(
            name="pyannote-speaker-segmentation",
            model_manager=PyannoteSpeakerSegmentationModelManager(0, config.unstable_ort_opts),  # HACK: hardcoded ttl
            model_registry=pyannote_speaker_segmentation_model_registry,
            task="voice-activity-detection",
        )
        self._vad_executor = Executor(
            name="vad",
            model_manager=SileroVADModelManager(config.vad_model_ttl, config.unstable_ort_opts),
            model_registry=silero_vad_model_registry,
            task="voice-activity-detection",
        )

    @property
    def transcription(self):  # noqa: ANN201
        return (self._whisper_executor, self._parakeet_executor)

    @property
    def translation(self):  # noqa: ANN201
        return (self._whisper_executor,)

    @property
    def text_to_speech(self):  # noqa: ANN201
        return (self._piper_executor, self._kokoro_executor)

    @property
    def speaker_embedding(self):  # noqa: ANN201
        return (self._wespeaker_speaker_embedding_executor,)

    @property
    def speaker_segmentation(self):  # noqa: ANN201
        return (self._pyannote_speaker_segmentation_executor,)

    @property
    def vad(self):  # noqa: ANN201
        return self._vad_executor

    def all_executors(self):  # noqa: ANN201
        return (
            self._whisper_executor,
            self._parakeet_executor,
            self._piper_executor,
            self._kokoro_executor,
            self._wespeaker_speaker_embedding_executor,
            self._pyannote_speaker_segmentation_executor,
            self._vad_executor,
        )

    def download_model_by_id(self, model_id: str) -> bool:
        for executor in self.all_executors():
            if model_id in [model.id for model in executor.model_registry.list_remote_models()]:
                return executor.model_registry.download_model_files_if_not_exist(model_id)
        raise ValueError(f"Model '{model_id}' not found")
