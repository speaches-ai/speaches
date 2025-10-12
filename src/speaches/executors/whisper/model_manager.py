from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from speaches.executors.base_model_manager import BaseModelManager

if TYPE_CHECKING:
    from speaches.config import (
        WhisperConfig,
    )

logger = logging.getLogger(__name__)


# TODO: enable concurrent model downloads


class WhisperModelManager(BaseModelManager[WhisperModel]):
    def __init__(self, ttl: int, whisper_config: WhisperConfig) -> None:
        super().__init__(ttl)
        self.whisper_config = whisper_config

    def _load_fn(self, model_id: str) -> WhisperModel:
        return WhisperModel(
            model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
        )
