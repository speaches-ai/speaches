from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


@runtime_checkable
class VadModel(Protocol):
    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ) -> NDArray[np.float32]: ...


class VadOptions(BaseModel):
    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


class SpeechTimestamp(BaseModel):
    start: int
    end: int
