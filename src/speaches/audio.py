import io
import logging
from typing import BinaryIO, cast

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# NOTE: `signal.resample_poly` **might** be a better option for resampling audio data
def resample_audio_data(
    data: np.typing.NDArray[np.float32], sample_rate: int, target_sample_rate: int
) -> np.typing.NDArray[np.float32]:
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    return np.interp(np.linspace(0, len(data), target_length), np.arange(len(data)), data).astype(np.float32)


# aip 'Write a function `resample_audio` which would take in RAW PCM 16-bit signed, little-endian audio data represented as bytes (`audio_bytes`) and resample it (either downsample or upsample) from `sample_rate` to `target_sample_rate` using numpy'
def resample_audio_bytes(audio_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_data) / sample_rate
    target_length = int(duration * target_sample_rate)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), target_length, endpoint=False), np.arange(len(audio_data)), audio_data
    )
    return resampled_data.astype(np.int16).tobytes()


def convert_audio_format(
    audio_bytes: bytes,
    sample_rate: int,
    audio_format: str,
    input_audio_format: str = "RAW",
    channels: int = 1,
    subtype: str = "PCM_16",
    endian: str = "LITTLE",
) -> bytes:
    # NOTE: the default dtype is float64. Should something else be used? Would that improve performance?
    data, _ = sf.read(
        io.BytesIO(audio_bytes),
        samplerate=sample_rate,
        format=input_audio_format,
        channels=channels,
        subtype=subtype,
        endian=endian,
    )
    converted_audio_bytes_buffer = io.BytesIO()
    sf.write(converted_audio_bytes_buffer, data, samplerate=sample_rate, format=audio_format)
    return converted_audio_bytes_buffer.getvalue()


def audio_samples_from_file(file: BinaryIO, sample_rate: int) -> np.typing.NDArray[np.float32]:
    audio_data, _sample_rate = sf.read(
        file,
        format="RAW",
        channels=1,
        samplerate=sample_rate,
        subtype="PCM_16",
        dtype="float32",
        endian="LITTLE",
    )
    return cast("np.typing.NDArray[np.float32]", audio_data)


class Audio:
    def __init__(
        self,
        data: np.typing.NDArray[np.float32],
        sample_rate: int,
    ) -> None:
        self.data = data
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        return f"Audio(duration={self.duration:.2f}s, sample_rate={self.sample_rate}Hz, samples={len(self.data)})"

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def size_in_bits(self) -> int:
        return self.data.nbytes * 8

    @property
    def size_in_bytes(self) -> int:
        return self.data.nbytes

    @property
    def size_in_kb(self) -> float:
        return self.size_in_bytes / 1024.0

    @property
    def size_in_mb(self) -> float:
        return self.size_in_bytes / (1024.0 * 1024.0)

    # def after(self, seconds: float) -> "Audio":
    #     assert seconds <= self.duration, f"Seconds ({seconds}) must be less than or equal to duration ({self.duration})"
    #     return Audio(self.data[int(seconds * self.sample_rate) :], sample_rate=self.sample_rate)

    def extend(self, data: np.typing.NDArray[np.float32]) -> None:
        self.data = np.append(self.data, data)
