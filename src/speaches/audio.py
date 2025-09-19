from __future__ import annotations

import io
import logging
import struct
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import soundfile as sf

from speaches.config import SAMPLES_PER_SECOND

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from speaches.routers.speech import ResponseFormat

logger = logging.getLogger(__name__)


# aip 'Write a function `resample_audio` which would take in RAW PCM 16-bit signed, little-endian audio data represented as bytes (`audio_bytes`) and resample it (either downsample or upsample) from `sample_rate` to `target_sample_rate` using numpy'
def resample_audio(audio_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_data) / sample_rate
    target_length = int(duration * target_sample_rate)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), target_length, endpoint=False), np.arange(len(audio_data)), audio_data
    )
    return resampled_data.astype(np.int16).tobytes()


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0xFFFFFFFF # placeholder value for streaming

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        0xFFFFFFFF, # placeholder file size value for streaming,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header


def convert_audio_format(
    audio_bytes: bytes,
    sample_rate: int,
    audio_format: ResponseFormat,
    format: str = "RAW",  # noqa: A002
    channels: int = 1,
    subtype: str = "PCM_16",
    endian: str = "LITTLE",
) -> bytes:
    # NOTE: the default dtype is float64. Should something else be used? Would that improve performance?
    data, _ = sf.read(
        io.BytesIO(audio_bytes),
        samplerate=sample_rate,
        format=format,
        channels=channels,
        subtype=subtype,
        endian=endian,
    )
    converted_audio_bytes_buffer = io.BytesIO()

    out_subtype = None
    if audio_format == "wav":
        header = create_wav_header(sample_rate=sample_rate, bits_per_sample=16, channels=channels)
        converted_audio_bytes_buffer.write(header)
        audio_format = 'RAW'
        out_subtype = "PCM_16"

    sf.write(converted_audio_bytes_buffer, data, samplerate=sample_rate, format=audio_format, subtype=out_subtype)
    return converted_audio_bytes_buffer.getvalue()


def audio_samples_from_file(file: BinaryIO) -> NDArray[np.float32]:
    audio_and_sample_rate = sf.read(
        file,
        format="RAW",
        channels=1,
        samplerate=SAMPLES_PER_SECOND,
        subtype="PCM_16",
        dtype="float32",
        endian="LITTLE",
    )
    audio = audio_and_sample_rate[0]
    return audio  # pyright: ignore[reportReturnType]


class Audio:
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
        self.data = data
        self.start = start

    def __repr__(self) -> str:
        return f"Audio(start={self.start:.2f}, end={self.end:.2f})"

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def duration(self) -> float:
        return len(self.data) / SAMPLES_PER_SECOND

    def after(self, ts: float) -> Audio:
        assert ts <= self.duration
        return Audio(self.data[int(ts * SAMPLES_PER_SECOND) :], start=ts)

    def extend(self, data: NDArray[np.float32]) -> None:
        self.data = np.append(self.data, data)
