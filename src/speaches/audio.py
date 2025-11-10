import base64
from collections.abc import Generator
import io
import logging
import queue as queue_module
import subprocess
import threading
from typing import BinaryIO, Literal, Self, cast

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


_SOUNDFILE_SUPPORTED_AUDIO_FORMATS = ("mp3", "flac", "wav")


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

    def extend(self, data: np.typing.NDArray[np.float32]) -> None:
        self.data = np.append(self.data, data)

    def as_bytes(self) -> bytes:
        # NOTE: a more correct approach would be to normalize the audio data first but I'd rather avoid doing that to not introduce any performance overhead given that I expect the data to already be in the [-1.0, 1.0] range
        # Clip to [-1.0, 1.0] to avoid overflow
        # audio = np.clip(self.data, -1.0, 1.0)

        # Scale to int16 range and convert
        audio = (self.data * 32767).astype(np.int16)

        return audio.tobytes()

    def to_base64(self) -> str:
        audio_bytes = self.as_bytes()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def resample(self, target_sample_rate: int) -> Self:
        if self.sample_rate == target_sample_rate:
            return self
        self.data = resample_audio_data(self.data, self.sample_rate, target_sample_rate)
        self.sample_rate = target_sample_rate
        return self

    @classmethod
    def concatenate(cls, audios: list["Audio"]) -> "Audio":
        if not audios:
            raise ValueError("No audio segments to concatenate")
        sample_rate = audios[0].sample_rate
        for audio in audios:
            if audio.sample_rate != sample_rate:
                raise ValueError("All audio segments must have the same sample rate to concatenate")
        concatenated_data = np.concatenate([audio.data for audio in audios])
        return Audio(concatenated_data, sample_rate=sample_rate)


def stream_audio_as_formatted_bytes(
    audio_generator: Generator[Audio],
    audio_format: Literal["aac", "pcm", "opus", "mp3", "flac", "wav"],
    sample_rate: int | None = None,
) -> Generator[bytes]:
    if audio_format == "pcm":
        for audio in audio_generator:
            if sample_rate is not None:
                audio.resample(sample_rate)
            yield audio.as_bytes()
        return

    first_audio = next(audio_generator, None)
    if first_audio is None:
        return

    source_sample_rate = first_audio.sample_rate
    target_sample_rate = sample_rate if sample_rate is not None else source_sample_rate

    format_args = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame"],
        "wav": ["-f", "wav"],
        "flac": ["-f", "flac"],
        "opus": ["-f", "opus", "-codec:a", "libopus"],
        "aac": ["-f", "adts", "-codec:a", "aac"],
    }

    cmd = [
        "ffmpeg",
        "-f",
        "s16le",
        "-ar",
        str(source_sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-ar",
        str(target_sample_rate),
        *format_args[audio_format],
        "pipe:1",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output_queue: queue_module.Queue[bytes | None] = queue_module.Queue(maxsize=10)
    error_queue: queue_module.Queue[Exception] = queue_module.Queue()

    def write_to_stdin() -> None:
        try:
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin is None")
            process.stdin.write(first_audio.as_bytes())
            for audio in audio_generator:
                if audio.sample_rate != source_sample_rate:
                    raise ValueError(
                        f"Inconsistent sample rate: expected {source_sample_rate}, got {audio.sample_rate}"
                    )
                process.stdin.write(audio.as_bytes())
            process.stdin.close()
        except Exception as e:  # noqa: BLE001
            error_queue.put(e)

    def read_from_stdout() -> None:
        try:
            if process.stdout is None:
                raise RuntimeError("ffmpeg stdout is None")
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                output_queue.put(chunk)
            output_queue.put(None)
        except Exception as e:  # noqa: BLE001
            error_queue.put(e)

    writer_thread = threading.Thread(target=write_to_stdin, daemon=True)
    reader_thread = threading.Thread(target=read_from_stdout, daemon=True)

    writer_thread.start()
    reader_thread.start()

    try:
        while True:
            if not error_queue.empty():
                raise error_queue.get()

            try:
                chunk = output_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue

            if chunk is None:
                break

            yield chunk

        writer_thread.join(timeout=5.0)
        reader_thread.join(timeout=5.0)

        return_code = process.wait(timeout=5.0)
        if return_code != 0:
            stderr_output = process.stderr.read().decode() if process.stderr else ""
            raise RuntimeError(f"ffmpeg failed with return code {return_code}: {stderr_output}")

    except Exception:
        process.kill()
        process.wait()
        raise
