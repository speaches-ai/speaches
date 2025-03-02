from collections.abc import AsyncGenerator
import logging
from pathlib import Path
import time
from typing import Literal

import httpx
import huggingface_hub
from kokoro_onnx import Kokoro
import numpy as np

from speaches.audio import resample_audio
from speaches.hf_utils import list_model_files

KOKORO_REVISION = "c97b7bbc3e60f447383c79b2f94fee861ff156ac"
SAMPLE_RATE = 24000  # the default sample rate for Kokoro
Language = Literal["en-us", "en-gb", "fr-fr", "ja", "ko", "cmn"]
LANGUAGES: list[Language] = ["en-us", "en-gb", "fr-fr", "ja", "ko", "cmn"]

VOICE_IDS = [
    "af",  # Default voice is a 50-50 mix of Bella & Sarah
    "af_bella",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
    "af_nicole",
    "af_sky",
]

logger = logging.getLogger(__name__)


def get_kokoro_model_path() -> Path:
    file_name = "kokoro-v0_19.onnx"
    onnx_files = list(list_model_files("hexgrad/Kokoro-82M", glob_pattern=f"**/{file_name}"))
    if len(onnx_files) == 0:
        raise ValueError(f"Could not find {file_name} file for 'hexgrad/Kokoro-82M' model")
    return onnx_files[0]


def download_kokoro_model() -> None:
    model_id = "hexgrad/Kokoro-82M"
    model_repo_path = Path(
        huggingface_hub.snapshot_download(
            model_id,
            repo_type="model",
            allow_patterns=["kokoro-v0_19.onnx"],
            revision=KOKORO_REVISION,
        )
    )
    # HACK
    res = httpx.get(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin", follow_redirects=True
    ).raise_for_status()
    voices_path = model_repo_path / "voices.bin"
    voices_path.touch(exist_ok=True)
    voices_path.write_bytes(res.content)


async def generate_audio(
    kokoro_tts: Kokoro,
    text: str,
    voice: str,
    *,
    language: Language = "en-us",
    speed: float = 1.0,
    sample_rate: int | None = None,
) -> AsyncGenerator[bytes, None]:
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    start = time.perf_counter()
    async for audio_data, _ in kokoro_tts.create_stream(text, voice, lang=language, speed=speed):
        assert isinstance(audio_data, np.ndarray) and audio_data.dtype == np.float32 and isinstance(sample_rate, int)
        normalized_audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
        audio_bytes = normalized_audio_data.tobytes()
        if sample_rate != SAMPLE_RATE:
            audio_bytes = resample_audio(audio_bytes, SAMPLE_RATE, sample_rate)
        yield audio_bytes
    logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start}s")
