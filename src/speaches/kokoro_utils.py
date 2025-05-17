from collections.abc import AsyncGenerator
import logging
from pathlib import Path
import time
from typing import Literal

import httpx
import huggingface_hub
from kokoro_onnx import Kokoro
import numpy as np

from speaches.api_types import Model, Voice
from speaches.audio import resample_audio
from speaches.hf_utils import list_model_files

KOKORO_REVISION = "c97b7bbc3e60f447383c79b2f94fee861ff156ac"
MODEL_ID = "hexgrad/Kokoro-82M"
FILE_NAME = "kokoro-v0_19.onnx"
VOICES_FILE_SOURCE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"

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


def get_kokoro_models() -> list[Model]:
    model = Model(id=MODEL_ID, owned_by=MODEL_ID.split("/")[0], task="text-to-speech")
    return [model]


def get_kokoro_model_path() -> Path:
    onnx_files = list(list_model_files(MODEL_ID, glob_pattern=f"**/{FILE_NAME}"))
    if not onnx_files:
        raise ValueError(f"Could not find {FILE_NAME} file for '{MODEL_ID}' model")
    return onnx_files[0]


def download_kokoro_model(allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None) -> None:
    model_repo_path = Path(
        huggingface_hub.snapshot_download(
            MODEL_ID,
            repo_type="model",
            revision=KOKORO_REVISION,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    res = httpx.get(VOICES_FILE_SOURCE, follow_redirects=True).raise_for_status()  # HACK
    voices_path = model_repo_path / "voices.bin"
    voices_path.touch(exist_ok=True)
    voices_path.write_bytes(res.content)


def list_kokoro_voice_names() -> list[str]:
    model_path = get_kokoro_model_path()
    voices_path = model_path.parent / "voices.bin"
    voices_npz = np.load(voices_path)
    return list(voices_npz.keys())


def list_kokoro_voices() -> list[Voice]:
    try:
        model_path = get_kokoro_model_path()
    except ValueError:
        return []
    voices_path = model_path.parent / "voices.bin"
    voices_npz = np.load(voices_path)
    voice_names: list[str] = list(voices_npz.keys())

    voices = [
        Voice(
            model_id=MODEL_ID,
            voice_id=voice_name,
            created=int(voices_path.stat().st_mtime),
            owned_by=MODEL_ID.split("/")[0],
            sample_rate=24000,
            model_path=model_path,  # HACK: not applicable for Kokoro
        )
        for voice_name in voice_names
    ]
    return voices


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
