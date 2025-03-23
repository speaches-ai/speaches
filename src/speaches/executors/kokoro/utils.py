import asyncio
from collections.abc import AsyncGenerator
import logging
from pathlib import Path
import time
from typing import Literal

from httpx import AsyncClient
import huggingface_hub
from kokoro_onnx import Kokoro
import numpy as np
from pydantic import BaseModel

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


logger = logging.getLogger(__name__)


class KokoroModelFiles(BaseModel):
    model: Path
    voices: Path


async def download_kokoro_model_files_if_not_exist(model_id: str = MODEL_ID) -> None:
    try:
        get_kokoro_model_files(model_id)
    except ValueError:
        await download_kokoro_model_files(model_id)


def list_kokoro_models() -> list[Model]:
    model = Model(id=MODEL_ID, owned_by=MODEL_ID.split("/")[0], task="text-to-speech")
    return [model]


def get_kokoro_model_files(model_id: str = MODEL_ID) -> KokoroModelFiles:
    assert model_id == MODEL_ID
    onnx_files = list(list_model_files(model_id, glob_pattern=f"**/{FILE_NAME}"))
    if len(onnx_files) == 0:
        raise ValueError(f"Could not find {FILE_NAME} file for '{model_id}' model")
    if len(onnx_files) > 1:
        raise ValueError(f"Found multiple {FILE_NAME} files for '{model_id}' model: {onnx_files}")
    model_path = onnx_files[0]
    voices_path = model_path.parent / "voices.bin"
    if not voices_path.exists():
        raise ValueError(f"Could not find voices.bin file for '{model_id}' model")
    return KokoroModelFiles(model=model_path, voices=voices_path)


async def download_kokoro_model_files(model_id: str = MODEL_ID) -> None:
    assert model_id == MODEL_ID
    model_repo_path = Path(
        await asyncio.to_thread(
            huggingface_hub.snapshot_download,
            model_id,
            repo_type="model",
            allow_patterns=[FILE_NAME],
            revision=KOKORO_REVISION,
        )
    )
    res = await AsyncClient().get(VOICES_FILE_SOURCE, follow_redirects=True)
    res = res.raise_for_status()  # HACK
    voices_path = model_repo_path / "voices.bin"
    voices_path.touch(exist_ok=True)
    voices_path.write_bytes(res.content)


def list_kokoro_voice_names() -> list[str]:
    model_files = get_kokoro_model_files()
    voices_npz = np.load(model_files.voices)
    return list(voices_npz.keys())


def list_kokoro_voices() -> list[Voice]:
    model_files = get_kokoro_model_files()
    voices_npz = np.load(model_files.voices)
    voice_names: list[str] = list(voices_npz.keys())

    voices = [
        Voice(
            model_id=MODEL_ID,
            voice_id=voice_name,
            created=int(model_files.voices.stat().st_mtime),
            owned_by=MODEL_ID.split("/")[0],
            sample_rate=SAMPLE_RATE,
            model_path=model_files.model,  # HACK: not applicable for Kokoro
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
