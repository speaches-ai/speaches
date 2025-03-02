from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from speaches.api_types import Voice
from speaches.audio import resample_audio
from speaches.hf_utils import list_model_files

if TYPE_CHECKING:
    from collections.abc import Generator

    from piper.voice import PiperVoice

MODEL_ID = "rhasspy/piper-voices"
PiperVoiceQuality = Literal["x_low", "low", "medium", "high"]
PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP: dict[PiperVoiceQuality, int] = {
    "x_low": 16000,
    "low": 22050,
    "medium": 22050,
    "high": 22050,
}

logger = logging.getLogger(__name__)


def list_piper_models() -> Generator[Voice, None, None]:
    model_weights_files = list_model_files(MODEL_ID, glob_pattern="**/*.onnx")
    for model_weights_file in model_weights_files:
        yield Voice(
            created=int(model_weights_file.stat().st_mtime),
            model_path=model_weights_file,
            voice_id=model_weights_file.name.removesuffix(".onnx"),
            model_id=MODEL_ID,
            owned_by=MODEL_ID.split("/")[0],
            sample_rate=PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP[
                model_weights_file.name.removesuffix(".onnx").split("-")[-1]
            ],  # pyright: ignore[reportArgumentType]
        )


# NOTE: It's debatable whether caching should be done here or by the caller. Should be revisited.
class PiperVoiceConfigAudio(BaseModel):
    sample_rate: int
    quality: int


class PiperVoiceConfig(BaseModel):
    audio: PiperVoiceConfigAudio
    # NOTE: there are more fields in the config, but we don't care about them


@lru_cache
def read_piper_voices_config() -> dict[str, Any]:
    voices_file = next(list_model_files(MODEL_ID, glob_pattern="**/voices.json"), None)
    if voices_file is None:
        raise FileNotFoundError("Could not find voices.json file")  # noqa: EM101
    return json.loads(voices_file.read_text())


@lru_cache
def get_piper_voice_model_file(voice: str) -> Path:
    model_file = next(list_model_files(MODEL_ID, glob_pattern=f"**/{voice}.onnx"), None)
    if model_file is None:
        raise FileNotFoundError(f"Could not find model file for '{voice}' voice")
    return model_file


@lru_cache
def read_piper_voice_config(voice: str) -> PiperVoiceConfig:
    model_config_file = next(list_model_files(MODEL_ID, glob_pattern=f"**/{voice}.onnx.json"), None)
    if model_config_file is None:
        raise FileNotFoundError(f"Could not find config file for '{voice}' voice")
    return PiperVoiceConfig.model_validate_json(model_config_file.read_text())


# TODO: async generator https://github.com/mikeshardmind/async-utils/blob/354b93a276572aa54c04212ceca5ac38fedf34ab/src/async_utils/gen_transform.py#L147
def generate_audio(
    piper_tts: PiperVoice, text: str, *, speed: float = 1.0, sample_rate: int | None = None
) -> Generator[bytes, None, None]:
    if sample_rate is None:
        sample_rate = piper_tts.config.sample_rate
    start = time.perf_counter()
    for audio_bytes in piper_tts.synthesize_stream_raw(text, length_scale=1.0 / speed):
        if sample_rate != piper_tts.config.sample_rate:
            audio_bytes = resample_audio(audio_bytes, piper_tts.config.sample_rate, sample_rate)  # noqa: PLW2901
        yield audio_bytes
    logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start}s")
