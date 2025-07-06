from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
import time
from typing import TYPE_CHECKING, Literal

import huggingface_hub
from pydantic import BaseModel, computed_field

from speaches.api_types import Model
from speaches.audio import resample_audio
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Generator

    from piper.voice import PiperVoice


PiperVoiceQuality = Literal["x_low", "low", "medium", "high"]
PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP: dict[PiperVoiceQuality, int] = {
    "x_low": 16000,
    "low": 22050,
    "medium": 22050,
    "high": 22050,
}


LIBRARY_NAME = "onnx"
TASK_NAME_TAG = "text-to-speech"
TAGS = {"speaches", "piper"}


class PiperModelFiles(BaseModel):
    model: Path
    config: Path


class PiperModelVoice(BaseModel):
    name: str
    language: str

    @computed_field
    @property
    def id(self) -> str:
        return self.name


class PiperModel(Model):
    sample_rate: int
    voices: list[PiperModelVoice]


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags=TAGS,
)


logger = logging.getLogger(__name__)


class PiperModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[PiperModel, None, None]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            model_id_parts = model.id.split("/")[-1].split("-")
            assert len(model_id_parts) == 4, model.id
            # HACK: all of the `speaches-ai` piper models have a prefix of `piper-`. That's why there are 4 parts.
            _prefix, _language_and_region, name, quality = model_id_parts
            assert quality in PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP, model
            languages = extract_language_list(model.card_data)
            assert len(languages) == 1, model
            yield PiperModel(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=languages,
                task=TASK_NAME_TAG,
                sample_rate=PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP[quality],
                voices=[
                    PiperModelVoice(
                        name=name,
                        language=languages[0],
                    )
                ],
            )

    def list_local_models(self) -> Generator[PiperModel, None, None]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(model_card_data):
                repo_id_parts = cached_repo_info.repo_id.split("/")[-1].split("-")
                # HACK: all of the `speaches-ai` piper models have a prefix of `piper-`. That's why there are 4 parts.
                assert len(repo_id_parts) == 4, repo_id_parts
                _prefix, _language_and_region, name, quality = repo_id_parts
                assert quality in PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP, cached_repo_info.repo_id
                sample_rate = PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP[quality]
                languages = extract_language_list(model_card_data)
                assert len(languages) == 1, model_card_data
                yield PiperModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                    sample_rate=sample_rate,
                    voices=[
                        PiperModelVoice(
                            name=name,
                            language=languages[0],
                        )
                    ],
                )

    def get_model_files(self, model_id: str) -> PiperModelFiles:
        model_files = list(list_model_files(model_id))
        model_file_path = next(file_path for file_path in model_files if file_path.name == "model.onnx")
        config_file_path = next(file_path for file_path in model_files if file_path.name == "config.json")

        return PiperModelFiles(
            model=model_file_path,
            config=config_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=["model.onnx", "config.json", "README.md"]
        )


model_registry = PiperModelRegistry(hf_model_filter=hf_model_filter)


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
