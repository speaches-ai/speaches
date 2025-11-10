from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003
import time
from typing import TYPE_CHECKING, Literal

import huggingface_hub
from onnxruntime import InferenceSession
from pydantic import BaseModel, computed_field

from speaches.api_types import SUPPORTED_NON_STREAMABLE_SPEECH_RESPONSE_FORMATS, Model
from speaches.audio import convert_audio_format, resample_audio
from speaches.config import OrtOptions  # noqa: TC001
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse  # noqa: TC001
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
            try:
                # Must have basic metadata
                if model.created_at is None or getattr(model, "card_data", None) is None:
                    logger.info(
                        f"Skipping (missing created_at/card_data): {model}",
                    )
                    continue
                assert model.card_data is not None

                # Expect repo name like: piper-<lang>_<REGION>-<voice>-<quality>
                repo_name = model.id.split("/")[-1]
                parts = repo_name.split("-")
                if len(parts) != 4:
                    logger.info(f"Skipping (unexpected repo name shape): {model.id}")
                    continue

                _prefix, _language_and_region, name, quality = parts

                # Quality must be known
                sample_rate = PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP.get(quality)  # pyright: ignore[reportArgumentType]
                if sample_rate is None:
                    logger.info(f"Skipping (unknown quality '{quality}'): {model.id}")
                    continue

                # Exactly one language required
                languages = extract_language_list(model.card_data)
                if not languages or len(languages) != 1:
                    logger.info("Skipping (languages parsed=%s): %s", languages, model.id)
                    continue

                yield PiperModel(
                    id=model.id,
                    created=int(model.created_at.timestamp()),
                    owned_by=model.id.split("/")[0],
                    language=languages,
                    task=TASK_NAME_TAG,
                    sample_rate=sample_rate,
                    voices=[PiperModelVoice(name=name, language=languages[0])],
                )

            except Exception:
                # Defensive: never let one bad model crash the whole listing
                logger.exception(f"Skipping (unexpected error): {model.id}")
                continue

    def list_local_models(self) -> Generator[PiperModel, None, None]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
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


piper_model_registry = PiperModelRegistry(hf_model_filter=hf_model_filter)


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


class PiperModelManager(BaseModelManager["PiperVoice"]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> PiperVoice:
        from piper.voice import PiperConfig, PiperVoice

        model_files = piper_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        inf_sess = InferenceSession(model_files.model, providers=providers)
        conf = PiperConfig.from_dict(json.loads(model_files.config.read_text()))
        return PiperVoice(session=inf_sess, config=conf)

    def handle_speech_request(
        self,
        request: SpeechRequest,
        **_kwargs,
    ) -> SpeechResponse:
        if request.speed < 0.25 or request.speed > 4.0:
            msg = (f"Speed must be between 0.25 and 4.0, got {request.speed}",)
            raise ValueError(msg)
        # TODO: maybe check voice
        with self.load_model(request.model) as piper_tts:
            audio_generator = generate_audio(
                piper_tts, request.input, speed=request.speed, sample_rate=request.sample_rate
            )
            # these file formats can't easily be streamed because they have headers and/or metadata
            if request.response_format in SUPPORTED_NON_STREAMABLE_SPEECH_RESPONSE_FORMATS:
                audio_data = b"".join(audio_bytes for audio_bytes in audio_generator)
                audio_data = convert_audio_format(
                    audio_data, request.sample_rate or piper_tts.config.sample_rate, request.response_format
                )
                return audio_data, f"audio/{request.response_format}"
            if request.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(
                        audio_bytes, request.sample_rate or piper_tts.config.sample_rate, request.response_format
                    )
                    for audio_bytes in audio_generator
                )
            return audio_generator, f"audio/{request.response_format}"
