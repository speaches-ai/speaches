"""FunASR transcription executor.

Adds support for Alibaba DAMO Academy's FunASR family of speech-recognition
models (SenseVoice, Paraformer, ...) as an OpenAI-compatible STT backend.

FunASR models are tagged with ``library_name: funasr`` and
``pipeline_tag: automatic-speech-recognition`` on the HuggingFace Hub, which is
how they are matched to this executor.

The ``funasr`` package (and ``torch``) are optional dependencies, so they are
imported lazily inside the model manager. Install them with::

    pip install speaches[funasr]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import huggingface_hub
import openai.types.audio
from opentelemetry import trace

from speaches.api_types import Model
from speaches.audio import resample_audio_data
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import (  # noqa: TC001
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
)
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    get_model_repo_path,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced, traced_generator

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from funasr import AutoModel

    from speaches.config import Device, FunasrConfig

LIBRARY_NAME = "funasr"
TASK_NAME_TAG = "automatic-speech-recognition"
SAMPLE_RATE = 16000  # FunASR models operate on 16 kHz mono audio

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


class FunasrModelFiles(TypedDict):
    model_dir: Path


class FunasrModelRegistry(ModelRegistry[Model, FunasrModelFiles]):
    def list_remote_models(self) -> Generator[Model]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> FunasrModelFiles:
        repo_path = get_model_repo_path(model_id)
        if repo_path is None:
            raise ValueError(f"Model {model_id} not found in the local cache. Download it first.")
        snapshots_path = repo_path / "snapshots"
        snapshot_dir = next((p for p in snapshots_path.iterdir() if p.is_dir()), None)
        if snapshot_dir is None:
            raise ValueError(f"No snapshot found for model {model_id}.")
        return FunasrModelFiles(model_dir=snapshot_dir)

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


funasr_model_registry = FunasrModelRegistry(hf_model_filter=hf_model_filter)


class FunasrModelManager(BaseModelManager["AutoModel"]):
    def __init__(self, ttl: int, funasr_config: FunasrConfig) -> None:
        super().__init__(ttl)
        self.funasr_config = funasr_config

    def _resolve_device(self) -> str:
        device: Device = self.funasr_config.inference_device
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_fn(self, model_id: str) -> AutoModel:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "The 'funasr' package is required to use FunASR models. Install it with: pip install speaches[funasr]"
            ) from e
        model_files = funasr_model_registry.get_model_files(model_id)
        return AutoModel(
            model=str(model_files["model_dir"]),
            device=self._resolve_device(),
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )

    @traced()
    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format not in ("text", "json"):
            raise ValueError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        audio_data = request.audio.data
        if request.audio.sample_rate != SAMPLE_RATE:
            audio_data = resample_audio_data(audio_data, request.audio.sample_rate, SAMPLE_RATE)

        with self.load_model(request.model) as model:
            # TODO: warn on unsupported params (temperature, timestamp_granularities, ...)
            # TODO: use request.speech_segments for chunking long audio
            results = model.generate(
                input=audio_data,
                cache={},
                language=request.language or "auto",
                use_itn=True,
                batch_size_s=60,
            )
            text = rich_transcription_postprocess(results[0]["text"])

            match request.response_format:
                case "text":
                    return text, "text/plain"
                case "json":
                    return openai.types.audio.Transcription(text=text)

    @traced_generator()
    def handle_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> Generator[StreamingTranscriptionEvent]:
        raise NotImplementedError(f"'{request.model}' model doesn't support streaming transcription.")

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)
