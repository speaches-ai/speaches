from collections.abc import Generator
import logging
from pathlib import Path
from typing import TypedDict

import huggingface_hub
import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models import NemoConformerTdt
import openai.types.audio

from speaches.api_types import Model
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
)
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

# TODO: support model quants

# LIBRARY_NAME = "onnx" # NOTE: library name is derived and not stored in the README
TASK_NAME_TAG = "automatic-speech-recognition"
# TAGS = {"nemo-conformer-tdt"} # NOTE: I've tried to use this tag however it seems to be derived (likely from config.json) and isn't present when parsing the local model card

logger = logging.getLogger(__name__)

hf_model_filter = HfModelFilter(
    model_name="istupakov/parakeet-tdt",
    # library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    # tags=TAGS,
)


class NemoConformerTdtModelFiles(TypedDict):
    encoder: Path
    decoder_joint: Path
    vocab: Path
    config: Path


class NemoConformerTdtModelRegistry(ModelRegistry[Model, NemoConformerTdtModelFiles]):
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

    def get_model_files(self, model_id: str) -> NemoConformerTdtModelFiles:
        model_files = list(list_model_files(model_id))

        encoder_file_path = next(file_path for file_path in model_files if file_path.name == "encoder-model.onnx")
        decoder_joint_file_path = next(
            file_path for file_path in model_files if file_path.name == "decoder_joint-model.onnx"
        )
        vocab_file_path = next(file_path for file_path in model_files if file_path.name == "vocab.txt")
        config_file_path = next(file_path for file_path in model_files if file_path.name == "config.json")

        return NemoConformerTdtModelFiles(
            encoder=encoder_file_path,
            decoder_joint=decoder_joint_file_path,
            vocab=vocab_file_path,
            config=config_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        allow_patterns = list(NemoConformerTdt._get_model_files(quantization=None).values())  # noqa: SLF001

        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=[*allow_patterns, "README.md"]
        )


parakeet_model_registry = NemoConformerTdtModelRegistry(hf_model_filter=hf_model_filter)


class ParakeetModelManager(BaseModelManager[TextResultsAsrAdapter]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> TextResultsAsrAdapter:
        providers = get_ort_providers_with_options(self.ort_opts)
        return onnx_asr.load_model(model_id, providers=providers)

    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format not in ("text", "json"):
            raise ValueError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        with self.load_model(request.model) as parakeet:
            # TODO: issue warnings when client specifies unsupported parameters like `prompt`, `temperature`, `hotwords`, etc.
            # TODO: Use request.speech_segments for audio chunking

            results = parakeet.with_timestamps().recognize(request.audio.data)

            match request.response_format:
                case "text":
                    return results.text, "text/plain"
                case "json":
                    return openai.types.audio.Transcription(text=results.text)

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
