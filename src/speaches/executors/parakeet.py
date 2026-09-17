from collections.abc import Generator
import hashlib
from itertools import chain
import logging
from pathlib import Path
from typing import TypedDict

import huggingface_hub
import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models import NemoConformerTdt
import openai.types.audio
from opentelemetry import trace

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
    load_repo_model_card_data,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced, traced_generator

# TODO: support model quants

# LIBRARY_NAME = "onnx" # NOTE: library name is derived and not stored in the README
TASK_NAME_TAG = "automatic-speech-recognition"
ORUKEET_MODEL_ID = "oruk/orukeet"
ORUKEET_REVISION = "1751fce6ecde442f14543cf1804800c49b3e415c"
ORUKEET_SUBFOLDER = "onnx/combined-v0.1.0-int8"
ORUKEET_FILES = {
    "encoder": "encoder-model.int8.onnx",
    "decoder_joint": "decoder_joint-model.int8.onnx",
    "vocab": "vocab.txt",
    "config": "config.json",
}
ORUKEET_SHA256 = {
    "encoder": "7b55f2a504a20a8e462899f5befd45f4a1784948d76ed0127902d9cf39405487",
    "decoder_joint": "95d3b1f53f9aadc5ef58e63664a3681a2184ee228b5010e1ef975a1c4ea8318a",
    "vocab": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    "config": "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
}
ORUKEET_METADATA = (
    "README.md",
    f"{ORUKEET_SUBFOLDER}/LICENSE-WEIGHTS",
    f"{ORUKEET_SUBFOLDER}/LICENSE-CONVERTER.txt",
    f"{ORUKEET_SUBFOLDER}/LICENSE-PREPROCESSOR.txt",
    f"{ORUKEET_SUBFOLDER}/NOTICE.md",
)
# TAGS = {"nemo-conformer-tdt"} # NOTE: I've tried to use this tag however it seems to be derived (likely from config.json) and isn't present when parsing the local model card

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ParakeetModelFilter(HfModelFilter):
    def passes_filter(self, model_id: str, model_card_data: huggingface_hub.ModelCardData) -> bool:
        if model_id == ORUKEET_MODEL_ID:
            return HfModelFilter(task=TASK_NAME_TAG).passes_filter(model_id, model_card_data)
        return super().passes_filter(model_id, model_card_data)


hf_model_filter = ParakeetModelFilter(
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


def validate_orukeet_files(files: NemoConformerTdtModelFiles) -> None:
    for key, path in (
        ("encoder", files["encoder"]),
        ("decoder_joint", files["decoder_joint"]),
        ("vocab", files["vocab"]),
        ("config", files["config"]),
    ):
        expected = ORUKEET_SHA256[key]
        with path.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Orukeet checksum mismatch for {path.name}. Delete and download the model again.")


class NemoConformerTdtModelRegistry(ModelRegistry[Model, NemoConformerTdtModelFiles]):
    def list_remote_models(self) -> Generator[Model]:
        models = chain(
            huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True),
            huggingface_hub.list_models(model_name=ORUKEET_MODEL_ID, cardData=True),
        )
        for model in models:
            if model.id != ORUKEET_MODEL_ID and not model.id.startswith("istupakov/parakeet-tdt"):
                continue
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
            if cached_repo_info.repo_id == ORUKEET_MODEL_ID:
                try:
                    self.get_model_files(ORUKEET_MODEL_ID)
                except FileNotFoundError:
                    continue
                readme = huggingface_hub.try_to_load_from_cache(
                    ORUKEET_MODEL_ID, "README.md", revision=ORUKEET_REVISION
                )
                if not isinstance(readme, str):
                    continue
                model_card_data = load_repo_model_card_data(readme)
            else:
                model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if cached_repo_info.repo_id == ORUKEET_MODEL_ID or self.hf_model_filter.passes_filter(
                cached_repo_info.repo_id, model_card_data
            ):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> NemoConformerTdtModelFiles:
        if model_id == ORUKEET_MODEL_ID:
            files = {}
            for key, filename in ORUKEET_FILES.items():
                path = huggingface_hub.try_to_load_from_cache(
                    model_id, f"{ORUKEET_SUBFOLDER}/{filename}", revision=ORUKEET_REVISION
                )
                if not isinstance(path, str):
                    raise FileNotFoundError(f"Missing cached Orukeet file: {filename}")
                files[key] = Path(path)
            for filename in ORUKEET_METADATA:
                if not isinstance(
                    huggingface_hub.try_to_load_from_cache(model_id, filename, revision=ORUKEET_REVISION), str
                ):
                    raise FileNotFoundError(f"Missing cached Orukeet metadata: {filename}")
            return NemoConformerTdtModelFiles(
                encoder=files["encoder"],
                decoder_joint=files["decoder_joint"],
                vocab=files["vocab"],
                config=files["config"],
            )

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
        if model_id == ORUKEET_MODEL_ID:
            huggingface_hub.snapshot_download(
                repo_id=model_id,
                revision=ORUKEET_REVISION,
                allow_patterns=[
                    "README.md",
                    *[f"{ORUKEET_SUBFOLDER}/{filename}" for filename in ORUKEET_FILES.values()],
                    f"{ORUKEET_SUBFOLDER}/LICENSE*",
                    f"{ORUKEET_SUBFOLDER}/NOTICE.md",
                ],
            )
            validate_orukeet_files(self.get_model_files(model_id))
            return

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
        if model_id == ORUKEET_MODEL_ID:
            files = parakeet_model_registry.get_model_files(model_id)
            validate_orukeet_files(files)
            return onnx_asr.load_model(
                "nemo-conformer-tdt", path=files["config"].parent, quantization="int8", providers=providers
            )
        return onnx_asr.load_model(model_id, providers=providers)

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
        with self.load_model(request.model) as parakeet:
            # TODO: issue warnings when client specifies unsupported parameters like `prompt`, `temperature`, `hotwords`, etc.
            # TODO: Use request.speech_segments for audio chunking

            results = parakeet.with_timestamps().recognize(request.audio.data)

            match request.response_format:
                case "text":
                    return results.text, "text/plain"
                case "json":
                    return openai.types.audio.Transcription(text=results.text)

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
