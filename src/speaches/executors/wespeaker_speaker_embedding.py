from collections.abc import Generator
import logging
from pathlib import Path

import huggingface_hub
from onnx_diarization.embedding import WeSpeakerEmbeddingModel
from onnx_diarization.fbank import FbankExtractor
from onnxruntime import InferenceSession, SessionOptions
from opentelemetry import trace
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.handler_protocol import SpeakerEmbeddingRequest, SpeakerEmbeddingResponse
from speaches.hf_utils import (
    HfModelFilter,
    get_cached_model_repos_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced

AVAILABLE_MODELS = {"Wespeaker/wespeaker-voxceleb-resnet34-LM"}

# LIBRARY_NAME = "onnx"
TASK_NAME_TAG = "speaker-embedding"
# TAGS = {"pyannote"}


class PyannoteModelFiles(BaseModel):
    model: Path
    readme: Path


hf_model_filter = HfModelFilter(
    model_name=next(iter(AVAILABLE_MODELS)),
    # library_name=LIBRARY_NAME,
    # task=TASK_NAME_TAG,
    # tags=TAGS,
)


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class WespeakerSpeakerEmbeddingModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[Model]:
        for model_id in AVAILABLE_MODELS:
            yield Model(
                id=model_id,
                created=0,
                owned_by=model_id.split("/")[0],
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id not in AVAILABLE_MODELS:
                continue
            # model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            # if model_card_data is None:
            #     continue
            # if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
            yield Model(
                id=cached_repo_info.repo_id,
                created=int(cached_repo_info.last_modified),
                owned_by=cached_repo_info.repo_id.split("/")[0],
                task=TASK_NAME_TAG,
            )

    def get_model_files(self, model_id: str) -> PyannoteModelFiles:
        model_files = list(list_model_files(model_id))
        model_file_path = next(file_path for file_path in model_files if file_path.name == "voxceleb_resnet34_LM.onnx")
        readme_file_path = next(file_path for file_path in model_files if file_path.name == "README.md")

        return PyannoteModelFiles(
            model=model_file_path,
            readme=readme_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=["voxceleb_resnet34_LM.onnx", "README.md"]
        )


wespeaker_speaker_embedding_model_registry = WespeakerSpeakerEmbeddingModelRegistry(hf_model_filter=hf_model_filter)


class WespeakerSpeakerEmbeddingModelManager(BaseModelManager[InferenceSession]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> InferenceSession:
        model_files = wespeaker_speaker_embedding_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        sess_options = SessionOptions()
        # XXX: why did i add the comment below
        # https://github.com/microsoft/onnxruntime/issues/1319#issuecomment-843945505
        # sess_options.log_severity_level = 3
        inf_sess = InferenceSession(model_files.model, providers=providers, sess_options=sess_options)
        return inf_sess

    @traced()
    def handle_speaker_embedding_request(self, request: SpeakerEmbeddingRequest, **_kwargs) -> SpeakerEmbeddingResponse:
        fbank_extractor = FbankExtractor()
        with self.load_model(request.model_id) as ort_session:
            model = WeSpeakerEmbeddingModel(ort_session, fbank_extractor)
            fbank_data = model.preprocess(request.audio.data)

            embeddings = model.extract(fbank_data)

            if embeddings.ndim == 2:
                embeddings = embeddings.squeeze()

            return embeddings
