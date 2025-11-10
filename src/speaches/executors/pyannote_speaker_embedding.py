from collections.abc import Generator
import logging
from pathlib import Path

import huggingface_hub
import numpy as np
from onnxruntime import InferenceSession, SessionOptions
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.handler_protocol import (
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResponse,
)
from speaches.hf_utils import (
    HfModelFilter,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

LIBRARY_NAME = "onnx"
TASK_NAME_TAG = "speaker-embedding"
TAGS = {"pyannote"}


class PyannoteModelFiles(BaseModel):
    model: Path
    readme: Path


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags=TAGS,
)


logger = logging.getLogger(__name__)

MODEL_ID_BLACKLIST = {
    "eek/wespeaker-voxceleb-resnet293-LM"  # reason: doesn't have `task` tag, also has pytorch binary file, onnx model file isn't named `model.onnx`
}


class PyannoteSpeakerEmbeddingModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[Model, None, None]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)

        for model in models:
            if model.id in MODEL_ID_BLACKLIST:
                continue
            try:
                if model.created_at is None or getattr(model, "card_data", None) is None:
                    logger.info(
                        f"Skipping (missing created_at/card_data): {model}",
                    )
                    continue
                assert model.card_data is not None

                yield Model(
                    id=model.id,
                    created=int(model.created_at.timestamp()),
                    owned_by=model.id.split("/")[0],
                    task=TASK_NAME_TAG,
                )

            except Exception:
                logger.exception(f"Skipping (unexpected error): {model.id}")
                continue

    def list_local_models(self) -> Generator[Model, None, None]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id in MODEL_ID_BLACKLIST:
                continue
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> PyannoteModelFiles:
        model_files = list(list_model_files(model_id))
        model_file_path = next(file_path for file_path in model_files if file_path.name == "model.onnx")
        readme_file_path = next(file_path for file_path in model_files if file_path.name == "README.md")

        return PyannoteModelFiles(
            model=model_file_path,
            readme=readme_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=["model.onnx", "README.md"]
        )


pyannote_speaker_embedding_model_registry = PyannoteSpeakerEmbeddingModelRegistry(hf_model_filter=hf_model_filter)


class PyannoteSpeakerEmbeddingModelManager(BaseModelManager[InferenceSession]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> InferenceSession:
        model_files = pyannote_speaker_embedding_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        sess_options = SessionOptions()
        # XXX: why did i add the comment below
        # https://github.com/microsoft/onnxruntime/issues/1319#issuecomment-843945505
        sess_options.log_severity_level = 3
        inf_sess = InferenceSession(model_files.model, providers=providers, sess_options=sess_options)
        return inf_sess

    def handle_speaker_embedding_request(self, request: SpeakerEmbeddingRequest, **_kwargs) -> SpeakerEmbeddingResponse:
        with self.load_model(request.model_id) as model:
            input_name = model.get_inputs()[0].name

            audio_data = request.audio_data.astype(np.float32)

            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(1, -1)

            outputs = model.run(None, {input_name: audio_data})
            embedding = outputs[0]

            if len(embedding.shape) > 1:  # pyright: ignore[reportAttributeAccessIssue]
                embedding = embedding.squeeze()  # pyright: ignore[reportAttributeAccessIssue]

            embedding = embedding / np.linalg.norm(embedding)  # pyright: ignore[reportCallIssue, reportArgumentType]

            return embedding


# if __name__ == "__main__":
#     from speaches.dependencies import get_config
#
#     config = get_config()
#
#     model_manager = PyannoteModelManager(ttl=config.tts_model_ttl, ort_opts=config.unstable_ort_opts)
#
#     remote_models = list(pyannote_model_registry.list_remote_models())
#     for model in remote_models:
#         pyannote_model_registry.download_model_files(model.id)
#     model_id = remote_models[0].id
#     with model_manager.load_model(model_id) as model:
#         print("=" * 50)
#         print("INPUT DETAILS:")
#         print("=" * 50)
#         for input_meta in model.get_inputs():
#             print(f"Name: {input_meta.name}")
#             print(f"Shape: {input_meta.shape}")
#             print(f"Type: {input_meta.type}")
#             print("-" * 50)
#
#         # Get output details
#         print("\n" + "=" * 50)
#         print("OUTPUT DETAILS:")
#         print("=" * 50)
#         for output_meta in model.get_outputs():
#             print(f"Name: {output_meta.name}")
#             print(f"Shape: {output_meta.shape}")
#             print(f"Type: {output_meta.type}")
#             print("-" * 50)
