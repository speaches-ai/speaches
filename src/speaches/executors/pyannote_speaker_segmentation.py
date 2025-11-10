from collections.abc import Generator
import logging
from pathlib import Path

import huggingface_hub
from onnxruntime import InferenceSession
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.hf_utils import (
    HfModelFilter,
    get_cached_model_repos_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry

LIBRARY_NAME = "onnx"
TASK_NAME_TAG = "speaker-segmentation"
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

# MODEL_ID_BLACKLIST = {
#     "eek/wespeaker-voxceleb-resnet293-LM"  # reason: doesn't have `task` tag, also has pytorch binary file, onnx model file isn't named `model.onnx`
# }
MODEL_ID_WHITELIST = {"onnx-community/pyannote-segmentation-3.0"}


class PyannoteSpeakerSegmentationModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[Model]:
        for model_id in MODEL_ID_WHITELIST:
            yield Model(
                id=model_id,
                created=0,
                owned_by=model_id.split("/")[0],
                task="speaker-embedding",
                # task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id in MODEL_ID_WHITELIST:
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    task="speaker-embedding",
                    # task=TASK_NAME_TAG,
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


pyannote_speaker_segmentation_model_registry = PyannoteSpeakerSegmentationModelRegistry(hf_model_filter=hf_model_filter)


class PyannoteSpeakerSegmentationModelManager(BaseModelManager[InferenceSession]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> InferenceSession:
        model_files = pyannote_speaker_segmentation_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        inf_sess = InferenceSession(model_files.model, providers=providers)
        return inf_sess


# ==================================================
# INPUT DETAILS:
# ==================================================
# Name: input_values
# Shape: ['batch_size', 'num_channels', 'num_samples']

# ==================================================
# OUTPUT DETAILS:
# ==================================================
# Name: logits
# Shape: ['batch_size', 'num_frames', 7]
# Type: tensor(float)

# if __name__ == "__main__":
#     from speaches.dependencies import get_config
#
#     config = get_config()
#
#     model_manager = PyannoteSegmentationModelManager(ttl=config.tts_model_ttl, ort_opts=config.unstable_ort_opts)
#
#     remote_models = list(pyannote_segmentation_model_registry.list_remote_models())
#     for model in remote_models:
#         pyannote_segmentation_model_registry.download_model_files(model.id)
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
