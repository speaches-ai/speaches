from collections.abc import Generator

from pydantic import BaseModel

from speaches.api_types import Model
from speaches.hf_utils import (
    HfModelFilter,
)


class ModelRegistry[ModelT: Model, ModelFilesT: BaseModel]:
    def __init__(self, hf_model_filter: HfModelFilter) -> None:
        self.hf_model_filter = hf_model_filter

    def list_remote_models(self) -> Generator[ModelT, None]: ...
    def list_local_models(self) -> Generator[ModelT, None]: ...
    def get_model(self, model_id: str) -> ModelT: ...
    def get_model_files(self, model_id: str) -> ModelFilesT: ...
    def download_model_files(self, model_id: str) -> None: ...
    def download_model_files_if_not_exist(self, model_id: str) -> bool:
        try:
            self.get_model_files(model_id)
        except Exception:  # noqa: BLE001
            self.download_model_files(model_id)
            return True
        return False
