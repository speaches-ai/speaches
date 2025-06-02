from collections.abc import Generator
import logging
from pathlib import Path
import shutil
import time
from typing import TypedDict

import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HfModelFilterDict(TypedDict):
    library: str | None
    task: str | None
    tags: list[str] | None


class HfModelFilter(BaseModel):
    library_name: str | None = None
    task: str | None = None
    tags: set[str] | None = None

    def passes_filter(self, model_card_data: huggingface_hub.ModelCardData) -> bool:
        # convert None to an empty set so it's easier to work with
        model_card_data_tags = set(model_card_data.tags) if model_card_data.tags is not None else set()
        if self.library_name is not None:
            # Handle both 'library_name' (correct) and 'library' (legacy/incorrect) fields
            model_library = model_card_data.library_name or getattr(model_card_data, "library", None)
            if model_library != self.library_name:
                return False
        if self.task is not None and (
            self.task != model_card_data.pipeline_tag and self.task not in model_card_data_tags
        ):
            return False
        if self.tags is not None and not self.tags.issubset(model_card_data_tags):  # noqa: SIM103
            return False
        return True

    def list_model_kwargs(self) -> HfModelFilterDict:
        kwargs: HfModelFilterDict = {
            "library": None,
            "task": None,
            "tags": None,
        }
        if self.library_name is not None:
            kwargs["library"] = self.library_name
        if self.task is not None:
            kwargs["task"] = self.task
        if self.tags is not None and len(self.tags) > 0:
            kwargs["tags"] = list(self.tags)
        return kwargs


def get_cached_model_repos_info() -> list[huggingface_hub.CachedRepoInfo]:
    hf_cache_info = huggingface_hub.scan_cache_dir()
    cache_repos_info = [repo for repo in list(hf_cache_info.repos) if repo.repo_type == "model"]
    return cache_repos_info


def get_model_card_data_from_cached_repo_info(
    cached_repo_info: huggingface_hub.CachedRepoInfo,
) -> huggingface_hub.ModelCardData | None:
    revisions = list(cached_repo_info.revisions)
    revision = revisions[0] if len(revisions) == 1 else next(rev for rev in revisions if "main" in rev.refs)
    files = list(revision.files)
    readme_cached_file_info = next((f for f in files if f.file_name == "README.md"), None)
    if readme_cached_file_info is None:
        return None
    readme_file_path = readme_cached_file_info.file_path
    model_card = huggingface_hub.ModelCard.load(readme_file_path, repo_type="model")
    assert isinstance(model_card.data, huggingface_hub.ModelCardData)
    return model_card.data


def load_repo_model_card_data(readme_file_path: str | Path) -> huggingface_hub.ModelCardData:
    model_card = huggingface_hub.ModelCard.load(readme_file_path, repo_type="model")
    assert isinstance(model_card.data, huggingface_hub.ModelCardData), model_card
    return model_card.data


def extract_language_list(card_data: huggingface_hub.ModelCardData) -> list[str]:
    assert card_data.language is None or isinstance(card_data.language, str | list)
    if card_data.language is None:
        language = []
    elif isinstance(card_data.language, str):
        language = [card_data.language]
    else:
        language = card_data.language
    return language


def list_local_model_ids() -> list[str]:
    model_dirs = list(Path(HF_HUB_CACHE).glob("models--*"))
    return [model_id_from_path(model_dir) for model_dir in model_dirs]


# alternative implementation that uses `huggingface_hub.scan_cache_dir`. Slightly cleaner but much slower
# def list_local_model_ids() -> list[str]:
#     start = time.perf_counter()
#     hf_cache = huggingface_hub.scan_cache_dir()
#     logger.debug(f"Scanned HuggingFace cache in {time.perf_counter() - start:.2f} seconds")
#     hf_models = [repo for repo in list(hf_cache.repos) if repo.repo_type == "model"]
#     return [model.repo_id for model in hf_models]


def does_local_model_exist(model_id: str) -> bool:
    return model_id in list_local_model_ids()


def model_id_from_path(repo_path: Path) -> str:
    repo_type, repo_id = repo_path.name.split("--", maxsplit=1)
    repo_type = repo_type[:-1]  # "models" -> "model"
    assert repo_type == "model"
    repo_id = repo_id.replace("--", "/")  # google--fleurs -> "google/fleurs"
    return repo_id


def get_model_repo_path(model_id: str, *, cache_dir: str | Path | None = None) -> Path | None:
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise huggingface_hub.CacheNotFound(
            f"Cache directory not found: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.",
            cache_dir=cache_dir,
        )

    if cache_dir.is_file():
        raise ValueError(
            f"Scan cache expects a directory but found a file: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable."
        )

    for repo_path in cache_dir.iterdir():
        if not repo_path.is_dir():
            continue
        if repo_path.name == ".locks":  # skip './.locks/' folder
            continue
        if "--" not in repo_path.name:  # cache might contain unrelated custom files
            continue
        repo_type, repo_id = repo_path.name.split("--", maxsplit=1)
        repo_type = repo_type[:-1]  # "models" -> "model"
        repo_id = repo_id.replace("--", "/")  # google--fleurs -> "google/fleurs"
        if repo_type != "model":
            continue
        if model_id == repo_id:
            return repo_path

    return None


def list_model_files(
    model_id: str, glob_pattern: str = "**/*", *, cache_dir: str | Path | None = None
) -> Generator[Path, None, None]:
    repo_path = get_model_repo_path(model_id, cache_dir=cache_dir)
    if repo_path is None:
        return None
    snapshots_path = repo_path / "snapshots"
    if not snapshots_path.exists():
        return None
    yield from list(snapshots_path.glob(glob_pattern))


def delete_local_model_repo(model_id: str) -> None:
    model_repo_path = get_model_repo_path(model_id)
    if model_repo_path is None:
        raise FileNotFoundError(f"Model repo not found: {model_id}")
    logger.debug(f"Deleting model repo: {model_repo_path}")
    start = time.perf_counter()
    shutil.rmtree(model_repo_path)
    logger.info(f"Deleted '{model_repo_path}' in {time.perf_counter() - start:.2f} seconds")
