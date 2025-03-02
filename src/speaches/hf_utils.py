from collections.abc import Generator
from pathlib import Path

import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE


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


def get_model_path(model_id: str, *, cache_dir: str | Path | None = None) -> Path | None:
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
    repo_path = get_model_path(model_id, cache_dir=cache_dir)
    if repo_path is None:
        return None
    snapshots_path = repo_path / "snapshots"
    if not snapshots_path.exists():
        return None
    yield from list(snapshots_path.glob(glob_pattern))
