from datetime import UTC, datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from huggingface_hub import CachedRepoInfo

from huggingface_hub import ModelCardData
import pytest
from pytest_mock import MockerFixture

from speaches.config import OrtOptions
from speaches.executors.parakeet import (
    ORUKEET_FILES,
    ORUKEET_METADATA,
    ORUKEET_MODEL_ID,
    ORUKEET_REVISION,
    ORUKEET_SUBFOLDER,
    NemoConformerTdtModelFiles,
    ParakeetModelManager,
    hf_model_filter,
    parakeet_model_registry,
    validate_orukeet_files,
)
from speaches.hf_utils import get_model_card_data_from_cached_repo_info


@pytest.fixture
def model_files(tmp_path: Path) -> NemoConformerTdtModelFiles:
    files = {key: tmp_path / filename for key, filename in ORUKEET_FILES.items()}
    for path in files.values():
        path.write_bytes(b"model fixture")
    return NemoConformerTdtModelFiles(
        encoder=files["encoder"], decoder_joint=files["decoder_joint"], vocab=files["vocab"], config=files["config"]
    )


def test_orukeet_routes_to_parakeet_only_for_exact_id() -> None:
    card = ModelCardData(pipeline_tag="automatic-speech-recognition", library_name="nemo")
    assert hf_model_filter.passes_filter(ORUKEET_MODEL_ID, card)
    assert hf_model_filter.passes_filter("istupakov/parakeet-tdt-0.6b-v3-onnx", card)
    assert not hf_model_filter.passes_filter("someone/orukeet", card)
    assert not hf_model_filter.passes_filter("oruk/orukeet-other", card)
    assert not hf_model_filter.passes_filter(ORUKEET_MODEL_ID, ModelCardData(pipeline_tag="text-to-speech"))


def test_remote_registry_includes_orukeet(mocker: MockerFixture) -> None:
    model = SimpleNamespace(
        id=ORUKEET_MODEL_ID,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        card_data=ModelCardData(language=["en", "de"], pipeline_tag="automatic-speech-recognition"),
    )
    mocker.patch("speaches.executors.parakeet.huggingface_hub.list_models", side_effect=[[], [model]])
    models = list(parakeet_model_registry.list_remote_models())
    assert [model.id for model in models] == [ORUKEET_MODEL_ID]
    assert models[0].language == ["en", "de"]


def test_cached_files_use_pinned_snapshot(mocker: MockerFixture, model_files: NemoConformerTdtModelFiles) -> None:
    cache = mocker.patch(
        "speaches.executors.parakeet.huggingface_hub.try_to_load_from_cache",
        side_effect=[*[str(path) for path in model_files.values()], *["cached" for _ in ORUKEET_METADATA]],
    )
    assert parakeet_model_registry.get_model_files(ORUKEET_MODEL_ID) == model_files
    assert cache.call_count == 4 + len(ORUKEET_METADATA)
    assert all(call.kwargs["revision"] == ORUKEET_REVISION for call in cache.call_args_list)
    assert all(call.args[1].startswith(ORUKEET_SUBFOLDER + "/") for call in cache.call_args_list[:4])


def test_missing_model_card_triggers_download(mocker: MockerFixture, model_files: NemoConformerTdtModelFiles) -> None:
    mocker.patch(
        "speaches.executors.parakeet.huggingface_hub.try_to_load_from_cache",
        side_effect=[*[str(path) for path in model_files.values()], None],
    )
    download = mocker.patch.object(parakeet_model_registry, "download_model_files")
    assert parakeet_model_registry.download_model_files_if_not_exist(ORUKEET_MODEL_ID) is True
    download.assert_called_once_with(ORUKEET_MODEL_ID)


@pytest.mark.parametrize("cached", [None, object()])
def test_missing_cached_file_is_reported(mocker: MockerFixture, cached: object) -> None:
    mocker.patch("speaches.executors.parakeet.huggingface_hub.try_to_load_from_cache", return_value=cached)
    with pytest.raises(FileNotFoundError, match=r"encoder-model\.int8\.onnx"):
        parakeet_model_registry.get_model_files(ORUKEET_MODEL_ID)


def test_download_includes_config_and_licenses(mocker: MockerFixture, model_files: NemoConformerTdtModelFiles) -> None:
    download = mocker.patch("speaches.executors.parakeet.huggingface_hub.snapshot_download")
    mocker.patch.object(parakeet_model_registry, "get_model_files", return_value=model_files)
    validate = mocker.patch("speaches.executors.parakeet.validate_orukeet_files")
    parakeet_model_registry.download_model_files(ORUKEET_MODEL_ID)
    assert download.call_args.kwargs["repo_id"] == ORUKEET_MODEL_ID
    assert download.call_args.kwargs["revision"] == ORUKEET_REVISION
    patterns = download.call_args.kwargs["allow_patterns"]
    assert f"{ORUKEET_SUBFOLDER}/config.json" in patterns
    assert f"{ORUKEET_SUBFOLDER}/LICENSE*" in patterns
    assert f"{ORUKEET_SUBFOLDER}/NOTICE.md" in patterns
    assert "README.md" in patterns
    validate.assert_called_once_with(model_files)


def test_failed_download_does_not_validate(mocker: MockerFixture) -> None:
    mocker.patch("speaches.executors.parakeet.huggingface_hub.snapshot_download", side_effect=OSError("offline"))
    validate = mocker.patch("speaches.executors.parakeet.validate_orukeet_files")
    with pytest.raises(OSError, match="offline"):
        parakeet_model_registry.download_model_files(ORUKEET_MODEL_ID)
    validate.assert_not_called()


def test_corrupt_model_is_rejected(mocker: MockerFixture, model_files: NemoConformerTdtModelFiles) -> None:
    expected = hashlib.sha256(b"model fixture").hexdigest()
    mocker.patch("speaches.executors.parakeet.ORUKEET_SHA256", dict.fromkeys(ORUKEET_FILES, expected))
    validate_orukeet_files(model_files)
    model_files["encoder"].write_bytes(b"corrupted")
    with pytest.raises(ValueError, match=r"checksum mismatch for encoder-model\.int8\.onnx"):
        validate_orukeet_files(model_files)


def test_cached_model_is_not_downloaded(mocker: MockerFixture, model_files: NemoConformerTdtModelFiles) -> None:
    mocker.patch.object(parakeet_model_registry, "get_model_files", return_value=model_files)
    download = mocker.patch.object(parakeet_model_registry, "download_model_files")
    assert parakeet_model_registry.download_model_files_if_not_exist(ORUKEET_MODEL_ID) is False
    download.assert_not_called()


def test_manager_loads_int8_locally(mocker: MockerFixture, model_files: NemoConformerTdtModelFiles) -> None:
    mocker.patch.object(parakeet_model_registry, "get_model_files", return_value=model_files)
    mocker.patch("speaches.executors.parakeet.validate_orukeet_files")
    mocker.patch(
        "speaches.executors.parakeet.get_ort_providers_with_options", return_value=[("CPUExecutionProvider", {})]
    )
    load = mocker.patch("speaches.executors.parakeet.onnx_asr.load_model")
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())
    with manager.load_model(ORUKEET_MODEL_ID):
        pass
    with manager.load_model(ORUKEET_MODEL_ID):
        pass
    load.assert_called_once_with(
        "nemo-conformer-tdt",
        path=model_files["config"].parent,
        quantization="int8",
        providers=[("CPUExecutionProvider", {})],
    )


def test_original_parakeet_load_is_unchanged(mocker: MockerFixture) -> None:
    mocker.patch("speaches.executors.parakeet.get_ort_providers_with_options", return_value=[])
    load = mocker.patch("speaches.executors.parakeet.onnx_asr.load_model")
    with ParakeetModelManager(ttl=-1, ort_opts=OrtOptions()).load_model("istupakov/parakeet-tdt-0.6b-v3-onnx"):
        pass
    load.assert_called_once_with("istupakov/parakeet-tdt-0.6b-v3-onnx", providers=[])


def test_incomplete_orukeet_cache_is_not_listed(mocker: MockerFixture) -> None:
    mocker.patch(
        "speaches.executors.parakeet.get_cached_model_repos_info",
        return_value=[SimpleNamespace(repo_id=ORUKEET_MODEL_ID)],
    )
    mocker.patch.object(parakeet_model_registry, "get_model_files", side_effect=FileNotFoundError)
    assert list(parakeet_model_registry.list_local_models()) == []


def test_pinned_revisions_without_main_have_a_model_card(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("---\nlanguage: [en]\npipeline_tag: automatic-speech-recognition\n---\n# Model\n")
    file = SimpleNamespace(file_name="README.md", file_path=readme)
    old = SimpleNamespace(refs=set(), last_modified=1, files=[])
    new = SimpleNamespace(refs=set(), last_modified=2, files=[file])
    repo = SimpleNamespace(revisions=[old, new], repo_id=ORUKEET_MODEL_ID)
    card = get_model_card_data_from_cached_repo_info(cast("CachedRepoInfo", repo))
    assert card is not None
    assert card.language == ["en"]
