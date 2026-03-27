import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient
import pytest

from speaches import DEFAULT_GPU_MEM_LIMIT
from speaches.config import Config
from speaches.types.realtime import ConversationItemContentInputAudio

# -- GPU memory limit --


def test_default_gpu_mem_limit_value() -> None:
    assert DEFAULT_GPU_MEM_LIMIT == 1073741824  # 1 GB


def test_config_gpu_mem_limit_default() -> None:
    config = Config()
    assert config.gpu_mem_limit == DEFAULT_GPU_MEM_LIMIT


def test_ct2_cuda_env_var_set_from_gpu_mem_limit() -> None:
    import os

    ct2_config = os.environ.get("CT2_CUDA_CACHING_ALLOCATOR_CONFIG")
    assert ct2_config is not None
    assert ct2_config.endswith(str(DEFAULT_GPU_MEM_LIMIT))


# -- conversation.item.create with input_audio limitation --


def test_conversation_item_input_audio_has_no_audio_data_field() -> None:
    fields = ConversationItemContentInputAudio.model_fields
    assert "audio" not in fields, "ConversationItemContentInputAudio should not have an 'audio' field"
    assert "transcript" in fields
    assert "type" in fields


# -- Model aliasing --


def test_model_alias_resolution(tmp_path: Path) -> None:
    aliases = {"whisper-1": "Systran/faster-whisper-large-v3", "tts-1": "speaches-ai/Kokoro-82M-v1.0-ONNX"}
    alias_file = tmp_path / "model_aliases.json"
    alias_file.write_text(json.dumps(aliases))

    with patch("speaches.model_aliases.MODEL_ID_ALIASES_PATH", alias_file):
        from speaches.model_aliases import load_model_id_aliases, resolve_model_id_alias

        load_model_id_aliases.cache_clear()
        assert resolve_model_id_alias("whisper-1") == "Systran/faster-whisper-large-v3"
        assert resolve_model_id_alias("tts-1") == "speaches-ai/Kokoro-82M-v1.0-ONNX"
        load_model_id_aliases.cache_clear()


def test_model_alias_passthrough(tmp_path: Path) -> None:
    aliases = {"whisper-1": "Systran/faster-whisper-large-v3"}
    alias_file = tmp_path / "model_aliases.json"
    alias_file.write_text(json.dumps(aliases))

    with patch("speaches.model_aliases.MODEL_ID_ALIASES_PATH", alias_file):
        from speaches.model_aliases import load_model_id_aliases, resolve_model_id_alias

        load_model_id_aliases.cache_clear()
        assert resolve_model_id_alias("unknown-model") == "unknown-model"
        load_model_id_aliases.cache_clear()


# -- HF cache directory default --


def test_hf_cache_default_directory() -> None:
    import os

    from huggingface_hub import constants

    if os.environ.get("HF_HUB_CACHE"):
        pytest.skip("HF_HUB_CACHE is overridden by the environment")
    assert str(constants.HF_HUB_CACHE).endswith(".cache/huggingface/hub")


# -- Chat completion extra_body params --


def test_chat_completion_params_include_transcription_and_speech_model() -> None:
    from speaches.routers.chat import CompletionCreateParamsBase

    fields = CompletionCreateParamsBase.model_fields
    assert "transcription_model" in fields
    assert "speech_model" in fields


# -- /v1/registry endpoint --


@pytest.mark.asyncio
async def test_registry_endpoint_returns_valid_response(aclient: AsyncClient) -> None:
    res = await aclient.get("/v1/registry")
    assert res.status_code == 200
    body = res.json()
    assert "data" in body
    assert "object" in body
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


# -- Startup behavior: preload_models --


@pytest.mark.asyncio
async def test_preload_models_downloads_and_warms_up() -> None:
    mock_registry = MagicMock()
    mock_registry.warmup_model = AsyncMock()
    mock_registry.warmup_local_models = AsyncMock()
    mock_registry.warmup_inference = AsyncMock()
    mock_registry.all_executors.return_value = ()

    config = Config(preload_models=["model-a", "model-b"], warmup_all_local_models=False)

    with (
        patch("speaches.main.get_config", return_value=config),
        patch("speaches.main.get_executor_registry", return_value=mock_registry),
    ):
        from speaches.main import lifespan

        app = MagicMock()
        async with lifespan(app):
            pass

    assert mock_registry.download_model_by_id.call_count == 2
    mock_registry.download_model_by_id.assert_any_call("model-a")
    mock_registry.download_model_by_id.assert_any_call("model-b")
    assert mock_registry.warmup_model.call_count == 2
    mock_registry.warmup_model.assert_any_call("model-a")
    mock_registry.warmup_model.assert_any_call("model-b")


# -- Startup behavior: warmup_all_local_models --


@pytest.mark.asyncio
async def test_warmup_all_local_models_calls_warmup() -> None:
    mock_registry = MagicMock()
    mock_registry.warmup_local_models = AsyncMock()
    mock_registry.warmup_inference = AsyncMock()
    mock_registry.all_executors.return_value = ()

    config = Config(warmup_all_local_models=True)

    with (
        patch("speaches.main.get_config", return_value=config),
        patch("speaches.main.get_executor_registry", return_value=mock_registry),
    ):
        from speaches.main import lifespan

        app = MagicMock()
        async with lifespan(app):
            pass

    mock_registry.warmup_local_models.assert_called_once()


@pytest.mark.asyncio
async def test_warmup_all_local_models_disabled_skips_warmup() -> None:
    mock_registry = MagicMock()
    mock_registry.warmup_local_models = AsyncMock()
    mock_registry.warmup_inference = AsyncMock()
    mock_registry.all_executors.return_value = ()

    config = Config(warmup_all_local_models=False)

    with (
        patch("speaches.main.get_config", return_value=config),
        patch("speaches.main.get_executor_registry", return_value=mock_registry),
    ):
        from speaches.main import lifespan

        app = MagicMock()
        async with lifespan(app):
            pass

    mock_registry.warmup_local_models.assert_not_called()
