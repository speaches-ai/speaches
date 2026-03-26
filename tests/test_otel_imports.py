import importlib

import pytest

OTEL_INSTRUMENTATION_MODULES = [
    "opentelemetry.instrumentation.asyncio",
    "opentelemetry.instrumentation.asgi",
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.instrumentation.httpx",
    "opentelemetry.instrumentation.logging",
    "opentelemetry.instrumentation.grpc",
]


@pytest.mark.parametrize("module_name", OTEL_INSTRUMENTATION_MODULES)
def test_otel_instrumentation_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_otel_util_http_has_expected_exports() -> None:
    from opentelemetry.util import http as util_http

    # These are the symbols that opentelemetry-instrumentation-httpx and
    # opentelemetry-instrumentation-fastapi import at module level.
    # A mismatch here is exactly what caused the production crash.
    expected = [
        "ExcludeList",
        "get_excluded_urls",
        "parse_excluded_urls",
    ]
    for name in expected:
        assert hasattr(util_http, name), f"opentelemetry.util.http is missing {name!r}"


def test_create_app_with_otel_does_not_crash() -> None:
    from unittest.mock import patch

    from pydantic import SecretStr

    from speaches.config import Config, WhisperConfig
    from speaches.executors.shared.registry import ExecutorRegistry

    config = Config(
        whisper=WhisperConfig(),
        stt_model_ttl=0,
        tts_model_ttl=0,
        vad_model_ttl=0,
        enable_ui=False,
        otel_exporter_otlp_endpoint="http://localhost:4317",
        chat_completion_base_url="http://localhost:11434/v1",
        chat_completion_api_key=SecretStr("test"),
        loopback_host_url=None,
    )

    with (
        patch("speaches.dependencies.get_config", return_value=config),
        patch("speaches.main.get_config", return_value=config),
        patch("speaches.dependencies.get_executor_registry", return_value=ExecutorRegistry(config)),
    ):
        from speaches.main import create_app

        app = create_app()
        assert app is not None
