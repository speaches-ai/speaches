import io
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError
import pytest
import soundfile as sf

from speaches.audio import Audio
from speaches.config import Config
from speaches.dependencies import get_config_async, get_executor_registry_async
from speaches.routers import speech


@pytest.mark.parametrize(
    ("configured_rate", "request_rate", "expected_rate"),
    [(None, None, 22050), ("44100", None, 44100), ("44100", 16000, 16000)],
)
def test_speech_output_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
    configured_rate: str | None,
    request_rate: int | None,
    expected_rate: int,
) -> None:
    if configured_rate is None:
        monkeypatch.delenv("TTS_DEFAULT_SAMPLE_RATE", raising=False)
    else:
        monkeypatch.setenv("TTS_DEFAULT_SAMPLE_RATE", configured_rate)
    config = Config()

    audio = Audio(np.zeros(2205, dtype=np.float32), sample_rate=22050)
    executor = SimpleNamespace(
        model_manager=SimpleNamespace(handle_speech_request=lambda _: (chunk for chunk in (audio,)))
    )
    app = FastAPI()
    app.include_router(speech.router)
    app.dependency_overrides[get_config_async] = lambda: config
    app.dependency_overrides[get_executor_registry_async] = lambda: SimpleNamespace(text_to_speech={})
    payload: dict[str, str | int] = {"model": "test/model", "voice": "test", "input": "Test", "response_format": "wav"}
    if request_rate is not None:
        payload["sample_rate"] = request_rate

    with (
        patch.object(speech, "get_model_card_data_or_raise"),
        patch.object(speech, "find_executor_for_model_or_raise", return_value=executor),
        TestClient(app) as client,
    ):
        response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    _, actual_rate = sf.read(io.BytesIO(response.content))
    assert actual_rate == expected_rate


@pytest.mark.parametrize("configured_rate", ["7999", "48001"])
def test_speech_output_sample_rate_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, configured_rate: str
) -> None:
    monkeypatch.setenv("TTS_DEFAULT_SAMPLE_RATE", configured_rate)
    with pytest.raises(ValidationError):
        Config()
