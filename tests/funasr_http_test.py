from __future__ import annotations

import io
from typing import TYPE_CHECKING
import wave

from huggingface_hub import ModelCardData
import numpy as np
import pytest

from speaches.executors.funasr import FunasrModelManager
from speaches.executors.silero_vad_v5 import SileroVADModelManager

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from httpx import AsyncClient
    from pytest_mock import MockerFixture

MODEL_ID = "FunAudioLLM/SenseVoiceSmall"


@pytest.fixture
def funasr_model(mocker: MockerFixture) -> MagicMock:
    model = mocker.MagicMock()
    model.generate.return_value = [{"text": "<|en|><|NEUTRAL|><|Speech|><|woitn|>hello"}]
    mocker.patch.object(FunasrModelManager, "_load_fn", return_value=model)
    mocker.patch.object(SileroVADModelManager, "handle_vad_request", return_value=[])
    mocker.patch(
        "speaches.routers.stt.get_model_card_data_or_raise",
        return_value=ModelCardData(library_name="funasr", pipeline_tag="automatic-speech-recognition"),
    )
    return model


@pytest.fixture
def wav_upload() -> tuple[str, bytes, str]:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x10" * 1600)
    return "sample.wav", buffer.getvalue(), "audio/wav"


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["json", "text"])
@pytest.mark.parametrize("language", [None, "zh"])
async def test_funasr_http_transcription(
    aclient: AsyncClient,
    funasr_model: MagicMock,
    wav_upload: tuple[str, bytes, str],
    response_format: str,
    language: str | None,
) -> None:
    data = {"model": MODEL_ID, "response_format": response_format}
    if language is not None:
        data["language"] = language
    response = await aclient.post("/v1/audio/transcriptions", data=data, files={"file": wav_upload})

    assert response.status_code == 200
    if response_format == "json":
        assert response.headers["content-type"] == "application/json"
        assert response.json()["text"] == "hello"
    else:
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "hello"
    funasr_model.generate.assert_called_once()
    arguments = funasr_model.generate.call_args.kwargs
    np.testing.assert_allclose(arguments["input"], np.full(1600, 0.125, dtype=np.float32))
    assert arguments["language"] == (language or "auto")
    assert arguments["cache"] == {}
    assert arguments["use_itn"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "stream", "rejected_option"),
    [
        ("verbose_json", "false", "verbose_json"),
        ("srt", "false", "srt"),
        ("vtt", "false", "vtt"),
        ("json", "true", "stream"),
        ("text", "true", "stream"),
    ],
)
async def test_funasr_http_rejects_unsupported_options_before_starting_response(
    aclient: AsyncClient,
    funasr_model: MagicMock,
    wav_upload: tuple[str, bytes, str],
    response_format: str,
    stream: str,
    rejected_option: str,
) -> None:
    response = await aclient.post(
        "/v1/audio/transcriptions",
        data={"model": MODEL_ID, "response_format": response_format, "stream": stream},
        files={"file": wav_upload},
    )

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert rejected_option in response.json()["detail"]
    funasr_model.generate.assert_not_called()
