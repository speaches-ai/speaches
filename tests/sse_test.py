import json
from pathlib import Path

import anyio
from httpx import AsyncClient
from httpx_sse import aconnect_sse
import pytest
import srt
import webvtt
import webvtt.vtt

from speaches.api_types import (
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
)

MODEL_ID = "Systran/faster-whisper-tiny.en"
FILE_PATHS = ["audio.wav"]  # HACK
ENDPOINTS = [
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
]


parameters = [(file_path, endpoint) for endpoint in ENDPOINTS for file_path in FILE_PATHS]


@pytest.mark.asyncio
@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.parametrize(("file_path", "endpoint"), parameters)
async def test_streaming_transcription_text(aclient: AsyncClient, file_path: str, endpoint: str) -> None:
    extension = Path(file_path).suffix[1:]
    async with await anyio.open_file(file_path, "rb") as f:
        data = await f.read()
    kwargs = {
        "files": {"file": (f"audio.{extension}", data, f"audio/{extension}")},
        "data": {"model": MODEL_ID, "response_format": "text", "stream": True},
    }
    async with aconnect_sse(aclient, "POST", endpoint, **kwargs) as event_source:
        async for event in event_source.aiter_sse():
            print(event)
            assert len(event.data) > 1  # HACK: 1 because of the space character that's always prepended


@pytest.mark.asyncio
@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.parametrize(("file_path", "endpoint"), parameters)
async def test_streaming_transcription_json(aclient: AsyncClient, file_path: str, endpoint: str) -> None:
    extension = Path(file_path).suffix[1:]
    async with await anyio.open_file(file_path, "rb") as f:
        data = await f.read()
    kwargs = {
        "files": {"file": (f"audio.{extension}", data, f"audio/{extension}")},
        "data": {"model": MODEL_ID, "response_format": "json", "stream": True},
    }
    async with aconnect_sse(aclient, "POST", endpoint, **kwargs) as event_source:
        async for event in event_source.aiter_sse():
            CreateTranscriptionResponseJson(**json.loads(event.data))


@pytest.mark.asyncio
@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.parametrize(("file_path", "endpoint"), parameters)
async def test_streaming_transcription_verbose_json(aclient: AsyncClient, file_path: str, endpoint: str) -> None:
    extension = Path(file_path).suffix[1:]
    async with await anyio.open_file(file_path, "rb") as f:
        data = await f.read()
    kwargs = {
        "files": {"file": (f"audio.{extension}", data, f"audio/{extension}")},
        "data": {"model": MODEL_ID, "response_format": "verbose_json", "stream": True},
    }
    async with aconnect_sse(aclient, "POST", endpoint, **kwargs) as event_source:
        async for event in event_source.aiter_sse():
            CreateTranscriptionResponseVerboseJson(**json.loads(event.data))


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_transcription_vtt(aclient: AsyncClient) -> None:
    async with await anyio.open_file("audio.wav", "rb") as f:
        data = await f.read()
    kwargs = {
        "files": {"file": ("audio.wav", data, "audio/wav")},
        "data": {"model": MODEL_ID, "response_format": "vtt", "stream": False},
    }
    response = await aclient.post("/v1/audio/transcriptions", **kwargs)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/vtt; charset=utf-8"
    text = response.text
    webvtt.from_string(text)
    text = text.replace("WEBVTT", "YO")
    with pytest.raises(webvtt.vtt.MalformedFileError):
        webvtt.from_string(text)


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_transcription_srt(aclient: AsyncClient) -> None:
    async with await anyio.open_file("audio.wav", "rb") as f:
        data = await f.read()
    kwargs = {
        "files": {"file": ("audio.wav", data, "audio/wav")},
        "data": {"model": MODEL_ID, "response_format": "srt", "stream": False},
    }
    response = await aclient.post("/v1/audio/transcriptions", **kwargs)
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    text = response.text
    list(srt.parse(text))
    text = text.replace("1", "YO")
    with pytest.raises(srt.SRTParseError):
        list(srt.parse(text))
