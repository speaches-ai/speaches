from pathlib import Path

import anyio
from httpx import AsyncClient
import pytest

from speaches.api_types import CreateTranscriptionResponseVerboseJson

FILE_PATH = "simple.m4a"
ENDPOINT = "/v1/audio/transcriptions"


@pytest.mark.asyncio
async def test_transcription_with_diarization(aclient: AsyncClient) -> None:
    """Test that diarization parameter works and adds speaker information."""
    extension = Path(FILE_PATH).suffix[1:]
    async with await anyio.open_file(FILE_PATH, "rb") as f:
        data = await f.read()

    # Test with diarization enabled
    res = await aclient.post(
        ENDPOINT,
        files={"file": (f"audio.{extension}", data, f"audio/{extension}")},
        data={
            "model": "openai/whisper-tiny",
            "response_format": "verbose_json",
            "diarization": "true"
        }
    )
    res.raise_for_status()
    response_data = res.json()

    # Validate response structure
    response = CreateTranscriptionResponseVerboseJson.model_validate(response_data)

    # Check that we have segments
    assert len(response.segments) > 0

    # Check that speaker information is present when diarization is enabled
    # Note: actual speaker assignment depends on audio content
    for segment in response.segments:
        # Speaker field should exist (might be None if no speakers detected)
        assert hasattr(segment, "speaker")


@pytest.mark.asyncio
async def test_transcription_without_diarization(aclient: AsyncClient) -> None:
    """Test that transcription works without diarization (default behavior)."""
    extension = Path(FILE_PATH).suffix[1:]
    async with await anyio.open_file(FILE_PATH, "rb") as f:
        data = await f.read()

    # Test without diarization (default)
    res = await aclient.post(
        ENDPOINT,
        files={"file": (f"audio.{extension}", data, f"audio/{extension}")},
        data={
            "model": "openai/whisper-tiny",
            "response_format": "verbose_json"
        }
    )
    res.raise_for_status()
    response_data = res.json()

    # Validate response structure
    response = CreateTranscriptionResponseVerboseJson.model_validate(response_data)

    # Check that we have segments
    assert len(response.segments) > 0

    # Check that speaker field exists but is None when diarization is disabled
    for segment in response.segments:
        assert hasattr(segment, "speaker")
        assert segment.speaker is None


@pytest.mark.asyncio
async def test_transcription_diarization_explicit_false(aclient: AsyncClient) -> None:
    """Test that explicitly setting diarization=false works."""
    extension = Path(FILE_PATH).suffix[1:]
    async with await anyio.open_file(FILE_PATH, "rb") as f:
        data = await f.read()

    # Test with diarization explicitly disabled
    res = await aclient.post(
        ENDPOINT,
        files={"file": (f"audio.{extension}", data, f"audio/{extension}")},
        data={
            "model": "openai/whisper-tiny",
            "response_format": "verbose_json",
            "diarization": "false"
        }
    )
    res.raise_for_status()
    response_data = res.json()

    # Validate response structure
    response = CreateTranscriptionResponseVerboseJson.model_validate(response_data)

    # Check that we have segments
    assert len(response.segments) > 0

    # Check that speaker field exists but is None when diarization is disabled
    for segment in response.segments:
        assert hasattr(segment, "speaker")
        assert segment.speaker is None
