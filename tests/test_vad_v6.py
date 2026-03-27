import logging
from pathlib import Path

from httpx import AsyncClient
import pytest

from speaches.executors.silero_vad_v5 import MODEL_ID_V6

logger = logging.getLogger(__name__)

AUDIO_FILE = Path(__file__).parent.parent / "audio.wav"


@pytest.mark.skipif(not AUDIO_FILE.exists(), reason=f"Audio file not found: {AUDIO_FILE}")
@pytest.mark.asyncio
async def test_model_listing_includes_v6(aclient: AsyncClient) -> None:
    res = await aclient.get("/v1/models")
    assert res.status_code == 200, f"Failed to list models: {res.text}"
    model_ids = [m["id"] for m in res.json()["data"]]
    assert MODEL_ID_V6 in model_ids, f"{MODEL_ID_V6} not in models: {model_ids}"


@pytest.mark.skipif(not AUDIO_FILE.exists(), reason=f"Audio file not found: {AUDIO_FILE}")
@pytest.mark.asyncio
async def test_vad_v6_basic(aclient: AsyncClient) -> None:
    audio_data = AUDIO_FILE.read_bytes()
    res = await aclient.post(
        "/v1/audio/speech/timestamps",
        files={"file": ("audio.wav", audio_data, "audio/wav")},
        data={"model": MODEL_ID_V6},
    )
    assert res.status_code == 200, f"Got {res.status_code}: {res.text}"
    timestamps = res.json()
    assert len(timestamps) == 1, f"Expected 1 segment, got {len(timestamps)}"
    seg = timestamps[0]
    assert seg["end"] > seg["start"], f"Segment end ({seg['end']}) should be after start ({seg['start']})"


@pytest.mark.skipif(not AUDIO_FILE.exists(), reason=f"Audio file not found: {AUDIO_FILE}")
@pytest.mark.asyncio
async def test_vad_v6_threshold(aclient: AsyncClient) -> None:
    audio_data = AUDIO_FILE.read_bytes()
    res = await aclient.post(
        "/v1/audio/speech/timestamps",
        files={"file": ("audio.wav", audio_data, "audio/wav")},
        data={"model": MODEL_ID_V6, "threshold": "0.5"},
    )
    assert res.status_code == 200, f"Got {res.status_code}: {res.text}"
    ts_low = res.json()
    assert isinstance(ts_low, list), f"Expected list response, got {type(ts_low)}"
    assert len(ts_low) >= 1, f"Expected at least 1 segment with threshold=0.5, got {len(ts_low)}"


@pytest.mark.skipif(not AUDIO_FILE.exists(), reason=f"Audio file not found: {AUDIO_FILE}")
@pytest.mark.asyncio
async def test_vad_v6_silence_duration(aclient: AsyncClient) -> None:
    audio_data = AUDIO_FILE.read_bytes()
    res = await aclient.post(
        "/v1/audio/speech/timestamps",
        files={"file": ("audio.wav", audio_data, "audio/wav")},
        data={"model": MODEL_ID_V6, "min_silence_duration_ms": "200"},
    )
    assert res.status_code == 200, f"Got {res.status_code}: {res.text}"
    ts_silence = res.json()
    assert isinstance(ts_silence, list), f"Expected list response, got {type(ts_silence)}"
    assert len(ts_silence) >= 1, f"Expected at least 1 segment with min_silence=200ms, got {len(ts_silence)}"


@pytest.mark.skipif(not AUDIO_FILE.exists(), reason=f"Audio file not found: {AUDIO_FILE}")
@pytest.mark.asyncio
async def test_vad_v6_response_schema(aclient: AsyncClient) -> None:
    audio_data = AUDIO_FILE.read_bytes()
    res = await aclient.post(
        "/v1/audio/speech/timestamps",
        files={"file": ("audio.wav", audio_data, "audio/wav")},
        data={"model": MODEL_ID_V6},
    )
    assert res.status_code == 200, f"Got {res.status_code}: {res.text}"
    timestamps = res.json()
    assert isinstance(timestamps, list), f"Expected list response, got {type(timestamps)}"
    for ts in timestamps:
        assert "start" in ts, f"Missing 'start' key in timestamp: {ts}"
        assert "end" in ts, f"Missing 'end' key in timestamp: {ts}"
        assert isinstance(ts["start"], int), f"'start' should be int, got {type(ts['start'])}"
        assert isinstance(ts["end"], int), f"'end' should be int, got {type(ts['end'])}"
