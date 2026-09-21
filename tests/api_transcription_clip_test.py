"""Tests for the non-standard `start`/`duration` form parameters of `/v1/audio/transcriptions`.

They transcribe only a clip of the uploaded audio; timestamps in the response are relative to the clip.
"""

from pathlib import Path

from httpx import AsyncClient
import pytest

TRANSCRIPTION_MODEL_ID = "Systran/faster-whisper-tiny.en"
AUDIO_FILE_PATH = Path("audio.wav")


async def transcribe(aclient: AsyncClient, **data: str) -> dict:  # noqa: ANN401
    with AUDIO_FILE_PATH.open("rb") as f:
        res = await aclient.post(
            "/v1/audio/transcriptions",
            files={"file": f},
            data={"model": TRANSCRIPTION_MODEL_ID, "response_format": "verbose_json", **data},
        )
    return {"status_code": res.status_code, "body": res.json()}


@pytest.mark.parametrize("pull_model_without_cleanup", [TRANSCRIPTION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_clip_is_shorter_and_timestamps_start_at_zero(aclient: AsyncClient) -> None:
    full = await transcribe(aclient)
    clip = await transcribe(aclient, start="1.0", duration="2.0")
    assert full["status_code"] == 200
    assert clip["status_code"] == 200
    assert clip["body"]["duration"] == pytest.approx(2.0, abs=0.05)
    assert clip["body"]["duration"] < full["body"]["duration"]
    assert all(segment["start"] < 2.0 for segment in clip["body"]["segments"])


@pytest.mark.parametrize("pull_model_without_cleanup", [TRANSCRIPTION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_clip_past_the_end_is_truncated(aclient: AsyncClient) -> None:
    full = await transcribe(aclient)
    clip = await transcribe(aclient, start="1.0", duration="100000")
    assert clip["status_code"] == 200
    assert clip["body"]["duration"] == pytest.approx(full["body"]["duration"] - 1.0, abs=0.05)


@pytest.mark.parametrize("pull_model_without_cleanup", [TRANSCRIPTION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {"start": "-1"},
        {"duration": "0"},
        {"duration": "-5"},
        {"duration": "100000"},
        {"start": "100000"},
    ],
)
async def test_invalid_clip_is_rejected(aclient: AsyncClient, data: dict[str, str]) -> None:
    res = await transcribe(aclient, **data)
    assert res["status_code"] == 400, res["body"]
