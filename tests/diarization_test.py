from pathlib import Path

import anyio
from httpx import AsyncClient
import numpy as np
import pytest
import soundfile as sf

from speaches.routers.diarization import DiarizationResponse

DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
ENDPOINT = "/v1/audio/diarization"
SAMPLE_RATE = 16000


def make_sine_wav(path: Path, frequency: float, duration: float, sample_rate: int = SAMPLE_RATE) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, data, sample_rate)


@pytest.mark.requires_gated_hf_model
@pytest.mark.parametrize("pull_model_without_cleanup", [DIARIZATION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_diarize_json_response_structure(aclient: AsyncClient, tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    make_sine_wav(audio_file, frequency=440, duration=10)

    async with await anyio.open_file(audio_file, "rb") as f:
        data = await f.read()

    response = await aclient.post(
        ENDPOINT,
        files={"file": ("audio.wav", data, "audio/wav")},
        data={"response_format": "json", "model": DIARIZATION_MODEL_ID},
    )

    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}. Response content: {response.text}"
    )

    result = DiarizationResponse.model_validate(response.json())
    assert result.duration == pytest.approx(10.0, abs=0.5)
    for segment in result.segments:
        assert segment.start >= 0
        assert segment.end > segment.start
        assert segment.end <= result.duration + 0.1
        assert isinstance(segment.speaker, str)
        assert len(segment.speaker) > 0


@pytest.mark.requires_gated_hf_model
@pytest.mark.parametrize("pull_model_without_cleanup", [DIARIZATION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_diarize_rttm_response_format(aclient: AsyncClient, tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    make_sine_wav(audio_file, frequency=440, duration=10)

    async with await anyio.open_file(audio_file, "rb") as f:
        data = await f.read()

    response = await aclient.post(
        ENDPOINT,
        files={"file": ("audio.wav", data, "audio/wav")},
        data={"response_format": "rttm", "model": DIARIZATION_MODEL_ID},
    )

    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}. Response content: {response.text}"
    )

    assert response.headers["content-type"].startswith("text/plain")
    for line in response.text.strip().splitlines():
        parts = line.split()
        assert parts[0] == "SPEAKER"
        assert len(parts) == 10
        assert float(parts[3]) >= 0  # start time
        assert float(parts[4]) > 0  # duration


@pytest.mark.requires_gated_hf_model
@pytest.mark.parametrize("pull_model_without_cleanup", [DIARIZATION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_diarize_real_audio(aclient: AsyncClient) -> None:
    async with await anyio.open_file("audio.wav", "rb") as f:
        data = await f.read()

    response = await aclient.post(
        ENDPOINT,
        files={"file": ("audio.wav", data, "audio/wav")},
        data={"response_format": "json", "model": DIARIZATION_MODEL_ID},
    )

    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}. Response content: {response.text}"
    )

    result = DiarizationResponse.model_validate(response.json())
    assert result.duration > 0
    assert len(result.segments) > 0


@pytest.mark.requires_gated_hf_model
@pytest.mark.parametrize("pull_model_without_cleanup", [DIARIZATION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_diarize_default_response_format_is_json(aclient: AsyncClient, tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    make_sine_wav(audio_file, frequency=440, duration=5)

    async with await anyio.open_file(audio_file, "rb") as f:
        data = await f.read()

    response = await aclient.post(
        ENDPOINT,
        files={"file": ("audio.wav", data, "audio/wav")},
        data={"model": DIARIZATION_MODEL_ID},
    )

    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}. Response content: {response.text}"
    )
    assert response.headers["content-type"].startswith("application/json")
    DiarizationResponse.model_validate(response.json())
