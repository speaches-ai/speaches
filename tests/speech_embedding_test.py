from pathlib import Path

from httpx import AsyncClient
import numpy as np
import pytest
import soundfile as sf

EMBEDDING_MODEL_ID = "deepghs/pyannote-embedding-onnx"


@pytest.mark.parametrize("pull_model_without_cleanup", [EMBEDDING_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_create_speech_embedding(aclient: AsyncClient, tmp_path: Path) -> None:
    sample_rate = 16000
    duration = 3
    audio_data = np.random.randn(sample_rate * duration).astype(np.float32)
    audio_file = tmp_path / "test_audio.wav"
    sf.write(audio_file, audio_data, sample_rate)

    with open(audio_file, "rb") as f:
        response = await aclient.post(
            "/v1/audio/speech/embedding",
            data={"model": EMBEDDING_MODEL_ID},
            files={"file": ("test_audio.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    result = response.json()

    assert result["object"] == "list"
    assert result["model"] == EMBEDDING_MODEL_ID
    assert len(result["data"]) == 1
    assert result["data"][0]["object"] == "embedding"
    assert result["data"][0]["index"] == 0
    assert isinstance(result["data"][0]["embedding"], list)
    assert len(result["data"][0]["embedding"]) == 512
    assert all(isinstance(x, float) for x in result["data"][0]["embedding"])
    assert "usage" in result
    assert "prompt_tokens" in result["usage"]
    assert "total_tokens" in result["usage"]


@pytest.mark.parametrize("pull_model_without_cleanup", [EMBEDDING_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_create_speech_embedding_with_real_audio(aclient: AsyncClient, tmp_path: Path) -> None:
    sample_rate = 16000
    duration = 2
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    audio_file = tmp_path / "sine_wave.wav"
    sf.write(audio_file, audio_data, sample_rate)

    with open(audio_file, "rb") as f:
        response = await aclient.post(
            "/v1/audio/speech/embedding",
            data={"model": EMBEDDING_MODEL_ID},
            files={"file": ("sine_wave.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    result = response.json()

    assert result["object"] == "list"
    assert len(result["data"]) == 1
    embedding = result["data"][0]["embedding"]
    assert len(embedding) == 512


@pytest.mark.asyncio
async def test_create_speech_embedding_model_not_found(aclient: AsyncClient, tmp_path: Path) -> None:
    sample_rate = 16000
    audio_data = np.random.randn(sample_rate).astype(np.float32)
    audio_file = tmp_path / "test_audio.wav"
    sf.write(audio_file, audio_data, sample_rate)

    with open(audio_file, "rb") as f:
        response = await aclient.post(
            "/v1/audio/speech/embedding",
            data={"model": "non-existent-model"},
            files={"file": ("test_audio.wav", f, "audio/wav")},
        )

    assert response.status_code == 404


@pytest.mark.parametrize("pull_model_without_cleanup", [EMBEDDING_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_embedding_similarity(aclient: AsyncClient, tmp_path: Path) -> None:
    sample_rate = 16000
    duration = 2
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data_1 = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    audio_file_1 = tmp_path / "audio_1.wav"
    sf.write(audio_file_1, audio_data_1, sample_rate)

    audio_data_2 = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    audio_file_2 = tmp_path / "audio_2.wav"
    sf.write(audio_file_2, audio_data_2, sample_rate)

    with open(audio_file_1, "rb") as f:
        response_1 = await aclient.post(
            "/v1/audio/speech/embedding",
            data={"model": EMBEDDING_MODEL_ID},
            files={"file": ("audio_1.wav", f, "audio/wav")},
        )

    with open(audio_file_2, "rb") as f:
        response_2 = await aclient.post(
            "/v1/audio/speech/embedding",
            data={"model": EMBEDDING_MODEL_ID},
            files={"file": ("audio_2.wav", f, "audio/wav")},
        )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    embedding_1 = np.array(response_1.json()["data"][0]["embedding"])
    embedding_2 = np.array(response_2.json()["data"][0]["embedding"])

    cosine_similarity = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

    assert cosine_similarity > 0.99
