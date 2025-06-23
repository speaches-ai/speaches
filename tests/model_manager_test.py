import asyncio

import anyio
import pytest

from speaches.config import Config, WhisperConfig
from tests.conftest import AclientFactory

MODEL_ID = "Systran/faster-whisper-tiny.en"


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_model_unloaded_after_ttl(aclient_factory: AclientFactory) -> None:
    ttl = 5
    config = Config(whisper=WhisperConfig(ttl=ttl), enable_ui=False)
    async with aclient_factory(config) as aclient:
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0
        await aclient.post(f"/api/ps/{MODEL_ID}")
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1
        await asyncio.sleep(ttl + 1)  # wait for the model to be unloaded
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_ttl_resets_after_usage(aclient_factory: AclientFactory) -> None:
    ttl = 5
    config = Config(whisper=WhisperConfig(ttl=ttl), enable_ui=False)
    async with aclient_factory(config) as aclient:
        await aclient.post(f"/api/ps/{MODEL_ID}")
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1
        await asyncio.sleep(ttl - 2)  # sleep for less than the ttl. The model should not be unloaded
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1

        async with await anyio.open_file("audio.wav", "rb") as f:
            data = await f.read()
        res = (
            await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", data, "audio/wav")},
                data={"model": MODEL_ID},
            )
        ).json()
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1
        await asyncio.sleep(ttl - 2)  # sleep for less than the ttl. The model should not be unloaded
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1

        await asyncio.sleep(3)  # sleep for a bit more. The model should be unloaded
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0

        # test the model can be used again after being unloaded
        # this just ensures the model can be loaded again after being unloaded
        res = (
            await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", data, "audio/wav")},
                data={"model": MODEL_ID},
            )
        ).json()


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_model_cant_be_unloaded_when_used(aclient_factory: AclientFactory) -> None:
    ttl = 0
    config = Config(whisper=WhisperConfig(ttl=ttl), enable_ui=False)
    async with aclient_factory(config) as aclient:
        async with await anyio.open_file("audio.wav", "rb") as f:
            data = await f.read()

        task = asyncio.create_task(
            aclient.post(
                "/v1/audio/transcriptions", files={"file": ("audio.wav", data, "audio/wav")}, data={"model": MODEL_ID}
            )
        )
        await asyncio.sleep(0.1)  # wait for the server to start processing the request
        res = await aclient.delete(f"/api/ps/{MODEL_ID}")
        assert res.status_code == 409

        await task
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_model_cant_be_loaded_twice(aclient_factory: AclientFactory) -> None:
    ttl = -1
    config = Config(whisper=WhisperConfig(ttl=ttl), enable_ui=False)
    async with aclient_factory(config) as aclient:
        res = await aclient.post(f"/api/ps/{MODEL_ID}")
        assert res.status_code == 201
        res = await aclient.post(f"/api/ps/{MODEL_ID}")
        assert res.status_code == 409
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1


@pytest.mark.parametrize("pull_model_without_cleanup", [MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_model_is_unloaded_after_request_when_ttl_is_zero(aclient_factory: AclientFactory) -> None:
    ttl = 0
    config = Config(whisper=WhisperConfig(ttl=ttl), enable_ui=False)
    async with aclient_factory(config) as aclient:
        async with await anyio.open_file("audio.wav", "rb") as f:
            data = await f.read()
        res = await aclient.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", data, "audio/wav")},
            data={"model": "Systran/faster-whisper-tiny.en"},
        )
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0
