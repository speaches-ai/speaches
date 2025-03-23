from pathlib import Path

import anyio
from httpx import AsyncClient
import pytest

from speaches.routers.vad import MODEL_ID, SpeechTimestamp

FILE_PATH = "audio.wav"
ENDPOINT = "/v1/audio/speech/timestamps"


@pytest.mark.asyncio
async def test_speech_timestamps_basic(aclient: AsyncClient) -> None:
    extension = Path(FILE_PATH).suffix[1:]
    async with await anyio.open_file(FILE_PATH, "rb") as f:
        data = await f.read()
    res = await aclient.post(
        ENDPOINT, files={"file": (f"audio.{extension}", data, f"audio/{extension}")}, data={"model": MODEL_ID}
    )
    res.raise_for_status()
    data = res.json()
    speech_timestamps = [SpeechTimestamp.model_validate(x) for x in data]
    assert len(speech_timestamps) == 1


# TODO: add more tests
