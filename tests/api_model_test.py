import openai
from openai import AsyncOpenAI
import pytest

MODEL_THAT_EXISTS = "Systran/faster-whisper-tiny.en"
MODEL_THAT_DOES_NOT_EXIST = "i-do-not-exist"
MIN_EXPECTED_NUMBER_OF_MODELS = 70  # At the time of the test creation there are 89 models


# TODO: re-enable this test. Was disabled as `POST /v1/models` only lists local models
# @pytest.mark.asyncio
# async def test_list_models(openai_client: AsyncOpenAI) -> None:
#     models = (await openai_client.models.list()).data
#     assert len(models) > MIN_EXPECTED_NUMBER_OF_MODELS


@pytest.mark.asyncio
async def test_model_exists(openai_client: AsyncOpenAI) -> None:
    model = await openai_client.models.retrieve(MODEL_THAT_EXISTS)
    assert model.id == MODEL_THAT_EXISTS


@pytest.mark.asyncio
async def test_model_does_not_exist(openai_client: AsyncOpenAI) -> None:
    with pytest.raises(openai.NotFoundError):
        await openai_client.models.retrieve(MODEL_THAT_DOES_NOT_EXIST)
