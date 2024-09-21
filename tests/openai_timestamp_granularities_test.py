"""OpenAI's handling of `response_format` and `timestamp_granularities` is a bit confusing and inconsistent. This test module exists to capture the OpenAI API's behavior with respect to these parameters."""  # noqa: E501

from faster_whisper_server.server_models import TIMESTAMP_GRANULARITIES_COMBINATIONS, TimestampGranularities
from openai import AsyncOpenAI, BadRequestError
import pytest


@pytest.mark.asyncio()
@pytest.mark.requires_openai()
@pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
async def test_openai_json_response_format_and_timestamp_granularities_combinations(
    actual_openai_client: AsyncOpenAI,
    timestamp_granularities: TimestampGranularities,
) -> None:
    audio_file = open("audio.wav", "rb")  # noqa: SIM115, ASYNC230

    if "word" in timestamp_granularities:
        with pytest.raises(BadRequestError):
            await actual_openai_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="json",
                timestamp_granularities=timestamp_granularities,
            )
    else:
        await actual_openai_client.audio.transcriptions.create(
            file=audio_file, model="whisper-1", response_format="json", timestamp_granularities=timestamp_granularities
        )


@pytest.mark.asyncio()
@pytest.mark.requires_openai()
@pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
async def test_openai_verbose_json_response_format_and_timestamp_granularities_combinations(
    actual_openai_client: AsyncOpenAI,
    timestamp_granularities: TimestampGranularities,
) -> None:
    audio_file = open("audio.wav", "rb")  # noqa: SIM115, ASYNC230

    transcription = await actual_openai_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=timestamp_granularities,
    )

    assert transcription.__pydantic_extra__
    if timestamp_granularities == ["word"]:
        # This is an exception where segments are not present
        assert transcription.__pydantic_extra__.get("segments") is None
        assert transcription.__pydantic_extra__.get("words") is not None
    elif "word" in timestamp_granularities:
        assert transcription.__pydantic_extra__.get("segments") is not None
        assert transcription.__pydantic_extra__.get("words") is not None
    else:
        # Unless explicitly requested, words are not present
        assert transcription.__pydantic_extra__.get("segments") is not None
        assert transcription.__pydantic_extra__.get("words") is None
