from pathlib import Path

from openai import AsyncOpenAI
import pytest

from speaches.routers.stt import RESPONSE_FORMATS


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("response_format", RESPONSE_FORMATS)
async def test_openai_supported_formats_for_non_whisper_models(
    actual_openai_client: AsyncOpenAI,
    response_format: str,
) -> None:
    file_path = Path("audio.wav")
    transcription_event_stream = await actual_openai_client.audio.transcriptions.create(  # pyright: ignore[reportCallIssue]
        file=file_path,
        model="gpt-4o-transcribe",
        response_format=response_format,  # pyright: ignore[reportArgumentType]
        stream=True,
    )
    async for event in transcription_event_stream:
        print(event)


# @pytest.mark.asyncio
# @pytest.mark.requires_openai
# @pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
# async def test_openai_verbose_json_response_format_and_timestamp_granularities_combinations(
#     actual_openai_client: AsyncOpenAI,
#     timestamp_granularities: TimestampGranularities,
# ) -> None:
#     file_path = Path("audio.wav")
#
#     transcription = await actual_openai_client.audio.transcriptions.create(
#         file=file_path,
#         model="whisper-1",
#         response_format="verbose_json",
#         timestamp_granularities=timestamp_granularities,
#     )
#
#     if timestamp_granularities == ["word"]:
#         # This is an exception where segments are not present
#         assert transcription.segments is None
#         assert transcription.words is not None
#     elif "word" in timestamp_granularities:
#         assert transcription.segments is not None
#         assert transcription.words is not None
#     else:
#         # Unless explicitly requested, words are not present
#         assert transcription.segments is not None
#         assert transcription.words is None
