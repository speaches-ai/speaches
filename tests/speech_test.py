import base64
import io
import json
import time

from httpx import AsyncClient
from openai import AsyncOpenAI, UnprocessableEntityError
import openai.types.audio
import pydub
import pytest
import soundfile as sf

from speaches.api_types import (
    DEFAULT_SPEECH_RESPONSE_FORMAT,
    SUPPORTED_SPEECH_RESPONSE_FORMATS,
    SpeechResponseFormat,
)

SPEECH_MODEL_ID = "speaches-ai/Kokoro-82M-v1.0-ONNX"
VOICE_ID = "af_heart"
DEFAULT_INPUT = "Hello, world!"


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", SUPPORTED_SPEECH_RESPONSE_FORMATS)
async def test_create_speech_formats(openai_client: AsyncOpenAI, response_format: SpeechResponseFormat) -> None:
    await openai_client.audio.speech.create(
        model=SPEECH_MODEL_ID,
        voice=VOICE_ID,
        input=DEFAULT_INPUT,
        response_format=response_format,
    )


GOOD_MODEL_VOICE_PAIRS: list[tuple[str, str]] = [
    ("tts-1", "alloy"),  # OpenAI and OpenAI
    ("tts-1-hd", "echo"),  # OpenAI and OpenAI
    ("tts-1", VOICE_ID),  # OpenAI and Piper
    (SPEECH_MODEL_ID, "echo"),  # Piper and OpenAI
    (SPEECH_MODEL_ID, VOICE_ID),  # Piper and Piper
]


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "voice"), GOOD_MODEL_VOICE_PAIRS)
async def test_create_speech_good_model_voice_pair(openai_client: AsyncOpenAI, model: str, voice: str) -> None:
    await openai_client.audio.speech.create(
        model=model,
        voice=voice,
        input=DEFAULT_INPUT,
        response_format=DEFAULT_SPEECH_RESPONSE_FORMAT,
    )


BAD_MODEL_VOICE_PAIRS: list[tuple[str, str]] = [
    ("tts-1", "invalid"),  # OpenAI and invalid
    ("invalid", "echo"),  # Invalid and OpenAI
    (SPEECH_MODEL_ID, "invalid"),  # Piper and invalid
    ("invalid", VOICE_ID),  # Invalid and Piper
    ("invalid", "invalid"),  # Invalid and invalid
]


# @pytest.mark.asyncio
# @pytest.mark.parametrize(("model", "voice"), BAD_MODEL_VOICE_PAIRS)
# async def test_create_speech_bad_model_voice_pair(openai_client: AsyncOpenAI, model: str, voice: str) -> None:
#     # NOTE: not sure why `APIConnectionError` is sometimes raised
#     with pytest.raises((UnprocessableEntityError, APIConnectionError)):
#         await openai_client.audio.speech.create(
#             model=model,
#             voice=voice,
#             input=DEFAULT_INPUT,
#             response_format=DEFAULT_RESPONSE_FORMAT,
#         )


SUPPORTED_SPEEDS = [0.5, 1.0, 2.0]


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_create_speech_with_varying_speed(openai_client: AsyncOpenAI) -> None:
    previous_size: int | None = None
    for speed in SUPPORTED_SPEEDS:
        res = await openai_client.audio.speech.create(
            model=SPEECH_MODEL_ID,
            voice=VOICE_ID,
            input=DEFAULT_INPUT,
            response_format="pcm",
            speed=speed,
        )
        audio_bytes = res.read()
        if previous_size is not None:
            assert len(audio_bytes) * 1.5 < previous_size  # TODO: document magic number
        previous_size = len(audio_bytes)


# UNSUPPORTED_SPEEDS = [0.1, 4.1]
#
#
# @pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
# @pytest.mark.usefixtures("pull_model_without_cleanup")
# @pytest.mark.asyncio
# @pytest.mark.parametrize("speed", UNSUPPORTED_SPEEDS)
# async def test_create_speech_with_unsupported_speed(openai_client: AsyncOpenAI, speed: float) -> None:
#     with pytest.raises(UnprocessableEntityError):
#         await openai_client.audio.speech.create(
#             model=SPEECH_MODEL_ID,
#             voice=VOICE_ID,
#             input=DEFAULT_INPUT,
#             response_format="pcm",
#             speed=speed,
#         )


VALID_SAMPLE_RATES = [16000, 22050, 24000, 48000]


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", VALID_SAMPLE_RATES)
async def test_speech_valid_resample(openai_client: AsyncOpenAI, sample_rate: int) -> None:
    res = await openai_client.audio.speech.create(
        model=SPEECH_MODEL_ID,
        voice=VOICE_ID,
        input=DEFAULT_INPUT,
        response_format="wav",
        extra_body={"sample_rate": sample_rate},
    )
    _, actual_sample_rate = sf.read(io.BytesIO(res.content))
    assert actual_sample_rate == sample_rate


INVALID_SAMPLE_RATES = [7999, 48001]


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", INVALID_SAMPLE_RATES)
async def test_speech_invalid_resample(openai_client: AsyncOpenAI, sample_rate: int) -> None:
    with pytest.raises(UnprocessableEntityError):
        await openai_client.audio.speech.create(
            model=SPEECH_MODEL_ID,
            voice=VOICE_ID,
            input=DEFAULT_INPUT,
            response_format="wav",
            extra_body={"sample_rate": sample_rate},
        )


_OPENAI_SPEECH_MODELS = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
_OPUS_SUPPORTED_SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)
_OPUS_PREFERRED_SAMPLE_RATE = 48000


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("speech_model_id", _OPENAI_SPEECH_MODELS)
async def test_openai_speech_opus_sample_rate(
    actual_openai_client: AsyncOpenAI, speech_model_id: openai.types.audio.SpeechModel
) -> None:
    res = await actual_openai_client.audio.speech.create(
        model=speech_model_id,
        voice="alloy",
        input=DEFAULT_INPUT,
        response_format="opus",
    )

    audio_data_bytes = res.content

    # Read opus audio using pydub
    audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_data_bytes), format="ogg", codec="libopus")

    assert audio_segment.frame_rate in _OPUS_SUPPORTED_SAMPLE_RATES
    assert audio_segment.frame_rate == _OPUS_PREFERRED_SAMPLE_RATE


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_speaches_speech_opus_sample_rate(
    openai_client: AsyncOpenAI,
) -> None:
    res = await openai_client.audio.speech.create(
        model=SPEECH_MODEL_ID,
        voice=VOICE_ID,
        input=DEFAULT_INPUT,
        response_format="opus",
    )

    audio_data_bytes = res.content

    # Read opus audio using pydub
    audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_data_bytes), format="ogg", codec="libopus")

    assert audio_segment.frame_rate in _OPUS_SUPPORTED_SAMPLE_RATES
    assert audio_segment.frame_rate == _OPUS_PREFERRED_SAMPLE_RATE


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("speech_model_id", _OPENAI_SPEECH_MODELS[:1])
@pytest.mark.parametrize("response_format", SUPPORTED_SPEECH_RESPONSE_FORMATS)
async def test_openai_speech_chunk_transfer_encoding(
    actual_openai_client: AsyncOpenAI,
    speech_model_id: openai.types.audio.SpeechModel,
    response_format: SpeechResponseFormat,
) -> None:
    start = time.perf_counter()
    async with actual_openai_client.audio.speech.with_streaming_response.create(
        model=speech_model_id,
        voice="alloy",
        input="How could trying to make sentences sound good help you do that? The clue to the answer is something I noticed 30 years ago when I was doing the layout for my first book. Sometimes when you're laying out text you have bad luck. For example, you get a section that runs one line longer than the page. I don't know what ordinary typesetters do in this situation, but what I did was rewrite the section to make it a line shorter. You'd expect such an arbitrary constraint to make the writing worse. But I found, to my surprise, that it never did. I always ended up with something I liked better.",
        response_format=response_format,
    ) as res:
        chunk_time_pairs = [(chunk, time.perf_counter() - start) async for chunk in res.iter_bytes()]

    print(
        f"Test for model {speech_model_id} with response format {response_format}. Res headers content-type: {res.headers.get('content-type')}"
    )
    percentiles = [1, 10, 25, 50, 75, 90, 95, 99]
    sorted_chunk_sizes = sorted(len(chunk) for chunk, _ in chunk_time_pairs)
    sorted_elapsed_times = sorted(elapsed for _chunk, elapsed in chunk_time_pairs)
    percentile_chunk_sizes = {}
    percentile_elapsed_times = {}
    for p in percentiles:
        index = max(0, len(sorted_chunk_sizes) * p // 100 - 1)
        percentile_chunk_sizes[p] = sorted_chunk_sizes[index]

        index = max(0, len(sorted_elapsed_times) * p // 100 - 1)
        percentile_elapsed_times[p] = sorted_elapsed_times[index]
    print(
        f"Total bytes: {sum(len(x) for x, _ in chunk_time_pairs)}, Total chunks: {len(chunk_time_pairs)}, \nPercentile chunk sizes: {percentile_chunk_sizes}\nPercentile elapsed times: {percentile_elapsed_times}"
    )


def validate_audio_reconstruction(  # noqa: RET503
    audio_bytes: bytes, sample_rate: int, response_format: SpeechResponseFormat
) -> tuple[float, int]:
    if response_format == "pcm":
        data, _ = sf.read(
            io.BytesIO(audio_bytes),
            samplerate=sample_rate,
            format="RAW",
            channels=1,
            subtype="PCM_16",
            endian="LITTLE",
            dtype="int16",
        )
        assert len(data) > 0, "PCM audio data is empty"
        duration = len(data) / sample_rate
        bit_depth = 16
        return duration, bit_depth
    elif response_format in ("mp3", "wav", "flac"):
        audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
        assert len(audio_segment) > 0, f"{response_format.upper()} audio is empty"
        assert audio_segment.frame_rate > 0, f"Invalid sample rate: {audio_segment.frame_rate}"
        duration = len(audio_segment) / 1000.0
        bit_depth = audio_segment.sample_width * 8
        return duration, bit_depth
    elif response_format == "opus":
        audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes), format="ogg", codec="libopus")
        assert len(audio_segment) > 0, "OPUS audio is empty"
        assert audio_segment.frame_rate in _OPUS_SUPPORTED_SAMPLE_RATES, (
            f"Invalid OPUS sample rate: {audio_segment.frame_rate}"
        )
        duration = len(audio_segment) / 1000.0
        bit_depth = audio_segment.sample_width * 8
        return duration, bit_depth
    elif response_format == "aac":
        audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes), format="aac")
        assert len(audio_segment) > 0, "AAC audio is empty"
        assert audio_segment.frame_rate > 0, f"Invalid AAC sample rate: {audio_segment.frame_rate}"
        duration = len(audio_segment) / 1000.0
        bit_depth = audio_segment.sample_width * 8
        return duration, bit_depth


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_speaches_speech_chunk_transfer_encoding(openai_client: AsyncOpenAI) -> None:
    test_input = "How could trying to make sentences sound good help you do that? The clue to the answer is something I noticed 30 years ago when I was doing the layout for my first book. Sometimes when you're laying out text you have bad luck. For example, you get a section that runs one line longer than the page. I don't know what ordinary typesetters do in this situation, but what I did was rewrite the section to make it a line shorter. You'd expect such an arbitrary constraint to make the writing worse. But I found, to my surprise, that it never did."

    format_results: dict[SpeechResponseFormat, tuple[float, int]] = {}

    for response_format in SUPPORTED_SPEECH_RESPONSE_FORMATS:
        start = time.perf_counter()
        async with openai_client.audio.speech.with_streaming_response.create(
            model=SPEECH_MODEL_ID,
            voice=VOICE_ID,
            input=test_input,
            response_format=response_format,
        ) as res:
            chunk_time_pairs = [(chunk, time.perf_counter() - start) async for chunk in res.iter_bytes()]

        print(
            f"\nTest for model {SPEECH_MODEL_ID} with response format {response_format}. Res headers content-type: {res.headers.get('content-type')}"
        )
        percentiles = [1, 10, 25, 50, 75, 90, 95, 99]
        sorted_chunk_sizes = sorted(len(chunk) for chunk, _ in chunk_time_pairs)
        sorted_elapsed_times = sorted(elapsed for _chunk, elapsed in chunk_time_pairs)
        percentile_chunk_sizes = {}
        percentile_elapsed_times = {}
        for p in percentiles:
            index = max(0, len(sorted_chunk_sizes) * p // 100 - 1)
            percentile_chunk_sizes[p] = sorted_chunk_sizes[index]

            index = max(0, len(sorted_elapsed_times) * p // 100 - 1)
            percentile_elapsed_times[p] = sorted_elapsed_times[index]
        print(
            f"Total bytes: {sum(len(x) for x, _ in chunk_time_pairs)}, Total chunks: {len(chunk_time_pairs)}, \nPercentile chunk sizes: {percentile_chunk_sizes}\nPercentile elapsed times: {percentile_elapsed_times}"
        )

        audio_bytes = b"".join(chunk for chunk, _ in chunk_time_pairs)
        duration, bit_depth = validate_audio_reconstruction(audio_bytes, 24000, response_format)
        format_results[response_format] = (duration, bit_depth)

        print(f"Duration: {duration:.2f}s, Bit depth: {bit_depth} bits")
        assert bit_depth == 16, f"Expected 16-bit audio, got {bit_depth}-bit for {response_format}"

    durations = [duration for duration, _ in format_results.values()]
    min_duration = min(durations)
    max_duration = max(durations)
    duration_diff = max_duration - min_duration

    print("\nDuration comparison across formats:")
    for fmt, (duration, bit_depth) in format_results.items():
        print(f"  {fmt}: {duration:.3f}s ({bit_depth}-bit)")
    print(f"Duration range: {min_duration:.3f}s - {max_duration:.3f}s (diff: {duration_diff:.3f}s)")

    assert duration_diff < 0.5, f"Duration difference {duration_diff:.3f}s exceeds threshold of 0.5s across formats"


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", SUPPORTED_SPEECH_RESPONSE_FORMATS)
async def test_speech_sse_stream_format(aclient: AsyncClient, response_format: SpeechResponseFormat) -> None:
    res = await aclient.post(
        "/v1/audio/speech",
        json={
            "model": SPEECH_MODEL_ID,
            "voice": VOICE_ID,
            "input": DEFAULT_INPUT,
            "response_format": response_format,
            "stream_format": "sse",
        },
    )

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = []
    lines = res.text.strip().split("\n\n")
    for line_block in lines:
        if line_block.startswith("data: "):
            event_data = line_block[6:]
            events.append(json.loads(event_data))

    assert len(events) > 0, "Expected at least one SSE event"

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    done_events = [e for e in events if e.get("type") == "speech.audio.done"]

    assert len(delta_events) > 0, "Expected at least one speech.audio.delta event"
    assert len(done_events) == 1, "Expected exactly one speech.audio.done event"

    for delta_event in delta_events:
        assert "audio" in delta_event, "Delta event missing audio field"
        audio_b64 = delta_event["audio"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0, "Audio chunk should not be empty"

    done_event = done_events[0]
    assert "token_usage" in done_event, "Done event missing token_usage field"
    assert "input_tokens" in done_event["token_usage"]
    assert "output_tokens" in done_event["token_usage"]
    assert "total_tokens" in done_event["token_usage"]


@pytest.mark.parametrize("pull_model_without_cleanup", [SPEECH_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_speech_default_stream_format_is_audio(aclient: AsyncClient) -> None:
    res = await aclient.post(
        "/v1/audio/speech",
        json={
            "model": SPEECH_MODEL_ID,
            "voice": VOICE_ID,
            "input": DEFAULT_INPUT,
            "response_format": "pcm",
        },
    )

    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/pcm"
    assert len(res.content) > 0


# TODO: add piper tests

# TODO: implement the following test

# NUMBER_OF_MODELS = 1
# NUMBER_OF_VOICES = 124
#
#
# @pytest.mark.asyncio
# async def test_list_tts_models(openai_client: AsyncOpenAI) -> None:
#     raise NotImplementedError
