from io import BytesIO
import wave

from openai import AsyncOpenAI
import pytest

TRANSCRIPTION_MODEL_ID = "Systran/faster-whisper-tiny.en"
SAMPLE_RATE = 16000
# Must exceed faster-whisper's 30s `chunk_length`. Shorter clips never reach
# the failing branch, because faster-whisper falls back to transcribing the
# whole clip when no clip timestamps are given.
SILENCE_DURATION_S = 45


def generate_silence() -> BytesIO:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * (SAMPLE_RATE * SILENCE_DURATION_S))
    buffer.name = "silence.wav"
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("pull_model_without_cleanup", [TRANSCRIPTION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_transcription_of_audio_without_speech(openai_client: AsyncOpenAI) -> None:
    transcription = await openai_client.audio.transcriptions.create(
        file=generate_silence(),
        model=TRANSCRIPTION_MODEL_ID,
        response_format="json",
    )

    assert transcription.text == ""


@pytest.mark.parametrize("pull_model_without_cleanup", [TRANSCRIPTION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_verbose_json_transcription_of_audio_without_speech(openai_client: AsyncOpenAI) -> None:
    transcription = await openai_client.audio.transcriptions.create(
        file=generate_silence(),
        model=TRANSCRIPTION_MODEL_ID,
        response_format="verbose_json",
    )

    assert transcription.text == ""
    assert transcription.segments == []
    assert transcription.words is None


@pytest.mark.parametrize("pull_model_without_cleanup", [TRANSCRIPTION_MODEL_ID], indirect=True)
@pytest.mark.usefixtures("pull_model_without_cleanup")
@pytest.mark.asyncio
async def test_streaming_transcription_of_audio_without_speech(openai_client: AsyncOpenAI) -> None:
    transcription_event_stream = (
        await openai_client.audio.transcriptions.create(  # pyrefly: ignore[no-matching-overload]
            file=generate_silence(),
            model=TRANSCRIPTION_MODEL_ID,
            response_format="json",
            stream=True,
        )
    )

    events = [event async for event in transcription_event_stream]

    assert len(events) == 1
    assert events[0].type == "transcript.text.done"
    assert events[0].text == ""
