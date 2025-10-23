import asyncio
import base64
import logging
import os

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session_update_event_param import (
    Session,
    SessionInputAudioTranscription,
    SessionTurnDetection,
)
import sounddevice as sd
import websockets

from speaches.logger import setup_logger

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
BYTERATE = SAMPLE_RATE * SAMPLE_WIDTH  # like "bitrate" but in bytes
MINIMUM_AUDIO_CHUNK_SIZE = 1024
DTYPE = "int16"
CHANNELS = 1
TRANSCRIPTION_MODEL = "Systran/faster-distil-whisper-small.en"
NO_AUDIO_TIMEOUT_SECONDS = 0.5

SPEACHES_BASE_URL = os.environ.get("SPEACHES_BASE_URL", "http://localhost:8000")
WEBSOCKET_BASE_URL = SPEACHES_BASE_URL.replace("http", "ws") + "/v1"


setup_logger("INFO")
logger = logging.getLogger(__name__)


async def print_events(conn: AsyncRealtimeConnection, final_event: str | None = None) -> None:
    try:
        async for event in conn:
            if event.type == "response.audio.delta":
                size = len(base64.b64decode(event.delta))
                event.delta = f"base64 encoded audio of size {size} bytes"
            print(event.model_dump_json())
            if final_event is not None and event.type == final_event:
                break
    except websockets.exceptions.ConnectionClosedError:
        logger.info("Connection closed")


async def send_mic_audio(
    connection: AsyncRealtimeConnection, *, no_audio_timeout_seconds: float | None = NO_AUDIO_TIMEOUT_SECONDS
) -> None:
    stream = sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype=DTYPE)
    stream.start()

    last_audio_time = asyncio.get_event_loop().time()

    try:
        while True:
            if stream.read_available < MINIMUM_AUDIO_CHUNK_SIZE:
                if stream.read_available == 0:
                    if no_audio_timeout_seconds is not None:
                        elapsed = asyncio.get_event_loop().time() - last_audio_time
                        if elapsed >= no_audio_timeout_seconds:
                            logger.warning(f"No audio for {elapsed:.2f} seconds, exiting.")
                            break
                await asyncio.sleep(0)
                continue

            data, _ = stream.read(MINIMUM_AUDIO_CHUNK_SIZE)
            last_audio_time = asyncio.get_event_loop().time()

            bytes_data = data.tobytes()
            assert len(bytes_data) == len(data) * SAMPLE_WIDTH
            await connection.input_audio_buffer.append(audio=base64.b64encode(bytes_data).decode())

    except asyncio.CancelledError:
        pass
    finally:
        stream.stop()
        stream.close()


async def main() -> None:
    devices = sd.query_devices()
    logger.info(f"Available audio devices:\n{devices}")

    logger.info(f"Using audio input device: {sd.query_devices(sd.default.device[0])}")
    logger.info(f"Using audio output device: {sd.query_devices(sd.default.device[1])}")

    # NOTE: below can be used to set specific input/output devices
    # sd.default.device = (0, 1)  # (input_id, output_id)

    realtime_client = AsyncOpenAI(
        api_key="does-not-matter", websocket_base_url=WEBSOCKET_BASE_URL, max_retries=0
    ).beta.realtime
    async with asyncio.TaskGroup() as tg:
        try:
            async with realtime_client.connect(model="does-not-matter") as conn:
                tg.create_task(print_events(conn, final_event=None))
                await conn.session.update(
                    session=Session(
                        input_audio_transcription=SessionInputAudioTranscription(
                            model=TRANSCRIPTION_MODEL  # controls the transcription model used
                        ),
                        turn_detection=SessionTurnDetection(
                            silence_duration_ms=1500,  # Shouldn't exceed 2500 due to how the current implementation works.
                            threshold=0.9,  # I've found this to be a good default value.
                            create_response=False,  # Ensures that the session is only used for audio transcription.
                        ),
                    )
                )
                await send_mic_audio(conn)
        except websockets.WebSocketException:
            logger.exception(f"Failed to connect to {WEBSOCKET_BASE_URL}")


if __name__ == "__main__":
    asyncio.run(main())
