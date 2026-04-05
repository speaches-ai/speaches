import base64
from io import BytesIO
import logging
from typing import Literal

import openai

from speaches.audio import audio_samples_from_file, resample_audio_data
from speaches.executors.silero_vad_v5 import VadOptions, get_speech_timestamps, to_ms_speech_timestamps
from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer import (
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    MS_SAMPLE_RATE,
    InputAudioBuffer,
    InputAudioBufferTranscriber,
)
from speaches.types.realtime import (
    Error,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    Response,
    TurnDetection,
    create_invalid_request_error,
    create_server_error,
)

MIN_AUDIO_BUFFER_DURATION_MS = 100  # based on the OpenAI's API response

logger = logging.getLogger(__name__)

event_router = EventRouter()

empty_input_audio_buffer_commit_error = Error(
    type="invalid_request_error",
    message="Error committing input audio buffer: the buffer is empty.",
)

type SpeechTimestamp = dict[Literal["start", "end"], int]


def vad_detection_flow(
    input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection, ctx: SessionContext
) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]

    speech_timestamps = to_ms_speech_timestamps(
        get_speech_timestamps(
            audio_window,
            model_manager=ctx.vad_model_manager,
            vad_options=VadOptions(
                threshold=turn_detection.threshold,
                min_silence_duration_ms=turn_detection.silence_duration_ms,
                speech_pad_ms=turn_detection.prefix_padding_ms,
            ),
        )
    )
    if len(speech_timestamps) > 1:
        logger.warning(f"More than one speech timestamp: {speech_timestamps}")

    speech_timestamp = speech_timestamps[-1] if len(speech_timestamps) > 0 else None

    # logger.debug(f"Speech timestamps: {speech_timestamps}")
    if input_audio_buffer.vad_state.audio_start_ms is None:
        if speech_timestamp is None:
            return None
        input_audio_buffer.vad_state.audio_start_ms = (
            input_audio_buffer.duration_ms - len(audio_window) // MS_SAMPLE_RATE + speech_timestamp.start
        )
        return InputAudioBufferSpeechStartedEvent(
            item_id=input_audio_buffer.id,
            audio_start_ms=input_audio_buffer.vad_state.audio_start_ms,
        )

    else:  # noqa: PLR5501
        if speech_timestamp is None:
            # TODO: not quite correct. dependent on window size
            input_audio_buffer.vad_state.audio_end_ms = (
                input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms
            )
            return InputAudioBufferSpeechStoppedEvent(
                item_id=input_audio_buffer.id,
                audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
            )

        elif speech_timestamp.end < 3000 and input_audio_buffer.duration_ms > 3000:  # FIX: magic number
            input_audio_buffer.vad_state.audio_end_ms = (
                input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms
            )

            return InputAudioBufferSpeechStoppedEvent(
                item_id=input_audio_buffer.id,
                audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
            )

    return None


# Client Events


@event_router.register("input_audio_buffer.append")
async def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    audio_chunk = audio_samples_from_file(BytesIO(base64.b64decode(event.audio)), 24000)
    # convert the audio data from 24kHz (sample rate defined in the API spec) to 16kHz (sample rate used by the VAD and for transcription)
    audio_chunk = resample_audio_data(audio_chunk, 24000, 16000)
    input_audio_buffer = ctx.audio_buffers.current
    input_audio_buffer.append(audio_chunk)
    if ctx.session.turn_detection is not None:
        vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection, ctx)
        if vad_event is not None:
            ctx.pubsub.publish_nowait(vad_event)
            if isinstance(vad_event, InputAudioBufferSpeechStoppedEvent):
                item_id = vad_event.item_id
                ctx.audio_buffers.rotate()
                await commit_and_transcribe(ctx, item_id)


@event_router.register("input_audio_buffer.commit")
async def handle_input_audio_buffer_commit(ctx: SessionContext, _event: InputAudioBufferCommitEvent) -> None:
    input_audio_buffer = ctx.audio_buffers.current
    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(
                message=f"Error committing input audio buffer: buffer too small. Expected at least {MIN_AUDIO_BUFFER_DURATION_MS}ms of audio, but buffer only has {input_audio_buffer.duration_ms}.00ms of audio."
            )
        )
    else:
        item_id = input_audio_buffer.id
        ctx.audio_buffers.rotate()
        await commit_and_transcribe(ctx, item_id)


@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, _event: InputAudioBufferClearEvent) -> None:
    ctx.audio_buffers.clear_current()
    # OpenAI's doesn't send an error if the buffer is already empty.
    ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent())


async def commit_and_transcribe(ctx: SessionContext, item_id: str) -> None:
    event = InputAudioBufferCommittedEvent(
        previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME
        item_id=item_id,
    )
    ctx.pubsub.publish_nowait(event)

    input_audio_buffer = ctx.audio_buffers.get(event.item_id)

    transcriber = InputAudioBufferTranscriber(
        pubsub=ctx.pubsub,
        transcription_client=ctx.transcription_client,
        input_audio_buffer=input_audio_buffer,
        session=ctx.session,
        conversation=ctx.conversation,
    )
    transcriber.start()
    assert transcriber.task is not None
    try:
        await transcriber.task
    except openai.APIStatusError as e:
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(message=e.message)
            if e.status_code < 500
            else create_server_error(
                message=e.message,
            )
        )
        return

    if ctx.session.turn_detection is None or not ctx.session.turn_detection.create_response:
        return

    await ctx.response_manager.create_and_run(
        model=ctx.session.model,
        configuration=Response(
            conversation="auto", input=list(ctx.conversation.items.values()), **ctx.session.model_dump()
        ),
        conversation=ctx.conversation,
    )
