import asyncio
import base64
from io import BytesIO
import logging
from typing import TYPE_CHECKING

import numpy as np
from openai.types.beta.realtime.error_event import Error

if TYPE_CHECKING:
    from numpy.typing import NDArray

from speaches.audio import Audio, audio_samples_from_file, resample_audio_data
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.shared.vad_types import VadOptions
from speaches.executors.silero_vad_v5 import get_speech_timestamps, to_ms_speech_timestamps
from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer import (
    MS_SAMPLE_RATE,
    SAMPLE_RATE,
    InputAudioBuffer,
    InputAudioBufferTranscriber,
)
from speaches.realtime.utils import task_done_callback
from speaches.types.realtime import (
    ConversationState,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferPartialTranscriptionEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
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


def vad_detection_flow(
    input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection, ctx: SessionContext
) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    if input_audio_buffer.vad_state.audio_end_ms is not None:
        # Speech stop already fired for this buffer; ignore further VAD on it.
        # This prevents duplicate speech_stopped events when the async handler
        # hasn't yet created a new buffer between audio appends.
        return None

    audio_window = input_audio_buffer.vad_data

    speech_timestamps = to_ms_speech_timestamps(
        get_speech_timestamps(
            audio_window,
            model_manager=ctx.vad_model_manager,
            model_id=ctx.vad_model_id,
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

    elif speech_timestamp is None:
        input_audio_buffer.vad_state.audio_end_ms = input_audio_buffer.duration_ms
        return InputAudioBufferSpeechStoppedEvent(
            item_id=input_audio_buffer.id,
            audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
        )

    else:
        window_ms = len(audio_window) // MS_SAMPLE_RATE
        trailing_silence_ms = window_ms - speech_timestamp.end
        if trailing_silence_ms >= turn_detection.silence_duration_ms:
            input_audio_buffer.vad_state.audio_end_ms = input_audio_buffer.duration_ms - trailing_silence_ms
            return InputAudioBufferSpeechStoppedEvent(
                item_id=input_audio_buffer.id,
                audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
            )

    return None


# Client Events


@event_router.register("input_audio_buffer.append")
def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    audio_chunk = audio_samples_from_file(BytesIO(base64.b64decode(event.audio)), 24000)
    # convert the audio data from 24kHz (sample rate defined in the API spec) to 16kHz (sample rate used by the VAD and for transcription)
    audio_chunk = resample_audio_data(audio_chunk, 24000, 16000)
    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
    input_audio_buffer.append(audio_chunk)
    if ctx.session.turn_detection is not None:
        vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection, ctx)
        if vad_event is not None:
            ctx.pubsub.publish_nowait(vad_event)


@event_router.register("input_audio_buffer.commit")
def handle_input_audio_buffer_commit(ctx: SessionContext, _event: InputAudioBufferCommitEvent) -> None:
    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(
                message=f"Error committing input audio buffer: buffer too small. Expected at least {MIN_AUDIO_BUFFER_DURATION_MS}ms of audio, but buffer only has {input_audio_buffer.duration_ms}.00ms of audio."
            )
        )
    else:
        input_audio_buffer.consolidate()
        ctx.pubsub.publish_nowait(
            InputAudioBufferCommittedEvent(
                previous_item_id=next(reversed(list(ctx.conversation.items)), None),  # FIXME
                item_id=input_audio_buffer_id,
            )
        )
        input_audio_buffer = InputAudioBuffer(ctx.pubsub)
        ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, _event: InputAudioBufferClearEvent) -> None:
    ctx.input_audio_buffers.popitem()
    # OpenAI's doesn't send an error if the buffer is already empty.
    ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent())
    input_audio_buffer = InputAudioBuffer(ctx.pubsub)
    ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


# Server Events


async def _partial_transcription_loop(ctx: SessionContext, input_audio_buffer: InputAudioBuffer, item_id: str) -> None:
    interval = 0.5
    min_samples = 8000  # 500ms of audio at 16kHz
    min_new_samples = 4000  # 250ms minimum new audio before re-transcribing
    last_snapshot_size = 0
    cached_snapshot: NDArray[np.float32] | None = None
    while True:
        await asyncio.sleep(interval)
        if input_audio_buffer.size < min_samples:
            continue
        if input_audio_buffer.size - last_snapshot_size < min_new_samples:
            continue
        if ctx.partial_transcription_lock.locked():
            continue
        async with ctx.partial_transcription_lock:
            current_size = input_audio_buffer.size
            if cached_snapshot is None:
                audio_snapshot = input_audio_buffer.data.copy()
            else:
                # Only copy the new samples and concatenate with cached prefix
                new_samples = input_audio_buffer.data[last_snapshot_size:current_size].copy()
                audio_snapshot = np.concatenate([cached_snapshot, new_samples])
            cached_snapshot = audio_snapshot
            last_snapshot_size = current_size
            audio = Audio(audio_snapshot, sample_rate=SAMPLE_RATE)
            request = TranscriptionRequest(
                audio=audio,
                model=ctx.session.input_audio_transcription.model,
                language=ctx.session.input_audio_transcription.language,
                response_format="text",
                speech_segments=[],
                vad_options=VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30),
                timestamp_granularities=["segment"],
            )
            try:
                result = await asyncio.to_thread(
                    ctx.stt_model_manager.handle_non_streaming_transcription_request, request
                )
                transcript = result[0] if isinstance(result, tuple) else result.text
                if transcript.strip():
                    ctx.pubsub.publish_nowait(
                        InputAudioBufferPartialTranscriptionEvent(item_id=item_id, transcript=transcript)
                    )
            except Exception:
                logger.exception("Partial transcription failed")


@event_router.register("input_audio_buffer.speech_started")
def handle_speech_started_interruption(ctx: SessionContext, event: InputAudioBufferSpeechStartedEvent) -> None:
    if ctx.barge_in_task is not None and not ctx.barge_in_task.done():
        ctx.barge_in_task.cancel()
        ctx.barge_in_task = None

    if ctx.state == ConversationState.GENERATING and ctx.response is not None:
        delay_ms = ctx.session.turn_detection.barge_in_delay_ms if ctx.session.turn_detection else 0
        response_to_cancel = ctx.response
        if delay_ms > 0:

            async def _delayed_barge_in() -> None:
                await asyncio.sleep(delay_ms / 1000)
                if ctx.response is response_to_cancel:
                    logger.info(f"Barge-in confirmed after {delay_ms}ms delay: cancelling active response")
                    response_to_cancel.stop()

            ctx.barge_in_task = asyncio.create_task(_delayed_barge_in(), name="barge_in_delay")
            ctx.barge_in_task.add_done_callback(task_done_callback)
        else:
            logger.info("Barge-in detected: cancelling active response")
            response_to_cancel.stop()

    ctx.state = ConversationState.LISTENING

    if ctx.partial_transcription_task is not None and not ctx.partial_transcription_task.done():
        ctx.partial_transcription_task.cancel()

    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
    ctx.partial_transcription_task = asyncio.create_task(
        _partial_transcription_loop(ctx, input_audio_buffer, event.item_id),
        name="partial_transcription",
    )
    ctx.partial_transcription_task.add_done_callback(task_done_callback)


@event_router.register("input_audio_buffer.speech_stopped")
def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
    if ctx.barge_in_task is not None and not ctx.barge_in_task.done():
        logger.info("Speech stopped before barge-in delay expired, cancelling barge-in")
        ctx.barge_in_task.cancel()
        ctx.barge_in_task = None

    ctx.state = ConversationState.PROCESSING

    if ctx.partial_transcription_task is not None and not ctx.partial_transcription_task.done():
        ctx.partial_transcription_task.cancel()
    ctx.partial_transcription_task = None

    committed_buffer = ctx.input_audio_buffers.get(event.item_id)
    if committed_buffer is not None:
        committed_buffer.consolidate()
    input_audio_buffer = InputAudioBuffer(ctx.pubsub)
    ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer
    ctx.pubsub.publish_nowait(
        InputAudioBufferCommittedEvent(
            previous_item_id=next(reversed(list(ctx.conversation.items)), None),  # FIXME
            item_id=event.item_id,
        )
    )


@event_router.register("input_audio_buffer.committed")
async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
    input_audio_buffer = ctx.input_audio_buffers[event.item_id]

    transcriber = InputAudioBufferTranscriber(
        pubsub=ctx.pubsub,
        stt_model_manager=ctx.stt_model_manager,
        input_audio_buffer=input_audio_buffer,
        session=ctx.session,
        conversation=ctx.conversation,
    )
    transcriber.start()
    assert transcriber.task is not None
    try:
        await transcriber.task
    except Exception:
        logger.exception("Transcription failed")
        ctx.pubsub.publish_nowait(create_server_error(message="Transcription failed"))
    finally:
        ctx.input_audio_buffers.pop(event.item_id, None)
        if ctx.state == ConversationState.PROCESSING:
            ctx.state = ConversationState.IDLE
