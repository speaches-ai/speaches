import asyncio
from collections import OrderedDict
import contextlib
from unittest.mock import MagicMock

import pytest

from speaches.realtime.conversation_event_router import Conversation
from speaches.realtime.input_audio_buffer import InputAudioBuffer
from speaches.realtime.input_audio_buffer_event_router import (
    handle_input_audio_buffer_speech_stopped,
    handle_speech_started_interruption,
)
from speaches.realtime.pubsub import EventPubSub
from speaches.types.realtime import (
    ConversationItemContentAudio,
    ConversationItemContentInputText,
    ConversationItemMessage,
    ConversationState,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
)


def _drain_events(q: asyncio.Queue) -> list:
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


class FakeSessionContext:
    def __init__(self) -> None:
        self.pubsub = EventPubSub()
        self.conversation = Conversation(self.pubsub)
        self.state = ConversationState.IDLE
        self.response = None
        self.session = MagicMock()
        self.session.turn_detection = None
        input_audio_buffer = InputAudioBuffer(self.pubsub)
        self.input_audio_buffers = OrderedDict({input_audio_buffer.id: input_audio_buffer})
        self.tts_model_manager = MagicMock()
        self.stt_model_manager = MagicMock()
        self.partial_transcription_task = None
        self.barge_in_task = None


def _make_speech_started_event(ctx: FakeSessionContext) -> InputAudioBufferSpeechStartedEvent:
    item_id = next(reversed(ctx.input_audio_buffers))
    return InputAudioBufferSpeechStartedEvent(item_id=item_id, audio_start_ms=0)


def _make_speech_stopped_event(ctx: FakeSessionContext, audio_end_ms: int = 1000) -> InputAudioBufferSpeechStoppedEvent:
    item_id = next(reversed(ctx.input_audio_buffers))
    return InputAudioBufferSpeechStoppedEvent(item_id=item_id, audio_end_ms=audio_end_ms)


# ---- State Machine: speech_started transitions ----


@pytest.mark.asyncio
async def test_speech_started_idle_to_listening() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.IDLE

    event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event)
    await asyncio.sleep(0)

    assert ctx.state == ConversationState.LISTENING
    assert ctx.response is None
    assert ctx.partial_transcription_task is not None
    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


@pytest.mark.asyncio
async def test_speech_started_generating_barge_in_immediate() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.GENERATING

    mock_response = MagicMock()
    mock_response.stop = MagicMock()
    ctx.response = mock_response

    # turn_detection is None -> delay_ms = 0 -> immediate stop
    event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event)
    await asyncio.sleep(0)

    mock_response.stop.assert_called_once()
    assert ctx.state == ConversationState.LISTENING
    assert ctx.barge_in_task is None
    assert ctx.partial_transcription_task is not None
    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


@pytest.mark.asyncio
async def test_speech_started_generating_barge_in_delayed() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.GENERATING

    mock_response = MagicMock()
    mock_response.stop = MagicMock()
    ctx.response = mock_response

    # Set turn_detection with barge_in_delay_ms
    td = MagicMock()
    td.barge_in_delay_ms = 100
    ctx.session.turn_detection = td

    event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event)
    await asyncio.sleep(0)

    # stop() should NOT be called yet (delayed)
    mock_response.stop.assert_not_called()
    assert ctx.barge_in_task is not None
    assert ctx.state == ConversationState.LISTENING

    # Wait for delay to expire
    await asyncio.sleep(0.15)

    mock_response.stop.assert_called_once()

    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


@pytest.mark.asyncio
async def test_speech_started_processing_to_listening() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.PROCESSING

    event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event)
    await asyncio.sleep(0)

    assert ctx.state == ConversationState.LISTENING
    assert ctx.response is None
    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


@pytest.mark.asyncio
async def test_speech_started_listening_stays_listening() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.LISTENING

    # Simulate an existing partial transcription task
    existing_task = asyncio.create_task(asyncio.sleep(10))
    ctx.partial_transcription_task = existing_task

    event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event)
    await asyncio.sleep(0)

    assert ctx.state == ConversationState.LISTENING
    # Old task should be cancelled and a new one created
    assert existing_task.cancelled()
    assert ctx.partial_transcription_task is not None
    assert ctx.partial_transcription_task is not existing_task
    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


# ---- State Machine: speech_stopped transitions ----


@pytest.mark.asyncio
async def test_speech_stopped_listening_to_processing() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.LISTENING

    q = ctx.pubsub.subscribe()
    event = _make_speech_stopped_event(ctx)
    handle_input_audio_buffer_speech_stopped(ctx, event)

    assert ctx.state == ConversationState.PROCESSING
    assert ctx.partial_transcription_task is None

    # Should have published a committed event
    events = _drain_events(q)
    committed_events = [e for e in events if e.type == "input_audio_buffer.committed"]
    assert len(committed_events) == 1
    assert committed_events[0].item_id == event.item_id


@pytest.mark.asyncio
async def test_speech_stopped_cancels_barge_in_task() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.LISTENING

    # Simulate pending barge-in
    barge_in_task = asyncio.create_task(asyncio.sleep(10))
    ctx.barge_in_task = barge_in_task

    event = _make_speech_stopped_event(ctx)
    handle_input_audio_buffer_speech_stopped(ctx, event)
    await asyncio.sleep(0)

    assert ctx.barge_in_task is None
    assert barge_in_task.cancelled()
    assert ctx.state == ConversationState.PROCESSING


@pytest.mark.asyncio
async def test_speech_stopped_cancels_partial_transcription() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.LISTENING

    partial_task = asyncio.create_task(asyncio.sleep(10))
    ctx.partial_transcription_task = partial_task

    event = _make_speech_stopped_event(ctx)
    handle_input_audio_buffer_speech_stopped(ctx, event)
    await asyncio.sleep(0)

    assert ctx.partial_transcription_task is None
    assert partial_task.cancelled()


@pytest.mark.asyncio
async def test_speech_stopped_creates_new_audio_buffer() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.LISTENING

    original_buffer_count = len(ctx.input_audio_buffers)
    event = _make_speech_stopped_event(ctx)
    handle_input_audio_buffer_speech_stopped(ctx, event)

    # A new buffer should be appended
    assert len(ctx.input_audio_buffers) == original_buffer_count + 1


# ---- VAD duplicate guard ----


def test_vad_does_not_produce_duplicate_speech_stopped() -> None:
    from speaches.realtime.input_audio_buffer_event_router import vad_detection_flow

    ctx = FakeSessionContext()
    td = MagicMock()
    td.threshold = 0.5
    td.silence_duration_ms = 300
    td.prefix_padding_ms = 300
    ctx.session.turn_detection = td

    buf_id = next(reversed(ctx.input_audio_buffers))
    buf = ctx.input_audio_buffers[buf_id]

    # Simulate: speech already started and stopped
    buf.vad_state.audio_start_ms = 100
    buf.vad_state.audio_end_ms = 900

    # VAD should NOT produce another speech_stopped event
    result = vad_detection_flow(buf, td, ctx)
    assert result is None


# ---- Delayed barge-in edge cases ----


@pytest.mark.asyncio
async def test_delayed_barge_in_cancelled_by_speech_stopped() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.GENERATING

    mock_response = MagicMock()
    mock_response.stop = MagicMock()
    ctx.response = mock_response

    td = MagicMock()
    td.barge_in_delay_ms = 500
    ctx.session.turn_detection = td

    # speech_started triggers delayed barge-in
    started_event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, started_event)
    assert ctx.barge_in_task is not None

    # speech_stopped before delay expires -> should cancel the barge-in
    stopped_event = _make_speech_stopped_event(ctx)
    handle_input_audio_buffer_speech_stopped(ctx, stopped_event)

    assert ctx.barge_in_task is None
    # stop() should never have been called
    mock_response.stop.assert_not_called()
    assert ctx.state == ConversationState.PROCESSING


@pytest.mark.asyncio
async def test_delayed_barge_in_skips_if_response_changed() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.GENERATING

    old_response = MagicMock()
    old_response.stop = MagicMock()
    ctx.response = old_response

    td = MagicMock()
    td.barge_in_delay_ms = 50
    ctx.session.turn_detection = td

    event = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event)
    assert ctx.barge_in_task is not None

    # Replace response before delay fires (simulates a new response starting)
    new_response = MagicMock()
    ctx.response = new_response

    await asyncio.sleep(0.1)

    # Old response should NOT be stopped (identity check fails)
    old_response.stop.assert_not_called()
    new_response.stop.assert_not_called()

    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


@pytest.mark.asyncio
async def test_second_speech_started_cancels_pending_barge_in() -> None:
    ctx = FakeSessionContext()
    ctx.state = ConversationState.GENERATING

    mock_response = MagicMock()
    mock_response.stop = MagicMock()
    ctx.response = mock_response

    td = MagicMock()
    td.barge_in_delay_ms = 500
    ctx.session.turn_detection = td

    # First speech_started
    event1 = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event1)
    first_barge_in = ctx.barge_in_task
    assert first_barge_in is not None

    # Second speech_started while first is still pending
    # (state is LISTENING now, so no new barge-in, but the old one should be cancelled)
    event2 = _make_speech_started_event(ctx)
    handle_speech_started_interruption(ctx, event2)
    await asyncio.sleep(0)

    assert first_barge_in.cancelled()
    # No new barge-in since state is now LISTENING (not GENERATING)
    assert ctx.barge_in_task is None

    ctx.partial_transcription_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ctx.partial_transcription_task


# ---- response.cancel ----


def test_response_cancel_no_active_response() -> None:
    from openai.types.beta.realtime import ResponseCancelEvent

    from speaches.realtime.response_event_router import handle_response_cancel_event

    ctx = FakeSessionContext()
    ctx.response = None

    q = ctx.pubsub.subscribe()
    event = ResponseCancelEvent(type="response.cancel", event_id="evt_1")
    handle_response_cancel_event(ctx, event)

    events = _drain_events(q)
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1


def test_response_cancel_with_active_response() -> None:
    from openai.types.beta.realtime import ResponseCancelEvent

    from speaches.realtime.response_event_router import handle_response_cancel_event

    ctx = FakeSessionContext()
    mock_response = MagicMock()
    mock_response.stop = MagicMock()
    ctx.response = mock_response

    event = ResponseCancelEvent(type="response.cancel", event_id="evt_2")
    handle_response_cancel_event(ctx, event)

    mock_response.stop.assert_called_once()


# ---- conversation.item.truncate ----


def test_truncate_nonexistent_item() -> None:
    from openai.types.beta.realtime import ConversationItemTruncateEvent

    from speaches.realtime.conversation_event_router import handle_conversation_item_truncate_event

    ctx = FakeSessionContext()

    q = ctx.pubsub.subscribe()
    event = ConversationItemTruncateEvent(
        type="conversation.item.truncate",
        event_id="evt_3",
        item_id="nonexistent",
        content_index=0,
        audio_end_ms=1000,
    )
    handle_conversation_item_truncate_event(ctx, event)

    events = _drain_events(q)
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "does not exist" in error_events[0].error.message


def test_truncate_assistant_audio_message() -> None:
    from openai.types.beta.realtime import ConversationItemTruncateEvent

    from speaches.realtime.conversation_event_router import handle_conversation_item_truncate_event

    ctx = FakeSessionContext()

    original_transcript = "Hello, I am a long response that should be truncated at some point"
    content = ConversationItemContentAudio(audio="base64data", transcript=original_transcript)
    item = ConversationItemMessage(
        id="item_assistant_1",
        role="assistant",
        content=[content],
        status="completed",
    )
    ctx.conversation.create_item(item)

    q = ctx.pubsub.subscribe()

    event = ConversationItemTruncateEvent(
        type="conversation.item.truncate",
        event_id="evt_4",
        item_id="item_assistant_1",
        content_index=0,
        audio_end_ms=500,
    )
    handle_conversation_item_truncate_event(ctx, event)

    events = _drain_events(q)
    truncated_events = [e for e in events if e.type == "conversation.item.truncated"]
    assert len(truncated_events) == 1
    assert truncated_events[0].item_id == "item_assistant_1"
    assert truncated_events[0].audio_end_ms == 500

    assert len(content.transcript) < len(original_transcript)
    assert content.transcript == original_transcript[: len(content.transcript)]


def test_truncate_non_assistant_message() -> None:
    from openai.types.beta.realtime import ConversationItemTruncateEvent

    from speaches.realtime.conversation_event_router import handle_conversation_item_truncate_event

    ctx = FakeSessionContext()

    item = ConversationItemMessage(
        id="item_user_1",
        role="user",
        content=[ConversationItemContentInputText(text="user text")],
        status="completed",
    )
    ctx.conversation.create_item(item)

    q = ctx.pubsub.subscribe()

    event = ConversationItemTruncateEvent(
        type="conversation.item.truncate",
        event_id="evt_5",
        item_id="item_user_1",
        content_index=0,
        audio_end_ms=500,
    )
    handle_conversation_item_truncate_event(ctx, event)

    events = _drain_events(q)
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "not an assistant message" in error_events[0].error.message


def test_truncate_content_index_out_of_range() -> None:
    from openai.types.beta.realtime import ConversationItemTruncateEvent

    from speaches.realtime.conversation_event_router import handle_conversation_item_truncate_event

    ctx = FakeSessionContext()

    content = ConversationItemContentAudio(audio="data", transcript="text")
    item = ConversationItemMessage(
        id="item_a1",
        role="assistant",
        content=[content],
        status="completed",
    )
    ctx.conversation.create_item(item)

    q = ctx.pubsub.subscribe()

    event = ConversationItemTruncateEvent(
        type="conversation.item.truncate",
        event_id="evt_6",
        item_id="item_a1",
        content_index=5,  # out of range
        audio_end_ms=500,
    )
    handle_conversation_item_truncate_event(ctx, event)

    events = _drain_events(q)
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "out of range" in error_events[0].error.message
