import asyncio
from asyncio import Queue
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import pytest

from speaches.realtime.conversation_event_router import Conversation
from speaches.realtime.pubsub import EventPubSub
from speaches.realtime.response_event_router import ResponseHandler, create_and_run_response
from speaches.types.realtime import (
    ConversationItemContentInputText,
    ConversationItemContentText,
    ConversationItemMessage,
    Event,
    Response,
)


def _drain_event_types(q: Queue[Event]) -> list[str]:
    types: list[str] = []
    while not q.empty():
        types.append(q.get_nowait().type)
    return types


def make_response_handler(*, cancelled: bool = False) -> ResponseHandler:
    pubsub = EventPubSub()
    conversation = Conversation(pubsub)
    configuration = Response(
        conversation="auto",
        input=[],
        instructions="test",
        max_response_output_tokens="inf",
        modalities=["text", "audio"],
        output_audio_format="pcm16",
        temperature=0.8,
        tool_choice="auto",
        tools=[],
        voice="af_heart",
    )
    handler = ResponseHandler(
        completion_client=MagicMock(),
        tts_model_manager=MagicMock(),
        model="test-model",
        speech_model="test-speech-model",
        configuration=configuration,
        conversation=conversation,
        pubsub=pubsub,
    )
    if cancelled:
        handler._cancelled = True  # noqa: SLF001
    return handler


def test_add_output_item_completed() -> None:
    handler = make_response_handler()
    q = handler.pubsub.subscribe()

    item = ConversationItemMessage(role="assistant", status="incomplete", content=[])

    with handler.add_output_item(item):
        pass

    assert item.status == "completed"

    event_types = _drain_event_types(q)
    assert "response.output_item.added" in event_types
    assert "response.output_item.done" in event_types


def test_add_output_item_cancelled() -> None:
    handler = make_response_handler(cancelled=True)
    q = handler.pubsub.subscribe()

    item = ConversationItemMessage(role="assistant", status="incomplete", content=[])

    with handler.add_output_item(item):
        pass

    assert item.status == "incomplete"

    event_types = _drain_event_types(q)
    assert "response.output_item.done" in event_types


def test_add_output_item_exception_cleanup() -> None:
    handler = make_response_handler()
    q = handler.pubsub.subscribe()

    item = ConversationItemMessage(role="assistant", status="incomplete", content=[])

    with pytest.raises(ValueError, match="something broke"), handler.add_output_item(item):
        raise ValueError("something broke")

    # Even with an exception, finally block should fire
    event_types = _drain_event_types(q)
    assert "response.output_item.done" in event_types


@pytest.mark.asyncio
async def test_response_handler_stop_sets_cancelled() -> None:
    handler = make_response_handler()
    handler.task = asyncio.get_running_loop().create_future()

    handler.stop()

    assert handler._cancelled is True  # noqa: SLF001
    assert handler.task.cancelled()


def test_response_handler_audio_duration_tracking() -> None:
    handler = make_response_handler()
    assert handler.audio_duration_ms == 0

    # Simulate audio duration accumulation
    handler.audio_duration_ms += 500
    handler.audio_duration_ms += 300
    assert handler.audio_duration_ms == 800


# --- _check_dismissed tests ---


def test_check_dismissed_with_matching_token() -> None:
    handler = make_response_handler()
    handler.no_response_token = "*"  # noqa: S105
    item = ConversationItemMessage(
        role="assistant",
        status="completed",
        content=[ConversationItemContentText(text="*")],
    )
    handler.conversation.create_item(item)

    handler._check_dismissed()  # noqa: SLF001
    assert handler.dismissed is True


def test_check_dismissed_no_match() -> None:
    handler = make_response_handler()
    handler.no_response_token = "*"  # noqa: S105
    item = ConversationItemMessage(
        role="assistant",
        status="completed",
        content=[ConversationItemContentText(text="Hello world")],
    )
    handler.conversation.create_item(item)

    handler._check_dismissed()  # noqa: SLF001
    assert handler.dismissed is False


def test_check_dismissed_no_token_configured() -> None:
    handler = make_response_handler()
    handler.no_response_token = None
    item = ConversationItemMessage(
        role="assistant",
        status="completed",
        content=[ConversationItemContentText(text="*")],
    )
    handler.conversation.create_item(item)

    handler._check_dismissed()  # noqa: SLF001
    assert handler.dismissed is False


def test_check_dismissed_non_assistant_message() -> None:
    handler = make_response_handler()
    handler.no_response_token = "*"  # noqa: S105
    item = ConversationItemMessage(
        role="user",
        status="completed",
        content=[ConversationItemContentInputText(text="*")],
    )
    handler.conversation.create_item(item)

    handler._check_dismissed()  # noqa: SLF001
    assert handler.dismissed is False


def test_check_dismissed_empty_conversation() -> None:
    handler = make_response_handler()
    handler.no_response_token = "*"  # noqa: S105

    handler._check_dismissed()  # noqa: SLF001
    assert handler.dismissed is False


# --- Helper to build mock async streams ---


def _make_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content),
                finish_reason=finish_reason,
            )
        ],
    )


async def _async_stream(chunks: list[ChatCompletionChunk]) -> AsyncGenerator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


# --- generate_response tests ---


@pytest.mark.asyncio
async def test_generate_response_text_mode() -> None:
    handler = make_response_handler()
    q = handler.pubsub.subscribe()
    handler.configuration = Response(
        conversation="auto",
        input=[],
        instructions="test",
        max_response_output_tokens="inf",
        modalities=["text"],
        output_audio_format="pcm16",
        temperature=0.8,
        tool_choice="auto",
        tools=[],
        voice="af_heart",
    )

    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(content=" world"),
        _make_chunk(finish_reason="stop"),
    ]
    handler.completion_client.create = AsyncMock(return_value=_async_stream(chunks))

    await handler.generate_response()

    assert len(handler.conversation.items) == 1
    item = next(iter(handler.conversation.items.values()))
    assert isinstance(item, ConversationItemMessage)
    assert item.role == "assistant"
    assert item.content[0].type == "text"
    assert item.content[0].text == "Hello world"  # pyright: ignore[reportAttributeAccessIssue]

    event_types = _drain_event_types(q)
    assert "response.output_item.added" in event_types
    assert "response.text.delta" in event_types
    assert "response.text.done" in event_types
    assert "response.output_item.done" in event_types


@pytest.mark.asyncio
async def test_generate_response_dismissed() -> None:
    handler = make_response_handler()
    handler.no_response_token = "*"  # noqa: S105
    handler.configuration = Response(
        conversation="auto",
        input=[],
        instructions="test",
        max_response_output_tokens="inf",
        modalities=["text"],
        output_audio_format="pcm16",
        temperature=0.8,
        tool_choice="auto",
        tools=[],
        voice="af_heart",
    )

    chunks = [
        _make_chunk(content="*"),
        _make_chunk(finish_reason="stop"),
    ]
    handler.completion_client.create = AsyncMock(return_value=_async_stream(chunks))

    await handler.generate_response()

    assert handler.dismissed is True


@pytest.mark.asyncio
async def test_create_and_run_response_sets_completed() -> None:
    pubsub = EventPubSub()
    conversation = Conversation(pubsub)
    configuration = Response(
        conversation="auto",
        input=[],
        instructions="test",
        max_response_output_tokens="inf",
        modalities=["text"],
        output_audio_format="pcm16",
        temperature=0.8,
        tool_choice="auto",
        tools=[],
        voice="af_heart",
    )

    chunks = [
        _make_chunk(content="Hi"),
        _make_chunk(finish_reason="stop"),
    ]

    q = pubsub.subscribe()

    ctx = MagicMock()
    ctx.completion_client = AsyncMock()
    ctx.completion_client.create = AsyncMock(return_value=_async_stream(chunks))
    ctx.tts_model_manager = MagicMock()
    ctx.session.model = "test-model"
    ctx.session.speech_model = "test-speech"
    ctx.session.no_response_token = None
    ctx.pubsub = pubsub
    ctx.conversation = conversation
    ctx.response = None
    ctx.state = None
    ctx.response_lock = asyncio.Lock()

    await create_and_run_response(ctx, configuration)

    events: list[Event] = []
    while not q.empty():
        events.append(q.get_nowait())
    response_done_events = [e for e in events if e.type == "response.done"]
    assert len(response_done_events) == 1
    assert response_done_events[0].response.status == "completed"  # pyright: ignore[reportAttributeAccessIssue]
