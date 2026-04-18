import asyncio
import logging
from typing import NamedTuple, cast

from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemRetrieveEvent,
    RealtimeError,
    RealtimeResponseCreateParams,
    ResponseCreateEvent,
    realtime_conversation_item_user_message,
)
from openai.types.realtime.realtime_client_event import RealtimeClientEvent
from openai.types.realtime.realtime_server_event import RealtimeServerEvent
import pytest

from speaches.logger import setup_logger
from tests.realtime.conftest import EndpointConfig
from tests.realtime.utils import (
    RecordingRealtimeConnection,
    consume_messages,
    extract_text_from_response_done_event,
    realtime_session_factory,
)

RealtimeEvent = RealtimeServerEvent | RealtimeClientEvent


setup_logger("INFO")
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_realtime_conversation_item_creation(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        await realtime_connection.send(
            ConversationItemCreateEvent(
                type="conversation.item.create",
                previous_item_id="",
                item=realtime_conversation_item_user_message.RealtimeConversationItemUserMessage(
                    role="user",
                    content=[realtime_conversation_item_user_message.Content(type="input_text", text="Hello, world!")],
                    type="message",
                ),
            )
        )
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert event_types[:3] == [
        "conversation.item.create",
        "conversation.item.added",
        "conversation.item.done",
    ], f"Unexpected event sequence: {event_types[:3]}"


@pytest.mark.asyncio
async def test_realtime_conversation_existing_item_deletion(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        await realtime_connection.send(ConversationItemDeleteEvent(type="conversation.item.delete", item_id="yo"))
        await message_consumer_task
        messages = recording_connection.messages

    events = [msg.message for msg in messages if not msg.message.type.startswith("session.")]
    assert [event.type for event in events] == ["conversation.item.delete", "error"]

    error_event = next(event for event in events if event.type == "error")
    assert error_event.error == RealtimeError(
        message="Error deleting item: the item with id 'yo' does not exist.",
        type="invalid_request_error",
        code="item_delete_invalid_item_id",
        event_id=None,
        param=None,
    )


def _create_user_message_event(
    text: str = "Hello, world!",
    previous_item_id: str = "",
    item_id: str | None = None,
) -> ConversationItemCreateEvent:
    return ConversationItemCreateEvent(
        type="conversation.item.create",
        previous_item_id=previous_item_id,
        item=realtime_conversation_item_user_message.RealtimeConversationItemUserMessage(
            id=item_id,
            role="user",
            content=[realtime_conversation_item_user_message.Content(type="input_text", text=text)],
            type="message",
        ),
    )


async def _recv_until_event_type(
    ws: RecordingRealtimeConnection, event_type: str, wait_timeout: float = 5.0
) -> RealtimeServerEvent:
    seen_events: list[RealtimeServerEvent] = []
    try:
        async with asyncio.timeout(wait_timeout):
            while True:
                event = await ws.recv_event()
                seen_events.append(event)
                if event.type == event_type:
                    return event
    except Exception as e:
        error_events = [event for event in seen_events if event.type == "error"]
        error_events_message = (
            f"Received {len(error_events)} error events. Errors: {error_events}"
            if error_events
            else "No error events received."
        )
        if isinstance(e, TimeoutError):
            raise TimeoutError(
                f"Timed out waiting for item creation to complete after {wait_timeout}s. Events received: {[event.type for event in seen_events]}. {error_events_message}"
            ) from None
        raise Exception(  # noqa: TRY002
            f"Error while waiting for event type '{event_type}'. Events received: {[event.type for event in seen_events]}. {error_events_message}"
        ) from e


class CreatedItem(NamedTuple):
    id: str
    previous_item_id: str | None


async def _create_conversation_item(
    realtime_connection: AsyncRealtimeConnection,
    recording_connection: RecordingRealtimeConnection,
    text: str = "Hello, world!",
    previous_item_id: str = "",
    item_id: str | None = None,
    wait_timeout: float = 2.5,
) -> CreatedItem:
    await realtime_connection.send(_create_user_message_event(text, previous_item_id, item_id))
    seen_events: list[RealtimeEvent] = []
    created_item_id: str | None = None
    try:
        async with asyncio.timeout(wait_timeout):
            while True:
                event = await recording_connection.recv_event()
                seen_events.append(event)
                if event.type == "conversation.item.added" and created_item_id is None:
                    created_item_id = event.item.id
                elif (
                    event.type == "conversation.item.done"
                    and created_item_id is not None
                    and event.item.id == created_item_id
                ):
                    return CreatedItem(id=created_item_id, previous_item_id=event.previous_item_id)
    except TimeoutError:
        error_events = [event for event in seen_events if event.type == "error"]
        error_events_message = (
            f"Received {len(error_events)} error events. Errors: {error_events}"
            if error_events
            else "No error events received."
        )
        raise TimeoutError(
            f"Timed out waiting for item creation to complete after {wait_timeout}s. "
            f"item_id={created_item_id}, events received: {[event.type for event in seen_events]}. {error_events_message}"
        ) from None


@pytest.mark.asyncio
async def test_realtime_conversation_item_deletion(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        item = await _create_conversation_item(realtime_connection, recording_connection)
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        await realtime_connection.send(ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item.id))
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert "conversation.item.deleted" in event_types


@pytest.mark.asyncio
async def test_realtime_conversation_item_double_deletion(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        item = await _create_conversation_item(realtime_connection, recording_connection)
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        delete_event = ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item.id)
        await realtime_connection.send(delete_event)
        await asyncio.sleep(1)  # Small delay to ensure the first delete is processed before sending the second
        await realtime_connection.send(delete_event)
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert event_types[-4:] == [
        "conversation.item.delete",
        "conversation.item.deleted",
        "conversation.item.delete",
        "error",
    ]
    # TODO: assert on error content


@pytest.mark.asyncio
async def test_realtime_conversation_item_retrieve(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        item = await _create_conversation_item(realtime_connection, recording_connection)
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        await realtime_connection.send(
            ConversationItemRetrieveEvent(type="conversation.item.retrieve", item_id=item.id)
        )
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert event_types[-2:] == [
        "conversation.item.retrieve",
        "conversation.item.retrieved",
    ]


@pytest.mark.asyncio
async def test_realtime_conversation_item_retrieve_non_existent(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        await realtime_connection.send(
            ConversationItemRetrieveEvent(type="conversation.item.retrieve", item_id="non_existent_item")
        )
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert event_types[-2:] == [
        "conversation.item.retrieve",
        "error",
    ]
    events = [msg.message for msg in messages if not msg.message.type.startswith("session.")]

    error_event = next(event for event in events if event.type == "error")
    assert error_event.error == RealtimeError(
        message="Error retrieving item: the item with id 'non_existent_item' does not exist.",
        type="invalid_request_error",
        code="item_retrieve_invalid_item_id",
        event_id=None,
        param=None,
    )


@pytest.mark.asyncio
async def test_realtime_conversation_item_delete_middle_and_check_order(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        item_a = await _create_conversation_item(realtime_connection, recording_connection, text="A")
        item_b = await _create_conversation_item(
            realtime_connection, recording_connection, text="B", previous_item_id=item_a.id
        )
        item_c = await _create_conversation_item(
            realtime_connection, recording_connection, text="C", previous_item_id=item_b.id
        )
        assert item_b.previous_item_id == item_a.id
        assert item_c.previous_item_id == item_b.id

        await realtime_connection.send(ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_b.id))
        await _recv_until_event_type(recording_connection, "conversation.item.deleted")

        item_d = await _create_conversation_item(
            realtime_connection, recording_connection, text="D", previous_item_id=item_a.id
        )

    assert item_d.previous_item_id == item_a.id


@pytest.mark.asyncio
async def test_realtime_conversation_item_delete_then_retrieve(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        item = await _create_conversation_item(realtime_connection, recording_connection)

        await realtime_connection.send(ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item.id))
        await _recv_until_event_type(recording_connection, "conversation.item.deleted")

        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )
        await realtime_connection.send(
            ConversationItemRetrieveEvent(type="conversation.item.retrieve", item_id=item.id)
        )
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert event_types[-2:] == [
        "conversation.item.retrieve",
        "error",
    ]


@pytest.mark.asyncio
async def test_realtime_conversation_item_create_with_previous_item_id(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        item_a = await _create_conversation_item(realtime_connection, recording_connection, text="alpha")
        await _create_conversation_item(
            realtime_connection, recording_connection, text="bravo", previous_item_id=item_a.id
        )
        # Insert charlie after alpha (between alpha and bravo). Order becomes: alpha -> charlie -> bravo
        await _create_conversation_item(
            realtime_connection, recording_connection, text="charlie", previous_item_id=item_a.id
        )

        instructions = (
            "Reply with all the user messages in order, separated by commas. No white space in-between. No other text."
        )
        response = await _trigger_response_and_get_text(
            realtime_connection, recording_connection, instructions=instructions
        )

    assert response.strip() == "alpha,charlie,bravo"


@pytest.mark.asyncio
async def test_realtime_conversation_item_create_with_nonexistent_previous_item_id(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        message_consumer_task = asyncio.create_task(
            consume_messages(
                realtime_connection,
                recording_connection,
                idle_timeout=1,
            )
        )

        await realtime_connection.send(_create_user_message_event(previous_item_id="nonexistent"))
        await message_consumer_task
        messages = recording_connection.messages

    event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
    assert "error" in event_types


@pytest.mark.asyncio
async def test_realtime_conversation_item_create_with_duplicate_id(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        original = await _create_conversation_item(realtime_connection, recording_connection, text="original")

        if endpoint.name == "speaches":
            # Speaches intentionally deviates from OpenAI: it returns an error for duplicate IDs
            message_consumer_task = asyncio.create_task(
                consume_messages(realtime_connection, recording_connection, idle_timeout=1)
            )
            await realtime_connection.send(_create_user_message_event(text="duplicate", item_id=original.id))
            await message_consumer_task
            messages = recording_connection.messages
            event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
            assert "error" in event_types
        else:
            duplicate = await _create_conversation_item(
                realtime_connection, recording_connection, text="duplicate", item_id=original.id
            )
            # OpenAI keeps the same ID but appends rather than replacing (see
            # test_realtime_conversation_item_duplicate_id_replaces_content for proof)
            assert duplicate.id == original.id
            # Quirk: previous_item_id points to itself (the original item's ID)
            assert duplicate.previous_item_id == duplicate.id


async def _trigger_response_and_get_text(
    realtime_connection: AsyncRealtimeConnection,
    recording_connection: RecordingRealtimeConnection,
    instructions: str = "",
    wait_timeout: float = 10.0,
) -> str:
    await realtime_connection.send(
        ResponseCreateEvent(
            type="response.create",
            response=RealtimeResponseCreateParams(instructions=instructions, output_modalities=["text"]),
            # response=RealtimeResponseCreateParams(instructions=instructions),
        )
    )
    event = await _recv_until_event_type(recording_connection, "response.done", wait_timeout=wait_timeout)
    assert event.type == "response.done"
    # HACK: needed because at the time of writing `speaches` doesn't properly handle `ResponseCreateEvent.response.output_modalities` and thus can return either text or audio content regardless of the requested output modalities. Once `speaches` is fixed this can be simplified to just extract the text content.
    return extract_text_from_response_done_event(event)


@pytest.mark.asyncio
async def test_realtime_conversation_item_duplicate_id_replaces_content(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")

        words = ["alpha", "bravo", "charlie", "delta", "echo"]
        items: list[CreatedItem] = []
        for word in words:
            item = await _create_conversation_item(realtime_connection, recording_connection, text=word)
            items.append(item)

        if endpoint.name == "speaches":
            # Speaches returns an error for duplicate IDs instead of silently ignoring them
            message_consumer_task = asyncio.create_task(
                consume_messages(realtime_connection, recording_connection, idle_timeout=1)
            )
            await realtime_connection.send(_create_user_message_event(text="REPLACED", item_id=items[2].id))
            await message_consumer_task
            messages = recording_connection.messages
            event_types = [msg.message.type for msg in messages if not msg.message.type.startswith("session.")]
            assert "error" in event_types
        else:
            # Create a duplicate of item "charlie" with text "REPLACED" using the same ID
            # This appends rather than replaces (see test_realtime_conversation_item_create_with_duplicate_id)
            await _create_conversation_item(
                realtime_connection, recording_connection, text="REPLACED", item_id=items[2].id
            )

            instructions = "Reply with all the user messages in order, separated by commas. No white space in-between. No other text."
            response = await _trigger_response_and_get_text(
                realtime_connection, recording_connection, instructions=instructions
            )

            # On `gpt-4o-realtime-preview`
            # Observed behavior: the duplicate ID item does not appear in the conversation.
            # The model only sees the original 5 items — the duplicate is silently ignored.
            # assert response.strip() == "alpha,bravo,charlie,delta,echo"
            # On `gpt-realtime-1.5` the item does appear at the end
            assert response.strip() == "alpha,bravo,charlie,delta,echo,REPLACED"
            print(recording_connection.messages)


@pytest.mark.asyncio
async def test_realtime_conversation_item_create_with_previous_item_id_root(endpoint: EndpointConfig) -> None:
    async with realtime_session_factory(
        endpoint.realtime_url("conversation"), endpoint.api_key.get_secret_value()
    ) as realtime_connection:
        recording_connection = cast("RecordingRealtimeConnection", realtime_connection._connection)  # noqa: SLF001
        await _recv_until_event_type(recording_connection, "session.created")
        print(recording_connection.messages)

        num_sequence = [-20, 10, 20, 30, 40]

        # Create items 10, 20, 30, 40
        for i in num_sequence[1:]:
            await _create_conversation_item(realtime_connection, recording_connection, text=str(i))

        item_1 = await _create_conversation_item(
            realtime_connection, recording_connection, text=str(num_sequence[0]), previous_item_id="root"
        )
        assert item_1.previous_item_id is None

        instructions = (
            "Reply with all the user messages in order, separated by commas. No white space in-between. No other text."
        )
        response = await _trigger_response_and_get_text(
            realtime_connection, recording_connection, instructions=instructions
        )

    assert response.strip() == ",".join(str(i) for i in num_sequence)
