import asyncio
import base64
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
import json
import logging
import time
from typing import Any, Literal, NamedTuple, cast

from openai._models import construct_type_unchecked
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import RealtimeConversationItemAssistantMessage, ResponseDoneEvent
from openai.types.realtime.realtime_client_event import RealtimeClientEvent
from openai.types.realtime.realtime_server_event import RealtimeServerEvent
import websockets
from websockets.asyncio.client import ClientConnection, connect

RealtimeEvent = RealtimeServerEvent | RealtimeClientEvent


logger = logging.getLogger(__name__)


def extract_text_from_response_done_event(event: ResponseDoneEvent) -> str:
    assert event.response.output is not None, event
    output_item = event.response.output[0]
    assert isinstance(output_item, RealtimeConversationItemAssistantMessage), output_item

    assert len(output_item.content) > 0, output_item

    content_item = output_item.content[0]
    if content_item.type == "output_audio":
        assert content_item.transcript is not None and content_item.text is None, output_item
        return content_item.transcript
    elif content_item.type == "output_text":
        assert content_item.text is not None and content_item.transcript is None, output_item
        return content_item.text
    else:
        raise ValueError(f"Unexpected content type: {content_item.type} in event: {event}")


def truncate_audio_messages(
    message: RealtimeServerEvent | RealtimeClientEvent,
) -> None:
    if message.type == "response.output_audio.delta":
        size = len(base64.b64decode(message.delta))
        message.delta = f"base64 encoded audio of size {size} bytes"
    elif message.type == "input_audio_buffer.append":
        size = len(base64.b64decode(message.audio))
        message.audio = f"base64 encoded audio of size {size} bytes"


async def consume_messages(
    conn: AsyncRealtimeConnection,
    ws_connection: "RecordingConnection[Any, Any]",
    idle_timeout: float | None = None,
    event_idle_timeouts: dict[str, float] | None = None,
) -> None:
    """Consume messages from a realtime connection until idle timeout or disconnection.

    Idle timeout accounts for both sent and received messages via `ws_connection.last_activity`.
    After receiving an event, the timeout for the next wait is determined by looking up the event's
    type in `event_idle_timeouts`. If the event type is not found (or `event_idle_timeouts` is None),
    `idle_timeout` is used as the fallback.
    """
    poll_interval = 0.5
    try:
        event_iter = conn.__aiter__()
        current_timeout = idle_timeout
        next_event_task: asyncio.Task[Any] | None = None
        while True:
            if next_event_task is None:
                next_event_task = asyncio.ensure_future(event_iter.__anext__())
            if current_timeout is not None:
                remaining = current_timeout - (time.monotonic() - ws_connection.last_activity)
                if remaining <= 0:
                    logger.info(
                        f"No activity for {time.monotonic() - ws_connection.last_activity:.1f}s"
                        f" (timeout: {current_timeout}s), exiting"
                    )
                    next_event_task.cancel()
                    return
                done, _ = await asyncio.wait({next_event_task}, timeout=min(remaining, poll_interval))
            else:
                done, _ = await asyncio.wait({next_event_task}, timeout=poll_interval)
            if next_event_task in done:
                try:
                    event = next_event_task.result()
                except StopAsyncIteration:
                    return
                next_event_task = None
                if event_idle_timeouts is not None and event.type in event_idle_timeouts:
                    current_timeout = event_idle_timeouts[event.type]
                else:
                    current_timeout = idle_timeout
    except websockets.exceptions.ConnectionClosedError:
        logger.info("Connection closed")


class TimestampedMessage[T](NamedTuple):
    timestamp: float
    direction: Literal["sent", "received"]
    message: T


class RecordingConnection[SendT, RecvT](ClientConnection):
    messages: list[TimestampedMessage[SendT] | TimestampedMessage[RecvT]]
    _send_listeners: list[Callable[[SendT], None]]
    _recv_listeners: list[Callable[[RecvT], None]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self._send_listeners = []
        self._recv_listeners = []
        self.last_activity: float = time.monotonic()

    def add_send_listener(self, callback: Callable[[SendT], None]) -> None:
        self._send_listeners.append(callback)

    def add_recv_listener(self, callback: Callable[[RecvT], None]) -> None:
        self._recv_listeners.append(callback)

    def _convert_sent(self, message: str | bytes) -> SendT:
        return message  # type: ignore[return-value]

    def _convert_received(self, message: str | bytes) -> RecvT:
        return message  # type: ignore[return-value]

    async def send(self, message: str | bytes, text: bool | None = None) -> None:  # type: ignore[override]
        await super().send(message, text=text)
        self.last_activity = time.monotonic()
        converted = self._convert_sent(message)
        self.messages.append(
            TimestampedMessage(time.monotonic(), "sent", converted)  # pyrefly: ignore[bad-argument-type]
        )
        for listener in self._send_listeners:
            listener(converted)

    async def recv(self, decode: bool | None = None) -> str | bytes:  # type: ignore[override]
        message = await super().recv(decode=decode)
        self.last_activity = time.monotonic()
        converted = self._convert_received(message)
        self.messages.append(TimestampedMessage(time.monotonic(), "received", converted))
        for listener in self._recv_listeners:
            listener(converted)
        return message

    async def recv_event(self) -> RecvT:
        await self.recv()
        return self.messages[-1].message  # type: ignore[return-value]


class RecordingRealtimeConnection(RecordingConnection[RealtimeClientEvent, RealtimeServerEvent]):
    def _convert_sent(self, message: str | bytes) -> RealtimeClientEvent:
        data = message if isinstance(message, str) else message.decode()
        return construct_type_unchecked(value=json.loads(data), type_=cast("Any", RealtimeClientEvent))

    def _convert_received(self, message: str | bytes) -> RealtimeServerEvent:
        data = message if isinstance(message, str) else message.decode()
        return construct_type_unchecked(value=json.loads(data), type_=cast("Any", RealtimeServerEvent))


TimestampedRealtimeMessage = TimestampedMessage[RealtimeServerEvent] | TimestampedMessage[RealtimeClientEvent]


@asynccontextmanager
async def realtime_session_factory(
    connection_url: str, api_token: str | None
) -> AsyncGenerator[AsyncRealtimeConnection, None]:
    additional_headers = {}
    if api_token is not None:
        additional_headers["Authorization"] = "Bearer " + api_token
    logger.debug(f"Connecting to {connection_url}")
    async with connect(
        connection_url, additional_headers=additional_headers, create_connection=RecordingRealtimeConnection
    ) as ws_connection:
        try:
            realtime_connection = AsyncRealtimeConnection(ws_connection)
            logger.info(f"Connected to {connection_url}")
            yield realtime_connection

        except websockets.WebSocketException:
            logger.exception(f"Failed to connect to {connection_url}")
