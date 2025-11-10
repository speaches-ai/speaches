import abc
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import fastapi
import httpx_ws
from openai.types.beta.realtime.error_event import Error
from pydantic import ValidationError

from speaches.realtime.pubsub import EventPubSub
from speaches.realtime.utils import task_done_callback
from speaches.types.realtime import (
    CLIENT_EVENT_TYPES,
    SERVER_EVENT_TYPES,
    ErrorEvent,
    Event,
    client_event_type_adapter,
    server_event_type_adapter,
)

logger = logging.getLogger(__name__)


class BaseMessageManager(abc.ABC):
    def __init__(self, event_pubsub: EventPubSub | None = None) -> None:
        if event_pubsub is None:
            event_pubsub = EventPubSub()
        self.event_pubsub = event_pubsub

    @abc.abstractmethod
    async def receiver(self, ws: Any) -> None: ...

    @abc.abstractmethod
    async def sender(self, ws: Any) -> None: ...

    async def wait_for(self, event_type: str) -> Event:
        q = self.event_pubsub.subscribe()
        try:
            while True:
                event = await q.get()
                if event.type == event_type:
                    return event
        finally:
            self.event_pubsub.subscribers.remove(q)

    async def run(self, ws: Any) -> None:
        async with asyncio.TaskGroup() as tg:
            receiver_task = tg.create_task(self.receiver(ws), name="receiver")
            sender_task = tg.create_task(self.sender(ws), name="sender")

            receiver_task.add_done_callback(task_done_callback)
            sender_task.add_done_callback(task_done_callback)

            await receiver_task
            sender_task.cancel()

    # HACK: for debugging purposes
    def dump_to_file(self, session_id: str) -> None:
        with Path(f"sessions/{session_id}.json").open("w") as f:
            print("Dumping events to file", session_id)
            f.write(json.dumps([m.model_dump() for m in self.event_pubsub.events], indent=2))


class WsClientMessageManager(BaseMessageManager):
    def __init__(self, receive_timeout: int | None = 5) -> None:
        self.event_pubsub = EventPubSub()
        self.receive_timeout = receive_timeout

    async def receiver(self, ws: httpx_ws.AsyncWebSocketSession) -> None:
        try:
            while True:
                data = await ws.receive_text(timeout=self.receive_timeout)
                try:
                    event = server_event_type_adapter.validate_json(data)
                except ValidationError:
                    logger.exception("Received an invalid server event")
                    continue

                self.event_pubsub.publish_nowait(event)
                logger.debug(f"Received {event.type} event")
        except TimeoutError:
            logger.info("Receiver task timed out")

    async def sender(self, ws: httpx_ws.AsyncWebSocketSession) -> None:
        q = self.event_pubsub.subscribe()
        try:
            while True:
                event = await q.get()
                if event.type not in CLIENT_EVENT_TYPES:
                    continue
                client_event = client_event_type_adapter.validate_python(event)
                try:
                    logger.debug(f"Sending {event.type} event")
                    await ws.send_text(client_event.model_dump_json())
                    logger.info(f"Sent {event.type} event")
                except fastapi.WebSocketDisconnect:
                    logger.info("Failed to send message due to disconnect")
                    break
        finally:
            self.event_pubsub.subscribers.remove(q)


class WsServerMessageManager(BaseMessageManager):
    async def receiver(self, ws: fastapi.WebSocket) -> None:
        logger.info("Receiver task started")
        while True:
            try:
                # logger.debug("Waiting for event")
                data = await ws.receive_text()
            except fastapi.WebSocketDisconnect:
                logger.info("Failed to receive message due to disconnect")
                break
            try:
                event = client_event_type_adapter.validate_json(data)
            except ValidationError as e:
                logger.exception("Received an invalid client event")
                await ws.send_text(
                    ErrorEvent(error=Error(type="invalid_request_error", message=str(e))).model_dump_json()
                )
                continue

            self.event_pubsub.publish_nowait(event)
            # logger.debug(f"Received {event.type} event")

    async def sender(self, ws: fastapi.WebSocket) -> None:
        logger.info("Sender task started")
        q = self.event_pubsub.subscribe()
        try:
            while True:
                # logger.debug("Waiting for event")
                event = await q.get()
                if event.type not in SERVER_EVENT_TYPES:
                    continue
                server_event = server_event_type_adapter.validate_python(event)
                try:
                    logger.debug(f"Sending {event.type} event")
                    await ws.send_text(server_event.model_dump_json())
                    logger.info(f"Sent {event.type} event")
                except fastapi.WebSocketDisconnect:
                    logger.info("Failed to send message due to disconnect")
                    break
        finally:
            self.event_pubsub.subscribers.remove(q)
