from __future__ import annotations

from collections import OrderedDict
import logging
from typing import TYPE_CHECKING

from speaches.realtime.event_router import EventRouter
from speaches.realtime.utils import generate_conversation_id
from speaches.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    Error,
    ErrorEvent,
    create_server_error,
)

if TYPE_CHECKING:
    from openai.types.realtime import ConversationItemTruncateEvent

    from speaches.realtime.context import SessionContext
    from speaches.realtime.pubsub import EventPubSub


logger = logging.getLogger(__name__)

event_router = EventRouter()


class Conversation:
    def __init__(self, pubsub: EventPubSub) -> None:
        self.id = generate_conversation_id()
        self.items = OrderedDict[str, ConversationItem]()
        self.pubsub = pubsub

    def create_item(self, item: ConversationItem, previous_item_id: str | None = None) -> None:
        # TODO: handle `previous_item_id == "root"`. See https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/create#realtime-client-events/conversation/item/create-previous_item_id
        if item.id in self.items:
            # NOTE: Weirdly OpenAI's API allows creating an item with an already existing ID! Their implementation doesn't seem to replace the existing item. Rather, it just adds a new item with the same ID at the end. Me returning an error here deviates from their implementation.
            self.pubsub.publish_nowait(
                ErrorEvent(
                    error=Error(
                        type="invalid_request_error",
                        message=f"Error adding item: the item with id '{item.id}' already exists.",
                    )
                )
            )
            return

        if previous_item_id is not None and previous_item_id not in self.items:
            self.pubsub.publish_nowait(
                ErrorEvent(
                    error=Error(
                        type="invalid_request_error",
                        message=f"Error adding item: the previous item with id '{previous_item_id}' does not exist.",
                    )
                )
            )
            return
        else:
            previous_item_id = next(reversed(self.items), None)

        self.items[item.id] = item
        self.pubsub.publish_nowait(ConversationItemCreatedEvent(previous_item_id=previous_item_id, item=item))

    def delete_item(self, item_id: str) -> None:
        if item_id not in self.items:
            self.pubsub.publish_nowait(
                ErrorEvent(
                    error=Error(
                        type="invalid_request_error",
                        message=f"Error deleting item: the item with id '{item_id}' does not exist.",
                    )
                )
            )
        else:
            # TODO: What should be done if this a conversation that's being currently genererated?
            del self.items[item_id]
            self.pubsub.publish_nowait(ConversationItemDeletedEvent(item_id=item_id))


# Client Events
@event_router.register("conversation.item.create")
def handle_conversation_item_create_event(ctx: SessionContext, event: ConversationItemCreateEvent) -> None:
    # TODO: What should happen if this get's called when a response is being generated?
    ctx.conversation.create_item(event.item)


@event_router.register("conversation.item.truncate")
def handle_conversation_item_truncate_event(ctx: SessionContext, event: ConversationItemTruncateEvent) -> None:
    ctx.pubsub.publish_nowait(
        create_server_error(f"Handling of the '{event.type}' event is not implemented.", event_id=event.event_id)
    )


@event_router.register("conversation.item.delete")
def handle_conversation_item_delete_event(ctx: SessionContext, event: ConversationItemDeleteEvent) -> None:
    ctx.conversation.delete_item(event.item_id)
