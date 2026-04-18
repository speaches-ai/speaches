from __future__ import annotations

from collections import OrderedDict
import logging
from typing import TYPE_CHECKING

from speaches.realtime.event_router import EventRouter
from speaches.realtime.utils import generate_conversation_id
from speaches.types.realtime import (
    ConversationItem,
    ConversationItemAddedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemDoneEvent,
    ConversationItemRetrievedEvent,
    Error,
    ErrorEvent,
    create_server_error,
)

if TYPE_CHECKING:
    from openai.types.realtime import ConversationItemRetrieveEvent, ConversationItemTruncateEvent

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
        if previous_item_id is not None and previous_item_id.strip() == "":
            logger.warning(f"Received empty string previous_item_id for item '{item.id}', treating as None")
            previous_item_id = None

        if item.id in self.items:
            # NOTE: OpenAI allows creating an item with an already existing ID without returning an error. Speaches intentionally deviates from this behavior and returns an error instead.
            self.pubsub.publish_nowait(
                ErrorEvent(
                    error=Error(
                        type="invalid_request_error",
                        message=f"Error adding item: the item with id '{item.id}' already exists.",
                        code="item_create_duplicate_id",
                    )
                )
            )
            return

        if previous_item_id == "root":
            # Insert at the beginning of the conversation
            actual_previous_item_id = None
            new_items = OrderedDict[str, ConversationItem]()
            new_items[item.id] = item
            new_items.update(self.items)
            self.items = new_items
        elif previous_item_id is not None:
            if previous_item_id not in self.items:
                self.pubsub.publish_nowait(
                    ErrorEvent(
                        error=Error(
                            type="invalid_request_error",
                            message=f"Error adding item: the previous item with id '{previous_item_id}' does not exist.",
                        )
                    )
                )
                return
            actual_previous_item_id = previous_item_id
            new_items = OrderedDict[str, ConversationItem]()
            for k, v in self.items.items():
                new_items[k] = v
                if k == previous_item_id:
                    new_items[item.id] = item
            self.items = new_items
        else:
            # Append at the end
            actual_previous_item_id = next(reversed(self.items), None)
            self.items[item.id] = item

        self.pubsub.publish_nowait(ConversationItemAddedEvent(previous_item_id=actual_previous_item_id, item=item))
        self.pubsub.publish_nowait(ConversationItemDoneEvent(previous_item_id=actual_previous_item_id, item=item))

    def delete_item(self, item_id: str) -> None:
        if item_id not in self.items:
            self.pubsub.publish_nowait(
                ErrorEvent(
                    error=Error(
                        type="invalid_request_error",
                        message=f"Error deleting item: the item with id '{item_id}' does not exist.",
                        code="item_delete_invalid_item_id",
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
    # TODO: What should happen if this gets called when a response is being generated?
    ctx.conversation.create_item(event.item, event.previous_item_id)


@event_router.register("conversation.item.retrieve")
def handle_conversation_item_retrieve_event(ctx: SessionContext, event: ConversationItemRetrieveEvent) -> None:
    item = ctx.conversation.items.get(event.item_id)
    if item is None:
        ctx.pubsub.publish_nowait(
            ErrorEvent(
                error=Error(
                    type="invalid_request_error",
                    message=f"Error retrieving item: the item with id '{event.item_id}' does not exist.",
                    code="item_retrieve_invalid_item_id",
                )
            )
        )
    else:
        ctx.pubsub.publish_nowait(ConversationItemRetrievedEvent(item=item))


@event_router.register("conversation.item.truncate")
def handle_conversation_item_truncate_event(ctx: SessionContext, event: ConversationItemTruncateEvent) -> None:
    ctx.pubsub.publish_nowait(
        create_server_error(f"Handling of the '{event.type}' event is not implemented.", event_id=event.event_id)
    )


@event_router.register("conversation.item.delete")
def handle_conversation_item_delete_event(ctx: SessionContext, event: ConversationItemDeleteEvent) -> None:
    ctx.conversation.delete_item(event.item_id)
