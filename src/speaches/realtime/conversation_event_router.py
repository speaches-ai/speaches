from __future__ import annotations

from collections import OrderedDict
import logging
from typing import TYPE_CHECKING

from openai.types.beta.realtime.error_event import Error

from speaches.realtime.event_router import EventRouter
from speaches.realtime.response_event_router import create_and_run_response
from speaches.realtime.utils import generate_conversation_id
from speaches.types.realtime import (
    ConversationItem,
    ConversationItemContentAudio,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemMessage,
    ConversationItemTruncatedEvent,
    ConversationState,
    ErrorEvent,
    Response,
    create_invalid_request_error,
)

if TYPE_CHECKING:
    from openai.types.beta.realtime import ConversationItemTruncateEvent

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
    item_id = event.item_id
    content_index = event.content_index
    audio_end_ms = event.audio_end_ms

    if item_id not in ctx.conversation.items:
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(
                message=f"Error truncating item: the item with id '{item_id}' does not exist.",
                event_id=event.event_id,
            )
        )
        return

    item = ctx.conversation.items[item_id]
    if not isinstance(item, ConversationItemMessage) or item.role != "assistant":
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(
                message=f"Error truncating item: item '{item_id}' is not an assistant message.",
                event_id=event.event_id,
            )
        )
        return

    if content_index >= len(item.content):
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(
                message=f"Error truncating item: content_index {content_index} is out of range.",
                event_id=event.event_id,
            )
        )
        return

    content = item.content[content_index]
    if isinstance(content, ConversationItemContentAudio) and content.transcript:
        # Estimate transcript position from audio_end_ms.
        # Without an exact audio-to-text mapping, use a rough character-rate estimate:
        # ~150 words/minute spoken, ~5 chars/word = ~12.5 chars/second
        chars_per_ms = 12.5 / 1000
        estimated_chars = int(audio_end_ms * chars_per_ms)
        content.transcript = content.transcript[:estimated_chars]

    ctx.pubsub.publish_nowait(
        ConversationItemTruncatedEvent(
            item_id=item_id,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        )
    )


@event_router.register("conversation.item.delete")
def handle_conversation_item_delete_event(ctx: SessionContext, event: ConversationItemDeleteEvent) -> None:
    ctx.conversation.delete_item(event.item_id)


# Server Events


@event_router.register("conversation.item.input_audio_transcription.completed")
async def handle_conversation_item_input_audio_transcription_completed_event(
    ctx: SessionContext, _event: ConversationItemInputAudioTranscriptionCompletedEvent
) -> None:
    if not _event.transcript or not _event.transcript.strip():
        ctx.state = ConversationState.IDLE
        return

    if ctx.session.turn_detection is None or not ctx.session.turn_detection.create_response:
        ctx.state = ConversationState.IDLE
        return

    if ctx.response is not None:
        ctx.response.stop()

    configuration = Response(
        conversation="auto", input=list(ctx.conversation.items.values()), **ctx.session.model_dump()
    )
    await create_and_run_response(ctx, configuration)
