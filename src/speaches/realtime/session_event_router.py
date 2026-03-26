from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.beta.realtime.error_event import Error
from pydantic import BaseModel

from speaches.realtime.event_router import EventRouter
from speaches.types.realtime import (
    NOT_GIVEN,
    ErrorEvent,
    NotGiven,
    PartialSession,
    PartialTurnDetection,
    Session,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    TurnDetection,
)

if TYPE_CHECKING:
    from speaches.realtime.context import SessionContext

logger = logging.getLogger(__name__)

event_router = EventRouter()

_SESSION_EXCLUDE_FIELDS = frozenset({"input_audio_format", "output_audio_format"})


def _build_session_update(session: Session, partial: PartialSession) -> dict:
    update: dict = {}
    for field_name in PartialSession.model_fields:
        if field_name in _SESSION_EXCLUDE_FIELDS:
            continue
        value = getattr(partial, field_name)
        if isinstance(value, NotGiven):
            continue
        if field_name == "turn_detection" and isinstance(value, PartialTurnDetection):
            existing = session.turn_detection
            if isinstance(existing, TurnDetection):
                value = existing.model_copy(
                    update={
                        k: v for k, v in value.model_dump(exclude_defaults=True).items() if k != "prefix_padding_ms"
                    }
                )
        elif isinstance(value, BaseModel):
            existing = getattr(session, field_name, None)
            if isinstance(existing, BaseModel):
                value = existing.model_copy(update=value.model_dump(exclude_defaults=True))
        update[field_name] = value
    return update


def unsupported_field_error(field: str) -> ErrorEvent:
    return ErrorEvent(
        error=Error(
            type="invalid_request_error",
            message=f"Specifying `{field}` is not supported. The server either does not support this field or it is not configurable.",
        )
    )


@event_router.register("session.update")
def handle_session_update_event(ctx: SessionContext, event: SessionUpdateEvent) -> None:
    if event.session.input_audio_format != NOT_GIVEN:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.input_audio_format"))
    if event.session.output_audio_format != NOT_GIVEN:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.output_audio_format"))
    if (
        event.session.turn_detection is not None
        and isinstance(event.session.turn_detection, PartialTurnDetection)
        and event.session.turn_detection.prefix_padding_ms != NOT_GIVEN
    ):
        ctx.pubsub.publish_nowait(unsupported_field_error("session.turn_detection.prefix_padding_ms"))

    session_update = _build_session_update(ctx.session, event.session)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Applying session configuration update: {session_update}")
        logger.debug(f"Session configuration before update: {ctx.session}")
    ctx.session = ctx.session.model_copy(update=session_update)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Session configuration after update: {ctx.session}")

    ctx.pubsub.publish_nowait(SessionUpdatedEvent(session=ctx.session))
