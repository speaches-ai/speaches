from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from speaches.types.realtime import CLIENT_EVENT_TYPES, Event

if TYPE_CHECKING:
    from collections.abc import Callable

    from speaches.realtime.context import SessionContext

logger = logging.getLogger(__name__)


class EventRouter:
    def __init__(self) -> None:
        self.event_handlers: dict[str, Callable] = {}

    def register(self, event_type: str) -> Callable:
        """Decorator to register an event handler for a specific event."""

        def decorator(func: Callable) -> Callable:
            if event_type in self.event_handlers:
                raise ValueError(f"An event handler for '{event_type}' is already registered.")

            # Register the handler for the event
            self.event_handlers[event_type] = func
            return func

        return decorator

    async def dispatch(self, ctx: SessionContext, event: Event) -> None:
        if event.type not in self.event_handlers:
            if event.type in CLIENT_EVENT_TYPES:
                logger.error(f"No handler registered for event: '{event.type}'")
            return

        handler = self.event_handlers[event.type]

        if inspect.iscoroutinefunction(handler):
            await handler(ctx, event)
        else:
            handler(ctx, event)

    def include_router(self, other_router: EventRouter) -> None:
        for event_type, handler in other_router.event_handlers.items():
            if event_type in self.event_handlers:
                raise ValueError(f"Conflict: An event handler for '{event_type}' is already registered.")

            self.event_handlers[event_type] = handler
