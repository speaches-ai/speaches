import asyncio
import contextlib
import logging

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketException,
    status,
)

from speaches.dependencies import (
    CompletionClientDependency,
    ConfigDependency,
    ExecutorRegistryDependency,
)
from speaches.realtime.context import SessionContext
from speaches.realtime.conversation_event_router import event_router as conversation_event_router
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer_event_router import (
    event_router as input_audio_buffer_event_router,
)
from speaches.realtime.message_manager import WsServerMessageManager
from speaches.realtime.response_event_router import event_router as response_event_router
from speaches.realtime.session import OPENAI_REALTIME_SESSION_DURATION_SECONDS, create_session_object_configuration
from speaches.realtime.session_event_router import event_router as session_event_router
from speaches.realtime.utils import verify_websocket_api_key
from speaches.types.realtime import Event, SessionCreatedEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

event_router = EventRouter()
event_router.include_router(conversation_event_router)
event_router.include_router(input_audio_buffer_event_router)
event_router.include_router(response_event_router)
event_router.include_router(session_event_router)


async def _safe_dispatch(ctx: SessionContext, event: Event) -> None:
    try:
        await event_router.dispatch(ctx, event)
    except Exception:
        logger.exception(f"Failed to handle {event.type} event")


async def event_listener(ctx: SessionContext) -> None:
    try:
        async with asyncio.TaskGroup() as tg:
            async for event in ctx.pubsub.poll():
                tg.create_task(_safe_dispatch(ctx, event))
    except asyncio.CancelledError:
        logger.info("Event listener task cancelled")
        raise
    finally:
        logger.info("Event listener task finished")


@router.websocket("/v1/realtime")
async def realtime(
    ws: WebSocket,
    model: str,
    config: ConfigDependency,
    completion_client: CompletionClientDependency,
    executor_registry: ExecutorRegistryDependency,
    intent: str = "conversation",
    language: str | None = None,
    transcription_model: str | None = None,
    instructions: str | None = None,
) -> None:
    """OpenAI Realtime API compatible WebSocket endpoint.

    According to OpenAI Realtime API specification:
    - 'model' parameter is the conversation model (e.g., gpt-4o-realtime-preview)
    - 'transcription_model' parameter is for input_audio_transcription.model
    - 'intent' parameter controls session behavior (conversation vs transcription)

    References:
    - https://platform.openai.com/docs/guides/realtime/overview
    - https://platform.openai.com/docs/api-reference/realtime-server-events/session/update

    """
    # Manually verify WebSocket authentication before accepting connection
    try:
        await verify_websocket_api_key(ws, config)
    except WebSocketException:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    await ws.accept()
    logger.info(f"Accepted websocket connection with intent: {intent}")

    session = create_session_object_configuration(
        model, intent, language, transcription_model, config.default_realtime_stt_model
    )
    if instructions is not None:
        session.instructions = instructions
    ctx = SessionContext(
        executor_registry=executor_registry,
        completion_client=completion_client,
        vad_model_manager=executor_registry.vad.model_manager,
        vad_model_id=executor_registry.vad_model_id,
        session=session,
    )
    message_manager = WsServerMessageManager(ctx.pubsub)
    mm_task: asyncio.Task[None] | None = None
    try:
        async with asyncio.TaskGroup() as tg:
            event_listener_task = tg.create_task(event_listener(ctx), name="event_listener")
            async with asyncio.timeout(OPENAI_REALTIME_SESSION_DURATION_SECONDS):
                mm_task = asyncio.create_task(message_manager.run(ws))
                await message_manager.ready.wait()
                ctx.pubsub.publish_nowait(SessionCreatedEvent(session=ctx.session))
                await mm_task
            event_listener_task.cancel()
    finally:
        if mm_task is not None and not mm_task.done():
            mm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mm_task
        if ctx.barge_in_task is not None and not ctx.barge_in_task.done():
            ctx.barge_in_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ctx.barge_in_task
        if ctx.partial_transcription_task is not None and not ctx.partial_transcription_task.done():
            ctx.partial_transcription_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ctx.partial_transcription_task
        if ctx.response is not None and ctx.response.task is not None and not ctx.response.task.done():
            ctx.response.stop()
            with contextlib.suppress(asyncio.CancelledError):
                await ctx.response.task
        logger.info(f"Finished handling '{ctx.session.id}' session")
