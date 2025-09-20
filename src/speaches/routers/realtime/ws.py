import asyncio
import logging

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketException,
    status,
)
from openai import AsyncOpenAI

from speaches.dependencies import (
    ConfigDependency,
    TranscriptionClientDependency,
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
from speaches.realtime.utils import task_done_callback, verify_websocket_api_key
from speaches.types.realtime import SessionCreatedEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

event_router = EventRouter()
event_router.include_router(conversation_event_router)
event_router.include_router(input_audio_buffer_event_router)
event_router.include_router(response_event_router)
event_router.include_router(session_event_router)


async def event_listener(ctx: SessionContext) -> None:
    try:
        async with asyncio.TaskGroup() as tg:
            async for event in ctx.pubsub.poll():
                # logger.debug(f"Received event: {event.type}")

                task = tg.create_task(event_router.dispatch(ctx, event))
                task.add_done_callback(task_done_callback)
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
    transcription_client: TranscriptionClientDependency,
    intent: str = "conversation",
    language: str | None = None,
    transcription_model: str | None = None,
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

    completion_client = AsyncOpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=0,
    ).chat.completions
    ctx = SessionContext(
        transcription_client=transcription_client,
        completion_client=completion_client,
        session=create_session_object_configuration(model, intent, language, transcription_model),
    )
    message_manager = WsServerMessageManager(ctx.pubsub)
    async with asyncio.TaskGroup() as tg:
        event_listener_task = tg.create_task(event_listener(ctx), name="event_listener")
        async with asyncio.timeout(OPENAI_REALTIME_SESSION_DURATION_SECONDS):
            mm_task = asyncio.create_task(message_manager.run(ws))
            # HACK: a tiny delay to ensure the message_manager.run() task is started. Otherwise, the `SessionCreatedEvent` will not be sent, as it's published before the `sender` task subscribes to the pubsub.
            await asyncio.sleep(0.001)
            ctx.pubsub.publish_nowait(SessionCreatedEvent(session=ctx.session))
            await mm_task
        event_listener_task.cancel()

    logger.info(f"Finished handling '{ctx.session.id}' session")
