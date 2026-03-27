from __future__ import annotations

import asyncio
import base64
import concurrent.futures
from contextlib import contextmanager
import logging
import threading
from typing import TYPE_CHECKING

import openai
from openai.types.beta.realtime.error_event import Error
from openai.types.beta.realtime.response_create_event import Response as OAIResponseConfig
from pydantic import BaseModel

from speaches import text_utils
from speaches.executors.shared.handler_protocol import SpeechHandler, SpeechRequest
from speaches.realtime.chat_utils import (
    create_completion_params,
    items_to_chat_messages,
)
from speaches.realtime.event_router import EventRouter
from speaches.realtime.session_event_router import unsupported_field_error
from speaches.realtime.utils import generate_response_id, task_done_callback
from speaches.text_utils import PhraseChunker
from speaches.types.realtime import (
    ConversationItemContentAudio,
    ConversationItemContentText,
    ConversationItemFunctionCall,
    ConversationItemMessage,
    ConversationState,
    ErrorEvent,
    RealtimeResponse,
    Response,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ServerConversationItem,
    Tool,
    create_server_error,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from openai.resources.chat import AsyncCompletions
    from openai.types.chat import ChatCompletionChunk

    from speaches.audio import Audio
    from speaches.realtime.context import SessionContext
    from speaches.realtime.conversation_event_router import Conversation
    from speaches.realtime.pubsub import EventPubSub
logger = logging.getLogger(__name__)

event_router = EventRouter()

_RESPONSE_EXCLUDE_FIELDS = frozenset({"conversation", "input", "output_audio_format", "metadata"})
_tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tts")


def _build_response_update(event_response: OAIResponseConfig) -> dict:
    update: dict = {}
    for field_name in OAIResponseConfig.model_fields:
        if field_name in _RESPONSE_EXCLUDE_FIELDS:
            continue
        value = getattr(event_response, field_name)
        if value is None:
            continue
        if field_name == "tools":
            value = [Tool.model_validate(t.model_dump()) for t in value]
        update[field_name] = value
    return update


class ChoiceDeltaAudio(BaseModel):
    id: str | None = None
    transcript: str | None = None
    data: str | None = None
    expires_at: int | None = None


class ResponseHandler:
    def __init__(
        self,
        *,
        completion_client: AsyncCompletions,
        tts_model_manager: SpeechHandler,
        model: str,
        speech_model: str,
        configuration: Response,
        conversation: Conversation,
        pubsub: EventPubSub,
        no_response_token: str | None = None,
    ) -> None:
        self.id = generate_response_id()
        self.completion_client = completion_client
        self.tts_model_manager = tts_model_manager
        self.model = model  # NOTE: unfortunatly `Response` doesn't have a `model` field
        self.speech_model = speech_model
        self.configuration = configuration
        self.conversation = conversation
        self.pubsub = pubsub
        self.no_response_token = no_response_token
        self.response = RealtimeResponse(
            id=self.id,
            status="incomplete",
            output=[],
            modalities=configuration.modalities,
        )
        self.task: asyncio.Task[None] | None = None
        self._cancelled = False
        self.dismissed = False
        self.pre_response_item_id: str | None = None
        self.audio_duration_ms: int = 0

    @contextmanager
    def add_output_item[T: ServerConversationItem](self, item: T) -> Generator[T, None, None]:
        self.response.output.append(item)
        self.pubsub.publish_nowait(ResponseOutputItemAddedEvent(response_id=self.id, item=item))
        try:
            yield item
        finally:
            if self._cancelled:
                item.status = "incomplete"
            else:
                item.status = "completed"
            self.pubsub.publish_nowait(ResponseOutputItemDoneEvent(response_id=self.id, item=item))

    @contextmanager
    def add_item_content[T: ConversationItemContentText | ConversationItemContentAudio](
        self, item: ConversationItemMessage, content: T
    ) -> Generator[T, None, None]:
        item.content.append(content)
        self.pubsub.publish_nowait(
            ResponseContentPartAddedEvent(response_id=self.id, item_id=item.id, part=content.to_part())
        )
        yield content
        self.pubsub.publish_nowait(
            ResponseContentPartDoneEvent(response_id=self.id, item_id=item.id, part=content.to_part())
        )

    async def conversation_item_message_text_handler(self, chunk_stream: AsyncGenerator[ChatCompletionChunk]) -> None:
        with self.add_output_item(ConversationItemMessage(role="assistant", status="incomplete", content=[])) as item:
            self.conversation.create_item(item)

            with self.add_item_content(item, ConversationItemContentText(text="")) as content:
                async for chunk in chunk_stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]

                    if choice.delta.content is not None:
                        content.text += choice.delta.content
                        self.pubsub.publish_nowait(
                            ResponseTextDeltaEvent(item_id=item.id, response_id=self.id, delta=choice.delta.content)
                        )

                self.pubsub.publish_nowait(
                    ResponseTextDoneEvent(item_id=item.id, response_id=self.id, text=content.text)
                )

    async def conversation_item_message_audio_handler(self, chunk_stream: AsyncGenerator[ChatCompletionChunk]) -> None:
        with self.add_output_item(ConversationItemMessage(role="assistant", status="incomplete", content=[])) as item:
            self.conversation.create_item(item)

            with self.add_item_content(item, ConversationItemContentAudio(audio="", transcript="")) as content:
                sentence_chunker = PhraseChunker()

                async def process_text_stream() -> None:
                    try:
                        async for chunk in chunk_stream:
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]

                            # Native audio path (e.g. OpenAI GPT-4o with audio)
                            audio = getattr(choice.delta, "audio", None)
                            if audio is not None:
                                assert isinstance(audio, dict), chunk
                                parsed_audio = ChoiceDeltaAudio(**audio)
                                if parsed_audio.transcript is not None:
                                    content.transcript += parsed_audio.transcript
                                    self.pubsub.publish_nowait(
                                        ResponseAudioTranscriptDeltaEvent(
                                            item_id=item.id, response_id=self.id, delta=parsed_audio.transcript
                                        )
                                    )
                                if parsed_audio.data is not None:
                                    self.pubsub.publish_nowait(
                                        ResponseAudioDeltaEvent(
                                            item_id=item.id, response_id=self.id, delta=parsed_audio.data
                                        )
                                    )
                                continue

                            # Text-only LLM path: feed text to sentence chunker for TTS
                            if choice.delta.content is not None:
                                content.transcript += choice.delta.content
                                self.pubsub.publish_nowait(
                                    ResponseAudioTranscriptDeltaEvent(
                                        item_id=item.id, response_id=self.id, delta=choice.delta.content
                                    )
                                )
                                sentence_chunker.add_token(choice.delta.content)
                    finally:
                        sentence_chunker.close()

                async def process_tts_stream() -> None:
                    async for sentence in sentence_chunker:
                        sentence_clean = text_utils.clean_for_tts(sentence)
                        if not sentence_clean:
                            continue
                        request = SpeechRequest(
                            model=self.speech_model,
                            voice=self.configuration.voice,
                            text=sentence_clean,
                            speed=1.0,
                        )
                        async for audio in self._stream_tts_chunks(request):
                            audio.resample(24000)
                            audio_bytes = audio.as_bytes()
                            # Track audio duration: PCM16 at 24kHz = 2 bytes per sample
                            audio_samples = len(audio_bytes) // 2
                            self.audio_duration_ms += (audio_samples * 1000) // 24000
                            audio_data = base64.b64encode(audio_bytes).decode("utf-8")
                            self.pubsub.publish_nowait(
                                ResponseAudioDeltaEvent(item_id=item.id, response_id=self.id, delta=audio_data)
                            )

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(process_text_stream())
                    tg.create_task(process_tts_stream())

                self.pubsub.publish_nowait(ResponseAudioDoneEvent(item_id=item.id, response_id=self.id))
                self.pubsub.publish_nowait(
                    ResponseAudioTranscriptDoneEvent(
                        item_id=item.id, response_id=self.id, transcript=content.transcript
                    )
                )

    async def conversation_item_function_call_handler(self, chunk_stream: AsyncGenerator[ChatCompletionChunk]) -> None:
        async for chunk in chunk_stream:
            if chunk.choices:
                break
        else:
            return

        assert len(chunk.choices) == 1, chunk
        choice = chunk.choices[0]
        assert choice.delta.tool_calls is not None and len(choice.delta.tool_calls) == 1, chunk
        tool_call = choice.delta.tool_calls[0]
        assert (
            tool_call.id is not None
            and tool_call.function is not None
            and tool_call.function.name is not None
            and tool_call.function.arguments is not None
        ), chunk
        item = ConversationItemFunctionCall(
            status="incomplete",
            call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        assert item.call_id is not None and item.arguments is not None and item.name is not None, item

        with self.add_output_item(item):
            self.conversation.create_item(item)

            async for chunk in chunk_stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]

                if choice.delta.tool_calls is not None:
                    assert len(choice.delta.tool_calls) == 1, chunk
                    tool_call = choice.delta.tool_calls[0]
                    assert tool_call.function is not None and tool_call.function.arguments is not None, chunk
                    self.pubsub.publish_nowait(
                        ResponseFunctionCallArgumentsDeltaEvent(
                            item_id=item.id,
                            response_id=self.id,
                            call_id=item.call_id,
                            delta=tool_call.function.arguments,
                        )
                    )
                    item.arguments += tool_call.function.arguments

            self.pubsub.publish_nowait(
                ResponseFunctionCallArgumentsDoneEvent(
                    arguments=item.arguments, call_id=item.call_id, item_id=item.id, response_id=self.id
                )
            )

    async def _stream_tts_chunks(self, request: SpeechRequest) -> AsyncGenerator[Audio, None]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Audio | None] = asyncio.Queue()
        stop_event = threading.Event()

        def _produce() -> None:
            try:
                for chunk in self.tts_model_manager.handle_speech_request(request):
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(_tts_executor, _produce)
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_event.set()

    def _check_dismissed(self) -> None:
        if not self.no_response_token:
            return
        last_item = next(reversed(self.conversation.items.values()), None)
        if not isinstance(last_item, ConversationItemMessage) or last_item.role != "assistant":
            return
        for content in last_item.content:
            transcript = getattr(content, "transcript", None) or getattr(content, "text", None) or ""
            if transcript.strip() == self.no_response_token:
                self.dismissed = True
                logger.info(f"Response dismissed: LLM returned no-response token {self.no_response_token!r}")
                return

    async def generate_response(self) -> None:
        self.pre_response_item_id = next(reversed(list(self.conversation.items)), None)
        try:
            messages = list(items_to_chat_messages(list(self.conversation.items.values())))
            completion_params = create_completion_params(self.model, messages, self.configuration)
            chunk_stream = await self.completion_client.create(**completion_params)
            async for chunk in chunk_stream:
                if chunk.choices:
                    break
            else:
                return

            is_tool_call = chunk.choices[0].delta.tool_calls is not None

            async def merge_chunks_and_chunk_stream(
                *chunks: ChatCompletionChunk, chunk_stream: openai.AsyncStream[ChatCompletionChunk]
            ) -> AsyncGenerator[ChatCompletionChunk]:
                for chunk in chunks:
                    yield chunk
                async for chunk in chunk_stream:
                    yield chunk

            if is_tool_call:
                await self.conversation_item_function_call_handler(
                    merge_chunks_and_chunk_stream(chunk, chunk_stream=chunk_stream)
                )
            else:
                if self.configuration.modalities == ["text"]:
                    handler = self.conversation_item_message_text_handler
                else:
                    handler = self.conversation_item_message_audio_handler
                await handler(merge_chunks_and_chunk_stream(chunk, chunk_stream=chunk_stream))
                self._check_dismissed()
        except openai.APIError:
            logger.exception("Error while generating response")
            self.pubsub.publish_nowait(
                ErrorEvent(error=Error(type="server_error", message="LLM provider returned an error"))
            )
            raise

    def start(self) -> None:
        assert self.task is None
        self.task = asyncio.create_task(self.generate_response())
        self.task.add_done_callback(task_done_callback)

    def stop(self) -> None:
        self._cancelled = True
        if self.task is not None and not self.task.done():
            self.task.cancel()


async def create_and_run_response(ctx: SessionContext, configuration: Response) -> None:
    async with ctx.response_lock:
        handler = ResponseHandler(
            completion_client=ctx.completion_client,
            tts_model_manager=ctx.tts_model_manager,
            model=ctx.session.model,
            speech_model=ctx.session.speech_model,
            configuration=configuration,
            conversation=ctx.conversation,
            pubsub=ctx.pubsub,
            no_response_token=ctx.session.no_response_token,
        )
        ctx.response = handler
        ctx.pubsub.publish_nowait(ResponseCreatedEvent(response=handler.response))
        ctx.state = ConversationState.GENERATING
        handler.start()
        assert handler.task is not None
    try:
        await handler.task
    except asyncio.CancelledError:
        logger.info(f"Response {handler.id} was cancelled")
        handler.response.status = "cancelled"
        ctx.pubsub.publish_nowait(ResponseDoneEvent(response=handler.response))
    except Exception:
        logger.exception(f"Response {handler.id} failed")
        handler.response.status = "failed"
        ctx.pubsub.publish_nowait(ResponseDoneEvent(response=handler.response))
    else:
        handler.response.status = "completed"
        ctx.pubsub.publish_nowait(ResponseDoneEvent(response=handler.response))
        if handler.dismissed:
            for item in handler.response.output:
                ctx.conversation.delete_item(item.id)
            if handler.pre_response_item_id:
                ctx.conversation.delete_item(handler.pre_response_item_id)
    finally:
        if ctx.response is handler:
            ctx.response = None
            if ctx.state == ConversationState.GENERATING:
                ctx.state = ConversationState.IDLE


@event_router.register("response.create")
async def handle_response_create_event(ctx: SessionContext, event: ResponseCreateEvent) -> None:
    if ctx.response is not None:
        ctx.response.stop()

    configuration = Response(
        conversation="auto", input=list(ctx.conversation.items.values()), **ctx.session.model_dump()
    )
    if event.response is not None:
        if event.response.conversation is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.conversation"))
        if event.response.input is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.input"))
        if event.response.output_audio_format is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.output_audio_format"))
        if event.response.metadata is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.metadata"))

        configuration_update = _build_response_update(event.response)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Applying response configuration update: {configuration_update}")
            logger.debug(f"Response configuration before update: {configuration}")
        configuration = configuration.model_copy(update=configuration_update)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Response configuration after update: {configuration}")

    await create_and_run_response(ctx, configuration)


@event_router.register("response.cancel")
def handle_response_cancel_event(ctx: SessionContext, event: ResponseCancelEvent) -> None:
    if ctx.response is None:
        ctx.pubsub.publish_nowait(create_server_error("No active response to cancel.", event_id=event.event_id))
        return
    ctx.response.stop()
