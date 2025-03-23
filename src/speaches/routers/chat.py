import base64
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from io import BytesIO
import logging
import time
from typing import Annotated, Self
from uuid import uuid4

import aiostream
from cachetools import TTLCache
from fastapi import APIRouter, Body, Response
from fastapi.responses import StreamingResponse
import openai
from openai import AsyncStream
from openai.resources.audio import AsyncSpeech
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudio,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import Field, model_validator

from speaches.dependencies import (
    CompletionClientDependency,
    SpeechClientDependency,
    TranscriptionClientDependency,
)
from speaches.routers.stt import format_as_sse
from speaches.text_utils import SentenceChunker
from speaches.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
)
from speaches.types.chat import (
    CompletionCreateParamsBase as OpenAICompletionCreateParamsBase,
)

# Resources:
# - https://platform.openai.com/docs/guides/audio

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-model
DEFAULT_SPEECH_MODEL = "tts-1"  # or "tts-1-hd"
# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-model
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
AUDIO_TRANSCRIPTION_CACHE_SIZE = 4096
AUDIO_TRANSCRIPTION_TTL_SECONDS = 60 * 60

logger = logging.getLogger(__name__)
router = APIRouter()
cache: TTLCache[str, str] = TTLCache(maxsize=AUDIO_TRANSCRIPTION_CACHE_SIZE, ttl=AUDIO_TRANSCRIPTION_TTL_SECONDS)


# NOTE: OpenAI doesn't use UUIDs


def generate_audio_id() -> str:
    return "audio_" + str(uuid4())


def generate_chat_completion_id() -> str:
    return "chatcmpl-" + str(uuid4())


# TODO: support model aliasing
class CompletionCreateParamsBase(OpenAICompletionCreateParamsBase):
    stream: bool = False
    trancription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    transcription_extra_body: dict | None = None
    speech_model: str = DEFAULT_SPEECH_MODEL
    speech_extra_body: dict | None = Field(default_factory=lambda: {"sample_rate": 24000})

    @model_validator(mode="after")
    def validate_audio_format_when_stream(self) -> Self:
        # NOTE: OpenAI only supports pcm format for streaming. We can support any format but keeping this hardcoded for consistency
        if self.stream and self.audio is not None and self.audio.format != "pcm16":
            raise ValueError(
                f"Unsupported value: 'audio.format' does not support '{self.audio.format}' when stream=true. Supported values are: 'pcm16'."
            )
        return self


def transform_choice_delta(choice_delta: ChoiceDelta) -> ChoiceDelta:
    if choice_delta.content is None:
        return choice_delta

    content = choice_delta.content
    choice_delta.content = None
    choice_delta.audio = {  # pyright: ignore[reportAttributeAccessIssue]
        "transcript": content,
    }
    return choice_delta


# FIXME: do not pass in `body`
async def transform_choice(speech_client: AsyncSpeech, choice: Choice, body: CompletionCreateParamsBase) -> Choice:
    assert body.audio is not None
    # HACK: because OpenAI alternates between `pcm16`(/v1/chat/completions) and `pcm`(/v1/audio/speech)
    audio_format = "pcm" if body.audio.format == "pcm16" else body.audio.format

    if choice.message.content is None:
        return choice
    res = await speech_client.create(
        input=choice.message.content,
        model=body.speech_model,
        voice=body.audio.voice,  # pyright: ignore[reportArgumentType]
        response_format=audio_format,
    )
    audio_bytes = res.read()
    audio_id = generate_audio_id()
    cache[audio_id] = choice.message.content
    choice.message.audio = ChatCompletionAudio(
        id=audio_id,
        data=base64.b64encode(audio_bytes).decode("utf-8"),
        transcript=choice.message.content,
        expires_at=int((datetime.now(UTC) + timedelta(seconds=AUDIO_TRANSCRIPTION_TTL_SECONDS)).timestamp()),
    )
    choice.message.content = None
    return choice


# TODO: document minor deviations from OAI
# TODO: rework modalities handling
class AudioChatStream:
    def __init__(
        self,
        chat_completion_chunk_stream: AsyncStream[ChatCompletionChunk],
        speech_client: AsyncSpeech,
        sentence_chunker: SentenceChunker,
        body: CompletionCreateParamsBase,  # FIXME: do not pass in `body`
    ) -> None:
        self.chat_completion_chunk_stream = chat_completion_chunk_stream
        self.speech_client = speech_client
        self.sentence_chunker = sentence_chunker  # NOTE: this should be for every choice is I want to support n > 1
        self.body = body
        self.audio_id = generate_audio_id()
        self.expires_at = int((datetime.now(UTC) + timedelta(seconds=AUDIO_TRANSCRIPTION_TTL_SECONDS)).timestamp())
        self.chat_completion_id: str
        self.created: int

    async def text_chat_completion_chunk_stream(self) -> AsyncGenerator[ChatCompletionChunk]:
        start = time.perf_counter()
        async for chunk in self.chat_completion_chunk_stream:
            if len(chunk.choices) == 0:
                logger.warning(f"Received a chunk with no choices: {chunk}")
                continue
            self.chat_completion_id = chunk.id
            self.created = chunk.created
            choice = chunk.choices[0]
            assert self.body.modalities is not None
            if "audio" not in self.body.modalities:  # do not transform the choice if audio is not in the modalities
                yield chunk
                continue
            if choice.delta.content is not None:
                self.sentence_chunker.add_token(choice.delta.content)
                choice.delta = transform_choice_delta(choice.delta)
                choice.delta.audio["id"] = self.audio_id  # pyright: ignore[reportAttributeAccessIssue]
                choice.delta.audio["expires_at"] = self.expires_at  # pyright: ignore[reportAttributeAccessIssue]
            # TODO: consider not sending the chunk if there's a finish_reason
            # if choice.finish_reason is None:
            yield chunk
        self.sentence_chunker.close()
        logger.info(f"Text generation took {time.perf_counter() - start:.2f} seconds")

    async def audio_chat_completion_chunk_stream(self) -> AsyncGenerator[ChatCompletionChunk]:
        assert self.body.audio is not None

        start = time.perf_counter()
        # TODO: parallelize
        async for sentence in self.sentence_chunker:
            res = await self.speech_client.create(
                input=sentence,
                model=self.body.speech_model,
                voice=self.body.audio.voice,  # pyright: ignore[reportArgumentType]
                response_format="pcm",
                extra_body={"sample_rate": 24000},
            )
            audio_bytes = res.read()
            audio_data = base64.b64encode(audio_bytes).decode("utf-8")
            delta = ChoiceDelta()
            delta.audio = {  # pyright: ignore[reportAttributeAccessIssue]
                "id": self.audio_id,
                "data": audio_data,
                "expires_at": self.expires_at,
            }
            yield ChatCompletionChunk(
                id=self.chat_completion_id,
                choices=[ChunkChoice(delta=delta, index=0)],
                created=self.created,
                model=self.body.speech_model,
                object="chat.completion.chunk",
            )
        logger.info(f"Audio generation took {time.perf_counter() - start:.2f} seconds")

    async def __aiter__(self) -> AsyncGenerator[ChatCompletionChunk]:
        assert self.body.modalities is not None

        if "audio" in self.body.modalities:
            merged_stream = aiostream.stream.merge(
                self.text_chat_completion_chunk_stream(), self.audio_chat_completion_chunk_stream()
            )
            # without stream I got warning
            async with merged_stream.stream() as stream:
                try:
                    async for chunk in stream:
                        yield chunk
                except openai.APIStatusError:
                    logger.exception("Audio chat generation failed yooo")
        else:
            async for chunk in self.text_chat_completion_chunk_stream():
                yield chunk


# TODO: maybe propagate 400 errors


# https://platform.openai.com/docs/api-reference/chat/create
@router.post("/v1/chat/completions", response_model=ChatCompletion | ChatCompletionChunk)
async def handle_completions(  # noqa: C901
    chat_completion_client: CompletionClientDependency,
    transcription_client: TranscriptionClientDependency,
    speech_client: SpeechClientDependency,
    body: Annotated[CompletionCreateParamsBase, Body()],
) -> Response | StreamingResponse:
    assert body.n is None or body.n == 1, "Multiple choices (`n` > 1) are not supported"

    for i, message in enumerate(body.messages):
        if message.role == "user":
            content = message.content
            # per https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in#quickstart, input audio should be within the `message.content` list
            if not isinstance(content, list):
                continue

            # TODO: parallelize
            for j in range(len(content)):
                content_part = content[j]
                if content_part.type == "input_audio":
                    audio_bytes = base64.b64decode(content_part.input_audio.data)
                    # TODO: how does the endpoint know the format lol?
                    transcript = await transcription_client.create(
                        file=BytesIO(audio_bytes),
                        model=body.trancription_model,
                        response_format="text",
                    )
                    content[j] = ChatCompletionContentPartTextParam(text=transcript, type="text")
                    logger.info(f"Transcript for message {i} content part {j}: {transcript}")

        elif message.role == "assistant" and message.audio is not None:
            transcript = cache[message.audio.id]
            body.messages[i] = ChatCompletionAssistantMessageParam(
                role="assistant",
                content=transcript,
                # NOTE: I believe the fields below will always be `None`
                name=message.name,
                refusal=message.refusal,
                tool_calls=message.tool_calls,
                function_call=message.function_call,
            )

    # NOTE: rather than doing a `model_copy` it might be better to override the fields when doing the `model_dump` and destructuring
    proxied_body = body.model_copy(deep=True)
    proxied_body.modalities = ["text"]
    proxied_body.audio = None
    # NOTE: Adding --use-one-literal-as-default breaks the `exclude_defaults=True` behavior
    try:
        chat_completion = await chat_completion_client.create(**proxied_body.model_dump(exclude_defaults=True))
    except openai.APIStatusError as e:
        return Response(content=e.message, status_code=e.status_code)
    if isinstance(chat_completion, AsyncStream):

        async def inner() -> AsyncGenerator[str]:
            audio_chat_stream = AudioChatStream(chat_completion, speech_client, SentenceChunker(), body)
            async for chunk in audio_chat_stream:
                yield format_as_sse(chunk.model_dump_json())

        return StreamingResponse(inner(), media_type="text/event-stream")
    elif isinstance(chat_completion, ChatCompletion):
        for i in range(len(chat_completion.choices)):
            if body.modalities is None or "audio" not in body.modalities:
                continue
            chat_completion.choices[i] = await transform_choice(speech_client, chat_completion.choices[i], body)
        return Response(content=chat_completion.model_dump_json(), media_type="application/json")

    raise ValueError(f"Unexpected chat completion type: {type(chat_completion)}")
