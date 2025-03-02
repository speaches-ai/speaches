import base64
from pathlib import Path
from typing import Self

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from pydantic import BaseModel, model_validator
import pytest

FILE_PATH = Path("audio.wav")  # transcription: Hello world
B64_AUDIO_DATA = base64.b64encode(FILE_PATH.read_bytes()).decode("utf-8")

OPENAI_MODEL = "gpt-4o-audio-preview"
TEXT_ONLY_MODEL = "gpt-4o-mini"
AUDIO_PARAM = ChatCompletionAudioParam(voice="alloy", format="wav")

# TODO: for non-streaming: validate audio format matches the one sent
# TODO: check how OpenAI behaves when more than two content parts are sent: (text, audio, audio), (audio, text, audio), etc.
# TODO: test with multiple input_audio content parts
# TODO: test with alternating content parts
# TODO: test with image content part


class AudioChatSessionArchive(BaseModel):
    res: ChatCompletion
    body: CompletionCreateParamsNonStreaming

    @model_validator(mode="after")
    def check_num_choices_matches_n(self) -> Self:
        assert len(self.res.choices) == self.body.get("n", 1)
        for i, choice in enumerate(self.res.choices):
            assert choice.index == i
            assert choice.finish_reason == "stop"
        return self

    @model_validator(mode="after")
    def correct_attributes_based_on_modality(self) -> Self:
        modalities = self.body.get("modalities")
        assert modalities is not None
        assert modalities in (["text"], ["text", "audio"])
        for choice in self.res.choices:
            if modalities == ["text"]:
                assert choice.message.content is not None and len(choice.message.content) > 0
                assert choice.message.audio is None
            elif modalities == ["text", "audio"]:
                assert choice.message.content is None
                assert choice.message.audio is not None
                assert isinstance(choice.message.audio.transcript, str)
                assert isinstance(choice.message.audio.id, str)

        return self


class AudioChatStreamingSessionArchive(BaseModel):
    res: list[ChatCompletionChunk]
    body: CompletionCreateParamsStreaming

    @model_validator(mode="after")
    def check_all_chat_completion_ids_are_the_same(self) -> Self:
        chat_completion_ids = {chat_completion.id for chat_completion in self.res}
        assert len(chat_completion_ids) == 1
        return self

    @model_validator(mode="after")
    def check_num_choices_matches_n(self) -> Self:
        for chat_completion in self.res:
            assert len(chat_completion.choices) == self.body.get("n", 1)
            for i, choice in enumerate(chat_completion.choices):
                assert choice.index == i
        return self

    @model_validator(mode="after")
    def check_created_fields_matches(self) -> Self:
        assert len({chat_completion.created for chat_completion in self.res}) == 1
        return self

    @model_validator(mode="after")
    def correct_attributes_based_on_modality(self) -> Self:
        modalities = self.body.get("modalities")
        assert modalities is not None
        assert modalities == ["text"] or modalities == ["text", "audio"]  # noqa: PLR1714
        for chat_completion in self.res:
            for choice in chat_completion.choices:
                if modalities == ["text"]:
                    with pytest.raises(AttributeError):
                        getattr(choice.delta, "audio")  # noqa: B009
                elif modalities == ["text", "audio"]:
                    assert choice.delta.content is None
        return self

    # TODO: validate
    # choices: List[Choice]
    # """A list of chat completion choices.
    # Can contain more than one elements if `n` is greater than 1. Can also be empty
    # for the last chunk if you set `stream_options: {"include_usage": true}`.
    # """

    # TODO: validate model is the same across chunks (Actually don't do this)
    # TODO: validate expires at is in 1h
    # TODO: validate audio ids are the same
    # TODO: validate finish_reason (although i'm not sure what to validate yet)
    # TODO: for streaming, validate audio format is pcm16 (validate that it's also valid and has reasonable length)


# TODO: test more audio formats
# TODO: test without sending the second content part containing input_audio


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("target", ["openai", "speaches"])
async def test_audio_chat_text(dynamic_openai_client: AsyncOpenAI, target: str) -> None:
    openai_client = dynamic_openai_client

    body = CompletionCreateParamsNonStreaming(
        model=OPENAI_MODEL if target == "openai" else TEXT_ONLY_MODEL,
        modalities=["text"],
        audio={"voice": "alloy", "format": "wav"},  # TODO: is this needed
        stream=False,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": B64_AUDIO_DATA, "format": "wav"}},
                ],
            },
        ],
    )

    chunk_completion = await openai_client.chat.completions.create(**body)
    AudioChatSessionArchive(res=chunk_completion, body=body)


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("target", ["openai", "speaches"])
async def test_audio_chat_text_audio(dynamic_openai_client: AsyncOpenAI, target: str) -> None:
    openai_client = dynamic_openai_client

    body = CompletionCreateParamsNonStreaming(
        model=OPENAI_MODEL if target == "openai" else TEXT_ONLY_MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        stream=False,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": B64_AUDIO_DATA, "format": "wav"}},
                ],
            },
        ],
    )

    chunk_completion = await openai_client.chat.completions.create(**body)
    AudioChatSessionArchive(res=chunk_completion, body=body)


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("target", ["openai", "speaches"])
async def test_audio_chat_text_stream(dynamic_openai_client: AsyncOpenAI, target: str) -> None:
    openai_client = dynamic_openai_client

    body = CompletionCreateParamsStreaming(
        model=OPENAI_MODEL if target == "openai" else TEXT_ONLY_MODEL,
        modalities=["text"],
        audio={"voice": "alloy", "format": "pcm16"},  # NOTE: is `audio`  necessary when modality is text only?
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": B64_AUDIO_DATA, "format": "wav"}},
                ],
            },
        ],
    )

    chunk_completion_chunks: list[ChatCompletionChunk] = [
        x async for x in await openai_client.chat.completions.create(**body)
    ]
    AudioChatStreamingSessionArchive(res=chunk_completion_chunks, body=body)


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("target", ["openai", "speaches"])
async def test_audio_chat_text_audio_stream(dynamic_openai_client: AsyncOpenAI, target: str) -> None:
    openai_client = dynamic_openai_client

    body = CompletionCreateParamsStreaming(
        model=OPENAI_MODEL if target == "openai" else TEXT_ONLY_MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},  # TODO: add a test ensuring this is the only supported format
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": B64_AUDIO_DATA, "format": "wav"}},
                ],
            },
        ],
    )

    chunk_completion_chunks: list[ChatCompletionChunk] = [
        x async for x in await openai_client.chat.completions.create(**body)
    ]
    AudioChatStreamingSessionArchive(res=chunk_completion_chunks, body=body)
