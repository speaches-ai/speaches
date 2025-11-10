import base64
from collections.abc import AsyncGenerator
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypedDict

import gradio as gr
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import Audio
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from pydantic import BaseModel

from speaches.audio import convert_audio_format
from speaches.config import Config
from speaches.ui.utils import openai_client_from_gradio_req

# Resources:
# - https://www.gradio.app/guides/creating-a-chatbot-fast
#   - https://www.gradio.app/guides/creating-a-chatbot-fast#multimodal-chat-interface
# - https://www.gradio.app/docs/gradio/chatinterface
# - https://www.gradio.app/docs/gradio/audio
# - https://www.gradio.app/guides/conversational-chatbot
# - https://www.gradio.app/guides/conversational-chatbot

logger = logging.getLogger(__name__)


type Modality = Literal["text", "audio"]

OUTPUT_AUDIO_SAMPLE_RATE = 24000  # FIXME: there's got to be a better way to handle this


class GradioMessage(TypedDict):
    text: str
    files: list[str]


class VoiceChatState(BaseModel):
    openai_messages: list[ChatCompletionMessageParam] = []


def gradio_message_to_openai_message(gradio_message: GradioMessage) -> ChatCompletionMessageParam:
    content: list[ChatCompletionContentPartParam] = []

    # openai_messages: list[ChatCompletionMessageParam] = []
    if len(gradio_message["text"]) > 0:
        content.append(ChatCompletionContentPartTextParam(text=gradio_message["text"], type="text"))

    for file_path in gradio_message["files"]:
        if not file_path:
            msg = "No audio file provided in gradio_message_to_openai_message (audio_chat.py). Please record or upload audio."
            raise ValueError(msg)
        content.append(
            ChatCompletionContentPartInputAudioParam(
                input_audio=InputAudio(
                    data=base64.b64encode(Path(file_path).read_bytes()).decode("utf-8"), format="wav"
                ),
                type="input_audio",
            )
        )

    openai_message = ChatCompletionUserMessageParam(content=content, role="user")
    return openai_message


def handle_text_reply(chat_completion: ChatCompletion, state: VoiceChatState) -> gr.ChatMessage:
    choice = chat_completion.choices[0]
    assert choice.message.content is not None
    text_chat_message = gr.ChatMessage(role="assistant", content=choice.message.content)
    state.openai_messages.append(
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=choice.message.content,
        )
    )
    return text_chat_message


def handle_text_audio_reply(chat_completion: ChatCompletion, state: VoiceChatState) -> list[gr.ChatMessage]:
    choice = chat_completion.choices[0]
    assert choice.message.audio is not None

    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(base64.b64decode(choice.message.audio.data))
        tmp_file_path = Path(tmp_file.name)

    text_chat_message = gr.ChatMessage(role="assistant", content=choice.message.audio.transcript)
    audio_chat_message = gr.ChatMessage(role="assistant", content=gr.FileData(path=str(tmp_file_path)))

    state.openai_messages.append(
        ChatCompletionAssistantMessageParam(
            role="assistant",
            audio=Audio(id=choice.message.audio.id),
        )
    )
    return [text_chat_message, audio_chat_message]


async def handle_text_stream_reply(
    chat_completion_chunk_stream: AsyncStream[ChatCompletionChunk], state: VoiceChatState
) -> AsyncGenerator[gr.ChatMessage]:
    text_chat_message = gr.ChatMessage(role="assistant", content="")
    assert isinstance(text_chat_message.content, str)

    async for chat_completion_chunk in chat_completion_chunk_stream:
        assert len(chat_completion_chunk.choices) == 1, "Multiple choices (`n` > 1) are not supported"
        choice = chat_completion_chunk.choices[0]
        if choice.delta.content is not None:
            text_chat_message.content += choice.delta.content
            yield text_chat_message

    state.openai_messages.append(
        ChatCompletionAssistantMessageParam(role="assistant", content=text_chat_message.content)
    )


async def create_text_audio_stream_reply(
    chat_completion_chunk_stream: AsyncStream[ChatCompletionChunk], state: VoiceChatState
) -> AsyncGenerator[list[gr.ChatMessage]]:
    text_chat_message = gr.ChatMessage(role="assistant", content="")
    assert isinstance(text_chat_message.content, str)

    audio_id: str | None = None
    full_audio_data_bytes = b""

    async for chat_completion_chunk in chat_completion_chunk_stream:
        choice = chat_completion_chunk.choices[0]
        audio_dict: dict = getattr(choice.delta, "audio", {})

        if id_ := audio_dict.get("id"):
            assert isinstance(id_, str)
            audio_id = id_

        transcript = audio_dict.get("transcript")
        audio_data = audio_dict.get("data")
        assert transcript is None or audio_data is None, "Assumption violated: transcript XOR audio_data"
        # NOTE: not a valid assumption but I want to keep it as a reference
        # assert transcript is not None or audio_data is not None, "Assumption violated: transcript OR audio_data"

        if transcript is not None:
            assert isinstance(transcript, str)
            text_chat_message.content += transcript
            yield [text_chat_message]
        elif audio_data is not None:
            logger.warning(f"Received audio data: {len(audio_data)} bytes")
            assert isinstance(audio_data, str)
            audio_data_bytes = base64.b64decode(audio_data)
            logger.warning(f"Received audio data: {len(audio_data_bytes)} bytes")
            full_audio_data_bytes += audio_data_bytes
        else:
            # NOTE: usually happens on the first and last message. Expected but keeping for reference
            logger.warning(f"Neither transcript nor audio data received: {audio_dict}")

    converted_audio_data_bytes = convert_audio_format(full_audio_data_bytes, OUTPUT_AUDIO_SAMPLE_RATE, "wav")

    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(converted_audio_data_bytes)

    yield [text_chat_message, gr.ChatMessage(role="assistant", content=gr.FileData(path=tmp_file.name))]

    assert audio_id is not None
    state.openai_messages.append(
        ChatCompletionAssistantMessageParam(
            role="assistant",
            audio=Audio(id=audio_id),
        )
    )


# NOTE: another option would have been to use `gr.load_chat` but I couldn't get it to work. Worth trying again in the future.
# gr.load_chat(
#     "https://api.openai.com/v1",
#     token="sk-xxx",
#     # model="gpt-4o",
#     model="gpt-4o-audio-preview",
#     multimodal=True,
#     # type="messages",
#     # textbox=gr.MultimodalTextbox(),
#     textbox=gr.MultimodalTextbox(
#         interactive=True,
#         file_count="multiple",
#         placeholder="Enter message or upload file...",
#         show_label=False,
#         sources=["microphone", "upload"],
#     ),
# )


def create_audio_chat_tab(config: Config) -> None:
    async def create_reply(
        message: GradioMessage,
        _history: list[gr.ChatMessage],
        model: str,
        stream: bool,
        state: VoiceChatState,
        request: gr.Request,
    ) -> AsyncGenerator[list[gr.ChatMessage] | gr.ChatMessage]:
        openai_client = openai_client_from_gradio_req(request, config)
        # openai_client = AsyncOpenAI(base_url="https://api.openai.com/v1")  # HACK: for easier testing

        state.openai_messages.append(gradio_message_to_openai_message(message))

        modalities: list[Modality] = ["text", "audio"]
        voice = "alloy"
        if "audio" not in modalities:
            audio = None
        elif stream:
            audio = ChatCompletionAudioParam(voice=voice, format="pcm16")
        else:
            audio = ChatCompletionAudioParam(
                voice=voice, format="wav"
            )  # TODO: make configurable. Will need to update temporary file suffix as well

        chat_completion = await openai_client.chat.completions.create(
            messages=state.openai_messages,
            model=model,
            modalities=modalities,
            audio=audio,
            stream=stream,
            n=1,  # explicitely set to 1 as multiple choices are not supported
        )

        if isinstance(chat_completion, ChatCompletion) and "audio" not in modalities:
            yield handle_text_reply(chat_completion, state)
        elif isinstance(chat_completion, ChatCompletion) and "audio" in modalities:
            yield handle_text_audio_reply(chat_completion, state)
        elif isinstance(chat_completion, AsyncStream) and "audio" not in modalities:
            async for chat_message in handle_text_stream_reply(chat_completion, state):
                yield chat_message
        elif isinstance(chat_completion, AsyncStream) and "audio" in modalities:
            async for chat_messages in create_text_audio_stream_reply(chat_completion, state):
                yield chat_messages
        else:
            raise ValueError(f"Unsupported response type: {type(chat_completion)}")

    async def update_chat_model_dropdown() -> gr.Dropdown:
        # NOTE: not using `openai_client_from_gradio_req` because we aren't intrested in making API calls to `speaches` but rather to whatever the user specified as LLM api
        openai_client = AsyncOpenAI(
            base_url=config.chat_completion_base_url,
            api_key=config.chat_completion_api_key.get_secret_value(),
            max_retries=0,
        )
        models = (await openai_client.models.list()).data
        model_ids: list[str] = [model.id for model in models]
        return gr.Dropdown(
            choices=model_ids,
            label="Chat Model",
            value=model_ids[0],
        )

    with gr.Tab(label="Audio Chat") as tab:
        state = gr.State(VoiceChatState())
        chat_model_dropdown = gr.Dropdown(
            choices=[],
            label="Chat Model",
            info="For this dropdown to be populated, you need to switch to a different tab and then come back to this tab.",
            value=None,
        )
        stream_checkbox = gr.Checkbox(label="Stream", value=True)
        gr.ChatInterface(
            type="messages",
            multimodal=True,  # I don't think this does anything according to https://www.gradio.app/guides/creating-a-chatbot-fast#multimodal-chat-interface
            fn=create_reply,
            textbox=gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",  # TODO: verify if this works
                placeholder="Enter message or upload file...",
                sources=["microphone", "upload"],
                # value="Count from 1 to 5",  # HACK: for easier testing
            ),
            additional_inputs=[chat_model_dropdown, stream_checkbox, state],
        )

        # FIXME: doesn't work when this is the first tab that gets loaded
        tab.select(
            update_chat_model_dropdown,
            inputs=None,
            outputs=[chat_model_dropdown],
        )
