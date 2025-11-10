import logging
from typing import Annotated, Any, Literal

from openai.types.beta.realtime import (
    ConversationCreatedEvent as OpenAIConversationCreatedEvent,
)
from openai.types.beta.realtime import (
    ConversationItemDeletedEvent as OpenAIConversationItemDeletedEvent,
)
from openai.types.beta.realtime import (
    ConversationItemDeleteEvent,
    ConversationItemTruncateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    RateLimitsUpdatedEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
)
from openai.types.beta.realtime import (
    ConversationItemInputAudioTranscriptionCompletedEvent as OpenAIConversationItemInputAudioTranscriptionCompletedEvent,
)
from openai.types.beta.realtime import (
    ConversationItemInputAudioTranscriptionFailedEvent as OpenAIConversationItemInputAudioTranscriptionFailedEvent,
)
from openai.types.beta.realtime import (
    ConversationItemTruncatedEvent as OpenAIConversationItemTruncatedEvent,
)
from openai.types.beta.realtime import (
    ErrorEvent as OpenAIErrorEvent,
)
from openai.types.beta.realtime import (
    InputAudioBufferClearedEvent as OpenAIInputAudioBufferClearedEvent,
)
from openai.types.beta.realtime import (
    InputAudioBufferSpeechStartedEvent as OpenAIInputAudioBufferSpeechStartedEvent,
)
from openai.types.beta.realtime import (
    InputAudioBufferSpeechStoppedEvent as OpenAIInputAudioBufferSpeechStoppedEvent,
)
from openai.types.beta.realtime import (
    ResponseAudioDeltaEvent as OpenAIResponseAudioDeltaEvent,
)
from openai.types.beta.realtime import (
    ResponseAudioDoneEvent as OpenAIResponseAudioDoneEvent,
)
from openai.types.beta.realtime import (
    ResponseAudioTranscriptDeltaEvent as OpenAIResponseAudioTranscriptDeltaEvent,
)
from openai.types.beta.realtime import (
    ResponseAudioTranscriptDoneEvent as OpenAIResponseAudioTranscriptDoneEvent,
)
from openai.types.beta.realtime import (
    ResponseFunctionCallArgumentsDeltaEvent as OpenAIResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.beta.realtime import (
    ResponseFunctionCallArgumentsDoneEvent as OpenAIResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.beta.realtime import (
    ResponseTextDeltaEvent as OpenAIResponseTextDeltaEvent,
)
from openai.types.beta.realtime import (
    ResponseTextDoneEvent as OpenAIResponseTextDoneEvent,
)
from openai.types.beta.realtime.error_event import Error
from pydantic import BaseModel, Discriminator, Field, model_validator
from pydantic.type_adapter import TypeAdapter

from speaches.realtime.utils import generate_event_id, generate_item_id

logger = logging.getLogger(__name__)


class NotGiven(BaseModel):
    pass


NOT_GIVEN = NotGiven()


class PartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class PartAudio(BaseModel):
    type: Literal["audio"] = "audio"
    transcript: str


type Part = PartText | PartAudio


# TODO: document that this type is fully custom and doesn't exist in the OpenAI API
class ConversationItemContentAudio(BaseModel):
    type: Literal["audio"] = "audio"
    transcript: str
    audio: str

    def to_part(self) -> PartAudio:
        return PartAudio(transcript=self.transcript)


class ConversationItemContentInputAudio(BaseModel):  # TODO: document the weirdness about this type
    type: Literal["input_audio"] = "input_audio"
    transcript: str | None
    # audio: str


class ConversationItemContentItemReference(BaseModel):
    type: Literal["item_reference"] = "item_reference"
    id: str


class ConversationItemContentText(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def to_part(self) -> PartText:
        return PartText(text=self.text)


class ConversationItemContentInputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


type ConversationItemContent = (
    ConversationItemContentInputText
    | ConversationItemContentInputAudio
    | ConversationItemContentItemReference
    | ConversationItemContentText
    | ConversationItemContentAudio
)


class BaseConversationItem(BaseModel):
    id: str = Field(default_factory=generate_item_id)
    object: Literal["realtime.item"] = "realtime.item"
    status: Literal["incomplete", "completed"]

    # https://docs.pydantic.dev/latest/concepts/validators/#model-validators
    @model_validator(mode="before")
    @classmethod
    # HACK: this is a workaround for `ConversationItemCreateEvent` as clients would rarely provide the status field causing a `ValidationError` to be raised. A `model_validator` is used instead of providing a default value because I want to bet getting typing errors from pyright if the field is not provided within the server code.
    def add_default_status_value(cls, data: Any) -> Any:
        if isinstance(data, dict) and "status" not in data:
            logger.warning(f"ConversationItem: {data} is missing 'status' field. Defaulting to 'completed'.")
            data["status"] = "completed"
        return data


class ConversationItemMessage(BaseConversationItem):
    type: Literal["message"] = "message"
    role: Literal["assistant", "user", "system"]
    content: list[ConversationItemContent]  # TODO: custom type


class ConversationItemFunctionCall(BaseConversationItem):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class ConversationItemFunctionCallOutput(BaseConversationItem):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


# NOTE: server can't generate "function_call_output"
type ServerConversationItem = ConversationItemMessage | ConversationItemFunctionCall
type ConversationItem = ConversationItemMessage | ConversationItemFunctionCall | ConversationItemFunctionCallOutput


class ConversationItemCreateEvent(BaseModel):
    type: Literal["conversation.item.create"] = "conversation.item.create"
    event_id: str = Field(default_factory=generate_event_id)
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemCreatedEvent(BaseModel):
    type: Literal["conversation.item.created"] = "conversation.item.created"
    event_id: str = Field(default_factory=generate_event_id)
    item: ConversationItem
    previous_item_id: str | None


class ResponseOutputItemAddedEvent(BaseModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    event_id: str = Field(default_factory=generate_event_id)
    output_index: int = 0
    response_id: str
    item: ServerConversationItem


class ResponseOutputItemDoneEvent(BaseModel):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    event_id: str = Field(default_factory=generate_event_id)
    output_index: int = 0
    response_id: str
    item: ServerConversationItem


class RealtimeResponse(BaseModel):
    id: str
    status: Literal["completed", "cancelled", "failed", "incomplete"]
    output: list[ServerConversationItem]
    modalities: list[Literal["text", "audio"]]
    object: Literal["realtime.response"] = "realtime.response"
    # TODO: add and support additional fields


class ResponseCreatedEvent(BaseModel):
    type: Literal["response.created"] = "response.created"
    event_id: str = Field(default_factory=generate_event_id)
    response: RealtimeResponse


class ResponseDoneEvent(BaseModel):
    type: Literal["response.done"] = "response.done"
    event_id: str = Field(default_factory=generate_event_id)
    response: RealtimeResponse


# Same as openai.types.beta.realtime.session_update_event.SessionTurnDetection but with all the fields made non-nullable
class TurnDetection(BaseModel):
    create_response: bool
    prefix_padding_ms: int
    silence_duration_ms: int
    threshold: float = Field(..., ge=0.0, le=1.0)
    type: Literal["server_vad"] = "server_vad"


class InputAudioTranscription(BaseModel):
    model: str
    # NOTE: `language` is a custom field not present in the OpenAI API. However, weirdly it can be found at https://github.com/openai/openai-openapi
    language: str | None = None


type AudioFormat = Literal["pcm16", "g711_ulaw", "g711_alaw"]
type Modality = Literal["text", "audio"]


class Tool(BaseModel):
    name: str
    description: str | None = None
    parameters: dict
    type: Literal["function"] = "function"


# ChatCompletionToolChoiceOptionParam: TypeAlias = Union[
#     Literal["none", "auto", "required"], ChatCompletionNamedToolChoiceParam
# ]


class Function(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    function: Function
    type: Literal["function"] = "function"


type ToolChoice = Literal["none", "auto", "required"] | NamedToolChoice


class Response(BaseModel):
    conversation: Literal["auto"]  # NOTE: there's also "none" but it's not supported in this implementation
    input: list[ConversationItem]
    instructions: str
    max_response_output_tokens: int | Literal["inf"]
    modalities: list[Modality]
    output_audio_format: AudioFormat
    temperature: float  # TODO: should there be lower and upper bounds?
    tool_choice: ToolChoice
    tools: list[Tool]
    voice: str


# TODO: which defaults should be set (if any)?
class Session(BaseModel):
    id: str  # TODO: should this be auto-generated?
    input_audio_format: AudioFormat
    input_audio_transcription: InputAudioTranscription  # NOTE: according to the spec None is a valid value here, but in this implementation it would be impossible to do anything without a transcription model
    instructions: str
    max_response_output_tokens: int | Literal["inf"]
    modalities: list[Modality]
    model: str
    output_audio_format: AudioFormat
    temperature: float  # TODO: should there be lower and upper bounds?
    tool_choice: ToolChoice
    tools: list[Tool]
    turn_detection: TurnDetection | None
    speech_model: str
    voice: str


class PartialSession(BaseModel):
    input_audio_format: AudioFormat | NotGiven = NOT_GIVEN
    input_audio_transcription: InputAudioTranscription | NotGiven = NOT_GIVEN
    instructions: str | NotGiven = NOT_GIVEN
    max_response_output_tokens: int | Literal["inf"] | NotGiven = NOT_GIVEN
    modalities: list[Modality] | NotGiven = NOT_GIVEN
    model: str | NotGiven = NOT_GIVEN
    output_audio_format: AudioFormat | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN
    tools: list[Tool] | NotGiven = NOT_GIVEN
    turn_detection: TurnDetection | NotGiven = NOT_GIVEN
    speech_model: str | NotGiven = NOT_GIVEN
    voice: str | NotGiven = NOT_GIVEN


class SessionUpdateEvent(BaseModel):
    type: Literal["session.update"] = "session.update"
    event_id: str | None = None
    session: PartialSession


class SessionCreatedEvent(BaseModel):
    type: Literal["session.created"] = "session.created"
    event_id: str = Field(default_factory=generate_event_id)
    session: Session


class SessionUpdatedEvent(BaseModel):
    type: Literal["session.updated"] = "session.updated"
    event_id: str = Field(default_factory=generate_event_id)
    session: Session


class InputAudioBufferCommittedEvent(BaseModel):
    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    event_id: str = Field(default_factory=generate_event_id)
    item_id: str
    previous_item_id: str | None


# The following classes are the same as the ones in openai.types.beta.realtime but with fields assigned some default values. This is to reduce the amount of boilerplate code when creating these events.


class InputAudioBufferSpeechStartedEvent(OpenAIInputAudioBufferSpeechStartedEvent):
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    event_id: str = Field(default_factory=generate_event_id)


class InputAudioBufferSpeechStoppedEvent(OpenAIInputAudioBufferSpeechStoppedEvent):
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    event_id: str = Field(default_factory=generate_event_id)


class ConversationCreatedEvent(OpenAIConversationCreatedEvent):
    type: Literal["conversation.created"] = "conversation.created"
    event_id: str = Field(default_factory=generate_event_id)


class ConversationItemDeletedEvent(OpenAIConversationItemDeletedEvent):
    type: Literal["conversation.item.deleted"] = "conversation.item.deleted"
    event_id: str = Field(default_factory=generate_event_id)


class ConversationItemInputAudioTranscriptionCompletedEvent(
    OpenAIConversationItemInputAudioTranscriptionCompletedEvent
):
    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0


class ConversationItemInputAudioTranscriptionFailedEvent(OpenAIConversationItemInputAudioTranscriptionFailedEvent):
    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    event_id: str = Field(default_factory=generate_event_id)


class InputAudioBufferClearedEvent(OpenAIInputAudioBufferClearedEvent):
    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"
    event_id: str = Field(default_factory=generate_event_id)


class ConversationItemTruncatedEvent(OpenAIConversationItemTruncatedEvent):
    type: Literal["conversation.item.truncated"] = "conversation.item.truncated"
    event_id: str = Field(default_factory=generate_event_id)


class ErrorEvent(OpenAIErrorEvent):
    type: Literal["error"] = "error"
    event_id: str = Field(default_factory=generate_event_id)


def create_invalid_request_error(
    message: str, code: str | None = None, event_id: str | None = None, param: str | None = None
) -> ErrorEvent:
    return ErrorEvent(
        error=Error(
            type="invalid_request_error",
            message=message,
            code=code,
            event_id=event_id,
            param=param,
        ),
    )


def create_server_error(
    message: str, code: str | None = None, event_id: str | None = None, param: str | None = None
) -> ErrorEvent:
    return ErrorEvent(
        error=Error(
            type="server_error",
            message=message,
            code=code,
            event_id=event_id,
            param=param,
        )
    )


class ResponseContentPartAddedEvent(BaseModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    event_id: str = Field(default_factory=generate_event_id)
    response_id: str
    item_id: str
    content_index: int = 0
    output_index: int = 0
    part: Part


class ResponseContentPartDoneEvent(BaseModel):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    event_id: str = Field(default_factory=generate_event_id)
    response_id: str
    item_id: str
    content_index: int = 0
    output_index: int = 0
    part: Part


class ResponseTextDeltaEvent(OpenAIResponseTextDeltaEvent):
    type: Literal["response.text.delta"] = "response.text.delta"
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0
    output_index: int = 0


class ResponseTextDoneEvent(OpenAIResponseTextDoneEvent):
    type: Literal["response.text.done"] = "response.text.done"
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0
    output_index: int = 0


class ResponseAudioTranscriptDeltaEvent(OpenAIResponseAudioTranscriptDeltaEvent):
    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0
    output_index: int = 0


class ResponseAudioDeltaEvent(OpenAIResponseAudioDeltaEvent):
    type: Literal["response.audio.delta"] = "response.audio.delta"
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0
    output_index: int = 0


class ResponseAudioDoneEvent(OpenAIResponseAudioDoneEvent):
    type: Literal["response.audio.done"] = "response.audio.done"
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0
    output_index: int = 0


class ResponseAudioTranscriptDoneEvent(OpenAIResponseAudioTranscriptDoneEvent):
    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    event_id: str = Field(default_factory=generate_event_id)
    content_index: int = 0
    output_index: int = 0


class ResponseFunctionCallArgumentsDeltaEvent(OpenAIResponseFunctionCallArgumentsDeltaEvent):
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    event_id: str = Field(default_factory=generate_event_id)
    output_index: int = 0


class ResponseFunctionCallArgumentsDoneEvent(OpenAIResponseFunctionCallArgumentsDoneEvent):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    event_id: str = Field(default_factory=generate_event_id)
    output_index: int = 0


type SessionClientEvent = SessionUpdateEvent
type SessionServerEvent = SessionCreatedEvent | SessionUpdatedEvent


type InputAudioBufferClientEvent = (
    InputAudioBufferAppendEvent | InputAudioBufferCommitEvent | InputAudioBufferClearEvent
)

type InputAudioBufferServerEvent = (
    InputAudioBufferCommittedEvent
    | InputAudioBufferClearedEvent
    | InputAudioBufferSpeechStartedEvent
    | InputAudioBufferSpeechStoppedEvent
)

type ConversationClientEvent = ConversationItemCreateEvent | ConversationItemTruncateEvent | ConversationItemDeleteEvent

type ConversationServerEvent = (
    ConversationCreatedEvent
    | ConversationItemCreatedEvent
    | ConversationItemInputAudioTranscriptionCompletedEvent
    | ConversationItemInputAudioTranscriptionFailedEvent
    | ConversationItemTruncatedEvent
    | ConversationItemDeletedEvent
)


type ResponseClientEvent = ResponseCreateEvent | ResponseCancelEvent

type ResponseContentDeltaEvent = (
    ResponseTextDeltaEvent
    | ResponseAudioTranscriptDeltaEvent
    | ResponseAudioDeltaEvent
    | ResponseFunctionCallArgumentsDeltaEvent
)

type ResponseContentDoneEvent = (
    ResponseTextDoneEvent
    | ResponseAudioTranscriptDoneEvent
    | ResponseAudioDoneEvent
    | ResponseFunctionCallArgumentsDoneEvent
)

type ResponseServerEvent = (
    ResponseCreatedEvent
    | ResponseOutputItemAddedEvent
    | ResponseContentPartAddedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseAudioTranscriptDeltaEvent
    | ResponseAudioTranscriptDoneEvent
    | ResponseAudioDeltaEvent
    | ResponseAudioDoneEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputItemDoneEvent
    | ResponseDoneEvent
)

# https://platform.openai.com/docs/guides/realtime/overview#events
CLIENT_EVENT_TYPES = {
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "conversation.item.create",
    "conversation.item.truncate",
    "conversation.item.delete",
    "response.create",
    "response.cancel",
}
SERVER_EVENT_TYPES = {
    "error",
    "session.created",
    "session.updated",
    "conversation.created",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "conversation.item.created",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.truncated",
    "conversation.item.deleted",
    "response.created",
    "response.done",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.text.delta",
    "response.text.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "response.audio.delta",
    "response.audio.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "rate_limits.updated",
}

ClientEvent = Annotated[
    SessionUpdateEvent | InputAudioBufferClientEvent | ConversationClientEvent | ResponseClientEvent,
    Discriminator("type"),
]

client_event_type_adapter = TypeAdapter[ClientEvent](ClientEvent)

ServerEvent = Annotated[
    SessionServerEvent
    | InputAudioBufferServerEvent
    | ConversationServerEvent
    | ResponseServerEvent
    | ErrorEvent
    | RateLimitsUpdatedEvent,
    Discriminator("type"),
]

server_event_type_adapter = TypeAdapter[ServerEvent](ServerEvent)


# Message fragmentation types for WebRTC
class FullMessageEvent(BaseModel):
    id: str
    type: Literal["full_message"] = "full_message"
    data: str


class PartialMessageEvent(BaseModel):
    id: str
    type: Literal["partial_message"] = "partial_message"
    data: str
    fragment_index: int
    total_fragments: int


type MessageFragment = FullMessageEvent | PartialMessageEvent

Event = ClientEvent | ServerEvent
