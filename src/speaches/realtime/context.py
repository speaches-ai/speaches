from collections import OrderedDict
from typing import TYPE_CHECKING

from openai.resources.audio import AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from speaches.executors.silero_vad_v5 import SileroVADModelManager
from speaches.realtime.conversation_event_router import Conversation
from speaches.realtime.input_audio_buffer import InputAudioBuffer
from speaches.realtime.pubsub import EventPubSub
from speaches.types.realtime import Session

if TYPE_CHECKING:
    from speaches.realtime.response_event_router import ResponseHandler


class SessionContext:
    def __init__(
        self,
        transcription_client: AsyncTranscriptions,
        completion_client: AsyncCompletions,
        vad_model_manager: SileroVADModelManager,
        session: Session,
    ) -> None:
        self.transcription_client = transcription_client
        self.completion_client = completion_client
        self.vad_model_manager = vad_model_manager

        self.session = session

        self.pubsub = EventPubSub()
        self.conversation = Conversation(self.pubsub)
        self.response: ResponseHandler | None = None

        input_audio_buffer = InputAudioBuffer(self.pubsub)
        self.input_audio_buffers = OrderedDict[str, InputAudioBuffer]({input_audio_buffer.id: input_audio_buffer})
