from openai.resources.audio import AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from speaches.executors.silero_vad_v5 import SileroVADModelManager
from speaches.realtime.conversation_event_router import Conversation
from speaches.realtime.input_audio_buffer import InputAudioBufferManager
from speaches.realtime.pubsub import EventPubSub
from speaches.realtime.response_event_router import ResponseManager
from speaches.types.realtime import Session


class SessionContext:
    def __init__(
        self,
        transcription_client: AsyncTranscriptions,
        completion_client: AsyncCompletions,
        vad_model_manager: SileroVADModelManager,
        session: Session,
    ) -> None:
        self.transcription_client = transcription_client
        self.vad_model_manager = vad_model_manager

        self.session = session

        self.pubsub = EventPubSub()
        self.conversation = Conversation(self.pubsub)
        self.response_manager = ResponseManager(completion_client=completion_client, pubsub=self.pubsub)
        self.audio_buffers = InputAudioBufferManager(self.pubsub)
