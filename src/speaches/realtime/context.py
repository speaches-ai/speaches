from __future__ import annotations

import asyncio
from collections import OrderedDict
import logging
from typing import TYPE_CHECKING

from speaches.realtime.conversation_event_router import Conversation
from speaches.realtime.input_audio_buffer import InputAudioBuffer
from speaches.realtime.pubsub import EventPubSub
from speaches.types.realtime import ConversationState, Session

if TYPE_CHECKING:
    from openai.resources.chat.completions import AsyncCompletions

    from speaches.executors.shared.handler_protocol import SpeechHandler, TranscriptionHandler
    from speaches.executors.shared.registry import ExecutorRegistry
    from speaches.executors.silero_vad_v5 import SileroVADModelManager
    from speaches.realtime.response_event_router import ResponseHandler

logger = logging.getLogger(__name__)

_VALID_TRANSITIONS: dict[ConversationState, set[ConversationState]] = {
    ConversationState.IDLE: {ConversationState.LISTENING, ConversationState.GENERATING},
    ConversationState.LISTENING: {ConversationState.PROCESSING, ConversationState.IDLE},
    ConversationState.PROCESSING: {ConversationState.IDLE, ConversationState.GENERATING},
    ConversationState.GENERATING: {ConversationState.IDLE, ConversationState.LISTENING},
}


class SessionContext:
    def __init__(
        self,
        executor_registry: ExecutorRegistry,
        completion_client: AsyncCompletions,
        vad_model_manager: SileroVADModelManager,
        vad_model_id: str,
        session: Session,
        tts_model_manager: SpeechHandler | None = None,
        stt_model_manager: TranscriptionHandler | None = None,
    ) -> None:
        self.executor_registry = executor_registry
        self.completion_client = completion_client
        self.vad_model_manager = vad_model_manager
        self.vad_model_id = vad_model_id

        self.session = session

        self.pubsub = EventPubSub()
        self.conversation = Conversation(self.pubsub)
        self.response: ResponseHandler | None = None
        self._state = ConversationState.IDLE
        self.response_lock = asyncio.Lock()
        self.partial_transcription_lock = asyncio.Lock()

        input_audio_buffer = InputAudioBuffer(self.pubsub)
        self.input_audio_buffers = OrderedDict[str, InputAudioBuffer]({input_audio_buffer.id: input_audio_buffer})
        self.partial_transcription_task: asyncio.Task[None] | None = None
        self.barge_in_task: asyncio.Task[None] | None = None

        self._tts_model_manager: SpeechHandler = (
            tts_model_manager
            if tts_model_manager is not None
            else executor_registry.resolve_tts_model_manager(session.speech_model)
        )
        self._tts_model_id: str = session.speech_model
        self._stt_model_manager: TranscriptionHandler = (
            stt_model_manager
            if stt_model_manager is not None
            else executor_registry.resolve_stt_model_manager(session.input_audio_transcription.model)
        )
        self._stt_model_id: str = session.input_audio_transcription.model

    @property
    def tts_model_manager(self) -> SpeechHandler:
        current = self.session.speech_model
        if current != self._tts_model_id:
            self._tts_model_manager = self.executor_registry.resolve_tts_model_manager(current)
            self._tts_model_id = current
        return self._tts_model_manager

    @property
    def stt_model_manager(self) -> TranscriptionHandler:
        current = self.session.input_audio_transcription.model
        if current != self._stt_model_id:
            self._stt_model_manager = self.executor_registry.resolve_stt_model_manager(current)
            self._stt_model_id = current
        return self._stt_model_manager

    @property
    def state(self) -> ConversationState:
        return self._state

    @state.setter
    def state(self, new_state: ConversationState) -> None:
        old = self._state
        if old == new_state:
            return
        if new_state not in _VALID_TRANSITIONS.get(old, set()):
            msg = f"Unexpected state transition: {old.value} -> {new_state.value}"
            if logger.isEnabledFor(logging.DEBUG):
                raise RuntimeError(msg)
            logger.warning(msg)
        self._state = new_state
