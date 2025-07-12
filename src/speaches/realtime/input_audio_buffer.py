from __future__ import annotations

import asyncio
from io import BytesIO
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from openai import NotGiven
from openai.types.beta.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)
from pydantic import BaseModel
import soundfile as sf

from speaches.realtime.utils import generate_item_id, task_done_callback
from speaches.types.realtime import (
    ConversationItemContentInputAudio,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemMessage,
    ServerEvent,
    Session,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from openai.resources.audio import AsyncTranscriptions

    from speaches.realtime.conversation_event_router import Conversation
    from speaches.realtime.pubsub import EventPubSub

SAMPLE_RATE = 16000
MS_SAMPLE_RATE = 16
MAX_VAD_WINDOW_SIZE_SAMPLES = 3000 * MS_SAMPLE_RATE

logger = logging.getLogger(__name__)


# NOTE not in `src/speaches/realtime/input_audio_buffer_event_router.py` due to circular import
class VadState(BaseModel):
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    # TODO: consider keeping track of what was the last audio timestamp that was processed. This value could be used to control how often the VAD is run.


# TODO: use `np.int16` instead of `np.float32` for audio data
class InputAudioBuffer:
    def __init__(self, pubsub: EventPubSub) -> None:
        self.id = generate_item_id()
        self.data: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.vad_state = VadState()
        self.pubsub = pubsub

    @property
    def size(self) -> int:
        """Number of samples in the buffer."""
        return len(self.data)

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return len(self.data) / SAMPLE_RATE

    @property
    def duration_ms(self) -> int:
        """Duration of the audio in milliseconds."""
        return len(self.data) // MS_SAMPLE_RATE

    def append(self, audio_chunk: NDArray[np.float32]) -> None:
        """Append an audio chunk to the buffer."""
        self.data = np.append(self.data, audio_chunk)

    # def commit(self) -> None:
    #     """Publish an event to indicate that the buffer is ready for processing."""
    #     self.pubsub.publish

    # TODO: come up with a better name
    @property
    def data_w_vad_applied(self) -> NDArray[np.float32]:
        if self.vad_state.audio_start_ms is None:
            return self.data
        else:
            assert self.vad_state.audio_end_ms is not None
            return self.data[
                self.vad_state.audio_start_ms * MS_SAMPLE_RATE : self.vad_state.audio_end_ms * MS_SAMPLE_RATE
            ]


class InputAudioBufferTranscriber:
    def __init__(
        self,
        *,
        pubsub: EventPubSub,
        transcription_client: AsyncTranscriptions,
        input_audio_buffer: InputAudioBuffer,
        session: Session,
        conversation: Conversation,
    ) -> None:
        self.pubsub = pubsub
        self.transcription_client = transcription_client
        self.input_audio_buffer = input_audio_buffer
        self.session = session
        self.conversation = conversation

        self.task: asyncio.Task[None] | None = None
        self.events = asyncio.Queue[ServerEvent]()

    async def _handler(self) -> None:
        content_item = ConversationItemContentInputAudio(transcript=None, type="input_audio")
        item = ConversationItemMessage(
            id=self.input_audio_buffer.id,
            role="user",
            content=[content_item],
            status="completed",  # `status == "completed"` as that's what OpenAI sends
        )
        self.conversation.create_item(item)

        file = BytesIO()
        sf.write(
            file,
            self.input_audio_buffer.data_w_vad_applied,
            samplerate=16000,
            subtype="PCM_16",
            endian="LITTLE",
            format="wav",
        )
        start = time.perf_counter()
        transcript = await self.transcription_client.create(
            file=file,
            model=self.session.input_audio_transcription.model,
            response_format="text",
            language=self.session.input_audio_transcription.language or NotGiven(),
        )
        logger.info(f"Transcription generation took {time.perf_counter() - start:.2f} seconds")
        content_item.transcript = transcript
        self.pubsub.publish_nowait(
            ConversationItemInputAudioTranscriptionCompletedEvent(
                item_id=item.id,
                transcript=transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=self.input_audio_buffer.duration,
                    type="duration",
                ),
            )
        )

    # TODO: add `timeout` parameter
    def start(self) -> None:
        assert self.task is None
        self.task = asyncio.create_task(self._handler())
        self.task.add_done_callback(task_done_callback)
