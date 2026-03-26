from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import openai.types.audio
from openai.types.beta.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)
from pydantic import BaseModel

from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import TranscriptionHandler, TranscriptionRequest
from speaches.executors.shared.vad_types import VadOptions
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

    from speaches.realtime.conversation_event_router import Conversation
    from speaches.realtime.pubsub import EventPubSub

SAMPLE_RATE = 16000
MS_SAMPLE_RATE = 16
MAX_VAD_WINDOW_SIZE_SAMPLES = 3000 * MS_SAMPLE_RATE
MAX_BUFFER_SIZE_SAMPLES = 30 * 60 * SAMPLE_RATE  # 30 minutes
_INITIAL_CAPACITY = 3200  # 200ms at 16kHz, ~12.5 KiB

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
        self._buffer: NDArray[np.float32] = np.empty(_INITIAL_CAPACITY, dtype=np.float32)
        self._size: int = 0
        self.vad_state = VadState()
        self.pubsub = pubsub
        self._vad_ring: NDArray[np.float32] = np.zeros(2 * MAX_VAD_WINDOW_SIZE_SAMPLES, dtype=np.float32)
        self._vad_ring_pos: int = 0
        self._vad_ring_filled: int = 0

    @property
    def data(self) -> NDArray[np.float32]:
        return self._buffer[: self._size]

    @property
    def vad_data(self) -> NDArray[np.float32]:
        if self._vad_ring_filled < MAX_VAD_WINDOW_SIZE_SAMPLES:
            return self._vad_ring[: self._vad_ring_filled]
        start = self._vad_ring_pos
        return self._vad_ring[start : start + MAX_VAD_WINDOW_SIZE_SAMPLES]

    @property
    def size(self) -> int:
        return self._size

    @property
    def duration(self) -> float:
        return self._size / SAMPLE_RATE

    @property
    def duration_ms(self) -> int:
        return self._size // MS_SAMPLE_RATE

    def append(self, audio_chunk: NDArray[np.float32]) -> None:
        chunk_len = len(audio_chunk)
        if self._size + chunk_len > MAX_BUFFER_SIZE_SAMPLES:
            logger.warning(
                f"Audio buffer size limit reached ({MAX_BUFFER_SIZE_SAMPLES} samples), dropping {chunk_len} new samples"
            )
            return
        required = self._size + chunk_len
        if required > len(self._buffer):
            new_capacity = max(required, len(self._buffer) * 2)
            new_buffer: NDArray[np.float32] = np.empty(new_capacity, dtype=np.float32)
            new_buffer[: self._size] = self._buffer[: self._size]
            self._buffer = new_buffer
        self._buffer[self._size : required] = audio_chunk
        self._size = required
        self._vad_ring_append(audio_chunk)

    def _vad_ring_append(self, audio_chunk: NDArray[np.float32]) -> None:
        n = len(audio_chunk)
        cap = MAX_VAD_WINDOW_SIZE_SAMPLES
        if n >= cap:
            tail = audio_chunk[-cap:]
            self._vad_ring[:cap] = tail
            self._vad_ring[cap : 2 * cap] = tail
            self._vad_ring_pos = 0
            self._vad_ring_filled = cap
            return
        pos = self._vad_ring_pos
        end = pos + n
        if end <= cap:
            self._vad_ring[pos:end] = audio_chunk
            self._vad_ring[pos + cap : end + cap] = audio_chunk
            new_pos = end % cap
        else:
            first = cap - pos
            self._vad_ring[pos:cap] = audio_chunk[:first]
            self._vad_ring[pos + cap : 2 * cap] = audio_chunk[:first]
            wrap = n - first
            self._vad_ring[:wrap] = audio_chunk[first:]
            self._vad_ring[cap : cap + wrap] = audio_chunk[first:]
            new_pos = wrap
        self._vad_ring_pos = new_pos
        self._vad_ring_filled = min(self._vad_ring_filled + n, cap)

    def consolidate(self) -> None:
        pass

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
        stt_model_manager: TranscriptionHandler,
        input_audio_buffer: InputAudioBuffer,
        session: Session,
        conversation: Conversation,
    ) -> None:
        self.pubsub = pubsub
        self.stt_model_manager = stt_model_manager
        self.input_audio_buffer = input_audio_buffer
        self.session = session
        self.conversation = conversation

        self.task: asyncio.Task[None] | None = None
        self.events = asyncio.Queue[ServerEvent]()

    async def _handler(self) -> None:
        audio = Audio(self.input_audio_buffer.data_w_vad_applied, sample_rate=SAMPLE_RATE)
        request = TranscriptionRequest(
            audio=audio,
            model=self.session.input_audio_transcription.model,
            language=self.session.input_audio_transcription.language,
            response_format="verbose_json",
            speech_segments=[],
            vad_options=VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30),
            timestamp_granularities=["segment"],
        )
        start = time.perf_counter()
        result = await asyncio.to_thread(self.stt_model_manager.handle_non_streaming_transcription_request, request)
        elapsed = time.perf_counter() - start

        # Extract transcript and check noise gate
        if isinstance(result, openai.types.audio.TranscriptionVerbose):
            transcript = result.text
            threshold = self.session.no_speech_prob_threshold
            if threshold is not None and result.segments:
                avg_no_speech = sum(s.no_speech_prob for s in result.segments) / len(result.segments)
                if avg_no_speech > threshold:
                    logger.info(
                        f"Noise gate: discarding audio (avg_no_speech_prob={avg_no_speech:.3f} > {threshold}, "
                        f"transcript={transcript!r}, elapsed={elapsed:.2f}s)"
                    )
                    self.pubsub.publish_nowait(
                        ConversationItemInputAudioTranscriptionCompletedEvent(
                            item_id=self.input_audio_buffer.id,
                            transcript="",
                            usage=UsageTranscriptTextUsageDuration(
                                seconds=self.input_audio_buffer.duration,
                                type="duration",
                            ),
                        )
                    )
                    return
        elif isinstance(result, tuple):
            transcript = result[0]
        else:
            transcript = result.text

        logger.debug(f"Transcription completed in {elapsed:.2f}s: {transcript!r}")

        if not transcript.strip():
            logger.info(f"Empty transcript: discarding audio (duration={self.input_audio_buffer.duration:.2f}s)")
            self.pubsub.publish_nowait(
                ConversationItemInputAudioTranscriptionCompletedEvent(
                    item_id=self.input_audio_buffer.id,
                    transcript="",
                    usage=UsageTranscriptTextUsageDuration(
                        seconds=self.input_audio_buffer.duration,
                        type="duration",
                    ),
                )
            )
            return

        content_item = ConversationItemContentInputAudio(transcript=transcript, type="input_audio")
        item = ConversationItemMessage(
            id=self.input_audio_buffer.id,
            role="user",
            content=[content_item],
            status="completed",
        )
        self.conversation.create_item(item)
        if item.id not in self.conversation.items:
            logger.warning(
                f"Item '{item.id}' was not added to conversation (likely duplicate), skipping transcription event"
            )
            return
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
