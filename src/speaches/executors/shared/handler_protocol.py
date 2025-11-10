from collections.abc import Generator
from typing import Protocol

import numpy as np
import openai.types.audio
from pydantic import BaseModel, ConfigDict

from speaches.api_types import SpeechResponseFormat, TimestampGranularities
from speaches.audio import Audio
from speaches.executors.silero_vad_v5 import SpeechTimestamp, VadOptions

MimeType = str


class SpeakerEmbeddingRequest(BaseModel):
    model_id: str
    audio: Audio

    model_config = ConfigDict(arbitrary_types_allowed=True)


type SpeakerEmbeddingResponse = np.typing.NDArray[np.float32]


class SpeakerEmbeddingHandler(Protocol):
    def handle_speaker_embedding_request(
        self, request: SpeakerEmbeddingRequest, **kwargs
    ) -> SpeakerEmbeddingResponse: ...


class SpeechRequest(BaseModel):
    model: str
    voice: str
    input: str
    instructions: str | None
    response_format: SpeechResponseFormat
    speed: float
    stream_format: str
    sample_rate: int | None


SpeechResponse = tuple[bytes, MimeType] | tuple[Generator[bytes], MimeType]


class SpeechHandler(Protocol):
    def handle_speech_request(self, request: SpeechRequest, **kwargs) -> SpeechResponse: ...


class VadRequest(BaseModel):
    audio: Audio
    vad_options: VadOptions
    model_id: str = "silero_vad_v5"
    sampling_rate: int = 16000

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VadHandler(Protocol):
    def handle_vad_request(self, request: VadRequest, **kwargs) -> list[SpeechTimestamp]: ...


class TranscriptionRequest(BaseModel):
    audio: Audio
    model: str
    stream: bool = False
    language: str | None = None
    prompt: str | None = None
    response_format: openai.types.AudioResponseFormat = "json"
    temperature: float = 0.0
    hotwords: str | None = None
    timestamp_granularities: TimestampGranularities
    speech_segments: list[SpeechTimestamp]
    vad_options: VadOptions
    without_timestamps: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


NonStreamingTranscriptionResponse = (
    tuple[str, MimeType] | openai.types.audio.Transcription | openai.types.audio.TranscriptionVerbose
)
StreamingTranscriptionEvent = (
    openai.types.audio.TranscriptionTextDeltaEvent | openai.types.audio.TranscriptionTextDoneEvent
)


class TranscriptionHandler(Protocol):
    def handle_non_streaming_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse: ...

    def handle_streaming_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> Generator[StreamingTranscriptionEvent]: ...

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)


class TranslationRequest(BaseModel):
    audio: Audio
    model: str
    prompt: str | None = None
    response_format: openai.types.AudioResponseFormat = "json"
    temperature: float = 0.0
    speech_segments: list[SpeechTimestamp]
    vad_options: VadOptions

    model_config = ConfigDict(arbitrary_types_allowed=True)


TranslationResponse = tuple[str, MimeType] | openai.types.audio.Translation | openai.types.audio.TranslationVerbose


class TranslationHandler(Protocol):
    def handle_translation_request(self, request: TranslationRequest, **kwargs) -> TranslationResponse: ...
