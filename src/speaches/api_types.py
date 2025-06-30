from collections.abc import Iterable
from typing import Literal

import faster_whisper.transcribe
from pydantic import BaseModel, ConfigDict

from speaches.text_utils import segments_to_text


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
class TranscriptionWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_segments(cls, segments: Iterable["TranscriptionSegment"]) -> list["TranscriptionWord"]:
        words: list[TranscriptionWord] = []
        for segment in segments:
            # NOTE: a temporary "fix" for https://github.com/speaches-ai/speaches/issues/58.
            # TODO: properly address the issue
            assert segment.words is not None, (
                "Segment must have words. If you are using an API ensure `timestamp_granularities[]=word` is set"
            )
            words.extend(segment.words)
        return words

    def offset(self, seconds: float) -> None:
        self.start += seconds
        self.end += seconds


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10938
class TranscriptionSegment(BaseModel):
    avg_logprob: float
    compression_ratio: float
    end: float
    id: int
    no_speech_prob: float
    seek: int
    start: float
    temperature: float
    text: str
    tokens: list[int]
    words: (
        list[TranscriptionWord] | None
    )  # TODO: why is here? It's not a field defined in the [OpenAI API spec](https://platform.openai.com/docs/api-reference/audio/verbose-json-object)
    # TODO: add `usage` field: https://platform.openai.com/docs/api-reference/audio/verbose-json-object#audio/verbose-json-object-usage

    @classmethod
    def from_faster_whisper_segments(
        cls, segments: Iterable[faster_whisper.transcribe.Segment]
    ) -> Iterable["TranscriptionSegment"]:
        for segment in segments:
            yield cls(
                id=segment.id,
                seek=segment.seek,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=segment.tokens,
                temperature=segment.temperature or 0,  # FIX: hardcoded
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                words=[
                    TranscriptionWord(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                    )
                    for word in segment.words
                ]
                if segment.words is not None
                else None,
            )


# https://platform.openai.com/docs/api-reference/audio/json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10924
class CreateTranscriptionResponseJson(BaseModel):
    text: str
    # NOTE: there's also a `logprobs` field it's only supported by non-whisper models, so we don't include it here (we can't `faster-whisper` doesn't provide it)
    # TODO: add `usage` field: https://platform.openai.com/docs/api-reference/audio/json-object#audio/json-object-usage

    @classmethod
    def from_segments(cls, segments: list[TranscriptionSegment]) -> "CreateTranscriptionResponseJson":
        return cls(text=segments_to_text(segments))


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11007
class CreateTranscriptionResponseVerboseJson(BaseModel):
    # NOTE: there's also a `logprobs` field it's only supported by non-whisper models, so we don't include it here (we can't `faster-whisper` doesn't provide it)
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[TranscriptionWord] | None
    segments: list[TranscriptionSegment]

    @classmethod
    def from_segment(
        cls, segment: TranscriptionSegment, transcription_info: faster_whisper.transcribe.TranscriptionInfo
    ) -> "CreateTranscriptionResponseVerboseJson":
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=segment.words if transcription_info.transcription_options.word_timestamps else None,
            segments=[segment],
        )

    @classmethod
    def from_segments(
        cls, segments: list[TranscriptionSegment], transcription_info: faster_whisper.transcribe.TranscriptionInfo
    ) -> "CreateTranscriptionResponseVerboseJson":
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=segments_to_text(segments),
            segments=segments,
            words=TranscriptionWord.from_segments(segments)
            if transcription_info.transcription_options.word_timestamps
            else None,
        )


ModelTask = Literal["automatic-speech-recognition", "text-to-speech"]  # TODO: add "voice-activity-detection"


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11146
class Model(BaseModel):
    """There may be additional fields in the response that are specific to the model type."""

    id: str
    """The model identifier, which can be referenced in the API endpoints."""
    created: int = 0
    """The Unix timestamp (in seconds) when the model was created."""
    object: Literal["model"] = "model"
    """The object type, which is always "model"."""
    owned_by: str
    """The organization that owns the model."""
    language: list[str] | None = None
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""

    task: ModelTask  # TODO: make a list?

    model_config = ConfigDict(extra="allow")


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8730
class ListModelsResponse(BaseModel):
    data: list[Model]
    object: Literal["list"] = "list"


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
TimestampGranularities = list[Literal["segment", "word"]]


DEFAULT_TIMESTAMP_GRANULARITIES: TimestampGranularities = ["segment"]
TIMESTAMP_GRANULARITIES_COMBINATIONS: list[TimestampGranularities] = [
    [],  # should be treated as ["segment"]. https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities
    ["segment"],
    ["word"],
    ["word", "segment"],
    ["segment", "word"],  # same as ["word", "segment"] but order is different
]
