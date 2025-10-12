import asyncio
from collections.abc import Generator, Iterable
import logging
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from faster_whisper.transcribe import BatchedInferencePipeline, TranscriptionInfo
from faster_whisper.vad import get_speech_timestamps
import numpy as np

from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
)
from speaches.dependencies import (
    AudioFileDependency,
    ConfigDependency,
    ExecutorRegistryDependency,
)
from speaches.executors.parakeet import ParakeetModelManager
from speaches.executors.whisper import WhisperModelManager
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import segments_to_srt, segments_to_text, segments_to_vtt

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]
RESPONSE_FORMATS = ("text", "json", "verbose_json", "srt", "vtt")

# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"


def segments_to_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    segments = list(segments)
    match response_format:
        case "text":
            return Response(segments_to_text(segments), media_type="text/plain")
        case "json":
            return Response(
                CreateTranscriptionResponseJson.from_segments(segments).model_dump_json(),
                media_type="application/json",
            )
        case "verbose_json":
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(segments, transcription_info).model_dump_json(),
                media_type="application/json",
            )
        case "vtt":
            return Response(
                "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), media_type="text/vtt"
            )
        case "srt":
            return Response(
                "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), media_type="text/plain"
            )


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> StreamingResponse:
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == "text":
                data = segment.text
            elif response_format == "json":
                data = CreateTranscriptionResponseJson.from_segments([segment]).model_dump_json()
            elif response_format == "verbose_json":
                data = CreateTranscriptionResponseVerboseJson.from_segment(
                    segment, transcription_info
                ).model_dump_json()
            elif response_format == "vtt":
                data = segments_to_vtt(segment, i)
            elif response_format == "srt":
                data = segments_to_srt(segment, i)
            yield format_as_sse(data)

    return StreamingResponse(segment_responses(), media_type="text/event-stream")


@router.post(
    "/v1/audio/translations",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def translate_file(
    config: ConfigDependency,
    executor_registry: ExecutorRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
    vad_filter: Annotated[bool | None, Form()] = None,
) -> Response | StreamingResponse:
    # Use config default if vad_filter not explicitly provided
    effective_vad_filter = vad_filter if vad_filter is not None else config._unstable_vad_filter  # noqa: SLF001

    # Translation is only supported by Whisper
    whisper_executor = executor_registry.transcription[0]  # Whisper is first
    with whisper_executor.model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task="translate",
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=effective_vad_filter,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)

        if stream:
            return segments_to_streaming_response(segments, transcription_info, response_format)
        else:
            return segments_to_response(segments, transcription_info, response_format)


# HACK: Since Form() doesn't support `alias`, we need to use a workaround.
async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities  # type: ignore[return-value]


# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def transcribe_file(  # noqa: C901, PLR0912
    config: ConfigDependency,
    executor_registry: ExecutorRegistryDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool | None, Form()] = None,
    without_timestamps: Annotated[bool, Form()] = True,
) -> Response | StreamingResponse:
    # Use config default if vad_filter not explicitly provided
    effective_vad_filter = vad_filter if vad_filter is not None else config._unstable_vad_filter  # noqa: SLF001

    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."
        )

    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.transcription)

    if isinstance(executor.model_manager, WhisperModelManager):
        with executor.model_manager.load_model(model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
            segments, transcription_info = whisper_model.transcribe(
                audio,
                task="transcribe",
                language=language,
                initial_prompt=prompt,
                word_timestamps="word" in timestamp_granularities,
                temperature=temperature,
                vad_filter=effective_vad_filter,
                hotwords=hotwords,
                without_timestamps=without_timestamps,
            )
            segments = TranscriptionSegment.from_faster_whisper_segments(segments)

            if stream:
                return segments_to_streaming_response(segments, transcription_info, response_format)
            else:
                return segments_to_response(segments, transcription_info, response_format)
    elif isinstance(executor.model_manager, ParakeetModelManager):
        if stream:
            raise HTTPException(status_code=500, detail=f"Model '{model}' does not support streaming yet.")
        if response_format not in ("text", "json"):
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model}' only supports 'text' and 'json' response formats for now.",
            )
        with executor.model_manager.load_model(model) as parakeet:
            # TODO: issue warnings when client specifies unsupported parameters like `prompt`, `temperature`, `hotwords`, etc.

            # Somewhat hacky work around for transcribing large audio files by splitting them into smaller chunks using VAD. May not work well for all use cases. Bug: https://github.com/istupakov/onnx-asr/issues/18

            # Apply VAD to split audio into speech segments
            speech_timestamps = get_speech_timestamps(audio, sampling_rate=16000)

            if not speech_timestamps:
                # No speech detected, return empty transcription
                match response_format:
                    case "text":
                        return Response("", media_type="text/plain")
                    case "json":
                        return Response(
                            CreateTranscriptionResponseJson(text="").model_dump_json(),
                            media_type="application/json",
                        )

            # Extract speech segments from audio
            waveforms = []
            waveforms_len = []
            for timestamp in speech_timestamps:
                start = timestamp["start"]
                end = timestamp["end"]
                segment = audio[start:end]
                waveforms.append(segment)
                waveforms_len.append(len(segment))

            # Prepare batch arrays
            max_len = max(waveforms_len)
            waveforms_batch = np.zeros((len(waveforms), max_len), dtype=np.float32)
            for i, waveform in enumerate(waveforms):
                waveforms_batch[i, : len(waveform)] = waveform
            waveforms_len_batch = np.array(waveforms_len, dtype=np.int64)

            # print all segment sizes in descending order

            logger.info(f"Transcribing {len(waveforms)} segments with lengths: {sorted(waveforms_len, reverse=True)}")
            # Process all segments in batch
            results = list(
                parakeet.with_timestamps().asr.recognize_batch(waveforms_batch, waveforms_len_batch, language=language)
            )

            # Combine results
            combined_text = " ".join(result.text for result in results)

            match response_format:
                case "text":
                    return Response(combined_text, media_type="text/plain")
                case "json":
                    return Response(
                        CreateTranscriptionResponseJson(
                            text=combined_text,
                        ).model_dump_json(),
                        media_type="application/json",
                    )

    raise HTTPException(
        status_code=500,
        detail=f"Executor for model '{model}' exists but is not properly configured. This is a bug.",
    )
