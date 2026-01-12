import asyncio
from collections.abc import Generator
import logging
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
import openai.types.audio

from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    TimestampGranularities,
)
from speaches.dependencies import (
    AudioFileDependency,
    ExecutorRegistryDependency,
)
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
    TranslationRequest,
    TranslationResponse,
    VadRequest,
)
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import format_as_sse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]
RESPONSE_FORMATS = ("text", "json", "verbose_json", "srt", "vtt")

# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"

# NOTE: copied from `faster_whisper.transcribe`
DEFAULT_VAD_OPTIONS = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30)


def translation_response_to_http_response(res: TranslationResponse) -> Response:  # noqa: RET503
    if isinstance(res, tuple):
        text, media_type = res
        return Response(content=text, media_type=media_type)
    elif isinstance(res, (openai.types.audio.Translation, openai.types.audio.TranslationVerbose)):
        return Response(content=res.model_dump_json(), media_type="application/json")


@router.post(
    "/v1/audio/translations",
    response_model=str | openai.types.audio.Translation | openai.types.audio.TranslationVerbose,
)
def translate_file(
    executor_registry: ExecutorRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
) -> Response:
    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.translation)

    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    translation_request = TranslationRequest(
        audio=audio,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
    )
    res = executor.model_manager.handle_translation_request(translation_request)
    return translation_response_to_http_response(res)


# HACK: Since Form() doesn't support `alias`, we need to use a workaround.
async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities  # pyright: ignore[reportReturnType]


def transcription_response_to_http_response(
    res: NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent],
) -> Response | StreamingResponse:
    if isinstance(res, tuple):
        text, media_type = res
        return Response(content=text, media_type=media_type)
    elif isinstance(res, (openai.types.audio.Transcription, openai.types.audio.TranscriptionVerbose)):
        return Response(content=res.model_dump_json(), media_type="application/json")
    else:
        return StreamingResponse(
            (format_as_sse(x.model_dump_json()) for x in res),
            media_type="text/event-stream",
        )


# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str | openai.types.audio.Transcription | openai.types.audio.TranscriptionVerbose,
)
def transcribe_file(
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
    # non standard parameters
    hotwords: Annotated[str | None, Form()] = None,
    without_timestamps: Annotated[bool, Form()] = True,
) -> Response | StreamingResponse:
    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."
        )

    transcription_model_card_data = get_model_card_data_or_raise(model)
    transcription_executor = find_executor_for_model_or_raise(
        model, transcription_model_card_data, executor_registry.transcription
    )

    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    transcription_request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        stream=stream,
        hotwords=hotwords,
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
        without_timestamps=without_timestamps,
    )
    res = transcription_executor.model_manager.handle_transcription_request(transcription_request)
    http_res = transcription_response_to_http_response(res)
    return http_res
