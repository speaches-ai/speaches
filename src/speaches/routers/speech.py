import logging

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from speaches.api_types import (
    DEFAULT_SPEECH_RESPONSE_FORMAT,
    MAX_SPEECH_SAMPLE_RATE,
    MIN_SPEECH_SAMPLE_RATE,
    SpeechResponseFormat,
)
from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.shared.handler_protocol import SpeechRequest
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import strip_emojis, strip_markdown_emphasis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])


class CreateSpeechRequestBody(BaseModel):
    model: ModelId
    input: str
    """The text to generate audio for."""
    voice: str
    response_format: SpeechResponseFormat = DEFAULT_SPEECH_RESPONSE_FORMAT
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
    speed: float = 1.0
    """The speed of the generated audio. 1.0 is the default. Different models have different supported speed ranges."""
    sample_rate: int | None = Field(None, ge=MIN_SPEECH_SAMPLE_RATE, le=MAX_SPEECH_SAMPLE_RATE)
    """Desired sample rate to convert the generated audio to. If not provided, the model's default sample rate will be used."""


# https://platform.openai.com/docs/api-reference/audio/createSpeech
# NOTE: `response_model=None` because `Response | StreamingResponse` are not serializable by Pydantic.
@router.post("/v1/audio/speech", response_model=None)
def synthesize(
    executor_registry: ExecutorRegistryDependency,
    body: CreateSpeechRequestBody,
) -> Response | StreamingResponse:
    model_card_data = get_model_card_data_or_raise(body.model)
    executor = find_executor_for_model_or_raise(body.model, model_card_data, executor_registry.text_to_speech)

    body.input = strip_emojis(body.input)
    body.input = strip_markdown_emphasis(body.input)

    speech_request = SpeechRequest(
        model=body.model,
        voice=body.voice,
        input=body.input,
        response_format=body.response_format,
        instructions=None,
        speed=body.speed,
        stream_format="pcm",
        sample_rate=body.sample_rate,
    )
    # HACK: here we assume that the ValueError is only raised for invalid input data which may not be correct. This is a workaround for avoiding raising `HTTPException` from the executor code
    try:
        res, mime_type = executor.model_manager.handle_speech_request(
            speech_request,
        )
    except ValueError as e:
        logger.exception("Value error during speech synthesis")
        raise HTTPException(status_code=422, detail=str(e)) from e
    if isinstance(res, bytes):
        return Response(res, media_type=mime_type)
    else:
        return StreamingResponse(res, media_type=mime_type)
