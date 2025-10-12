from collections.abc import AsyncGenerator
import logging
from pathlib import Path

import gradio as gr
import httpx
from httpx_sse import aconnect_sse

from speaches.config import Config
from speaches.routers.stt import RESPONSE_FORMATS, ResponseFormat
from speaches.ui.utils import http_client_from_gradio_req, openai_client_from_gradio_req
from speaches.utils import APIProxyError, format_api_proxy_error

logger = logging.getLogger(__name__)

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"

DEFAULT_RESPONSE_FORMAT: ResponseFormat = "text"


def create_stt_tab(config: Config) -> None:  # noqa: C901, PLR0915
    async def update_whisper_model_dropdown(request: gr.Request) -> gr.Dropdown:
        openai_client = openai_client_from_gradio_req(request, config)
        models = (await openai_client.models.list()).data
        model_ids: list[str] = [model.id for model in models]
        recommended_models = {model for model in model_ids if model.startswith("Systran")}
        other_models = [model for model in model_ids if model not in recommended_models]
        model_ids = list(recommended_models) + other_models
        return gr.Dropdown(choices=model_ids, label="Model")

    async def audio_task(
        http_client: httpx.AsyncClient,
        file_path: str,
        endpoint: str,
        response_format: ResponseFormat,
        temperature: float,
        model: str,
    ) -> str:
        try:
            if not file_path:
                msg = "No audio file provided in audio_task (stt.py). Please record or upload audio."
                raise APIProxyError(msg, suggestions=["Please record or upload an audio file."])
            with Path(file_path).open("rb") as file:  # noqa: ASYNC230
                response = await http_client.post(
                    endpoint,
                    files={"file": file},
                    data={
                        "model": model,
                        "response_format": response_format,
                        "temperature": temperature,
                    },
                )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.exception("STT audio_task error")
            if not isinstance(e, APIProxyError):
                e = APIProxyError(str(e))
            return format_api_proxy_error(e, context="audio_task")

    async def streaming_audio_task(
        http_client: httpx.AsyncClient,
        file_path: str,
        endpoint: str,
        response_format: ResponseFormat,
        temperature: float,
        model: str,
    ) -> AsyncGenerator[str, None]:
        try:
            with Path(file_path).open("rb") as file:  # noqa: ASYNC230
                kwargs = {
                    "files": {"file": file},
                    "data": {
                        "response_format": response_format,
                        "temperature": temperature,
                        "model": model,
                        "stream": True,
                    },
                }
                async with aconnect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                    async for event in event_source.aiter_sse():
                        yield event.data
        except Exception as e:
            logger.exception("STT streaming error")
            if not isinstance(e, APIProxyError):
                e = APIProxyError(str(e))
            yield format_api_proxy_error(e, context="streaming_audio_task")

    async def whisper_handler(
        file_path: str,
        model: str,
        task: str,
        response_format: ResponseFormat,
        temperature: float,
        stream: bool,
        request: gr.Request,
    ) -> AsyncGenerator[str, None]:
        try:
            if not file_path:
                msg = "No audio file provided in whisper_handler (stt.py). Please record or upload audio."
                raise APIProxyError(msg, suggestions=["Please record or upload an audio file."])
            http_client = http_client_from_gradio_req(request, config)
            endpoint = TRANSCRIPTION_ENDPOINT if task == "transcribe" else TRANSLATION_ENDPOINT

            if stream:
                previous_transcription = ""
                async for transcription in streaming_audio_task(
                    http_client, file_path, endpoint, response_format, temperature, model
                ):
                    previous_transcription += transcription
                    yield previous_transcription
            else:
                result = await audio_task(http_client, file_path, endpoint, response_format, temperature, model)
                yield result
        except Exception as e:
            logger.exception("STT handler error")
            if not isinstance(e, APIProxyError):
                e = APIProxyError(str(e))
            yield format_api_proxy_error(e, context="whisper_handler")

    with gr.Tab(label="Speech-to-Text") as tab:
        audio = gr.Audio(type="filepath")
        whisper_model_dropdown = gr.Dropdown(choices=[], label="Model")
        task_dropdown = gr.Dropdown(choices=["transcribe", "translate"], label="Task", value="transcribe")
        response_format = gr.Dropdown(choices=RESPONSE_FORMATS, label="Response Format", value=DEFAULT_RESPONSE_FORMAT)
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
        stream_checkbox = gr.Checkbox(label="Stream", value=True)
        button = gr.Button("Generate")

        output = gr.Textbox()

        # NOTE: the inputs order must match the `whisper_handler` signature
        button.click(
            whisper_handler,
            [audio, whisper_model_dropdown, task_dropdown, response_format, temperature_slider, stream_checkbox],
            output,
        )

        tab.select(update_whisper_model_dropdown, inputs=None, outputs=whisper_model_dropdown)
