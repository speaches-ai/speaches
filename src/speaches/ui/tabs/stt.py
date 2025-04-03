from collections.abc import AsyncGenerator
from pathlib import Path

import gradio as gr
import httpx
from httpx_sse import aconnect_sse

from speaches.config import Config
from speaches.ui.utils import http_client_from_gradio_req, openai_client_from_gradio_req

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"


def create_stt_tab(config: Config) -> None:
    async def update_whisper_model_dropdown(request: gr.Request) -> gr.Dropdown:
        openai_client = openai_client_from_gradio_req(request, config)
        models = (await openai_client.models.list()).data
        model_ids: list[str] = [model.id for model in models]
        recommended_models = {model for model in model_ids if model.startswith("Systran")}
        other_models = [model for model in model_ids if model not in recommended_models]
        model_ids = list(recommended_models) + other_models
        return gr.Dropdown(choices=model_ids, label="Model")

    async def audio_task(
        http_client: httpx.AsyncClient, file_path: str, endpoint: str, temperature: float, model: str
    ) -> str:
        with Path(file_path).open("rb") as file:  # noqa: ASYNC230
            response = await http_client.post(
                endpoint,
                files={"file": file},
                data={
                    "model": model,
                    "response_format": "text",
                    "temperature": temperature,
                },
                follow_redirects=True
            )

        response.raise_for_status()
        return response.text

    async def streaming_audio_task(
        http_client: httpx.AsyncClient, file_path: str, endpoint: str, temperature: float, model: str
    ) -> AsyncGenerator[str, None]:
        with Path(file_path).open("rb") as file:  # noqa: ASYNC230
            kwargs = {
                "files": {"file": file},
                "data": {
                    "response_format": "text",
                    "temperature": temperature,
                    "model": model,
                    "stream": True,
                },
            }
            async with aconnect_sse(http_client, "POST", endpoint, follow_redirects=True, **kwargs) as event_source:
                async for event in event_source.aiter_sse():
                    yield event.data

    async def whisper_handler(
        file_path: str, model: str, task: str, temperature: float, stream: bool, request: gr.Request
    ) -> AsyncGenerator[str, None]:
        http_client = http_client_from_gradio_req(request, config)
        endpoint = TRANSCRIPTION_ENDPOINT if task == "transcribe" else TRANSLATION_ENDPOINT

        if stream:
            previous_transcription = ""
            async for transcription in streaming_audio_task(http_client, file_path, endpoint, temperature, model):
                previous_transcription += transcription
                yield previous_transcription
        else:
            yield await audio_task(http_client, file_path, endpoint, temperature, model)

    with gr.Tab(label="Speech-to-Text") as tab:
        audio = gr.Audio(type="filepath")
        whisper_model_dropdown = gr.Dropdown(choices=[], label="Model")
        task_dropdown = gr.Dropdown(choices=["transcribe", "translate"], label="Task", value="transcribe")
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
        stream_checkbox = gr.Checkbox(label="Stream", value=True)
        button = gr.Button("Generate")

        output = gr.Textbox()

        # NOTE: the inputs order must match the `whisper_handler` signature
        button.click(
            whisper_handler,
            [audio, whisper_model_dropdown, task_dropdown, temperature_slider, stream_checkbox],
            output,
        )

        tab.select(update_whisper_model_dropdown, inputs=None, outputs=whisper_model_dropdown)
