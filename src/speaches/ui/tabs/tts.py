from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import gradio as gr
import httpx

from speaches import kokoro_utils
from speaches.api_types import Voice
from speaches.config import Config
from speaches.routers.speech import (
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SUPPORTED_RESPONSE_FORMATS,
)
from speaches.ui.utils import http_client_from_gradio_req, openai_client_from_gradio_req

DEFAULT_TEXT = "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky."


def create_tts_tab(config: Config) -> None:
    async def update_voices_and_language_dropdown(model_id: str | None, request: gr.Request) -> dict:
        params = httpx.QueryParams({"model_id": model_id})
        http_client = http_client_from_gradio_req(request, config)
        res = (await http_client.get("/v1/audio/speech/voices", params=params)).raise_for_status()
        voice_ids = [Voice.model_validate(x).voice_id for x in res.json()]
        return {
            voice_dropdown: gr.update(choices=voice_ids, value=voice_ids[0]),
            language_dropdown: gr.update(visible=model_id == "hexgrad/Kokoro-82M"),
        }

    async def handle_audio_speech(
        text: str,
        model: str,
        voice: str,
        language: str | None,
        response_format: str,
        speed: float,
        sample_rate: int | None,
        request: gr.Request,
    ) -> Path:
        openai_client = openai_client_from_gradio_req(request, config)
        res = await openai_client.audio.speech.create(
            input=text,
            model=model,
            voice=voice,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            speed=speed,
            extra_body={"language": language, "sample_rate": sample_rate},
        )
        audio_bytes = res.response.read()
        with NamedTemporaryFile(suffix=f".{response_format}", delete=False) as file:
            file.write(audio_bytes)
            file_path = Path(file.name)
        return file_path

    with gr.Tab(label="Text-to-Speech") as tab:
        model_dropdown_choices = ["hexgrad/Kokoro-82M", "rhasspy/piper-voices"]
        if platform.machine() != "x86_64":
            model_dropdown_choices.remove("rhasspy/piper-voices")
            gr.Textbox("Speech generation using `rhasspy/piper-voices` model is only supported on x86_64 machines.")

        text = gr.Textbox(label="Input Text", value=DEFAULT_TEXT, lines=3)
        stt_model_dropdown = gr.Dropdown(choices=model_dropdown_choices, label="Model", value=model_dropdown_choices[0])
        voice_dropdown = gr.Dropdown(choices=[], label="Voice")
        language_dropdown = gr.Dropdown(choices=kokoro_utils.LANGUAGES, label="Language", value="en-us", visible=True)
        stt_model_dropdown.change(
            update_voices_and_language_dropdown,
            inputs=[stt_model_dropdown],
            outputs=[voice_dropdown, language_dropdown],
        )
        response_fromat_dropdown = gr.Dropdown(
            choices=SUPPORTED_RESPONSE_FORMATS,
            label="Response Format",
            value="wav",
        )
        speed_slider = gr.Slider(minimum=0.25, maximum=4.0, step=0.05, label="Speed", value=1.0)
        sample_rate_slider = gr.Number(
            minimum=MIN_SAMPLE_RATE,
            maximum=MAX_SAMPLE_RATE,
            label="Desired Sample Rate",
            info="""
Setting this will resample the generated audio to the desired sample rate.
You may want to set this if you are going to use 'rhasspy/piper-voices' with voices of different qualities but want to keep the same sample rate.
Default: None (No resampling)
""",
            value=lambda: None,
        )
        button = gr.Button("Generate Speech")
        output = gr.Audio(type="filepath")
        button.click(
            handle_audio_speech,
            [
                text,
                stt_model_dropdown,
                voice_dropdown,
                language_dropdown,
                response_fromat_dropdown,
                speed_slider,
                sample_rate_slider,
            ],
            output,
        )

        tab.select(
            update_voices_and_language_dropdown,
            inputs=[stt_model_dropdown],
            outputs=[voice_dropdown, language_dropdown],
        )
