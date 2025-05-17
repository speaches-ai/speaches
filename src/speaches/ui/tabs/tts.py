# ruff: noqa: PLR0915, C901
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote_plus

import gradio as gr
import httpx

from speaches import kokoro_utils
from speaches.api_types import Voice
from speaches.config import Config
from speaches.piper_utils import PiperDownloadOptions
from speaches.routers.speech import (
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SUPPORTED_RESPONSE_FORMATS,
)
from speaches.ui.utils import http_client_from_gradio_req, openai_client_from_gradio_req

DEFAULT_TEXT = "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky."


def create_tts_tab(config: Config) -> None:
    async def update_model_dropdown(request: gr.Request) -> gr.Dropdown:
        openai_client = openai_client_from_gradio_req(request, config)
        models = (await openai_client.models.list(extra_query={"task": "text-to-speech"})).data
        model_ids: list[str] = [model.id for model in models]
        default = None
        if "hexgrad/Kokoro-82M" in model_ids:
            default = "hexgrad/Kokoro-82M"
        return gr.Dropdown(choices=model_ids, label="Model", value=default)

    async def update_voices_and_language_dropdown(model_id: str | None, request: gr.Request) -> dict:
        params = httpx.QueryParams({"model_id": model_id}) if model_id is not None else None
        http_client = http_client_from_gradio_req(request, config)
        res = (
            await http_client.get("/v1/audio/speech/voices", params=params, follow_redirects=True)
        ).raise_for_status()
        voice_ids = [Voice.model_validate(x).voice_id for x in res.json()]
        if not voice_ids:
            return {
                voice_dropdown: gr.update(choices=[], value=None),
                language_dropdown: gr.update(visible=False),
                model_present: False,
                execute_group: gr.update(visible=False),
            }
        return {
            voice_dropdown: gr.update(choices=voice_ids, value=voice_ids[0]),
            language_dropdown: gr.update(visible=model_id == "hexgrad/Kokoro-82M"),
            model_present: True,
            execute_group: gr.update(visible=True),
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

    async def download_model_and_voices(
        model_id: str, lang_voices: str | None, lang_custom: str | None, request: gr.Request
    ) -> None:
        http_client = http_client_from_gradio_req(request, config)
        params = {}
        if model_id == "rhasspy/piper-voices":
            match lang_voices:
                case PiperDownloadOptions.ENGLISH_ONLY:
                    params["allow_patterns"] = ["en/**/*"]
                case PiperDownloadOptions.US_ENGLISH_ONLY:
                    params["allow_patterns"] = ["en/en_US/**/*"]
                case PiperDownloadOptions.US_ENGLISH_AMY:
                    params["allow_patterns"] = ["en/en_US/amy/**/*"]
                case PiperDownloadOptions.CUSTOM:
                    params["allow_patterns"] = [lang_custom]
        model_id_enc = quote_plus(model_id)
        res = await http_client.post(url=f"/api/pull/{model_id_enc}", json=params, follow_redirects=True)
        res.raise_for_status()

        return {
            stt_model_dropdown: gr.update(value=model_id),
        }

    async def toggle_download_section(model_present: bool | None, model_id: str) -> None:
        if model_present:
            return {
                download_section: gr.update(visible=False),
            }
        print("model_id", model_id)
        if model_id == "rhasspy/piper-voices":
            return {
                download_section: gr.update(visible=True),
                lang_voices_section: gr.update(visible=True),
            }
        return {
            download_section: gr.update(visible=True),
            lang_voices_section: gr.update(visible=False),
        }

    with gr.Tab(label="Text-to-Speech") as tab:
        text = gr.Textbox(label="Input Text", value=DEFAULT_TEXT, lines=3)
        stt_model_dropdown = gr.Dropdown(choices=[], label="Model")
        model_present = gr.State(value=None)

        with gr.Column(visible=False) as download_section:
            with gr.Column(visible=False) as lang_voices_section:
                lang_voices = gr.Radio(
                    choices=[o.value for o in list(PiperDownloadOptions)],
                    label="Language/Voices",
                    info="Downloading all voices may take a while. We recommend downloading only the voices you need",
                    value=PiperDownloadOptions.US_ENGLISH_AMY,
                    interactive=True,
                )
                lang_custom = gr.Textbox(
                    label="Custom Language Pattern",
                    info="Use a Glob pattern to specify the voices you want to download from HuggingFace.",
                    placeholder="en/en_US/amy/**/*",
                    visible=False,
                    interactive=True,
                )
                lang_voices.change(
                    lambda x: gr.update(visible=x == "Custom"),
                    inputs=[lang_voices],
                    outputs=[lang_custom],
                )
            download_button = gr.Button("Download Model and Voices")
        model_present.change(
            toggle_download_section,
            inputs=[model_present, stt_model_dropdown],
            outputs=[download_section, lang_voices_section],
        )

        with gr.Column(visible=False) as execute_group:
            voice_dropdown = gr.Dropdown(choices=[], label="Voice")
            language_dropdown = gr.Dropdown(
                choices=kokoro_utils.LANGUAGES, label="Language", value="en-us", visible=True
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

        stt_model_dropdown.change(
            update_voices_and_language_dropdown,
            inputs=[stt_model_dropdown],
            outputs=[voice_dropdown, language_dropdown, model_present, execute_group],
        ).then(
            toggle_download_section,
            inputs=[model_present, stt_model_dropdown],
            outputs=[download_section, lang_voices_section],
        )

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

        download_button.click(
            download_model_and_voices,
            inputs=[stt_model_dropdown, lang_voices, lang_custom],
            outputs=[stt_model_dropdown],
        ).then(
            update_voices_and_language_dropdown,
            inputs=[stt_model_dropdown],
            outputs=[voice_dropdown, language_dropdown, model_present, execute_group],
        )

        tab.select(update_model_dropdown, inputs=None, outputs=stt_model_dropdown)
        tab.select(
            update_voices_and_language_dropdown,
            inputs=[stt_model_dropdown],
            outputs=[voice_dropdown, language_dropdown, model_present, execute_group],
        )
