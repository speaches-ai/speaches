import gradio as gr

from speaches.config import Config
from speaches.ui.tabs.audio_chat import create_audio_chat_tab
from speaches.ui.tabs.stt import create_stt_tab
from speaches.ui.tabs.tts import create_tts_tab

# NOTE: `gr.Request` seems to be passed in as the last positional (not keyword) argument


def create_gradio_demo(config: Config) -> gr.Blocks:
    with gr.Blocks(title="Speaches Playground") as demo:
        gr.Markdown("# Speaches Playground")
        gr.Markdown(
            "### Consider supporting the project by starring the [speaches-ai/speaches repository on GitHub](https://github.com/speaches-ai/speaches)."
        )
        gr.Markdown("### Documentation Website: https://speaches.ai")
        gr.Markdown(
            "### For additional details regarding the parameters, see the [API Documentation](https://speaches.ai/api)"
        )

        create_audio_chat_tab(config)
        create_stt_tab(config)
        create_tts_tab(config)

    return demo
