import gradio as gr

from speaches.config import Config
from speaches.ui.tabs.audio_chat import create_audio_chat_tab
from speaches.ui.tabs.stt import create_stt_tab
from speaches.ui.tabs.tts import create_tts_tab

# NOTE: `gr.Request` seems to be passed in as the last positional (not keyword) argument


def create_gradio_demo(config: Config) -> gr.Blocks:
    with gr.Blocks(
        title="Speaches Playground",
        head="""
<script>
    const API_KEY_STORAGE_KEY = 'speaches_api_key';

    function saveApiKey(apiKey) {
        try {
            localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
        } catch (e) {
            console.error('Failed to save API key to localStorage:', e);
        }
    }

    function loadApiKey() {
        try {
            return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
        } catch (e) {
            console.error('Failed to load API key from localStorage:', e);
            return '';
        }
    }
</script>
""",
    ) as demo:
        gr.Markdown("# Speaches Playground")
        gr.Markdown(
            "### Consider supporting the project by starring the [speaches-ai/speaches repository on GitHub](https://github.com/speaches-ai/speaches)."
        )
        gr.Markdown("### Documentation Website: https://speaches.ai")
        gr.Markdown(
            "### For additional details regarding the parameters, see the [API Documentation](https://speaches.ai/api)"
        )

        with gr.Row():
            with gr.Column(scale=9):
                api_key_input = gr.Textbox(
                    label="API Key (Optional)",
                    placeholder="Enter your API key if authentication is enabled",
                    type="password",
                    value="",
                    info="Leave empty if no API key is configured on the server. Your key is stored in browser localStorage. Note: You may need to refresh the page after entering your API key for it to take effect.",
                    elem_id="api_key_input",
                )
            with gr.Column(scale=1, min_width=120):
                show_api_key_btn = gr.Button("Show Key", size="sm", elem_id="show_api_key_btn")

        # Add JavaScript for persistence and visibility toggle
        demo.load(
            None,
            None,
            api_key_input,
            js="""() => { return loadApiKey(); }""",
        )

        api_key_input.change(
            None,
            api_key_input,
            None,
            js="""(apiKey) => { saveApiKey(apiKey); }""",
        )

        show_api_key_btn.click(
            None,
            None,
            None,
            js="""() => {
                const input = document.querySelector('#api_key_input input');
                const btn = document.querySelector('#show_api_key_btn');
                if (input && btn) {
                    if (input.type === 'password') {
                        input.type = 'text';
                        btn.textContent = 'Hide Key';
                    } else {
                        input.type = 'password';
                        btn.textContent = 'Show Key';
                    }
                }
            }""",
        )

        create_audio_chat_tab(config, api_key_input)
        create_stt_tab(config, api_key_input)
        create_tts_tab(config, api_key_input)

    return demo
