# Speaches

> [!NOTE]
> This project was previously named `faster-whisper-server`. I've decided to change the name from `faster-whisper-server`, as the project has evolved to support more than just ASR.

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. Speach-to-Text is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for Text-to-Speech [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) are used. This project aims to be Ollama, but for TTS/STT models.

See the documentation for installation instructions and usage: [speaches.ai](https://speaches.ai/)

## Quick Start

Get a fully functional `speaches` server running in a few commands.

### 1. Installation

Install the `speaches` command-line tool and all its dependencies using `uv`. The default installation includes the web server and UI.

```bash

git clone https://github.com/speaches-ai/speaches.git
cd speaches
uv venv
source .venv/bin/activate
uv sync --all-extras --upgrade
uv tool install .

# Downloading a Text To Speech (TTS) model:
uvx speaches model download speaches-ai/Kokoro-82M-v1.0-ONNX

# Downloading a Speech To Text (STT) model:
uvx speaches model download Systran/faster-distil-whisper-small.en

# run the speaches server then open http://localhost:8000 in your web browser to try speaches
speaches serve --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000 in your web browser.

The server will start, and the console will display the correct URL (e.g., `http://localhost:8000`) to access the Gradio web UI. Once the server is running, you can open a new terminal to use client commands like `speaches model ls`.

## Features:

- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
- Streaming support (transcription is sent via SSE as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
- Text-to-Speech via `kokoro`(Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
- GPU and CPU support.
- [Deployable via Docker Compose / Docker](https://speaches.ai/installation/)
- [Highly configurable](https://speaches.ai/configuration/)
- [Realtime API](https://speaches.ai/usage/realtime-api/)

Please create an issue if you find a bug, have a question, or a feature suggestion.

## Demos

### Realtime API

https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc

(Excuse the breathing lol. Didn't have enough time to record a better demo)

### Streaming Transcription

TODO

### Speech Generation

https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b
