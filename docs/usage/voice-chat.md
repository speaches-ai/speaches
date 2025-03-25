!!! note

    Before proceeding, you should be familiar with [OpenAI Audio Generation Guide](https://platform.openai.com/docs/guides/audio). The guide explains how the API works and provides examples on how to use. Unless stated otherwise in [limitations](#limitations) if a feature is supported by OpenAI, it should be supported by this project as well.

## Prerequisites

Start `speaches` with the following environmental variables:

- `CHAT_COMPLETION_BASE_URL` to the base URL of an OpenAI API compatible endpoint | [Config](../configuration.md#speaches.config.Config.chat_completion_base_url)
- `CHAT_COMPLETION_API_KEY` if the API you are using requires authentication | [Config](../configuration.md#speaches.config.Config.chat_completion_api_key)

Ollama example:

```bash
export CHAT_COMPLETION_BASE_URL=http://localhost:11434/v1
```

OpenAI example:

```bash
export CHAT_COMPLETION_BASE_URL=https://api.openai.com/v1
export CHAT_COMPLETION_API_KEY=sk-xxx
```

Download a TTS model and a STT model. See [Text-to-Speech](./text-to-speech.md) and [Speech-to-Text](./speech-to-text.md) for more details.

## How does it work?

`speaches` acts as a proxy between the client and the OpenAI compatible API. It receives the user's input audio message, transcribes it, and replaces the input audio with a transcription that a regular LLM can understand. The request is then proxied to the provided endpoint, the response is transcribed back to audio, and returned to the client. Below is the flow of the process for non-streaming text + audio in → text + audio out:

- Receive the `messages` containing audio inputs

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this recording?"
      },
      {
        "type": "input_audio",
        "input_audio": {
          "data": "<bytes omitted>",
          "format": "wav"
        }
      }
    ]
  }
]
```

- Transcribe the audio (`POST /v1/audio/transcriptions`) and replace the input audio with the transcription

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this recording?"
      },
      {
        "type": "text",
        "text": "Hello World!"
      }
    ]
  }
]
```

- Proxy the request to the OpenAI compatible API specified in the environmental variables and receive the response

```json
[
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "The recording says Hello World!."
      }
    ]
  }
]
```

- Transcribe the response and replace the text with the audio (`POST /v1/audio/speech`). The audio is then returned to the client

```json
[
  {
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "refusal": null,
      "audio": {
        "id": "audio_abc123",
        "expires_at": 1729018505,
        "data": "<bytes omitted>",
        "transcript": "The recording says Hello World!."
      }
    },
    "finish_reason": "stop"
  }
]
```

## Customization

The chat completion endpoint exposes additional parameters for customization. The following parameters are available:

- `transcription_model`: The model to use for transcribing the audio.
- `speech_model`: The model to use for generating the audio.

When using OpenAI's Python SDK the parameters can be set using `extra_body` parameter. For example:

```python
openai_client.chat.completions.create(
        model="gpt-4o-mini",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        stream=False,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": "<bytes ommitted>", "format": "wav"}},
                ],
            },
        ],
        extra_body={"transcription_model": "Systran/faster-whisper-tiny.en", "speech_model": "hexgrad/Kokoro-82M"}
    )
```

## Limitations

- User's input audio message are not cached. That means the user's input audio message will be transcribed each time it sent. This can be a performance issue when doing long multi-turn conversations.
- Multiple choices (`n` > 1) are not supported

This features utilizes [Text-to-Speech](./text-to-speech.md) and [Speech-to-Text](./speech-to-text.md) features. Therefore, the limitations of those features apply here as well.
