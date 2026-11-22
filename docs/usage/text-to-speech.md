!!! note

    Before proceeding, you should be familiar with the [OpenAI Text-to-Speech](https://platform.openai.com/docs/guides/text-to-speech) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## Download a TTS model

```bash
export SPEACHES_BASE_URL="http://localhost:8000"

# Listing all available TTS models
uvx speaches-cli registry ls --task text-to-speech | jq '.data | [].id'

# Downloading a TTS model
uvx speaches-cli model download speaches-ai/Kokoro-82M-v1.0-ONNX

# Check that the model has been installed
uvx speaches-cli model ls --task text-to-speech | jq '.data | map(select(.id == "speaches-ai/Kokoro-82M-v1.0-ONNX"))'
```

## Default output sample rate

Set `TTS_DEFAULT_SAMPLE_RATE=44100` in the server environment to request
44.1 kHz audio from `/v1/audio/speech` when the client omits `sample_rate`.
The setting accepts 8,000 to 48,000 Hz, subject to the selected audio format's
supported rates. A `sample_rate` in the request takes precedence. When both
are omitted, audio uses the model's native sample rate. This setting applies
to `stream_format="audio"`.

## Usage

### Curl

```bash
export SPEACHES_BASE_URL="http://localhost:8000"
export SPEECH_MODEL_ID="speaches-ai/Kokoro-82M-v1.0-ONNX"

# List available voices
# TODO

export VOICE_ID="af_heart"

# Generate speech
curl "$SPEACHES_BASE_URL/v1/audio/speech" -s -H "Content-Type: application/json" \
  --output audio.mp3 \
  --data @- << EOF
{
  "input": "Hello World!",
  "model": "$SPEECH_MODEL_ID",
  "voice": "$VOICE_ID"
}
EOF

curl "$SPEACHES_BASE_URL/v1/audio/speech" -s -H "Content-Type: application/json" \
  --output audio.wav \
  --data @- << EOF
{
  "input": "Hello World!",
  "model": "$SPEECH_MODEL_ID",
  "voice": "$VOICE_ID",
  "response_format": "wav"
}
EOF

curl "$SPEACHES_BASE_URL/v1/audio/speech" -s -H "Content-Type: application/json" \
  --output audio.mp3 \
  --data @- << EOF
{
  "input": "Hello World!",
  "model": "$SPEECH_MODEL_ID",
  "voice": "$VOICE_ID",
  "speed": 2.0
}
EOF
```

### Python

=== "httpx"

    ```python
    from pathlib import Path

    import httpx

    client = httpx.Client(base_url="http://localhost:8000/")
    model_id = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    voice_id = "af_heart"
    res = client.post(
        "v1/audio/speech",
        json={
            "model": model_id,
            "voice": voice_id,
            "input": "Hello, world!",
            "response_format": "mp3",
            "speed": 1,
        },
    ).raise_for_status()
    with Path("output.mp3").open("wb") as f:
        f.write(res.read())
    ```

### OpenAI SDKs

!!! note

    Although this project doesn't require an API key, all OpenAI SDKs require an API key. Therefore, you will need to set it to a non-empty value. Additionally, you will need to overwrite the base URL to point to your server.

    This can be done by setting the `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables or by passing them as arguments to the SDK.

=== "Python"

    ```python
    from pathlib import Path

    from openai import OpenAI

    openai = OpenAI(base_url="http://localhost:8000/v1", api_key="cant-be-empty")
    model_id = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    voice_id = "af_heart"
    res = openai.audio.speech.create(
        model=model_id,
        voice=voice_id,
        input="Hello, world!",
        response_format="mp3",
        speed=1,
    )
    with Path("output.mp3").open("wb") as f:
        f.write(res.response.read())
    ```

=== "Other"

    See [OpenAI libraries](https://platform.openai.com/docs/libraries)

## Limitations

- `response_format`: `opus` and `aac` are not supported
