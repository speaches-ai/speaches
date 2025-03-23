!!! note

    Before proceeding, you should be familiar with the [OpenAI Text-to-Speech](https://platform.openai.com/docs/guides/text-to-speech) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## Usage

### Curl

```bash
export SPEACHES_BASE_URL="http://localhost:8000"


# Listing all available TTS models
curl -s "$SPEACHES_BASE_URL/v1/registry?task=text-to-speech"

# Downloading a TTS model
export MODEL_ID="speaches-ai/Kokoro-82M-v1.0-ONNX"
curl -s -X POST "$SPEACHES_BASE_URL/v1/models" \
  -H "Content-Type: application/json" \
  --data "{\"id\": \"$MODEL_ID\"}"

# Check that the model has been installed
curl -s "$SPEACHES_BASE_URL/v1/models/$MODEL_ID"

# List available voices
# TODO
export VOICE_ID="af_heart"

# Generate speech from text using the default values (model="hexgrad/Kokoro-82M", voice="af", response_format="mp3", speed=1.0, etc.)
curl $SPEACHES_BASE_URL/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!"}' --output audio.mp3
# Specifying the output format
curl $SPEACHES_BASE_URL/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "response_format": "wav"}' --output audio.wav
# Specifying the audio speed
curl $SPEACHES_BASE_URL/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "speed": 2.0}' --output audio.mp3

# List available (downloaded) voices
curl --silent $SPEACHES_BASE_URL/v1/audio/speech/voices
# List just the voice names
curl --silent $SPEACHES_BASE_URL/v1/audio/speech/voices | jq --raw-output '.[] | .voice_id'
# List just the rhasspy/piper-voices voice names
curl --silent '$SPEACHES_BASE_URL/v1/audio/speech/voices?model_id=rhasspy/piper-voices' | jq --raw-output '.[] | .voice_id'
# List just the hexgrad/Kokoro-82M voice names
curl --silent '$SPEACHES_BASE_URL/v1/audio/speech/voices?model_id=hexgrad/Kokoro-82M' | jq --raw-output '.[] | .voice_id'

# List just the voices in your language (piper)
curl --silent $SPEACHES_BASE_URL/v1/audio/speech/voices | jq --raw-output '.[] | select(.voice | startswith("en")) | .voice_id'

curl $SPEACHES_BASE_URL/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "voice": "af_sky"}' --output audio.mp3
```

### Python

=== "httpx"

    ```python
    from pathlib import Path

    import httpx

    client = httpx.Client(base_url="http://localhost:8000/")
    res = client.post(
        "v1/audio/speech",
        json={
            "model": "hexgrad/Kokoro-82M",
            "voice": "af",
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
    res = openai.audio.speech.create(
        model="hexgrad/Kokoro-82M",
        voice="af",  # pyright: ignore[reportArgumentType]
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
- Maximum audio generation length is 10 seconds for Piper models
