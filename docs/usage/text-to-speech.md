!!! note

    Before proceeding, you should be familiar with the [OpenAI Text-to-Speech](https://platform.openai.com/docs/guides/text-to-speech) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## Prerequisite

Download the Kokoro model and voices.

```bash
export KOKORO_REVISION=c97b7bbc3e60f447383c79b2f94fee861ff156ac
# Download the ONNX model (~346 MBs)
docker exec -it speaches huggingface-cli download hexgrad/Kokoro-82M --include 'kokoro-v0_19.onnx' --revision $KOKORO_REVISION
# Download the voices.bin (~5.5 MBs) file
docker exec -it speaches curl --location --output /home/ubuntu/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/$KOKORO_REVISION/voices.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin
```

!!! note

    `rhasspy/piper-voices` audio samples can be found [here](https://rhasspy.github.io/piper-samples/)

Download the piper voices from [HuggingFace model repository](https://huggingface.co/rhasspy/piper-voices)

```bash
# Download all voices (~15 minutes / 7.7 GBs)
docker exec -it speaches huggingface-cli download rhasspy/piper-voices
# Download all English voices (~4.5 minutes)
docker exec -it speaches huggingface-cli download rhasspy/piper-voices --include 'en/**/*' 'voices.json'
# Download all qualities of a specific voice (~4 seconds)
docker exec -it speaches huggingface-cli download rhasspy/piper-voices --include 'en/en_US/amy/**/*' 'voices.json'
# Download specific quality of a specific voice (~2 seconds)
docker exec -it speaches huggingface-cli download rhasspy/piper-voices --include 'en/en_US/amy/medium/*' 'voices.json'
```

## Usage

### Curl

```bash
# Generate speech from text using the default values (model="hexgrad/Kokoro-82M", voice="af", response_format="mp3", speed=1.0, etc.)
curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!"}' --output audio.mp3
# Specifying the output format
curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "response_format": "wav"}' --output audio.wav
# Specifying the audio speed
curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "speed": 2.0}' --output audio.mp3

# List available (downloaded) voices
curl --silent http://localhost:8000/v1/audio/speech/voices
# List just the voice names
curl --silent http://localhost:8000/v1/audio/speech/voices | jq --raw-output '.[] | .voice_id'
# List just the rhasspy/piper-voices voice names
curl --silent 'http://localhost:8000/v1/audio/speech/voices?model_id=rhasspy/piper-voices' | jq --raw-output '.[] | .voice_id'
# List just the hexgrad/Kokoro-82M voice names
curl --silent 'http://localhost:8000/v1/audio/speech/voices?model_id=hexgrad/Kokoro-82M' | jq --raw-output '.[] | .voice_id'

# List just the voices in your language (piper)
curl --silent http://localhost:8000/v1/audio/speech/voices | jq --raw-output '.[] | select(.voice | startswith("en")) | .voice_id'

curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "voice": "af_sky"}' --output audio.mp3
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
- Maximum audio generation length is 10 seconds for `rhasspy/piper-voices` and 30 seconds for `hexgrad/Kokoro-82M`
