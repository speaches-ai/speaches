TODO: add a note about automatic downloads
TODO: mention streaming
TODO: add a demo
TODO: talk about audio format
TODO: add a note about performance
TODO: add a note about vad

!!! note

    Before proceeding, you should be familiar with the [OpenAI Speech-to-Text](https://platform.openai.com/docs/guides/speech-to-text) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)

## Download a STT model

```bash
export SPEACHES_BASE_URL="http://localhost:8000"

# Listing all available STT models
uvx speaches-cli registry ls --task automatic-speech-recognition | jq '.data | [].id'

# Downloading a Systran/faster-distil-whisper-small.en model
uvx speaches-cli model download Systran/faster-distil-whisper-small.en

# Check that the model has been installed
uvx speaches-cli model ls --task text-to-speech | jq '.data | map(select(.id == "Systran/faster-distil-whisper-small.en"))'
```

## Usage

### Orukeet

Orukeet is an optional, local 25-language recognizer based on Parakeet TDT v3. Install it through the model registry:

```bash
uvx speaches-cli model download oruk/orukeet
curl "$SPEACHES_BASE_URL/v1/audio/transcriptions" \
  -F "file=@audio.wav" -F "model=oruk/orukeet" -F "response_format=json"
```

The server downloads the pinned INT8 ONNX export and its license files from [Hugging Face](https://huggingface.co/oruk/orukeet/tree/1751fce6ecde442f14543cf1804800c49b3e415c/onnx/combined-v0.1.0-int8), verifies SHA-256 hashes, and reuses the Hugging Face cache. The required `config.json` download participates in Hugging Face's normal model download accounting. Audio stays on your server; transcription does not contact a hosted inference service.

Use `json` or `text` responses. This executor transcribes complete recordings; streaming, translation, subtitle formats and language forcing are not supported. The weights use CC BY-SA 4.0; the downloaded notices include the converter and preprocessor terms.

### Curl

```bash
export SPEACHES_BASE_URL="http://localhost:8000"
export TRANSCRIPTION_MODEL_ID="Systran/faster-distil-whisper-small.en"

curl -s "$SPEACHES_BASE_URL/v1/audio/transcriptions" -F "file=@audio.wav" -F "model=$TRANSCRIPTION_MODEL_ID"
```

### Python

=== "httpx"

    ```python
    import httpx

    with open('audio.wav', 'rb') as f:
        files = {'file': ('audio.wav', f)}
        response = httpx.post('http://localhost:8000/v1/audio/transcriptions', files=files)

    print(response.text)
    ```

### OpenAI SDKs

!!! note

    Although this project doesn't require an API key, all OpenAI SDKs require an API key. Therefore, you will need to set it to a non-empty value. Additionally, you will need to overwrite the base URL to point to your server.

    This can be done by setting the `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables or by passing them as arguments to the SDK.

=== "Python"

    ```python
    from pathlib import Path

    from openai import OpenAI

    client = OpenAI()

    with Path("audio.wav").open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(model="Systran/faster-whisper-small", file=audio_file)

    print(transcription.text)
    ```

=== "CLI"

    ```bash
    export OPENAI_BASE_URL=http://localhost:8000/v1/
    export OPENAI_API_KEY="cant-be-empty"
    openai api audio.transcriptions.create -m Systran/faster-whisper-small -f audio.wav --response-format text
    ```

=== "Other"

    See [OpenAI libraries](https://platform.openai.com/docs/libraries).
