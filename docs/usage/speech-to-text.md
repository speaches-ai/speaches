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

### Curl

```bash
export SPEACHES_BASE_URL="http://localhost:8000"
export MODEL_ID="Systran/faster-distil-whisper-small.en"

curl -s http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "model=$MODEL_ID"
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
