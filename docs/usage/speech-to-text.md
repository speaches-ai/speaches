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
uvx speaches-cli model ls --task automatic-speech-recognition | jq '.data | map(select(.id == "Systran/faster-distil-whisper-small.en"))'
```

## Usage

### Curl

<!-- Verified by: tests/api_timestamp_granularities_test.py::test_api_json_response_format_and_timestamp_granularities_combinations -->
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

<!-- Verified by: tests/sse_test.py::test_streaming_transcription_text -->
## Streaming

Speaches supports streaming transcription via Server-Sent Events (SSE). The transcription is sent as the audio is processed — you don't need to wait for the entire audio to be transcribed.

### Curl

```bash
export SPEACHES_BASE_URL="http://localhost:8000"
export TRANSCRIPTION_MODEL_ID="Systran/faster-distil-whisper-small.en"

curl -N -s "$SPEACHES_BASE_URL/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=$TRANSCRIPTION_MODEL_ID" \
  -F "stream=true"
```

<!-- Verified by: tests/vad_test.py::test_speech_timestamps_basic -->
## Voice Activity Detection

By default, speaches applies a VAD (Voice Activity Detection) filter to remove silence and non-speech segments before transcription. This reduces hallucinations caused by background silence. The VAD filter can be controlled per-request or globally via the `_UNSTABLE_VAD_FILTER` environment variable.
