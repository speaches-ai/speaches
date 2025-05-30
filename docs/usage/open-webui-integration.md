## Using the UI

1. Go to the [Admin Settings](http://localhost:8080/admin/settings) page
2. Click on the "Audio" tab
3. Update settings
   - Speech-to-Text Engine: OpenAI
   - API Base URL: http://speaches:8000/v1
   - API Key: does-not-matter-what-you-put-but-should-not-be-empty
   - Model: Systran/faster-distil-whisper-large-v3
4. Click "Save"

## Using environment variables (Docker Compose)

!!! warning

    This doesn't seem to work when you've previously used the UI to set the STT engine.

```yaml
# NOTE: Some parts of the file are omitted for brevity.
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ...
    environment:
      ...
      # Environment variables are documented here https://docs.openwebui.com/getting-started/env-configuration#speech-to-text
      AUDIO_STT_ENGINE: "openai"
      AUDIO_STT_OPENAI_API_BASE_URL: "http://speaches:8000/v1"
      AUDIO_STT_OPENAI_API_KEY: "does-not-matter-what-you-put-but-should-not-be-empty"
      AUDIO_STT_MODEL: "Systran/faster-distil-whisper-large-v3"
  speaches:
    image: ghcr.io/speaches-ai/speaches:latest-cuda
    ...
```
