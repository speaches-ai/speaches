# Speaches

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. Speach-to-Text is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for Text-to-Speech [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/speaches-ai/Kokoro-82M-v1.0-ONNX) are used. This project aims to be Ollama, but for TTS/STT models.

## Features:

- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
<!-- Verified by: tests/sse_test.py::test_streaming_transcription_text -->
- Streaming support (transcription is sent via SSE as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
<!-- Verified by: tests/speech_test.py::test_create_speech_formats -->
- Text-to-Speech via `kokoro`(Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
<!-- Verified by: tests/speech_test.py::test_create_speech_formats (mp3, wav, flac, opus, aac) -->
- Speech response formats: opus, aac, wav, mp3, and flac.
<!-- Verified by: tests/speech_embedding_test.py::test_create_speech_embedding -->
- Speaker diarization and speech embeddings.
<!-- Verified by: tests/vad_test.py::test_speech_timestamps_basic, tests/test_vad_v6.py::test_vad_v6_basic -->
- Voice Activity Detection (VAD) via Silero VAD v5 and v6.
<!-- Verified by: tests/test_doc_claims.py::test_default_gpu_mem_limit_value, test_config_gpu_mem_limit_default, test_ct2_cuda_env_var_set_from_gpu_mem_limit -->
- GPU and CPU support with configurable GPU memory limits shared across inference backends.
<!-- Verified by: flake.nix NixOS e2e test (opentelemetry module import verification) -->
- OpenTelemetry observability support.
- [Deployable via Docker Compose / Docker or Nix/NixOS](https://speaches.ai/installation/)
<!-- Verified by: tests/realtime_api_test.py, tests/e2e_realtime.py -->
- [Realtime API](https://speaches.ai/usage/realtime-api)
- [Highly configurable](https://speaches.ai/configuration/)

Please create an issue if you find a bug, have a question, or a feature suggestion.

## Demos

### Realtime API

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc" type="video/webm">
</video>

(Excuse the breathing lol. Didn't have enough time to record a better demo)

### Streaming Transcription

TODO

### Speech Generation

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b" type="video/webm">
</video>
