# Running Tests

## Prerequisites

### Nix (recommended)

The Nix development shell includes all dependencies needed to run the test suite.

```bash
git clone https://github.com/speaches-ai/speaches.git
cd speaches
nix develop
```

Once inside the shell, tests are run with:

```bash
PYTHONPATH=src:. python3 -m pytest tests/ -v
```

Or using the Taskfile:

```bash
task test
```

### uv

```bash
git clone https://github.com/speaches-ai/speaches.git
cd speaches
uv python install
uv venv && source .venv/bin/activate
uv sync
pytest tests/ -v
```

## Test Groups

The test suite is organized into several groups based on what they exercise and what infrastructure they require.

### Unit Tests (no external dependencies)

These tests run entirely in-process with no network, no models, and no external services. They always run.

| File | Description |
|------|-------------|
| `test_doc_claims.py` | Verifies documentation claims against source code and config |
| `test_input_audio_buffer.py` | Realtime API audio buffer ring-buffer logic |
| `test_conversation_state.py` | Realtime API state machine transitions |
| `test_session_config.py` | Realtime session configuration defaults |
| `test_interruption.py` | Realtime barge-in and interruption handling |
| `test_response_handler.py` | Realtime response generation and cancellation |
| `test_phrase_chunker.py` | Text chunking for TTS streaming |
| `text_utils_test.py` | SRT/VTT formatting, markdown stripping, text chunking |
| `realtime_api_test.py` | WebSocket auth, session config, realtime app creation |

### Integration Tests (require local models)

These tests start the speaches ASGI app in-process and exercise real model loading, inference, and API responses. Models are downloaded automatically on first run.

| File | Description |
|------|-------------|
| `speech_test.py` | TTS: format encoding, resampling, streaming, speed |
| `speech_embedding_test.py` | Speaker embedding generation and similarity |
| `vad_test.py` | VAD v5 speech timestamps |
| `test_vad_v6.py` | VAD v6 speech timestamps, threshold, silence duration |
| `sse_test.py` | Streaming transcription via SSE, VTT and SRT output |
| `api_timestamp_granularities_test.py` | Transcription with timestamp granularity combinations |
| `api_model_test.py` | Model listing and retrieval endpoints |
| `model_manager_test.py` | TTL-based model loading and unloading |
| `auth_test.py` | API key authentication on protected endpoints |

### Voice Chat Tests (require a chat completion backend)

The voice chat tests (`api_chat_test.py`) exercise the full STT → LLM → TTS pipeline. The `target="speaches"` variants run the pipeline locally but proxy the chat completion to an external LLM.

These tests are skipped unless a chat completion backend is configured:

```bash
# Using Ollama (recommended for local testing)
export CHAT_COMPLETION_BASE_URL=http://localhost:11434/v1
export CHAT_COMPLETION_MODEL=llama3.2  # or any model you have pulled

# Using OpenAI
export OPENAI_API_KEY=sk-xxx
# CHAT_COMPLETION_BASE_URL defaults to https://api.openai.com/v1

# Then run:
PYTHONPATH=src:. python3 -m pytest tests/api_chat_test.py -v -k speaches
```

| Marker | Trigger | What it tests |
|--------|---------|---------------|
| `requires_chat_backend` | `CHAT_COMPLETION_BASE_URL` or `OPENAI_API_KEY` is set | `target="speaches"` — full pipeline with local STT/TTS, proxied LLM |
| `requires_openai` | `OPENAI_API_KEY` is set | `target="openai"` — direct OpenAI API conformance |

### OpenAI Conformance Tests (require `OPENAI_API_KEY`)

These tests compare speaches behavior against the real OpenAI API. They are always skipped without `OPENAI_API_KEY`.

| File | Description |
|------|-------------|
| `openai_timestamp_granularities_test.py` | Timestamp granularity behavior on OpenAI |
| `openai_transcription_test.py` | Transcription format support on OpenAI |
| `speech_test.py` (OpenAI variants) | TTS opus sample rate and chunked encoding on OpenAI |

### End-to-End Tests (Nix only)

Full system tests that start a real speaches server, load models from a pre-built cache, and exercise the HTTP API. These use NixOS VM testing infrastructure.

```bash
# Run the default e2e test (CPU, Python 3.12)
nix build .#checks.x86_64-linux.e2e

# Run for a specific Python version
nix build .#checks.x86_64-linux.e2e-python313

# Run the realtime WebSocket e2e test (uses a mock LLM)
nix run .#e2e-test-realtime

# Run e2e with real models on your host (no VM)
nix run .#e2e-test
```

The NixOS e2e tests (`nix build .#checks.*`) run inside a NixOS VM and test:

- Service startup via systemd
- Health endpoint
- Model listing
- TTS → STT round-trip pipeline (generates audio, transcribes it back, checks word overlap)
- OpenTelemetry module imports

The realtime e2e test (`e2e-test-realtime`) tests the WebSocket-based realtime API with a mock LLM backend — no OpenAI key needed.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_COMPLETION_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint for voice chat tests |
| `CHAT_COMPLETION_API_KEY` | falls back to `OPENAI_API_KEY` | API key for the chat completion backend |
| `CHAT_COMPLETION_MODEL` | `gpt-4o-mini` | Model name sent in voice chat test requests |
| `OPENAI_API_KEY` | (none) | Enables OpenAI conformance tests and serves as fallback for chat backend |

## Common Recipes

```bash
# Run only unit tests (fast, no models needed)
PYTHONPATH=src:. python3 -m pytest tests/test_doc_claims.py tests/test_input_audio_buffer.py tests/test_conversation_state.py tests/test_session_config.py tests/test_interruption.py tests/test_response_handler.py tests/test_phrase_chunker.py tests/text_utils_test.py tests/realtime_api_test.py -v

# Run integration tests (requires models, downloads on first run)
PYTHONPATH=src:. python3 -m pytest tests/ -v -k "not requires_openai and not requires_chat_backend"

# Run voice chat tests with Ollama
CHAT_COMPLETION_BASE_URL=http://localhost:11434/v1 CHAT_COMPLETION_MODEL=llama3.2 \
  PYTHONPATH=src:. python3 -m pytest tests/api_chat_test.py -v -k speaches

# Run everything (requires both Ollama and OPENAI_API_KEY)
CHAT_COMPLETION_BASE_URL=http://localhost:11434/v1 CHAT_COMPLETION_MODEL=llama3.2 \
  OPENAI_API_KEY=sk-xxx \
  PYTHONPATH=src:. python3 -m pytest tests/ -v
```
