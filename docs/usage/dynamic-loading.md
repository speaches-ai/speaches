# Dynamic Model Loading

<!-- Verified by: tests/model_manager_test.py::test_model_cant_be_loaded_twice (prevents duplicate, implies multiple distinct models allowed) -->
Speaches dynamically loads and unloads models based on demand, similar to how Ollama manages LLM models. Unlike Ollama, speaches supports keeping multiple models loaded simultaneously.

## How It Works

<!-- Verified by: tests/model_manager_test.py::test_ttl_resets_after_usage -->
When you make a request specifying a model, speaches automatically loads it into memory if it isn't already loaded. After a configurable period of inactivity, the model is unloaded to free resources.

Each model type has its own TTL (time-to-live) setting:

<!-- Verified by: tests/model_manager_test.py::test_model_unloaded_after_ttl -->
| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `STT_MODEL_TTL` | Seconds until an STT model is unloaded after last use | `-1` (never) |
| `TTS_MODEL_TTL` | Seconds until a TTS model is unloaded after last use | `-1` (never) |
| `VAD_MODEL_TTL` | Seconds until a VAD model is unloaded after last use | `-1` (never) |

<!-- Verified by: tests/model_manager_test.py::test_model_unloaded_after_ttl (TTL behavior), test_model_is_unloaded_after_request_when_ttl_is_zero (TTL=0) -->
A value of `-1` means the model is never unloaded. A value of `0` means the model is unloaded immediately after each request.

<!-- Verified by: tests/test_doc_claims.py::test_preload_models_downloads_and_warms_up -->
## Preloading Models

You can preload specific models at startup using the `PRELOAD_MODELS` environment variable:

```bash
export PRELOAD_MODELS='["Systran/faster-whisper-tiny", "speaches-ai/Kokoro-82M-v1.0-ONNX"]'
```

Models listed here will be downloaded (if not already cached) and loaded into memory during startup. The application will exit if any listed model fails to download.

<!-- Verified by: tests/test_doc_claims.py::test_warmup_all_local_models_calls_warmup, test_warmup_all_local_models_disabled_skips_warmup -->
## Startup Behavior

By default, speaches loads all locally cached models into memory at startup. This can be controlled with the `WARMUP_ALL_LOCAL_MODELS` environment variable:

- `true` (default): All locally cached models are loaded on startup
- `false`: Only models listed in `PRELOAD_MODELS` are loaded on startup; other models are loaded on first request
