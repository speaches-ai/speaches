---
id: troubleshooting
aliases: []
tags: []
---

#### `uvx` command not found

You are either running the command from within the Docker container or the `uvx` command is not installed. This likely means you are trying to run the command from within the Docker container, but the `uvx` command line tool is not installed in the container. You have two options to resolve this issue:

- (**Recommended**) Run the `uvx` command from the host machine instead of the Docker container. If you don't have the `uvx` command line tool installed on your host machine, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
- You could also install the `uvx` command line tool inside the Docker container. NOTE: any changes you make to the container will be lost when the container is stopped or removed. To install the `uvx` command line tool inside the Docker container, you can should use the same installation instructions as you would for the host machine.

#### CUDA out of memory (OOM) errors

<!-- Verified by: src/speaches/__init__.py::DEFAULT_GPU_MEM_LIMIT (536870912 = 512MB) -->
If you're seeing CUDA OOM errors during inference, you can limit GPU memory usage via the `GPU_MEM_LIMIT` environment variable (value in bytes, default is 512MB / `536870912`). This limit is shared across ONNX Runtime and CTranslate2 inference backends.

```bash
export GPU_MEM_LIMIT=1073741824  # 1GB
```

<!-- Verified by: src/speaches/config.py::WhisperConfig.batch_size (default=4) -->
<!-- Verified by: src/speaches/config.py::WhisperConfig.max_concurrency (default=1) -->
For Whisper specifically, you can also reduce `WHISPER__BATCH_SIZE` (default: 4) or limit concurrent inference with `WHISPER__MAX_CONCURRENCY` (default: 1).

#### Models not loading on startup

<!-- Verified by: src/speaches/config.py::Config.warmup_all_local_models (default=True) -->
<!-- Verified by: src/speaches/config.py::Config.preload_models (config field) -->
By default, speaches loads all locally cached models into memory at startup (`WARMUP_ALL_LOCAL_MODELS=true`). If you want to only load specific models, set `WARMUP_ALL_LOCAL_MODELS=false` and use `PRELOAD_MODELS` to specify which models to load:

```bash
export WARMUP_ALL_LOCAL_MODELS=false
export PRELOAD_MODELS='["Systran/faster-whisper-tiny", "speaches-ai/Kokoro-82M-v1.0-ONNX"]'
```
