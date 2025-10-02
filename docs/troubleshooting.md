---
id: troubleshooting
aliases: []
tags: []
---

#### `uvx` command not found

You are either running the command from within the Docker container or the `uvx` command is not installed. This likely means you are trying to run the command from within the Docker container, but the `uvx` command line tool is not installed in the container. You have two options to resolve this issue:

- (**Recommended**) Run the `uvx` command from the host machine instead of the Docker container. If you don't have the `uvx` command line tool installed on your host machine, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
- You could also install the `uvx` command line tool inside the Docker container. NOTE: any changes you make to the container will be lost when the container is stopped or removed. To install the `uvx` command line tool inside the Docker container, you can should use the same installation instructions as you would for the host machine.

#### Pyannote speaker diarization crashes the container when using CUDA

When running Pyannote speaker diarization alongside GPU-based transcription (e.g., Whisper with CUDA), the container may crash during concurrent operations. This is likely due to GPU memory contention or resource conflicts when both models attempt to use CUDA simultaneously.

**Workaround**: Configure Pyannote to use CPU while keeping transcription on GPU by setting the following environment variable:

```bash
export PYANNOTE__INFERENCE_DEVICE=cpu
```

This configuration allows:
- Whisper/transcription models to run on GPU (faster transcription)
- Pyannote diarization to run on CPU (preventing crashes)

The performance impact on diarization is typically minimal compared to the stability benefit. If you need maximum performance for both models, consider:
- Running them in separate containers with dedicated GPU memory allocations
- Increasing GPU memory if available
- Running operations sequentially instead of concurrently
