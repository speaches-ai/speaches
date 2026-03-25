ARG BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/speaches-ai/speaches"
LABEL org.opencontainers.image.licenses="MIT"
# `ffmpeg` is installed because without it `gradio` won't work with mp3(possible others as well) files
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Workdir outside $HOME to avoid permission issues when running as non-root user
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv
COPY . .
# Specify the cache and install directories for `uv` to avoid permission issues when running as non-root user
ENV UV_CACHE_DIR=/app/.uv-cache
ENV UV_PYTHON_INSTALL_DIR=/app/.uv-python
RUN uv sync --frozen --compile-bytecode --no-dev
# Change the group ownership and permissions of the installed packages to allow access for non-root users
RUN chgrp -R 0 /app/.venv/bin/ && \
    chmod -R g=u  /app/.venv/bin/

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV PATH="/app/.venv/bin:$PATH"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubenablehftransfer
# NOTE: I've disabled this because it doesn't inside of Docker container. I couldn't pinpoint the exact reason. This doesn't happen when running the server locally.
# RuntimeError: An error occurred while downloading using `hf_transfer`. Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#donottrack
# https://www.reddit.com/r/StableDiffusion/comments/1f6asvd/gradio_sends_ip_address_telemetry_by_default/
ENV DO_NOT_TRACK=1
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1
EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
