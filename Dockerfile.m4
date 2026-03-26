ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Dependencies
# ffmpeg is needed for audio processing
# curl/ca-certificates for downloads
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# User setup (matching original)
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH
WORKDIR $HOME/speaches

# Install uv (Python package manager)
COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

# Install dependencies using uv
# --mount=type=cache is supported by default in Docker Desktop for Mac
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --no-dev

# Copy source code
COPY --chown=ubuntu . .

# Final sync to install the project itself (if needed) or verify
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode --no-dev

# Create cache directory for HuggingFace
RUN mkdir -p $HOME/.cache/huggingface/hub

# Environment variables
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV PATH="$HOME/speaches/.venv/bin:$PATH"
# Disable HF transfer acceleration as it can be problematic in containers
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# Privacy settings
ENV DO_NOT_TRACK=1
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
