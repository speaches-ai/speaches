# Use the same base image as the original project for GPU support
ARG BASE_IMAGE=nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/Daniel-OS01/speaches"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies required by speaches and its dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    python3-pip \
    python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user 'ubuntu' to run the application
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
WORKDIR /home/ubuntu/app

# Set up a virtual environment
ENV VIRTUAL_ENV=/home/ubuntu/app/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install 'uv' for Python package management, as used in the original project
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency definition files
COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock ./
COPY --chown=ubuntu:ubuntu runpod_requirements.txt ./

# Install Python dependencies
# First, install from uv.lock for the main application, including UI extras for full functionality
RUN uv sync --frozen --compile-bytecode --extra ui
# Then, install Runpod-specific dependencies
RUN pip install -r runpod_requirements.txt

# Copy the application source code and necessary files
COPY --chown=ubuntu:ubuntu src/ ./src
COPY --chown=ubuntu:ubuntu realtime-console/ ./realtime-console
COPY --chown=ubuntu:ubuntu handler.py ./
COPY --chown=ubuntu:ubuntu model_aliases.json ./

# Environment variables for the Speaches server
ENV UVICORN_HOST=127.0.0.1
ENV UVICORN_PORT=8000
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Set the command to run the Runpod handler script
CMD ["python", "-u", "handler.py"]

