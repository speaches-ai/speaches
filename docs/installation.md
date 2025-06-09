!!! warning

    Additional steps are required to use the text-to-speech feature. Please see the [Text-to-Speech](./usage/text-to-speech.md).

## Docker Compose (Recommended)

!!! note

    I'm using newer Docker Compose features. If you are using an older version of Docker Compose, you may need need to update.

Download the necessary Docker Compose files

=== "CUDA"

    ```bash
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cuda.yaml
    export COMPOSE_FILE=compose.cuda.yaml
    ```

=== "CUDA (with CDI feature enabled)"

    ```bash
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cuda.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cuda-cdi.yaml
    export COMPOSE_FILE=compose.cuda-cdi.yaml
    ```

=== "CPU"

    ```bash
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cpu.yaml
    export COMPOSE_FILE=compose.cpu.yaml
    ```

Start the service

```bash
docker compose up --detach
```

??? note "Build from source"

    ```bash
    # NOTE: you need to install and enable [buildx](https://github.com/docker/buildx) for multi-platform builds

    # Build image with CUDA support
    docker compose --file compose.cuda.yaml build

    # Build image without CUDA support
    docker compose --file compose.cpu.yaml build
    ```

## Docker

=== "CUDA"

    ```bash
    docker run \
      --rm \
      --detach \
      --publish 8000:8000 \
      --name speaches \
      --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
      --gpus=all \
      ghcr.io/speaches-ai/speaches:latest-cuda
    ```

=== "CUDA (with CDI feature enabled)"

    ```bash
    docker run \
      --rm \
      --detach \
      --publish 8000:8000 \
      --name speaches \
      --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
      --device=nvidia.com/gpu=all \
      ghcr.io/speaches-ai/speaches:latest-cuda
    ```

=== "CPU"

    ```bash
    docker run \
      --rm \
      --detach \
      --publish 8000:8000 \
      --name speaches \
      --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
      ghcr.io/speaches-ai/speaches:latest-cpu
    ```

??? note "Build from source"

    ```bash
    docker build --tag speaches .

    # NOTE: you need to install and enable [buildx](https://github.com/docker/buildx) for multi-platform builds
    # Build image for both amd64 and arm64
    docker buildx build --tag speaches --platform linux/amd64,linux/arm64 .

    # Build image without CUDA support
    docker build --tag speaches --build-arg BASE_IMAGE=ubuntu:24.04 .
    ```

## Python (requires Python 3.12+ and `uv` package manager)

```bash
git clone https://github.com/speaches-ai/speaches.git
cd speaches
uv venv
source .venv/bin/activate
uv sync --all-extras
uvicorn --factory --host 0.0.0.0 speaches.main:create_app
```
