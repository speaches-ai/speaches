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

# Installation

The `speaches` package is distributed as a single, "batteries-included" application. The standard installation provides all features, including the API server, web UI, and client tools.

## For Users

The recommended way to install `speaches` is as a command-line tool using `uv`. This installs the application and its dependencies into an isolated environment, making the `speaches` command available globally on your system.

```bash
git clone https://github.com/speaches-ai/speaches.git
cd speaches
uv venv
source .venv/bin/activate
uv sync --all-extras --upgrade
uv tool install .
speaches serve --host 0.0.0.0 --port 8000
```

After installation, you can run the server with `speaches serve` or explore other commands with `speaches --help`.

## For Developers (Contributing to Speaches)

If you plan to contribute to the `speaches` project, you must install it in "editable" mode from a local clone of the repository. This setup links the `speaches` command directly to your source code, so your edits are reflected immediately without reinstalling.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/path/to/speaches.git
    cd speaches
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install in Editable Mode with Development Extras:**
    This command installs the project along with all optional dependencies needed for running tests and other development tasks.
    ```bash
s   uv pip install -e '.[dev]'
    ```
The `speaches` command is now available in your shell for development and testing.
