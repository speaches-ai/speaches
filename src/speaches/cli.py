# speaches/cli.py
"""
This module defines the main command-line interface for the Speaches application,
powered by Typer. It serves as a unified entry point for both running the API
server and interacting with it as a client.

After installation, you can use it like this:
  - `speaches serve` to start the API server (requires [server] extra).
  - `speaches model ls` to list available models on a running server.
  - `speaches --help` for a full list of commands.
"""
import json
import os
from typing import Optional

import httpx
import typer
# `import uvicorn` is now moved inside the `serve` command.

# Create the main Typer application
app = typer.Typer(
    name="speaches",
    help="A unified tool to serve and interact with the Speaches API.",
    add_completion=False, # Can be enabled for more advanced shell completion
)

# --- Server Command (`speaches serve`) ---

@app.command()
def serve(
    host: Optional[str] = typer.Option(
        None, "--host", help="Host to bind the server to. Overrides environment variables."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", help="Port to bind the server to. Overrides environment variables."
    ),
    ssl_keyfile: Optional[str] = typer.Option(
        None, "--ssl-keyfile", help="Path to the SSL key file. Overrides environment variables."
    ),
    ssl_certfile: Optional[str] = typer.Option(
        None, "--ssl-certfile", help="Path to the SSL certificate file. Overrides environment variables."
    ),
):
    """
    Starts the Speaches FastAPI server using Uvicorn.

    This command loads configuration from environment variables (e.g., SPEACHES_HOST,
    UVICORN_HOST) and .env files. Command-line options provided here will take the
    highest precedence.

    NOTE: This command requires the 'server' extras to be installed.
    Install with: `pip install 'speaches[server]'` or `uv tool install 'speaches[server]'`
    """
    try:
        # Lazy import: Uvicorn is only imported when `serve` is called.
        import uvicorn
    except ImportError:
        typer.secho(
            "Error: Uvicorn is not installed. The 'serve' command requires it.",
            fg=typer.colors.RED,
            bold=True,
        )
        typer.echo("To install the server dependencies, please run:")
        typer.secho("  pip install 'speaches[server]'", fg=typer.colors.CYAN)
        typer.echo("or, if using uv:")
        typer.secho("  uv tool install 'speaches[server]'", fg=typer.colors.CYAN)
        raise typer.Exit(code=1)

    from speaches.dependencies import get_config
    from speaches.main import create_app

    # Load configuration from Pydantic. This respects the established precedence rules.
    config = get_config()

    # CLI options take final precedence over everything.
    # This allows for maximum flexibility during runtime.
    final_host = host or config.resolved_host
    final_port = port or config.resolved_port
    final_ssl_keyfile = ssl_keyfile or config.ssl_keyfile
    final_ssl_certfile = ssl_certfile or config.ssl_certfile

    # Create the FastAPI app instance.
    fastapi_app = create_app()

    # Bridge the gap between the runner and the app: populate app.state.
    # This ensures the lifespan events have 100% accurate info.
    fastapi_app.state.server_host = final_host
    fastapi_app.state.server_port = final_port
    fastapi_app.state.server_is_ssl = bool(final_ssl_keyfile and final_ssl_certfile)

    # Run the Uvicorn server programmatically.
    uvicorn.run(
        fastapi_app,
        host=final_host,
        port=final_port,
        ssl_keyfile=final_ssl_keyfile,
        ssl_certfile=final_ssl_certfile,
        log_level=config.log_level.lower(),
    )


# --- Client Commands (`speaches model`, `speaches registry`) ---

# Client-side configuration and helper
SPEACHES_BASE_URL = os.getenv("SPEACHES_BASE_URL", "http://localhost:8000")
SPEACHES_OPENAI_BASE_URL = SPEACHES_BASE_URL + "/v1"
MODELS_URL = f"{SPEACHES_OPENAI_BASE_URL}/models"
REGISTRY_URL = f"{SPEACHES_OPENAI_BASE_URL}/registry"

try:
    client = httpx.Client(base_url=SPEACHES_BASE_URL, timeout=httpx.Timeout(None))
except httpx.InvalidURL:
    typer.secho(f"Error: Invalid SPEACHES_BASE_URL: '{SPEACHES_BASE_URL}'", fg=typer.colors.RED)
    raise typer.Exit(code=1)


def dump_response(response: httpx.Response) -> None:
    """Pretty-prints a JSON response or prints raw text."""
    if response.status_code >= 400:
        typer.secho(f"Error: Received status code {response.status_code}", fg=typer.colors.RED)

    if response.headers.get("Content-Type") == "application/json":
        try:
            data = response.json()
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError:
            typer.echo("Received non-JSON response:")
            typer.echo(response.text)
    else:
        typer.echo(response.text)

    if response.status_code >= 400:
        raise typer.Exit(code=1)

# Create sub-typers for command organization
registry_app = typer.Typer(help="Interact with the model registry.")
model_app = typer.Typer(help="Manage local models on a running server.")
audio_app = typer.Typer() # Retaining for future use
audio_speech_app = typer.Typer() # Retaining for future use

# Add the sub-commands to the main app
app.add_typer(registry_app, name="registry")
app.add_typer(model_app, name="model")

# Define the client commands
@registry_app.command("ls")
def registry_ls(task: Optional[str] = typer.Option(None, help="Filter registry by task type.")):
    """Lists all available models in the public registry."""
    params: dict[str, str] = {}
    if task is not None:
        params["task"] = task
    try:
        response = client.get(REGISTRY_URL, params=params)
        dump_response(response)
    except httpx.ConnectError:
        typer.secho(f"Error: Connection to {SPEACHES_BASE_URL} failed. Is the server running?", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@model_app.command("ls")
def models_ls(task: Optional[str] = typer.Option(None, help="Filter local models by task type.")):
    """Lists locally downloaded and available models."""
    params: dict[str, str] = {}
    if task is not None:
        params["task"] = task
    try:
        response = client.get(MODELS_URL, params=params)
        dump_response(response)
    except httpx.ConnectError:
        typer.secho(f"Error: Connection to {SPEACHES_BASE_URL} failed. Is the server running?", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@model_app.command("rm")
def model_rm(model_id: str = typer.Argument(..., help="The ID of the model to remove (e.g., 'Systran/faster-whisper-large-v3').")):
    """Removes (unloads) a model from memory."""
    try:
        response = client.delete(f"{MODELS_URL}/{model_id}")
        dump_response(response)
    except httpx.ConnectError:
        typer.secho(f"Error: Connection to {SPEACHES_BASE_URL} failed. Is the server running?", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@model_app.command("download")
def model_download(model_id: str = typer.Argument(..., help="The ID of the model to download (e.g., 'Systran/faster-whisper-large-v3').")):
    """Downloads a model from the registry to the local cache."""
    try:
        response = client.post(f"{MODELS_URL}/{model_id}")
        dump_response(response)
    except httpx.ConnectError:
        typer.secho(f"Error: Connection to {SPEACHES_BASE_URL} failed. Is the server running?", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# This check is useful if you ever want to run this script directly for debugging
if __name__ == "__main__":
    app()