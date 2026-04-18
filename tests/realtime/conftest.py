from collections.abc import Generator
import logging
import os
from typing import Literal

from openai.types.realtime.realtime_client_event import RealtimeClientEvent
from openai.types.realtime.realtime_server_event import RealtimeServerEvent
from pydantic import BaseModel, Secret
import pytest

from speaches.logger import setup_logger

RealtimeEvent = RealtimeServerEvent | RealtimeClientEvent


setup_logger("INFO")
logger = logging.getLogger(__name__)


class EndpointConfig(BaseModel):
    name: str
    base_url: str
    api_key: Secret[str]
    realtime_model_id: str

    def realtime_url(self, intent: Literal["transcription", "conversation"]) -> str:
        ws_base_url = self.base_url.replace("http", "ws")
        if intent == "transcription":
            return f"{ws_base_url}/realtime?intent=transcription"
        else:
            return f"{ws_base_url}/realtime?model={self.realtime_model_id}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}


@pytest.fixture(scope="session")
def speaches_server() -> Generator[EndpointConfig]:
    import multiprocessing
    import socket
    import time

    import uvicorn

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = _find_free_port()
    config = uvicorn.Config(
        app="speaches.main:create_app", factory=True, host="0.0.0.0", port=port, log_level="warning"
    )
    os.environ["PORT"] = str(port)
    os.environ["HOST"] = "0.0.0.0"
    os.environ["ENABLE_UI"] = "false"
    os.environ["LOOPBACK_HOST_URL"] = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(config)
    # NOTE: multiprocessing.Process inherits the parent's environment, so env vars like
    # CHAT_COMPLETION_BASE_URL and CHAT_COMPLETION_API_KEY are automatically available to the child.
    process = multiprocessing.Process(target=server.run, daemon=True)
    process.start()

    base_url = f"http://127.0.0.1:{port}/v1"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        process.kill()
        raise RuntimeError(f"Speaches server failed to start on port {port}")

    yield EndpointConfig(
        name="speaches",
        base_url=base_url,
        api_key=Secret(os.environ.get("SPEACHES_API_KEY", "sk-xxx")),
        realtime_model_id=os.environ.get("SPEACHES_REALTIME_MODEL_ID", "gpt-4o-mini"),
    )

    process.kill()
    process.join(timeout=5)


def _get_external_endpoints() -> list[str]:
    endpoints: list[str] = []
    if os.environ.get("OPENAI_API_KEY"):
        endpoints.append("openai")
    else:
        logger.warning("OPENAI_API_KEY not set, skipping OpenAI endpoint tests")
    return endpoints


def _resolve_endpoint(name: str) -> EndpointConfig:
    if name == "openai":
        return EndpointConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=Secret(os.environ["OPENAI_API_KEY"]),
            realtime_model_id=os.environ.get("OPENAI_REALTIME_MODEL_ID", "gpt-realtime-1.5"),
        )
    raise ValueError(f"Unknown external endpoint: {name}")


@pytest.fixture(params=["speaches", *_get_external_endpoints()])
def endpoint(request: pytest.FixtureRequest, speaches_server: EndpointConfig) -> EndpointConfig:
    if request.param == "speaches":
        return speaches_server
    return _resolve_endpoint(request.param)
