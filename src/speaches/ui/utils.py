import gradio as gr
import httpx
from openai import AsyncOpenAI

from speaches.config import Config

TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)


def base_url_from_gradio_req(request: gr.Request | None, config: Config) -> str:
    if config.loopback_host_url is not None and len(config.loopback_host_url) > 0:
        return config.loopback_host_url
    if request is None:
        msg = "`request` is None (this happens when running the service behind a reverse-proxy) you should set config.loopback_host_url"
        raise ValueError(msg)
    assert request.request is not None, request
    # NOTE: `request.request.url` seems to always have a path of "/gradio_api/queue/join"
    return f"{request.request.url.scheme}://{request.request.url.netloc}"


def http_client_from_gradio_req(
    request: gr.Request, config: Config, user_api_key: str | None = None
) -> httpx.AsyncClient:
    base_url = base_url_from_gradio_req(request, config)
    headers = {}
    if user_api_key:
        headers["Authorization"] = f"Bearer {user_api_key}"
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=TIMEOUT,
        headers=headers if headers else None,
    )


def openai_client_from_gradio_req(request: gr.Request, config: Config, user_api_key: str | None = None) -> AsyncOpenAI:
    base_url = base_url_from_gradio_req(request, config)
    api_key_value = user_api_key if user_api_key else "cant-be-empty"
    return AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key=api_key_value,
        max_retries=0,
    )
