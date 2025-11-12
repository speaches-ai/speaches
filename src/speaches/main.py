from __future__ import annotations

import logging
import os
import uuid

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.exception_handlers import (
    http_exception_handler,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse

from speaches.dependencies import ApiKeyDependency, get_config
from speaches.logger import setup_logger
from speaches.routers.chat import (
    router as chat_router,
)
from speaches.routers.misc import (
    router as misc_router,
)
from speaches.routers.models import (
    router as models_router,
)
from speaches.routers.realtime_rtc import (
    router as realtime_rtc_router,
)
from speaches.routers.realtime_ws import (
    router as realtime_ws_router,
)
from speaches.routers.speech import (
    router as speech_router,
)
from speaches.routers.speech_embedding import (
    router as speech_embedding_router,
)
from speaches.routers.stt import (
    router as stt_router,
)
from speaches.routers.vad import (
    router as vad_router,
)
from speaches.utils import APIProxyError

# https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
# https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags
TAGS_METADATA = [
    {"name": "automatic-speech-recognition"},
    {"name": "speech-to-text"},
    {"name": "speaker-embedding"},
    {"name": "realtime"},
    {"name": "models"},
    {"name": "diagnostic"},
    {
        "name": "experimental",
        "description": "Not meant for public use yet. May change or be removed at any time.",
    },
]


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    # Initialize OpenTelemetry if endpoint is configured
    if config.otel_exporter_otlp_endpoint:
        from speaches.tracing import setup_telemetry

        setup_telemetry(config.otel_exporter_otlp_endpoint, config.otel_service_name)

        # Auto-instrument common libraries
        from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        AsyncioInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        LoggingInstrumentor().instrument()

    # Create main app WITHOUT global authentication
    app = FastAPI(
        title="Speaches",
        version="0.8.3",  # TODO: update this on release
        license_info={"name": "MIT License", "identifier": "MIT"},
        openapi_tags=TAGS_METADATA,
    )

    # Instrument FastAPI app if telemetry is enabled
    if config.otel_exporter_otlp_endpoint:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    # Register global exception handler for APIProxyError
    @app.exception_handler(APIProxyError)
    async def _api_proxy_error_handler(_request: Request, exc: APIProxyError) -> JSONResponse:
        error_id = str(uuid.uuid4())
        logger.exception(f"[{{error_id}}] {exc.message}")
        content = {
            "detail": exc.message,
            "hint": exc.hint,
            "suggested_fixes": exc.suggestions,
            "error_id": error_id,
        }

        # HACK: replace with something else
        log_level = os.getenv("SPEACHES_LOG_LEVEL", "INFO").upper()
        if log_level == "DEBUG" and exc.debug:
            content["debug"] = exc.debug
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(StarletteHTTPException)
    async def _custom_http_exception_handler(request: Request, exc: HTTPException) -> Response:
        logger.error(f"HTTP error: {exc}")
        return await http_exception_handler(request, exc)

    # HTTP routers WITH authentication (if API key is configured)
    http_dependencies = []
    if config.api_key is not None:
        http_dependencies.append(ApiKeyDependency)

    app.include_router(chat_router, dependencies=http_dependencies)
    app.include_router(stt_router, dependencies=http_dependencies)
    app.include_router(models_router, dependencies=http_dependencies)
    app.include_router(misc_router, dependencies=http_dependencies)
    app.include_router(realtime_rtc_router, dependencies=http_dependencies)
    app.include_router(speech_router, dependencies=http_dependencies)
    app.include_router(speech_embedding_router, dependencies=http_dependencies)
    app.include_router(vad_router, dependencies=http_dependencies)

    # WebSocket router WITHOUT authentication (handles its own)
    app.include_router(realtime_ws_router)

    # HACK: move this elsewhere
    app.get("/v1/realtime", include_in_schema=False)(lambda: RedirectResponse(url="/v1/realtime/"))
    app.mount("/v1/realtime", StaticFiles(directory="realtime-console/dist", html=True))

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if config.enable_ui:
        import gradio as gr

        from speaches.ui.app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="")

        logger = logging.getLogger("speaches.main")
        if config.host and config.port:
            display_host = "localhost" if config.host in ("0.0.0.0", "127.0.0.1") else config.host
            url = f"http://{display_host}:{config.port}/"
            logger.info(f"\n\nTo view the gradio web ui of speaches open your browser and visit:\n\n{url}\n\n")
        # If host or port is missing, do not print a possibly incorrect URL.

    return app
