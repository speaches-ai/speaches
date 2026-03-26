from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import TYPE_CHECKING
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

from speaches.dependencies import ApiKeyDependency, get_config, get_executor_registry
from speaches.logger import setup_logger
from speaches.routers.chat import (
    router as chat_router,
)
from speaches.routers.diarization import (
    router as diarization_router,
)
from speaches.routers.misc import (
    public_router as misc_public_router,
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
from speaches.utils import APIProxyError, CudaOutOfMemoryError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from speaches.config import Config
    from speaches.executors.shared.registry import ExecutorRegistry

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


def _format_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / (1 << 10):.1f} KB"


def _log_startup_summary(config: Config, executor_registry: ExecutorRegistry) -> None:
    logger = logging.getLogger(__name__)
    lines: list[str] = [""]
    lines.append("=" * 60)
    lines.append("  Speaches startup summary")
    lines.append("=" * 60)

    # Server
    lines.append(f"  Host: {config.host}:{config.port}")
    lines.append(f"  Log level: {config.log_level}")
    lines.append(f"  Auth: {'enabled' if config.api_key else 'disabled'}")
    lines.append(f"  UI: {'enabled' if config.enable_ui else 'disabled'}")
    if config.otel_exporter_otlp_endpoint:
        lines.append(f"  OpenTelemetry: {config.otel_exporter_otlp_endpoint}")

    # GPU / Memory
    lines.append("")
    lines.append("  -- GPU / Memory --")
    lines.append(f"  GPU memory limit: {_format_bytes(config.gpu_mem_limit)}")
    lines.append(f"  Whisper device: {config.whisper.inference_device}")
    lines.append(f"  Whisper compute type: {config.whisper.compute_type}")
    lines.append(f"  Whisper batch size: {config.whisper.batch_size}")
    lines.append(f"  Whisper max concurrency: {config.whisper.max_concurrency}")
    lines.append(f"  Whisper num workers: {config.whisper.num_workers}")
    if config.whisper.flash_attention:
        lines.append("  Whisper flash attention: enabled")

    # Model TTLs
    def _ttl_str(ttl: int) -> str:
        if ttl == -1:
            return "never unload"
        if ttl == 0:
            return "immediate"
        return f"{ttl}s"

    lines.append("")
    lines.append("  -- Model TTLs --")
    lines.append(f"  STT: {_ttl_str(config.stt_model_ttl)}")
    lines.append(f"  TTS: {_ttl_str(config.tts_model_ttl)}")
    lines.append(f"  VAD: {_ttl_str(config.vad_model_ttl)}")

    # Executors and loaded models
    lines.append("")
    lines.append("  -- Executors --")
    for executor in executor_registry.all_executors():
        loaded = list(executor.model_manager.loaded_models.keys())
        try:
            local_models = [m.id for m in executor.model_registry.list_local_models()]
        except (OSError, ValueError):
            local_models = []
        status = f"{len(loaded)} loaded, {len(local_models)} local"
        lines.append(f"  {executor.name} [{executor.task}]: {status}")
        lines.extend(f"    * {model_id} (loaded)" for model_id in loaded)
        lines.extend(f"    - {model_id} (local)" for model_id in local_models if model_id not in loaded)

    # VAD
    lines.append("")
    lines.append(f"  VAD model: {config.vad_model}")
    lines.append(f"  Warmup all local models: {config.warmup_all_local_models}")

    lines.append("=" * 60)
    logger.info("\n".join(lines))


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger = logging.getLogger(__name__)
    config = get_config()
    executor_registry = get_executor_registry()

    if config.preload_models:
        logger.info(f"Preloading {len(config.preload_models)} models on startup")

        for model_id in config.preload_models:
            logger.info(f"Downloading model: {model_id}")
            executor_registry.download_model_by_id(model_id)
            logger.info(f"Successfully downloaded model: {model_id}")

        for model_id in config.preload_models:
            logger.info(f"Warming up model: {model_id}")
            await executor_registry.warmup_model(model_id)
            logger.info(f"Model '{model_id}' is warm")

    if config.warmup_all_local_models:
        await executor_registry.warmup_local_models()
    await executor_registry.warmup_inference()

    _log_startup_summary(config, executor_registry)

    yield


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

        AsyncioInstrumentor().instrument()  # pyrefly: ignore[missing-attribute]
        HTTPXClientInstrumentor().instrument()  # pyrefly: ignore[missing-attribute]
        LoggingInstrumentor().instrument()  # pyrefly: ignore[missing-attribute]

    # Create main app WITHOUT global authentication
    app = FastAPI(
        title="Speaches",
        version="0.8.3",  # TODO: update this on release
        license_info={"name": "MIT License", "identifier": "MIT"},
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
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

    @app.exception_handler(CudaOutOfMemoryError)
    async def _cuda_oom_handler(_request: Request, exc: CudaOutOfMemoryError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": str(exc),
                "hint": "The GPU does not have enough memory for this request. Try again shortly or reduce audio length.",
            },
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _custom_http_exception_handler(request: Request, exc: HTTPException) -> Response:
        logger.error(f"HTTP error: {exc}")
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Public routers WITHOUT authentication
    app.include_router(misc_public_router)

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
    app.include_router(diarization_router, dependencies=http_dependencies)

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
