# speaches/main.py
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from speaches.config import Config
from speaches.dependencies import ApiKeyDependency, get_config
from speaches.logger import setup_logger
from speaches.routers.chat import router as chat_router
from speaches.routers.misc import router as misc_router
from speaches.routers.models import router as models_router
from speaches.routers.realtime.rtc import router as realtime_rtc_router
from speaches.routers.realtime.ws import router as realtime_ws_router
from speaches.routers.speech import router as speech_router
from speaches.routers.stt import router as stt_router
from speaches.routers.vad import router as vad_router
from speaches.utils import APIProxyError

# https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
# https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags
TAGS_METADATA = [
    {"name": "automatic-speech-recognition"},
    {"name": "speech-to-text"},
    {"name": "realtime"},
    {"name": "models"},
    {"name": "diagnostic"},
    {
        "name": "experimental",
        "description": "Not meant for public use yet. May change or be removed at any time.",
    },
]

# --- Helper functions for clean app creation ---

def _register_routers(app: FastAPI):
    """Includes all the API routers in the FastAPI app."""
    app.include_router(chat_router)
    app.include_router(stt_router)
    app.include_router(models_router)
    app.include_router(misc_router)
    app.include_router(realtime_rtc_router)
    app.include_router(realtime_ws_router)
    app.include_router(speech_router)
    app.include_router(vad_router)

def _register_exception_handlers(app: FastAPI):
    """Registers global exception handlers."""
    @app.exception_handler(APIProxyError)
    async def api_proxy_error_handler(request: Request, exc: APIProxyError):
        error_id = str(uuid.uuid4())
        # Use the module-level logger from the top of the file
        logger.exception(f"[{{error_id}}] {exc.message}")
        content = {
            "detail": exc.message,
            "hint": exc.hint,
            "suggested_fixes": exc.suggestions,
            "error_id": error_id,
        }
        # Avoid importing os inside a function
        import os
        log_level = os.getenv("SPEACHES_LOG_LEVEL", "INFO").upper()
        if log_level == "DEBUG" and exc.debug:
            content["debug"] = exc.debug
        return JSONResponse(status_code=exc.status_code, content=content)

def _mount_ui_and_static(app: FastAPI, config: Config):
    """Mounts static files and the Gradio UI if enabled."""
    # HACK: move this elsewhere
    app.get("/v1/realtime", include_in_schema=False)(lambda: RedirectResponse(url="/v1/realtime/"))
    app.mount("/v1/realtime", StaticFiles(directory="realtime-console/dist", html=True))

    if config.enable_ui:
        import gradio as gr
        from speaches.ui.app import create_gradio_demo
        # Mount the Gradio app. The original `app = gr.mount...` reassignment is avoided.
        gr.mount_gradio_app(app, create_gradio_demo(config), path="")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events (startup and shutdown).

    This context manager is the modern replacement for the deprecated
    `@app.on_event("startup")` decorator. Code before the `yield` runs on
    startup; code after runs on shutdown.
    """
    # --- STARTUP LOGIC ---
    logger = logging.getLogger(__name__)
    app_config = get_config()

    # Prioritize runtime info from app.state (set by run.py), with a fallback to static config.
    host = getattr(app.state, "server_host", app_config.resolved_host)
    port = getattr(app.state, "server_port", app_config.resolved_port)
    is_ssl = getattr(app.state, "server_is_ssl", bool(app_config.ssl_keyfile and app_config.ssl_certfile))

    protocol = "https" if is_ssl else "http"
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    access_url = f"{protocol}://{display_host}:{port}"
    app.state.access_url = access_url  # Store for potential later use

    # Dynamically update CORS origins if the middleware is present.
    # We find the middleware and modify its `allow_origins` list directly.
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            ui_origin = urlparse(access_url)._replace(path="", params="", query="", fragment="").geturl()
            if ui_origin not in middleware.options["allow_origins"]:
                middleware.options["allow_origins"].append(ui_origin)
                logger.info(f"Dynamically added '{ui_origin}' to allowed CORS origins.")
            break # Stop after finding the first CORS middleware

    # Log the final, user-friendly message.
    if app_config.enable_ui:
        if app_config.host and app_config.port: # Preserving original logic check
            logger.info(f"\n\nTo view the gradio web ui of speaches open your browser and visit:\n\n{access_url}\n\n")
        # If host or port is missing, do not print a possibly incorrect URL.
        # This original check is now less relevant with our robust config, but is kept for fidelity.

    yield
    # --- SHUTDOWN LOGIC (if any) ---
    logger.info("Speaches server shutting down.")


# --- Main Application Factory ---

def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(
        dependencies=dependencies,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan # Use the modern lifespan manager
    )

    # Register global exception handler for APIProxyError
    _register_exception_handlers(app)

    # Include all API routers
    _register_routers(app)

    # Mount static files and the Gradio UI
    _mount_ui_and_static(app, config)

    # The original CORS middleware block, now corrected and fully preserved.
    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.allow_origins), # Use a mutable list copy
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app
