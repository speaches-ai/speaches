import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
import os
from typing import Any
import uuid


class APIProxyError(Exception):
    """Exception for structured, actionable API or proxy errors.

    Args:
        message: Human-readable error message.
        hint: Short actionable hint for the user.
        suggestions: List of actionable suggestions for the user.
        status_code: HTTP status code (default 500).
        debug: Optional debug info (stack trace, request ID, etc.).
        error_id: Unique error ID for traceability.
        timestamp: When the error occurred (ISO 8601, UTC).

    """

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        suggestions: list[str] | None = None,
        status_code: int = 500,
        debug: Any = None,
        error_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.message = message
        self.hint = hint
        self.suggestions = suggestions or []
        self.status_code = status_code
        self.debug = debug
        self.error_id = error_id or uuid.uuid4().hex
        self.timestamp = timestamp or datetime.now(UTC).isoformat()


def format_api_proxy_error(exc: "APIProxyError", context: str = "") -> str:
    debug_mode = os.environ.get("SPEACHES_LOG_LEVEL", "").lower() == "debug"
    user_message = f"An error occurred: {exc.message} (Error ID: {exc.error_id}). Please try again or contact support."
    suggestions = exc.suggestions or [
        "Double-check your input data and file format (e.g., ensure audio files are WAV/MP3 and not corrupted).",
        "Verify your API key and endpoint configuration in the settings.",
        "Check your internet/network connection.",
        "If the error persists, restart the application or server.",
        "Contact support with the error ID and debug info if available.",
    ]
    debug_info = (
        f"Debug: {exc.debug}\nContext: {context}\nTimestamp: {exc.timestamp}" if debug_mode and exc.debug else ""
    )
    return f"[ERROR] {user_message}\nSuggestions: {', '.join(suggestions)}" + (f"\n{debug_info}" if debug_info else "")


def async_to_sync_generator[T](async_gen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        should_close_loop = True
    else:
        should_close_loop = False

    try:
        while True:
            try:
                # Get the next item from the async generator
                if should_close_loop:
                    item = loop.run_until_complete(async_gen.__anext__())
                else:
                    # If there's already a running loop, we need to run in a thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, async_gen.__anext__())
                        item = future.result()
                yield item
            except StopAsyncIteration:
                break
    finally:
        # Clean up the async generator
        try:
            if should_close_loop:
                loop.run_until_complete(async_gen.aclose())
            else:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_gen.aclose())
                    future.result()
        except Exception:  # noqa: BLE001, S110
            pass  # Ignore cleanup errors

        if should_close_loop:
            loop.close()
