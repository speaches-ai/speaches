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
        debug: Any = None,  # noqa: ANN401
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
