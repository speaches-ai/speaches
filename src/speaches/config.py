# speaches/config.py
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Constants (unchanged) ---
SAMPLES_PER_SECOND = 16000
SAMPLE_WIDTH = 2
BYTES_PER_SECOND = SAMPLES_PER_SECOND * SAMPLE_WIDTH
# 2 BYTES = 16 BITS = 1 SAMPLE
# 1 SECOND OF AUDIO = 32000 BYTES = 16000 SAMPLES

type Device = Literal["cpu", "cuda", "auto"]
# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md#quantize-on-model-conversion
type Quantization = Literal[
    "int8", "int8_float16", "int8_bfloat16", "int8_float32", "int16", "float16", "bfloat16", "float32", "default"
]

# --- Nested Config Models ---
class WhisperConfig(BaseModel):
    """Configuration for the faster-whisper model.

    See: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599.
    """

    inference_device: Device = Field(default="auto", description="The device to use for inference ('cpu', 'cuda', 'auto').")
    device_index: int | list[int] = Field(default=0, description="A list of device IDs to use for inference.")
    compute_type: Quantization = Field(default="default", description="The quantization type to use for the model.")
    cpu_threads: int = Field(default=0, description="Number of threads to use when running on CPU (0 = auto).")
    num_workers: int = Field(default=1, description="The number of workers to use for parallel transcription.")
    ttl: int = Field(default=300, ge=-1, description="Time in seconds until the model is unloaded if unused. -1: never unload; 0: unload immediately.")
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """
    use_batched_mode: bool = Field(default=False, description="Whether to use batch mode for inference. This may become the default in the future.")
    """
    Whether to use batch mode(introduced in 1.1.0 `faster-whisper` release) for inference. This will likely become the default in the future and the configuration option will be removed.
    """

# --- Main Config Class ---
# TODO: document `alias` behaviour within the docstring
class Config(BaseSettings):
    """Defines the application's configuration settings.Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__INFERENCE_DEVICE`(note the double underscore)
    to `whisper.inference_device`, to set quantization to int8, use `WHISPER__COMPUTE_TYPE=int8`, etc.

    Values are loaded from environment variables or a .env file.
    The system uses a clear precedence for host and port settings:
    1. SPEACHES_HOST / SPEACHES_PORT (highest priority)
    2. UVICORN_HOST / UVICORN_PORT (fallback for compatibility)
    3. Default value in the code (lowest priority)

    For other settings, use the `SPEACHES_` prefix. For nested models like `whisper`,
    use a double underscore delimiter (e.g., `SPEACHES_WHISPER__INFERENCE_DEVICE=cpu`).
    """

    model_config = SettingsConfigDict(
        env_prefix="speaches_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # We define fields for both namespaces
    host: str | None = Field(default=None, description="Application-specific host. Overrides UVICORN_HOST.")
    uvicorn_host: str = Field(default="0.0.0.0", alias="UVICORN_HOST", description="Standard Uvicorn host, used as a fallback.")

    port: int | None = Field(default=None, description="Application-specific port. Overrides UVICORN_PORT.")
    uvicorn_port: int = Field(default=34331, alias="UVICORN_PORT", description="Standard Uvicorn port, used as a fallback.")

    # These are the final, resolved values that the app will use.
    # They are not environment variables themselves but are computed fields.
    resolved_host: str = "0.0.0.0"
    resolved_port: int = 34331

    @field_validator('resolved_host', mode='before')
    @classmethod
    def _resolve_host(cls, v, info: ValidationInfo) -> str:
        """Computes the definitive host based on the precedence rule."""
        # `info.data` holds the values of the other fields being validated.
        return info.data.get('host') or info.data.get('uvicorn_host')

    @field_validator('resolved_port', mode='before')
    @classmethod
    def _resolve_port(cls, v, info: ValidationInfo) -> int:
        """Computes the definitive port based on the precedence rule."""
        return info.data.get('port') or info.data.get('uvicorn_port')

    # --- Other configuration fields (fully documented) ---
    api_key: SecretStr | None = Field(default=None, description="If set, this API key will be required for all requests via the 'Authorization' header.")
    """
    If set, the API key will be required for all requests.
    """

    log_level: str = Field(default="debug", description="Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.")
    """
    Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.
    """

    allow_origins: list[str] | None = Field(default=None, description="A list of origins that are allowed to make cross-site requests. Use '[\"*\"]' to allow all.")
    """
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#parsing-environment-variable-values
    Usage:
        `export ALLOW_ORIGINS='["http://localhost:3000", "http://localhost:3001"]'`
        `export ALLOW_ORIGINS='["*"]'`
    """

    enable_ui: bool = Field(default=True, description="Enable the Gradio web UI. Disable to reduce dependencies and improve startup time.")
    """
    Whether to enable the Gradio UI. You may want to disable this if you want to minimize the dependencies and slightly improve the startup time.
    """

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    loopback_host_url: str | None = Field(default=None, description="URL for the Gradio app to connect to the API. If unset, it uses the user's browser URL.")

    """
    If set this is the URL that the gradio app will use to connect to the API server hosting speaches.
    If not set the gradio app will use the url that the user connects to the gradio app on.
    """
    # TODO: document the below configuration options
    chat_completion_base_url: str = Field(default="http://localhost:11434/v1", description="The base URL for the chat completion API endpoint (e.g., Ollama).")
    chat_completion_api_key: SecretStr = Field(default=SecretStr("not-required"), description="The API key for the chat completion service, if required.")
    ssl_keyfile: str | None = Field(default=None, description="Path to the SSL private key file for enabling HTTPS.")
    ssl_certfile: str | None = Field(default=None, description="Path to the SSL certificate file for enabling HTTPS.")