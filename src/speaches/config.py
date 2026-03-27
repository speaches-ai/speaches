from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from speaches import DEFAULT_GPU_MEM_LIMIT

type Device = Literal["cpu", "cuda", "auto"]

# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md#quantize-on-model-conversion
type Quantization = Literal[
    "auto",
    "int8",
    "int8_float16",
    "int8_bfloat16",
    "int8_float32",
    "int16",
    "float16",
    "bfloat16",
    "float32",
    "default",
]


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    inference_device: Device = "auto"
    device_index: int | list[int] = 0
    compute_type: Quantization = "int8"
    cpu_threads: int = 0
    num_workers: int = 3
    flash_attention: bool = False
    max_queued_batches: int = 0
    tensor_parallel: bool = False
    batch_size: int = Field(default=4, ge=1)
    """
    Number of audio segments to decode in a single GPU forward pass.
    Lower values use less VRAM but are slower. On CUDA OOM, the system
    automatically retries with batch_size=1.
    """
    max_concurrency: int = Field(default=1, ge=1)
    """
    Maximum number of concurrent Whisper inference requests.
    A value of 1 prevents CUDA OOM errors from concurrent GPU usage.
    Increase only if GPU VRAM is sufficient for parallel inference.
    """


class OrtOptions(BaseModel):
    exclude_providers: list[str] = ["TensorrtExecutionProvider"]
    """
    List of ORT providers to exclude from the inference session.
    """
    provider_priority: dict[str, int] = {"CUDAExecutionProvider": 100}
    """
    Dictionary of ORT providers and their priority. The higher the value, the higher the priority. Default priority for a provider if not specified is 0.
    """
    provider_opts: dict[str, dict[str, Any]] = {
        "CUDAExecutionProvider": {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "cudnn_conv_algo_search": "HEURISTIC",
        },
    }
    """
    Dictionary of ORT provider options. The keys are provider names, and the values are dictionaries of options.
    Example: {"CUDAExecutionProvider": {"cudnn_conv_algo_search": "DEFAULT"}}
    """
    gpu_mem_limit: int | None = None
    """
    GPU memory limit in bytes for ONNX Runtime's CUDA execution provider arena.
    Populated from Config.gpu_mem_limit at startup. None means no limit.
    """
    enable_cpu_mem_arena: bool = False
    """
    Whether to enable the CPU memory arena in ONNX Runtime sessions.
    Disabling reduces peak CPU memory usage.
    """
    enable_mem_pattern: bool = True
    """
    Whether to enable memory pattern optimization in ONNX Runtime sessions.
    """


# TODO: document `alias` behaviour within the docstring
class Config(BaseSettings):
    """Configuration for the application. Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__INFERENCE_DEVICE`(note the double underscore) to `whisper.inference_device`, to set quantization to int8, use `WHISPER__COMPUTE_TYPE=int8`, etc.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    stt_model_ttl: int = Field(default=-1, ge=-1)
    """
    Time in seconds until a speech to text (stt) model is unloaded after last usage.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """

    tts_model_ttl: int = Field(default=-1, ge=-1)
    """
    Time in seconds until a text to speech (tts) model is unloaded after last usage.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """

    vad_model_ttl: int = Field(default=-1, ge=-1)
    """
    Time in seconds until a voice activation detection (VAD) model is unloaded after last usage.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """

    vad_model: str = "silero_vad_v6"
    """
    Which Silero VAD model to use. Options: 'silero_vad_v5', 'silero_vad_v6'.
    Only the selected model is loaded on startup.
    """

    api_key: SecretStr | None = None
    """
    If set, the API key will be required for all API requests.
    The following endpoints remain publicly accessible without authentication:
    - /health (health check endpoint)
    - /docs (API documentation)
    - /openapi.json (OpenAPI schema)
    - Web UI (Gradio interface)
    """
    log_level: str = "debug"
    """
    Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.
    """
    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(alias="UVICORN_PORT", default=8000)
    allow_origins: list[str] | None = None
    """
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#parsing-environment-variable-values
    Usage:
        `export ALLOW_ORIGINS='["http://localhost:3000", "http://localhost:3001"]'`
        `export ALLOW_ORIGINS='["*"]'`
    """

    enable_ui: bool = False
    """
    Whether to enable the Gradio UI. You may want to disable this if you want to minimize the dependencies and slightly improve the startup time.
    """

    whisper: WhisperConfig = WhisperConfig()

    # TODO: remove the underscore prefix from the field name
    _unstable_vad_filter: bool = True
    """
    Default value for VAD (Voice Activity Detection) filter in speech recognition endpoints.
    When enabled, the model will filter out non-speech segments. Useful for removing hallucinations in speech recognition caused by background silences.


    NOTE: having `_unstable_vad_filter: True` technically deviates from the OpenAI API specification, so you may want to set it to `False`.

    NOTE: This is an unstable feature and may change in the future.
    """

    loopback_host_url: str | None = None
    """
    If set this is the URL that the gradio app will use to connect to the API server hosting speaches.
    If not set the gradio app will use the url that the user connects to the gradio app on.
    """

    # TODO: document the below configuration options
    chat_completion_base_url: str = "http://localhost:11434/v1"
    chat_completion_api_key: SecretStr = SecretStr("cant-be-empty")

    unstable_ort_opts: OrtOptions = OrtOptions()

    otel_exporter_otlp_endpoint: str | None = None
    """
    OpenTelemetry OTLP exporter endpoint. If set, telemetry will be enabled.
    Example: 'http://localhost:4317'
    Shadows OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
    """

    otel_service_name: str = "speaches"
    """
    OpenTelemetry service name for identifying this application in traces.
    Shadows OTEL_SERVICE_NAME environment variable.
    """

    preload_models: list[str] = []
    """
    List of model IDs to download during application startup.
    Models will be downloaded sequentially if they do not already exist locally.
    Application will exit if any model fails to download or is not found in the registry.
    Example: ["Systran/faster-whisper-tiny", "rhasspy/piper-voices"]
    """

    gpu_mem_limit: int = DEFAULT_GPU_MEM_LIMIT
    """
    GPU memory limit in bytes shared across inference backends (512MB default).
    Controls both ONNX Runtime CUDA arena size and CTranslate2 caching allocator limit.
    Set via GPU_MEM_LIMIT environment variable.
    """

    default_realtime_stt_model: str = "Systran/faster-distil-whisper-small.en"
    """
    Default speech-to-text model used for the realtime WebSocket/WebRTC API
    when no explicit transcription_model is provided by the client.
    Set via DEFAULT_REALTIME_STT_MODEL environment variable.
    """

    warmup_all_local_models: bool = True
    """
    Whether to automatically load all locally cached models into memory at startup.
    When False, only models listed in preload_models are loaded on startup.
    Other models are loaded on first request.
    """
