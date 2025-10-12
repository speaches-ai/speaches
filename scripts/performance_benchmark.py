import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
import contextlib
from datetime import datetime
import io
import logging
import logging.config
from pathlib import Path
import time
from typing import TypedDict

from httpx import AsyncClient
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.audio import SpeechCreateParams, TranscriptionCreateParams

# from openai.types.audio.transcription_create_params import TranscriptionCreateParamsNonStreaming
from pydantic import BaseModel, ConfigDict, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings
import soundfile as sf

INPUT_TEXT_1 = """You can now select additional permissions when creating an API key to use in any third-party libraries or software that integrate with Immich. This mechanism will give you better control over what the other applications or libraries can do with your Immich’s instance."""  # noqa: RUF001
INPUT_TEXT_2 = """I figured that surely, someone has had this idea and built it before. On eBay you'll find cubes of resin embedding various random components from mechanical watches, but they are typically sold as "steampunk art" and bear little resemblance to the proper assembly of a mechanical watch movement. Sometimes, you'll find resin castings showing every component of a movement spread out in a plane like a buffet---very cool, but not what I'm looking for. Despite my best efforts, I haven't found anyone who makes what I'm after, and I have a sneaking suspicion as to why that is. Building an exploded view of a mechanical watch movement is undoubtedly very fiddly work and requires working knowledge about how a mechanical watch is assembled. People with that skillset are called watchmakers. Maker, not "destroyer for the sake of art". I guess it falls to me, then, to give this project an honest shot. """

logger: logging.Logger


class SpeechBenchmarkResultEntry(BaseModel):
    time_to_first_chunk: float
    average_chunk_time: float
    total_chunks: int
    total_time: float
    input_text_character_count: int


class SpeechBenchmarkResultsSummary(BaseModel):
    total_iterations: int
    total_time: float
    average_time_to_first_chunk: float
    average_chunk_time: float
    total_chunks: int


class SpeechBenchmarkScenarioResults(BaseModel):
    entries: list[SpeechBenchmarkResultEntry] = []

    @computed_field
    @property
    def summary(self) -> SpeechBenchmarkResultsSummary:
        total_iterations = len(self.entries)
        total_time = sum(entry.total_time for entry in self.entries)
        average_time_to_first_chunk = (
            sum(entry.time_to_first_chunk for entry in self.entries) / total_iterations if total_iterations > 0 else 0
        )
        average_chunk_time = (
            sum(entry.average_chunk_time for entry in self.entries) / total_iterations if total_iterations > 0 else 0
        )
        total_chunks = sum(entry.total_chunks for entry in self.entries)

        return SpeechBenchmarkResultsSummary(
            total_iterations=total_iterations,
            total_time=total_time,
            average_time_to_first_chunk=average_time_to_first_chunk,
            average_chunk_time=average_chunk_time,
            total_chunks=total_chunks,
        )


class SpeechBenchmarkScenario(BaseModel):
    request_count: int
    request_concurrency: int
    request_params: SpeechCreateParams

    results: SpeechBenchmarkScenarioResults | None = None


class TranscriptionBenchmarkResultEntry(BaseModel):
    start_time: float
    end_time: float
    file_duration_seconds: float


class TranscriptionBenchmarkResultsSummary(BaseModel):
    total_processing_time_seconds: float
    total_file_duration_seconds: float


class TranscriptionBenchmarkScenarioResults(BaseModel):
    entries: list[TranscriptionBenchmarkResultEntry] = []

    @computed_field
    @property
    def summary(self) -> TranscriptionBenchmarkResultsSummary:
        earliest_start = min(entry.start_time for entry in self.entries) if self.entries else 0
        latest_end = max(entry.end_time for entry in self.entries) if self.entries else 0
        total_time = latest_end - earliest_start
        total_file_duration_seconds = sum(entry.file_duration_seconds for entry in self.entries)

        return TranscriptionBenchmarkResultsSummary(
            total_processing_time_seconds=total_time,
            total_file_duration_seconds=total_file_duration_seconds,
        )


# class TranscriptionRequestParams(BaseModel):
#     file: BufferedReader = Field(exclude=True)
#     model: str
#     language: str | None = None
#     prompt: str | None = None
#
#     model_config = ConfigDict(arbitrary_types_allowed=True)


class TranscriptionBenchmarkScenario(BaseModel):
    request_count: int
    request_concurrency: int
    request_params: TranscriptionCreateParams

    results: TranscriptionBenchmarkScenarioResults | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VadCreateParams(TypedDict):
    model: str
    file: Path


class VadBenchmarkScenario(BaseModel):
    request_count: int
    request_concurrency: int
    request_params: VadCreateParams
    warmup_count: int = 2

    results: TranscriptionBenchmarkScenarioResults | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# with Path("/Users/fedir/code/speaches/audio.wav").open("rb") as f:
#     AUDIO_FILE_CONTENT = f.read()

DEFAULT_SCENARIOS = [
    # SpeechBenchmarkScenario(
    #     request_count=2,
    #     request_concurrency=1,
    #     request_params=SpeechCreateParams(
    #         model="speaches-ai/Kokoro-82M-v1.0-ONNX",
    #         voice="af_heart",
    #         input=INPUT_TEXT_1,
    #     ),
    # ),
    # TranscriptionBenchmarkScenario(
    #     request_count=1,
    #     request_concurrency=1,
    #     request_params=TranscriptionCreateParamsNonStreaming(
    #         # file=Path("/Users/fedir/code/speaches/data/modified_test/BillGates_2010_modified.wav").open("rb"),
    #         file=Path("/Users/fedir/code/speaches/data/modified_test/BillGates_2010_modified_2min.wav"),
    #         # file=Path("/Users/fedir/code/speaches/audio.wav"),
    #         # model="Systran/faster-whisper-tiny.en",
    #         model="istupakov/parakeet-tdt-0.6b-v3-onnx",
    #     ),
    # ),
    VadBenchmarkScenario(
        request_count=40,
        request_concurrency=20,
        warmup_count=2,
        request_params=VadCreateParams(
            # file=Path("/Users/fedir/code/speaches/data/modified_test/BillGates_2010_modified.wav").open("rb"),
            file=Path("/Users/fedir/code/speaches/data/modified_test/BillGates_2010_modified_2min.wav"),
            # file=Path("/Users/fedir/code/speaches/audio.wav"),
            # model="Systran/faster-whisper-tiny.en",
            model="silero_vad_v5",
        ),
    ),
    # BenchmarkScenario(
    #     request_count=2,
    #     request_concurrency=1,
    #     request_params=SpeechRequestParams(
    #         model="speaches-ai/Kokoro-82M-v1.0-ONNX",
    #         voice="af_heart",
    #         input=INPUT_TEXT_2,
    #     ),
    # ),
]


class Config(BaseSettings):
    speaches_base_url: SecretStr = SecretStr("http://localhost:8000")
    api_key: SecretStr = SecretStr("does-not-matter")
    log_level: str = "debug"
    """
    Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.
    """
    output_directory: Path = Path("benchmarks")
    file_prefix: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  # noqa: DTZ005

    scenarios: Sequence[SpeechBenchmarkScenario | TranscriptionBenchmarkScenario | VadBenchmarkScenario]


class Output(BaseModel):
    config: Config


def limit_concurrency[**P, R](
    coro: Callable[P, Coroutine[None, None, R]], limit: int
) -> Callable[P, Coroutine[None, None, R]]:
    semaphore = asyncio.Semaphore(limit)

    async def wrapped_coro(*args: P.args, **kwargs: P.kwargs) -> R:
        async with semaphore:
            return await coro(*args, **kwargs)

    return wrapped_coro


@contextlib.asynccontextmanager
async def model_available_fixture(model_id: str, client: AsyncClient) -> AsyncGenerator[None, None]:
    get_running_models_res = await client.get("/api/ps")
    running_models = get_running_models_res.json()["models"]
    for running_model in running_models:
        if running_model == model_id:
            continue
        stop_model_res = await client.delete(f"/api/ps/{running_model}")
        stop_model_res.raise_for_status()

    if model_id not in running_models:
        get_model_res = await client.get(f"/v1/models/{model_id}")
        if get_model_res.status_code == 404:
            download_model_res = await client.post(f"/v1/models/{model_id}")
            download_model_res.raise_for_status()

        load_model_res = await client.post(f"/api/ps/{model_id}")
        load_model_res.raise_for_status()

    yield

    get_running_models_res = await client.get("/api/ps")
    running_models = get_running_models_res.json()["models"]
    for running_model in running_models:
        stop_model_res = await client.delete(f"/api/ps/{running_model}")
        stop_model_res.raise_for_status()


async def speech_benchmark_runner(
    openai_client: AsyncOpenAI, scenario: SpeechBenchmarkScenario
) -> SpeechBenchmarkScenarioResults:
    entries: list[SpeechBenchmarkResultEntry] = []

    async def create_speech() -> None:
        async with openai_client.audio.speech.with_streaming_response.create(
            input=scenario.request_params["input"],
            model=scenario.request_params["model"],
            voice=scenario.request_params["voice"],  # pyright: ignore[reportArgumentType]
        ) as res:
            chunk_times: list[float] = []
            start = time.perf_counter()
            prev_chunk_time = time.perf_counter()
            async for _ in res.iter_bytes():
                chunk_times.append(time.perf_counter() - prev_chunk_time)
                prev_chunk_time = time.perf_counter()
            stat = SpeechBenchmarkResultEntry(
                time_to_first_chunk=chunk_times[0],
                average_chunk_time=sum(chunk_times) / len(chunk_times),
                total_chunks=len(chunk_times),
                total_time=time.perf_counter() - start,
                input_text_character_count=len(scenario.request_params["input"]),
            )
        entries.append(stat)
        logger.debug(stat.model_dump_json())

    create_speech_with_limited_concurrency = limit_concurrency(create_speech, scenario.request_concurrency)

    start = time.perf_counter()

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(create_speech_with_limited_concurrency()) for _ in range(scenario.request_count)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        logger.info(f"All tasks completed in {time.perf_counter() - start:.2f} seconds")

    return SpeechBenchmarkScenarioResults(
        entries=entries,
    )


async def vad_benchmark_runner(
    openai_client: AsyncOpenAI, scenario: VadBenchmarkScenario
) -> TranscriptionBenchmarkScenarioResults:
    entries: list[TranscriptionBenchmarkResultEntry] = []

    # Pre-load and convert file data to PCM 16-bit
    audio_data, samplerate = sf.read(scenario.request_params["file"], dtype="int16")

    # Calculate actual file duration
    file_duration_seconds = len(audio_data) / samplerate

    # Convert to WAV format in memory (PCM 16-bit)
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, samplerate, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    file_data = buffer.read()

    async def create_vad() -> None:
        start = time.perf_counter()
        _res = await openai_client._client.post(
            "/v1/audio/speech/timestamps",
            data={"model": scenario.request_params["model"]},
            files={"file": file_data},
        )
        end = time.perf_counter()

        stat = TranscriptionBenchmarkResultEntry(
            start_time=start,
            end_time=end,
            file_duration_seconds=file_duration_seconds,
        )

        entries.append(stat)

    # Warmup phase
    logger.info(f"Running {scenario.warmup_count} warmup requests...")
    for _ in range(scenario.warmup_count):
        await create_vad()
    entries.clear()
    logger.info("Warmup complete, starting benchmark...")

    # Actual benchmark
    create_vad_with_limited_concurrency = limit_concurrency(create_vad, scenario.request_concurrency)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(create_vad_with_limited_concurrency()) for _ in range(scenario.request_count)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        logger.info(f"All tasks completed in {time.perf_counter() - start:.2f} seconds")

    return TranscriptionBenchmarkScenarioResults(
        entries=entries,
    )


async def transcription_benchmark_runner(
    openai_client: AsyncOpenAI, scenario: TranscriptionBenchmarkScenario
) -> TranscriptionBenchmarkScenarioResults:
    entries: list[TranscriptionBenchmarkResultEntry] = []

    async def create_transcription() -> None:
        start = time.perf_counter()
        _res = await openai_client.audio.transcriptions.create(
            file=scenario.request_params["file"],
            model=scenario.request_params["model"],
            language=scenario.request_params.get("language") or NOT_GIVEN,
            # prompt=scenario.request_params["prompt"],
        )
        end = time.perf_counter()

        with sf.SoundFile(scenario.request_params["file"]) as f:
            file_duration_seconds = len(f) / f.samplerate

        stat = TranscriptionBenchmarkResultEntry(
            start_time=start,
            end_time=end,
            file_duration_seconds=file_duration_seconds,
        )
        entries.append(stat)
        logger.debug(stat.model_dump_json())

    create_speech_with_limited_concurrency = limit_concurrency(create_transcription, scenario.request_concurrency)

    start = time.perf_counter()

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(create_speech_with_limited_concurrency()) for _ in range(scenario.request_count)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        logger.info(f"All tasks completed in {time.perf_counter() - start:.2f} seconds")

    return TranscriptionBenchmarkScenarioResults(
        entries=entries,
    )


async def main(config: Config) -> None:
    client = AsyncClient(base_url=config.speaches_base_url.get_secret_value(), timeout=180)
    openai_client = AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=f"{config.speaches_base_url.get_secret_value()}/v1",
        http_client=client,
        max_retries=0,
    )

    for scenario in config.scenarios:
        logger.info(f"Running benchmark scenario: {scenario.model_dump_json()}")
        if isinstance(scenario, SpeechBenchmarkScenario):
            async with model_available_fixture(scenario.request_params["model"], client):
                scenario_output = await speech_benchmark_runner(openai_client, scenario)
                scenario.results = scenario_output
        elif isinstance(scenario, TranscriptionBenchmarkScenario):
            async with model_available_fixture(scenario.request_params["model"], client):
                scenario_output = await transcription_benchmark_runner(openai_client, scenario)
                scenario.results = scenario_output
        elif isinstance(scenario, VadBenchmarkScenario):
            scenario_output = await vad_benchmark_runner(openai_client, scenario)
            scenario.results = scenario_output
        else:
            raise TypeError(f"Unknown scenario type: {type(scenario)}")
        logger.info(f"Finished benchmark scenario: {scenario.model_dump_json(indent=2)}")

    output = Output(config=config)
    output_file_path = Path(config.output_directory, f"{config.file_prefix}_output.json")
    with output_file_path.open("w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.write(output.model_dump_json(indent=2))
    logger.info(f"Wrote output to {output_file_path}")


if __name__ == "__main__":
    config = Config(scenarios=DEFAULT_SCENARIOS)

    config.output_directory.mkdir(parents=True, exist_ok=True)
    logging_config = {
        "version": 1,  # required
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": f"{config.output_directory}/{config.file_prefix}_output.log",
                "formatter": "simple",
            },
        },
        "loggers": {
            "root": {
                "level": config.log_level.upper(),
                "handlers": ["stdout", "file"],
            },
            "asyncio": {
                "level": "INFO",
            },
            "httpx": {
                "level": "INFO",
            },
            "python_multipart": {
                "level": "INFO",
            },
            "httpcore": {
                "level": "INFO",
            },
            "openai": {
                "level": "INFO",
            },
        },
    }

    logging.config.dictConfig(logging_config)
    # logging.basicConfig(level=config.log_level.upper())
    logger = logging.getLogger(__name__)
    asyncio.run(main(config))
