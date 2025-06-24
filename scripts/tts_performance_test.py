import asyncio
from collections.abc import Callable, Coroutine
import logging
import logging.config
from pathlib import Path
import time

from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings

INPUT_TEXT_1 = """You can now select additional permissions when creating an API key to use in any third-party libraries or software that integrate with Immich. This mechanism will give you better control over what the other applications or libraries can do with your Immich’s instance."""  # noqa: RUF001
INPUT_TEXT_2 = """I figured that surely, someone has had this idea and built it before. On eBay you'll find cubes of resin embedding various random components from mechanical watches, but they are typically sold as "steampunk art" and bear little resemblance to the proper assembly of a mechanical watch movement. Sometimes, you'll find resin castings showing every component of a movement spread out in a plane like a buffet---very cool, but not what I'm looking for. Despite my best efforts, I haven't found anyone who makes what I'm after, and I have a sneaking suspicion as to why that is. Building an exploded view of a mechanical watch movement is undoubtedly very fiddly work and requires working knowledge about how a mechanical watch is assembled. People with that skillset are called watchmakers. Maker, not "destroyer for the sake of art". I guess it falls to me, then, to give this project an honest shot. """


class Config(BaseSettings):
    api_key: SecretStr = SecretStr("does-not-matter")
    log_level: str = "debug"
    """
    Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.
    """
    logs_directory: str = "logs"

    speaches_base_url: SecretStr = SecretStr("http://localhost:8000")
    speech_model_id: str = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    voice_id: str = "af_heart"
    input_text: str = INPUT_TEXT_2

    iterations: int = 5
    """
    The number of iterations to run the performance test.
    """
    concurrency: int = 1
    """
    Maximum number of concurrent requests made to the API.
    """


def limit_concurrency[**P, R](
    coro: Callable[P, Coroutine[None, None, R]], limit: int
) -> Callable[P, Coroutine[None, None, R]]:
    semaphore = asyncio.Semaphore(limit)

    async def wrapped_coro(*args: P.args, **kwargs: P.kwargs) -> R:
        async with semaphore:
            return await coro(*args, **kwargs)

    return wrapped_coro


async def main(config: Config) -> None:
    logger = logging.getLogger(__name__)
    logger.debug("Config: %s", config.model_dump_json())
    client = AsyncClient(
        base_url=config.speaches_base_url.get_secret_value(),
        headers={"Authorization": f"Bearer {config.api_key.get_secret_value()}"},
    )
    oai_client = AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=f"{config.speaches_base_url.get_secret_value()}/v1",
        http_client=client,
    )

    logger.debug(f"Attempting to pull model {config.speech_model_id}")
    res = await client.post(f"{config.speaches_base_url}/v1/models/{config.speech_model_id}")
    logger.info(f"Finished attempting to pull model {config.speech_model_id}. Response: {res.text}")

    # INFO: Make initial request so that the model is loaded into memory.
    await oai_client.audio.speech.create(
        input="Hello",
        model=config.speech_model_id,
        voice=config.voice_id,  # type: ignore  # noqa: PGH003
    )

    async def create_speech() -> None:
        async with oai_client.audio.speech.with_streaming_response.create(
            input=config.input_text,
            model=config.speech_model_id,
            voice=config.voice_id,  # type: ignore  # noqa: PGH003
        ) as res:
            chunk_times: list[float] = []
            start = time.perf_counter()
            prev_chunk_time = time.perf_counter()
            async for _ in res.iter_bytes():
                chunk_times.append(time.perf_counter() - prev_chunk_time)
                prev_chunk_time = time.perf_counter()
            stats = {
                "time_to_first_token": chunk_times[0],
                "average_chunk_time": sum(chunk_times) / len(chunk_times),
                "total_chunks": len(chunk_times),
                "total_time": time.perf_counter() - start,
            }
            logger.debug(stats)

    create_speech_with_limited_concurrency = limit_concurrency(create_speech, config.concurrency)

    start = time.perf_counter()

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(create_speech_with_limited_concurrency()) for _ in range(config.iterations)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        logger.info(f"All tasks completed in {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    config = Config()

    Path.mkdir(Path(config.logs_directory), exist_ok=True)
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
                "filename": f"{config.logs_directory}/{time.strftime('%Y-%m-%d_%H-%M-%S')}_tts_performance_test.log",  # TODO: there's a better way to do this, but this is good enough for now
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
                "handlers": ["stdout"],
            },
            "httpx": {
                "level": "WARNING",
                "handlers": ["stdout"],
            },
            "python_multipart": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "httpcore": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
            "openai": {
                "level": "INFO",
                "handlers": ["stdout"],
            },
        },
    }

    logging.config.dictConfig(logging_config)
    logging.basicConfig(level=config.log_level.upper())
    asyncio.run(main(config))
