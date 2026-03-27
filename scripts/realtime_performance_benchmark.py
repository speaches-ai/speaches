import asyncio
import atexit
import base64
import json
import logging
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI
import httpx
from pydantic import BaseModel, computed_field
from pydantic_settings import BaseSettings
from starlette.requests import Request
from starlette.responses import StreamingResponse
import uvicorn
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("realtime_benchmark")


# ---- Config & Result Models ----


class BenchmarkConfig(BaseSettings):
    speaches_bin: str | None = None
    speaches_port: int = 18000
    mock_llm_port: int = 18001
    mock_llm_token_delay: float = 0.02
    tts_voice: str = "af_heart"
    stt_model: str = "Systran/faster-whisper-base"
    input_text: str = "Hello, how are you today?"
    mock_response: str = "I am doing great, thank you for asking!"
    iterations: int = 5
    concurrent_sessions: int = 1
    warmup_rounds: int = 1
    timeout: float = 120.0


class RoundResult(BaseModel):
    iteration: int
    stt_latency: float
    llm_first_audio: float
    full_response: float
    voice_to_voice: float
    transcript: str
    audio_duration_s: float


class BargeInResult(BaseModel):
    iteration: int
    time_to_cancellation: float


class MetricSummary(BaseModel):
    min: float
    mean: float
    p50: float
    p95: float
    max: float


class BenchmarkSummary(BaseModel):
    stt_latency: MetricSummary
    llm_first_audio: MetricSummary
    full_response: MetricSummary
    voice_to_voice: MetricSummary


class BenchmarkResults(BaseModel):
    config: BenchmarkConfig
    rounds: list[RoundResult]
    concurrency_results: list[RoundResult] | None = None
    barge_in_results: list[BargeInResult] | None = None

    @computed_field
    @property
    def summary(self) -> BenchmarkSummary | None:
        if not self.rounds:
            return None
        return BenchmarkSummary(
            stt_latency=_summarize([r.stt_latency for r in self.rounds]),
            llm_first_audio=_summarize([r.llm_first_audio for r in self.rounds]),
            full_response=_summarize([r.full_response for r in self.rounds]),
            voice_to_voice=_summarize([r.voice_to_voice for r in self.rounds]),
        )

    @computed_field
    @property
    def concurrency_summary(self) -> BenchmarkSummary | None:
        if not self.concurrency_results:
            return None
        return BenchmarkSummary(
            stt_latency=_summarize([r.stt_latency for r in self.concurrency_results]),
            llm_first_audio=_summarize([r.llm_first_audio for r in self.concurrency_results]),
            full_response=_summarize([r.full_response for r in self.concurrency_results]),
            voice_to_voice=_summarize([r.voice_to_voice for r in self.concurrency_results]),
        )


def _summarize(values: list[float]) -> MetricSummary:
    sorted_values = sorted(values)
    return MetricSummary(
        min=sorted_values[0],
        mean=statistics.mean(sorted_values),
        p50=statistics.median(sorted_values),
        p95=sorted_values[int(0.95 * (len(sorted_values) - 1))],
        max=sorted_values[-1],
    )


# ---- Mock LLM Server ----

_mock_config: BenchmarkConfig | None = None
_use_slow_response = False
mock_app = FastAPI()

SLOW_RESPONSE = (
    "This is a very long response that takes a while to stream out word by word "
    "so that we have time to interrupt it during generation. "
    "It keeps going and going with many words to fill up time. "
    "We need enough words here to ensure the streaming takes several seconds "
    "so the barge-in test has time to fire its interrupt."
)


@mock_app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    body = await request.json()
    logger.debug(f"Mock LLM received request with {len(body.get('messages', []))} messages")

    assert _mock_config is not None
    response_text = SLOW_RESPONSE if _use_slow_response else _mock_config.mock_response
    token_delay = 0.05 if _use_slow_response else _mock_config.mock_llm_token_delay

    async def generate():  # noqa: ANN202
        words = response_text.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": "mock-llm",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": token} if i == 0 else {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(token_delay)

        chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-llm",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-llm",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def start_mock_llm(port: int) -> None:
    uvicorn.run(mock_app, host="127.0.0.1", port=port, log_level="warning")


# ---- WebSocket Helpers ----


async def send_audio_chunks(ws: websockets.ClientConnection, audio_pcm: bytes, chunk_size: int = 9600) -> int:
    chunks_sent = 0
    for i in range(0, len(audio_pcm), chunk_size):
        chunk = audio_pcm[i : i + chunk_size]
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }
        await ws.send(json.dumps(event))
        chunks_sent += 1
        await asyncio.sleep(0.01)
    return chunks_sent


async def send_silence(ws: websockets.ClientConnection, duration_seconds: float = 3.5) -> None:
    chunk_size = 9600
    silence_samples = int(24000 * duration_seconds)
    silence_bytes = b"\x00" * (silence_samples * 2)
    for i in range(0, len(silence_bytes), chunk_size):
        chunk = silence_bytes[i : i + chunk_size]
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }
        await ws.send(json.dumps(event))
        await asyncio.sleep(0.01)


async def collect_events_until(
    ws: websockets.ClientConnection,
    stop_type: str,
    timeout_seconds: float = 120,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                events.append(event)
                if event["type"] == stop_type:
                    break
    except TimeoutError:
        event_types = [e["type"] for e in events]
        raise TimeoutError(
            f"Timed out after {timeout_seconds}s waiting for {stop_type!r}. Got {len(events)} events: {event_types}"
        ) from None
    return events


# ---- Core Measurement ----


async def generate_audio(client: httpx.AsyncClient, config: BenchmarkConfig) -> bytes:
    response = await client.post(
        f"http://127.0.0.1:{config.speaches_port}/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": config.input_text,
            "voice": config.tts_voice,
            "response_format": "pcm",
            "sample_rate": 24000,
        },
    )
    response.raise_for_status()
    logger.info(f"Generated {len(response.content)} bytes of 24kHz 16-bit PCM audio")
    return response.content


async def run_single_round(
    ws: websockets.ClientConnection,
    audio_pcm: bytes,
    config: BenchmarkConfig,
    iteration: int,
) -> RoundResult:
    audio_duration_s = len(audio_pcm) / (24000 * 2)

    await send_audio_chunks(ws, audio_pcm)
    await send_silence(ws)

    t_speech_stopped: float | None = None
    t_transcription_done: float | None = None
    t_first_audio_delta: float | None = None
    t_response_done: float | None = None
    transcript = ""

    try:
        async with asyncio.timeout(config.timeout):
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                etype = event["type"]
                now = time.perf_counter()

                if etype == "input_audio_buffer.speech_stopped":
                    t_speech_stopped = now
                elif etype == "conversation.item.input_audio_transcription.completed":
                    t_transcription_done = now
                    transcript = event.get("transcript", "")
                elif etype == "response.audio.delta" and t_first_audio_delta is None:
                    t_first_audio_delta = now
                elif etype == "response.done":
                    t_response_done = now
                    break
    except TimeoutError:
        raise TimeoutError(f"Iteration {iteration}: timed out after {config.timeout}s") from None

    if (
        t_speech_stopped is None
        or t_transcription_done is None
        or t_first_audio_delta is None
        or t_response_done is None
    ):
        missing = []
        if t_speech_stopped is None:
            missing.append("speech_stopped")
        if t_transcription_done is None:
            missing.append("transcription_completed")
        if t_first_audio_delta is None:
            missing.append("first audio delta")
        if t_response_done is None:
            missing.append("response_done")
        raise RuntimeError(f"Iteration {iteration}: missing timing events: {missing}")

    stt_latency = t_transcription_done - t_speech_stopped
    llm_first_audio = t_first_audio_delta - t_transcription_done
    full_response = t_response_done - t_first_audio_delta
    voice_to_voice = t_first_audio_delta - t_speech_stopped

    return RoundResult(
        iteration=iteration,
        stt_latency=round(stt_latency, 4),
        llm_first_audio=round(llm_first_audio, 4),
        full_response=round(full_response, 4),
        voice_to_voice=round(voice_to_voice, 4),
        transcript=transcript,
        audio_duration_s=round(audio_duration_s, 3),
    )


async def run_barge_in_round(
    ws: websockets.ClientConnection,
    audio_pcm: bytes,
    config: BenchmarkConfig,
    iteration: int,
) -> BargeInResult:
    global _use_slow_response  # noqa: PLW0603
    _use_slow_response = True

    await send_audio_chunks(ws, audio_pcm)
    await send_silence(ws)

    # Wait for response to start streaming
    while True:
        raw = await ws.recv()
        event = json.loads(raw)
        if event["type"] == "response.audio.delta":
            break

    # Now send speech to trigger barge-in
    t_barge_start = time.perf_counter()
    await send_audio_chunks(ws, audio_pcm)
    await send_silence(ws, duration_seconds=1.0)

    # Collect until we see speech_started (barge-in detection)
    t_cancellation: float | None = None
    try:
        async with asyncio.timeout(config.timeout):
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                if event["type"] == "input_audio_buffer.speech_started":
                    t_cancellation = time.perf_counter()
                    break
    except TimeoutError:
        raise TimeoutError(f"Barge-in iteration {iteration}: timed out waiting for speech_started") from None

    # Drain remaining events from this round
    try:
        async with asyncio.timeout(10):
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                if event["type"] == "response.done":
                    break
    except TimeoutError:
        pass

    _use_slow_response = False

    return BargeInResult(
        iteration=iteration,
        time_to_cancellation=round(t_cancellation - t_barge_start, 4),
    )


# ---- Server Startup ----


def wait_for_server(url: str, name: str, max_attempts: int = 30) -> None:
    for attempt in range(max_attempts):
        try:
            resp = httpx.get(url)
            logger.info(f"{name} is running (status {resp.status_code})")
            return
        except httpx.ConnectError:
            if attempt == max_attempts - 1:
                logger.warning(f"Failed to connect to {name} after {max_attempts} attempts")
                sys.exit(1)
            time.sleep(0.5)


def start_servers(config: BenchmarkConfig) -> None:
    global _mock_config  # noqa: PLW0603
    _mock_config = config

    # Start mock LLM in background thread
    logger.info(f"Starting mock LLM on port {config.mock_llm_port}")
    mock_thread = threading.Thread(target=start_mock_llm, args=(config.mock_llm_port,), daemon=True)
    mock_thread.start()
    wait_for_server(f"http://127.0.0.1:{config.mock_llm_port}/docs", "Mock LLM")

    # Start speaches as a subprocess via uvicorn CLI
    speaches_env = {
        **os.environ,
        "CHAT_COMPLETION_BASE_URL": f"http://127.0.0.1:{config.mock_llm_port}/v1",
        "LOOPBACK_HOST_URL": f"http://127.0.0.1:{config.speaches_port}",
        "ENABLE_UI": "false",
        "LOG_LEVEL": "warning",
        "UVICORN_HOST": "127.0.0.1",
        "UVICORN_PORT": str(config.speaches_port),
    }

    # Try explicit binary, then PATH lookup, then uvicorn fallback
    speaches_bin = config.speaches_bin or shutil.which("speaches")
    if speaches_bin:
        cmd = [speaches_bin, "--host", "127.0.0.1", "--port", str(config.speaches_port)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "speaches.main:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(config.speaches_port),
        ]
        # Ensure speaches package is importable via PYTHONPATH
        src_dir = str(pathlib.Path(__file__).resolve().parent.parent / "src")
        speaches_env["PYTHONPATH"] = src_dir + os.pathsep + speaches_env.get("PYTHONPATH", "")

    logger.info(f"Starting speaches on port {config.speaches_port}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=speaches_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cleanup_speaches() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    atexit.register(cleanup_speaches)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    wait_for_server(f"http://127.0.0.1:{config.speaches_port}/docs", "Speaches", max_attempts=120)


# ---- Orchestrator ----


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    start_servers(config)

    ws_url = f"ws://127.0.0.1:{config.speaches_port}/v1/realtime?model=mock-llm&transcription_model={config.stt_model}"

    async with httpx.AsyncClient(timeout=180.0) as client:
        audio_pcm = await generate_audio(client, config)

    # -- Sequential rounds --
    rounds: list[RoundResult] = []
    async with websockets.connect(ws_url) as ws:
        session_msg = json.loads(await ws.recv())
        assert session_msg["type"] == "session.created", f"Expected session.created, got {session_msg['type']}"

        # Warmup
        for i in range(config.warmup_rounds):
            logger.info(f"Warmup round {i + 1}/{config.warmup_rounds}")
            await run_single_round(ws, audio_pcm, config, iteration=-1)

        # Measured rounds
        for i in range(config.iterations):
            logger.info(f"Round {i + 1}/{config.iterations}")
            result = await run_single_round(ws, audio_pcm, config, iteration=i + 1)
            rounds.append(result)
            logger.info(
                f"  STT={result.stt_latency:.3f}s  LLM+TTS={result.llm_first_audio:.3f}s  "
                f"Full={result.full_response:.3f}s  V2V={result.voice_to_voice:.3f}s  "
                f"transcript={result.transcript!r}"
            )

    # -- Concurrent sessions --
    concurrency_results: list[RoundResult] | None = None
    if config.concurrent_sessions > 1:
        logger.info(f"Running {config.concurrent_sessions} concurrent sessions")
        concurrency_results = []

        async def run_concurrent_session(session_id: int) -> list[RoundResult]:
            session_rounds: list[RoundResult] = []
            async with websockets.connect(ws_url) as ws:
                session_msg = json.loads(await ws.recv())
                assert session_msg["type"] == "session.created"
                for i in range(config.iterations):
                    result = await run_single_round(ws, audio_pcm, config, iteration=session_id * 100 + i + 1)
                    session_rounds.append(result)
                    logger.info(f"  Session {session_id} round {i + 1}: V2V={result.voice_to_voice:.3f}s")
            return session_rounds

        session_results = await asyncio.gather(*[run_concurrent_session(s) for s in range(config.concurrent_sessions)])
        for sr in session_results:
            concurrency_results.extend(sr)

    # -- Barge-in --
    barge_in_results: list[BargeInResult] | None = None
    if config.iterations > 0:
        logger.info("Running barge-in test")
        barge_in_results = []
        async with websockets.connect(ws_url) as ws:
            session_msg = json.loads(await ws.recv())
            assert session_msg["type"] == "session.created"
            try:
                result = await run_barge_in_round(ws, audio_pcm, config, iteration=1)
                barge_in_results.append(result)
                logger.info(f"  Barge-in cancellation latency: {result.time_to_cancellation:.3f}s")
            except Exception:
                logger.exception("Barge-in test failed")
                barge_in_results = None

    return BenchmarkResults(
        config=config,
        rounds=rounds,
        concurrency_results=concurrency_results,
        barge_in_results=barge_in_results,
    )


# ---- Output ----


def print_table(results: BenchmarkResults) -> None:
    print("\n=== Realtime WebSocket Benchmark Results ===\n", file=sys.stderr)

    print(f"  Iterations: {len(results.rounds)}", file=sys.stderr)
    print(f"  Input text: {results.config.input_text!r}", file=sys.stderr)
    print(f"  Mock response: {results.config.mock_response!r}", file=sys.stderr)
    print(f"  Mock LLM token delay: {results.config.mock_llm_token_delay}s", file=sys.stderr)
    print(file=sys.stderr)

    if results.summary:
        s = results.summary
        header = f"{'Metric':<20} {'Min':>8} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}"
        print(header, file=sys.stderr)
        print("-" * len(header), file=sys.stderr)
        for name, metric in [
            ("STT Latency", s.stt_latency),
            ("LLM First Audio", s.llm_first_audio),
            ("Full Response", s.full_response),
            ("Voice-to-Voice", s.voice_to_voice),
        ]:
            print(
                f"{name:<20} {metric.min:>7.3f}s {metric.mean:>7.3f}s {metric.p50:>7.3f}s "
                f"{metric.p95:>7.3f}s {metric.max:>7.3f}s",
                file=sys.stderr,
            )

    if results.concurrency_summary:
        print(f"\n--- Concurrent ({results.config.concurrent_sessions} sessions) ---", file=sys.stderr)
        s = results.concurrency_summary
        header = f"{'Metric':<20} {'Min':>8} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}"
        print(header, file=sys.stderr)
        print("-" * len(header), file=sys.stderr)
        for name, metric in [
            ("STT Latency", s.stt_latency),
            ("LLM First Audio", s.llm_first_audio),
            ("Full Response", s.full_response),
            ("Voice-to-Voice", s.voice_to_voice),
        ]:
            print(
                f"{name:<20} {metric.min:>7.3f}s {metric.mean:>7.3f}s {metric.p50:>7.3f}s "
                f"{metric.p95:>7.3f}s {metric.max:>7.3f}s",
                file=sys.stderr,
            )

    if results.barge_in_results:
        print("\n--- Barge-in ---", file=sys.stderr)
        for r in results.barge_in_results:
            print(f"  Cancellation latency: {r.time_to_cancellation:.3f}s", file=sys.stderr)

    print(file=sys.stderr)


def main() -> None:
    config = BenchmarkConfig()
    logger.info(f"Benchmark config: {config.model_dump_json(indent=2)}")

    results = asyncio.run(run_benchmark(config))
    print_table(results)
    print(results.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
