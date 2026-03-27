"""Allocation profiler for the speaches realtime pipeline.

Directly imports and exercises each hot-path component with tracemalloc
to measure per-stage memory allocations. No subprocess needed.

Usage:
    PYTHONPATH=src python scripts/allocation_benchmark.py
"""

import asyncio
import base64
import gc
from io import BytesIO
import linecache
import logging
import sys
import tracemalloc

import numpy as np
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("allocation_benchmark")

ITERATIONS = 3
TRACEMALLOC_FRAMES = 5

rng = np.random.default_rng(42)


# ---- Snapshot helpers ----


class AllocationEntry(BaseModel):
    file: str
    line: int
    source_line: str
    size_bytes: int
    count: int


def _shorten_path(path: str) -> str:
    if "speaches/" in path:
        return path.split("speaches/", 1)[-1]
    return path


def snapshot_diff(
    before: tracemalloc.Snapshot,
    after: tracemalloc.Snapshot,
    top_n: int = 20,
) -> list[AllocationEntry]:
    filt = [
        tracemalloc.Filter(True, "*/speaches/*"),  # noqa: FBT003
        tracemalloc.Filter(False, "*tracemalloc*"),  # noqa: FBT003
    ]
    stats = after.filter_traces(filt).compare_to(before.filter_traces(filt), "lineno")
    entries: list[AllocationEntry] = []
    for stat in stats[:top_n]:
        if stat.size_diff <= 0:
            continue
        frame = stat.traceback[0]
        entries.append(
            AllocationEntry(
                file=_shorten_path(frame.filename),
                line=frame.lineno,
                source_line=linecache.getline(frame.filename, frame.lineno).strip(),
                size_bytes=stat.size_diff,
                count=stat.count_diff,
            )
        )
    return entries


def print_table(entries: list[AllocationEntry], title: str) -> None:
    if not entries:
        print(f"\n  {title}: (no allocations)", file=sys.stderr)
        return

    def fmt(size: int) -> str:
        if size >= 1_048_576:
            return f"{size / 1_048_576:.1f} MiB"
        if size >= 1024:
            return f"{size / 1024:.1f} KiB"
        return f"{size} B"

    print(f"\n{'=' * 110}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'=' * 110}", file=sys.stderr)
    print(f"  {'File:Line':<50} {'Size':>10} {'Count':>7}  Source", file=sys.stderr)
    print(f"  {'-' * 105}", file=sys.stderr)

    total_size = 0
    total_count = 0
    for e in entries:
        loc = f"{e.file}:{e.line}"
        src = e.source_line[:55] if e.source_line else ""
        print(f"  {loc:<50} {fmt(e.size_bytes):>10} {e.count:>7}  {src}", file=sys.stderr)
        total_size += e.size_bytes
        total_count += e.count

    print(f"  {'-' * 105}", file=sys.stderr)
    print(f"  {'TOTAL':<50} {fmt(total_size):>10} {total_count:>7}", file=sys.stderr)


class StageResult(BaseModel):
    entries: list[AllocationEntry]
    peak_bytes: int
    alloc_blocks_delta: int


def measure_stage(_name: str, func: object) -> StageResult:
    """Run func() once for warmup, then start fresh tracemalloc, run once, snapshot, stop."""
    # Warmup (no tracing)
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    func()  # type: ignore[operator]
    gc.collect()

    # Measure block delta (lightweight, no tracemalloc)
    blocks_before = sys.getallocatedblocks()
    func()  # type: ignore[operator]
    block_delta = sys.getallocatedblocks() - blocks_before
    gc.collect()

    # Start fresh tracemalloc, run once, snapshot, stop
    tracemalloc.start(TRACEMALLOC_FRAMES)
    func()  # type: ignore[operator]
    _, peak = tracemalloc.get_traced_memory()
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()

    filt = [
        tracemalloc.Filter(True, "*/speaches/*"),  # noqa: FBT003
        tracemalloc.Filter(False, "*tracemalloc*"),  # noqa: FBT003
    ]
    stats = snap.filter_traces(filt).statistics("lineno")
    entries = []
    for stat in stats[:15]:
        frame = stat.traceback[0]
        entries.append(
            AllocationEntry(
                file=_shorten_path(frame.filename),
                line=frame.lineno,
                source_line=linecache.getline(frame.filename, frame.lineno).strip(),
                size_bytes=stat.size,
                count=stat.count,
            )
        )

    return StageResult(entries=entries, peak_bytes=peak, alloc_blocks_delta=block_delta)


async def measure_stage_async(_name: str, func: object) -> StageResult:
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    await func()  # type: ignore[operator]
    gc.collect()

    blocks_before = sys.getallocatedblocks()
    await func()  # type: ignore[operator]
    block_delta = sys.getallocatedblocks() - blocks_before
    gc.collect()

    tracemalloc.start(TRACEMALLOC_FRAMES)
    await func()  # type: ignore[operator]
    _, peak = tracemalloc.get_traced_memory()
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()

    filt = [
        tracemalloc.Filter(True, "*/speaches/*"),  # noqa: FBT003
        tracemalloc.Filter(False, "*tracemalloc*"),  # noqa: FBT003
    ]
    stats = snap.filter_traces(filt).statistics("lineno")
    entries = []
    for stat in stats[:15]:
        frame = stat.traceback[0]
        entries.append(
            AllocationEntry(
                file=_shorten_path(frame.filename),
                line=frame.lineno,
                source_line=linecache.getline(frame.filename, frame.lineno).strip(),
                size_bytes=stat.size,
                count=stat.count,
            )
        )

    return StageResult(entries=entries, peak_bytes=peak, alloc_blocks_delta=block_delta)


# ---- Pipeline stage benchmarks ----


def make_test_audio_chunk(duration_ms: int = 100, sample_rate: int = 24000) -> bytes:
    """Create PCM16 audio bytes (like a WebSocket client would send)."""
    samples = int(sample_rate * duration_ms / 1000)
    data = rng.integers(-1000, 1000, size=samples, dtype=np.int16)
    return data.tobytes()


def bench_base64_decode_and_parse() -> StageResult:
    from speaches.audio import audio_samples_from_file

    raw_pcm = make_test_audio_chunk(100, 24000)
    b64_audio = base64.b64encode(raw_pcm).decode("utf-8")

    def run() -> None:
        audio_samples_from_file(BytesIO(base64.b64decode(b64_audio)), 24000)

    return measure_stage("base64_decode + audio_samples_from_file", run)


def bench_resample() -> StageResult:
    from speaches.audio import audio_samples_from_file, resample_audio_data

    raw_pcm = make_test_audio_chunk(100, 24000)
    audio_chunk = audio_samples_from_file(BytesIO(raw_pcm), 24000)

    def run() -> None:
        resample_audio_data(audio_chunk, 24000, 16000)

    return measure_stage("resample_audio_data (24kHz -> 16kHz)", run)


def bench_input_audio_buffer_append() -> StageResult:
    from speaches.realtime.input_audio_buffer import InputAudioBuffer
    from speaches.realtime.pubsub import EventPubSub

    pubsub = EventPubSub()
    buf = InputAudioBuffer(pubsub)
    chunk = rng.standard_normal(1600).astype(np.float32)

    def run() -> None:
        buf.append(chunk)

    return measure_stage("InputAudioBuffer.append (ring buffer)", run)


def bench_vad_ring_buffer_read() -> StageResult:
    from speaches.realtime.input_audio_buffer import InputAudioBuffer
    from speaches.realtime.pubsub import EventPubSub

    pubsub = EventPubSub()
    buf = InputAudioBuffer(pubsub)
    for _ in range(200):
        buf.append(rng.standard_normal(1600).astype(np.float32))

    def run() -> None:
        _ = buf.vad_data

    return measure_stage("InputAudioBuffer.vad_data (ring concat)", run)


def bench_audio_buffer_consolidate() -> StageResult:
    from speaches.realtime.input_audio_buffer import InputAudioBuffer
    from speaches.realtime.pubsub import EventPubSub

    def run() -> None:
        pubsub = EventPubSub()
        buf = InputAudioBuffer(pubsub)
        for _ in range(20):
            buf.append(rng.standard_normal(1600).astype(np.float32))
        buf.consolidate()

    return measure_stage("InputAudioBuffer.consolidate (20 chunks)", run)


def bench_audio_as_bytes() -> StageResult:
    from speaches.audio import Audio

    data = rng.standard_normal(24000).astype(np.float32)
    audio = Audio(data, sample_rate=24000)

    def run() -> None:
        audio.as_bytes()

    return measure_stage("Audio.as_bytes (1s, 24kHz)", run)


def bench_audio_base64_encode() -> StageResult:
    from speaches.audio import Audio

    data = rng.standard_normal(24000).astype(np.float32)
    audio = Audio(data, sample_rate=24000)

    def run() -> None:
        audio_bytes = audio.as_bytes()
        base64.b64encode(audio_bytes).decode("utf-8")

    return measure_stage("as_bytes + base64 encode (1s, 24kHz)", run)


def bench_audio_resample_output() -> StageResult:
    from speaches.audio import Audio

    data = rng.standard_normal(24000).astype(np.float32)

    def run() -> None:
        a = Audio(data.copy(), sample_rate=22050)
        a.resample(24000)

    return measure_stage("Audio.resample (22050 -> 24kHz, 1s)", run)


def bench_pubsub_publish() -> StageResult:
    from speaches.realtime.pubsub import EventPubSub
    from speaches.types.realtime import InputAudioBufferSpeechStartedEvent

    pubsub = EventPubSub()
    _sub = pubsub.subscribe()
    event = InputAudioBufferSpeechStartedEvent(item_id="test-123", audio_start_ms=100)

    def run() -> None:
        pubsub.publish_nowait(event)

    return measure_stage("EventPubSub.publish_nowait (1 subscriber)", run)


def bench_event_model_copy() -> StageResult:
    from speaches.types.realtime import ResponseAudioDeltaEvent

    event = ResponseAudioDeltaEvent(
        item_id="item-123",
        response_id="resp-456",
        delta=base64.b64encode(b"\x00" * 4800).decode("utf-8"),
    )

    def run() -> None:
        event.model_copy()

    return measure_stage("ResponseAudioDeltaEvent.model_copy (200ms audio)", run)


def bench_event_model_dump_json() -> StageResult:
    from speaches.types.realtime import ResponseAudioDeltaEvent

    event = ResponseAudioDeltaEvent(
        item_id="item-123",
        response_id="resp-456",
        delta=base64.b64encode(b"\x00" * 4800).decode("utf-8"),
    )

    def run() -> None:
        event.model_dump_json()

    return measure_stage("ResponseAudioDeltaEvent.model_dump_json (200ms audio)", run)


def bench_phrase_chunker() -> StageResult:
    from speaches.text_utils import PhraseChunker

    tokens = ["I ", "am ", "doing ", "great, ", "thank ", "you ", "for ", "asking!"]

    async def run() -> None:
        chunker = PhraseChunker()
        for token in tokens:
            chunker.add_token(token)
        chunker.close()
        async for _ in chunker:
            pass

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(measure_stage_async("PhraseChunker (8 tokens)", run))
    loop.close()
    return result


def bench_clean_for_tts() -> StageResult:
    from speaches.text_utils import clean_for_tts

    text = "I am doing great, thank you for asking!"

    def run() -> None:
        clean_for_tts(text)

    return measure_stage("clean_for_tts", run)


# ---- Main ----


def main() -> None:
    stages = [
        ("1. base64 decode + audio parse", bench_base64_decode_and_parse),
        ("2. resample 24kHz -> 16kHz", bench_resample),
        ("3. InputAudioBuffer.append", bench_input_audio_buffer_append),
        ("4. VAD ring buffer read", bench_vad_ring_buffer_read),
        ("5. buffer consolidate (20 chunks)", bench_audio_buffer_consolidate),
        ("6. Audio.as_bytes (float->int16->bytes)", bench_audio_as_bytes),
        ("7. as_bytes + base64 encode (output)", bench_audio_base64_encode),
        ("8. Audio.resample (22050->24kHz)", bench_audio_resample_output),
        ("9. PubSub.publish_nowait", bench_pubsub_publish),
        ("10. model_copy (audio delta event)", bench_event_model_copy),
        ("11. model_dump_json (audio delta event)", bench_event_model_dump_json),
        ("12. PhraseChunker (8 tokens)", bench_phrase_chunker),
        ("13. clean_for_tts", bench_clean_for_tts),
    ]

    print("\n" + "=" * 110, file=sys.stderr)
    print("  REALTIME PIPELINE ALLOCATION PROFILE (per-call averages)", file=sys.stderr)
    print("=" * 110, file=sys.stderr)

    all_results: list[tuple[str, StageResult]] = []

    for stage_name, bench_func in stages:
        logger.info(f"Profiling: {stage_name}")
        result = bench_func()
        all_results.append((stage_name, result))
        print_table(result.entries, stage_name)

    # Summary table
    print(f"\n\n{'=' * 90}", file=sys.stderr)
    print("  SUMMARY: Per-call allocation by pipeline stage", file=sys.stderr)
    print(f"{'=' * 90}", file=sys.stderr)
    print(f"  {'Stage':<50} {'Traced':>10} {'Allocs':>8} {'Peak':>12}", file=sys.stderr)
    print(f"  {'-' * 85}", file=sys.stderr)

    def fmt(size: int) -> str:
        if size >= 1_048_576:
            return f"{size / 1_048_576:.1f} MiB"
        if size >= 1024:
            return f"{size / 1024:.1f} KiB"
        return f"{size} B"

    for stage_name, result in all_results:
        total_size = sum(e.size_bytes for e in result.entries)
        total_count = sum(e.count for e in result.entries)
        print(
            f"  {stage_name:<50} {fmt(total_size):>10} {total_count:>8} {fmt(result.peak_bytes):>12}",
            file=sys.stderr,
        )



if __name__ == "__main__":
    main()
