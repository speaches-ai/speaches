from collections.abc import Generator
import io
import queue
import shutil
import subprocess
import threading
from typing import Any

import numpy as np
import pytest

from speaches.audio import Audio, stream_audio_as_formatted_bytes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")


@pytest.mark.parametrize("wait_for_backpressure", [False, True])
def test_closing_encoded_audio_stream_reaps_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, wait_for_backpressure: bool
) -> None:
    processes: list[subprocess.Popen[bytes]] = []
    threads: list[threading.Thread] = []
    real_popen = subprocess.Popen
    real_thread = threading.Thread
    stop = threading.Event()
    backpressure = threading.Event()

    class ObservedQueue(queue.Queue[Any]):
        def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
            if self.maxsize == 10 and self.full():
                backpressure.set()
            super().put(item, block=block, timeout=timeout)

    samples = np.random.default_rng(0).uniform(-0.5, 0.5, 16000).astype(np.float32)

    def track_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def track_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    def source() -> Generator[Audio]:
        while not stop.is_set():
            yield Audio(samples, sample_rate=16000)

    monkeypatch.setattr(queue, "Queue", ObservedQueue)
    monkeypatch.setattr(subprocess, "Popen", track_popen)
    monkeypatch.setattr(threading, "Thread", track_thread)
    stream = stream_audio_as_formatted_bytes(source(), audio_format="flac")
    try:
        assert next(stream)
        assert processes[0].poll() is None
        if wait_for_backpressure:
            assert backpressure.wait(timeout=5)
        stream.close()
        assert processes[0].poll() is not None
        assert all(not thread.is_alive() for thread in threads)
        assert all(
            pipe is not None and pipe.closed for pipe in (processes[0].stdin, processes[0].stdout, processes[0].stderr)
        )
    finally:
        stop.set()
        stream.close()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)


def test_encoded_audio_stream_finishes_normally() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    stream = stream_audio_as_formatted_bytes(iter([Audio(samples, sample_rate=16000)]), audio_format="flac")
    assert b"".join(stream).startswith(b"fLaC")


def test_encoded_audio_stream_preserves_producer_error() -> None:
    def source() -> Generator[Audio]:
        yield Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        raise ValueError("fixture producer error")

    with pytest.raises(ValueError, match="fixture producer error"):
        list(stream_audio_as_formatted_bytes(source(), audio_format="flac"))


@pytest.mark.parametrize("block_stage", ["producer", "write"])
def test_close_does_not_close_a_blocked_writers_pipe(monkeypatch: pytest.MonkeyPatch, block_stage: str) -> None:
    entered = threading.Event()
    release = threading.Event()
    killed = threading.Event()
    close_done = threading.Event()
    violations: list[str] = []
    threads: list[threading.Thread] = []
    close_errors: list[BaseException] = []
    real_thread = threading.Thread

    class InputPipe(io.BytesIO):
        def write(self, data: Any) -> int:
            if block_stage == "write":
                entered.set()
                release.wait()
            return super().write(data)

        def close(self) -> None:
            if threads and threads[0].is_alive() and threading.current_thread() is not threads[0]:
                violations.append("stdin closed outside its active writer")
                if block_stage == "write":
                    release.wait()
            super().close()

    class OutputPipe(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.first = True

        def read(self, _size: int = -1) -> bytes:
            if self.first:
                self.first = False
                return b"encoded fixture"
            killed.wait()
            return b""

    class Process:
        def __init__(self) -> None:
            self.stdin = InputPipe()
            self.stdout = OutputPipe()
            self.stderr = io.BytesIO()

        def poll(self) -> int | None:
            return -9 if killed.is_set() else None

        def kill(self) -> None:
            killed.set()

        def wait(self, timeout: float | None = None) -> int:
            assert killed.wait(timeout=timeout)
            return -9

    process = Process()

    def source() -> Generator[Audio]:
        yield Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        if block_stage == "producer":
            entered.set()
            release.wait()
        yield Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def make_process(*_args: Any, **_kwargs: Any) -> Process:
        return process

    def track_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(subprocess, "Popen", make_process)
    monkeypatch.setattr(threading, "Thread", track_thread)
    stream = stream_audio_as_formatted_bytes(source(), audio_format="flac")

    def close_stream() -> None:
        try:
            stream.close()
        except BaseException as error:  # noqa: BLE001
            close_errors.append(error)
        finally:
            close_done.set()

    closer = real_thread(target=close_stream, daemon=True)
    try:
        assert next(stream)
        assert entered.wait(timeout=5)
        closer.start()
        assert close_done.wait(timeout=7), "close blocked on an active writer's pipe"
        assert not close_errors
        assert killed.is_set()
        assert threads[0].is_alive()
        assert not process.stdin.closed
        assert not violations
    finally:
        release.set()
        killed.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
        stream.close()
    assert not closer.is_alive()
    assert all(not thread.is_alive() for thread in threads)
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert not violations
