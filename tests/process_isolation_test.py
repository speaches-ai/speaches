from collections.abc import Generator
import multiprocessing
import os
from pathlib import Path
import threading
import time

import pytest

from speaches.executors.shared.handler_protocol import SpeechRequest, TranslationRequest
from speaches.executors.shared.isolated_model_manager import IsolatedModelManager
from speaches.executors.shared.model_worker import ModelWorkerDiedError
from tests.fake_isolated_manager import FakeModelManager

MODEL_ID = "fake-model"

pytestmark = pytest.mark.timeout(60)


def speech_request(text: str, model: str = MODEL_ID) -> SpeechRequest:
    return SpeechRequest(model=model, voice="voice", text=text, speed=1.0)


def make_manager(ttl: int = -1) -> IsolatedModelManager:
    return IsolatedModelManager(ttl, FakeModelManager, (), "info")


def wait_until(predicate, timeout: float = 10.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def manager() -> Generator[IsolatedModelManager]:
    manager = make_manager()
    yield manager
    manager.shutdown()


def test_stream_order(manager: IsolatedModelManager) -> None:
    assert list(manager.handle_speech_request(speech_request("count:50"))) == list(range(50))


def test_concurrent_requests_share_one_worker(manager: IsolatedModelManager) -> None:
    results: dict[int, list[tuple[str, int]]] = {}

    def run(n: int) -> None:
        results[n] = list(manager.handle_speech_request(speech_request(f"tagged:{n}")))

    threads = [threading.Thread(target=run, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {n: [(str(n), i) for i in range(20)] for n in range(5)}
    assert len(multiprocessing.active_children()) == 1


def test_non_streaming_value(manager: IsolatedModelManager) -> None:
    request = TranslationRequest.model_construct(model=MODEL_ID)
    assert manager.handle_translation_request(request) == (f"translated by {MODEL_ID}", "text/plain")


def test_value_error_type_preserved(manager: IsolatedModelManager) -> None:
    with pytest.raises(ValueError, match="speed must be between"):
        list(manager.handle_speech_request(speech_request("value_error:speed must be between 0.5 and 2.0")))


def test_load_error_raised(manager: IsolatedModelManager) -> None:
    with pytest.raises(ValueError, match="cannot load bad-model"):
        list(manager.handle_speech_request(speech_request("count:1", model="bad-model")))
    assert list(manager.handle_speech_request(speech_request("count:2"))) == [0, 1]


def test_worker_crash_mid_stream_does_not_hang_and_respawns(manager: IsolatedModelManager) -> None:
    (pid,) = manager.handle_speech_request(speech_request("pid:"))
    stream = manager.handle_speech_request(speech_request("die:"))
    assert next(stream) == 0
    with pytest.raises(ModelWorkerDiedError) as exc_info:
        next(stream)
    assert exc_info.value.exitcode == 3
    assert wait_until(lambda: MODEL_ID not in manager.loaded_models)
    (new_pid,) = manager.handle_speech_request(speech_request("pid:"))
    assert new_pid != pid


def test_ttl_zero_terminates_worker_after_request() -> None:
    manager = make_manager(ttl=0)
    (pid,) = manager.handle_speech_request(speech_request("pid:"))
    assert manager.loaded_models == {}
    assert not process_exists(pid)
    assert multiprocessing.active_children() == []


def test_ttl_timer_terminates_idle_worker() -> None:
    manager = make_manager(ttl=1)
    try:
        (pid,) = manager.handle_speech_request(speech_request("pid:"))
        assert process_exists(pid)
        assert wait_until(lambda: not process_exists(pid))
        assert wait_until(lambda: manager.loaded_models == {})
    finally:
        manager.shutdown()


def test_closing_stream_cancels_in_worker(manager: IsolatedModelManager, tmp_path: Path) -> None:
    marker = tmp_path / "cancelled"
    stream = manager.handle_speech_request(speech_request(f"forever:{marker}"))
    assert next(stream) == 0
    stream.close()
    assert wait_until(marker.exists)
    assert manager.loaded_models[MODEL_ID].ref_count == 0


def test_shutdown_leaves_no_child_processes() -> None:
    manager = make_manager()
    stream = manager.handle_speech_request(speech_request("forever:" + os.devnull))
    next(stream)
    assert len(multiprocessing.active_children()) == 1
    manager.shutdown()
    assert multiprocessing.active_children() == []
    with pytest.raises(ModelWorkerDiedError):
        next(stream)
