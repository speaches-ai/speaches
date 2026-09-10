# Unit tests for `BaseModelManager`'s registry bookkeeping.
#
# No model download and no HTTP client, so these run in milliseconds anywhere.

import threading
import time

import pytest

from speaches.executors.shared.base_model_manager import BaseModelManager, SelfDisposingModel


class FakeModelManager(BaseModelManager[dict]):
    def _load_fn(self, model_id: str) -> dict:
        return {"model_id": model_id}


@pytest.fixture
def manager() -> FakeModelManager:
    m = FakeModelManager(ttl=300)
    yield m
    for thread in threading.enumerate():
        if isinstance(thread, threading.Timer):
            thread.cancel()


def test_unload_model_removes_the_entry_on_success(manager: FakeModelManager) -> None:
    handle = manager.load_model("m")
    with handle:
        pass
    manager.unload_model("m")
    assert handle.model is None
    assert "m" not in manager.loaded_models


def test_unload_model_raises_for_an_unknown_model(manager: FakeModelManager) -> None:
    with pytest.raises(KeyError):
        manager.unload_model("does-not-exist")


def test_unload_model_keeps_the_entry_when_the_model_is_in_use(manager: FakeModelManager) -> None:
    # A refused unload must not remove the model from `loaded_models`.
    #
    # `unload_model` used to delete the entry before calling `unload()`, which
    # raises while `ref_count > 0`. The caller got the error, but the model was
    # already gone from `loaded_models` -- so `GET /api/ps` stopped listing it
    # while its weights were still resident, and `DELETE /api/ps/{model_id}`
    # could only answer 404 from then on. The memory was unreachable until the
    # model's own TTL timer happened to fire.
    handle = manager.load_model("m")
    with handle:
        with pytest.raises(ValueError, match="still in use"):
            manager.unload_model("m")

        assert "m" in manager.loaded_models, "a refused unload dropped the model from the registry"
        assert handle.model is not None, "nothing was actually unloaded"

    # Still addressable afterwards, so a retry succeeds.
    assert "m" in manager.loaded_models
    manager.unload_model("m")
    assert handle.model is None
    assert "m" not in manager.loaded_models


def test_load_model_waits_out_an_in_flight_unload(manager: FakeModelManager) -> None:
    # While `unload_model` holds the registry lock across `unload()`, a
    # concurrent `load_model` must wait for the entry to be removed rather
    # than receive the handle being torn down -- otherwise it would reload
    # weights into a handle the registry no longer tracks.
    first = manager.load_model("m")
    with first:
        pass

    unload_started = threading.Event()
    finish_unload = threading.Event()
    original_unload = first.unload

    def gated_unload() -> None:
        unload_started.set()
        finish_unload.wait(timeout=10)
        original_unload()

    first.unload = gated_unload

    unloader = threading.Thread(target=manager.unload_model, args=("m",))
    unloader.start()
    assert unload_started.wait(timeout=10)

    loaded: list[SelfDisposingModel[dict]] = []
    loader = threading.Thread(target=lambda: loaded.append(manager.load_model("m")))
    loader.start()
    time.sleep(0.2)
    assert not loaded, "load_model returned a handle while its unload was in flight"
    finish_unload.set()
    unloader.join(timeout=10)
    loader.join(timeout=10)

    second = loaded[0]
    assert second is not first
    assert manager.loaded_models["m"] is second
    with second as model:
        assert model == {"model_id": "m"}
    with pytest.raises(ValueError, match="has been unloaded"), first:
        pass


def test_load_model_replaces_a_dead_entry_before_its_unload_callback_lands(manager: FakeModelManager) -> None:
    # `SelfDisposingModel.unload` fires its callback after releasing `rlock`,
    # so a TTL unload leaves the registry entry present-but-dead for a moment.
    # A `load_model` in that gap must swap in a fresh handle, and the late
    # callback must not delete the replacement.
    first = manager.load_model("m")
    with first:
        pass

    loaded_during_callback: list[SelfDisposingModel[dict]] = []
    original_callback = first.model_unloaded_callback
    assert original_callback is not None

    def load_inside_callback(model: SelfDisposingModel[dict]) -> None:
        loaded_during_callback.append(manager.load_model("m"))
        original_callback(model)

    first.model_unloaded_callback = load_inside_callback
    first.unload()  # the path a TTL timer takes

    second = loaded_during_callback[0]
    assert second is not first
    assert manager.loaded_models["m"] is second
    with second as model:
        assert model == {"model_id": "m"}
