"""Unit tests for `BaseModelManager`'s registry bookkeeping.

No model download and no HTTP client, so these run in milliseconds anywhere.
"""

import threading

import pytest

from speaches.executors.shared.base_model_manager import BaseModelManager


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
    """A refused unload must not remove the model from `loaded_models`.

    `unload_model` used to delete the entry before calling `unload()`, which
    raises while `ref_count > 0`. The caller got the error, but the model was
    already gone from `loaded_models` -- so `GET /api/ps` stopped listing it
    while its weights were still resident, and `DELETE /api/ps/{model_id}`
    could only answer 404 from then on. The memory was unreachable until the
    model's own TTL timer happened to fire.
    """
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
