from collections.abc import Generator
import os
from pathlib import Path
import time
from typing import Any

from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import SpeechRequest, TranslationRequest


class FakeModelManager(BaseModelManager[dict[str, Any]]):
    def _load_fn(self, model_id: str) -> dict[str, Any]:
        if model_id == "bad-model":
            raise ValueError("cannot load bad-model")
        return {"model_id": model_id}

    def handle_speech_request(self, request: SpeechRequest, **_kwargs) -> Generator[Any]:
        command, _, arg = request.text.partition(":")
        with self.load_model(request.model):
            match command:
                case "count":
                    for i in range(int(arg)):
                        yield i
                case "pid":
                    yield os.getpid()
                case "tagged":
                    for i in range(20):
                        time.sleep(0.001)
                        yield (arg, i)
                case "value_error":
                    raise ValueError(arg)
                case "die":
                    yield 0
                    os._exit(3)
                case "forever":
                    try:
                        while True:
                            time.sleep(0.01)
                            yield 0
                    finally:
                        Path(arg).touch()

    def handle_translation_request(self, request: TranslationRequest, **_kwargs) -> tuple[str, str]:
        with self.load_model(request.model):
            return (f"translated by {request.model}", "text/plain")
