from __future__ import annotations

from typing import TYPE_CHECKING, Any

from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.model_worker import ModelWorker

if TYPE_CHECKING:
    from collections.abc import Generator

    from speaches.executors.shared.handler_protocol import (
        NonStreamingTranscriptionResponse,
        SpeechRequest,
        SpeechResponse,
        StreamingTranscriptionEvent,
        TranscriptionRequest,
        TranslationRequest,
        TranslationResponse,
    )


class IsolatedModelManager(BaseModelManager[ModelWorker]):
    def __init__(
        self,
        ttl: int,
        manager_cls: type[BaseModelManager[Any]],
        ctor_args: tuple[Any, ...],
        log_level: str,
    ) -> None:
        super().__init__(ttl)
        self.manager_cls = manager_cls
        self.ctor_args = ctor_args
        self.log_level = log_level

    def _load_fn(self, model_id: str) -> ModelWorker:
        worker = ModelWorker(
            model_id,
            self.manager_cls,
            self.ctor_args,
            self.log_level,
            on_death=lambda w: self._on_worker_death(model_id, w),
        )
        worker.start()
        return worker

    def _close_fn(self, model: ModelWorker) -> None:
        model.stop()

    def _on_worker_death(self, model_id: str, worker: ModelWorker) -> None:
        with self._lock:
            entry = self.loaded_models.get(model_id)
            # Drop the entry so the next request spawns a fresh worker.
            if entry is not None and entry.model is worker:
                del self.loaded_models[model_id]

    def handle_speech_request(self, request: SpeechRequest, **kwargs) -> SpeechResponse:
        with self.load_model(request.model) as worker:
            yield from worker.stream("handle_speech_request", request, kwargs)

    def handle_streaming_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> Generator[StreamingTranscriptionEvent]:
        with self.load_model(request.model) as worker:
            yield from worker.stream("handle_streaming_transcription_request", request, kwargs)

    def handle_non_streaming_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse:
        with self.load_model(request.model) as worker:
            return worker.value("handle_non_streaming_transcription_request", request, kwargs)

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        return self.handle_non_streaming_transcription_request(request, **kwargs)

    def handle_translation_request(self, request: TranslationRequest, **kwargs) -> TranslationResponse:
        with self.load_model(request.model) as worker:
            return worker.value("handle_translation_request", request, kwargs)
