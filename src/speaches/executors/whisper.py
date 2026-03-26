from __future__ import annotations

import gc
import logging
import threading
import time
from typing import TYPE_CHECKING

from faster_whisper import BatchedInferencePipeline, WhisperModel
import faster_whisper.transcribe
import huggingface_hub
import openai.types.audio
from opentelemetry import trace
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import (  # noqa: TC001
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
    TranslationRequest,
    TranslationResponse,
)
from speaches.executors.shared.vad_types import SpeechTimestamp, VadOptions  # noqa: TC001
from speaches.executors.silero_vad_v5 import SAMPLE_RATE, merge_segments
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.text_utils import format_as_srt, format_as_vtt
from speaches.tracing import traced, traced_generator
from speaches.utils import CudaOutOfMemoryError

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path

    from speaches.config import (
        WhisperConfig,
    )
    from speaches.routers.stt import ResponseFormat


LIBRARY_NAME = "ctranslate2"
TASK_NAME_TAG = "automatic-speech-recognition"


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower() and "CUDA" in str(exc)


def _try_clear_cuda_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _transcribe_with_oom_retry(
    whisper_model: BatchedInferencePipeline,
    batch_size: int,
    audio_duration: float | None,
    **transcribe_kwargs,
) -> tuple[list[faster_whisper.transcribe.Segment], faster_whisper.transcribe.TranscriptionInfo]:
    batch_sizes = [batch_size] if batch_size <= 1 else [batch_size, 1]
    for i, bs in enumerate(batch_sizes):
        try:
            segments, info = whisper_model.transcribe(batch_size=bs, **transcribe_kwargs)
            return list(segments), info
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            _try_clear_cuda_cache()
            if i == len(batch_sizes) - 1:
                logger.exception(f"CUDA OOM during transcription of {audio_duration}s audio (batch_size={bs})")
                raise CudaOutOfMemoryError(audio_duration) from e
            logger.warning(f"CUDA OOM with batch_size={bs} for {audio_duration}s audio, retrying with batch_size=1")
    raise AssertionError("unreachable")


def build_clip_timestamps(
    speech_segments: list[SpeechTimestamp],
    vad_options: VadOptions,
) -> list[dict[str, float]] | None:
    if not speech_segments:
        return None
    merged = merge_segments(speech_segments, vad_options)
    return [{"start": seg["start"] / SAMPLE_RATE, "end": seg["end"] / SAMPLE_RATE} for seg in merged]


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


class WhisperModelFiles(BaseModel):
    model: Path
    config: Path
    tokenizer: Path
    preprocessor_config: Path


class WhisperModelRegistry(ModelRegistry[Model, WhisperModelFiles]):
    def list_remote_models(self) -> Generator[Model]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> WhisperModelFiles:
        model_files = list(list_model_files(model_id))

        # the necessary files are specified in `faster_whisper.transcribe`
        model_file_path = next(file_path for file_path in model_files if file_path.name == "model.bin")
        config_file_path = next(
            file_path for file_path in model_files if file_path.name == "config.json"
        )  # NOTE: I don't think this file is used
        tokenizer_file_path = next(file_path for file_path in model_files if file_path.name == "tokenizer.json")
        preprocessor_config_file_path = next(
            file_path for file_path in model_files if file_path.name == "preprocessor_config.json"
        )
        return WhisperModelFiles(
            model=model_file_path,
            config=config_file_path,
            tokenizer=tokenizer_file_path,
            preprocessor_config=preprocessor_config_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        # Taken from faster_whisper/utils.py
        allow_patterns = [
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ]
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=[*allow_patterns, "README.md"]
        )


whisper_model_registry = WhisperModelRegistry(hf_model_filter=hf_model_filter)


class WhisperModelManager(BaseModelManager[BatchedInferencePipeline]):
    def __init__(self, ttl: int, whisper_config: WhisperConfig) -> None:
        super().__init__(ttl)
        self.whisper_config = whisper_config
        self._inference_semaphore = threading.Semaphore(whisper_config.max_concurrency)

    def _load_fn(self, model_id: str) -> BatchedInferencePipeline:
        model = WhisperModel(
            model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
            flash_attention=self.whisper_config.flash_attention,
            max_queued_batches=self.whisper_config.max_queued_batches,
            tensor_parallel=self.whisper_config.tensor_parallel,
        )
        return BatchedInferencePipeline(model=model)

    @traced()
    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format == "diarized_json":
            raise NotImplementedError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        timelog_start = time.perf_counter()
        with self._inference_semaphore, self.load_model(request.model) as whisper_model:
            clip_timestamps = build_clip_timestamps(request.speech_segments, request.vad_options)
            segments, transcription_info = _transcribe_with_oom_retry(
                whisper_model,
                batch_size=self.whisper_config.batch_size,
                audio_duration=request.audio.duration,
                audio=request.audio.data,
                task="transcribe",
                language=request.language,
                initial_prompt=request.prompt,
                word_timestamps="word" in request.timestamp_granularities,
                temperature=request.temperature,
                vad_filter=clip_timestamps is None,
                clip_timestamps=clip_timestamps,  # pyrefly: ignore[bad-argument-type]
                hotwords=request.hotwords,
                without_timestamps=request.without_timestamps,
            )

            res = segments_to_transcription_response(
                segments,
                transcription_info,
                response_format=request.response_format,
            )
            logger.info(
                f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - timelog_start} seconds"
            )
            return res

    @traced_generator()
    def handle_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> Generator[StreamingTranscriptionEvent]:
        timelog_start = time.perf_counter()
        with self._inference_semaphore, self.load_model(request.model) as whisper_model:
            clip_timestamps = build_clip_timestamps(request.speech_segments, request.vad_options)
            # Streaming cannot retry mid-stream, so use batch_size=1 for safety
            try:
                segments, _transcription_info = whisper_model.transcribe(
                    request.audio.data,
                    batch_size=1,
                    task="transcribe",
                    language=request.language,
                    initial_prompt=request.prompt,
                    word_timestamps="word" in request.timestamp_granularities,
                    temperature=request.temperature,
                    vad_filter=clip_timestamps is None,
                    clip_timestamps=clip_timestamps,  # pyrefly: ignore[bad-argument-type]
                    hotwords=request.hotwords,
                    without_timestamps=request.without_timestamps,
                )

                all_text = []
                for segment in segments:
                    all_text.append(segment.text)
                    yield openai.types.audio.TranscriptionTextDeltaEvent(
                        type="transcript.text.delta", delta=segment.text, logprobs=None
                    )

                yield openai.types.audio.TranscriptionTextDoneEvent(
                    type="transcript.text.done", text="".join(all_text), logprobs=None
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    logger.exception(f"CUDA OOM during streaming transcription of {request.audio.duration}s audio")
                    _try_clear_cuda_cache()
                    raise CudaOutOfMemoryError(request.audio.duration) from e
                raise
        logger.info(
            f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - timelog_start} seconds"
        )

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)

    @traced()
    def handle_translation_request(
        self,
        request: TranslationRequest,
        **_kwargs,
    ) -> TranslationResponse:
        if request.response_format == "diarized_json":
            raise NotImplementedError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        with self._inference_semaphore, self.load_model(request.model) as whisper_model:
            segments, transcription_info = _transcribe_with_oom_retry(
                whisper_model,
                batch_size=self.whisper_config.batch_size,
                audio_duration=request.audio.duration,
                audio=request.audio.data,
                task="translate",
                initial_prompt=request.prompt,
                temperature=request.temperature,
            )

            return segments_to_translation_response(
                segments,
                transcription_info,
                response_format=request.response_format,
            )


def segments_to_text(segments: Iterable[faster_whisper.transcribe.Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def segments_to_transcription_response(
    segments: list[faster_whisper.transcribe.Segment],
    transcription_info: faster_whisper.transcribe.TranscriptionInfo,
    response_format: ResponseFormat,
) -> NonStreamingTranscriptionResponse:
    match response_format:
        case "text":
            return segments_to_text(segments), "text/plain"
        case "json":
            return openai.types.audio.Transcription(
                text=segments_to_text(segments),
            )

        case "verbose_json":
            return openai.types.audio.TranscriptionVerbose(
                language=transcription_info.language,
                duration=transcription_info.duration,
                text=segments_to_text(segments),
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=segment.id,
                        seek=segment.seek,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        tokens=segment.tokens,
                        temperature=segment.temperature or 0,  # FIX: hardcoded
                        avg_logprob=segment.avg_logprob,
                        compression_ratio=segment.compression_ratio,
                        no_speech_prob=segment.no_speech_prob,
                    )
                    for segment in segments
                ],
                words=[
                    openai.types.audio.TranscriptionWord(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                    )
                    for segment in segments
                    for word in (segment.words or [])
                ]
                if transcription_info.transcription_options.word_timestamps
                else None,
            )

        case "vtt":
            return "".join(
                format_as_vtt(segment.text, segment.start, segment.end, i) for i, segment in enumerate(segments)
            ), "text/vtt"

        case "srt":
            return "".join(
                format_as_srt(segment.text, segment.start, segment.end, i) for i, segment in enumerate(segments)
            ), "text/plain"


def segments_to_translation_response(
    segments: list[faster_whisper.transcribe.Segment],
    transcription_info: faster_whisper.transcribe.TranscriptionInfo,
    response_format: ResponseFormat,
) -> TranslationResponse:
    match response_format:
        case "text":
            return segments_to_text(segments), "text/plain"
        case "json":
            return openai.types.audio.Translation(
                text=segments_to_text(segments),
            )

        case "verbose_json":
            return openai.types.audio.TranslationVerbose(
                language=transcription_info.language,
                duration=transcription_info.duration,
                text=segments_to_text(segments),
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=segment.id,
                        seek=segment.seek,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        tokens=segment.tokens,
                        temperature=segment.temperature or 0,  # FIX: hardcoded
                        avg_logprob=segment.avg_logprob,
                        compression_ratio=segment.compression_ratio,
                        no_speech_prob=segment.no_speech_prob,
                    )
                    for segment in segments
                ],
            )

        case "vtt":
            return "".join(
                format_as_vtt(segment.text, segment.start, segment.end, i) for i, segment in enumerate(segments)
            ), "text/vtt"

        case "srt":
            return "".join(
                format_as_srt(segment.text, segment.start, segment.end, i) for i, segment in enumerate(segments)
            ), "text/plain"
