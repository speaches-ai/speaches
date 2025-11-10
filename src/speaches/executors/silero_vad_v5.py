from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, TypedDict

from faster_whisper.utils import get_assets_path
import numpy as np
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.hf_utils import HfModelFilter
from speaches.model_registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from speaches.config import OrtOptions
    from speaches.executors.shared.handler_protocol import VadRequest


SAMPLE_RATE = 16000
MODEL_ID = "silero_vad_v5"
SAMPLE_RATE_MS = SAMPLE_RATE // 1000

logger = logging.getLogger(__name__)


# The code below is adapted from https://github.com/snakers4/silero-vad.
class VadOptions(BaseModel):
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      neg_threshold: Silence threshold for determining the end of speech. If a probability is lower
        than neg_threshold, it is always considered silence. Values higher than neg_threshold
        are only considered speech if the previous sample was classified as speech; otherwise,
        they are treated as silence. This parameter helps refine the detection of speech
         transitions, ensuring smoother segment boundaries.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side

    """

    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


class SpeechTimestamp(BaseModel):
    start: int
    end: int


class SileroVADModelFiles(BaseModel):
    encoder: Path
    decoder: Path


class SileroVADModel:
    def __init__(self, encoder_path: Path, decoder_path: Path, providers: list[tuple[str, dict]]) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        # opts.inter_op_num_threads = 1
        # opts.intra_op_num_threads = 1
        # opts.enable_cpu_mem_arena = False
        # opts.log_severity_level = 4

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_path,
            providers=providers,
            sess_options=opts,
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path,
            providers=providers,
            sess_options=opts,
        )

    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ) -> NDArray[np.float32]:
        timelog_start_1 = time.perf_counter()
        assert audio.ndim == 2, "Input should be a 2D array with size (batch_size, num_samples)"
        assert audio.shape[1] % num_samples == 0, "Input size should be a multiple of num_samples"

        batch_size = audio.shape[0]

        state = np.zeros((2, batch_size, 128), dtype=np.float32)
        context = np.zeros(
            (batch_size, context_size_samples),
            dtype=np.float32,
        )

        batched_audio = audio.reshape(batch_size, -1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[:, -1] = 0
        context = np.roll(context, 1, 1)
        batched_audio = np.concatenate([context, batched_audio], 2)

        batched_audio = batched_audio.reshape(-1, num_samples + context_size_samples)

        encoder_batch_size = 10000
        num_segments = batched_audio.shape[0]
        encoder_outputs = []
        for i in range(0, num_segments, encoder_batch_size):
            encoder_output = self.encoder_session.run(None, {"input": batched_audio[i : i + encoder_batch_size]})[0]
            encoder_outputs.append(encoder_output)

        encoder_output = np.concatenate(encoder_outputs, axis=0)
        encoder_output = encoder_output.reshape(batch_size, -1, 128)

        decoder_outputs = []
        for window in np.split(encoder_output, encoder_output.shape[1], axis=1):
            out, state = self.decoder_session.run(None, {"input": window.squeeze(1), "state": state})
            decoder_outputs.append(out)

        out = np.stack(decoder_outputs, axis=1).squeeze(-1)
        logger.debug(f"VAD model inference took {time.perf_counter() - timelog_start_1:.4f}s")
        return out


class SileroVADModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[Model, None, None]:
        return
        yield  # pyright: ignore[reportUnreachable]

    def list_local_models(self) -> Generator[Model, None, None]:
        encoder_path = Path(get_assets_path()) / "silero_encoder_v5.onnx"
        if encoder_path.exists():
            yield Model(
                id=MODEL_ID,
                created=int(encoder_path.stat().st_mtime),
                owned_by="snakers4",
                task="voice-activity-detection",
            )

    def get_model_files(self, model_id: str) -> SileroVADModelFiles:
        assert model_id == MODEL_ID, f"Only '{MODEL_ID}' model is supported"
        encoder_path = Path(get_assets_path()) / "silero_encoder_v5.onnx"
        decoder_path = Path(get_assets_path()) / "silero_decoder_v5.onnx"
        return SileroVADModelFiles(encoder=encoder_path, decoder=decoder_path)


silero_vad_model_registry = SileroVADModelRegistry(
    hf_model_filter=HfModelFilter(library_name="onnx", task="voice-activity-detection")
)


class SileroVADModelManager(BaseModelManager[SileroVADModel]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> SileroVADModel:
        model_files = silero_vad_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        return SileroVADModel(model_files.encoder, model_files.decoder, providers)

    def handle_vad_request(self, request: VadRequest, **_kwargs) -> list[SpeechTimestamp]:
        return get_speech_timestamps(
            request.audio.data,
            model_manager=self,
            model_id=request.model_id,
            vad_options=request.vad_options,
            sampling_rate=request.sampling_rate,
        )


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: VadOptions,
    model_manager: SileroVADModelManager,
    model_id: str = MODEL_ID,
    sampling_rate: int = SAMPLE_RATE,
) -> list[SpeechTimestamp]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      model_manager: The model manager instance for loading the VAD model.
      model_id: The model ID to use for VAD.
      vad_options: Options for VAD processing.
      sampling rate: Sampling rate of the audio.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.

    """
    _perf_start = time.perf_counter()

    threshold = vad_options.threshold
    neg_threshold = vad_options.neg_threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = 512
    speech_pad_ms = vad_options.speech_pad_ms
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    with model_manager.load_model(model_id) as model:
        padded_audio = np.pad(audio, (0, window_size_samples - audio.shape[0] % window_size_samples))
        speech_probs = model(padded_audio.reshape(1, -1)).squeeze(0)

        triggered = False
        speeches = []
        current_speech = {}
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)

        # to save potential segment end (and tolerate some silence)
        temp_end = 0
        # to save potential segment limits in case of maximum segment size reached
        prev_end = next_start = 0

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = window_size_samples * i

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech["start"] = window_size_samples * i
                continue

            if triggered and (window_size_samples * i) - current_speech["start"] > max_speech_samples:
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    # previously reached silence (< neg_thres) and is still not speech (< thres)
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech["end"] = window_size_samples * i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                # condition to avoid cutting in very short silence
                if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                    prev_end = temp_end
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                current_speech["end"] = temp_end
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - silence_duration // 2))
                else:
                    speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    elapsed = time.perf_counter() - _perf_start
    logger.debug(f"VAD processing took {elapsed:.4f}s for {audio_length_samples / sampling_rate:.2f}s audio")
    return [SpeechTimestamp(**speech) for speech in speeches]


def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    return [SpeechTimestamp(start=ts.start // SAMPLE_RATE_MS, end=ts.end // SAMPLE_RATE_MS) for ts in speech_timestamps]


class MergedSegment(TypedDict):
    start: int
    end: int
    segments: list[tuple[int, int]]


def merge_segments(
    segments_list: list[SpeechTimestamp], vad_options: VadOptions, sampling_rate: int = SAMPLE_RATE
) -> list[MergedSegment]:
    if not segments_list:
        return []

    curr_end = 0
    seg_idxs = []
    merged_segments = []
    edge_padding = vad_options.speech_pad_ms * sampling_rate // 1000
    chunk_length = vad_options.max_speech_duration_s * sampling_rate

    curr_start = segments_list[0].start

    for idx, seg in enumerate(segments_list):
        # if any segment start timing is less than previous segment end timing,
        # reset the edge padding. Similarly for end timing.
        if idx > 0:
            if seg.start < segments_list[idx - 1].end:
                seg.start += edge_padding
        if idx < len(segments_list) - 1:
            if seg.end > segments_list[idx + 1].start:
                seg.end -= edge_padding

        if seg.end - curr_start > chunk_length and curr_end - curr_start > 0:
            merged_segments.append(
                {
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                }
            )
            curr_start = seg.start
            seg_idxs = []
        curr_end = seg.end
        seg_idxs.append((seg.start, seg.end))
    # add final
    merged_segments.append(
        {
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        }
    )
    return merged_segments


# NOTE: below code was copied over from `faster_whisper.vad` but is not used anywhere. Kept for reference.
#
#
# class ChunkMetadata(TypedDict):
#     start_time: float
#     end_time: float
#
# def collect_chunks(
#     audio: np.ndarray, chunks: list[dict], sampling_rate: int = SAMPLE_RATE
# ) -> tuple[list[np.ndarray], list[ChunkMetadata]]:
#     """Collects audio chunks."""
#     if not chunks:
#         chunk_metadata: ChunkMetadata = {
#             "start_time": 0,
#             "end_time": 0,
#         }
#         return [np.array([], dtype=np.float32)], [chunk_metadata]
#
#     audio_chunks = []
#     chunks_metadata = []
#     for chunk in chunks:
#         chunk_metadata = {
#             "start_time": chunk["start"] / sampling_rate,
#             "end_time": chunk["end"] / sampling_rate,
#         }
#         audio_chunks.append(audio[chunk["start"] : chunk["end"]])
#         chunks_metadata.append(chunk_metadata)
#     return audio_chunks, chunks_metadata
#
#
# class SpeechTimestampsMap:
#     """Helper class to restore original speech timestamps."""
#
#     def __init__(self, chunks: list[dict], sampling_rate: int, time_precision: int = 2) -> None:
#         self.sampling_rate = sampling_rate
#         self.time_precision = time_precision
#         self.chunk_end_sample = []
#         self.total_silence_before = []
#
#         previous_end = 0
#         silent_samples = 0
#
#         for chunk in chunks:
#             silent_samples += chunk["start"] - previous_end
#             previous_end = chunk["end"]
#
#             self.chunk_end_sample.append(chunk["end"] - silent_samples)
#             self.total_silence_before.append(silent_samples / sampling_rate)
#
#     def get_original_time(
#         self,
#         time: float,
#         chunk_index: int | None = None,
#     ) -> float:
#         if chunk_index is None:
#             chunk_index = self.get_chunk_index(time)
#
#         total_silence_before = self.total_silence_before[chunk_index]
#         return round(total_silence_before + time, self.time_precision)
#
#     def get_chunk_index(self, time: float) -> int:
#         sample = int(time * self.sampling_rate)
#         return min(
#             bisect.bisect(self.chunk_end_sample, sample),
#             len(self.chunk_end_sample) - 1,
#         )
