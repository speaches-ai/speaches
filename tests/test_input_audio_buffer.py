from unittest.mock import MagicMock

import numpy as np

from speaches.realtime.input_audio_buffer import (
    MAX_BUFFER_SIZE_SAMPLES,
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    InputAudioBuffer,
)


def _make_buffer() -> InputAudioBuffer:
    return InputAudioBuffer(pubsub=MagicMock())


def _make_chunk(n: int, value: float = 1.0) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


class TestAppendAndSize:
    def test_empty_buffer(self) -> None:
        buf = _make_buffer()
        assert buf.size == 0
        assert buf.duration_ms == 0
        assert len(buf.data) == 0

    def test_single_append(self) -> None:
        buf = _make_buffer()
        chunk = _make_chunk(1600)
        buf.append(chunk)
        assert buf.size == 1600
        assert buf.duration_ms == 100
        np.testing.assert_array_equal(buf.data, chunk)

    def test_multiple_appends(self) -> None:
        buf = _make_buffer()
        buf.append(_make_chunk(800, 1.0))
        buf.append(_make_chunk(800, 2.0))
        assert buf.size == 1600
        assert buf.data[0] == 1.0
        assert buf.data[800] == 2.0

    def test_max_buffer_size_cap(self) -> None:
        buf = _make_buffer()
        buf.append(_make_chunk(MAX_BUFFER_SIZE_SAMPLES))
        assert buf.size == MAX_BUFFER_SIZE_SAMPLES
        buf.append(_make_chunk(100))
        assert buf.size == MAX_BUFFER_SIZE_SAMPLES


class TestConsolidate:
    def test_consolidate_preserves_data(self) -> None:
        buf = _make_buffer()
        buf.append(_make_chunk(100, 1.0))
        buf.append(_make_chunk(100, 2.0))
        buf.consolidate()
        assert buf.size == 200
        assert buf.data[0] == 1.0
        assert buf.data[100] == 2.0

    def test_consolidate_noop_empty(self) -> None:
        buf = _make_buffer()
        buf.consolidate()
        assert buf.size == 0


class TestVadRingBuffer:
    def test_vad_data_small_buffer(self) -> None:
        buf = _make_buffer()
        chunk = _make_chunk(100, 0.5)
        buf.append(chunk)
        vad = buf.vad_data
        assert len(vad) == 100
        np.testing.assert_array_equal(vad, chunk)

    def test_vad_data_exact_window(self) -> None:
        buf = _make_buffer()
        chunk = _make_chunk(MAX_VAD_WINDOW_SIZE_SAMPLES, 0.7)
        buf.append(chunk)
        vad = buf.vad_data
        assert len(vad) == MAX_VAD_WINDOW_SIZE_SAMPLES
        np.testing.assert_array_equal(vad, chunk)

    def test_vad_data_wrap_around(self) -> None:
        buf = _make_buffer()
        half = MAX_VAD_WINDOW_SIZE_SAMPLES // 2
        buf.append(_make_chunk(half + 100, 1.0))
        buf.append(_make_chunk(half + 100, 2.0))
        vad = buf.vad_data
        assert len(vad) == MAX_VAD_WINDOW_SIZE_SAMPLES
        # After wrapping, the oldest data (1.0) should be at the start,
        # newest data (2.0) at the end
        assert vad[-1] == 2.0

    def test_vad_data_overflow_single_large_chunk(self) -> None:
        buf = _make_buffer()
        chunk = _make_chunk(MAX_VAD_WINDOW_SIZE_SAMPLES + 500, 3.0)
        buf.append(chunk)
        vad = buf.vad_data
        assert len(vad) == MAX_VAD_WINDOW_SIZE_SAMPLES
        np.testing.assert_array_equal(vad, _make_chunk(MAX_VAD_WINDOW_SIZE_SAMPLES, 3.0))

    def test_vad_ring_preserves_order_across_wraps(self) -> None:
        buf = _make_buffer()
        chunk_size = MAX_VAD_WINDOW_SIZE_SAMPLES // 4
        for i in range(6):
            buf.append(_make_chunk(chunk_size, float(i)))
        vad = buf.vad_data
        assert len(vad) == MAX_VAD_WINDOW_SIZE_SAMPLES
        # Last 4 chunks should be in order: 2.0, 3.0, 4.0, 5.0
        assert vad[0] == 2.0
        assert vad[chunk_size] == 3.0
        assert vad[2 * chunk_size] == 4.0
        assert vad[3 * chunk_size] == 5.0


class TestDataWithVadApplied:
    def test_no_vad_state_returns_full_data(self) -> None:
        buf = _make_buffer()
        buf.append(_make_chunk(1600))
        np.testing.assert_array_equal(buf.data_w_vad_applied, buf.data)

    def test_vad_state_slices_data(self) -> None:
        buf = _make_buffer()
        buf.append(_make_chunk(16000))  # 1 second
        buf.vad_state.audio_start_ms = 100
        buf.vad_state.audio_end_ms = 500
        result = buf.data_w_vad_applied
        expected_samples = (500 - 100) * 16  # 6400 samples
        assert len(result) == expected_samples
