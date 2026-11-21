import numpy as np

from speaches.audio import Audio


def test_as_bytes_saturates_pcm_samples_without_changing_in_range_values() -> None:
    samples = np.array([-1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25], dtype=np.float32)

    pcm = np.frombuffer(Audio(samples, sample_rate=24000).as_bytes(), dtype=np.int16)

    assert pcm.tolist() == [-32768, -32767, -16383, 0, 16383, 32767, 32767]
    assert samples.tolist() == [-1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25]
