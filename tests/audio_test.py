import io

import numpy as np
import pydub
import pytest
import soundfile as sf

from speaches.audio import Audio


@pytest.fixture
def sample_audio() -> Audio:
    """Create a sample Audio object with a 440Hz sine wave"""
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

    return Audio(data=audio_data, sample_rate=sample_rate)


def test_pcm_format(sample_audio: Audio) -> None:
    """Test conversion to PCM format (raw bytes)"""
    pcm_bytes = sample_audio.as_formatted_bytes("pcm")

    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) > 0
    # PCM 16-bit should have 2 bytes per sample
    assert len(pcm_bytes) == len(sample_audio.data) * 2


def test_pcm_format_is_default(sample_audio: Audio) -> None:
    """Test that PCM is the default format"""
    default_bytes = sample_audio.as_formatted_bytes()
    pcm_bytes = sample_audio.as_formatted_bytes("pcm")

    assert default_bytes == pcm_bytes


def test_wav_format(sample_audio: Audio) -> None:
    """Test conversion to WAV format"""
    wav_bytes = sample_audio.as_formatted_bytes("wav")

    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

    # Verify WAV format by reading it back
    audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes))
    assert sample_rate == sample_audio.sample_rate
    assert len(audio_data) == len(sample_audio.data)
    # Check duration is approximately preserved (within 1%)
    assert abs(len(audio_data) - len(sample_audio.data)) / len(sample_audio.data) < 0.01


def test_flac_format(sample_audio: Audio) -> None:
    """Test conversion to FLAC format"""
    flac_bytes = sample_audio.as_formatted_bytes("flac")

    assert isinstance(flac_bytes, bytes)
    assert len(flac_bytes) > 0

    # Verify FLAC format by reading it back
    audio_data, sample_rate = sf.read(io.BytesIO(flac_bytes))
    assert sample_rate == sample_audio.sample_rate
    # FLAC is lossless, so data should be very close
    assert len(audio_data) == len(sample_audio.data)


def test_mp3_format(sample_audio: Audio) -> None:
    """Test conversion to MP3 format"""
    mp3_bytes = sample_audio.as_formatted_bytes("mp3")

    assert isinstance(mp3_bytes, bytes)
    assert len(mp3_bytes) > 0

    # Verify MP3 format by reading it back
    audio_data, sample_rate = sf.read(io.BytesIO(mp3_bytes))
    assert sample_rate == sample_audio.sample_rate
    # MP3 is lossy, duration should be approximately preserved
    expected_duration = sample_audio.duration
    actual_duration = len(audio_data) / sample_rate
    assert abs(actual_duration - expected_duration) < 0.1  # Within 100ms


def test_aac_format(sample_audio: Audio) -> None:
    """Test conversion to AAC format"""
    aac_bytes = sample_audio.as_formatted_bytes("aac")

    assert isinstance(aac_bytes, bytes)
    assert len(aac_bytes) > 0

    # Verify AAC format by reading it back with pydub
    audio_segment = pydub.AudioSegment.from_file(io.BytesIO(aac_bytes), format="aac")
    assert audio_segment.frame_rate == sample_audio.sample_rate
    # Check duration is approximately preserved (within 10%)
    expected_duration_ms = sample_audio.duration * 1000
    assert abs(len(audio_segment) - expected_duration_ms) / expected_duration_ms < 0.1


def test_opus_format(sample_audio: Audio) -> None:
    """Test conversion to Opus format"""
    opus_bytes = sample_audio.as_formatted_bytes("opus")

    assert isinstance(opus_bytes, bytes)
    assert len(opus_bytes) > 0

    # Verify Opus format by reading it back with pydub
    audio_segment = pydub.AudioSegment.from_file(io.BytesIO(opus_bytes), format="ogg", codec="libopus")
    # Opus may resample to supported rates (8k, 12k, 16k, 24k, 48k)
    # Check duration is approximately preserved
    expected_duration_ms = sample_audio.duration * 1000
    assert abs(len(audio_segment) - expected_duration_ms) / expected_duration_ms < 0.1


def test_all_formats_produce_different_outputs(sample_audio: Audio) -> None:
    """Test that different formats produce different outputs"""
    formats: list[str] = ["pcm", "wav", "flac", "mp3", "aac", "opus"]
    outputs = {fmt: sample_audio.as_formatted_bytes(fmt) for fmt in formats}  # type: ignore[arg-type]

    # Check that all formats produce different outputs
    # (except we already know pcm is used as default, but others should differ)
    unique_outputs = set(outputs.values())
    # At minimum, we should have several different outputs
    # PCM will be raw, others will have headers/containers
    assert len(unique_outputs) >= 5
