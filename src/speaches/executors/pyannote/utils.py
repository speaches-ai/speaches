import logging

from numpy import float32
from numpy.typing import NDArray
from pyannote.audio import Pipeline
import torch

logger = logging.getLogger(__name__)


def run_diarization(audio: NDArray[float32], pipeline: Pipeline, sample_rate: int = 16000) -> dict[tuple[float, float], str]:
    """Run speaker diarization on audio data.

    Args:
        audio: Audio array (mono, float32)
        pipeline: Pyannote diarization pipeline
        sample_rate: Sample rate of the audio

    Returns:
        Dictionary mapping (start_time, end_time) tuples to speaker labels
    """
    logger.debug("Running speaker diarization")

    # Convert numpy array to torch tensor
    # Pyannote expects shape (channels, samples), so add channel dimension if needed
    waveform = torch.from_numpy(audio)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension: (samples,) -> (1, samples)

    # Run diarization by passing waveform and sample_rate directly
    # This avoids the torchaudio file loading deprecation warnings
    diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Convert pyannote output to our format
    speaker_segments = {}
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments[(segment.start, segment.end)] = speaker

    logger.debug(f"Diarization found {len(speaker_segments)} speaker segments")
    return speaker_segments
