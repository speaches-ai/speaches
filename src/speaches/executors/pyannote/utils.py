import io
import logging

from numpy import float32
from numpy.typing import NDArray
from pyannote.audio import Pipeline
import soundfile as sf

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

    # Convert numpy array to audio format that pyannote can process
    # Create an in-memory file-like object
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, sample_rate, format="WAV")
    audio_buffer.seek(0)

    # Run diarization
    diarization_result = pipeline({"uri": "temp", "audio": audio_buffer})

    # Convert pyannote output to our format
    speaker_segments = {}
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments[(segment.start, segment.end)] = speaker

    logger.debug(f"Diarization found {len(speaker_segments)} speaker segments")
    return speaker_segments
