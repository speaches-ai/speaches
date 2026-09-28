from huggingface_hub import ModelCardData

from speaches.config import Config
from speaches.executors.funasr import hf_model_filter
from speaches.executors.shared.registry import ExecutorRegistry


def test_funasr_executor_is_registered() -> None:
    registry = ExecutorRegistry(Config())
    assert "funasr" in [executor.name for executor in registry.transcription]


def test_funasr_filter_matches_funasr_asr_models() -> None:
    card_data = ModelCardData(library_name="funasr", pipeline_tag="automatic-speech-recognition")
    assert hf_model_filter.passes_filter("FunAudioLLM/SenseVoiceSmall", card_data)


def test_funasr_filter_rejects_non_funasr_models() -> None:
    card_data = ModelCardData(library_name="ctranslate2", pipeline_tag="automatic-speech-recognition")
    assert not hf_model_filter.passes_filter("Systran/faster-whisper-tiny", card_data)
