from collections.abc import Generator
import logging
from pathlib import Path
import re
import time
from typing import Literal

import huggingface_hub
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession
from pydantic import BaseModel, computed_field

from speaches.api_types import (
    OPENAI_SUPPORTED_SPEECH_VOICE_NAMES,
    Model,
)
from speaches.audio import Audio
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import (
    ModelRegistry,
)
from speaches.tracing import traced_generator
from speaches.utils import async_to_sync_generator

SAMPLE_RATE = 24000  # the default sample rate for Kokoro
MAX_CHUNK_CHARS = 400  # conservative limit to stay well under 510 phoneme limit
LIBRARY_NAME = "onnx"
TASK_NAME_TAG = "text-to-speech"
TAGS = {"speaches", "kokoro"}


def normalize_text_for_tts(text: str) -> str:
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    current_chunk = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {word}" if current_chunk else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class KokoroModelFiles(BaseModel):
    model: Path
    voices: Path


class KokoroModelVoice(BaseModel):
    name: str
    language: str
    gender: Literal["male", "female"]

    @computed_field
    @property
    def id(self) -> str:
        return self.name


VOICES = [
    # American English
    KokoroModelVoice(name="af_heart", language="en-us", gender="female"),
    KokoroModelVoice(name="af_alloy", language="en-us", gender="female"),
    KokoroModelVoice(name="af_aoede", language="en-us", gender="female"),
    KokoroModelVoice(name="af_bella", language="en-us", gender="female"),
    KokoroModelVoice(name="af_jessica", language="en-us", gender="female"),
    KokoroModelVoice(name="af_kore", language="en-us", gender="female"),
    KokoroModelVoice(name="af_nicole", language="en-us", gender="female"),
    KokoroModelVoice(name="af_nova", language="en-us", gender="female"),
    KokoroModelVoice(name="af_river", language="en-us", gender="female"),
    KokoroModelVoice(name="af_sarah", language="en-us", gender="female"),
    KokoroModelVoice(name="af_sky", language="en-us", gender="female"),
    KokoroModelVoice(name="am_adam", language="en-us", gender="male"),
    KokoroModelVoice(name="am_echo", language="en-us", gender="male"),
    KokoroModelVoice(name="am_eric", language="en-us", gender="male"),
    KokoroModelVoice(name="am_fenrir", language="en-us", gender="male"),
    KokoroModelVoice(name="am_liam", language="en-us", gender="male"),
    KokoroModelVoice(name="am_michael", language="en-us", gender="male"),
    KokoroModelVoice(name="am_onyx", language="en-us", gender="male"),
    KokoroModelVoice(name="am_puck", language="en-us", gender="male"),
    KokoroModelVoice(name="am_santa", language="en-us", gender="male"),
    # British English
    KokoroModelVoice(name="bf_alice", language="en-gb", gender="female"),
    KokoroModelVoice(name="bf_emma", language="en-gb", gender="female"),
    KokoroModelVoice(name="bf_isabella", language="en-gb", gender="female"),
    KokoroModelVoice(name="bf_lily", language="en-gb", gender="female"),
    KokoroModelVoice(name="bm_daniel", language="en-gb", gender="male"),
    KokoroModelVoice(name="bm_fable", language="en-gb", gender="male"),
    KokoroModelVoice(name="bm_george", language="en-gb", gender="male"),
    KokoroModelVoice(name="bm_lewis", language="en-gb", gender="male"),
    # Japanese
    KokoroModelVoice(name="jf_alpha", language="ja", gender="female"),
    KokoroModelVoice(name="jf_gongitsune", language="ja", gender="female"),
    KokoroModelVoice(name="jf_nezumi", language="ja", gender="female"),
    KokoroModelVoice(name="jf_tebukuro", language="ja", gender="female"),
    KokoroModelVoice(name="jm_kumo", language="ja", gender="male"),
    # Mandarin Chinese
    KokoroModelVoice(name="zf_xiaobei", language="zh", gender="female"),
    KokoroModelVoice(name="zf_xiaoni", language="zh", gender="female"),
    KokoroModelVoice(name="zf_xiaoxiao", language="zh", gender="female"),
    KokoroModelVoice(name="zf_xiaoyi", language="zh", gender="female"),
    KokoroModelVoice(name="zm_yunjian", language="zh", gender="male"),
    KokoroModelVoice(name="zm_yunxi", language="zh", gender="male"),
    KokoroModelVoice(name="zm_yunxia", language="zh", gender="male"),
    KokoroModelVoice(name="zm_yunyang", language="zh", gender="male"),
    # Spanish
    KokoroModelVoice(name="ef_dora", language="es", gender="female"),
    KokoroModelVoice(name="em_alex", language="es", gender="male"),
    KokoroModelVoice(name="em_santa", language="es", gender="male"),
    # French
    KokoroModelVoice(name="ff_siwis", language="fr-fr", gender="female"),
    # Hindi
    KokoroModelVoice(name="hf_alpha", language="hi", gender="female"),
    KokoroModelVoice(name="hf_beta", language="hi", gender="female"),
    KokoroModelVoice(name="hm_omega", language="hi", gender="male"),
    KokoroModelVoice(name="hm_psi", language="hi", gender="male"),
    # Italian
    KokoroModelVoice(name="if_sara", language="it", gender="female"),
    KokoroModelVoice(name="im_nicola", language="it", gender="male"),
    # Brazilian Portuguese
    KokoroModelVoice(name="pf_dora", language="pt-br", gender="female"),
    KokoroModelVoice(name="pm_alex", language="pt-br", gender="male"),
    KokoroModelVoice(name="pm_santa", language="pt-br", gender="male"),
]


class KokoroModel(Model):
    sample_rate: int
    voices: list[KokoroModelVoice]


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags=TAGS,
)


logger = logging.getLogger(__name__)


class KokoroModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[KokoroModel]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield KokoroModel(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
                sample_rate=SAMPLE_RATE,
                voices=VOICES,
            )

    def list_local_models(self) -> Generator[KokoroModel]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield KokoroModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=VOICES,
                )

    def get_model_files(self, model_id: str) -> KokoroModelFiles:
        model_files = list(list_model_files(model_id))

        model_file_path = next(file_path for file_path in model_files if file_path.name == "model.onnx")
        voices_file_path = next(file_path for file_path in model_files if file_path.name == "voices.bin")

        return KokoroModelFiles(
            model=model_file_path,
            voices=voices_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=["model.onnx", "voices.bin", "README.md"]
        )


kokoro_model_registry = KokoroModelRegistry(hf_model_filter=hf_model_filter)


class KokoroModelManager(BaseModelManager[Kokoro]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> Kokoro:
        model_files = kokoro_model_registry.get_model_files(model_id)
        providers = get_ort_providers_with_options(self.ort_opts)
        inf_sess = InferenceSession(model_files.model, providers=providers)
        return Kokoro.from_session(inf_sess, str(model_files.voices))

    @traced_generator()
    def handle_speech_request(
        self,
        request: SpeechRequest,
        **_kwargs,
    ) -> SpeechResponse:
        if request.speed < 0.5 or request.speed > 2.0:
            msg = f"Speed must be between 0.5 and 2.0, got {request.speed}"
            raise ValueError(msg)
        if request.voice not in [v.name for v in VOICES]:
            if request.voice in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
                logger.warning(
                    f"Voice '{request.voice}' is not supported by the model '{request.model}'. It will be replaced with '{VOICES[0].name}'. The behaviour of substituting OpenAI voices may be removed in the future without warning."
                )
                request.voice = VOICES[0].name
            else:
                msg = f"Voice '{request.voice}' is not supported. Supported voices: {VOICES}"
                raise ValueError(msg)

        normalized_text = normalize_text_for_tts(request.text)
        chunks = split_text_into_chunks(normalized_text)

        if not chunks:
            return

        voice_language = next(v.language for v in VOICES if v.name == request.voice)
        with self.load_model(request.model) as tts:
            start = time.perf_counter()
            for chunk in chunks:
                try:
                    async_stream = tts.create_stream(
                        chunk,
                        request.voice,
                        lang=voice_language,
                        speed=request.speed,
                    )
                    # HACK: converting an async generator to a sync generator
                    sync_stream = async_to_sync_generator(async_stream)
                    for audio_data, _ in sync_stream:
                        yield Audio(audio_data, sample_rate=SAMPLE_RATE)
                except RuntimeError as e:
                    if "number of lines" in str(e):
                        logger.warning(f"Phonemizer error for chunk, skipping: {e}")
                        continue
                    raise
                except IndexError as e:
                    if "out of bounds" in str(e):
                        logger.warning(f"Phoneme limit exceeded for chunk, skipping: {e}")
                        continue
                    raise

        logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start}s")
