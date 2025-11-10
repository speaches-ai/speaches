from collections.abc import Generator
import logging
from pathlib import Path
import time
from typing import Literal

import huggingface_hub
from kokoro_onnx import Kokoro
import numpy as np
from onnxruntime import InferenceSession
from pydantic import BaseModel, computed_field

from speaches.api_types import (
    OPENAI_SUPPORTED_SPEECH_VOICE_NAMES,
    SUPPORTED_NON_STREAMABLE_SPEECH_RESPONSE_FORMATS,
    Model,
)
from speaches.audio import convert_audio_format, resample_audio_bytes
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
from speaches.utils import async_to_sync_generator

SAMPLE_RATE = 24000  # the default sample rate for Kokoro
LIBRARY_NAME = "onnx"
TASK_NAME_TAG = "text-to-speech"
TAGS = {"speaches", "kokoro"}


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
    def list_remote_models(self) -> Generator[KokoroModel, None, None]:
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

    def list_local_models(self) -> Generator[KokoroModel, None, None]:
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

        with self.load_model(request.model) as tts:
            audio_generator = generate_audio(
                tts,
                request.input,
                request.voice,
                speed=request.speed,
                sample_rate=request.sample_rate,
            )
            # these file formats can't easily be streamed because they have headers and/or metadata
            if request.response_format in SUPPORTED_NON_STREAMABLE_SPEECH_RESPONSE_FORMATS:
                audio_data = b"".join(list(audio_generator))
                audio_data = convert_audio_format(
                    audio_data, request.sample_rate or SAMPLE_RATE, request.response_format
                )
                return audio_data, f"audio/{request.response_format}"
            if request.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(audio_bytes, request.sample_rate or SAMPLE_RATE, request.response_format)
                    for audio_bytes in audio_generator
                )
            return audio_generator, f"audio/{request.response_format}"


def generate_audio(
    kokoro_tts: Kokoro,
    text: str,
    voice: str,
    *,
    speed: float = 1.0,
    sample_rate: int | None = None,
) -> Generator[bytes]:
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    voice_language = next(v.language for v in VOICES if v.name == voice)
    start = time.perf_counter()
    async_stream = kokoro_tts.create_stream(
        text,
        voice,
        lang=voice_language,
        speed=speed,
    )
    # HACK: converting an async generator to a sync generator
    sync_stream = async_to_sync_generator(async_stream)
    for audio_data, _ in sync_stream:
        assert isinstance(audio_data, np.ndarray) and audio_data.dtype == np.float32 and isinstance(sample_rate, int)
        normalized_audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
        audio_bytes = normalized_audio_data.tobytes()
        if sample_rate != SAMPLE_RATE:
            audio_bytes = resample_audio_bytes(audio_bytes, SAMPLE_RATE, sample_rate)
        yield audio_bytes
    logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start}s")
