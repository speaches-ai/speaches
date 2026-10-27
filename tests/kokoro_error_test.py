from collections.abc import Iterator
from contextlib import contextmanager

from kokoro_onnx import Kokoro
import numpy as np
import pytest

from speaches.config import OrtOptions
from speaches.executors.kokoro import KokoroModelManager
from speaches.executors.shared.handler_protocol import SpeechRequest


class FakeTokenizer:
    def phonemize(self, text: str, lang: str) -> str:  # noqa: ARG002
        return text


def make_fake_kokoro(fail_on_batch: int) -> Kokoro:
    tts = Kokoro.__new__(Kokoro)
    tts.tokenizer = FakeTokenizer()  # pyrefly: ignore[bad-assignment]
    tts.voices = {"af_heart": np.zeros(1, dtype=np.float32)}  # pyrefly: ignore[bad-assignment]
    calls = {"count": 0}

    def fake_create_audio(phonemes: str, voice: object, speed: float) -> tuple[np.ndarray, int]:  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == fail_on_batch:
            msg = "Failed to allocate memory"
            raise RuntimeError(msg)
        return np.ones(2400, dtype=np.float32), 24000

    tts._create_audio = fake_create_audio  # type: ignore[method-assign]  # noqa: SLF001
    return tts


@pytest.mark.timeout(30)
@pytest.mark.parametrize("fail_on_batch", [1, 2])
def test_kokoro_synthesis_error_is_raised_not_hung(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fail_on_batch: int
) -> None:
    manager = KokoroModelManager(ttl=0, ort_opts=OrtOptions())
    tts = make_fake_kokoro(fail_on_batch)

    @contextmanager
    def fake_load_model(_model_id: str) -> Iterator[Kokoro]:
        yield tts

    monkeypatch.setattr(manager, "load_model", fake_load_model)
    request = SpeechRequest(model="fake/kokoro", voice="af_heart", text="Hello there. General Kenobi.", speed=1.0)
    # force two batches so the failure can also happen mid-stream
    monkeypatch.setattr(tts, "_split_phonemes", lambda p: p.split(". "), raising=False)

    with pytest.raises(RuntimeError, match="Failed to allocate memory"):
        list(manager.handle_speech_request(request))
    assert "Kokoro speech synthesis failed" in caplog.text
