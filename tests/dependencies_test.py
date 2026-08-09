import io

from fastapi import UploadFile
import numpy as np
import pytest

from speaches.audio import Audio
from speaches.dependencies import audio_file_dependency


def test_zero_duration_audio_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for https://github.com/speaches-ai/speaches/issues/667
    # A well-formed but zero-duration container decodes to an empty sample array;
    # decode_audio() succeeds, so the av.error.* handlers are not triggered. The
    # RTF debug log line then eagerly evaluated `elapsed / audio.duration` and
    # crashed with ZeroDivisionError — even when debug logging was disabled.
    monkeypatch.setattr(
        "speaches.dependencies.decode_audio",
        lambda _file, **_kwargs: np.array([], dtype=np.float32),
    )
    upload_file = UploadFile(file=io.BytesIO(b""), filename="empty.wav")

    audio = audio_file_dependency(upload_file)

    assert isinstance(audio, Audio)
    assert audio.duration == 0.0
