import logging
from unittest.mock import MagicMock, patch

import pytest

from speaches.realtime.context import _VALID_TRANSITIONS, SessionContext
from speaches.types.realtime import ConversationState


def _make_session_context() -> SessionContext:
    mock_registry = MagicMock()
    mock_completion_client = MagicMock()
    mock_vad_manager = MagicMock()
    mock_tts = MagicMock()
    mock_stt = MagicMock()
    mock_session = MagicMock()
    mock_session.speech_model = "test-tts"
    mock_session.input_audio_transcription.model = "test-stt"

    return SessionContext(
        executor_registry=mock_registry,
        completion_client=mock_completion_client,
        vad_model_manager=mock_vad_manager,
        vad_model_id="test-vad",
        session=mock_session,
        tts_model_manager=mock_tts,
        stt_model_manager=mock_stt,
    )


def test_valid_transitions() -> None:
    ctx = _make_session_context()
    for source, targets in _VALID_TRANSITIONS.items():
        for target in targets:
            ctx._state = source  # noqa: SLF001
            ctx.state = target
            assert ctx.state == target, f"Expected state {target} after transition from {source}"


def test_invalid_transition_raises_in_debug() -> None:
    ctx = _make_session_context()
    all_states = set(ConversationState)
    with patch.object(
        logging.getLogger("speaches.realtime.context"),
        "isEnabledFor",
        return_value=True,
    ):
        for source, valid_targets in _VALID_TRANSITIONS.items():
            invalid_targets = all_states - valid_targets - {source}
            for target in invalid_targets:
                ctx._state = source  # noqa: SLF001
                with pytest.raises(RuntimeError, match="Unexpected state transition"):
                    ctx.state = target


def test_same_state_is_noop() -> None:
    ctx = _make_session_context()
    with patch.object(
        logging.getLogger("speaches.realtime.context"),
        "isEnabledFor",
        return_value=True,
    ):
        for s in ConversationState:
            ctx._state = s  # noqa: SLF001
            ctx.state = s
            assert ctx.state == s


def test_invalid_transition_warns_in_production() -> None:
    ctx = _make_session_context()
    all_states = set(ConversationState)
    context_logger = logging.getLogger("speaches.realtime.context")
    with (
        patch.object(context_logger, "isEnabledFor", return_value=False),
        patch.object(context_logger, "warning") as mock_warning,
    ):
        for source, valid_targets in _VALID_TRANSITIONS.items():
            invalid_targets = all_states - valid_targets - {source}
            for target in invalid_targets:
                ctx._state = source  # noqa: SLF001
                mock_warning.reset_mock()
                ctx.state = target
                assert ctx.state == target, f"State should have changed to {target} in production mode"
                mock_warning.assert_called_once()
                assert "Unexpected state transition" in mock_warning.call_args[0][0]
