"""Tests for Realtime API WebSocket functionality."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest
from pytest_mock import MockerFixture

from speaches.config import Config, WhisperConfig
from speaches.main import create_app
from speaches.realtime.session import create_session_object_configuration
from speaches.realtime.utils import verify_websocket_api_key


class TestRealtimeWebSocketAuthentication:
    """Test WebSocket authentication functionality."""

    @pytest.mark.asyncio
    async def test_websocket_auth_with_bearer_token(self) -> None:
        """Test WebSocket authentication with Authorization Bearer token."""
        # Mock WebSocket
        mock_ws = MagicMock()
        mock_ws.headers = {"authorization": "Bearer test-api-key"}
        mock_ws.query_params = {}

        # Mock config with API key
        config = Config(
            api_key=SecretStr("test-api-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        # Should not raise exception
        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_with_x_api_key(self) -> None:
        """Test WebSocket authentication with X-API-Key header."""
        mock_ws = MagicMock()
        mock_ws.headers = {"x-api-key": "test-api-key"}
        mock_ws.query_params = {}

        config = Config(
            api_key=SecretStr("test-api-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_with_query_param(self) -> None:
        """Test WebSocket authentication with api_key query parameter."""
        mock_ws = MagicMock()
        mock_ws.headers = {}
        mock_ws.query_params = {"api_key": "test-api-key"}

        config = Config(
            api_key=SecretStr("test-api-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_no_api_key_configured(self) -> None:
        """Test WebSocket authentication when no API key is configured."""
        mock_ws = MagicMock()
        mock_ws.headers = {}
        mock_ws.query_params = {}

        config = Config(
            api_key=None,  # No API key configured
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        # Should not raise exception when no API key is configured
        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_invalid_key(self) -> None:
        """Test WebSocket authentication with invalid API key."""
        from fastapi import WebSocketException

        mock_ws = MagicMock()
        mock_ws.headers = {"authorization": "Bearer wrong-key"}
        mock_ws.query_params = {}

        config = Config(
            api_key=SecretStr("correct-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        with pytest.raises(WebSocketException):
            await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_missing_key(self) -> None:
        """Test WebSocket authentication with missing API key."""
        from fastapi import WebSocketException

        mock_ws = MagicMock()
        mock_ws.headers = {}
        mock_ws.query_params = {}

        config = Config(
            api_key=SecretStr("required-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        with pytest.raises(WebSocketException):
            await verify_websocket_api_key(mock_ws, config)


class TestRealtimeSessionConfiguration:
    """Test session configuration for different modes."""

    def test_conversation_mode_default(self) -> None:
        """Test default conversation mode session configuration."""
        session = create_session_object_configuration("gpt-4o-realtime-preview")

        assert session.model == "gpt-4o-realtime-preview"
        assert session.input_audio_transcription.model == "Systran/faster-distil-whisper-small.en"
        assert session.turn_detection is not None and session.turn_detection.create_response is True
        assert session.input_audio_transcription.language is None

    def test_conversation_mode_with_custom_transcription(self) -> None:
        """Test conversation mode with custom transcription model."""
        session = create_session_object_configuration(
            model="gpt-4o-realtime-preview", intent="conversation", transcription_model="whisper-1"
        )

        assert session.model == "gpt-4o-realtime-preview"
        assert session.input_audio_transcription.model == "whisper-1"
        assert session.turn_detection is not None and session.turn_detection.create_response is True

    def test_transcription_only_mode(self) -> None:
        """Test transcription-only mode configuration."""
        session = create_session_object_configuration(
            model="deepdml/faster-whisper-large-v3-turbo-ct2", intent="transcription"
        )

        assert session.model == "gpt-4o-realtime-preview"  # Default conversation model (unused)
        assert session.input_audio_transcription.model == "deepdml/faster-whisper-large-v3-turbo-ct2"
        assert session.turn_detection is not None and session.turn_detection.create_response is False

    def test_transcription_mode_with_language(self) -> None:
        """Test transcription mode with language specification."""
        session = create_session_object_configuration(model="whisper-1", intent="transcription", language="ru")

        assert session.input_audio_transcription.language == "ru"
        assert session.turn_detection is not None and session.turn_detection.create_response is False

    def test_transcription_mode_with_explicit_models(self) -> None:
        """Test transcription mode with explicit transcription model."""
        session = create_session_object_configuration(
            model="gpt-4o-realtime-preview", intent="transcription", transcription_model="custom-whisper-model"
        )

        assert session.model == "gpt-4o-realtime-preview"
        assert session.input_audio_transcription.model == "custom-whisper-model"
        assert session.turn_detection is not None and session.turn_detection.create_response is False

    def test_session_configuration_logging(self, caplog) -> None:  # noqa: ANN001
        """Test that session configuration produces appropriate logging."""
        with caplog.at_level("INFO"):
            create_session_object_configuration(model="test-model", intent="transcription")

        assert "Transcription-only mode" in caplog.text
        assert "test-model" in caplog.text

    def test_conversation_mode_logging(self, caplog) -> None:  # noqa: ANN001
        """Test conversation mode logging."""
        with caplog.at_level("INFO"):
            create_session_object_configuration(model="gpt-4o-realtime-preview", intent="conversation")

        assert "Conversation mode (OpenAI standard)" in caplog.text


class TestRealtimeWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    def test_websocket_endpoint_exists(self, mocker: MockerFixture) -> None:
        """Test that WebSocket endpoint is properly registered."""
        from pydantic import SecretStr

        from speaches.config import Config, WhisperConfig

        # Create config without UI to avoid gradio dependency
        config = Config(
            api_key=None,
            whisper=WhisperConfig(),
            enable_ui=False,  # Disable UI to avoid gradio import
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        # Mock get_config before create_app is called
        mocker.patch("speaches.main.get_config", return_value=config)
        mocker.patch("speaches.dependencies.get_config", return_value=config)

        app = create_app()
        client = TestClient(app)

        # Test that the endpoint exists (will fail auth but endpoint should be found)
        with client.websocket_connect("/v1/realtime?model=test-model") as _:
            # Connection should be closed due to auth failure, but endpoint exists
            pass

    @pytest.mark.asyncio
    async def test_websocket_parameter_parsing(self) -> None:
        """Test that WebSocket parameters are correctly parsed."""
        # This would require more complex mocking of the WebSocket connection
        # For now, we test the session configuration which is the core logic

        # Test default parameters
        session = create_session_object_configuration("test-model")
        assert session.model == "test-model"

        # Test with intent parameter
        session = create_session_object_configuration("test-model", intent="transcription")
        assert session.turn_detection is not None and session.turn_detection.create_response is False

        # Test with all parameters
        session = create_session_object_configuration(
            model="test-model", intent="transcription", language="en", transcription_model="custom-model"
        )
        assert session.input_audio_transcription.model == "custom-model"
        assert session.input_audio_transcription.language == "en"


class TestRealtimeAPICompatibility:
    """Test OpenAI Realtime API compatibility."""

    def test_default_models_configuration(self) -> None:
        """Test that default models are properly configured."""
        session = create_session_object_configuration("gpt-4o-realtime-preview")

        # Check default models match documentation
        assert session.input_audio_transcription.model == "Systran/faster-distil-whisper-small.en"
        assert session.speech_model == "speaches-ai/Kokoro-82M-v1.0-ONNX"
        assert session.voice == "af_heart"

    def test_openai_standard_behavior(self) -> None:
        """Test OpenAI standard behavior is maintained."""
        session = create_session_object_configuration(model="gpt-4o-realtime-preview", intent="conversation")

        # OpenAI standard: model param is conversation model
        assert session.model == "gpt-4o-realtime-preview"
        # Default transcription model is used
        assert session.input_audio_transcription.model == "Systran/faster-distil-whisper-small.en"
        # Response generation is enabled
        assert session.turn_detection is not None and session.turn_detection.create_response is True

    def test_speaches_extension_behavior(self) -> None:
        """Test Speaches extension behavior for transcription-only mode."""
        session = create_session_object_configuration(model="custom-whisper-model", intent="transcription")

        # Speaches extension: model param is transcription model
        assert session.input_audio_transcription.model == "custom-whisper-model"
        # Default conversation model is used (but unused)
        assert session.model == "gpt-4o-realtime-preview"
        # Response generation is disabled
        assert session.turn_detection is not None and session.turn_detection.create_response is False

    def test_session_structure_compatibility(self) -> None:
        """Test that session structure matches expected OpenAI format."""
        session = create_session_object_configuration("test-model")

        # Check required fields exist
        assert hasattr(session, "id")
        assert hasattr(session, "model")
        assert hasattr(session, "modalities")
        assert hasattr(session, "input_audio_transcription")
        assert hasattr(session, "turn_detection")
        assert hasattr(session, "speech_model")
        assert hasattr(session, "voice")

        # Check types
        assert isinstance(session.modalities, list)
        assert "audio" in session.modalities
        assert "text" in session.modalities
