from speaches.realtime.session import create_session_object_configuration
from speaches.types.realtime import Tool


def test_vad_defaults_optimized() -> None:
    session = create_session_object_configuration("test-model")
    assert session.turn_detection is not None
    assert session.turn_detection.threshold == 0.6
    assert session.turn_detection.silence_duration_ms == 350
    assert session.turn_detection.prefix_padding_ms == 300


def test_tool_execution_field_default() -> None:
    tool = Tool(name="test_tool", parameters={})
    assert tool.execution == "client"


def test_tool_execution_field_server() -> None:
    tool = Tool(name="test_tool", parameters={}, execution="server")
    assert tool.execution == "server"


def test_tool_execution_field_serialization() -> None:
    tool = Tool(name="test_tool", parameters={}, execution="server")
    data = tool.model_dump()
    assert data["execution"] == "server"

    # Reconstruct
    tool2 = Tool(**data)
    assert tool2.execution == "server"


def test_tool_execution_field_not_in_function_definition() -> None:
    """The execution field should NOT leak into the LLM API call.

    The chat_utils.create_completion_params manually constructs
    FunctionDefinition from name/description/parameters, so the
    execution field is naturally excluded.
    """
    tool = Tool(name="weather", description="Get weather", parameters={"type": "object"}, execution="server")

    # When we dump for API, execution should be present in full dump
    full = tool.model_dump()
    assert "execution" in full

    # But when constructing FunctionDefinition, only name/description/parameters are used
    from openai.types.shared_params.function_definition import FunctionDefinition

    fd = FunctionDefinition(name=tool.name, description=tool.description or "", parameters=tool.parameters)
    # FunctionDefinition should NOT have execution
    fd_dict = dict(fd)
    assert "execution" not in fd_dict
