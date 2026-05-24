from __future__ import annotations

from typing import Any

import pytest
from harnify_agent import AgentTool, AgentToolResult
from harnify_agent.harness.agent_harness import AgentHarness
from harnify_ai.providers.faux import faux_assistant_message, faux_tool_call
from harnify_ai.types import TextContent

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"],
    "additionalProperties": False,
}


def _assistant_text(message: Any) -> str:
    return "\n".join(block.text for block in getattr(message, "content", []) if getattr(block, "type", None) == "text")


def calculate_tool(executed: list[str]) -> AgentTool:
    async def execute(_tool_call_id: str, params: Any, _signal=None, _on_update=None) -> AgentToolResult:
        result = str(eval(params["expression"], {"__builtins__": {}}, {}))
        executed.append(result)
        return AgentToolResult(content=[TextContent(text=result)], details={"result": result})

    return AgentTool(
        name="calculate",
        label="Calculate",
        description="Evaluate simple arithmetic expressions",
        parameters=TOOL_SCHEMA,
        execute=execute,
    )


@pytest.mark.asyncio
async def test_faux_provider_agent_harness_integration_exercises_tool_and_jsonl_session(
    faux_provider_factory,
    jsonl_session_factory,
    session_scaffold,
) -> None:
    registration = faux_provider_factory({"api": "faux-integration", "tokenSize": {"min": 32, "max": 32}})
    registration.set_responses(
        [
            faux_assistant_message(
                faux_tool_call("calculate", {"expression": "2 + 2"}, options={"id": "call-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message("final answer"),
        ]
    )
    model = registration.get_model()
    assert model is not None

    session = await jsonl_session_factory()
    executed: list[str] = []
    event_types: list[str] = []
    harness = AgentHarness(
        {
            "env": session_scaffold.env,
            "session": session,
            "model": model,
            "tools": [calculate_tool(executed)],
        }
    )
    harness.subscribe(lambda event, _signal: event_types.append(event.type))

    response = await harness.prompt("hello")
    context = await session.buildContext()
    roles = [getattr(message, "role", None) for message in context.messages]
    metadata = await session.getMetadata()

    assert executed == ["4"]
    assert _assistant_text(response) == "final answer"
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types
    assert "agent_end" in event_types
    assert roles == ["user", "assistant", "toolResult", "assistant"]
    assert _assistant_text(context.messages[-1]) == "final answer"
    assert metadata["cwd"] == str(session_scaffold.cwd)
    assert metadata["path"].startswith(str(session_scaffold.sessions_root))
