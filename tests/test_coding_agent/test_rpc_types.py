from __future__ import annotations

from typing import get_args

from harnify_ai.types import ModelCost
from harnify_coding_agent.core.bash_executor import BashResult
from harnify_coding_agent.core.source_info import SourceInfo
from harnify_coding_agent.modes.rpc.rpc_types import (
    RpcAbortCommand,
    RpcBashResponse,
    RpcCommandType,
    RpcErrorResponse,
    RpcExtensionUIRequest,
    RpcExtensionUIResponse,
    RpcGetStateResponse,
    RpcSessionState,
    RpcSetModelCommand,
    RpcSlashCommand,
)


def test_rpc_types_expose_discriminated_payload_shapes() -> None:
    model = {
        "id": "claude",
        "name": "claude",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://example.test",
        "reasoning": True,
        "input": ["text"],
        "cost": ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        "contextWindow": 200_000,
        "maxTokens": 8_192,
    }
    state = RpcSessionState(
        model=model,
        thinkingLevel="high",
        isStreaming=False,
        isCompacting=False,
        steeringMode="all",
        followUpMode="one-at-a-time",
        sessionId="session-1",
        autoCompactionEnabled=True,
        messageCount=4,
        pendingMessageCount=0,
    )
    slash_command = RpcSlashCommand(
        name="demo",
        source="extension",
        sourceInfo=SourceInfo(
            path="/tmp/demo.py",
            source="extension",
            scope="temporary",
            origin="top-level",
        ),
    )

    assert RpcAbortCommand(type="abort")["type"] == "abort"
    assert RpcSetModelCommand(type="set_model", provider="anthropic", modelId="claude")["modelId"] == "claude"
    assert RpcGetStateResponse(type="response", command="get_state", success=True, data=state)["data"]["sessionId"] == "session-1"
    assert RpcBashResponse(
        type="response",
        command="bash",
        success=True,
        data=BashResult(output="ok", exitCode=0, cancelled=False, truncated=False),
    )["data"].output == "ok"
    assert RpcErrorResponse(type="response", command="set_model", success=False, error="boom")["error"] == "boom"

    ui_request: RpcExtensionUIRequest = {
        "type": "extension_ui_request",
        "id": "req-1",
        "method": "select",
        "title": "Pick one",
        "options": ["A", "B"],
    }
    ui_response: RpcExtensionUIResponse = {
        "type": "extension_ui_response",
        "id": "req-1",
        "value": "A",
    }
    assert ui_request["method"] == "select"
    assert ui_response["value"] == "A"
    assert slash_command["sourceInfo"].path == "/tmp/demo.py"

    command_type_value = getattr(RpcCommandType, "__value__", RpcCommandType)
    assert "prompt" in get_args(command_type_value)
    assert "get_commands" in get_args(command_type_value)
