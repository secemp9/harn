from __future__ import annotations

from harnify_coding_agent.core.compaction import CompactionResult
from harnify_coding_agent.core.extensions import types as extension_types


def test_extension_types_export_event_and_result_surface() -> None:
    context_usage: extension_types.ContextUsage = {
        "tokens": 128,
        "contextWindow": 2048,
        "percent": 6.25,
    }
    compact_options: extension_types.CompactOptions = {
        "customInstructions": "Focus on active blockers",
        "onComplete": lambda _result: None,
        "onError": lambda _error: None,
    }
    message_options: extension_types.MessageRenderOptions = {"expanded": True}
    tool_render_options: extension_types.ToolRenderResultOptions = {"expanded": False, "isPartial": True}
    discover_event: extension_types.ResourcesDiscoverEvent = {
        "type": "resources_discover",
        "cwd": "/tmp/project",
        "reason": "startup",
    }
    discover_result: extension_types.ResourcesDiscoverResult = {"skillPaths": ["/tmp/skills"]}
    before_compact: extension_types.SessionBeforeCompactEvent = {
        "type": "session_before_compact",
        "preparation": object(),
        "branchEntries": [],
        "signal": object(),
    }
    before_compact_result: extension_types.SessionBeforeCompactResult = {
        "compaction": CompactionResult(
            summary="## Goal\nCheckpoint",
            firstKeptEntryId="entry-1",
            tokensBefore=512,
            details=None,
        )
    }
    before_agent_start: extension_types.BeforeAgentStartEvent = {
        "type": "before_agent_start",
        "prompt": "run tests",
        "systemPrompt": "system",
        "systemPromptOptions": {"cwd": "/tmp/project"},
    }
    before_agent_start_result: extension_types.BeforeAgentStartEventResult = {
        "message": {"customType": "notice", "content": "hello"},
        "systemPrompt": "override",
    }
    input_event: extension_types.InputEvent = {
        "type": "input",
        "text": "/review",
        "source": "interactive",
    }
    input_result: extension_types.InputEventResult = {"action": "transform", "text": "review changed"}
    tool_call: extension_types.ToolCallEvent = {
        "type": "tool_call",
        "toolCallId": "call-1",
        "toolName": "bash",
        "input": {"command": "pwd"},
    }
    tool_call_result: extension_types.ToolCallEventResult = {"block": True, "reason": "policy"}
    tool_result: extension_types.ToolResultEvent = {
        "type": "tool_result",
        "toolCallId": "call-1",
        "toolName": "bash",
        "input": {"command": "pwd"},
        "content": [{"type": "text", "text": "ok"}],
        "details": None,
        "isError": False,
    }
    tool_result_result: extension_types.ToolResultEventResult = {
        "content": [{"type": "text", "text": "rewritten"}],
        "isError": False,
    }
    session_tree: extension_types.SessionTreeEvent = {
        "type": "session_tree",
        "newLeafId": "leaf-new",
        "oldLeafId": "leaf-old",
    }
    extension_event: extension_types.ExtensionEvent = session_tree
    user_bash_result: extension_types.UserBashEventResult = {"operations": object()}

    assert context_usage["contextWindow"] == 2048
    assert compact_options["customInstructions"] == "Focus on active blockers"
    assert message_options["expanded"] is True
    assert tool_render_options["isPartial"] is True
    assert discover_event["reason"] == "startup"
    assert discover_result["skillPaths"] == ["/tmp/skills"]
    assert before_compact["type"] == "session_before_compact"
    assert before_compact_result["compaction"].tokensBefore == 512
    assert before_agent_start["prompt"] == "run tests"
    assert before_agent_start_result["systemPrompt"] == "override"
    assert input_event["source"] == "interactive"
    assert input_result["action"] == "transform"
    assert tool_call["toolName"] == "bash"
    assert tool_call_result["block"] is True
    assert tool_result["isError"] is False
    assert tool_result_result["content"][0]["text"] == "rewritten"
    assert extension_event["type"] == "session_tree"
    assert "operations" in user_bash_result

    for exported_name in (
        "ExtensionEvent",
        "InputEventResult",
        "ResourcesDiscoverEvent",
        "SessionBeforeCompactResult",
        "ToolCallEvent",
        "ToolRenderResultOptions",
    ):
        assert exported_name in extension_types.__all__
