from __future__ import annotations

import asyncio
import importlib
import re
from types import SimpleNamespace

import pytest
from harnify_agent.types import AgentToolResult
from harnify_ai.types import AssistantMessage, TextContent, ToolCall, Usage, UsageCost
from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.edit_diff import EditDiffResult
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.tools.truncate import TruncationResult
from harnify_coding_agent.modes.interactive.components import (
    AssistantMessageComponent,
    BashExecutionComponent,
    ToolExecutionComponent,
    UserMessageComponent,
    truncate_to_visual_lines,
)
import harnify_coding_agent.modes.interactive.components.assistant_message as assistant_message_module
import harnify_coding_agent.modes.interactive.components.bash_execution as bash_execution_module
from harnify_tui import Text, setKeybindings, visibleWidth

OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"
BG_RESET = "\x1b[49m"
_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)
interactive_theme_module = importlib.import_module("harnify_coding_agent.modes.interactive.theme.theme")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _assistant_message(content: list[TextContent | ToolCall | dict[str, object]]) -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api="openai-responses",
        provider="openai",
        model="demo",
        usage=Usage(
            input=0,
            output=0,
            cacheRead=0,
            cacheWrite=0,
            totalTokens=0,
            cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
        ),
        stopReason="stop",
        timestamp=0,
    )


class FakeUi:
    def __init__(self) -> None:
        self.render_calls = 0
        self.terminal = SimpleNamespace(columns=120, rows=40)

    def requestRender(self) -> None:
        self.render_calls += 1


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    interactive_theme_module.init_theme("dark")


def test_assistant_message_adds_osc_markers_without_tool_calls() -> None:
    component = AssistantMessageComponent(_assistant_message([TextContent(text="hello")]))
    lines = component.render(40)

    assert lines
    assert OSC133_ZONE_START in lines[0]
    assert lines[-1].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)


def test_assistant_message_omits_osc_markers_for_tool_calls_and_can_hide_thinking() -> None:
    component = AssistantMessageComponent(
        _assistant_message(
            [
                {"type": "thinking", "thinking": "secret plan"},
                TextContent(text="calling tool"),
                ToolCall(id="tool-1", name="read", arguments={"path": "demo.txt"}),
            ]
        ),
        hideThinkingBlock=True,
        hiddenThinkingLabel="Thinking hidden",
    )

    rendered = "\n".join(component.render(80))
    stripped = _strip_ansi(rendered)

    assert OSC133_ZONE_START not in rendered
    assert OSC133_ZONE_END not in rendered
    assert "Thinking hidden" in stripped
    assert "secret plan" not in stripped


def test_user_message_keeps_box_height_and_prefixes_closing_zone_markers() -> None:
    component = UserMessageComponent("hello")
    lines = component.render(20)

    assert len(lines) == 3
    assert OSC133_ZONE_START in lines[0]
    assert lines[0].endswith(BG_RESET)
    assert OSC133_ZONE_END not in lines[0]
    assert "hello" in _strip_ansi(lines[1])
    assert lines[2].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)


def test_assistant_message_module_exports_match_ts_surface() -> None:
    assert assistant_message_module.__all__ == ["AssistantMessageComponent"]


def test_bash_execution_module_exports_match_ts_surface() -> None:
    assert bash_execution_module.__all__ == ["BashExecutionComponent"]


def test_truncate_to_visual_lines_counts_wrapped_lines() -> None:
    result = truncate_to_visual_lines("x" * 100, 2, 20, 0)

    assert len(result.visualLines) == 2
    assert result.skippedCount > 0


def test_bash_execution_collapsed_preview_respects_render_time_width() -> None:
    component = BashExecutionComponent("pwd", FakeUi())
    long_line = "x" * 150
    component.appendOutput(f"{long_line}\n{long_line}\n")
    component.setComplete(0, False)

    lines_200 = component.render(200)
    lines_60 = component.render(60)

    for line in lines_200:
        assert visibleWidth(line) <= 200
    for line in lines_60:
        assert visibleWidth(line) <= 60


@pytest.mark.asyncio
async def test_tool_execution_uses_custom_renderers_and_shared_state() -> None:
    async def execute(
        _tool_call_id: str,
        _params: object,
        _signal: object | None,
        _on_update: object | None,
        _ctx: object | None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    definition = ToolDefinition(
        name="custom_tool",
        label="custom_tool",
        description="custom tool",
        parameters={},
        execute=execute,
        renderCall=lambda _args, _theme, context: Text(
            f"custom call {context.state.setdefault('token', 'shared-token')}",
            0,
            0,
        ),
        renderResult=lambda _result, _options, _theme, context: Text(
            f"custom result {context.state.get('token')} arg:{context.args['foo']}",
            0,
            0,
        ),
    )

    component = ToolExecutionComponent(
        "custom_tool",
        "tool-1",
        {"foo": "bar"},
        {},
        definition,
        FakeUi(),
        ".",
    )
    component.updateResult({"content": [{"type": "text", "text": "done"}], "details": {}, "isError": False}, False)
    rendered = _strip_ansi("\n".join(component.render(120)))

    assert "custom call shared-token" in rendered
    assert "custom result shared-token arg:bar" in rendered


@pytest.mark.asyncio
async def test_tool_execution_generic_fallback_includes_args_and_output() -> None:
    async def execute(
        _tool_call_id: str,
        _params: object,
        _signal: object | None,
        _on_update: object | None,
        _ctx: object | None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="done")], details={})

    definition = ToolDefinition(
        name="custom_tool",
        label="custom_tool",
        description="custom tool",
        parameters={},
        execute=execute,
    )

    component = ToolExecutionComponent(
        "custom_tool",
        "tool-2",
        {"foo": "bar"},
        {},
        definition,
        FakeUi(),
        ".",
    )
    component.updateResult({"content": [{"type": "text", "text": "done"}], "details": {}, "isError": False}, False)
    rendered = _strip_ansi("\n".join(component.render(80)))

    assert "custom_tool" in rendered
    assert '"foo": "bar"' in rendered
    assert "done" in rendered


def test_tool_execution_uses_builtin_bash_renderer() -> None:
    component = ToolExecutionComponent(
        "bash",
        "tool-3",
        {"command": "printf 'alpha\\n'", "timeout": 5},
        {},
        None,
        FakeUi(),
        ".",
    )
    component.markExecutionStarted()
    component.updateResult({"content": [{"type": "text", "text": "alpha\n"}], "details": None, "isError": False}, False)
    rendered = _strip_ansi("\n".join(component.render(120)))

    assert "$ printf 'alpha\\n'" in rendered
    assert "(timeout 5s)" in rendered
    assert "alpha" in rendered
    assert "Took " in rendered


def test_builtin_bash_renderer_dedupes_truncation_footer() -> None:
    truncation = TruncationResult(
        content="line 1\nline 2",
        truncated=True,
        truncatedBy="lines",
        totalLines=10,
        totalBytes=60,
        outputLines=2,
        outputBytes=13,
        lastLinePartial=False,
        firstLineExceedsLimit=False,
        maxLines=2000,
        maxBytes=51200,
    )
    component = ToolExecutionComponent(
        "bash",
        "tool-4",
        {"command": "tail -n 2"},
        {},
        None,
        FakeUi(),
        ".",
    )
    component.markExecutionStarted()
    component.updateResult(
        {
            "content": [
                {
                    "type": "text",
                    "text": "line 1\nline 2\n\n[Showing lines 9-10 of 10. Full output: /tmp/full.log]",
                }
            ],
            "details": {"truncation": truncation, "fullOutputPath": "/tmp/full.log"},
            "isError": False,
        },
        False,
    )
    rendered = _strip_ansi("\n".join(component.render(120)))

    assert rendered.count("Full output: /tmp/full.log") == 1
    assert "Truncated: showing 2 of 10 lines" in rendered


@pytest.mark.asyncio
async def test_tool_execution_uses_builtin_edit_preview_renderer(tmp_path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    component = ToolExecutionComponent(
        "edit",
        "tool-edit-1",
        {"path": str(target), "edits": [{"oldText": "world\n", "newText": "WORLD\n"}]},
        {},
        None,
        FakeUi(),
        str(tmp_path),
    )
    component.setArgsComplete()
    component.render(120)
    await asyncio.sleep(0.05)
    rendered = _strip_ansi("\n".join(component.render(120)))

    assert "edit" in rendered
    assert str(target) in rendered
    assert "-2 world" in rendered
    assert "+2 WORLD" in rendered


@pytest.mark.asyncio
async def test_tool_execution_edit_result_dedupes_preview_diff(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    preview = EditDiffResult(diff="-2 world\n+2 WORLD", firstChangedLine=2)

    async def fake_compute_edits_diff(_path, _edits, _cwd):
        return preview

    monkeypatch.setattr(
        "harnify_coding_agent.core.tools.edit.compute_edits_diff",
        fake_compute_edits_diff,
    )

    component = ToolExecutionComponent(
        "edit",
        "tool-edit-2",
        {"path": str(tmp_path / "demo.txt"), "edits": [{"oldText": "world\n", "newText": "WORLD\n"}]},
        {},
        None,
        FakeUi(),
        str(tmp_path),
    )
    component.setArgsComplete()
    component.render(120)
    await asyncio.sleep(0)
    component.updateResult(
        {
            "content": [{"type": "text", "text": "Successfully replaced 1 block(s) in demo.txt."}],
            "details": {"diff": "-2 world\n+2 WORLD", "patch": "--- demo\n+++ demo", "firstChangedLine": 2},
            "isError": False,
        },
        False,
    )
    rendered = _strip_ansi("\n".join(component.render(120)))

    assert rendered.count("-2 world") == 1
    assert rendered.count("+2 WORLD") == 1
