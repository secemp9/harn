from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from harnify_coding_agent.core.export_html import (
    ansi_lines_to_html,
    ansi_to_html,
    create_tool_html_renderer,
    export_from_file,
    export_session_to_html,
    get_resolved_theme_colors,
    get_theme_export_colors,
    trim_rendered_result_lines,
)
from harnify_coding_agent.core.session_manager import SessionManager


def test_export_html_module_exports_match_ts_surface() -> None:
    import importlib

    module = importlib.import_module("harnify_coding_agent.core.export_html")
    assert module.__all__ == [
        "ExportOptions",
        "ToolHtmlRenderer",
        "exportFromFile",
        "exportSessionToHtml",
    ]


def test_tool_renderer_module_exports_match_ts_surface() -> None:
    import importlib

    module = importlib.import_module("harnify_coding_agent.core.export_html.tool_renderer")
    assert module.__all__ == [
        "ToolHtmlRenderer",
        "ToolHtmlRendererDeps",
        "createToolHtmlRenderer",
    ]


def test_ansi_to_html_handles_styles_and_escaping() -> None:
    rendered = ansi_to_html('plain \x1b[31m<red>\x1b[0m \x1b[1;4m"\'&"\x1b[0m')
    assert "plain " in rendered
    assert '<span style="color:#800000">&lt;red&gt;</span>' in rendered
    assert 'font-weight:bold' in rendered
    assert 'text-decoration:underline' in rendered
    assert "&quot;&#039;&amp;&quot;" in rendered


def test_ansi_to_html_handles_extended_colors_and_blank_lines() -> None:
    rendered = ansi_to_html("\x1b[38;5;196mhello\x1b[0m \x1b[48;2;1;2;3mrgb\x1b[0m")
    assert 'color:#ff0000' in rendered
    assert 'background-color:rgb(1,2,3)' in rendered

    lines_html = ansi_lines_to_html(["", "ok"])
    assert '<div class="ansi-line">&nbsp;</div>' in lines_html
    assert '<div class="ansi-line">ok</div>' in lines_html


def test_ansi_to_html_module_exports_match_ts_surface() -> None:
    import importlib

    module = importlib.import_module("harnify_coding_agent.core.export_html.ansi_to_html")
    assert module.__all__ == [
        "ansiLinesToHtml",
        "ansiToHtml",
    ]


def test_trim_rendered_result_lines_ignores_ansi_only_padding() -> None:
    trimmed = trim_rendered_result_lines(["\x1b[31m\x1b[0m", "  ", "\x1b[32mvalue\x1b[0m", "\x1b[0m"])
    assert trimmed == ["\x1b[32mvalue\x1b[0m"]


@dataclass(slots=True)
class _FakeComponent:
    lines: list[str]

    def render(self, _width: int) -> list[str]:
        return list(self.lines)


class _FakeToolDefinition:
    def __init__(self) -> None:
        self.call_contexts: list[Any] = []
        self.result_contexts: list[Any] = []

    def renderCall(self, args: Any, theme: Any, ctx: Any) -> _FakeComponent:
        self.call_contexts.append((args, theme, ctx))
        return _FakeComponent(["\x1b[36mcall\x1b[0m"])

    def renderResult(self, result: Any, options: Any, theme: Any, ctx: Any) -> _FakeComponent:
        self.result_contexts.append((result, options, theme, ctx))
        label = "expanded" if options["expanded"] else "collapsed"
        return _FakeComponent(["", f"\x1b[33m{label}\x1b[0m", ""])


def test_tool_html_renderer_renders_calls_and_results() -> None:
    tool_def = _FakeToolDefinition()
    renderer = create_tool_html_renderer(
        {
            "getToolDefinition": lambda name: tool_def if name == "demo" else None,
            "theme": {"name": "demo-theme"},
            "cwd": "/tmp/project",
            "width": 80,
        }
    )

    call_html = renderer.renderCall("tool-1", "demo", {"path": "demo.txt"})
    assert call_html == '<div class="ansi-line"><span style="color:#008080">call</span></div>'

    rendered = renderer.renderResult(
        "tool-1",
        "demo",
        [{"type": "text", "text": "done"}],
        {"code": 0},
        False,
    )
    assert rendered is not None
    assert rendered.collapsed == '<div class="ansi-line"><span style="color:#808000">collapsed</span></div>'
    assert rendered.expanded == '<div class="ansi-line"><span style="color:#808000">expanded</span></div>'

    args, theme, ctx = tool_def.call_contexts[0]
    assert args == {"path": "demo.txt"}
    assert theme == {"name": "demo-theme"}
    assert ctx.cwd == "/tmp/project"
    assert ctx.toolCallId == "tool-1"
    assert ctx.args == {"path": "demo.txt"}

    first_result, first_options, _, first_ctx = tool_def.result_contexts[0]
    assert first_result["details"] == {"code": 0}
    assert first_options == {"expanded": False, "isPartial": False}
    assert first_ctx.isError is False


def test_tool_html_renderer_falls_back_on_missing_or_broken_renderers() -> None:
    class _BrokenToolDefinition:
        def renderCall(self, _args: Any, _theme: Any, _ctx: Any) -> Any:
            raise RuntimeError("boom")

        def renderResult(self, _result: Any, _options: Any, _theme: Any, _ctx: Any) -> Any:
            raise RuntimeError("boom")

    renderer = create_tool_html_renderer(
        {
            "getToolDefinition": lambda name: _BrokenToolDefinition() if name == "broken" else object(),
            "theme": None,
            "cwd": "/tmp",
        }
    )

    assert renderer.renderCall("tool-2", "missing", {}) is None
    assert renderer.renderCall("tool-2", "broken", {}) is None
    assert renderer.renderResult("tool-2", "missing", [], None, False) is None
    assert renderer.renderResult("tool-2", "broken", [], None, False) is None


def test_theme_helpers_resolve_builtin_colors() -> None:
    dark_colors = get_resolved_theme_colors("dark")
    light_colors = get_resolved_theme_colors("light")
    dark_export = get_theme_export_colors("dark")
    light_export = get_theme_export_colors("light")

    assert dark_colors["accent"] == "#8abeb7"
    assert dark_colors["userMessageBg"] == "#343541"
    assert light_colors["text"] == "#1f2328"
    assert dark_export["pageBg"] == "#18181e"
    assert light_export["cardBg"] == "#ffffff"


def _extract_session_payload(html: str) -> dict[str, Any]:
    match = re.search(r'<script id="session-data" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None
    return json.loads(base64.b64decode(match.group(1)).decode("utf-8"))


@pytest.mark.asyncio
async def test_export_session_to_html_writes_html_and_payload(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    session_dir = tmp_path / "sessions"
    session_manager = SessionManager.create(str(cwd), str(session_dir))
    session_manager.appendMessage({"role": "assistant", "content": [{"type": "text", "text": "hello"}]})

    output_path = await export_session_to_html(
        session_manager,
        {
            "systemPrompt": "system rules",
            "tools": [{"name": "demo", "description": "Demo tool", "parameters": {"type": "object"}}],
        },
        {"themeName": "dark"},
    )

    html = Path(output_path).read_text(encoding="utf-8")
    payload = _extract_session_payload(html)

    assert Path(output_path).exists()
    assert Path(output_path).name.startswith("harnify-session-")
    assert "<!DOCTYPE html>" in html
    assert "--accent: #8abeb7;" in html
    assert payload["systemPrompt"] == "system rules"
    assert payload["tools"][0]["name"] == "demo"
    assert payload["entries"][0]["message"]["content"][0]["text"] == "hello"
    assert payload["leafId"] == session_manager.getLeafId()


@pytest.mark.asyncio
async def test_export_from_file_honors_output_path_and_renders_custom_tool_html(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    session_dir = tmp_path / "sessions"
    session_manager = SessionManager.create(str(cwd), str(session_dir))
    session_manager.appendMessage(
        {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "tool-1", "name": "custom", "arguments": {"path": "x.txt"}}],
        }
    )
    session_manager.appendMessage(
        {
            "role": "toolResult",
            "toolCallId": "tool-1",
            "toolName": "custom",
            "content": [{"type": "text", "text": "done"}],
            "details": {"ok": True},
            "isError": False,
        }
    )

    class _Renderer:
        def renderCall(self, toolCallId: str, toolName: str, args: Any) -> str | None:
            assert toolCallId == "tool-1"
            assert toolName == "custom"
            assert args == {"path": "x.txt"}
            return "<div>call html</div>"

        def renderResult(
            self,
            toolCallId: str,
            toolName: str,
            result: list[dict[str, Any]],
            details: Any,
            isError: bool,
        ) -> Any:
            assert toolCallId == "tool-1"
            assert toolName == "custom"
            assert result == [{"type": "text", "text": "done"}]
            assert details == {"ok": True}
            assert isError is False
            return type("Rendered", (), {"collapsed": "<div>collapsed</div>", "expanded": "<div>expanded</div>"})()

    live_output = await export_session_to_html(
        session_manager,
        None,
        {"themeName": "light", "toolRenderer": _Renderer(), "outputPath": str(tmp_path / "live.html")},
    )
    live_payload = _extract_session_payload(Path(live_output).read_text(encoding="utf-8"))
    assert live_payload["renderedTools"]["tool-1"]["callHtml"] == "<div>call html</div>"
    assert live_payload["renderedTools"]["tool-1"]["resultHtmlExpanded"] == "<div>expanded</div>"

    session_file = session_manager.getSessionFile()
    assert session_file is not None
    exported_path = await export_from_file(
        session_file,
        {"outputPath": str(tmp_path / "exported.html"), "themeName": "light"},
    )
    html = Path(exported_path).read_text(encoding="utf-8")
    payload = _extract_session_payload(html)

    assert exported_path == str(tmp_path / "exported.html")
    assert "--exportPageBg: #f8f8f8;" in html
    assert payload["entries"][0]["message"]["content"][0]["name"] == "custom"
    assert "systemPrompt" not in payload
    assert "tools" not in payload


@pytest.mark.asyncio
async def test_export_from_file_missing_input_raises_runtime_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(RuntimeError, match=rf"File not found: {re.escape(str(missing))}"):
        await export_from_file(str(missing))


@pytest.mark.asyncio
async def test_export_session_to_html_preserves_empty_rendered_tool_result_strings(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    session_dir = tmp_path / "sessions"
    session_manager = SessionManager.create(str(cwd), str(session_dir))
    session_manager.appendMessage(
        {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "tool-2", "name": "custom", "arguments": {}}],
        }
    )
    session_manager.appendMessage(
        {
            "role": "toolResult",
            "toolCallId": "tool-2",
            "toolName": "custom",
            "content": [{"type": "text", "text": "done"}],
            "details": {},
            "isError": False,
        }
    )

    class _EmptyRenderer:
        def renderCall(self, _toolCallId: str, _toolName: str, _args: Any) -> str | None:
            return "<div>call</div>"

        def renderResult(
            self,
            _toolCallId: str,
            _toolName: str,
            _result: list[dict[str, Any]],
            _details: Any,
            _isError: bool,
        ) -> Any:
            return type("Rendered", (), {"collapsed": "", "expanded": ""})()

    output_path = await export_session_to_html(
        session_manager,
        None,
        {"toolRenderer": _EmptyRenderer(), "outputPath": str(tmp_path / "empty-rendered.html")},
    )
    payload = _extract_session_payload(Path(output_path).read_text(encoding="utf-8"))

    assert payload["renderedTools"]["tool-2"]["resultHtmlCollapsed"] == ""
    assert payload["renderedTools"]["tool-2"]["resultHtmlExpanded"] == ""
