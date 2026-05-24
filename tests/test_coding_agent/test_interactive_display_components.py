from __future__ import annotations

import asyncio
import importlib
import re
from types import SimpleNamespace

import pytest
from harnify_agent.harness.messages import BranchSummaryMessage, CompactionSummaryMessage, CustomMessage
from harnify_agent.types import AgentState
from harnify_ai.types import Model, ModelCost, TextContent
from harnify_coding_agent.core.agent_session import ParsedSkillBlock
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.modes.interactive.components import (
    BorderedLoader,
    BranchSummaryMessageComponent,
    CompactionSummaryMessageComponent,
    CustomMessageComponent,
    FooterComponent,
    SkillInvocationMessageComponent,
    ToolExecutionComponent,
    render_diff,
)
from harnify_tui import Text, setKeybindings, visibleWidth

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)
interactive_theme_module = importlib.import_module("harnify_coding_agent.modes.interactive.theme.theme")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    interactive_theme_module.init_theme("dark")


def test_branch_and_compaction_summary_components_expand() -> None:
    branch = BranchSummaryMessage(summary="branch body", fromId="branch-1", timestamp=0)
    branch_component = BranchSummaryMessageComponent(branch)
    assert "Branch summary" in _strip_ansi("\n".join(branch_component.render(80)))
    branch_component.setExpanded(True)
    assert "branch body" in _strip_ansi("\n".join(branch_component.render(80)))

    compaction = CompactionSummaryMessage(summary="summary body", tokensBefore=12345, timestamp=0)
    compaction_component = CompactionSummaryMessageComponent(compaction)
    assert "Compacted from 12,345 tokens" in _strip_ansi("\n".join(compaction_component.render(80)))
    compaction_component.setExpanded(True)
    assert "summary body" in _strip_ansi("\n".join(compaction_component.render(80)))


def test_custom_message_uses_custom_renderer_and_falls_back() -> None:
    message = CustomMessage(
        customType="demo",
        content=[TextContent(text="default body")],
        display=True,
        timestamp=0,
        details={"value": 1},
    )
    calls: list[tuple[object, object]] = []

    def renderer(msg, options, _theme):  # noqa: ANN001
        calls.append((msg, options))
        return Text("custom render", 0, 0)

    component = CustomMessageComponent(message, renderer)
    rendered = _strip_ansi("\n".join(component.render(80)))
    assert "custom render" in rendered
    assert calls and calls[0][1] == {"expanded": False}

    fallback = CustomMessageComponent(message, lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    fallback_rendered = _strip_ansi("\n".join(fallback.render(80)))
    assert "[demo]" in fallback_rendered
    assert "default body" in fallback_rendered


def test_skill_invocation_component_collapses_and_expands() -> None:
    skill = ParsedSkillBlock(name="attio", location="/tmp/attio", content="## Steps\n\nDo the thing", userMessage=None)
    component = SkillInvocationMessageComponent(skill)
    collapsed = _strip_ansi("\n".join(component.render(80)))
    assert "[skill]" in collapsed
    assert "attio" in collapsed
    assert "Do the thing" not in collapsed

    component.setExpanded(True)
    expanded = _strip_ansi("\n".join(component.render(80)))
    assert "Do the thing" in expanded


def test_bordered_loader_exposes_abort_signal() -> None:
    ui = SimpleNamespace(requestRender=lambda: None)
    loader = BorderedLoader(ui, interactive_theme_module.theme, "Loading", {"cancellable": True})
    called: list[bool] = []
    loader.onAbort = lambda: called.append(True)
    loader.handleInput("\x1b")
    assert loader.signal.aborted is True
    assert called == [True]


def test_render_diff_highlights_single_line_replacement() -> None:
    rendered = render_diff("-  10 old value\n+  10 new value\n  11 untouched")
    stripped = _strip_ansi(rendered)
    assert "-  10 old value" in stripped
    assert "+  10 new value" in stripped
    assert "\x1b[7m" in rendered


class _FooterData:
    def __init__(self, provider_count: int) -> None:
        self.provider_count = provider_count

    def getGitBranch(self) -> str | None:
        return "main"

    def getExtensionStatuses(self) -> dict[str, str]:
        return {}

    def getAvailableProviderCount(self) -> int:
        return self.provider_count

    def onBranchChange(self, callback):  # noqa: ANN001, ANN201
        del callback
        return lambda: None


def _create_session(
    *,
    session_name: str,
    model_id: str = "test-model",
    provider: str = "test",
    reasoning: bool = False,
    thinking_level: str = "off",
    usage: dict[str, object] | None = None,
):
    entries = [] if usage is None else [{"type": "message", "message": {"role": "assistant", "usage": usage}}]
    model = Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider=provider,
        baseUrl="https://example.com",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=200000,
        maxTokens=4096,
    )
    state = AgentState(model=model, thinkingLevel=thinking_level)
    return SimpleNamespace(
        state=state,
        sessionManager=SimpleNamespace(
            getEntries=lambda: entries,
            getSessionName=lambda: session_name,
            getCwd=lambda: "/tmp/project",
        ),
        getContextUsage=lambda: {"contextWindow": 200000, "percent": 12.3},
        modelRegistry=SimpleNamespace(isUsingOAuth=lambda _model: False),
    )


def test_footer_component_respects_width_for_wide_names() -> None:
    width = 93
    session = _create_session(session_name="한글" * 30)
    footer = FooterComponent(session, _FooterData(1), interactive_theme_module.theme)
    lines = footer.render(width)
    for line in lines:
        assert visibleWidth(line) <= width


def test_footer_component_respects_width_for_wide_model_provider_names() -> None:
    width = 60
    session = _create_session(
        session_name="",
        model_id="模" * 30,
        provider="공급자",
        reasoning=True,
        thinking_level="high",
        usage={
            "input": 12345,
            "output": 6789,
            "cacheRead": 0,
            "cacheWrite": 0,
            "cost": {"total": 1.234},
        },
    )
    footer = FooterComponent(session, _FooterData(2), interactive_theme_module.theme)
    lines = footer.render(width)
    for line in lines:
        assert visibleWidth(line) <= width


@pytest.mark.asyncio
async def test_tool_execution_converts_non_png_images_for_kitty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.components.tool_execution.getCapabilities",
        lambda: SimpleNamespace(images="kitty"),
    )

    async def fake_convert_to_png(data: str, mime_type: str):
        assert data == "jpeg-data"
        assert mime_type == "image/jpeg"
        return SimpleNamespace(data="png-data", mimeType="image/png")

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.components.tool_execution.convert_to_png",
        fake_convert_to_png,
    )

    component = ToolExecutionComponent("read", "tool-1", {}, ui=SimpleNamespace(requestRender=lambda: None))
    component.updateResult({"content": [{"type": "image", "data": "jpeg-data", "mimeType": "image/jpeg"}]})
    await asyncio.sleep(0)

    assert component._converted_images[0] == ("png-data", "image/png")
