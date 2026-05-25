from __future__ import annotations

import asyncio
import builtins
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harnify_ai.types import Model
from harnify_coding_agent.core.agent_session_runtime import SessionImportFileNotFoundError
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.session_cwd import MissingSessionCwdError, SessionCwdIssue
from harnify_coding_agent.config import APP_NAME, APP_TITLE
import harnify_coding_agent.modes.interactive.interactive_mode as interactive_mode_module
from harnify_coding_agent.modes.interactive.interactive_mode import (
    ANTHROPIC_SUBSCRIPTION_AUTH_WARNING,
    InteractiveMode,
)
from harnify_coding_agent.modes.interactive.components.assistant_message import AssistantMessageComponent
from harnify_coding_agent.modes.interactive.components.tool_execution import ToolExecutionComponent
from harnify_coding_agent.modes.interactive.components.user_message import UserMessageComponent
from harnify_coding_agent.modes.interactive.theme.theme import init_theme
from harnify_coding_agent.utils.changelog import ChangelogEntry
from harnify_coding_agent.utils.version_check import LatestPiRelease
from harnify_tui import Container, Text, setKeybindings

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
    return None


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    init_theme("dark")


class FakeUi:
    def __init__(self) -> None:
        self.children: list[Any] = []
        self.render_calls: list[bool | None] = []
        self.started = 0
        self.stopped = 0
        self.focused: Any | None = None
        self.invalidated = 0
        self.overlays: list[tuple[Any, dict[str, Any]]] = []
        self.input_listeners: list[Any] = []
        self.overlay_handles: list[Any] = []
        self.terminal_titles: list[str] = []
        self.terminal = SimpleNamespace(
            setProgress=lambda _value: None,
            setTitle=lambda value: self.terminal_titles.append(value),
        )

    def requestRender(self, force: bool | None = None) -> None:
        self.render_calls.append(force)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def addChild(self, component: Any) -> None:
        self.children.append(component)

    def removeChild(self, component: Any) -> None:
        if component in self.children:
            self.children.remove(component)

    def setFocus(self, component: Any) -> None:
        self.focused = component

    def invalidate(self) -> None:
        self.invalidated += 1

    def showOverlay(self, component: Any, options: dict[str, Any] | None = None) -> Any:
        self.overlays.append((component, dict(options or {})))
        self.focused = component
        hidden = {"value": False}
        handle = SimpleNamespace(
            hide=lambda: hidden.__setitem__("value", True),
            focus=lambda: self.setFocus(component),
            component=component,
            hidden=hidden,
        )
        self.overlay_handles.append(handle)
        return handle

    def hideOverlay(self) -> None:
        if self.overlay_handles:
            self.overlay_handles[-1].hide()

    def addInputListener(self, listener: Any) -> Any:
        self.input_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self.input_listeners:
                self.input_listeners.remove(listener)

        return unsubscribe


class FakeEditor:
    def __init__(self) -> None:
        self.providers: list[Any] = []
        self.text = ""
        self.onEscape = None
        self.onCtrlD = None
        self.onPasteImage = None
        self.onExtensionShortcut = None
        self.onChange = None
        self.onSubmit = None
        self.history: list[str] = []
        self.actions: dict[str, Any] = {}
        self.actionHandlers = self.actions
        self.paddingX = 0
        self.autocompleteMaxVisible = 0
        self.borderColor = None
        self.inserted: list[str] = []

    def setAutocompleteProvider(self, provider: Any) -> None:
        self.providers.append(provider)

    def setText(self, text: str) -> None:
        self.text = text

    def getText(self) -> str:
        return self.text

    def getExpandedText(self) -> str:
        return self.text

    def addToHistory(self, text: str) -> None:
        self.history.append(text)

    def onAction(self, action: str, handler: Any) -> None:
        self.actions[action] = handler

    def insertTextAtCursor(self, text: str) -> None:
        self.inserted.append(text)
        self.text += text

    def handleInput(self, data: str) -> None:
        self.text += data

    def setPaddingX(self, padding: int) -> None:
        self.paddingX = padding

    def setAutocompleteMaxVisible(self, value: int) -> None:
        self.autocompleteMaxVisible = value


def _model(provider: str, model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider=provider,
        baseUrl=f"https://{provider}.example.com",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2},
        contextWindow=200_000,
        maxTokens=16_000,
    )


def test_show_status_coalesces_and_appends_when_interleaved() -> None:
    ui = FakeUi()
    mode = InteractiveMode(chatContainer=Container(), ui=ui)

    mode.showStatus("STATUS_ONE")
    assert len(mode.chatContainer.children) == 2
    assert "STATUS_ONE" in _strip_ansi("\n".join(mode.lastStatusText.render(120)))

    mode.showStatus("STATUS_TWO")
    assert len(mode.chatContainer.children) == 2
    assert "STATUS_TWO" in _strip_ansi("\n".join(mode.lastStatusText.render(120)))
    assert "STATUS_ONE" not in _strip_ansi("\n".join(mode.lastStatusText.render(120)))

    mode.chatContainer.addChild(Text("OTHER", 0, 0))
    mode.showStatus("STATUS_THREE")

    assert len(mode.chatContainer.children) == 5
    assert "STATUS_THREE" in _strip_ansi("\n".join(mode.lastStatusText.render(120)))
    assert ui.render_calls == [None, None, None]


def test_set_tools_expanded_updates_header_and_chat_children() -> None:
    header_calls: list[bool] = []
    child_calls: list[bool] = []
    ui = FakeUi()
    mode = InteractiveMode(
        ui=ui,
        builtInHeader=SimpleNamespace(setExpanded=lambda expanded: header_calls.append(expanded)),
        chatContainer=SimpleNamespace(
            children=[SimpleNamespace(setExpanded=lambda expanded: child_calls.append(expanded))]
        ),
    )

    mode.setToolsExpanded(True)

    assert mode.toolOutputExpanded is True
    assert header_calls == [True]
    assert child_calls == [True]
    assert ui.render_calls == [None]


def test_toggle_thinking_block_visibility_rebuilds_chat_and_reports_status() -> None:
    statuses: list[str] = []
    mode = InteractiveMode(
        chatContainer=Container(),
        settingsManager=SimpleNamespace(setHideThinkingBlock=lambda _value: None),
    )
    mode.rebuildChatFromMessages = lambda: statuses.append("rebuilt")  # type: ignore[method-assign]
    mode.showStatus = statuses.append  # type: ignore[method-assign]

    mode.toggleThinkingBlockVisibility()

    assert mode.hideThinkingBlock is True
    assert statuses == ["rebuilt", "Thinking blocks: hidden"]


def test_render_current_session_state_resets_pending_and_streaming_before_render() -> None:
    ui = FakeUi()
    pending = Container()
    pending.addChild(Text("queued", 0, 0))
    chat = Container()
    chat.addChild(Text("stale", 0, 0))
    calls: list[str] = []
    mode = InteractiveMode(ui=ui, chatContainer=chat, pendingMessagesContainer=pending)
    mode.compactionQueuedMessages = [{"text": "queued", "mode": "followUp"}]
    mode.streamingComponent = object()
    mode.streamingMessage = {"role": "assistant"}
    mode._toolComponentsById = {"tool-1": object()}
    mode.renderInitialMessages = lambda: calls.append("rendered")  # type: ignore[method-assign]

    mode.renderCurrentSessionState()

    assert len(chat.children) == 0
    assert len(pending.children) == 0
    assert mode.compactionQueuedMessages == []
    assert mode.streamingComponent is None
    assert mode.streamingMessage is None
    assert mode._toolComponentsById == {}
    assert calls == ["rendered"]
    assert ui.render_calls == [None]


def test_render_session_context_matches_ts_footer_history_and_tool_results() -> None:
    ui = FakeUi()
    editor = FakeEditor()
    footer_calls: list[str] = []
    mode = InteractiveMode(
        ui=ui,
        chatContainer=Container(),
        defaultEditor=editor,
        editor=editor,
        footer=SimpleNamespace(invalidate=lambda: footer_calls.append("footer")),
        session=SimpleNamespace(
            retryAttempt=0,
            extensionRunner=SimpleNamespace(get_message_renderer=lambda _custom_type: None),
            getToolDefinition=lambda _name: None,
            state=SimpleNamespace(thinkingLevel="off"),
        ),
        sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project"),
        settingsManager=SimpleNamespace(
            getShowImages=lambda: True,
            getImageWidthCells=lambda: 40,
            getCodeBlockIndent=lambda: "  ",
        ),
    )
    context = SimpleNamespace(
        messages=[
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [{"type": "toolCall", "id": "tool-1", "name": "read", "arguments": {"path": "x"}}],
                "stopReason": "end",
            },
            {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "content": [{"type": "text", "text": "done"}],
                "isError": False,
            },
        ]
    )

    mode.renderSessionContext(context, {"updateFooter": True, "populateHistory": True})

    assert editor.history == ["hello"]
    assert footer_calls == ["footer"]
    assert mode._toolComponentsById == {}
    assert any(isinstance(child, UserMessageComponent) for child in mode.chatContainer.children)
    assert any(isinstance(child, AssistantMessageComponent) for child in mode.chatContainer.children)
    assert any(isinstance(child, ToolExecutionComponent) for child in mode.chatContainer.children)
    assert ui.render_calls


def test_update_editor_border_color_matches_ts_and_requests_render() -> None:
    ui = FakeUi()
    editor = FakeEditor()
    mode = InteractiveMode(
        ui=ui,
        editor=editor,
        defaultEditor=editor,
        session=SimpleNamespace(state=SimpleNamespace(thinkingLevel="high")),
    )

    mode.updateEditorBorderColor()

    expected = interactive_mode_module.interactive_theme.theme.getThinkingBorderColor("high")
    assert callable(editor.borderColor)
    assert editor.borderColor("sample") == expected("sample")
    assert ui.render_calls == [None]


def test_extension_ui_context_persists_theme_and_rebuilds_autocomplete() -> None:
    current_theme = {"value": "dark"}
    settings = SimpleNamespace(
        getTheme=lambda: current_theme["value"],
        setTheme=lambda value: current_theme.__setitem__("value", value),
    )
    ui = FakeUi()
    mode = InteractiveMode(settingsManager=settings, ui=ui)
    rebuilds: list[str] = []
    mode.setupAutocompleteProvider = lambda: rebuilds.append("rebuilt")  # type: ignore[method-assign]

    ctx = mode.createExtensionUIContext()
    ok = ctx.setTheme("light")
    bad = ctx.setTheme("__missing_theme__")
    ctx.addAutocompleteProvider(lambda current: current)

    assert ok["success"] is True
    assert bad["success"] is False
    assert current_theme["value"] == "light"
    assert len(mode.autocompleteProviderWrappers) == 1
    assert rebuilds == ["rebuilt"]
    assert ui.render_calls == [None]


@pytest.mark.asyncio
async def test_extension_ui_context_dialog_methods_delegate() -> None:
    mode = InteractiveMode()
    mode.showExtensionSelector = lambda title, options, opts=None: asyncio.sleep(0, result=options[0])  # type: ignore[method-assign]
    mode.showExtensionConfirm = lambda title, message, opts=None: asyncio.sleep(0, result=True)  # type: ignore[method-assign]
    mode.showExtensionInput = lambda title, placeholder=None, opts=None: asyncio.sleep(0, result=placeholder)  # type: ignore[method-assign]
    mode.showExtensionEditor = lambda title, prefill=None: asyncio.sleep(0, result=prefill)  # type: ignore[method-assign]

    ctx = mode.createExtensionUIContext()

    assert await ctx.select("Pick", ["A", "B"]) == "A"
    assert await ctx.confirm("Confirm", "Question?") is True
    assert await ctx.input("Input", "placeholder") == "placeholder"
    assert await ctx.editor("Editor", "prefill") == "prefill"


def test_extension_ui_context_status_widget_and_terminal_helpers() -> None:
    ui = FakeUi()
    mode = InteractiveMode(ui=ui, defaultEditor=FakeEditor(), editor=FakeEditor())
    ctx = mode.createExtensionUIContext()

    calls: list[str] = []
    unsubscribe = ctx.onTerminalInput(lambda data: {"consume": True, "data": data})
    ctx.setStatus("build", "Busy")
    ctx.setWidget("summary", ["line one", "line two"])
    ctx.setTitle("Title from extension")
    ctx.setEditorText("hello")
    ctx.pasteToEditor(" world")

    assert "build" in mode.footerDataProvider.getExtensionStatuses()
    assert len(mode.widgetContainerAbove.children) == 2
    assert ui.terminal_titles == ["Title from extension"]
    assert "world" in ctx.getEditorText()
    assert len(ui.input_listeners) == 1

    unsubscribe()
    ctx.setStatus("build", None)
    assert mode.footerDataProvider.getExtensionStatuses() == {}
    assert ui.input_listeners == []


def test_set_extension_header_footer_and_reset_ui_restore_builtins() -> None:
    ui = FakeUi()
    built_in_header = Text("Header", 0, 0)
    header_container = Container()
    header_container.addChild(built_in_header)
    footer = Text("Footer", 0, 0)
    mode = InteractiveMode(
        ui=ui,
        headerContainer=header_container,
        builtInHeader=built_in_header,
        footer=footer,
        defaultEditor=FakeEditor(),
        editor=FakeEditor(),
    )
    ui.addChild(footer)
    disposed: list[str] = []

    def make_component(label: str) -> Any:
        return SimpleNamespace(
            label=label,
            render=lambda _width: [label],
            dispose=lambda: disposed.append(label),
        )

    mode.setExtensionHeader(lambda _ui, _theme: make_component("custom-header"))
    mode.setExtensionFooter(lambda _ui, _theme, _footer_data: make_component("custom-footer"))
    mode.setExtensionWidget("widget", ["one"])
    mode.setExtensionStatus("status", "Active")
    mode.resetExtensionUI()

    assert mode.customHeader is None
    assert mode.customFooter is None
    assert mode.headerContainer.children[0] is built_in_header
    assert footer in ui.children
    assert mode.extensionWidgetsAbove == {}
    assert mode.footerDataProvider.getExtensionStatuses() == {}
    assert {"custom-header", "custom-footer"} <= set(disposed)


@pytest.mark.asyncio
async def test_setup_extension_shortcuts_wires_default_and_custom_editors() -> None:
    default_editor = FakeEditor()
    custom_editor = FakeEditor()
    mode = InteractiveMode(ui=FakeUi(), defaultEditor=default_editor, editor=custom_editor)
    shortcut_contexts: list[Any] = []

    async def shortcut_handler(ctx: Any) -> None:
        shortcut_contexts.append(ctx)

    extension_runner = SimpleNamespace(
        getShortcuts=lambda _config: {"k": SimpleNamespace(handler=shortcut_handler)},
        createContext=lambda: {"source": "extension-shortcut"},
    )

    mode.setupExtensionShortcuts(extension_runner)
    assert default_editor.onExtensionShortcut("k") is True
    assert custom_editor.onExtensionShortcut("k") is True
    await asyncio.sleep(0)

    assert shortcut_contexts == [{"source": "extension-shortcut"}, {"source": "extension-shortcut"}]


def test_set_custom_editor_component_preserves_text_and_handlers() -> None:
    ui = FakeUi()
    default_editor = FakeEditor()
    default_editor.setText("seed")
    default_editor.onSubmit = lambda text: None
    default_editor.onChange = lambda text: None
    default_editor.onEscape = lambda: None
    default_editor.onCtrlD = lambda: None
    default_editor.onPasteImage = lambda: None
    default_editor.onExtensionShortcut = lambda _data: True
    default_editor.onAction("app.clear", lambda: None)
    container = Container()
    container.addChild(default_editor)
    mode = InteractiveMode(ui=ui, defaultEditor=default_editor, editor=default_editor, editorContainer=container)

    class CustomSwapEditor(FakeEditor):
        pass

    mode.setCustomEditorComponent(lambda _ui, _theme, _keybindings: CustomSwapEditor())
    swapped = mode.editor
    assert swapped is not default_editor
    assert swapped.getText() == "seed"
    assert swapped.onSubmit is default_editor.onSubmit
    assert swapped.onChange is default_editor.onChange
    assert "app.clear" in swapped.actionHandlers

    swapped.setText("changed")
    mode.setCustomEditorComponent(None)

    assert mode.editor is default_editor
    assert default_editor.getText() == "changed"


@pytest.mark.asyncio
async def test_show_extension_custom_restores_editor_inline_and_overlay() -> None:
    ui = FakeUi()
    editor = FakeEditor()
    editor.setText("draft")
    container = Container()
    container.addChild(editor)
    mode = InteractiveMode(ui=ui, defaultEditor=editor, editor=editor, editorContainer=container)

    captured_inline: dict[str, Any] = {}

    async def inline_factory(_ui: Any, _theme: Any, _keybindings: Any, done: Any) -> Any:
        captured_inline["done"] = done
        return SimpleNamespace(render=lambda _width: ["inline"], dispose=lambda: None)

    inline_task = asyncio.create_task(mode.showExtensionCustom(inline_factory))
    await asyncio.sleep(0)
    assert mode.editorContainer.children[0] is not editor
    captured_inline["done"]("inline-ok")
    assert await inline_task == "inline-ok"
    assert mode.editorContainer.children[0] is editor
    assert editor.getText() == "draft"

    captured_overlay: dict[str, Any] = {}
    handles: list[Any] = []

    async def overlay_factory(_ui: Any, _theme: Any, _keybindings: Any, done: Any) -> Any:
        captured_overlay["done"] = done
        return SimpleNamespace(render=lambda _width: ["overlay"], dispose=lambda: None)

    overlay_task = asyncio.create_task(
        mode.showExtensionCustom(
            overlay_factory,
            {"overlay": True, "onHandle": handles.append},
        )
    )
    await asyncio.sleep(0)
    assert ui.overlays and handles
    captured_overlay["done"]("overlay-ok")
    assert await overlay_task == "overlay-ok"
    assert handles[0].hidden["value"] is True


@pytest.mark.asyncio
async def test_update_terminal_title_and_reload_command_binding() -> None:
    ui = FakeUi()
    mode = InteractiveMode(
        ui=ui,
        sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project", getSessionName=lambda: "Session"),
    )

    mode.updateTerminalTitle()
    assert ui.terminal_titles == [f"{APP_TITLE} - Session - project"]

    calls: list[str] = []

    async def fake_reload() -> None:
        calls.append("reload")

    mode.handleReloadCommand = fake_reload  # type: ignore[method-assign]
    actions = mode._build_command_context_actions()
    await actions["reload"]()

    assert calls == ["reload"]


@pytest.mark.asyncio
async def test_rebind_current_session_shows_loaded_resources_and_diagnostics() -> None:
    source_info = lambda path, base_dir: SimpleNamespace(  # noqa: E731
        path=path,
        source="local",
        scope="project",
        baseDir=base_dir,
    )
    skill = SimpleNamespace(
        name="beads",
        filePath="/tmp/project/.harnify/skills/beads/SKILL.md",
        sourceInfo=source_info(
            "/tmp/project/.harnify/skills/beads/SKILL.md",
            "/tmp/project/.harnify/skills",
        ),
    )
    prompt = SimpleNamespace(
        name="review",
        filePath="/tmp/project/.harnify/prompts/review.md",
        sourceInfo=source_info(
            "/tmp/project/.harnify/prompts/review.md",
            "/tmp/project/.harnify/prompts",
        ),
    )
    extension = SimpleNamespace(
        path="/tmp/project/.harnify/extensions/example.py",
        sourceInfo=source_info(
            "/tmp/project/.harnify/extensions/example.py",
            "/tmp/project/.harnify/extensions",
        ),
    )
    theme = SimpleNamespace(
        name="custom",
        sourcePath="/tmp/project/.harnify/themes/custom.json",
        sourceInfo=source_info(
            "/tmp/project/.harnify/themes/custom.json",
            "/tmp/project/.harnify/themes",
        ),
    )
    resource_loader = SimpleNamespace(
        getSkills=lambda: {
            "skills": [skill],
            "diagnostics": [SimpleNamespace(type="warning", message="skill warning", path=skill.filePath)],
        },
        getPrompts=lambda: {"prompts": [prompt], "diagnostics": []},
        getThemes=lambda: {"themes": [theme], "diagnostics": []},
        getAgentsFiles=lambda: {"agentsFiles": [{"path": "/tmp/project/AGENTS.md"}]},
        getExtensions=lambda: SimpleNamespace(extensions=[extension], errors=[]),
    )
    session = SimpleNamespace(
        resourceLoader=resource_loader,
        promptTemplates=[prompt],
        extensionRunner=SimpleNamespace(
            getCommandDiagnostics=lambda: [],
            getShortcutDiagnostics=lambda: [
                SimpleNamespace(type="warning", message="shortcut warning", path=extension.path)
            ],
            get_registered_commands=lambda: [],
            get_message_renderer=lambda _custom_type: None,
        ),
        bindExtensions=_noop_async,
        subscribe=lambda _listener: (lambda: None),
        modelRegistry=SimpleNamespace(getAvailable=lambda: []),
        state=SimpleNamespace(messages=[]),
        autoCompactionEnabled=False,
        isStreaming=False,
    )
    mode = InteractiveMode(
        ui=FakeUi(),
        session=session,
        sessionManager=SimpleNamespace(
            getCwd=lambda: "/tmp/project",
            getSessionName=lambda: None,
        ),
    )

    await mode.rebindCurrentSession()

    rendered = "\n".join(
        line
        for child in mode.chatContainer.children
        if hasattr(child, "render")
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)

    assert "[Context]" in stripped
    assert "[Skills]" in stripped
    assert "[Prompts]" in stripped
    assert "[Extensions]" in stripped
    assert "[Themes]" in stripped
    assert "AGENTS.md" in stripped
    assert "beads" in stripped
    assert "/review" in stripped
    assert "example" in stripped
    assert "custom" in stripped
    assert "shortcut warning" in stripped
    assert "skill warning" in stripped


@pytest.mark.asyncio
async def test_show_extension_selector_and_confirm_use_real_overlay_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Any] = []

    class FakeExtensionSelectorComponent:
        def __init__(self, title: str, options: list[str], onSelect: Any, onCancel: Any, opts: Any) -> None:
            self.title = title
            self.options = options
            self.onSelect = onSelect
            self.onCancel = onCancel
            self.opts = opts
            captured.append(self)

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.ExtensionSelectorComponent",
        FakeExtensionSelectorComponent,
    )

    ui = FakeUi()
    mode = InteractiveMode(ui=ui)

    select_task = asyncio.create_task(mode.showExtensionSelector("Pick one", ["A", "B"]))
    await asyncio.sleep(0)
    captured[-1].onSelect("B")
    assert await select_task == "B"

    confirm_task = asyncio.create_task(mode.showExtensionConfirm("Delete", "Really?"))
    await asyncio.sleep(0)
    captured[-1].onSelect("Yes")
    assert await confirm_task is True


def test_setup_autocomplete_provider_stacks_wrappers() -> None:
    calls: list[str] = []

    def wrap1(current):
        class Provider:
            async def getSuggestions(self, lines, cursorLine, cursorCol, options):
                calls.append("getSuggestions:wrap1")
                return await current.getSuggestions(lines, cursorLine, cursorCol, options)

            def applyCompletion(self, lines, cursorLine, cursorCol, item, prefix):
                calls.append("applyCompletion:wrap1")
                return current.applyCompletion(lines, cursorLine, cursorCol, item, prefix)

            def shouldTriggerFileCompletion(self, lines, cursorLine, cursorCol):
                calls.append("shouldTrigger:wrap1")
                return current.shouldTriggerFileCompletion(lines, cursorLine, cursorCol)

        return Provider()

    def wrap2(current):
        class Provider:
            async def getSuggestions(self, lines, cursorLine, cursorCol, options):
                calls.append("getSuggestions:wrap2")
                return await current.getSuggestions(lines, cursorLine, cursorCol, options)

            def applyCompletion(self, lines, cursorLine, cursorCol, item, prefix):
                calls.append("applyCompletion:wrap2")
                return current.applyCompletion(lines, cursorLine, cursorCol, item, prefix)

            def shouldTriggerFileCompletion(self, lines, cursorLine, cursorCol):
                calls.append("shouldTrigger:wrap2")
                return current.shouldTriggerFileCompletion(lines, cursorLine, cursorCol)

        return Provider()

    default_editor = FakeEditor()
    custom_editor = FakeEditor()
    mode = InteractiveMode(
        defaultEditor=default_editor,
        editor=custom_editor,
        sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project"),
        autocompleteProviderWrappers=[wrap1, wrap2],
    )

    mode.setupAutocompleteProvider()

    assert len(default_editor.providers) == 1
    assert default_editor.providers[0] is custom_editor.providers[0]
    assert default_editor.providers[0].shouldTriggerFileCompletion(["foo"], 0, 3) is True
    assert calls == ["shouldTrigger:wrap2", "shouldTrigger:wrap1"]


@pytest.mark.asyncio
async def test_warns_once_for_anthropic_subscription_auth() -> None:
    warnings: list[str] = []
    mode = InteractiveMode(
        settingsManager=SimpleNamespace(getWarnings=lambda: {}),
        session=SimpleNamespace(
            modelRegistry=SimpleNamespace(
                authStorage=SimpleNamespace(get=lambda _provider: {"type": "oauth"}),
                getApiKeyForProvider=lambda _provider: None,
            )
        ),
    )
    mode.showWarning = warnings.append  # type: ignore[method-assign]

    await mode.maybeWarnAboutAnthropicSubscriptionAuth(SimpleNamespace(provider="anthropic"))
    await mode.maybeWarnAboutAnthropicSubscriptionAuth(SimpleNamespace(provider="anthropic"))

    assert warnings == [ANTHROPIC_SUBSCRIPTION_AUTH_WARNING]


def test_handle_ctrl_z_windows_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    mode = InteractiveMode()
    statuses: list[str] = []
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "platform", "win32", raising=False)

    mode.handleCtrlZ()

    assert statuses == ["Suspend to background is not supported on Windows"]


def test_handle_ctrl_z_suspends_and_restores_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = FakeUi()
    installed_handlers: dict[int, Any] = {}

    class FakeTimer:
        def __init__(self, _seconds: int, _fn: Any) -> None:
            self.started = False
            self.cancelled = False

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

    timers: list[FakeTimer] = []
    previous_sigint = object()
    previous_sigcont = object()

    def fake_timer(seconds: int, fn: Any) -> FakeTimer:
        timer = FakeTimer(seconds, fn)
        timers.append(timer)
        return timer

    def fake_signal(signum: int, handler: Any) -> Any:
        previous = installed_handlers.get(signum)
        installed_handlers[signum] = handler
        return previous

    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    monkeypatch.setattr(threading, "Timer", fake_timer)
    monkeypatch.setattr(
        signal,
        "getsignal",
        lambda signum: previous_sigint if signum == signal.SIGINT else previous_sigcont,
    )
    monkeypatch.setattr(signal, "signal", fake_signal)
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, signum: kill_calls.append((pid, signum)))

    mode = InteractiveMode(ui=ui)
    mode.handleCtrlZ()

    assert ui.stopped == 1
    assert timers and timers[0].started is True
    assert kill_calls == [(0, signal.SIGTSTP)]

    installed_handlers[signal.SIGCONT](signal.SIGCONT, None)

    assert timers[0].cancelled is True
    assert ui.started == 1
    assert ui.render_calls[-1] is True
    assert installed_handlers[signal.SIGINT] is previous_sigint
    assert installed_handlers[signal.SIGCONT] is previous_sigcont


@pytest.mark.asyncio
async def test_handle_ctrl_c_matches_ts_double_sigint_shutdown_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    mode = InteractiveMode()
    mode.clearEditor = lambda: calls.append("clear")  # type: ignore[method-assign]

    async def shutdown() -> int:
        calls.append("shutdown")
        return 0

    mode.shutdown = shutdown  # type: ignore[method-assign]
    times = iter([1.0, 1.3])
    monkeypatch.setattr(interactive_mode_module.time, "time", lambda: next(times))

    mode.handleCtrlC()
    await asyncio.sleep(0)
    mode.handleCtrlC()
    await asyncio.sleep(0)

    assert calls == ["clear", "shutdown"]


def test_register_signal_handlers_and_stop_match_ts_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    installed_handlers: dict[int, Any] = {}
    previous_sigterm = object()
    previous_sighup = object()
    cleanup_calls: list[Any] = []

    class FakeStream:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, event: str, handler: Any) -> None:
            self.handlers[event] = handler

        def off(self, event: str, handler: Any) -> None:
            if self.handlers.get(event) is handler:
                self.handlers.pop(event, None)

    stdout = FakeStream()
    stderr = FakeStream()

    monkeypatch.setattr(
        signal,
        "getsignal",
        lambda signum: previous_sigterm if signum == signal.SIGTERM else previous_sighup,
    )
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed_handlers.__setitem__(signum, handler))
    monkeypatch.setattr(interactive_mode_module.sys, "stdout", stdout)
    monkeypatch.setattr(interactive_mode_module.sys, "stderr", stderr)

    ui = FakeUi()
    ui.terminal = SimpleNamespace(
        setProgress=lambda active: cleanup_calls.append(("progress", active)),
        setTitle=lambda _value: None,
    )
    mode = InteractiveMode(
        ui=ui,
        settingsManager=SimpleNamespace(getShowTerminalProgress=lambda: True),
        footer=SimpleNamespace(dispose=lambda: cleanup_calls.append("footer")),
        footerDataProvider=SimpleNamespace(dispose=lambda: cleanup_calls.append("footer-data")),
    )
    mode.extensionTerminalInputUnsubscribers = {lambda: cleanup_calls.append("ext-unsub")}
    mode.loadingAnimation = SimpleNamespace(stop=lambda: cleanup_calls.append("loader-stop"))
    mode._sessionUnsubscribe = lambda: cleanup_calls.append("session-unsub")
    mode.isInitialized = True

    mode.registerSignalHandlers()

    assert signal.SIGTERM in installed_handlers
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in installed_handlers
    assert "error" in stdout.handlers
    assert "error" in stderr.handlers
    assert mode.signalCleanupHandlers

    mode.stop()

    assert ("progress", False) in cleanup_calls
    assert "loader-stop" in cleanup_calls
    assert "ext-unsub" in cleanup_calls
    assert "footer" in cleanup_calls
    assert "footer-data" in cleanup_calls
    assert "session-unsub" in cleanup_calls
    assert installed_handlers[signal.SIGTERM] is previous_sigterm
    if hasattr(signal, "SIGHUP"):
        assert installed_handlers[signal.SIGHUP] is previous_sighup
    assert stdout.handlers == {}
    assert stderr.handlers == {}
    assert mode.signalCleanupHandlers == []
    assert ui.stopped == 1
    assert mode.isInitialized is False


def test_register_signal_handlers_dead_terminal_error_uses_emergency_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, event: str, handler: Any) -> None:
            self.handlers[event] = handler

        def off(self, event: str, handler: Any) -> None:
            if self.handlers.get(event) is handler:
                self.handlers.pop(event, None)

    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(signal, "signal", lambda _signum, _handler: None)
    monkeypatch.setattr(interactive_mode_module.sys, "stdout", stdout)
    monkeypatch.setattr(interactive_mode_module.sys, "stderr", stderr)

    mode = InteractiveMode()
    mode.emergencyTerminalExit = lambda: (_ for _ in ()).throw(SystemExit(129))  # type: ignore[method-assign]
    mode.registerSignalHandlers()

    with pytest.raises(SystemExit) as exc_info:
        stdout.handlers["error"](SimpleNamespace(code="EPIPE"))

    assert exc_info.value.code == 129
    mode.unregisterSignalHandlers()


@pytest.mark.asyncio
async def test_request_shutdown_defers_until_check_shutdown_requested_matches_ts() -> None:
    calls: list[Any] = []
    ui = FakeUi()
    ui.terminal = SimpleNamespace(
        drainInput=lambda max_ms: asyncio.sleep(0, result=calls.append(("drain", max_ms))),
        setProgress=lambda active: calls.append(("progress", active)),
        setTitle=lambda _value: None,
    )
    runtime_host = SimpleNamespace(dispose=lambda: asyncio.sleep(0, result=calls.append("runtime-dispose")))
    mode = InteractiveMode(
        ui=ui,
        runtimeHost=runtime_host,
        settingsManager=SimpleNamespace(getShowTerminalProgress=lambda: True),
        footer=SimpleNamespace(dispose=lambda: calls.append("footer")),
        footerDataProvider=SimpleNamespace(dispose=lambda: calls.append("footer-data")),
        session=SimpleNamespace(isStreaming=True),
    )
    mode.isInitialized = True
    mode.loadingAnimation = SimpleNamespace(stop=lambda: calls.append("loader-stop"))
    mode.extensionTerminalInputUnsubscribers = {lambda: calls.append("ext-unsub")}
    mode._sessionUnsubscribe = lambda: calls.append("session-unsub")
    mode.signalCleanupHandlers = [lambda: calls.append("signals-unregistered")]
    mode._shutdownFuture = asyncio.get_running_loop().create_future()

    mode.requestShutdown()

    assert mode.shutdownRequested is True
    assert mode._shutdownFuture.done() is False
    assert calls == []

    mode.session = SimpleNamespace(isStreaming=False)
    await mode.checkShutdownRequested()

    assert calls == [
        "signals-unregistered",
        ("drain", 1000),
        ("progress", False),
        "loader-stop",
        "ext-unsub",
        "footer",
        "footer-data",
        "session-unsub",
        "runtime-dispose",
    ]
    assert ui.stopped == 1
    assert mode._shutdownFuture.done() is True
    assert await asyncio.shield(mode._shutdownFuture) == 0


def test_emergency_terminal_exit_matches_ts_exit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        interactive_mode_module,
        "kill_tracked_detached_children",
        lambda: calls.append("kill"),
    )
    mode = InteractiveMode()
    mode.unregisterSignalHandlers = lambda: calls.append("unregister")  # type: ignore[method-assign]

    with pytest.raises(SystemExit) as exc_info:
        mode.emergencyTerminalExit()

    assert exc_info.value.code == 129
    assert mode.isShuttingDown is True
    assert calls == ["unregister", "kill"]


def test_uncaught_crash_restores_tui_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        interactive_mode_module,
        "kill_tracked_detached_children",
        lambda: calls.append("kill"),
    )
    ui = FakeUi()
    mode = InteractiveMode(ui=ui)
    mode.unregisterSignalHandlers = lambda: calls.append("unregister")  # type: ignore[method-assign]

    with pytest.raises(SystemExit) as exc_info:
        mode.uncaughtCrash(RuntimeError("boom"))

    assert exc_info.value.code == 1
    assert mode.isShuttingDown is True
    assert ui.stopped == 1
    assert calls == ["unregister", "kill"]


@pytest.mark.asyncio
async def test_handle_fatal_runtime_error_matches_ts_shutdown_and_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        interactive_mode_module.interactive_theme,
        "stop_theme_watcher",
        lambda: calls.append("theme"),
    )
    mode = InteractiveMode()
    mode.showError = lambda message: calls.append(("error", message))  # type: ignore[method-assign]
    mode.stop = lambda: calls.append("stop")  # type: ignore[method-assign]

    with pytest.raises(SystemExit) as exc_info:
        await mode.handleFatalRuntimeError("Failed to import session", RuntimeError("boom"))

    assert exc_info.value.code == 1
    assert calls == [("error", "Failed to import session: boom"), "theme", "stop"]


@pytest.mark.asyncio
async def test_import_command_retries_with_selected_cwd() -> None:
    calls: list[tuple[str, str | None]] = []
    statuses: list[str] = []
    stopped: list[bool] = []
    cleared: list[bool] = []
    rendered: list[bool] = []
    issue = SessionCwdIssue(
        sessionCwd="/missing/project",
        fallbackCwd="/current/project",
        sessionFile="/tmp/session.jsonl",
    )

    async def import_from_jsonl(path: str, cwd_override: str | None = None) -> dict[str, bool]:
        calls.append((path, cwd_override))
        if cwd_override is None:
            raise MissingSessionCwdError(issue)
        return {"cancelled": False}

    mode = InteractiveMode(
        runtimeHost=SimpleNamespace(importFromJsonl=import_from_jsonl),
        statusContainer=SimpleNamespace(clear=lambda: cleared.append(True)),
        loadingAnimation=SimpleNamespace(stop=lambda: stopped.append(True)),
    )
    mode.showExtensionConfirm = lambda _title, _message: True  # type: ignore[method-assign]
    mode.promptForMissingSessionCwd = lambda _error: "/current/project"  # type: ignore[method-assign]
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.renderCurrentSessionState = lambda: rendered.append(True)  # type: ignore[method-assign]

    await mode.handleImportCommand('/import "path/to/session.jsonl"')

    assert calls == [("path/to/session.jsonl", None), ("path/to/session.jsonl", "/current/project")]
    assert stopped == [True]
    assert cleared == [True]
    assert rendered == [True]
    assert statuses == ["Session imported from: path/to/session.jsonl"]


@pytest.mark.asyncio
async def test_import_command_reports_missing_file_nonfatally() -> None:
    errors: list[str] = []

    async def import_from_jsonl(_path: str, _cwd_override: str | None = None) -> dict[str, bool]:
        raise SessionImportFileNotFoundError("/tmp/missing-session.jsonl")

    mode = InteractiveMode(
        runtimeHost=SimpleNamespace(importFromJsonl=import_from_jsonl),
        statusContainer=SimpleNamespace(clear=lambda: None),
    )
    mode.showExtensionConfirm = lambda _title, _message: True  # type: ignore[method-assign]
    mode.showError = errors.append  # type: ignore[method-assign]

    await mode.handleImportCommand("/import /tmp/missing-session.jsonl")

    assert errors == ["Failed to import session: File not found: /tmp/missing-session.jsonl"]


@pytest.mark.asyncio
async def test_clone_command_and_compaction_end_rebuild_chat() -> None:
    statuses: list[str] = []
    fork_calls: list[tuple[str, dict[str, str]]] = []
    rendered: list[bool] = []
    editor = FakeEditor()

    async def fork(entry_id: str, options: dict[str, str]) -> dict[str, bool]:
        fork_calls.append((entry_id, options))
        return {"cancelled": False}

    mode = InteractiveMode(
        sessionManager=SimpleNamespace(getLeafId=lambda: "leaf-123", getCwd=lambda: os.getcwd()),
        runtimeHost=SimpleNamespace(fork=fork),
        editor=editor,
    )
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.renderCurrentSessionState = lambda: rendered.append(True)  # type: ignore[method-assign]

    await mode.handleCloneCommand()

    assert fork_calls == [("leaf-123", {"position": "at"})]
    assert rendered == [True]
    assert editor.text == ""
    assert statuses == ["Cloned to new session"]

    compaction_messages: list[Any] = []
    flush_calls: list[dict[str, bool]] = []
    footer_calls: list[bool] = []
    mode.chatContainer = SimpleNamespace(clear=lambda: rendered.append(False))
    mode.rebuildChatFromMessages = lambda: rendered.append(True)  # type: ignore[method-assign]
    mode.addMessageToChat = compaction_messages.append  # type: ignore[method-assign]
    mode.flushCompactionQueue = lambda options: flush_calls.append(options)  # type: ignore[method-assign]
    mode.footer = SimpleNamespace(invalidate=lambda: footer_calls.append(True))
    mode.statusContainer = SimpleNamespace(clear=lambda: None)
    mode.isInitialized = True

    await mode.handleEvent(
        {
            "type": "compaction_end",
            "reason": "manual",
            "result": {"summary": "summary", "tokensBefore": 123},
            "aborted": False,
            "willRetry": False,
        }
    )

    assert rendered[-2:] == [False, True]
    assert compaction_messages and compaction_messages[0].role == "compactionSummary"
    assert compaction_messages[0].tokensBefore == 123
    assert compaction_messages[0].summary == "summary"
    assert flush_calls == [{"willRetry": False}]
    assert footer_calls == [True]


@pytest.mark.asyncio
async def test_handle_event_compaction_start_sets_escape_handler_loader_and_progress() -> None:
    progress: list[bool] = []
    status_children: list[Any] = []
    aborts: list[bool] = []
    editor = FakeEditor()
    original_escape = lambda: None
    editor.onEscape = original_escape

    mode = InteractiveMode(
        ui=FakeUi(),
        defaultEditor=editor,
        editor=editor,
        session=SimpleNamespace(abortCompaction=lambda: aborts.append(True)),
        statusContainer=SimpleNamespace(
            clear=lambda: status_children.append("cleared"),
            addChild=lambda child: status_children.append(child),
        ),
        settingsManager=SimpleNamespace(getShowTerminalProgress=lambda: True),
    )
    mode.ui.terminal = SimpleNamespace(
        setProgress=lambda value: progress.append(value),
        setTitle=lambda _value: None,
    )
    mode.isInitialized = True

    await mode.handleEvent({"type": "compaction_start", "reason": "manual"})

    assert progress == [True]
    assert mode.autoCompactionEscapeHandler is original_escape
    assert mode.defaultEditor.onEscape is not original_escape
    mode.defaultEditor.onEscape()
    assert aborts == [True]
    assert status_children[0] == "cleared"
    assert mode.autoCompactionLoader is status_children[1]
    assert mode.ui.render_calls and mode.ui.render_calls[-1] is None


@pytest.mark.asyncio
async def test_handle_event_agent_start_restores_retry_state_and_starts_working_loader() -> None:
    progress: list[bool] = []
    status_calls: list[Any] = []
    retry_calls: list[str] = []
    editor = FakeEditor()
    original_escape = lambda: None
    editor.onEscape = lambda: None

    mode = InteractiveMode(
        ui=FakeUi(),
        defaultEditor=editor,
        editor=editor,
        statusContainer=SimpleNamespace(
            clear=lambda: status_calls.append("cleared"),
            addChild=lambda child: status_calls.append(child),
        ),
        settingsManager=SimpleNamespace(getShowTerminalProgress=lambda: True),
    )
    mode.ui.terminal = SimpleNamespace(
        setProgress=lambda value: progress.append(value),
        setTitle=lambda _value: None,
    )
    mode.isInitialized = True
    mode._toolComponentsById = {"tool-1": object()}
    mode.retryEscapeHandler = original_escape
    mode.retryCountdown = SimpleNamespace(dispose=lambda: retry_calls.append("countdown-dispose"))
    mode.retryLoader = SimpleNamespace(stop=lambda: retry_calls.append("retry-loader-stop"))
    mode.loadingAnimation = SimpleNamespace(stop=lambda: retry_calls.append("working-stop"))
    mode.createWorkingLoader = lambda: "working-loader"  # type: ignore[method-assign]

    await mode.handleEvent({"type": "agent_start"})

    assert progress == [True]
    assert mode._toolComponentsById == {}
    assert mode.defaultEditor.onEscape is original_escape
    assert mode.retryEscapeHandler is None
    assert mode.retryCountdown is None
    assert mode.retryLoader is None
    assert mode.loadingAnimation == "working-loader"
    assert retry_calls == ["countdown-dispose", "retry-loader-stop", "working-stop"]
    assert status_calls == ["cleared", "working-loader"]


@pytest.mark.asyncio
async def test_handle_event_session_info_changed_updates_terminal_title() -> None:
    updates: list[str] = []
    renders: list[str] = []
    mode = InteractiveMode()
    mode.updateTerminalTitle = lambda: updates.append("title")  # type: ignore[method-assign]
    mode.footer = SimpleNamespace(invalidate=lambda: renders.append("footer"))
    mode._request_render = lambda force=None: renders.append("render")  # type: ignore[method-assign]
    mode.isInitialized = True

    await mode.handleEvent({"type": "session_info_changed"})

    assert updates == ["title"]
    assert renders == ["footer", "render"]


@pytest.mark.asyncio
async def test_handle_event_assistant_streaming_updates_tool_components_and_aborts() -> None:
    footer_calls: list[str] = []
    mode = InteractiveMode(
        ui=FakeUi(),
        chatContainer=Container(),
        sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project"),
        settingsManager=SimpleNamespace(getShowImages=lambda: True, getImageWidthCells=lambda: 48),
        session=SimpleNamespace(retryAttempt=0),
        footer=SimpleNamespace(invalidate=lambda: footer_calls.append("footer")),
    )
    mode.isInitialized = True

    tool_call = {"type": "toolCall", "id": "tool-1", "name": "read", "arguments": {"filePath": "a.txt"}}

    await mode.handleEvent({"type": "message_start", "message": {"role": "assistant", "content": []}})
    assert mode.streamingComponent is not None

    await mode.handleEvent(
        {
            "type": "message_update",
            "message": {"role": "assistant", "content": [tool_call]},
        }
    )

    component = mode._toolComponentsById["tool-1"]
    assert component.args == {"filePath": "a.txt"}

    await mode.handleEvent(
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [tool_call], "stopReason": "aborted"},
        }
    )

    assert mode.streamingComponent is None
    assert mode.streamingMessage is None
    assert mode._toolComponentsById == {}
    assert component.result is not None
    assert component.result.isError is True
    assert component.result.content[0]["text"] == "Operation aborted"
    assert footer_calls[-1] == "footer"


@pytest.mark.asyncio
async def test_handle_event_tool_execution_lifecycle_updates_component() -> None:
    mode = InteractiveMode(
        ui=FakeUi(),
        chatContainer=Container(),
        sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project"),
        settingsManager=SimpleNamespace(getShowImages=lambda: True, getImageWidthCells=lambda: 48),
        footer=SimpleNamespace(invalidate=lambda: None),
    )
    mode.isInitialized = True

    await mode.handleEvent(
        {
            "type": "tool_execution_start",
            "toolName": "grep",
            "toolCallId": "tool-2",
            "args": {"pattern": "needle"},
        }
    )

    component = mode._toolComponentsById["tool-2"]
    assert component.executionStarted is True

    await mode.handleEvent(
        {
            "type": "tool_execution_update",
            "toolCallId": "tool-2",
            "partialResult": {"content": [{"type": "text", "text": "partial"}]},
        }
    )
    assert component.result is not None
    assert component.result.content[0]["text"] == "partial"

    await mode.handleEvent(
        {
            "type": "tool_execution_end",
            "toolCallId": "tool-2",
            "result": {"content": [{"type": "text", "text": "done"}]},
            "isError": False,
        }
    )

    assert "tool-2" not in mode._toolComponentsById
    assert component.result is not None
    assert component.result.content[0]["text"] == "done"


@pytest.mark.asyncio
async def test_handle_event_auto_retry_start_and_end_manage_escape_and_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_events: list[str] = []
    status_calls: list[Any] = []
    aborts: list[bool] = []
    errors: list[str] = []
    editor = FakeEditor()
    original_escape = lambda: None
    editor.onEscape = original_escape

    class FakeLoader:
        def __init__(self, _ui: Any, _spinner: Any, _message_color: Any, message: str, _indicator: Any = None) -> None:
            self.message = message

        def stop(self) -> None:
            loader_events.append("loader-stop")

        def setMessage(self, message: str) -> None:
            self.message = message
            loader_events.append(f"message:{message}")

    class FakeCountdown:
        def __init__(self, timeoutMs: int, _ui: Any, onTick: Any, _onExpire: Any) -> None:
            loader_events.append(f"countdown:{timeoutMs}")
            onTick(3)

        def dispose(self) -> None:
            loader_events.append("countdown-dispose")

    monkeypatch.setattr(interactive_mode_module, "Loader", FakeLoader)
    monkeypatch.setattr(interactive_mode_module, "CountdownTimer", FakeCountdown)

    mode = InteractiveMode(
        ui=FakeUi(),
        defaultEditor=editor,
        editor=editor,
        session=SimpleNamespace(abortRetry=lambda: aborts.append(True)),
        statusContainer=SimpleNamespace(
            clear=lambda: status_calls.append("cleared"),
            addChild=lambda child: status_calls.append(child),
        ),
        footer=SimpleNamespace(invalidate=lambda: None),
    )
    mode.showError = errors.append  # type: ignore[method-assign]
    mode.isInitialized = True

    await mode.handleEvent(
        {
            "type": "auto_retry_start",
            "attempt": 1,
            "maxAttempts": 2,
            "delayMs": 3000,
        }
    )

    assert mode.retryEscapeHandler is original_escape
    assert mode.defaultEditor.onEscape is not original_escape
    mode.defaultEditor.onEscape()
    assert aborts == [True]
    assert status_calls[0] == "cleared"
    assert mode.retryLoader is status_calls[1]

    await mode.handleEvent(
        {
            "type": "auto_retry_end",
            "success": False,
            "attempt": 1,
            "finalError": "boom",
        }
    )

    assert mode.defaultEditor.onEscape is original_escape
    assert mode.retryEscapeHandler is None
    assert mode.retryCountdown is None
    assert mode.retryLoader is None
    assert "countdown:3000" in loader_events
    assert "countdown-dispose" in loader_events
    assert "loader-stop" in loader_events
    assert errors == ["Retry failed after 1 attempts: boom"]


@pytest.mark.asyncio
async def test_run_seeds_initial_messages_and_starts_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = FakeUi()
    prompts: list[tuple[str, dict[str, Any] | None]] = []
    warnings: list[str] = []

    async def prompt(text: str, options: dict[str, Any] | None = None) -> None:
        prompts.append((text, options))

    async def fake_version_check(_version: str) -> None:
        return None

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.check_for_new_pi_version",
        fake_version_check,
    )
    monkeypatch.setattr(interactive_mode_module, "ensureTool", _noop_async)

    mode = InteractiveMode(
        ui=ui,
        options={
            "initialMessage": "first",
            "initialImages": ["img-1"],
            "initialMessages": ["second"],
            "modelFallbackMessage": "fallback",
        },
    )
    mode.session.prompt = prompt
    mode.showWarning = warnings.append  # type: ignore[method-assign]
    mode.checkForPackageUpdates = lambda: asyncio.sleep(0, result=[])  # type: ignore[method-assign]
    mode.checkTmuxKeyboardSetup = lambda: asyncio.sleep(0, result=None)  # type: ignore[method-assign]

    run_task = asyncio.create_task(mode.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    mode.requestShutdown()
    exit_code = await run_task

    assert exit_code == 0
    assert ui.started == 1
    assert prompts == [
        ("first", {"images": ["img-1"]}),
        ("second", None),
    ]
    assert warnings == ["fallback"]


@pytest.mark.asyncio
async def test_init_ensures_tools_and_logs_scoped_model_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_calls: list[str] = []
    printed: list[str] = []
    ui = FakeUi()
    mode = InteractiveMode(
        ui=ui,
        session=SimpleNamespace(
            resourceLoader=SimpleNamespace(getThemes=lambda: {"themes": []}),
            scopedModels=[{"model": _model("anthropic", "claude-scope"), "thinkingLevel": "high"}],
            state=SimpleNamespace(messages=[]),
        ),
        settingsManager=SimpleNamespace(
            getQuietStartup=lambda: False,
            getTheme=lambda: "dark",
        ),
        options={"verbose": True},
    )
    mode.keybindings.getKeys = lambda action: ["ctrl+k"] if action == "app.model.cycleForward" else []  # type: ignore[method-assign]
    mode.registerSignalHandlers = lambda: None  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: None  # type: ignore[method-assign]
    mode.setupAutocompleteProvider = lambda: None  # type: ignore[method-assign]
    mode.renderWidgets = lambda: None  # type: ignore[method-assign]
    mode.setupKeyHandlers = lambda: None  # type: ignore[method-assign]
    mode.setupEditorSubmitHandler = lambda: None  # type: ignore[method-assign]
    mode.updateAvailableProviderCount = lambda: None  # type: ignore[method-assign]
    mode.rebindCurrentSession = _noop_async  # type: ignore[method-assign]
    mode.renderInitialMessages = lambda: None  # type: ignore[method-assign]

    async def fake_ensure_tool(name: str, *args: Any, **kwargs: Any) -> str:
        tool_calls.append(name)
        return f"/tmp/{name}"

    monkeypatch.setattr(interactive_mode_module, "ensureTool", fake_ensure_tool)
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(str(message)))
    monkeypatch.setattr(interactive_mode_module.interactive_theme, "on_theme_change", lambda _callback: None)

    await mode.init()

    assert tool_calls == ["fd", "rg"]
    assert mode.fdPath == "/tmp/fd"
    assert printed and "Model scope: claude-scope:high" in _strip_ansi(printed[0])
    assert "to cycle" in _strip_ansi(printed[0])


@pytest.mark.asyncio
async def test_init_matches_ts_startup_header_and_render_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    ui = FakeUi()
    footer_data_provider = SimpleNamespace(onBranchChange=lambda _callback: calls.append("branchWatcher"))
    mode = InteractiveMode(
        ui=ui,
        footerDataProvider=footer_data_provider,
        settingsManager=SimpleNamespace(
            getQuietStartup=lambda: True,
            getTheme=lambda: "dark",
        ),
        options={"verbose": True},
    )
    mode.registerSignalHandlers = lambda: calls.append("signals")  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: calls.append("border")  # type: ignore[method-assign]
    mode.setupAutocompleteProvider = lambda: calls.append("autocomplete")  # type: ignore[method-assign]
    mode.renderWidgets = lambda: calls.append("widgets")  # type: ignore[method-assign]
    mode.setupKeyHandlers = lambda: calls.append("keyHandlers")  # type: ignore[method-assign]
    mode.setupEditorSubmitHandler = lambda: calls.append("submitHandlers")  # type: ignore[method-assign]
    mode.updateAvailableProviderCount = lambda: calls.append("count")  # type: ignore[method-assign]

    async def fake_rebind(*_args: Any, **_kwargs: Any) -> None:
        calls.append("rebind")

    mode.rebindCurrentSession = fake_rebind  # type: ignore[method-assign]
    mode.renderInitialMessages = lambda: calls.append("renderInitialMessages")  # type: ignore[method-assign]
    monkeypatch.setattr(
        interactive_mode_module.interactive_theme,
        "on_theme_change",
        lambda _callback: calls.append("themeWatcher"),
    )
    monkeypatch.setattr(interactive_mode_module, "ensureTool", _noop_async)

    await mode.init()

    assert ui.children == [
        mode.headerContainer,
        mode.chatContainer,
        mode.pendingMessagesContainer,
        mode.statusContainer,
        mode.widgetContainerAbove,
        mode.editorContainer,
        mode.widgetContainerBelow,
        mode.footer,
    ]
    assert ui.focused is mode.editor
    assert ui.started == 1
    assert calls == [
        "signals",
        "border",
        "autocomplete",
        "widgets",
        "keyHandlers",
        "submitHandlers",
        "rebind",
        "renderInitialMessages",
        "themeWatcher",
        "branchWatcher",
        "count",
    ]

    mode.builtInHeader.setExpanded(False)
    collapsed = _strip_ansi(str(getattr(mode.builtInHeader, "text", "")))
    assert "clear/exit" in collapsed
    assert "show full startup help and loaded resources" in collapsed
    assert "Ask it how to use or extend Pi." in collapsed

    mode.builtInHeader.setExpanded(True)
    expanded = _strip_ansi(str(getattr(mode.builtInHeader, "text", "")))
    assert "to suspend" in expanded
    assert "to queue follow-up" in expanded
    assert "to edit all queued messages" in expanded
    assert "drop files" in expanded


@pytest.mark.asyncio
async def test_run_shows_version_notification_from_background_check(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = FakeUi()

    async def fake_version_check(_version: str) -> LatestPiRelease | None:
        return LatestPiRelease(version="9.9.9", note="*New bits*")

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.check_for_new_pi_version",
        fake_version_check,
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.get_update_instruction",
        lambda _package_name: "Run: upgrade-pi",
    )
    monkeypatch.setattr(interactive_mode_module, "ensureTool", _noop_async)

    mode = InteractiveMode(ui=ui)
    mode.checkForPackageUpdates = lambda: asyncio.sleep(0, result=[])  # type: ignore[method-assign]
    mode.checkTmuxKeyboardSetup = lambda: asyncio.sleep(0, result=None)  # type: ignore[method-assign]

    run_task = asyncio.create_task(mode.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    mode.requestShutdown()
    assert await run_task == 0

    rendered = "\n".join(
        line
        for child in mode.chatContainer.children
        if hasattr(child, "render")
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)
    assert "Update Available" in stripped
    assert "New version 9.9.9 is available. Run: upgrade-pi" in stripped
    assert "Changelog:" in stripped


@pytest.mark.asyncio
async def test_run_surfaces_package_updates_tmux_warning_and_models_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_notifications: list[list[str]] = []
    warnings: list[str] = []
    errors: list[str] = []
    mode = InteractiveMode(
        ui=FakeUi(),
        session=SimpleNamespace(
            prompt=_noop_async,
            state=SimpleNamespace(messages=[]),
            promptTemplates=[],
            scopedModels=[],
            autoCompactionEnabled=False,
            isStreaming=False,
            isCompacting=False,
            extensionRunner=SimpleNamespace(
                get_registered_commands=lambda: [],
                get_message_renderer=lambda _custom_type: None,
            ),
            resourceLoader=SimpleNamespace(getSkills=lambda: {"skills": []}, getThemes=lambda: {"themes": []}),
            modelRegistry=SimpleNamespace(
                authStorage=SimpleNamespace(get=lambda *_args, **_kwargs: None),
                getApiKeyForProvider=_noop_async,
                getAvailable=lambda: [],
                isUsingOAuth=lambda _model: False,
                getError=lambda: "invalid models.json",
            ),
            subscribe=lambda _listener: (lambda: None),
            bindExtensions=_noop_async,
        ),
    )
    mode.checkForPackageUpdates = lambda: asyncio.sleep(0, result=["pkg-a", "pkg-b"])  # type: ignore[method-assign]
    mode.checkTmuxKeyboardSetup = lambda: asyncio.sleep(0, result="tmux warning")  # type: ignore[method-assign]
    mode.showPackageUpdateNotification = package_notifications.append  # type: ignore[method-assign]
    mode.showWarning = warnings.append  # type: ignore[method-assign]
    mode.showError = errors.append  # type: ignore[method-assign]
    monkeypatch.setattr(interactive_mode_module, "ensureTool", _noop_async)

    run_task = asyncio.create_task(mode.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    mode.requestShutdown()
    assert await run_task == 0

    assert package_notifications == [["pkg-a", "pkg-b"]]
    assert warnings == ["tmux warning"]
    assert errors == ["models.json error: invalid models.json"]


def test_get_changelog_for_display_returns_new_entries_and_updates_last_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_versions: list[str] = []
    telemetry_reports: list[str] = []
    mode = InteractiveMode(
        settingsManager=SimpleNamespace(
            getLastChangelogVersion=lambda: "1.0.0",
            setLastChangelogVersion=lambda version: saved_versions.append(version),
            getEnableInstallTelemetry=lambda: False,
        ),
        session=SimpleNamespace(state=SimpleNamespace(messages=[])),
    )
    mode.reportInstallTelemetry = lambda version: telemetry_reports.append(version)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.get_changelog_path",
        lambda: "/tmp/CHANGELOG.md",
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.parse_changelog",
        lambda _path: [
            ChangelogEntry(major=1, minor=1, patch=0, content="## 1.1.0\n- Added"),
            ChangelogEntry(major=1, minor=0, patch=0, content="## 1.0.0\n- Old"),
        ],
    )

    assert mode.getChangelogForDisplay() == "## 1.1.0\n- Added"
    assert saved_versions == [mode.version]
    assert telemetry_reports == [mode.version]


def test_get_changelog_for_display_skips_resumed_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    mode = InteractiveMode(
        settingsManager=SimpleNamespace(getLastChangelogVersion=lambda: "1.0.0"),
        session=SimpleNamespace(state=SimpleNamespace(messages=[{"role": "user"}])),
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.parse_changelog",
        lambda _path: called.append("parse") or [],
    )

    assert mode.getChangelogForDisplay() is None
    assert called == []


def test_show_startup_notices_condenses_changelog() -> None:
    mode = InteractiveMode(
        ui=FakeUi(),
        chatContainer=Container(),
        settingsManager=SimpleNamespace(getCollapseChangelog=lambda: True),
    )
    mode.changelogMarkdown = "## 1.2.3\n- Added"

    mode.showStartupNoticesIfNeeded()

    rendered = "\n".join(
        line
        for child in mode.chatContainer.children
        if isinstance(child, Text)
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)
    assert "Updated to v1.2.3." in stripped
    assert "/changelog" in stripped


def test_handle_changelog_command_renders_full_changelog(monkeypatch: pytest.MonkeyPatch) -> None:
    mode = InteractiveMode(ui=FakeUi(), chatContainer=Container())
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.get_changelog_path",
        lambda: "/tmp/CHANGELOG.md",
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.parse_changelog",
        lambda _path: [
            ChangelogEntry(major=1, minor=1, patch=0, content="## 1.1.0\n- Added"),
            ChangelogEntry(major=1, minor=0, patch=0, content="## 1.0.0\n- Old"),
        ],
    )

    mode.handleChangelogCommand()

    markdown_texts = [
        child.text for child in mode.chatContainer.children if child.__class__.__name__ == "Markdown"
    ]
    assert markdown_texts == ["## 1.0.0\n- Old\n\n## 1.1.0\n- Added"]


@pytest.mark.asyncio
async def test_check_for_package_updates_returns_display_names(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict[str, Any]] = []

    class FakePackageManager:
        def __init__(self, options: dict[str, Any]) -> None:
            constructed.append(dict(options))

        async def checkForAvailableUpdates(self) -> list[dict[str, str]]:
            return [
                {"displayName": "pkg-one"},
                {"displayName": "pkg-two"},
            ]

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.DefaultPackageManager",
        FakePackageManager,
    )

    mode = InteractiveMode(sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project"))
    assert await mode.checkForPackageUpdates() == ["pkg-one", "pkg-two"]
    assert constructed and constructed[0]["cwd"] == "/tmp/project"


def test_interactive_mode_module_exports_match_ts_surface() -> None:
    assert interactive_mode_module.__all__ == [
        "InteractiveMode",
        "InteractiveModeOptions",
        "isApiKeyLoginProvider",
    ]


def test_is_api_key_login_provider_matches_ts_semantics() -> None:
    assert interactive_mode_module.isApiKeyLoginProvider("anthropic", set()) is True
    assert interactive_mode_module.isApiKeyLoginProvider("custom-built-in", set(), {"custom-built-in"}) is False
    assert interactive_mode_module.isApiKeyLoginProvider("custom-oauth", {"custom-oauth"}) is False
    assert interactive_mode_module.isApiKeyLoginProvider("custom-api-key", {"custom-oauth"}) is True


def test_show_package_update_notification_matches_ts_copy() -> None:
    mode = InteractiveMode(ui=FakeUi(), chatContainer=Container())

    mode.showPackageUpdateNotification(["pkg-a", "pkg-b"])

    rendered = "\n".join(
        line
        for child in mode.chatContainer.children
        if isinstance(child, Text)
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)
    assert "Package Updates Available" in stripped
    assert "Package updates are available. Run" in stripped
    assert f"{APP_NAME} update" in stripped
    assert "Configured package updates are available" not in stripped
    assert "Packages:" in stripped
    assert "- pkg-a" in stripped
    assert "- pkg-b" in stripped


@pytest.mark.asyncio
async def test_check_tmux_keyboard_setup_warns_for_xterm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "1")
    responses = {
        "extended-keys": "always",
        "extended-keys-format": "xterm",
    }

    class FakeProcess:
        def __init__(self, option: str) -> None:
            self.option = option
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return responses[self.option].encode("utf-8"), b""

        async def wait(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    async def fake_create_subprocess_exec(*args: Any, **_kwargs: Any) -> FakeProcess:
        return FakeProcess(str(args[-1]))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    mode = InteractiveMode()
    warning = await mode.checkTmuxKeyboardSetup()
    assert warning is not None
    assert "extended-keys-format is xterm" in warning


@pytest.mark.asyncio
async def test_check_tmux_keyboard_setup_warns_for_disabled_extended_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TMUX", "1")
    responses = {
        "extended-keys": "off",
        "extended-keys-format": "csi-u",
    }

    class FakeProcess:
        def __init__(self, option: str) -> None:
            self.option = option
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return responses[self.option].encode("utf-8"), b""

        async def wait(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    async def fake_create_subprocess_exec(*args: Any, **_kwargs: Any) -> FakeProcess:
        return FakeProcess(str(args[-1]))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    mode = InteractiveMode()
    warning = await mode.checkTmuxKeyboardSetup()
    assert warning == (
        "tmux extended-keys is off. Modified Enter keys may not work. "
        "Add `set -g extended-keys on` to ~/.tmux.conf and restart tmux."
    )


@pytest.mark.asyncio
async def test_prompt_for_missing_session_cwd_uses_ts_title() -> None:
    prompts: list[tuple[str, str]] = []
    issue = SessionCwdIssue(
        sessionCwd="/missing/project",
        fallbackCwd="/current/project",
        sessionFile="/tmp/session.jsonl",
    )

    async def confirm(title: str, message: str, _opts: dict[str, Any] | None = None) -> bool:
        prompts.append((title, message))
        return True

    mode = InteractiveMode()
    mode.showExtensionConfirm = confirm  # type: ignore[method-assign]

    assert await mode.promptForMissingSessionCwd(MissingSessionCwdError(issue)) == "/current/project"
    assert prompts and prompts[0][0] == "Session cwd not found"
    assert "/missing/project" in prompts[0][1]


@pytest.mark.asyncio
async def test_handle_submitted_text_routes_commands_and_prompts() -> None:
    calls: list[Any] = []
    editor = FakeEditor()

    async def prompt(text: str, options: dict[str, Any] | None = None) -> None:
        calls.append(("prompt", text, options))

    async def handle_theme(theme: str) -> None:
        calls.append(("theme-command", theme))

    async def handle_import(text: str) -> None:
        calls.append(("import", text))

    async def handle_export(text: str) -> None:
        calls.append(("export", text))

    async def handle_clone() -> None:
        calls.append("clone")

    async def handle_models() -> None:
        calls.append("models")

    async def handle_clear() -> None:
        calls.append("new")

    async def handle_quit() -> int:
        calls.append("quit")
        return 0

    async def handle_bash(command: str, exclude: bool = False) -> None:
        calls.append(("bash", command, exclude))

    async def handle_share() -> None:
        calls.append("share")

    async def handle_copy() -> None:
        calls.append("copy")

    async def handle_login(mode: str) -> None:
        calls.append(("auth", mode))

    async def handle_reload() -> None:
        calls.append("reload")

    async def handle_compact(custom_instructions: str | None = None) -> None:
        calls.append(("compact", custom_instructions))

    mode = InteractiveMode(
        editor=editor,
        defaultEditor=editor,
        session=SimpleNamespace(
            prompt=prompt,
            isStreaming=False,
            state=SimpleNamespace(messages=[]),
        ),
    )
    mode.showSessionSelector = lambda: calls.append("resume-selector")  # type: ignore[method-assign]
    mode.showModelSelector = lambda search=None: calls.append(("model-selector", search))  # type: ignore[method-assign]
    mode.showThemeSelector = lambda: calls.append("theme-selector")  # type: ignore[method-assign]
    mode.showSettingsSelector = lambda: calls.append("settings-selector")  # type: ignore[method-assign]
    mode.handleChangelogCommand = lambda: calls.append("changelog")  # type: ignore[method-assign]
    mode.handleHotkeysCommand = lambda: calls.append("hotkeys")  # type: ignore[method-assign]
    mode.handleSessionCommand = lambda: calls.append("session-info")  # type: ignore[method-assign]
    mode.handleNameCommand = lambda text: calls.append(("name", text))  # type: ignore[method-assign]
    mode.showUserMessageSelector = lambda: calls.append("fork-selector")  # type: ignore[method-assign]
    mode.showTreeSelector = lambda initialSelectedId=None: calls.append(("tree-selector", initialSelectedId))  # type: ignore[method-assign]
    mode.showModelsSelector = handle_models  # type: ignore[method-assign]
    mode.handleThemeCommand = handle_theme  # type: ignore[method-assign]
    mode.handleExportCommand = handle_export  # type: ignore[method-assign]
    mode.handleImportCommand = handle_import  # type: ignore[method-assign]
    mode.handleCloneCommand = handle_clone  # type: ignore[method-assign]
    mode.handleShareCommand = handle_share  # type: ignore[method-assign]
    mode.handleCopyCommand = handle_copy  # type: ignore[method-assign]
    mode.showOAuthSelector = handle_login  # type: ignore[method-assign]
    mode.handleClearCommand = handle_clear  # type: ignore[method-assign]
    mode.handleCompactCommand = handle_compact  # type: ignore[method-assign]
    mode.handleReloadCommand = handle_reload  # type: ignore[method-assign]
    mode.handleDebugCommand = lambda: calls.append("debug")  # type: ignore[method-assign]
    mode.handleArminSaysHi = lambda: calls.append("armin")  # type: ignore[method-assign]
    mode.handleDementedDelves = lambda: calls.append("demented")  # type: ignore[method-assign]
    mode.shutdown = handle_quit  # type: ignore[method-assign]
    mode.handleBashCommand = handle_bash  # type: ignore[method-assign]
    mode.onInputCallback = lambda text: calls.append(("input-callback", text))

    await mode.handleSubmittedText("/resume")
    await mode.handleSubmittedText("/changelog")
    await mode.handleSubmittedText("/model sonnet")
    await mode.handleSubmittedText("/scoped-models")
    await mode.handleSubmittedText("/models")
    await mode.handleSubmittedText("/settings")
    await mode.handleSubmittedText('/export "session.html"')
    await mode.handleSubmittedText("/theme")
    await mode.handleSubmittedText("/theme light")
    await mode.handleSubmittedText('/import "session.jsonl"')
    await mode.handleSubmittedText("/clone")
    await mode.handleSubmittedText("/share")
    await mode.handleSubmittedText("/copy")
    await mode.handleSubmittedText("/name renamed")
    await mode.handleSubmittedText("/session")
    await mode.handleSubmittedText("/hotkeys")
    await mode.handleSubmittedText("/fork")
    await mode.handleSubmittedText("/tree")
    await mode.handleSubmittedText("/login")
    await mode.handleSubmittedText("/logout")
    await mode.handleSubmittedText("/new")
    await mode.handleSubmittedText("/compact focus on tests")
    await mode.handleSubmittedText("/reload")
    await mode.handleSubmittedText("/debug")
    await mode.handleSubmittedText("/arminsayshi")
    await mode.handleSubmittedText("/dementedelves")
    await mode.handleSubmittedText("/quit")
    await mode.handleSubmittedText("! ls -la")
    await mode.handleSubmittedText("!! pwd")
    await mode.handleSubmittedText("hello")

    assert calls == [
        "resume-selector",
        "changelog",
        ("model-selector", "sonnet"),
        "models",
        "models",
        "settings-selector",
        ("export", '/export "session.html"'),
        "theme-selector",
        ("theme-command", "light"),
        ("import", '/import "session.jsonl"'),
        "clone",
        "share",
        "copy",
        ("name", "/name renamed"),
        "session-info",
        "hotkeys",
        "fork-selector",
        ("tree-selector", None),
        ("auth", "login"),
        ("auth", "logout"),
        "new",
        ("compact", "focus on tests"),
        "reload",
        "debug",
        "armin",
        "demented",
        "quit",
        ("bash", "ls -la", False),
        ("bash", "pwd", True),
        ("input-callback", "hello"),
        ("prompt", "hello", None),
    ]
    assert editor.history == ["! ls -la", "!! pwd", "hello"]
    assert editor.text == ""


@pytest.mark.asyncio
async def test_handle_login_provider_select_routes_bedrock_to_setup_dialog() -> None:
    calls: list[tuple[str, str, str]] = []
    done_calls: list[bool] = []
    mode = InteractiveMode()
    mode.showBedrockSetupDialog = lambda provider_id, provider_name: calls.append(  # type: ignore[method-assign]
        ("bedrock", provider_id, provider_name)
    )

    async def fail_api_key_login(_provider_id: str, _provider_name: str) -> None:
        raise AssertionError("showApiKeyLoginDialog should not be called for Bedrock")

    async def fail_oauth_login(_provider_id: str, _provider_name: str) -> None:
        raise AssertionError("showLoginDialog should not be called for Bedrock")

    mode.showApiKeyLoginDialog = fail_api_key_login  # type: ignore[method-assign]
    mode.showLoginDialog = fail_oauth_login  # type: ignore[method-assign]

    await mode._handle_login_provider_select(
        [SimpleNamespace(id="amazon-bedrock", name="Amazon Bedrock", authType="api_key")],
        "amazon-bedrock",
        lambda: done_calls.append(True),
    )

    assert done_calls == [True]
    assert calls == [("bedrock", "amazon-bedrock", "Amazon Bedrock")]


@pytest.mark.asyncio
async def test_handle_logout_provider_select_matches_ts_refresh_path() -> None:
    calls: list[Any] = []
    statuses: list[str] = []
    mode = InteractiveMode(
        session=SimpleNamespace(
            modelRegistry=SimpleNamespace(
                authStorage=SimpleNamespace(logout=lambda provider_id: calls.append(("logout", provider_id))),
                refresh=lambda: calls.append("refresh"),
            )
        )
    )
    mode.updateAvailableProviderCount = lambda: calls.append("count")  # type: ignore[method-assign]
    mode.setupAutocompleteProvider = lambda: calls.append("autocomplete")  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: calls.append("border")  # type: ignore[method-assign]
    mode.footer = SimpleNamespace(invalidate=lambda: calls.append("footer"))
    mode.showStatus = statuses.append  # type: ignore[method-assign]

    await mode._handle_logout_provider_select(
        [SimpleNamespace(id="anthropic", name="Anthropic", authType="oauth")],
        "anthropic",
        lambda: calls.append("done"),
    )

    assert calls == ["done", ("logout", "anthropic"), "refresh", "count"]
    assert statuses == ["Logged out of Anthropic"]


@pytest.mark.asyncio
async def test_complete_provider_authentication_selects_default_model_for_unknown_current_model() -> None:
    statuses: list[str] = []
    errors: list[str] = []
    set_model_calls: list[str] = []
    warnings: list[str | None] = []
    daxnuts: list[str] = []
    default_model = SimpleNamespace(provider="anthropic", id="claude-opus-4-7")
    unknown_model = SimpleNamespace(provider="unknown", id="unknown", api="unknown")
    mode = InteractiveMode(
        session=SimpleNamespace(
            setModel=lambda model: set_model_calls.append(model.id),
            modelRegistry=SimpleNamespace(
                refresh=lambda: None,
                getAvailable=lambda: [default_model],
            ),
        )
    )
    mode.footer = SimpleNamespace(invalidate=lambda: None)
    mode.updateAvailableProviderCount = lambda: None  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: None  # type: ignore[method-assign]
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.showError = errors.append  # type: ignore[method-assign]

    async def maybe_warn(model: Any = None) -> None:
        warnings.append(None if model is None else model.id)

    mode.maybeWarnAboutAnthropicSubscriptionAuth = maybe_warn  # type: ignore[method-assign]
    mode.checkDaxnutsEasterEgg = lambda model: daxnuts.append(model.id)  # type: ignore[method-assign]

    await mode.completeProviderAuthentication("anthropic", "Anthropic", "oauth", unknown_model)

    assert set_model_calls == ["claude-opus-4-7"]
    assert statuses == [
        f"Logged in to Anthropic. Selected claude-opus-4-7. Credentials saved to {interactive_mode_module.get_auth_path()}"
    ]
    assert errors == []
    assert warnings == ["claude-opus-4-7"]
    assert daxnuts == ["claude-opus-4-7"]


def test_show_bedrock_setup_dialog_renders_info_with_docs_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeDialog:
        def __init__(
            self,
            tui: Any,
            providerId: str,
            onComplete: Any,
            providerName: str,
            titleOverride: str | None = None,
        ) -> None:
            captured["tui"] = tui
            captured["providerId"] = providerId
            captured["onComplete"] = onComplete
            captured["providerName"] = providerName
            captured["titleOverride"] = titleOverride

        def showInfo(self, lines: list[str]) -> None:
            captured["lines"] = lines

    monkeypatch.setattr(interactive_mode_module, "LoginDialogComponent", FakeDialog)
    monkeypatch.setattr(interactive_mode_module, "get_docs_path", lambda: "/tmp/docs")

    ui = FakeUi()
    editor = FakeEditor()
    mode = InteractiveMode(ui=ui, editor=editor, defaultEditor=editor)

    mode.showBedrockSetupDialog("amazon-bedrock", "Amazon Bedrock")

    assert captured["providerId"] == "amazon-bedrock"
    assert captured["providerName"] == "Amazon Bedrock"
    assert captured["titleOverride"] == "Amazon Bedrock setup"
    assert any("/tmp/docs/providers.md" in line for line in captured["lines"])
    assert mode.editorContainer.children == [ui.focused]
    assert ui.render_calls == [None]


def test_update_pending_messages_display_renders_session_and_compaction_queues() -> None:
    mode = InteractiveMode(
        pendingMessagesContainer=Container(),
        session=SimpleNamespace(
            getSteeringMessages=lambda: ["queued steer"],
            getFollowUpMessages=lambda: ["queued follow"],
        ),
    )
    mode.compactionQueuedMessages = [
        {"text": "compaction steer", "mode": "steer"},
        {"text": "compaction follow", "mode": "followUp"},
    ]

    mode.updatePendingMessagesDisplay()

    rendered = "\n".join(
        line
        for child in mode.pendingMessagesContainer.children
        if hasattr(child, "render")
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)
    assert "Steering: queued steer" in stripped
    assert "Steering: compaction steer" in stripped
    assert "Follow-up: queued follow" in stripped
    assert "Follow-up: compaction follow" in stripped
    assert "to edit all queued messages" in stripped


def test_restore_queued_messages_to_editor_merges_session_and_compaction_queues_and_aborts() -> None:
    editor = FakeEditor()
    editor.setText("draft")
    aborts: list[bool] = []
    mode = InteractiveMode(
        editor=editor,
        defaultEditor=editor,
        pendingMessagesContainer=Container(),
        session=SimpleNamespace(
            clearQueue=lambda: {"steering": ["queued steer"], "followUp": ["queued follow"]},
            agent=SimpleNamespace(abort=lambda: aborts.append(True)),
        ),
    )
    mode.compactionQueuedMessages = [
        {"text": "compaction steer", "mode": "steer"},
        {"text": "compaction follow", "mode": "followUp"},
    ]

    restored = mode.restoreQueuedMessagesToEditor({"abort": True})

    assert restored == 4
    assert (
        editor.text
        == "queued steer\n\ncompaction steer\n\nqueued follow\n\ncompaction follow\n\ndraft"
    )
    assert aborts == [True]
    assert mode.compactionQueuedMessages == []


@pytest.mark.asyncio
async def test_handle_follow_up_streaming_queues_follow_up_and_updates_pending() -> None:
    editor = FakeEditor()
    editor.setText("later")
    calls: list[Any] = []
    ui = FakeUi()

    async def prompt(text: str, options: dict[str, Any] | None = None) -> None:
        calls.append(("prompt", text, options))

    mode = InteractiveMode(
        ui=ui,
        editor=editor,
        defaultEditor=editor,
        session=SimpleNamespace(prompt=prompt, isStreaming=True, isCompacting=False),
    )
    mode.updatePendingMessagesDisplay = lambda: calls.append("pending")  # type: ignore[method-assign]

    await mode.handleFollowUp()

    assert calls == [("prompt", "later", {"streamingBehavior": "followUp"}), "pending"]
    assert editor.history == ["later"]
    assert editor.text == ""
    assert ui.render_calls == [None]


@pytest.mark.asyncio
async def test_handle_submitted_text_queues_message_during_compaction() -> None:
    editor = FakeEditor()
    statuses: list[str] = []
    mode = InteractiveMode(
        ui=FakeUi(),
        editor=editor,
        defaultEditor=editor,
        pendingMessagesContainer=Container(),
        session=SimpleNamespace(
            prompt=_noop_async,
            isStreaming=False,
            isCompacting=True,
            extensionRunner=SimpleNamespace(getCommand=lambda _name: None),
            state=SimpleNamespace(messages=[]),
        ),
    )
    mode.showStatus = statuses.append  # type: ignore[method-assign]

    await mode.handleSubmittedText("hello while compacting")

    assert editor.history == ["hello while compacting"]
    assert editor.text == ""
    assert mode.compactionQueuedMessages == [{"text": "hello while compacting", "mode": "steer"}]
    assert statuses == ["Queued message for after compaction"]


@pytest.mark.asyncio
async def test_handle_submitted_text_streaming_uses_steer_and_updates_pending() -> None:
    editor = FakeEditor()
    calls: list[Any] = []
    ui = FakeUi()

    async def prompt(text: str, options: dict[str, Any] | None = None) -> None:
        calls.append(("prompt", text, options))

    mode = InteractiveMode(
        ui=ui,
        editor=editor,
        defaultEditor=editor,
        session=SimpleNamespace(prompt=prompt, isStreaming=True, isCompacting=False, state=SimpleNamespace(messages=[])),
    )
    mode.onInputCallback = lambda text: calls.append(("input", text))
    mode.updatePendingMessagesDisplay = lambda: calls.append("pending")  # type: ignore[method-assign]

    await mode.handleSubmittedText("hello while streaming")

    assert calls == [("prompt", "hello while streaming", {"streamingBehavior": "steer"}), "pending"]
    assert editor.history == ["hello while streaming"]
    assert editor.text == ""
    assert ui.render_calls == [None]


@pytest.mark.asyncio
async def test_flush_compaction_queue_routes_messages_by_mode() -> None:
    calls: list[tuple[str, Any]] = []

    async def prompt(text: str, options: dict[str, Any] | None = None) -> None:
        calls.append(("prompt", text))

    async def follow_up(text: str) -> None:
        calls.append(("followUp", text))

    async def steer(text: str) -> None:
        calls.append(("steer", text))

    mode = InteractiveMode(
        session=SimpleNamespace(
            prompt=prompt,
            followUp=follow_up,
            steer=steer,
            extensionRunner=SimpleNamespace(getCommand=lambda name: object() if name == "ext" else None),
        ),
    )
    mode.compactionQueuedMessages = [
        {"text": "/ext do-thing", "mode": "steer"},
        {"text": "primary prompt", "mode": "steer"},
        {"text": "queued follow-up", "mode": "followUp"},
        {"text": "queued steer", "mode": "steer"},
    ]

    await mode.flushCompactionQueue({"willRetry": False})
    await asyncio.sleep(0)

    assert calls[0] == ("prompt", "/ext do-thing")
    assert set(calls[1:]) == {
        ("prompt", "primary prompt"),
        ("followUp", "queued follow-up"),
        ("steer", "queued steer"),
    }
    assert mode.compactionQueuedMessages == []


def test_flush_pending_bash_components_moves_components_to_chat() -> None:
    pending = Container()
    chat = Container()
    first = Text("one", 0, 0)
    second = Text("two", 0, 0)
    pending.addChild(first)
    pending.addChild(second)

    mode = InteractiveMode(pendingMessagesContainer=pending, chatContainer=chat)
    mode.pendingBashComponents = [first, second]

    mode.flushPendingBashComponents()

    assert pending.children == []
    assert chat.children[-2:] == [first, second]
    assert mode.pendingBashComponents == []


@pytest.mark.asyncio
async def test_handle_event_queue_update_refreshes_pending_messages() -> None:
    calls: list[str] = []
    mode = InteractiveMode()
    mode.updatePendingMessagesDisplay = lambda: calls.append("pending")  # type: ignore[method-assign]
    mode._request_render = lambda force=None: calls.append("render")  # type: ignore[method-assign]
    mode.isInitialized = True

    await mode.handleEvent({"type": "queue_update"})

    assert calls == ["pending", "render"]


@pytest.mark.asyncio
async def test_handle_event_initializes_mode_before_processing() -> None:
    calls: list[str] = []
    mode = InteractiveMode()
    mode.init = lambda: asyncio.sleep(0, result=calls.append("init"))  # type: ignore[method-assign]
    mode.updatePendingMessagesDisplay = lambda: calls.append("pending")  # type: ignore[method-assign]
    mode._request_render = lambda force=None: calls.append("render")  # type: ignore[method-assign]

    await mode.handleEvent({"type": "queue_update"})

    assert calls[:2] == ["init", "pending"]


def test_check_daxnuts_easter_egg_renders_for_kimi_k25_opencode_model() -> None:
    calls: list[str] = []
    mode = InteractiveMode()
    mode.handleDaxnuts = lambda: calls.append("daxnuts")  # type: ignore[method-assign]

    mode.checkDaxnutsEasterEgg({"provider": "opencode", "id": "KIMI-K2.5-special"})
    mode.checkDaxnutsEasterEgg({"provider": "openai", "id": "gpt-4.1"})

    assert calls == ["daxnuts"]


def test_create_base_autocomplete_provider_includes_restored_builtin_commands() -> None:
    mode = InteractiveMode(
        session=SimpleNamespace(getSlashCommands=lambda: [], state=SimpleNamespace(messages=[])),
        sessionManager=SimpleNamespace(getCwd=lambda: "/tmp/project"),
    )
    provider = mode.createBaseAutocompleteProvider()
    command_names = {command.name for command in provider.commands}
    assert {
        "settings",
        "model",
        "scoped-models",
        "export",
        "share",
        "copy",
        "name",
        "session",
        "hotkeys",
        "login",
        "logout",
        "compact",
        "reload",
        "quit",
        "models",
        "theme",
    }.issubset(command_names)


@pytest.mark.asyncio
async def test_handle_model_command_prefers_exact_match_before_selector() -> None:
    statuses: list[str] = []
    selected: list[str] = []
    shown: list[str | None] = []
    model = _model("openai", "gpt-4o-mini")

    async def set_model(next_model: Model) -> None:
        selected.append(next_model.id)

    mode = InteractiveMode(
        session=SimpleNamespace(
            scopedModels=[],
            modelRegistry=SimpleNamespace(refresh=lambda: None, getAvailable=lambda: [model]),
            setModel=set_model,
            state=SimpleNamespace(thinkingLevel="off"),
        ),
        footer=SimpleNamespace(invalidate=lambda: None),
        ui=FakeUi(),
    )
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.showModelSelector = lambda search=None: shown.append(search)  # type: ignore[method-assign]
    mode.updateAvailableProviderCount = lambda: None  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: None  # type: ignore[method-assign]
    mode.maybeWarnAboutAnthropicSubscriptionAuth = lambda _model=None: asyncio.sleep(0)  # type: ignore[method-assign]

    await mode.handleModelCommand("openai/gpt-4o-mini")
    await mode.handleModelCommand("missing")

    assert selected == ["gpt-4o-mini"]
    assert statuses == ["Model: gpt-4o-mini"]
    assert shown == ["missing"]


def test_setup_key_handlers_registers_session_fork_action() -> None:
    editor = FakeEditor()
    calls: list[str] = []
    mode = InteractiveMode(defaultEditor=editor, editor=editor)
    mode.showUserMessageSelector = lambda: calls.append("fork")  # type: ignore[method-assign]

    mode.setupKeyHandlers()
    editor.actions["app.session.fork"]()

    assert calls == ["fork"]


def test_setup_key_handlers_registers_session_tree_action() -> None:
    editor = FakeEditor()
    calls: list[str] = []
    mode = InteractiveMode(defaultEditor=editor, editor=editor)
    mode.showTreeSelector = lambda initialSelectedId=None: calls.append(str(initialSelectedId))  # type: ignore[method-assign]

    mode.setupKeyHandlers()
    editor.actions["app.session.tree"]()

    assert calls == ["None"]


def test_setup_key_handlers_escape_clears_bash_mode_input() -> None:
    editor = FakeEditor()
    editor.setText("! ls -la")
    border_updates: list[bool] = []
    mode = InteractiveMode(defaultEditor=editor, editor=editor)
    mode.updateEditorBorderColor = lambda: border_updates.append(True)  # type: ignore[method-assign]

    mode.setupKeyHandlers()
    assert editor.onEscape is not None
    editor.onEscape()

    assert editor.text == ""
    assert border_updates == [True]


def test_setup_key_handlers_double_escape_opens_tree_selector() -> None:
    editor = FakeEditor()
    calls: list[Any] = []
    mode = InteractiveMode(
        defaultEditor=editor,
        editor=editor,
        settingsManager=SimpleNamespace(getDoubleEscapeAction=lambda: "tree"),
    )
    mode.showTreeSelector = lambda initialSelectedId=None: calls.append(initialSelectedId)  # type: ignore[method-assign]
    mode.lastEscapeTime = time.monotonic() * 1000

    mode.setupKeyHandlers()
    assert editor.onEscape is not None
    editor.onEscape()

    assert calls == [None]
    assert mode.lastEscapeTime == 0


@pytest.mark.asyncio
async def test_setup_key_handlers_registers_new_session_action_via_handle_clear_command() -> None:
    editor = FakeEditor()
    calls: list[str] = []
    mode = InteractiveMode(defaultEditor=editor, editor=editor)

    async def handle_clear() -> None:
        calls.append("new")

    mode.handleClearCommand = handle_clear  # type: ignore[method-assign]

    mode.setupKeyHandlers()
    editor.actions["app.session.new"]()
    await asyncio.sleep(0)

    assert calls == ["new"]


@pytest.mark.asyncio
async def test_setup_key_handlers_registers_external_editor_action() -> None:
    editor = FakeEditor()
    calls: list[str] = []
    mode = InteractiveMode(defaultEditor=editor, editor=editor)

    async def open_external() -> None:
        calls.append("external")

    mode.openExternalEditor = open_external  # type: ignore[method-assign]

    mode.setupKeyHandlers()
    editor.actions["app.editor.external"]()
    await asyncio.sleep(0)

    assert calls == ["external"]


def test_setup_key_handlers_registers_paste_image_handler() -> None:
    editor = FakeEditor()
    mode = InteractiveMode(defaultEditor=editor, editor=editor, ui=FakeUi())

    mode.setupKeyHandlers()

    assert editor.onPasteImage is not None


@pytest.mark.asyncio
async def test_handle_clear_command_renders_new_session_message() -> None:
    started: list[str] = []
    stopped: list[bool] = []
    cleared: list[bool] = []
    mode = InteractiveMode(
        ui=FakeUi(),
        chatContainer=Container(),
        runtimeHost=SimpleNamespace(newSession=lambda: asyncio.sleep(0, result={"cancelled": False})),
        statusContainer=SimpleNamespace(clear=lambda: cleared.append(True)),
        loadingAnimation=SimpleNamespace(stop=lambda: stopped.append(True)),
    )
    mode.renderCurrentSessionState = lambda: started.append("rendered")  # type: ignore[method-assign]

    await mode.handleClearCommand()

    rendered = "\n".join(
        line
        for child in mode.chatContainer.children
        if isinstance(child, Text)
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)
    assert started == ["rendered"]
    assert stopped == [True]
    assert cleared == [True]
    assert "✓ New session started" in stripped


@pytest.mark.asyncio
async def test_open_external_editor_warns_when_editor_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    warnings: list[str] = []

    mode = InteractiveMode(editor=FakeEditor(), defaultEditor=FakeEditor())
    mode.showWarning = warnings.append  # type: ignore[method-assign]

    await mode.openExternalEditor()

    assert warnings == ["No editor configured. Set $VISUAL or $EDITOR environment variable."]


@pytest.mark.asyncio
async def test_open_external_editor_ignores_launch_failure_and_restores_ui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EDITOR", "missing-editor")
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing-editor")),
    )

    ui = FakeUi()
    editor = FakeEditor()
    editor.setText("original text")
    mode = InteractiveMode(ui=ui, editor=editor, defaultEditor=editor)

    await mode.openExternalEditor()

    assert editor.text == "original text"
    assert ui.stopped == 1
    assert ui.started == 1
    assert ui.render_calls == [True]


def test_handle_debug_command_writes_log_and_renders_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    debug_log_path = tmp_path / "debug.log"
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.get_debug_log_path",
        lambda: str(debug_log_path),
    )

    ui = FakeUi()
    ui.render = lambda width: [f"width={width}", "hello"]  # type: ignore[method-assign]
    ui.terminal = SimpleNamespace(
        columns=80,
        rows=24,
        setProgress=lambda _value: None,
        setTitle=lambda _value: None,
    )
    mode = InteractiveMode(
        ui=ui,
        chatContainer=Container(),
        session=SimpleNamespace(messages=[{"role": "user", "content": "hi"}]),
    )

    mode.handleDebugCommand()

    content = debug_log_path.read_text(encoding="utf-8")
    assert "Terminal: 80x24" in content
    assert '[0] (w=8) "width=80"' in content
    assert '{"role": "user", "content": "hi"}' in content

    rendered = "\n".join(
        line
        for child in mode.chatContainer.children
        if isinstance(child, Text)
        for line in child.render(120)
    )
    stripped = _strip_ansi(rendered)
    assert "✓ Debug log written" in stripped
    assert str(debug_log_path) in stripped


@pytest.mark.asyncio
async def test_cycle_model_matches_ts_status_messages() -> None:
    statuses: list[str] = []
    warnings: list[str] = []
    mode = InteractiveMode(
        session=SimpleNamespace(
            cycleModel=lambda direction: asyncio.sleep(
                0,
                result=SimpleNamespace(
                    model=SimpleNamespace(id="sonnet", name="Claude Sonnet", reasoning=True),
                    thinkingLevel="high",
                ),
            ),
            scopedModels=[],
        ),
    )
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: warnings.append("border")  # type: ignore[method-assign]
    mode.maybeWarnAboutAnthropicSubscriptionAuth = lambda _model: asyncio.sleep(0)  # type: ignore[method-assign]

    await mode._cycle_model("forward")

    assert statuses == ["Switched to Claude Sonnet (thinking: high)"]
    assert warnings == ["border"]


@pytest.mark.asyncio
async def test_cycle_model_reports_single_model_and_errors_like_ts() -> None:
    statuses: list[str] = []
    errors: list[str] = []
    mode = InteractiveMode(session=SimpleNamespace(cycleModel=lambda _direction: asyncio.sleep(0, result=None), scopedModels=[1]))
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.showError = errors.append  # type: ignore[method-assign]

    await mode._cycle_model("forward")

    assert statuses == ["Only one model in scope"]
    assert errors == []

    async def fail_cycle(_direction: str) -> Any:
        raise RuntimeError("boom")

    mode.session = SimpleNamespace(cycleModel=fail_cycle, scopedModels=[])
    await mode._cycle_model("forward")

    assert errors == ["boom"]


@pytest.mark.asyncio
async def test_cycle_thinking_level_matches_ts_status_messages() -> None:
    statuses: list[str] = []
    border_updates: list[bool] = []
    mode = InteractiveMode(session=SimpleNamespace(cycleThinkingLevel=lambda: None))
    mode.showStatus = statuses.append  # type: ignore[method-assign]

    await mode._cycle_thinking_level()

    assert statuses == ["Current model does not support thinking"]

    mode.session = SimpleNamespace(cycleThinkingLevel=lambda: "high")
    mode.updateEditorBorderColor = lambda: border_updates.append(True)  # type: ignore[method-assign]
    await mode._cycle_thinking_level()

    assert statuses[-1] == "Thinking level: high"
    assert border_updates == [True]


@pytest.mark.asyncio
async def test_command_context_navigate_tree_updates_chat_and_editor() -> None:
    renders: list[bool] = []
    statuses: list[str] = []
    flushes: list[dict[str, bool]] = []
    editor = FakeEditor()

    async def navigate_tree(_target_id: str, _options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"cancelled": False, "editorText": "restored draft"}

    mode = InteractiveMode(
        editor=editor,
        defaultEditor=editor,
        session=SimpleNamespace(navigateTree=navigate_tree),
    )
    mode.renderCurrentSessionState = lambda: renders.append(True)  # type: ignore[method-assign]
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.flushCompactionQueue = lambda options: flushes.append(options)  # type: ignore[method-assign]

    actions = mode._build_command_context_actions()
    result = await actions["navigateTree"]("entry-1", {"summarize": False})

    assert result["cancelled"] is False
    assert renders == [True]
    assert editor.text == "restored draft"
    assert statuses == ["Navigated to selected point"]
    assert flushes == [{"willRetry": False}]


@pytest.mark.asyncio
async def test_handle_tree_select_prompts_for_summary_and_passes_custom_instructions() -> None:
    calls: list[tuple[str, Any]] = []
    statuses: list[str] = []
    flushes: list[dict[str, bool]] = []
    editor = FakeEditor()

    async def navigate_tree(_target_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append(("navigate", dict(options or {})))
        return {"cancelled": False, "editorText": "restored draft"}

    mode = InteractiveMode(
        editor=editor,
        defaultEditor=editor,
        session=SimpleNamespace(
            navigateTree=navigate_tree,
            abortBranchSummary=lambda: calls.append(("abort", None)),
        ),
        statusContainer=SimpleNamespace(
            children=[],
            clear=lambda: calls.append(("clear-status", None)),
            addChild=lambda child: calls.append(("loader", child)),
        ),
        settingsManager=SimpleNamespace(getBranchSummarySkipPrompt=lambda: False),
    )
    mode.showExtensionSelector = lambda title, options, opts=None: asyncio.sleep(  # type: ignore[method-assign]
        0,
        result="Summarize with custom prompt",
    )
    mode.showExtensionEditor = lambda title, prefill=None: asyncio.sleep(0, result="focus on files")  # type: ignore[method-assign]
    mode.renderCurrentSessionState = lambda: calls.append(("render", None))  # type: ignore[method-assign]
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.flushCompactionQueue = lambda options: flushes.append(options)  # type: ignore[method-assign]

    await mode._handle_tree_select("entry-2", "entry-1", lambda: calls.append(("done", None)))

    assert ("done", None) in calls
    assert any(name == "loader" for name, _value in calls)
    assert ("render", None) in calls
    assert calls[-1] == ("clear-status", None)
    assert editor.text == "restored draft"
    assert statuses == ["Navigated to selected point"]
    assert flushes == [{"willRetry": False}]
    assert calls.count(("navigate", {"summarize": True, "customInstructions": "focus on files"})) == 1


@pytest.mark.asyncio
async def test_handle_clipboard_image_paste_writes_temp_file_and_inserts_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ui = FakeUi()
    editor = FakeEditor()
    mode = InteractiveMode(defaultEditor=editor, editor=editor, ui=ui)

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.read_clipboard_image",
        lambda: asyncio.sleep(0, result=SimpleNamespace(bytes=b"\x89PNG\r\n\x1a\n", mimeType="image/png")),
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.uuid4",
        lambda: "fixed-id",
    )

    await mode.handleClipboardImagePaste()

    expected_path = tmp_path / "pi-clipboard-fixed-id.png"
    assert expected_path.read_bytes() == b"\x89PNG\r\n\x1a\n"
    assert editor.inserted == [str(expected_path)]
    assert ui.render_calls == [None]


@pytest.mark.asyncio
async def test_handle_tree_select_reopens_tree_when_summary_prompt_cancelled() -> None:
    reopened: list[str | None] = []
    mode = InteractiveMode(
        session=SimpleNamespace(navigateTree=lambda *_args, **_kwargs: asyncio.sleep(0)),
        settingsManager=SimpleNamespace(getBranchSummarySkipPrompt=lambda: False),
    )
    mode.showExtensionSelector = lambda title, options, opts=None: asyncio.sleep(0, result=None)  # type: ignore[method-assign]
    mode.showTreeSelector = lambda initialSelectedId=None: reopened.append(initialSelectedId)  # type: ignore[method-assign]

    await mode._handle_tree_select("entry-7", "entry-1", lambda: None)

    assert reopened == ["entry-7"]


def test_show_settings_selector_builds_live_settings_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeSettingsSelectorComponent:
        def __init__(self, config: Any, callbacks: Any) -> None:
            captured["config"] = config
            captured["callbacks"] = callbacks

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.SettingsSelectorComponent",
        FakeSettingsSelectorComponent,
    )

    settings_calls: list[tuple[str, Any]] = []
    tool_calls: list[tuple[str, Any]] = []
    ui = FakeUi()
    mode = InteractiveMode(
        ui=ui,
        settingsManager=SimpleNamespace(
            getShowImages=lambda: True,
            setShowImages=lambda value: settings_calls.append(("showImages", value)),
            getImageWidthCells=lambda: 48,
            setImageWidthCells=lambda value: settings_calls.append(("imageWidthCells", value)),
            getImageAutoResize=lambda: True,
            setImageAutoResize=lambda value: settings_calls.append(("imageAutoResize", value)),
            getBlockImages=lambda: False,
            setBlockImages=lambda value: settings_calls.append(("blockImages", value)),
            getEnableSkillCommands=lambda: True,
            setEnableSkillCommands=lambda value: settings_calls.append(("enableSkillCommands", value)),
            getTransport=lambda: "sse",
            setTransport=lambda value: settings_calls.append(("transport", value)),
            getHttpIdleTimeoutMs=lambda: 300_000,
            setHttpIdleTimeoutMs=lambda value: settings_calls.append(("httpIdleTimeoutMs", value)),
            getTheme=lambda: "dark",
            getCollapseChangelog=lambda: True,
            setCollapseChangelog=lambda value: settings_calls.append(("collapseChangelog", value)),
            getEnableInstallTelemetry=lambda: True,
            setEnableInstallTelemetry=lambda value: settings_calls.append(("installTelemetry", value)),
            getDoubleEscapeAction=lambda: "tree",
            setDoubleEscapeAction=lambda value: settings_calls.append(("doubleEscapeAction", value)),
            getTreeFilterMode=lambda: "default",
            setTreeFilterMode=lambda value: settings_calls.append(("treeFilterMode", value)),
            getShowHardwareCursor=lambda: False,
            setShowHardwareCursor=lambda value: settings_calls.append(("showHardwareCursor", value)),
            getEditorPaddingX=lambda: 1,
            setEditorPaddingX=lambda value: settings_calls.append(("editorPaddingX", value)),
            getAutocompleteMaxVisible=lambda: 5,
            setAutocompleteMaxVisible=lambda value: settings_calls.append(("autocompleteMaxVisible", value)),
            getQuietStartup=lambda: False,
            setQuietStartup=lambda value: settings_calls.append(("quietStartup", value)),
            getClearOnShrink=lambda: False,
            setClearOnShrink=lambda value: settings_calls.append(("clearOnShrink", value)),
            getShowTerminalProgress=lambda: False,
            setShowTerminalProgress=lambda value: settings_calls.append(("showTerminalProgress", value)),
            getWarnings=lambda: {"anthropicExtraUsage": True},
            setWarnings=lambda value: settings_calls.append(("warnings", value)),
            setCompactionEnabled=lambda value: settings_calls.append(("autoCompact", value)),
            setHideThinkingBlock=lambda value: settings_calls.append(("hideThinkingBlock", value)),
        ),
        session=SimpleNamespace(
            autoCompactionEnabled=False,
            steeringMode="one-at-a-time",
            followUpMode="one-at-a-time",
            state=SimpleNamespace(thinkingLevel="off"),
            getAvailableThinkingLevels=lambda: ["off", "high"],
            setSteeringMode=lambda value: settings_calls.append(("steeringMode", value)),
            setFollowUpMode=lambda value: settings_calls.append(("followUpMode", value)),
            setThinkingLevel=lambda value: settings_calls.append(("thinkingLevel", value)),
        ),
        footer=SimpleNamespace(setAutoCompactEnabled=lambda value: settings_calls.append(("footerAutoCompact", value))),
        chatContainer=SimpleNamespace(
            children=[
                SimpleNamespace(
                    setShowImages=lambda value: tool_calls.append(("showImages", value)),
                    setImageWidthCells=lambda value: tool_calls.append(("imageWidthCells", value)),
                    setHideThinkingBlock=lambda value: tool_calls.append(("hideThinkingBlock", value)),
                )
            ]
        ),
        defaultEditor=FakeEditor(),
        editor=FakeEditor(),
    )
    mode.setupAutocompleteProvider = lambda: settings_calls.append(("autocomplete", True))  # type: ignore[method-assign]
    mode.renderCurrentSessionState = lambda: settings_calls.append(("rebuild", True))  # type: ignore[method-assign]
    mode.showStatus = lambda message: settings_calls.append(("status", message))  # type: ignore[method-assign]
    mode.updateEditorBorderColor = lambda: settings_calls.append(("border", True))  # type: ignore[method-assign]

    mode.showSettingsSelector()

    config = captured["config"]
    callbacks = captured["callbacks"]
    assert config.currentTheme == "dark"
    assert config.availableThinkingLevels == ["off", "high"]

    callbacks.onShowImagesChange(False)
    callbacks.onImageWidthCellsChange(64)
    callbacks.onEnableSkillCommandsChange(False)
    callbacks.onHideThinkingBlockChange(True)

    assert ("showImages", False) in settings_calls
    assert ("imageWidthCells", 64) in settings_calls
    assert ("enableSkillCommands", False) in settings_calls
    assert ("hideThinkingBlock", True) in settings_calls
    assert ("showImages", False) in tool_calls
    assert ("imageWidthCells", 64) in tool_calls
    assert ("hideThinkingBlock", True) in tool_calls
    assert ("autocomplete", True) in settings_calls
    assert ("rebuild", True) in settings_calls


@pytest.mark.asyncio
async def test_show_models_selector_updates_session_scope_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    all_models = [_model("openai", "gpt-4o-mini"), _model("anthropic", "claude-sonnet-4-5")]

    class FakeScopedModelsSelectorComponent:
        def __init__(self, config: Any, callbacks: Any) -> None:
            captured["config"] = config
            captured["callbacks"] = callbacks

    async def fake_resolve_model_scope(patterns: list[str], _registry: Any) -> list[Any]:
        resolved: list[Any] = []
        for pattern in patterns:
            provider, model_id = pattern.split("/", 1)
            model = next(model for model in all_models if model.provider == provider and model.id == model_id)
            resolved.append(SimpleNamespace(model=model, thinkingLevel="high"))
        return resolved

    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.ScopedModelsSelectorComponent",
        FakeScopedModelsSelectorComponent,
    )
    monkeypatch.setattr(
        "harnify_coding_agent.modes.interactive.interactive_mode.resolveModelScope",
        fake_resolve_model_scope,
    )

    scoped_updates: list[list[dict[str, Any]]] = []
    persisted: list[Any] = []
    statuses: list[str] = []

    mode = InteractiveMode(
        ui=FakeUi(),
        session=SimpleNamespace(
            scopedModels=[],
            modelRegistry=SimpleNamespace(refresh=lambda: None, getAvailable=lambda: list(all_models)),
            setScopedModels=lambda scoped: scoped_updates.append(scoped),
        ),
        settingsManager=SimpleNamespace(
            getEnabledModels=lambda: ["openai/gpt-4o-mini"],
            setEnabledModels=lambda value: persisted.append(value),
        ),
    )
    mode.showStatus = statuses.append  # type: ignore[method-assign]
    mode.updateAvailableProviderCount = lambda: None  # type: ignore[method-assign]

    await mode.showModelsSelector()

    config = captured["config"]
    callbacks = captured["callbacks"]
    assert config.enabledModelIds == ["openai/gpt-4o-mini"]

    callbacks.onChange(["anthropic/claude-sonnet-4-5"])
    await asyncio.sleep(0)
    callbacks.onPersist(["anthropic/claude-sonnet-4-5"])

    assert scoped_updates == [[{"model": all_models[1], "thinkingLevel": "high"}]]
    assert persisted == [["anthropic/claude-sonnet-4-5"]]
    assert statuses == ["Model selection saved to settings"]
