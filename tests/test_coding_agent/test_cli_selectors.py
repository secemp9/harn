from __future__ import annotations

from pathlib import Path

import pytest
from harnify_coding_agent.cli.config_selector import ConfigSelectorOptions, select_config
from harnify_coding_agent.cli.session_picker import select_session
from harnify_coding_agent.config import APP_NAME
from harnify_coding_agent.core.session_cwd import SessionCwdIssue
from harnify_coding_agent.core.package_manager import ResolvedPaths
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.main import main, prompt_for_missing_session_cwd
from harnify_coding_agent.package_manager_cli import handle_config_command


class _FakeTerminal:
    rows = 29


class _FakeTui:
    def __init__(self, terminal: object) -> None:
        self.terminal = terminal
        self.child: object | None = None
        self.focus: object | None = None
        self.started = False
        self.stopped = False
        self.renderRequested = 0

    def addChild(self, child: object) -> None:
        self.child = child

    def setFocus(self, focus: object) -> None:
        self.focus = focus

    def requestRender(self) -> None:
        self.renderRequested += 1

    def start(self) -> None:
        self.started = True
        assert self.child is not None
        if hasattr(self.child, "onClose"):
            self.child.onClose()  # type: ignore[attr-defined]
        elif hasattr(self.child, "onSelect"):
            self.child.onSelect("/tmp/resume.jsonl")  # type: ignore[attr-defined]

    def stop(self) -> None:
        self.stopped = True


class _FakeConfigSelectorComponent:
    def __init__(
        self,
        resolvedPaths: ResolvedPaths,
        settingsManager: SettingsManager,
        cwd: str,
        agentDir: str,
        onClose,
        onExit,
        requestRender,
        terminalHeight: int | None,
    ) -> None:
        self.resolvedPaths = resolvedPaths
        self.settingsManager = settingsManager
        self.cwd = cwd
        self.agentDir = agentDir
        self.onClose = onClose
        self.onExit = onExit
        self.requestRender = requestRender
        self.terminalHeight = terminalHeight
        self.resourceList = object()

    def getResourceList(self) -> object:
        return self.resourceList


class _FakeSessionSelectorComponent:
    def __init__(
        self,
        currentSessionsLoader,
        allSessionsLoader,
        onSelect,
        onCancel,
        onExit,
        requestRender,
        options,
        currentSessionFilePath=None,
    ) -> None:
        self.currentSessionsLoader = currentSessionsLoader
        self.allSessionsLoader = allSessionsLoader
        self.onSelect = onSelect
        self.onCancel = onCancel
        self.onExit = onExit
        self.requestRender = requestRender
        self.options = options
        self.currentSessionFilePath = currentSessionFilePath
        self.sessionList = object()

    def getSessionList(self) -> object:
        return self.sessionList


class _FakePromptTui(_FakeTui):
    def start(self) -> None:
        self.started = True
        assert self.child is not None
        self.child.onSelect("Continue")  # type: ignore[attr-defined]


class _FakePromptSelectorComponent:
    def __init__(self, title, options, onSelect, onCancel, opts=None) -> None:
        self.title = title
        self.options = options
        self.onSelect = onSelect
        self.onCancel = onCancel
        self.opts = opts or {}


@pytest.mark.asyncio
async def test_select_config_initializes_theme_and_mounts_component() -> None:
    settings = SettingsManager.inMemory({"theme": "light"})
    init_calls: list[tuple[str | None, bool]] = []
    stop_calls: list[str] = []

    await select_config(
        ConfigSelectorOptions(
            resolvedPaths=ResolvedPaths(),
            settingsManager=settings,
            cwd="/tmp/project",
            agentDir="/tmp/agent",
        ),
        terminalFactory=_FakeTerminal,
        uiFactory=_FakeTui,
        componentFactory=_FakeConfigSelectorComponent,
        initTheme=lambda theme_name, enable_watcher: init_calls.append((theme_name, enable_watcher)),
        stopThemeWatcher=lambda: stop_calls.append("stopped"),
    )

    assert init_calls == [("light", True)]
    assert stop_calls == ["stopped"]


@pytest.mark.asyncio
async def test_select_session_sets_keybindings_and_returns_selected_path() -> None:
    sentinel_keybindings = object()
    bound: list[object] = []

    selected = await select_session(
        lambda _progress=None: _async_sessions([]),
        lambda _progress=None: _async_sessions([]),
        terminalFactory=_FakeTerminal,
        uiFactory=_FakeTui,
        componentFactory=_FakeSessionSelectorComponent,
        keybindingsFactory=lambda: sentinel_keybindings,  # type: ignore[return-value]
        setKeybindingsFn=lambda keybindings: bound.append(keybindings),
    )

    assert selected == "/tmp/resume.jsonl"
    assert bound == [sentinel_keybindings]


@pytest.mark.asyncio
async def test_handle_config_command_uses_real_package_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    captured: list[dict[str, object]] = []

    async def fake_select_config(options: object) -> None:
        if isinstance(options, ConfigSelectorOptions):
            captured.append(
                {
                    "cwd": options.cwd,
                    "agentDir": options.agentDir,
                    "settingsManager": options.settingsManager,
                    "resolvedPaths": options.resolvedPaths,
                }
            )
            return
        assert isinstance(options, dict)
        captured.append(dict(options))

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.select_config", fake_select_config)

    code = await handle_config_command(["config"])

    assert code == 0
    assert captured[0]["cwd"] == str(project)
    assert captured[0]["agentDir"] == str(agent_dir)
    assert isinstance(captured[0]["settingsManager"], SettingsManager)
    assert isinstance(captured[0]["resolvedPaths"], ResolvedPaths)


@pytest.mark.asyncio
async def test_main_routes_config_command_before_normal_arg_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    called: list[str] = []

    async def fake_select_config(_options: object) -> None:
        called.append("config")

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.select_config", fake_select_config)

    code = await main(["config"])

    assert code == 0
    assert called == ["config"]


@pytest.mark.asyncio
async def test_main_routes_package_command_before_normal_arg_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_package_command(_args: list[str]) -> int | None:
        return 23

    async def fail_config_command(_args: list[str]) -> int | None:
        raise AssertionError("config command should not run")

    monkeypatch.setattr("harnify_coding_agent.main.handle_package_command", fake_package_command)
    monkeypatch.setattr("harnify_coding_agent.main.handle_config_command", fail_config_command)

    assert await main(["install", "demo"]) == 23


@pytest.mark.asyncio
async def test_prompt_for_missing_session_cwd_returns_fallback_and_sets_keybindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SettingsManager.inMemory({"theme": "light"})
    init_calls: list[tuple[str | None, bool]] = []
    bound_keybindings: list[object] = []
    sentinel_keybindings = object()

    monkeypatch.setattr("harnify_coding_agent.main.init_theme", lambda name, enable=False: init_calls.append((name, enable)))

    result = await prompt_for_missing_session_cwd(
        SessionCwdIssue(sessionCwd="/missing/project", fallbackCwd="/tmp/project", sessionFile="/tmp/session.jsonl"),
        settings,
        terminal_factory=_FakeTerminal,
        ui_factory=_FakePromptTui,
        component_factory=_FakePromptSelectorComponent,
        keybindings_factory=lambda: sentinel_keybindings,  # type: ignore[return-value]
        set_keybindings_fn=lambda keybindings: bound_keybindings.append(keybindings),
    )

    assert result == "/tmp/project"
    assert init_calls == [("light", False)]
    assert bound_keybindings == [sentinel_keybindings]


async def _async_sessions(value: list[object]) -> list[object]:
    return value
