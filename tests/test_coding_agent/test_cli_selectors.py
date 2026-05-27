from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import harnify_coding_agent.cli.config_selector as config_selector_module
import harnify_coding_agent.cli.session_picker as session_picker_module
from harnify_coding_agent.cli.config_selector import ConfigSelectorOptions, select_config
from harnify_coding_agent.cli.session_picker import select_session
from harnify_coding_agent.config import APP_NAME
from harnify_coding_agent.core.session_cwd import SessionCwdIssue
from harnify_coding_agent.core.package_manager import ConfiguredPackage, ResolvedPaths
from harnify_coding_agent.core.settings_manager import SettingsError, SettingsManager
from harnify_coding_agent.main import main, prompt_for_missing_session_cwd
import harnify_coding_agent.package_manager_cli as package_manager_cli_module
from harnify_coding_agent.package_manager_cli import handle_config_command, handle_package_command
from harnify_coding_agent.utils.version_check import LatestHarnifyRelease


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

    await config_selector_module._select_config(
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
    assert config_selector_module.__all__ == ["ConfigSelectorOptions", "selectConfig"]


@pytest.mark.asyncio
async def test_select_session_sets_keybindings_and_returns_selected_path() -> None:
    sentinel_keybindings = object()
    bound: list[object] = []

    selected = await session_picker_module._select_session(
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
    assert session_picker_module.__all__ == ["selectSession"]


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

    handled = await handle_config_command(["config"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert captured[0]["cwd"] == str(project)
    assert captured[0]["agentDir"] == str(agent_dir)
    assert isinstance(captured[0]["settingsManager"], SettingsManager)
    assert isinstance(captured[0]["resolvedPaths"], ResolvedPaths)


@pytest.mark.asyncio
async def test_handle_config_command_reports_settings_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    settings = SettingsManager.inMemory()
    settings.errors.append(SettingsError(scope="global", error=Exception("bad global settings")))
    stderr = io.StringIO()
    called: list[str] = []

    async def fake_select_config(_options: object) -> None:
        called.append("config")

    monkeypatch.setattr("sys.stderr", stderr)
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: settings)
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.select_config", fake_select_config)

    handled = await handle_config_command(["config"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert called == ["config"]
    assert "Warning (config command, global settings): bad global settings" in stderr.getvalue()


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
async def test_main_routes_package_command_boolean_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_package_command(_args: list[str]) -> bool:
        return True

    async def fail_config_command(_args: list[str]) -> int | None:
        raise AssertionError("config command should not run")

    monkeypatch.setattr("harnify_coding_agent.main.handle_package_command", fake_package_command)
    monkeypatch.setattr("harnify_coding_agent.main.handle_config_command", fail_config_command)
    monkeypatch.setattr("harnify_coding_agent.main._take_command_exit_code", lambda: 17)

    assert await main(["install", "demo"]) == 17


@pytest.mark.asyncio
async def test_handle_package_command_reports_invalid_option_for_command(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)

    handled = await handle_package_command(["install", "--self"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 1
    assert 'Unknown option --self for "install".' in stderr.getvalue()
    assert f'Use "{APP_NAME} --help" or "{APP_NAME} install <source> [-l]".' in stderr.getvalue()


@pytest.mark.asyncio
async def test_handle_package_command_help_uses_app_name_and_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)

    handled = await handle_package_command(["install", "--help"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert f"  {APP_NAME} install <source> [-l]" in stdout.getvalue()
    assert f"  {APP_NAME} install ./local/path" in stdout.getvalue()


@pytest.mark.asyncio
async def test_handle_package_command_supports_extension_update_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())

    calls: list[str | None] = []
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        async def update(self, source: str | None = None) -> None:
            calls.append(source)

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)

    handled = await handle_package_command(["update", "--extension", "npm:demo"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert calls == ["npm:demo"]
    assert stdout.getvalue() == "Updated npm:demo\n"
    assert stderr.getvalue() == ""


@pytest.mark.asyncio
async def test_handle_package_command_list_matches_ts_sections_and_filtered_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        def setProgressCallback(self, _callback) -> None:
            return None

        def listConfiguredPackages(self) -> list[ConfiguredPackage]:
            return [
                ConfiguredPackage(
                    source="npm:user-demo",
                    scope="user",
                    filtered=False,
                    installedPath="/tmp/user-demo",
                ),
                ConfiguredPackage(
                    source="npm:project-demo",
                    scope="project",
                    filtered=True,
                    installedPath=None,
                ),
            ]

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)

    handled = await handle_package_command(["list"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert stdout.getvalue() == (
        "User packages:\n"
        "  npm:user-demo\n"
        "    /tmp/user-demo\n"
        "\n"
        "Project packages:\n"
        "  npm:project-demo (filtered)\n"
    )
    assert stderr.getvalue() == ""


@pytest.mark.asyncio
async def test_handle_package_command_install_reports_settings_warning_progress_and_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    settings = SettingsManager.inMemory()
    settings.errors.append(SettingsError(scope="project", error=Exception("bad project settings")))
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: settings)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            self._progress = None

        def setProgressCallback(self, callback) -> None:
            self._progress = callback

        async def installAndPersist(self, source: str, options: dict[str, bool] | None = None) -> None:
            assert source == "npm:demo"
            assert options == {"local": True}
            assert self._progress is not None
            self._progress(SimpleNamespace(type="start", message="Installing npm:demo..."))

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)

    handled = await handle_package_command(["install", "npm:demo", "--local"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert stdout.getvalue() == "Installing npm:demo...\nInstalled npm:demo\n"
    assert "Warning (package command, project settings): bad project settings" in stderr.getvalue()


@pytest.mark.asyncio
async def test_handle_package_command_remove_reports_missing_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        def setProgressCallback(self, _callback) -> None:
            return None

        async def removeAndPersist(self, source: str, options: dict[str, bool] | None = None) -> bool:
            assert source == "npm:missing"
            assert options == {"local": False}
            return False

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)

    handled = await handle_package_command(["remove", "npm:missing"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "No matching package found for npm:missing\n"


@pytest.mark.asyncio
async def test_handle_package_command_update_self_reports_already_up_to_date(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_latest_harnify_release",
        lambda _version: _return_latest_release(LatestHarnifyRelease(version=package_manager_cli_module.VERSION)),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        def setProgressCallback(self, _callback) -> None:
            return None

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)

    handled = await handle_package_command(["update", "--self"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert stdout.getvalue() == f"{APP_NAME} is already up to date (v{package_manager_cli_module.VERSION})\n"
    assert stderr.getvalue() == ""


@pytest.mark.asyncio
async def test_handle_package_command_update_self_unavailable_sets_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_latest_harnify_release",
        lambda _version: _return_latest_release(None),
    )
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_self_update_command",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_self_update_unavailable_instruction",
        lambda *_args, **_kwargs: "Update it yourself manually.",
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        def setProgressCallback(self, _callback) -> None:
            return None

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)

    handled = await handle_package_command(["update", "--self"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 1
    assert stdout.getvalue() == ""
    assert f"error: {APP_NAME} cannot self-update this installation." in stderr.getvalue()
    assert "Update it yourself manually." in stderr.getvalue()


@pytest.mark.asyncio
async def test_handle_package_command_update_all_reports_extension_and_self_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_latest_harnify_release",
        lambda _version: _return_latest_release(None),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    update_calls: list[str | None] = []
    self_update_calls: list[tuple[str, str]] = []

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        def setProgressCallback(self, _callback) -> None:
            return None

        async def update(self, source: str | None = None) -> None:
            update_calls.append(source)

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_self_update_command",
        lambda *_args, **_kwargs: SimpleNamespace(command="uv", args=("tool", "upgrade"), display="uv tool upgrade harnify", steps=None),
    )

    async def fake_run_self_update(command) -> None:
        self_update_calls.append((command.command, command.display))

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli._run_self_update", fake_run_self_update)

    handled = await handle_package_command(["update"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 0
    assert update_calls == [None]
    assert self_update_calls == [("uv", "uv tool upgrade harnify")]
    assert stdout.getvalue() == "Updated packages\nUpdated harnify\n"
    assert stderr.getvalue() == ""


@pytest.mark.asyncio
async def test_handle_package_command_update_self_failure_prints_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.SettingsManager.create", lambda *_args: SettingsManager.inMemory())
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_latest_harnify_release",
        lambda _version: _return_latest_release(None),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    class FakePackageManager:
        def __init__(self, _options: object) -> None:
            return None

        def setProgressCallback(self, _callback) -> None:
            return None

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli.DefaultPackageManager", FakePackageManager)
    monkeypatch.setattr(
        "harnify_coding_agent.package_manager_cli.get_self_update_command",
        lambda *_args, **_kwargs: SimpleNamespace(command="uv", args=("tool", "upgrade"), display="uv tool upgrade harnify", steps=None),
    )

    async def fake_run_self_update(_command) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("harnify_coding_agent.package_manager_cli._run_self_update", fake_run_self_update)

    handled = await handle_package_command(["update", "--self"])

    assert handled is True
    assert package_manager_cli_module._take_command_exit_code() == 1
    assert stdout.getvalue() == ""
    assert "Error: boom" in stderr.getvalue()
    assert "If this keeps failing, run this command yourself: uv tool upgrade harnify" in stderr.getvalue()


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


async def _return_latest_release(value):
    return value
