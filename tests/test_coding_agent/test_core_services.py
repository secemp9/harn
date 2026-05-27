from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from harnify_coding_agent.core.bash_executor import execute_bash_with_operations
from harnify_coding_agent.core.event_bus import createEventBus
from harnify_coding_agent.core.exec import exec_command
from harnify_coding_agent.core.keybindings import KeybindingsManager, migrateKeybindingsConfig
from harnify_coding_agent.core.output_guard import (
    flushRawStdout,
    isStdoutTakenOver,
    restoreStdout,
    takeOverStdout,
    writeRawStdout,
)
from harnify_coding_agent.core import source_info as source_info_module
from harnify_coding_agent.core import slash_commands as slash_commands_module
from harnify_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES


class _AbortSignal:
    def __init__(self) -> None:
        self.aborted = False
        self._event = asyncio.Event()

    def abort(self) -> None:
        self.aborted = True
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


class _FakeBashOperations:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        exit_code: int | None = 0,
        wait_for_abort: bool = False,
    ) -> None:
        self._chunks = chunks
        self._exit_code = exit_code
        self._wait_for_abort = wait_for_abort

    async def exec(self, command: str, cwd: str, options: dict[str, Any]) -> dict[str, int | None]:
        assert command
        assert cwd
        for chunk in self._chunks:
            options["onData"](chunk)
        if self._wait_for_abort:
            signal = options["signal"]
            await signal.wait()
            raise RuntimeError("aborted")
        return {"exitCode": self._exit_code}


def test_slash_commands_surface_matches_ts() -> None:
    assert slash_commands_module.__all__ == [
        "SlashCommandSource",
        "SlashCommandInfo",
        "BuiltinSlashCommand",
        "BUILTIN_SLASH_COMMANDS",
    ]
    assert slash_commands_module.BUILTIN_SLASH_COMMANDS[2].description == "Enable/disable models for Ctrl+P cycling"
    assert "_LOCAL_ALIAS_SLASH_COMMANDS" in vars(slash_commands_module)


def test_source_info_surface_and_copy_semantics_match_ts() -> None:
    assert source_info_module.__all__ == [
        "SourceScope",
        "SourceOrigin",
        "SourceInfo",
        "createSourceInfo",
        "createSyntheticSourceInfo",
    ]

    info = source_info_module.createSourceInfo(
        "/tmp/demo",
        {
            "source": "pkg:test",
            "scope": "project",
            "origin": "package",
            "baseDir": "/tmp",
        },
    )
    assert info.source == "pkg:test"
    assert info.scope == "project"
    assert info.origin == "package"
    assert info.baseDir == "/tmp"

    synthetic = source_info_module.createSyntheticSourceInfo("/tmp/demo", {"source": "local"})
    assert synthetic.scope == "temporary"
    assert synthetic.origin == "top-level"


def test_install_telemetry_surface_and_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from harnify_coding_agent.core import telemetry as telemetry_module

    class _Settings:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled

        def getEnableInstallTelemetry(self) -> bool:
            return self.enabled

    assert telemetry_module.__all__ == ["isInstallTelemetryEnabled"]

    monkeypatch.setenv("HARNIFY_TELEMETRY", "yes")
    assert telemetry_module.isInstallTelemetryEnabled(_Settings(False)) is True
    assert telemetry_module.isInstallTelemetryEnabled(_Settings(True), "0") is False
    assert telemetry_module.isInstallTelemetryEnabled(_Settings(True), None) is True


def test_timings_surface_matches_ts() -> None:
    from harnify_coding_agent.core import timings as timings_module

    assert timings_module.__all__ == ["resetTimings", "time", "printTimings"]


@pytest.mark.asyncio
async def test_event_bus_supports_sync_async_and_unsubscribe() -> None:
    bus = createEventBus()
    seen: list[tuple[str, int]] = []
    async_seen = asyncio.Event()

    def sync_handler(data: Any) -> None:
        seen.append(("sync", data["value"]))

    async def async_handler(data: Any) -> None:
        seen.append(("async", data["value"]))
        async_seen.set()

    unsubscribe_sync = bus.on("demo", sync_handler)
    bus.on("demo", async_handler)

    bus.emit("demo", {"value": 1})
    await asyncio.wait_for(async_seen.wait(), timeout=1)
    assert seen == [("sync", 1), ("async", 1)]

    unsubscribe_sync()
    async_seen.clear()
    bus.emit("demo", {"value": 2})
    await asyncio.wait_for(async_seen.wait(), timeout=1)
    assert seen == [("sync", 1), ("async", 1), ("async", 2)]

    bus.clear()
    async_seen.clear()
    bus.emit("demo", {"value": 3})
    await asyncio.sleep(0.05)
    assert seen == [("sync", 1), ("async", 1), ("async", 2)]


def test_event_bus_reports_handler_errors(capsys: pytest.CaptureFixture[str]) -> None:
    bus = createEventBus()

    def bad_handler(_data: Any) -> None:
        raise RuntimeError("boom")

    bus.on("demo", bad_handler)
    bus.emit("demo", {"value": 1})

    captured = capsys.readouterr()
    assert "Event handler error (demo): boom" in captured.err


def test_event_bus_async_handler_does_not_block_without_running_loop() -> None:
    bus = createEventBus()
    finished = threading.Event()

    async def async_handler(_data: Any) -> None:
        await asyncio.sleep(0.05)
        finished.set()

    bus.on("demo", async_handler)

    started = time.perf_counter()
    bus.emit("demo", {"value": 1})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.04
    assert finished.wait(timeout=1)


def test_event_bus_module_exports_match_ts_surface() -> None:
    from harnify_coding_agent.core import event_bus

    assert event_bus.__all__ == [
        "EventBus",
        "EventBusController",
        "createEventBus",
    ]


def test_core_package_exports_match_ts_surface() -> None:
    import harnify_coding_agent.core as core_package

    assert core_package.__all__ == [
        "AgentEndEvent",
        "AgentSession",
        "AgentSessionConfig",
        "AgentSessionEvent",
        "AgentSessionEventListener",
        "AgentSessionRuntime",
        "AgentSessionRuntimeDiagnostic",
        "AgentSessionServices",
        "AgentStartEvent",
        "AgentToolResult",
        "AgentToolUpdateCallback",
        "BashExecutorOptions",
        "BashResult",
        "BeforeAgentStartEvent",
        "BeforeAgentStartEventResult",
        "BuildSystemPromptOptions",
        "CompactionResult",
        "ContextEvent",
        "CreateAgentSessionFromServicesOptions",
        "CreateAgentSessionRuntimeFactory",
        "CreateAgentSessionRuntimeResult",
        "CreateAgentSessionServicesOptions",
        "EventBus",
        "EventBusController",
        "ExecOptions",
        "ExecResult",
        "Extension",
        "ExtensionAPI",
        "ExtensionCommandContext",
        "ExtensionContext",
        "ExtensionError",
        "ExtensionEvent",
        "ExtensionFactory",
        "ExtensionFlag",
        "ExtensionHandler",
        "ExtensionRunner",
        "ExtensionShortcut",
        "ExtensionUIContext",
        "LoadExtensionsResult",
        "MessageRenderer",
        "ModelCycleResult",
        "PromptOptions",
        "RegisteredCommand",
        "SessionBeforeCompactEvent",
        "SessionBeforeForkEvent",
        "SessionBeforeSwitchEvent",
        "SessionBeforeTreeEvent",
        "SessionCompactEvent",
        "SessionShutdownEvent",
        "SessionStartEvent",
        "SessionStats",
        "SessionTreeEvent",
        "ToolCallEvent",
        "ToolCallEventResult",
        "ToolDefinition",
        "ToolRenderResultOptions",
        "ToolResultEvent",
        "TurnEndEvent",
        "TurnStartEvent",
        "WorkingIndicatorOptions",
        "createAgentSessionFromServices",
        "createAgentSessionRuntime",
        "createAgentSessionServices",
        "createEventBus",
        "createSyntheticSourceInfo",
        "defineTool",
        "discoverAndLoadExtensions",
        "executeBashWithOperations",
    ]


@pytest.mark.asyncio
async def test_exec_command_uses_argument_cwd_and_ignores_python_only_options(tmp_path: Path) -> None:
    ignored_cwd = tmp_path / "ignored"
    ignored_cwd.mkdir()
    result = await exec_command(
        sys.executable,
        [
            "-c",
            (
                "import os, pathlib, sys; "
                "print(os.getenv('DEMO')); "
                "print(pathlib.Path.cwd().name); "
                "print(sys.stdin.read(), end='')"
            ),
        ],
        str(tmp_path),
        {
            "cwd": str(ignored_cwd),
            "env": {"DEMO": "ok"},
            "input": "payload",
        },
    )

    assert result.code == 0
    assert result.killed is False
    assert result.stdout == f"None\n{tmp_path.name}\n"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_exec_command_uses_millisecond_timeout_and_abort(tmp_path: Path) -> None:
    started = asyncio.get_running_loop().time()
    timeout_result = await exec_command(
        sys.executable,
        ["-c", "import time; time.sleep(1)"],
        str(tmp_path),
        {"timeout": 50},
    )
    elapsed = asyncio.get_running_loop().time() - started
    assert timeout_result.killed is True
    assert timeout_result.code == 0
    assert elapsed < 0.5

    signal = _AbortSignal()
    task = asyncio.create_task(
        exec_command(
            sys.executable,
            ["-c", "import time; time.sleep(5)"],
            str(tmp_path),
            {"signal": signal},
        )
    )
    await asyncio.sleep(0.05)
    signal.abort()
    aborted_result = await task
    assert aborted_result.killed is True
    assert aborted_result.code == 0


@pytest.mark.asyncio
async def test_exec_command_returns_code_1_for_spawn_errors(tmp_path: Path) -> None:
    result = await exec_command("__missing_exec_command__", [], str(tmp_path))

    assert result == type(result)(stdout="", stderr="", code=1, killed=False)


def test_exec_module_exports_match_ts_surface() -> None:
    import importlib

    exec_module = importlib.import_module("harnify_coding_agent.core.exec")
    assert exec_module.__all__ == [
        "ExecOptions",
        "ExecResult",
        "execCommand",
    ]


@pytest.mark.asyncio
async def test_bash_executor_sanitizes_and_streams_output(tmp_path: Path) -> None:
    streamed: list[str] = []
    result = await execute_bash_with_operations(
        "echo demo",
        str(tmp_path),
        _FakeBashOperations([b"\x1b[31mhello\x1b[0m\r\n", b"world\n"]),
        {"onChunk": streamed.append},
    )

    assert result.output == "hello\nworld\n"
    assert result.exitCode == 0
    assert result.cancelled is False
    assert streamed == ["hello\n", "world\n"]


@pytest.mark.asyncio
async def test_bash_executor_truncates_and_preserves_cancelled_output(tmp_path: Path) -> None:
    large_text = ("x" * (DEFAULT_MAX_BYTES + 256)).encode("utf-8")
    truncated = await execute_bash_with_operations(
        "echo big",
        str(tmp_path),
        _FakeBashOperations([large_text]),
    )
    assert truncated.truncated is True
    assert truncated.fullOutputPath is not None
    assert os.path.exists(truncated.fullOutputPath)

    signal = _AbortSignal()
    task = asyncio.create_task(
        execute_bash_with_operations(
            "sleep",
            str(tmp_path),
            _FakeBashOperations([b"partial\n"], wait_for_abort=True),
            {"signal": signal},
        )
    )
    await asyncio.sleep(0.05)
    signal.abort()
    cancelled = await task
    assert cancelled.cancelled is True
    assert cancelled.exitCode is None
    assert cancelled.output == "partial\n"


@pytest.mark.asyncio
async def test_output_guard_redirects_stdout_but_preserves_raw_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    try:
        takeOverStdout()
        assert isStdoutTakenOver() is True
        assert sys.stdout is stdout

        print("redirected", end="")
        writeRawStdout("raw")
        await flushRawStdout()

        assert stderr.getvalue() == "redirected"
        assert stdout.getvalue() == "raw"
        assert callable(stdout.write)

        restoreStdout()
        assert isStdoutTakenOver() is False
        print("after", end="")
        assert stdout.getvalue() == "rawafter"
    finally:
        restoreStdout()


def test_output_guard_exports_match_ts_surface() -> None:
    from harnify_coding_agent.core import output_guard

    assert output_guard.__all__ == [
        "flushRawStdout",
        "isStdoutTakenOver",
        "restoreStdout",
        "takeOverStdout",
        "writeRawStdout",
    ]
    assert not hasattr(output_guard, "flush_raw_stdout")
    assert not hasattr(output_guard, "is_stdout_taken_over")
    assert not hasattr(output_guard, "restore_stdout")
    assert not hasattr(output_guard, "take_over_stdout")
    assert not hasattr(output_guard, "write_raw_stdout")


def test_keybindings_manager_loads_migrated_config(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "keybindings.json").write_text(
        json.dumps(
            {
                "selectConfirm": "enter",
                "interrupt": "ctrl+x",
            }
        ),
        encoding="utf-8",
    )

    keybindings = KeybindingsManager.create(str(agent_dir))

    assert keybindings.getUserBindings() == {
        "tui.select.confirm": "enter",
        "app.interrupt": "ctrl+x",
    }
    effective = keybindings.getEffectiveConfig()
    assert effective["tui.select.confirm"] == "enter"
    assert effective["app.interrupt"] == "ctrl+x"


def test_migrate_keybindings_config_prefers_namespaced_entries() -> None:
    migrated = migrateKeybindingsConfig(
        {
            "expandTools": "ctrl+x",
            "app.tools.expand": "ctrl+y",
        }
    )

    assert migrated == {
        "config": {
            "app.tools.expand": "ctrl+y",
        },
        "migrated": True,
    }


def test_keybindings_module_exports_match_ts_surface() -> None:
    from harnify_coding_agent.core import keybindings

    assert keybindings.__all__ == [
        "AppKeybinding",
        "AppKeybindings",
        "KEYBINDINGS",
        "Keybinding",
        "KeyId",
        "KeybindingsConfig",
        "KeybindingsManager",
        "migrateKeybindingsConfig",
    ]
