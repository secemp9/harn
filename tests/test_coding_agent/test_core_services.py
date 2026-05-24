from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from harnify_coding_agent.core.bash_executor import execute_bash_with_operations
from harnify_coding_agent.core.event_bus import create_event_bus
from harnify_coding_agent.core.exec import exec_command
from harnify_coding_agent.core.keybindings import KeybindingsManager, migrate_keybindings_config
from harnify_coding_agent.core.output_guard import (
    flush_raw_stdout,
    is_stdout_taken_over,
    restore_stdout,
    take_over_stdout,
    write_raw_stdout,
)
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


@pytest.mark.asyncio
async def test_event_bus_supports_sync_async_and_unsubscribe() -> None:
    bus = create_event_bus()
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
    bus = create_event_bus()

    def bad_handler(_data: Any) -> None:
        raise RuntimeError("boom")

    bus.on("demo", bad_handler)
    bus.emit("demo", {"value": 1})

    captured = capsys.readouterr()
    assert "Event handler error (demo): boom" in captured.err


@pytest.mark.asyncio
async def test_exec_command_supports_cwd_env_and_stdin(tmp_path: Path) -> None:
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
            "env": {"DEMO": "ok"},
            "input": "payload",
        },
    )

    assert result.code == 0
    assert result.killed is False
    assert result.stdout == f"ok\n{tmp_path.name}\npayload"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_exec_command_supports_timeout_and_abort(tmp_path: Path) -> None:
    timeout_result = await exec_command(
        sys.executable,
        ["-c", "import time; time.sleep(5)"],
        str(tmp_path),
        {"timeoutMs": 50},
    )
    assert timeout_result.killed is True

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
        take_over_stdout()
        assert is_stdout_taken_over() is True

        print("redirected", end="")
        write_raw_stdout("raw")
        await flush_raw_stdout()

        assert stderr.getvalue() == "redirected"
        assert stdout.getvalue() == "raw"

        restore_stdout()
        assert is_stdout_taken_over() is False
        print("after", end="")
        assert stdout.getvalue() == "rawafter"
    finally:
        restore_stdout()


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
    effective = keybindings.get_effective_config()
    assert effective["tui.select.confirm"] == "enter"
    assert effective["app.interrupt"] == "ctrl+x"


def test_migrate_keybindings_config_prefers_namespaced_entries() -> None:
    migrated = migrate_keybindings_config(
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
