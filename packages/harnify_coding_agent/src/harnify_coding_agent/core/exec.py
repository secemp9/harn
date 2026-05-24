"""Shared subprocess execution helpers for extensions and session services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from harnify_coding_agent.utils.paths import resolve_path
from harnify_coding_agent.utils.shell import kill_process_tree, track_detached_child_pid, untrack_detached_child_pid


class ExecOptions(TypedDict, total=False):
    signal: Any
    timeout: float
    timeoutMs: int
    cwd: str
    env: dict[str, str]
    input: str


@dataclass(slots=True)
class ExecResult:
    stdout: str
    stderr: str
    code: int | None
    killed: bool

    @property
    def exitCode(self) -> int | None:
        return self.code


def _is_aborted(signal: Any | None) -> bool:
    return bool(getattr(signal, "aborted", False))


async def _wait_for_abort(signal: Any) -> None:
    wait = getattr(signal, "wait", None)
    if callable(wait):
        result = wait()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            await result
            return
    while not _is_aborted(signal):
        await asyncio.sleep(0.01)


async def _read_stream(stream: asyncio.StreamReader | None, chunks: list[bytes]) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        chunks.append(chunk)


def _resolve_timeout_seconds(options: Mapping[str, Any]) -> float | None:
    timeout = options.get("timeout")
    if timeout is not None:
        return float(timeout)
    timeout_ms = options.get("timeoutMs")
    if timeout_ms is None:
        return None
    return float(timeout_ms) / 1000


async def exec_command(
    command: str,
    args: list[str],
    cwd: str,
    options: ExecOptions | Mapping[str, Any] | None = None,
) -> ExecResult:
    resolved_options = dict(options or {})
    resolved_cwd = resolve_path(str(resolved_options.get("cwd") or cwd))
    env = os.environ.copy()
    env.update(resolved_options.get("env") or {})
    input_text = resolved_options.get("input")
    timeout = _resolve_timeout_seconds(resolved_options)
    signal = resolved_options.get("signal")

    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        cwd=resolved_cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    if process.pid is not None:
        track_detached_child_pid(process.pid)

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_task = asyncio.create_task(_read_stream(process.stdout, stdout_chunks))
    stderr_task = asyncio.create_task(_read_stream(process.stderr, stderr_chunks))
    wait_task = asyncio.create_task(process.wait())
    abort_task = asyncio.create_task(_wait_for_abort(signal)) if signal is not None else None
    stdin_task = None
    killed = False

    try:
        if process.stdin is not None:
            process.stdin.write(input_text.encode("utf-8"))
            stdin_task = asyncio.create_task(process.stdin.drain())
            await stdin_task
            process.stdin.close()

        pending = [wait_task]
        if abort_task is not None:
            pending.append(abort_task)
        done, _ = await asyncio.wait(
            pending,
            timeout=timeout if timeout and timeout > 0 else None,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if wait_task in done:
            await wait_task
        else:
            killed = True
            if process.pid is not None:
                kill_process_tree(process.pid)
            else:
                process.kill()
            await wait_task

        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        return ExecResult(
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            code=process.returncode,
            killed=killed,
        )
    finally:
        if stdin_task is not None:
            await asyncio.gather(stdin_task, return_exceptions=True)
        if abort_task is not None:
            abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)
        if process.pid is not None:
            untrack_detached_child_pid(process.pid)


execCommand = exec_command

__all__ = [
    "ExecOptions",
    "ExecResult",
    "execCommand",
    "exec_command",
]
