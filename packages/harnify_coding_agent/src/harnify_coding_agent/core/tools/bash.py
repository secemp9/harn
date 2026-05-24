"""Bash tool for command execution with streaming, truncation, and timeouts."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import TextContent
from pydantic import BaseModel, ConfigDict, Field

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.output_accumulator import OutputAccumulator, OutputAccumulatorOptions
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition
from harnify_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from harnify_coding_agent.utils.shell import (
    get_shell_config,
    get_shell_env,
    kill_process_tree,
    track_detached_child_pid,
    untrack_detached_child_pid,
)

BASH_UPDATE_THROTTLE_SECONDS = 0.1


class BashToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(description="Bash command to execute")
    timeout: float | None = Field(default=None, description="Timeout in seconds (optional, no default timeout)")


@dataclass(slots=True)
class BashToolDetails:
    truncation: TruncationResult | None = None
    fullOutputPath: str | None = None


class BashExecOptions(TypedDict, total=False):
    onData: Callable[[bytes], None]
    signal: Any
    timeout: float
    env: dict[str, str]


class BashOperations(Protocol):
    async def exec(self, command: str, cwd: str, options: BashExecOptions) -> dict[str, int | None]: ...


@dataclass(slots=True)
class BashSpawnContext:
    command: str
    cwd: str
    env: dict[str, str]


type BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


@dataclass(slots=True)
class BashToolOptions:
    operations: BashOperations | None = None
    commandPrefix: str | None = None
    shellPath: str | None = None
    spawnHook: BashSpawnHook | None = None


@dataclass(slots=True)
class _LocalBashOperations:
    shellPath: str | None = None

    async def exec(self, command: str, cwd: str, options: BashExecOptions) -> dict[str, int | None]:
        if not os.path.exists(cwd):
            raise RuntimeError(f"Working directory does not exist: {cwd}\nCannot execute bash commands.")

        shell_config = get_shell_config(self.shellPath)
        env = options.get("env") or get_shell_env()
        signal = options.get("signal")
        timeout = options.get("timeout")
        on_data = options["onData"]

        process = await asyncio.create_subprocess_exec(
            shell_config.shell,
            *shell_config.args,
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        if process.pid is not None:
            track_detached_child_pid(process.pid)

        async def _read_stream(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                on_data(chunk)

        stdout_task = asyncio.create_task(_read_stream(process.stdout))
        stderr_task = asyncio.create_task(_read_stream(process.stderr))
        wait_task = asyncio.create_task(process.wait())
        abort_task = (
            asyncio.create_task(_wait_for_abort(signal))
            if signal is not None and hasattr(signal, "wait")
            else asyncio.create_task(_poll_abort(signal))
            if signal is not None
            else None
        )

        try:
            pending = [wait_task]
            if abort_task is not None:
                pending.append(abort_task)
            done, _ = await asyncio.wait(
                pending,
                timeout=timeout if timeout and timeout > 0 else None,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if wait_task not in done:
                if process.pid is not None:
                    kill_process_tree(process.pid)
                await wait_task
                if abort_task is not None and abort_task in done:
                    raise RuntimeError("aborted")
                raise RuntimeError(f"timeout:{timeout}")

            await wait_task
            if abort_task is not None and abort_task.done() and _is_aborted(signal):
                raise RuntimeError("aborted")
            return {"exitCode": process.returncode}
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            if abort_task is not None:
                abort_task.cancel()
                await asyncio.gather(abort_task, return_exceptions=True)
            if process.pid is not None:
                untrack_detached_child_pid(process.pid)


def create_local_bash_operations(options: Mapping[str, Any] | None = None) -> BashOperations:
    shell_path = options.get("shellPath") if options else None
    return _LocalBashOperations(shellPath=shell_path)


def _coerce_options(options: BashToolOptions | Mapping[str, Any] | None) -> BashToolOptions:
    if options is None:
        return BashToolOptions()
    if isinstance(options, BashToolOptions):
        return options
    return BashToolOptions(
        operations=options.get("operations"),
        commandPrefix=options.get("commandPrefix"),
        shellPath=options.get("shellPath"),
        spawnHook=options.get("spawnHook"),
    )


def _prepare_arguments(args: Any) -> dict[str, Any]:
    return BashToolInput.model_validate(args).model_dump(exclude_none=True)


def _resolve_spawn_context(command: str, cwd: str, spawn_hook: BashSpawnHook | None = None) -> BashSpawnContext:
    context = BashSpawnContext(command=command, cwd=cwd, env={**get_shell_env()})
    return spawn_hook(context) if spawn_hook else context


def _is_aborted(signal: Any | None) -> bool:
    return bool(getattr(signal, "aborted", False))


async def _wait_for_abort(signal: Any) -> None:
    wait = getattr(signal, "wait", None)
    if callable(wait):
        await wait()
        return
    await _poll_abort(signal)


async def _poll_abort(signal: Any) -> None:
    while not _is_aborted(signal):
        await asyncio.sleep(0.01)


def _make_text_result(text: str, details: BashToolDetails | None = None) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)], details=details)


def _append_status(text: str, status: str) -> str:
    return f"{text}\n\n{status}" if text else status


def create_bash_tool_definition(
    cwd: str,
    options: BashToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], BashToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or create_local_bash_operations({"shellPath": resolved_options.shellPath})

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
        _ctx: Any = None,
    ) -> AgentToolResult:
        parsed = BashToolInput.model_validate(params)
        resolved_command = (
            f"{resolved_options.commandPrefix}\n{parsed.command}" if resolved_options.commandPrefix else parsed.command
        )
        spawn_context = _resolve_spawn_context(resolved_command, cwd, resolved_options.spawnHook)
        output = OutputAccumulator(OutputAccumulatorOptions(tempFilePrefix="pi-bash"))
        loop = asyncio.get_running_loop()
        update_handle: asyncio.TimerHandle | None = None
        update_dirty = False
        last_update_at = 0.0

        def emit_output_update() -> None:
            nonlocal update_dirty, last_update_at, update_handle
            if on_update is None or not update_dirty:
                return
            update_dirty = False
            last_update_at = loop.time()
            update_handle = None
            snapshot = output.snapshot(persistIfTruncated=True)
            on_update(
                AgentToolResult(
                    content=[TextContent(text=snapshot.content or "")],
                    details=BashToolDetails(
                        truncation=snapshot.truncation if snapshot.truncation.truncated else None,
                        fullOutputPath=snapshot.fullOutputPath,
                    ),
                )
            )

        def clear_update_handle() -> None:
            nonlocal update_handle
            if update_handle is not None:
                update_handle.cancel()
                update_handle = None

        def schedule_output_update() -> None:
            nonlocal update_dirty, update_handle
            if on_update is None:
                return
            update_dirty = True
            delay = BASH_UPDATE_THROTTLE_SECONDS - (loop.time() - last_update_at)
            if delay <= 0:
                clear_update_handle()
                emit_output_update()
                return
            if update_handle is None:
                update_handle = loop.call_later(delay, emit_output_update)

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        def handle_data(data: bytes) -> None:
            output.append(data)
            schedule_output_update()

        async def finish_output() -> tuple[str, BashToolDetails | None]:
            output.finish()
            clear_update_handle()
            emit_output_update()
            snapshot = output.snapshot(persistIfTruncated=True)
            await output.close_temp_file()

            truncation = snapshot.truncation
            text = snapshot.content or "(no output)"
            details: BashToolDetails | None = None
            if truncation.truncated:
                details = BashToolDetails(truncation=truncation, fullOutputPath=snapshot.fullOutputPath)
                start_line = truncation.totalLines - truncation.outputLines + 1
                end_line = truncation.totalLines
                if truncation.lastLinePartial:
                    last_line_size = format_size(output.get_last_line_bytes())
                    text += (
                        f"\n\n[Showing last {format_size(truncation.outputBytes)} of line {end_line} "
                        f"(line is {last_line_size}). Full output: {snapshot.fullOutputPath}]"
                    )
                elif truncation.truncatedBy == "lines":
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {truncation.totalLines}. "
                        f"Full output: {snapshot.fullOutputPath}]"
                    )
                else:
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {truncation.totalLines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {snapshot.fullOutputPath}]"
                    )
            return text, details

        try:
            try:
                result = await operations.exec(
                    spawn_context.command,
                    spawn_context.cwd,
                    {
                        "onData": handle_data,
                        "signal": signal,
                        "timeout": parsed.timeout,
                        "env": spawn_context.env,
                    },
                )
                exit_code = result.get("exitCode")
            except Exception as error:
                output_text, _details = await finish_output()
                message = str(error)
                if message == "aborted":
                    status_base = output_text if output_text != "(no output)" else ""
                    raise RuntimeError(_append_status(status_base, "Command aborted")) from None
                if message.startswith("timeout:"):
                    timeout_secs = message.split(":", 1)[1]
                    raise RuntimeError(
                        _append_status(
                            output_text if output_text != "(no output)" else "",
                            f"Command timed out after {timeout_secs} seconds",
                        )
                    ) from None
                raise

            output_text, details = await finish_output()
            if exit_code not in {0, None}:
                raise RuntimeError(_append_status(output_text, f"Command exited with code {exit_code}"))
            return _make_text_result(output_text, details)
        finally:
            clear_update_handle()

    return ToolDefinition(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES / 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
        ),
        parameters=BashToolInput,
        prepareArguments=_prepare_arguments,
        execute=execute,
    )


def create_bash_tool(cwd: str, options: BashToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_bash_tool_definition(cwd, options))


createBashTool = create_bash_tool
createBashToolDefinition = create_bash_tool_definition
createLocalBashOperations = create_local_bash_operations

__all__ = [
    "BashExecOptions",
    "BashOperations",
    "BashSpawnContext",
    "BashToolDetails",
    "BashToolInput",
    "BashToolOptions",
    "createBashTool",
    "createBashToolDefinition",
    "createLocalBashOperations",
    "create_bash_tool",
    "create_bash_tool_definition",
    "create_local_bash_operations",
]
