"""Print mode for single-shot non-interactive execution."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from harnify_ai.types import ImageContent

from harnify_coding_agent.core.output_guard import (
    flushRawStdout,
    restoreStdout,
    takeOverStdout,
    writeRawStdout,
)
from harnify_coding_agent.modes.rpc.jsonl import serialize_json_line


@dataclass(slots=True)
class PrintModeOptions:
    mode: str = "text"
    messages: list[str] = field(default_factory=list)
    initialMessage: str | None = None
    initialImages: list[ImageContent] | None = None


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _message_role(message: Any) -> str | None:
    role = _value(message, "role")
    return role if isinstance(role, str) else None


def _message_content(message: Any) -> Any:
    return _value(message, "content")


def _content_type(block: Any) -> str | None:
    block_type = _value(block, "type")
    return block_type if isinstance(block_type, str) else None


async def run_print_mode(runtime_host: Any, options: PrintModeOptions | dict[str, Any]) -> int:
    resolved = options if isinstance(options, PrintModeOptions) else PrintModeOptions(**dict(options or {}))
    session = runtime_host.session
    unsubscribe = None
    disposed = False
    takeOverStdout()

    async def dispose_runtime() -> None:
        nonlocal disposed, unsubscribe
        if disposed:
            return
        disposed = True
        if callable(unsubscribe):
            unsubscribe()
            unsubscribe = None
        await runtime_host.dispose()

    async def rebind_session() -> None:
        nonlocal session, unsubscribe
        session = runtime_host.session
        await session.bindExtensions(
            {
                "commandContextActions": {
                    "waitForIdle": lambda: session.agent.waitForIdle(),
                    "newSession": lambda new_session_options=None: runtime_host.newSession(new_session_options),
                    "fork": lambda entry_id, fork_options=None: runtime_host.fork(entry_id, fork_options),
                    "navigateTree": lambda _target_id, _options=None: {"cancelled": False},
                    "switchSession": lambda session_path, switch_options=None: runtime_host.switchSession(
                        session_path,
                        switch_options,
                    ),
                    "reload": lambda: None,
                },
                "onError": lambda error: print(
                    f"Extension error ({_value(error, 'extensionPath', '<unknown>')}): "
                    f"{_value(error, 'error', error)}",
                    file=sys.stderr,
                ),
            }
        )
        if callable(unsubscribe):
            unsubscribe()
        if resolved.mode == "json":
            unsubscribe = session.subscribe(lambda event: writeRawStdout(serialize_json_line(event)))

    try:
        if resolved.mode == "json":
            header = session.sessionManager.getHeader()
            if header:
                writeRawStdout(serialize_json_line(header))

        if hasattr(runtime_host, "setRebindSession"):
            runtime_host.setRebindSession(rebind_session)
        await rebind_session()

        if resolved.initialMessage:
            await session.prompt(
                resolved.initialMessage,
                {
                    "images": list(resolved.initialImages or []),
                },
            )

        for message in resolved.messages:
            await session.prompt(message)

        if resolved.mode == "text":
            last_message = session.state.messages[-1] if session.state.messages else None
            if _message_role(last_message) == "assistant":
                stop_reason = _value(last_message, "stopReason")
                if stop_reason in {"error", "aborted"}:
                    print(_value(last_message, "errorMessage") or f"Request {stop_reason}", file=sys.stderr)
                    return 1
                for content in _message_content(last_message) or []:
                    if _content_type(content) == "text":
                        writeRawStdout(f"{_value(content, 'text', '')}\n")

        return 0
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 1
    finally:
        await dispose_runtime()
        await flushRawStdout()
        restoreStdout()

runPrintMode = run_print_mode

__all__ = ["PrintModeOptions", "runPrintMode", "run_print_mode"]
