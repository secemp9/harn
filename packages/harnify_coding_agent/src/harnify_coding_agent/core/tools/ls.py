"""Directory listing tool."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import TextContent
from pydantic import BaseModel, ConfigDict, Field

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.path_utils import resolve_to_cwd
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition
from harnify_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    TruncationOptions,
    TruncationResult,
    format_size,
    truncate_head,
)

DEFAULT_LIMIT = 500


class LsToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str | None = Field(default=None, description="Directory to list (default: current directory)")
    limit: int | None = Field(default=None, description="Maximum number of entries to return (default: 500)")


@dataclass(slots=True)
class LsToolDetails:
    truncation: TruncationResult | None = None
    entryLimitReached: int | None = None


class LsOperations(Protocol):
    exists: Callable[[str], Awaitable[bool] | bool]
    stat: Callable[[str], Awaitable[os.stat_result | Any] | os.stat_result | Any]
    readdir: Callable[[str], Awaitable[list[str]] | list[str]]


@dataclass(slots=True)
class LsToolOptions:
    operations: LsOperations | None = None


@dataclass(slots=True)
class _DefaultLsOperations:
    def exists(self, absolute_path: str) -> bool:
        return os.path.exists(absolute_path)

    def stat(self, absolute_path: str) -> os.stat_result:
        return os.stat(absolute_path)

    def readdir(self, absolute_path: str) -> list[str]:
        return os.listdir(absolute_path)


def _coerce_options(options: LsToolOptions | Mapping[str, Any] | None) -> LsToolOptions:
    if options is None:
        return LsToolOptions()
    if isinstance(options, LsToolOptions):
        return options
    return LsToolOptions(operations=options.get("operations"))


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if asyncio.isfuture(value) or hasattr(value, "__await__"):
        return await value
    return value


def _is_directory(stat_result: Any) -> bool:
    if hasattr(stat_result, "is_dir"):
        return bool(stat_result.is_dir())
    if hasattr(stat_result, "isDirectory"):
        return bool(stat_result.isDirectory())
    mode = getattr(stat_result, "st_mode", None)
    if isinstance(mode, int):
        return stat.S_ISDIR(mode)
    raise TypeError("Unsupported stat result")


def _details_or_none(details: LsToolDetails) -> LsToolDetails | None:
    if any(getattr(details, field.name) is not None for field in fields(details)):
        return details
    return None


def create_ls_tool_definition(
    cwd: str,
    options: LsToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], LsToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultLsOperations()

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        _ctx: Any = None,
    ) -> AgentToolResult:
        parsed = LsToolInput.model_validate(params)
        dir_path = resolve_to_cwd(parsed.path or ".", cwd)
        effective_limit = parsed.limit or DEFAULT_LIMIT

        if getattr(signal, "aborted", False):
            raise RuntimeError("Operation aborted")
        if not await _maybe_await(operations.exists(dir_path)):
            raise RuntimeError(f"Path not found: {dir_path}")

        stat_result = await _maybe_await(operations.stat(dir_path))
        if not _is_directory(stat_result):
            raise RuntimeError(f"Not a directory: {dir_path}")

        try:
            entries = await _maybe_await(operations.readdir(dir_path))
        except Exception as error:
            raise RuntimeError(f"Cannot read directory: {error}") from error

        entries = sorted(entries, key=lambda item: item.lower())
        results: list[str] = []
        entry_limit_reached = False
        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break
            full_path = os.path.join(dir_path, entry)
            try:
                entry_stat = await _maybe_await(operations.stat(full_path))
            except Exception:
                continue
            results.append(f"{entry}/" if _is_directory(entry_stat) else entry)

        if not results:
            return AgentToolResult(content=[TextContent(text="(empty directory)")], details=None)

        truncation = truncate_head("\n".join(results), TruncationOptions(maxLines=2**31 - 1))
        output = truncation.content
        details = LsToolDetails()
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
            details.entryLimitReached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=_details_or_none(details),
        )

    return ToolDefinition(
        name="ls",
        label="ls",
        description="List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories.",
        parameters=LsToolInput,
        execute=execute,
    )


def create_ls_tool(cwd: str, options: LsToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_ls_tool_definition(cwd, options))


createLsTool = create_ls_tool
createLsToolDefinition = create_ls_tool_definition

__all__ = [
    "DEFAULT_LIMIT",
    "LsOperations",
    "LsToolDetails",
    "LsToolInput",
    "LsToolOptions",
    "createLsTool",
    "createLsToolDefinition",
    "create_ls_tool",
    "create_ls_tool_definition",
]
