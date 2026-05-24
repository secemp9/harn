"""Ripgrep-backed content search tool."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import TextContent
from pydantic import BaseModel, ConfigDict, Field

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.path_utils import resolve_to_cwd
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition
from harnify_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationOptions,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
)
from harnify_coding_agent.utils.tools_manager import ensure_tool

DEFAULT_LIMIT = 100


class GrepToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pattern: str = Field(description="Search pattern (regex or literal string)")
    path: str | None = Field(default=None, description="Directory or file to search (default: current directory)")
    glob: str | None = Field(default=None, description="Filter files by glob pattern, e.g. '*.ts'")
    ignoreCase: bool | None = Field(default=None, description="Case-insensitive search (default: false)")
    literal: bool | None = Field(default=None, description="Treat pattern as literal string instead of regex")
    context: int | None = Field(default=None, description="Number of lines to show before and after each match")
    limit: int | None = Field(default=None, description="Maximum number of matches to return (default: 100)")


@dataclass(slots=True)
class GrepToolDetails:
    truncation: TruncationResult | None = None
    matchLimitReached: int | None = None
    linesTruncated: bool | None = None


class GrepOperations(Protocol):
    isDirectory: Callable[[str], Awaitable[bool] | bool]
    readFile: Callable[[str], Awaitable[str] | str]


@dataclass(slots=True)
class GrepToolOptions:
    operations: GrepOperations | None = None


@dataclass(slots=True)
class _DefaultGrepOperations:
    def isDirectory(self, absolute_path: str) -> bool:
        path = Path(absolute_path)
        path.stat()
        return path.is_dir()

    def readFile(self, absolute_path: str) -> str:
        return Path(absolute_path).read_text(encoding="utf-8")


def _coerce_options(options: GrepToolOptions | Mapping[str, Any] | None) -> GrepToolOptions:
    if options is None:
        return GrepToolOptions()
    if isinstance(options, GrepToolOptions):
        return options
    return GrepToolOptions(operations=options.get("operations"))


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if asyncio.isfuture(value) or hasattr(value, "__await__"):
        return await value
    return value


def _format_match_path(file_path: str, search_path: str, is_directory: bool) -> str:
    if is_directory:
        relative = os.path.relpath(file_path, search_path)
        if relative and not relative.startswith(".."):
            return relative.replace(os.sep, "/")
    return os.path.basename(file_path)


def _details_or_none(details: GrepToolDetails) -> GrepToolDetails | None:
    if any(getattr(details, field.name) is not None for field in fields(details)):
        return details
    return None


def create_grep_tool_definition(
    cwd: str,
    options: GrepToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], GrepToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultGrepOperations()

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        _ctx: Any = None,
    ) -> AgentToolResult:
        parsed = GrepToolInput.model_validate(params)
        rg_path = await ensure_tool("rg", silent=True)
        if not rg_path:
            raise RuntimeError("ripgrep (rg) is not available")

        search_path = resolve_to_cwd(parsed.path or ".", cwd)
        try:
            is_directory = await _maybe_await(operations.isDirectory(search_path))
        except Exception:
            raise RuntimeError(f"Path not found: {search_path}") from None
        context_value = parsed.context if parsed.context and parsed.context > 0 else 0
        effective_limit = max(1, parsed.limit or DEFAULT_LIMIT)

        args: list[str] = [
            "--json",
            "--line-number",
            "--color=never",
            "--hidden",
            "--max-count",
            str(effective_limit + 1),
        ]
        if parsed.ignoreCase:
            args.append("--ignore-case")
        if parsed.literal:
            args.append("--fixed-strings")
        if parsed.glob:
            args.extend(["--glob", parsed.glob])
        args.extend(["--", parsed.pattern, search_path])

        process = await asyncio.create_subprocess_exec(
            rg_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        communicate_task = asyncio.create_task(process.communicate())
        abort_task = asyncio.create_task(signal.wait()) if signal is not None and hasattr(signal, "wait") else None
        if abort_task is not None:
            done, _pending = await asyncio.wait({communicate_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if abort_task in done and not communicate_task.done():
                process.kill()
                await communicate_task
                raise RuntimeError("Operation aborted")

        stdout, stderr = await communicate_task
        if abort_task is not None:
            abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)

        if getattr(signal, "aborted", False):
            raise RuntimeError("Operation aborted")

        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode not in {0, 1}:
            raise RuntimeError(stderr_text or f"ripgrep exited with code {process.returncode}")

        matches: list[tuple[str, int, str | None]] = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            path_data = data.get("path") or {}
            file_path = path_data.get("text")
            line_number = data.get("line_number")
            line_text = ((data.get("lines") or {}).get("text")) if isinstance(data.get("lines"), dict) else None
            if isinstance(file_path, str) and isinstance(line_number, int):
                matches.append((file_path, line_number, line_text))

        match_limit_reached = len(matches) > effective_limit
        if match_limit_reached:
            matches = matches[:effective_limit]

        if not matches:
            return AgentToolResult(content=[TextContent(text="No matches found")], details=None)

        file_cache: dict[str, list[str]] = {}
        lines_truncated = False
        output_lines: list[str] = []

        async def get_file_lines(file_path: str) -> list[str]:
            cached = file_cache.get(file_path)
            if cached is not None:
                return cached
            try:
                content = await _maybe_await(operations.readFile(file_path))
            except Exception:
                content = ""
            lines = str(content).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            file_cache[file_path] = lines
            return lines

        for file_path, line_number, line_text in matches:
            display_path = _format_match_path(file_path, search_path, bool(is_directory))
            if context_value == 0 and line_text is not None:
                sanitized = line_text.replace("\r\n", "\n").replace("\r", "").removesuffix("\n")
                truncated = truncate_line(sanitized)
                lines_truncated = lines_truncated or bool(truncated["wasTruncated"])
                output_lines.append(f"{display_path}:{line_number}: {truncated['text']}")
                continue

            file_lines = await get_file_lines(file_path)
            if not file_lines:
                output_lines.append(f"{display_path}:{line_number}: (unable to read file)")
                continue
            start = max(1, line_number - context_value)
            end = min(len(file_lines), line_number + context_value)
            for current_line in range(start, end + 1):
                text = (file_lines[current_line - 1] if current_line - 1 < len(file_lines) else "").replace("\r", "")
                truncated = truncate_line(text)
                lines_truncated = lines_truncated or bool(truncated["wasTruncated"])
                if current_line == line_number:
                    output_lines.append(f"{display_path}:{current_line}: {truncated['text']}")
                else:
                    output_lines.append(f"{display_path}-{current_line}- {truncated['text']}")

        raw_output = "\n".join(output_lines)
        truncation = truncate_head(raw_output, TruncationOptions(maxLines=2**31 - 1))
        output = truncation.content
        details = GrepToolDetails()
        notices: list[str] = []
        if match_limit_reached:
            notices.append(
                f"{effective_limit} matches limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.matchLimitReached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if lines_truncated:
            notices.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines")
            details.linesTruncated = True
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=_details_or_none(details),
        )

    return ToolDefinition(
        name="grep",
        label="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
            "Respects .gitignore."
        ),
        parameters=GrepToolInput,
        execute=execute,
    )


def create_grep_tool(cwd: str, options: GrepToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_grep_tool_definition(cwd, options))


createGrepTool = create_grep_tool
createGrepToolDefinition = create_grep_tool_definition

__all__ = [
    "DEFAULT_LIMIT",
    "GrepOperations",
    "GrepToolDetails",
    "GrepToolInput",
    "GrepToolOptions",
    "createGrepTool",
    "createGrepToolDefinition",
    "create_grep_tool",
    "create_grep_tool_definition",
]
