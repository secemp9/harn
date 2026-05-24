"""Write tool for creating and overwriting files."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import TextContent
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.file_mutation_queue import with_file_mutation_queue
from harnify_coding_agent.core.tools.path_utils import resolve_to_cwd
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition


class WriteToolInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str = Field(
        description="Path to the file to write (relative or absolute)",
        validation_alias=AliasChoices("path", "file_path"),
    )
    content: str = Field(description="Content to write to the file")


class WriteOperations(Protocol):
    writeFile: Callable[[str, str], Awaitable[None]]
    mkdir: Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class WriteToolOptions:
    operations: WriteOperations | None = None


@dataclass(slots=True)
class _DefaultWriteOperations:
    async def writeFile(self, absolute_path: str, content: str) -> None:
        await asyncio.to_thread(Path(absolute_path).write_text, content, encoding="utf-8")

    async def mkdir(self, directory: str) -> None:
        await asyncio.to_thread(os.makedirs, directory, exist_ok=True)


def _coerce_options(options: WriteToolOptions | Mapping[str, Any] | None) -> WriteToolOptions:
    if options is None:
        return WriteToolOptions()
    if isinstance(options, WriteToolOptions):
        return options
    return WriteToolOptions(operations=options.get("operations"))


def _prepare_arguments(args: Any) -> dict[str, Any]:
    return WriteToolInput.model_validate(args).model_dump(exclude_none=True)


def _is_aborted(signal: Any | None) -> bool:
    return bool(getattr(signal, "aborted", False))


def _ensure_not_aborted(signal: Any | None) -> None:
    if _is_aborted(signal):
        raise RuntimeError("Operation aborted")


def create_write_tool_definition(
    cwd: str,
    options: WriteToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultWriteOperations()

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        _ctx: Any = None,
    ) -> AgentToolResult:
        parsed = WriteToolInput.model_validate(params)
        absolute_path = resolve_to_cwd(parsed.path, cwd)
        directory = os.path.dirname(absolute_path)

        async def mutate() -> AgentToolResult:
            _ensure_not_aborted(signal)
            await operations.mkdir(directory)
            _ensure_not_aborted(signal)
            await operations.writeFile(absolute_path, parsed.content)
            _ensure_not_aborted(signal)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote {len(parsed.content)} bytes to {parsed.path}")],
                details=None,
            )

        return await with_file_mutation_queue(absolute_path, mutate)

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters=WriteToolInput,
        prepareArguments=_prepare_arguments,
        execute=execute,
    )


def create_write_tool(cwd: str, options: WriteToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_write_tool_definition(cwd, options))


createWriteTool = create_write_tool
createWriteToolDefinition = create_write_tool_definition

__all__ = [
    "WriteOperations",
    "WriteToolInput",
    "WriteToolOptions",
    "createWriteTool",
    "createWriteToolDefinition",
    "create_write_tool",
    "create_write_tool_definition",
]
