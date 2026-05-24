"""Edit tool for exact and fuzzy-aware search/replace operations."""

from __future__ import annotations

import asyncio
import errno as errno_module
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import TextContent
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from harnify_coding_agent.core.tools.file_mutation_queue import with_file_mutation_queue
from harnify_coding_agent.core.tools.path_utils import resolve_to_cwd
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition


class ReplaceEditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oldText: str = Field(
        description=(
            "Exact text for one targeted replacement. It must be unique in the original file "
            "and must not overlap with any other edits[].oldText in the same call."
        )
    )
    newText: str = Field(description="Replacement text for this targeted edit.")


class EditToolInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str = Field(
        description="Path to the file to edit (relative or absolute)",
        validation_alias=AliasChoices("path", "file_path"),
    )
    edits: list[ReplaceEditInput] = Field(
        description=(
            "One or more targeted replacements. Each edit is matched against the original file, not incrementally. "
            "Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, "
            "merge them into one edit instead."
        )
    )


@dataclass(slots=True)
class EditToolDetails:
    diff: str
    patch: str
    firstChangedLine: int | None = None


class EditOperations(Protocol):
    readFile: Callable[[str], Awaitable[bytes]]
    writeFile: Callable[[str, str], Awaitable[None]]
    access: Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class EditToolOptions:
    operations: EditOperations | None = None


@dataclass(slots=True)
class _DefaultEditOperations:
    async def readFile(self, absolute_path: str) -> bytes:
        return await asyncio.to_thread(Path(absolute_path).read_bytes)

    async def writeFile(self, absolute_path: str, content: str) -> None:
        await asyncio.to_thread(Path(absolute_path).write_text, content, encoding="utf-8")

    async def access(self, absolute_path: str) -> None:
        def _check() -> None:
            with open(absolute_path, "rb"):
                pass
            with open(absolute_path, "r+b"):
                pass

        await asyncio.to_thread(_check)


def _coerce_options(options: EditToolOptions | Mapping[str, Any] | None) -> EditToolOptions:
    if options is None:
        return EditToolOptions()
    if isinstance(options, EditToolOptions):
        return options
    return EditToolOptions(operations=options.get("operations"))


def prepare_edit_arguments(input_value: Any) -> Any:
    if not isinstance(input_value, dict):
        return input_value

    args = input_value
    changed = False
    edits_value = args.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            args = dict(args)
            args["edits"] = parsed
            changed = True

    old_text = args.get("oldText")
    new_text = args.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = list(args.get("edits")) if isinstance(args.get("edits"), list) else []
        edits.append({"oldText": old_text, "newText": new_text})
        next_args = dict(args)
        next_args.pop("oldText", None)
        next_args.pop("newText", None)
        next_args["edits"] = edits
        return next_args

    return args if not changed else dict(args)


def _validate_edit_input(input_value: EditToolInput) -> tuple[str, list[Edit]]:
    if not input_value.edits:
        raise RuntimeError("Edit tool input is invalid. edits must contain at least one replacement.")
    return input_value.path, [Edit(oldText=edit.oldText, newText=edit.newText) for edit in input_value.edits]


def _format_access_error(error: BaseException) -> str:
    if isinstance(error, OSError) and error.errno is not None:
        code = errno_module.errorcode.get(error.errno)
        if code:
            return f"Error code: {code}"
    if isinstance(error, Exception):
        return f"Error: {error}"
    return str(error)


def create_edit_tool_definition(
    cwd: str,
    options: EditToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[EditToolInput | dict[str, Any], EditToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultEditOperations()

    async def execute(
        _tool_call_id: str,
        input_value: EditToolInput | dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        _ctx: Any = None,
    ) -> AgentToolResult:
        parsed = EditToolInput.model_validate(input_value)
        path, edits = _validate_edit_input(parsed)
        absolute_path = resolve_to_cwd(path, cwd)

        async def mutate() -> AgentToolResult:
            if getattr(signal, "aborted", False):
                raise RuntimeError("Operation aborted")

            try:
                await operations.access(absolute_path)
            except Exception as error:
                raise RuntimeError(f"Could not edit file: {path}. {_format_access_error(error)}.") from None

            if getattr(signal, "aborted", False):
                raise RuntimeError("Operation aborted")

            buffer = await operations.readFile(absolute_path)
            if getattr(signal, "aborted", False):
                raise RuntimeError("Operation aborted")

            raw_content = buffer.decode("utf-8")
            bom, content = strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)

            if getattr(signal, "aborted", False):
                raise RuntimeError("Operation aborted")

            final_content = bom + restore_line_endings(applied.newContent, original_ending)
            await operations.writeFile(absolute_path, final_content)
            if getattr(signal, "aborted", False):
                raise RuntimeError("Operation aborted")

            diff_result = generate_diff_string(applied.baseContent, applied.newContent)
            patch = generate_unified_patch(path, applied.baseContent, applied.newContent)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=EditToolDetails(
                    diff=diff_result.diff,
                    patch=patch,
                    firstChangedLine=diff_result.firstChangedLine,
                ),
            )

        return await with_file_mutation_queue(absolute_path, mutate)

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file."
        ),
        parameters=EditToolInput,
        prepareArguments=prepare_edit_arguments,
        execute=execute,
    )


def create_edit_tool(cwd: str, options: EditToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_edit_tool_definition(cwd, options))


createEditTool = create_edit_tool
createEditToolDefinition = create_edit_tool_definition
prepareEditArguments = prepare_edit_arguments

__all__ = [
    "EditOperations",
    "EditToolDetails",
    "EditToolInput",
    "EditToolOptions",
    "ReplaceEditInput",
    "createEditTool",
    "createEditToolDefinition",
    "create_edit_tool",
    "create_edit_tool_definition",
    "prepareEditArguments",
    "prepare_edit_arguments",
]
