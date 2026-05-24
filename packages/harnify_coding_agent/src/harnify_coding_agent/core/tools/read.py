"""Read tool for text and image files."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import ImageContent, Model, TextContent
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.path_utils import resolve_read_path
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition
from harnify_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)
from harnify_coding_agent.utils.image_resize import format_dimension_note, resize_image
from harnify_coding_agent.utils.mime import detect_supported_image_mime_type_from_file


class ReadToolInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str = Field(
        description="Path to the file to read (relative or absolute)",
        validation_alias=AliasChoices("path", "file_path"),
    )
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")


@dataclass(slots=True)
class ReadToolDetails:
    truncation: TruncationResult | None = None


class ReadOperations(Protocol):
    readFile: Callable[[str], Awaitable[bytes]]
    access: Callable[[str], Awaitable[None]]
    detectImageMimeType: Callable[[str], Awaitable[str | None]] | None


@dataclass(slots=True)
class ReadToolOptions:
    autoResizeImages: bool = True
    operations: ReadOperations | None = None


@dataclass(slots=True)
class _DefaultReadOperations:
    async def readFile(self, absolute_path: str) -> bytes:
        return await asyncio.to_thread(Path(absolute_path).read_bytes)

    async def access(self, absolute_path: str) -> None:
        def _check() -> None:
            with open(absolute_path, "rb"):
                return None

        await asyncio.to_thread(_check)

    async def detectImageMimeType(self, absolute_path: str) -> str | None:
        return await detect_supported_image_mime_type_from_file(absolute_path)


def _coerce_options(options: ReadToolOptions | Mapping[str, Any] | None) -> ReadToolOptions:
    if options is None:
        return ReadToolOptions()
    if isinstance(options, ReadToolOptions):
        return options
    return ReadToolOptions(
        autoResizeImages=options.get("autoResizeImages", True),
        operations=options.get("operations"),
    )


def _prepare_arguments(args: Any) -> dict[str, Any]:
    return ReadToolInput.model_validate(args).model_dump(exclude_none=True)


def _is_aborted(signal: Any | None) -> bool:
    return bool(getattr(signal, "aborted", False))


def _ensure_not_aborted(signal: Any | None) -> None:
    if _is_aborted(signal):
        raise RuntimeError("Operation aborted")


def _non_vision_image_note(model: Model | None) -> str | None:
    if model is None or "image" in model.input:
        return None
    return "[Current model does not support images. The image will be omitted from this request.]"


def _make_text_result(text: str, details: ReadToolDetails | None = None) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)], details=details)


def create_read_tool_definition(
    cwd: str,
    options: ReadToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], ReadToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultReadOperations()

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        parsed = ReadToolInput.model_validate(params)
        absolute_path = resolve_read_path(parsed.path, cwd)

        _ensure_not_aborted(signal)
        await operations.access(absolute_path)
        _ensure_not_aborted(signal)

        detect_mime = getattr(operations, "detectImageMimeType", None)
        mime_type = await detect_mime(absolute_path) if callable(detect_mime) else None
        non_vision_note = _non_vision_image_note(getattr(ctx, "model", None))

        if mime_type:
            buffer = await operations.readFile(absolute_path)
            _ensure_not_aborted(signal)
            base64_data = base64.b64encode(buffer).decode("ascii")
            if resolved_options.autoResizeImages:
                resized = await resize_image(ImageContent(data=base64_data, mimeType=mime_type))
                _ensure_not_aborted(signal)
                if not resized:
                    text_note = (
                        f"Read image file [{mime_type}]\n"
                        "[Image omitted: could not be resized below the inline image size limit.]"
                    )
                    if non_vision_note:
                        text_note += f"\n{non_vision_note}"
                    return _make_text_result(text_note)

                text_note = f"Read image file [{resized.mimeType}]"
                dimension_note = format_dimension_note(resized)
                if dimension_note:
                    text_note += f"\n{dimension_note}"
                if non_vision_note:
                    text_note += f"\n{non_vision_note}"
                return AgentToolResult(
                    content=[
                        TextContent(text=text_note),
                        ImageContent(data=resized.data, mimeType=resized.mimeType),
                    ],
                    details=None,
                )

            text_note = f"Read image file [{mime_type}]"
            if non_vision_note:
                text_note += f"\n{non_vision_note}"
            return AgentToolResult(
                content=[TextContent(text=text_note), ImageContent(data=base64_data, mimeType=mime_type)],
                details=None,
            )

        buffer = await operations.readFile(absolute_path)
        _ensure_not_aborted(signal)
        text_content = buffer.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)

        start_line = max(0, parsed.offset - 1) if parsed.offset else 0
        start_line_display = start_line + 1
        if start_line >= len(all_lines):
            raise RuntimeError(f"Offset {parsed.offset} is beyond end of file ({len(all_lines)} lines total)")

        selected_content: str
        user_limited_lines: int | None = None
        if parsed.limit is not None:
            end_line = min(start_line + parsed.limit, len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected_content)
        details: ReadToolDetails | None = None
        if truncation.firstLineExceedsLimit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {parsed.path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = ReadToolDetails(truncation=truncation)
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.outputLines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncatedBy == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = ReadToolDetails(truncation=truncation)
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = (
                f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )
        else:
            output_text = truncation.content

        return _make_text_result(output_text, details)

    return ToolDefinition(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
            f"Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} "
            f"lines or {DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). Use offset/limit for large files. "
            "When you need the full file, continue with offset until complete."
        ),
        parameters=ReadToolInput,
        prepareArguments=_prepare_arguments,
        execute=execute,
    )


def create_read_tool(cwd: str, options: ReadToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_read_tool_definition(cwd, options))


createReadTool = create_read_tool
createReadToolDefinition = create_read_tool_definition

__all__ = [
    "ReadOperations",
    "ReadToolDetails",
    "ReadToolInput",
    "ReadToolOptions",
    "createReadTool",
    "createReadToolDefinition",
    "create_read_tool",
    "create_read_tool_definition",
]
