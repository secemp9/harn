"""File discovery tool with gitignore-aware glob matching."""

from __future__ import annotations

import asyncio
import os
import posixpath
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

from harnify_agent.types import AgentTool, AgentToolResult
from harnify_ai.types import TextContent
from pathspec import GitIgnoreSpec
from pydantic import BaseModel, ConfigDict, Field
from wcmatch import glob

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

DEFAULT_LIMIT = 1000
_GLOB_FLAGS = glob.GLOBSTAR | glob.DOTMATCH


class FindToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pattern: str = Field(description="Glob pattern to match files")
    path: str | None = Field(default=None, description="Directory to search in (default: current directory)")
    limit: int | None = Field(default=None, description="Maximum number of results (default: 1000)")


@dataclass(slots=True)
class FindToolDetails:
    truncation: TruncationResult | None = None
    resultLimitReached: int | None = None


class FindOperations(Protocol):
    exists: Callable[[str], Awaitable[bool] | bool]
    glob: Callable[[str, str, dict[str, Any]], Awaitable[list[str]] | list[str]]


@dataclass(slots=True)
class FindToolOptions:
    operations: FindOperations | None = None


@dataclass(slots=True)
class _IgnoreSpecEntry:
    base_dir: str
    spec: GitIgnoreSpec


@dataclass(slots=True)
class _DefaultFindOperations:
    def exists(self, absolute_path: str) -> bool:
        return os.path.exists(absolute_path)

    def glob(self, pattern: str, cwd: str, options: dict[str, Any]) -> list[str]:
        return _glob_files(pattern, cwd, limit=int(options["limit"]))


def _coerce_options(options: FindToolOptions | Mapping[str, Any] | None) -> FindToolOptions:
    if options is None:
        return FindToolOptions()
    if isinstance(options, FindToolOptions):
        return options
    return FindToolOptions(operations=options.get("operations"))


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if asyncio.isfuture(value) or hasattr(value, "__await__"):
        return await value
    return value


def _validate_glob_pattern(pattern: str) -> None:
    if pattern.count("[") != pattern.count("]"):
        raise RuntimeError(f"error parsing glob: {pattern}")


def _to_posix_path(value: str) -> str:
    return value.replace(os.sep, "/")


def _is_ancestor(base_dir: str, relative_path: str) -> bool:
    return relative_path == base_dir or relative_path.startswith(f"{base_dir}/")


def _load_gitignore_spec(directory: str) -> GitIgnoreSpec | None:
    gitignore_path = Path(directory) / ".gitignore"
    if not gitignore_path.exists():
        return None
    lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    return GitIgnoreSpec.from_lines(lines)


def _check_ignored(relative_path: str, specs: list[_IgnoreSpecEntry], *, is_dir: bool = False) -> bool:
    candidate = f"{relative_path}/" if is_dir else relative_path
    ignored = False
    for entry in specs:
        if entry.base_dir and not _is_ancestor(entry.base_dir, relative_path):
            continue
        subpath = candidate
        if entry.base_dir:
            subpath = posixpath.relpath(candidate, entry.base_dir)
        result = entry.spec.check_file(subpath)
        if result.include is not None:
            ignored = bool(result.include)
    return ignored


def _matches_pattern(relative_path: str, pattern: str) -> bool:
    if "/" in pattern:
        return glob.globmatch(relative_path, pattern, flags=_GLOB_FLAGS)
    return glob.globmatch(posixpath.basename(relative_path), pattern, flags=_GLOB_FLAGS)


def _normalize_result_path(path_value: str, search_path: str) -> str:
    had_trailing_slash = path_value.endswith("/") or path_value.endswith("\\")
    normalized_path = path_value.rstrip("/\\") if had_trailing_slash else path_value
    if os.path.isabs(normalized_path):
        relative_path = os.path.relpath(normalized_path, search_path)
    else:
        relative_path = normalized_path
    posix_value = _to_posix_path(relative_path)
    if had_trailing_slash and not posix_value.endswith("/"):
        posix_value += "/"
    return posix_value


def _details_or_none(details: FindToolDetails) -> FindToolDetails | None:
    if any(getattr(details, field.name) is not None for field in fields(details)):
        return details
    return None


def _glob_files(pattern: str, cwd: str, *, limit: int) -> list[str]:
    _validate_glob_pattern(pattern)
    root = Path(cwd)
    results: list[str] = []
    limit_reached = False
    spec_cache: dict[str, list[_IgnoreSpecEntry]] = {"": []}

    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        rel_dir = os.path.relpath(current_root, root)
        rel_dir_posix = "" if rel_dir in {".", ""} else _to_posix_path(rel_dir)
        parent_rel = posixpath.dirname(rel_dir_posix) if rel_dir_posix else ""
        active_specs = list(spec_cache[parent_rel] if rel_dir_posix else [])
        local_spec = _load_gitignore_spec(current_root)
        if local_spec is not None:
            active_specs.append(_IgnoreSpecEntry(base_dir=rel_dir_posix, spec=local_spec))
        spec_cache[rel_dir_posix] = active_specs

        dirnames.sort(key=str.lower)
        filenames.sort(key=str.lower)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            if dirname in {".git", "node_modules"}:
                continue
            child_rel = dirname if not rel_dir_posix else f"{rel_dir_posix}/{dirname}"
            if _check_ignored(child_rel, active_specs, is_dir=True):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            child_rel = filename if not rel_dir_posix else f"{rel_dir_posix}/{filename}"
            if _check_ignored(child_rel, active_specs):
                continue
            if not _matches_pattern(child_rel, pattern):
                continue
            if len(results) >= limit:
                limit_reached = True
                break
            results.append(child_rel)
        if limit_reached:
            break

    return results


def create_find_tool_definition(
    cwd: str,
    options: FindToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], FindToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultFindOperations()

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        _ctx: Any = None,
    ) -> AgentToolResult:
        parsed = FindToolInput.model_validate(params)
        search_path = resolve_to_cwd(parsed.path or ".", cwd)
        effective_limit = parsed.limit or DEFAULT_LIMIT

        if not await _maybe_await(operations.exists(search_path)):
            raise RuntimeError(f"Path not found: {search_path}")
        if getattr(signal, "aborted", False):
            raise RuntimeError("Operation aborted")

        try:
            results = await _maybe_await(
                operations.glob(
                    parsed.pattern,
                    search_path,
                    {"ignore": ["**/node_modules/**", "**/.git/**"], "limit": effective_limit},
                )
            )
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(str(error)) from error

        if getattr(signal, "aborted", False):
            raise RuntimeError("Operation aborted")

        if not results:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

        relativized = sorted(
            (_normalize_result_path(path_value, search_path) for path_value in results),
            key=lambda item: item.lower(),
        )
        result_limit_reached = len(relativized) >= effective_limit
        truncation = truncate_head("\n".join(relativized), TruncationOptions(maxLines=2**31 - 1))
        output = truncation.content
        details = FindToolDetails()
        notices: list[str] = []
        if result_limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.resultLimitReached = effective_limit
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
        name="find",
        label="find",
        description="Search for files by glob pattern. Returns matching file paths relative to the search directory.",
        parameters=FindToolInput,
        execute=execute,
    )


def create_find_tool(cwd: str, options: FindToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrap_tool_definition(create_find_tool_definition(cwd, options))


createFindTool = create_find_tool
createFindToolDefinition = create_find_tool_definition

__all__ = [
    "DEFAULT_LIMIT",
    "FindOperations",
    "FindToolDetails",
    "FindToolInput",
    "FindToolOptions",
    "createFindTool",
    "createFindToolDefinition",
    "create_find_tool",
    "create_find_tool_definition",
]
