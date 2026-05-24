"""Shared tool exports for the coding-agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from harnify_agent.types import AgentTool

from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools.bash import (
    BashExecOptions,
    BashOperations,
    BashSpawnContext,
    BashSpawnHook,
    BashToolDetails,
    BashToolInput,
    BashToolOptions,
    create_bash_tool,
    create_bash_tool_definition,
    create_local_bash_operations,
    createBashTool,
    createBashToolDefinition,
    createLocalBashOperations,
)
from harnify_coding_agent.core.tools.edit import (
    EditOperations,
    EditToolDetails,
    EditToolInput,
    EditToolOptions,
    ReplaceEditInput,
    create_edit_tool,
    create_edit_tool_definition,
    createEditTool,
    createEditToolDefinition,
    prepare_edit_arguments,
    prepareEditArguments,
)
from harnify_coding_agent.core.tools.edit_diff import (
    AppliedEditsResult,
    Edit,
    EditDiffError,
    EditDiffResult,
    FuzzyMatchResult,
    apply_edits_to_normalized_content,
    applyEditsToNormalizedContent,
    compute_edit_diff,
    compute_edits_diff,
    computeEditDiff,
    computeEditsDiff,
    detect_line_ending,
    detectLineEnding,
    fuzzy_find_text,
    fuzzyFindText,
    generate_diff_string,
    generate_unified_patch,
    generateDiffString,
    generateUnifiedPatch,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    normalizeForFuzzyMatch,
    normalizeToLF,
    restore_line_endings,
    restoreLineEndings,
    strip_bom,
    stripBom,
)
from harnify_coding_agent.core.tools.file_mutation_queue import with_file_mutation_queue, withFileMutationQueue
from harnify_coding_agent.core.tools.find import (
    FindOperations,
    FindToolDetails,
    FindToolInput,
    FindToolOptions,
    create_find_tool,
    create_find_tool_definition,
    createFindTool,
    createFindToolDefinition,
)
from harnify_coding_agent.core.tools.grep import (
    GrepOperations,
    GrepToolDetails,
    GrepToolInput,
    GrepToolOptions,
    create_grep_tool,
    create_grep_tool_definition,
    createGrepTool,
    createGrepToolDefinition,
)
from harnify_coding_agent.core.tools.ls import (
    LsOperations,
    LsToolDetails,
    LsToolInput,
    LsToolOptions,
    create_ls_tool,
    create_ls_tool_definition,
    createLsTool,
    createLsToolDefinition,
)
from harnify_coding_agent.core.tools.output_accumulator import (
    OutputAccumulator,
    OutputAccumulatorOptions,
    OutputSnapshot,
    byte_length,
    byteLength,
    default_temp_file_path,
    defaultTempFilePath,
)
from harnify_coding_agent.core.tools.path_utils import (
    expand_path,
    expandPath,
    resolve_read_path,
    resolve_to_cwd,
    resolveReadPath,
    resolveToCwd,
)
from harnify_coding_agent.core.tools.read import (
    ReadOperations,
    ReadToolDetails,
    ReadToolInput,
    ReadToolOptions,
    create_read_tool,
    create_read_tool_definition,
    createReadTool,
    createReadToolDefinition,
)
from harnify_coding_agent.core.tools.render_utils import (
    ToolRenderResultLike,
    get_text_output,
    getTextOutput,
    invalid_arg_text,
    invalidArgText,
    normalize_display_text,
    normalizeDisplayText,
    replace_tabs,
    replaceTabs,
    shorten_path,
    shortenPath,
    str,
    str_value,
)
from harnify_coding_agent.core.tools.tool_definition_wrapper import (
    create_tool_definition_from_agent_tool,
    createToolDefinitionFromAgentTool,
    wrap_tool_definition,
    wrap_tool_definitions,
    wrapToolDefinition,
    wrapToolDefinitions,
)
from harnify_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationOptions,
    TruncationResult,
    format_size,
    formatSize,
    truncate_head,
    truncate_line,
    truncate_tail,
    truncateHead,
    truncateLine,
    truncateTail,
)
from harnify_coding_agent.core.tools.write import (
    WriteOperations,
    WriteToolInput,
    WriteToolOptions,
    create_write_tool,
    create_write_tool_definition,
    createWriteTool,
    createWriteToolDefinition,
)

type Tool = AgentTool
type ToolDef = ToolDefinition[Any, Any]
type ToolName = Literal["read", "bash", "edit", "write", "grep", "find", "ls"]

all_tool_names: set[ToolName] = {"read", "bash", "edit", "write", "grep", "find", "ls"}
allToolNames = all_tool_names


class ToolsOptions(TypedDict, total=False):
    read: ReadToolOptions | Mapping[str, Any]
    bash: BashToolOptions | Mapping[str, Any]
    write: WriteToolOptions | Mapping[str, Any]
    edit: EditToolOptions | Mapping[str, Any]
    grep: GrepToolOptions | Mapping[str, Any]
    find: FindToolOptions | Mapping[str, Any]
    ls: LsToolOptions | Mapping[str, Any]


def _get_tool_options(options: ToolsOptions | Mapping[str, Any] | None, key: ToolName) -> Any:
    if options is None:
        return None
    return options.get(key)


def create_tool_definition(
    tool_name: ToolName,
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> ToolDef:
    match tool_name:
        case "read":
            return create_read_tool_definition(cwd, _get_tool_options(options, "read"))
        case "bash":
            return create_bash_tool_definition(cwd, _get_tool_options(options, "bash"))
        case "edit":
            return create_edit_tool_definition(cwd, _get_tool_options(options, "edit"))
        case "write":
            return create_write_tool_definition(cwd, _get_tool_options(options, "write"))
        case "grep":
            return create_grep_tool_definition(cwd, _get_tool_options(options, "grep"))
        case "find":
            return create_find_tool_definition(cwd, _get_tool_options(options, "find"))
        case "ls":
            return create_ls_tool_definition(cwd, _get_tool_options(options, "ls"))
        case _:
            raise ValueError(f"Unknown tool name: {tool_name}")


def create_tool(
    tool_name: ToolName,
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> Tool:
    match tool_name:
        case "read":
            return create_read_tool(cwd, _get_tool_options(options, "read"))
        case "bash":
            return create_bash_tool(cwd, _get_tool_options(options, "bash"))
        case "edit":
            return create_edit_tool(cwd, _get_tool_options(options, "edit"))
        case "write":
            return create_write_tool(cwd, _get_tool_options(options, "write"))
        case "grep":
            return create_grep_tool(cwd, _get_tool_options(options, "grep"))
        case "find":
            return create_find_tool(cwd, _get_tool_options(options, "find"))
        case "ls":
            return create_ls_tool(cwd, _get_tool_options(options, "ls"))
        case _:
            raise ValueError(f"Unknown tool name: {tool_name}")


def create_coding_tool_definitions(
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> list[ToolDef]:
    return [
        create_read_tool_definition(cwd, _get_tool_options(options, "read")),
        create_bash_tool_definition(cwd, _get_tool_options(options, "bash")),
        create_edit_tool_definition(cwd, _get_tool_options(options, "edit")),
        create_write_tool_definition(cwd, _get_tool_options(options, "write")),
    ]


def create_read_only_tool_definitions(
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> list[ToolDef]:
    return [
        create_read_tool_definition(cwd, _get_tool_options(options, "read")),
        create_grep_tool_definition(cwd, _get_tool_options(options, "grep")),
        create_find_tool_definition(cwd, _get_tool_options(options, "find")),
        create_ls_tool_definition(cwd, _get_tool_options(options, "ls")),
    ]


def create_all_tool_definitions(
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> dict[ToolName, ToolDef]:
    return {
        "read": create_read_tool_definition(cwd, _get_tool_options(options, "read")),
        "bash": create_bash_tool_definition(cwd, _get_tool_options(options, "bash")),
        "edit": create_edit_tool_definition(cwd, _get_tool_options(options, "edit")),
        "write": create_write_tool_definition(cwd, _get_tool_options(options, "write")),
        "grep": create_grep_tool_definition(cwd, _get_tool_options(options, "grep")),
        "find": create_find_tool_definition(cwd, _get_tool_options(options, "find")),
        "ls": create_ls_tool_definition(cwd, _get_tool_options(options, "ls")),
    }


def create_coding_tools(
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> list[Tool]:
    return [
        create_read_tool(cwd, _get_tool_options(options, "read")),
        create_bash_tool(cwd, _get_tool_options(options, "bash")),
        create_edit_tool(cwd, _get_tool_options(options, "edit")),
        create_write_tool(cwd, _get_tool_options(options, "write")),
    ]


def create_read_only_tools(
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> list[Tool]:
    return [
        create_read_tool(cwd, _get_tool_options(options, "read")),
        create_grep_tool(cwd, _get_tool_options(options, "grep")),
        create_find_tool(cwd, _get_tool_options(options, "find")),
        create_ls_tool(cwd, _get_tool_options(options, "ls")),
    ]


def create_all_tools(
    cwd: str,
    options: ToolsOptions | Mapping[str, Any] | None = None,
) -> dict[ToolName, Tool]:
    return {
        "read": create_read_tool(cwd, _get_tool_options(options, "read")),
        "bash": create_bash_tool(cwd, _get_tool_options(options, "bash")),
        "edit": create_edit_tool(cwd, _get_tool_options(options, "edit")),
        "write": create_write_tool(cwd, _get_tool_options(options, "write")),
        "grep": create_grep_tool(cwd, _get_tool_options(options, "grep")),
        "find": create_find_tool(cwd, _get_tool_options(options, "find")),
        "ls": create_ls_tool(cwd, _get_tool_options(options, "ls")),
    }


createToolDefinition = create_tool_definition
createTool = create_tool
createCodingToolDefinitions = create_coding_tool_definitions
createReadOnlyToolDefinitions = create_read_only_tool_definitions
createAllToolDefinitions = create_all_tool_definitions
createCodingTools = create_coding_tools
createReadOnlyTools = create_read_only_tools
createAllTools = create_all_tools

__all__ = [
    "AppliedEditsResult",
    "BashExecOptions",
    "BashOperations",
    "BashSpawnHook",
    "BashSpawnContext",
    "BashToolDetails",
    "BashToolInput",
    "BashToolOptions",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "Edit",
    "EditDiffError",
    "EditDiffResult",
    "EditOperations",
    "EditToolDetails",
    "EditToolInput",
    "EditToolOptions",
    "FindOperations",
    "FindToolDetails",
    "FindToolInput",
    "FindToolOptions",
    "FuzzyMatchResult",
    "GrepOperations",
    "GrepToolDetails",
    "GrepToolInput",
    "GrepToolOptions",
    "GREP_MAX_LINE_LENGTH",
    "LsOperations",
    "LsToolDetails",
    "LsToolInput",
    "LsToolOptions",
    "OutputAccumulator",
    "OutputAccumulatorOptions",
    "OutputSnapshot",
    "ReadOperations",
    "ReadToolDetails",
    "ReadToolInput",
    "ReadToolOptions",
    "Tool",
    "ToolDef",
    "ToolName",
    "ToolRenderResultLike",
    "ToolsOptions",
    "TruncationOptions",
    "TruncationResult",
    "ReplaceEditInput",
    "WriteOperations",
    "WriteToolInput",
    "WriteToolOptions",
    "allToolNames",
    "all_tool_names",
    "applyEditsToNormalizedContent",
    "apply_edits_to_normalized_content",
    "byteLength",
    "byte_length",
    "createAllToolDefinitions",
    "createAllTools",
    "createBashTool",
    "createBashToolDefinition",
    "createCodingToolDefinitions",
    "createCodingTools",
    "createEditTool",
    "createEditToolDefinition",
    "createFindTool",
    "createFindToolDefinition",
    "createGrepTool",
    "createGrepToolDefinition",
    "createLsTool",
    "createLsToolDefinition",
    "createReadOnlyToolDefinitions",
    "createReadOnlyTools",
    "createTool",
    "createToolDefinitionFromAgentTool",
    "createToolDefinition",
    "createLocalBashOperations",
    "createReadTool",
    "createReadToolDefinition",
    "createWriteTool",
    "createWriteToolDefinition",
    "create_all_tool_definitions",
    "create_all_tools",
    "create_bash_tool",
    "create_bash_tool_definition",
    "create_coding_tool_definitions",
    "create_coding_tools",
    "create_edit_tool",
    "create_edit_tool_definition",
    "create_find_tool",
    "create_find_tool_definition",
    "create_grep_tool",
    "create_grep_tool_definition",
    "create_ls_tool",
    "create_ls_tool_definition",
    "create_read_only_tool_definitions",
    "create_read_only_tools",
    "create_tool",
    "create_tool_definition_from_agent_tool",
    "create_tool_definition",
    "create_local_bash_operations",
    "create_read_tool",
    "create_read_tool_definition",
    "create_write_tool",
    "create_write_tool_definition",
    "computeEditDiff",
    "computeEditsDiff",
    "compute_edit_diff",
    "compute_edits_diff",
    "defaultTempFilePath",
    "default_temp_file_path",
    "detectLineEnding",
    "detect_line_ending",
    "expandPath",
    "expand_path",
    "fuzzyFindText",
    "fuzzy_find_text",
    "formatSize",
    "format_size",
    "generateDiffString",
    "generateUnifiedPatch",
    "generate_diff_string",
    "generate_unified_patch",
    "getTextOutput",
    "get_text_output",
    "invalidArgText",
    "invalid_arg_text",
    "normalizeForFuzzyMatch",
    "normalizeDisplayText",
    "normalizeToLF",
    "normalize_for_fuzzy_match",
    "normalize_display_text",
    "normalize_to_lf",
    "prepareEditArguments",
    "prepare_edit_arguments",
    "replaceTabs",
    "replace_tabs",
    "resolveReadPath",
    "resolveToCwd",
    "resolve_read_path",
    "resolve_to_cwd",
    "restoreLineEndings",
    "restore_line_endings",
    "shortenPath",
    "shorten_path",
    "str",
    "stripBom",
    "strip_bom",
    "str_value",
    "truncateHead",
    "truncateLine",
    "truncateTail",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
    "withFileMutationQueue",
    "with_file_mutation_queue",
    "wrapToolDefinition",
    "wrapToolDefinitions",
    "wrap_tool_definition",
    "wrap_tool_definitions",
]
