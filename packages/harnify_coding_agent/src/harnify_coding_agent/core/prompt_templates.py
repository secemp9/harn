"""Prompt-template loading and expansion helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from ruamel.yaml import YAML

from harnify_coding_agent.core.source_info import SourceInfo, create_synthetic_source_info
from harnify_coding_agent.utils.paths import resolve_path

CONFIG_DIR_NAME = ".harnify"


@dataclass(slots=True)
class PromptTemplate:
    name: str
    description: str
    content: str
    sourceInfo: SourceInfo
    filePath: str
    argumentHint: str | None = None


class LoadPromptTemplatesOptions(TypedDict):
    cwd: str
    agentDir: str
    promptPaths: list[str]
    includeDefaults: bool


def parse_command_args(args_string: str) -> list[str]:
    args: list[str] = []
    current = ""
    in_quote: str | None = None
    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in {'"', "'"}:
            in_quote = char
        elif char.isspace():
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    def replace_index(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return args[index] if 0 <= index < len(args) else ""

    def replace_slice(match: re.Match[str]) -> str:
        start = max(int(match.group(1)) - 1, 0)
        length = match.group(2)
        if length is not None:
            return " ".join(args[start : start + int(length)])
        return " ".join(args[start:])

    result = re.sub(r"\$(\d+)", replace_index, content)
    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", replace_slice, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    return result.replace("$@", all_args)


def load_prompt_templates(options: LoadPromptTemplatesOptions) -> list[PromptTemplate]:
    resolved_cwd = resolve_path(options["cwd"])
    resolved_agent_dir = resolve_path(options.get("agentDir") or _default_agent_dir())
    prompt_paths = list(options.get("promptPaths", []))
    include_defaults = bool(options.get("includeDefaults", True))

    templates: list[PromptTemplate] = []
    global_prompts_dir = os.path.join(resolved_agent_dir, "prompts")
    project_prompts_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "prompts")

    def is_under_path(target: str, root: str) -> bool:
        normalized_root = os.path.abspath(root)
        normalized_target = os.path.abspath(target)
        if normalized_target == normalized_root:
            return True
        prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
        return normalized_target.startswith(prefix)

    def get_source_info(resolved_path: str) -> SourceInfo:
        if is_under_path(resolved_path, global_prompts_dir):
            return create_synthetic_source_info(
                resolved_path,
                {"source": "local", "scope": "user", "baseDir": global_prompts_dir},
            )
        if is_under_path(resolved_path, project_prompts_dir):
            return create_synthetic_source_info(
                resolved_path,
                {"source": "local", "scope": "project", "baseDir": project_prompts_dir},
            )
        base_dir = resolved_path if os.path.isdir(resolved_path) else os.path.dirname(resolved_path)
        return create_synthetic_source_info(
            resolved_path,
            {"source": "local", "baseDir": base_dir},
        )

    if include_defaults:
        templates.extend(_load_templates_from_dir(global_prompts_dir, get_source_info))
        templates.extend(_load_templates_from_dir(project_prompts_dir, get_source_info))

    for raw_path in prompt_paths:
        resolved = resolve_path(raw_path, resolved_cwd, trim=True)
        if not os.path.exists(resolved):
            continue
        try:
            if os.path.isdir(resolved):
                templates.extend(_load_templates_from_dir(resolved, get_source_info))
            elif os.path.isfile(resolved) and resolved.endswith(".md"):
                template = _load_template_from_file(resolved, get_source_info(resolved))
                if template is not None:
                    templates.append(template)
        except OSError:
            continue

    return templates


def expand_prompt_template(text: str, templates: list[PromptTemplate]) -> str:
    if not text.startswith("/"):
        return text
    match = re.match(r"^/([^\s]+)(?:\s+([\s\S]*))?$", text)
    if match is None:
        return text
    template_name = match.group(1)
    args_string = match.group(2) or ""
    template = next((candidate for candidate in templates if candidate.name == template_name), None)
    if template is None:
        return text
    return substitute_args(template.content, parse_command_args(args_string))


def _load_templates_from_dir(dir_path: str, get_source_info: Any) -> list[PromptTemplate]:
    if not os.path.isdir(dir_path):
        return []
    templates: list[PromptTemplate] = []
    try:
        for entry in sorted(Path(dir_path).iterdir(), key=lambda candidate: candidate.name):
            entry_path = str(entry)
            is_file = entry.is_file()
            if entry.is_symlink():
                try:
                    is_file = entry.stat().st_mode is not None and Path(entry_path).is_file()
                except OSError:
                    continue
            if is_file and entry.name.endswith(".md"):
                template = _load_template_from_file(entry_path, get_source_info(entry_path))
                if template is not None:
                    templates.append(template)
    except OSError:
        return []
    return templates


def _load_template_from_file(file_path: str, source_info: SourceInfo) -> PromptTemplate | None:
    try:
        raw_content = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None
    frontmatter, body = _parse_frontmatter(raw_content)
    name = Path(file_path).stem
    description = str(frontmatter.get("description") or "")
    if not description:
        first_line = next((line for line in body.split("\n") if line.strip()), None)
        if first_line:
            description = first_line[:60] + ("..." if len(first_line) > 60 else "")
    return PromptTemplate(
        name=name,
        description=description,
        argumentHint=_string_or_none(frontmatter.get("argument-hint")),
        content=body,
        sourceInfo=source_info,
        filePath=file_path,
    )


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    data = _yaml_load(yaml_string) or {}
    return (data if isinstance(data, dict) else {}, body)


def _yaml_load(content: str) -> Any:
    yaml = YAML(typ="safe")
    return yaml.load(content)


def _string_or_none(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _default_agent_dir() -> str:
    return str(Path.home() / ".harnify" / "agent")


expandPromptTemplate = expand_prompt_template
loadPromptTemplates = load_prompt_templates
parseCommandArgs = parse_command_args
substituteArgs = substitute_args

__all__ = [
    "CONFIG_DIR_NAME",
    "LoadPromptTemplatesOptions",
    "PromptTemplate",
    "expandPromptTemplate",
    "expand_prompt_template",
    "loadPromptTemplates",
    "load_prompt_templates",
    "parseCommandArgs",
    "parse_command_args",
    "substituteArgs",
    "substitute_args",
]
