"""Skill discovery, validation, and prompt formatting helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from pathspec import GitIgnoreSpec
from ruamel.yaml import YAML

from harnify_coding_agent.core.diagnostics import ResourceCollision, ResourceDiagnostic
from harnify_coding_agent.core.source_info import SourceInfo, create_synthetic_source_info
from harnify_coding_agent.utils.paths import canonicalize_path, resolve_path

CONFIG_DIR_NAME = ".harnify"
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    filePath: str
    baseDir: str
    sourceInfo: SourceInfo
    disableModelInvocation: bool


@dataclass(slots=True)
class LoadSkillsResult:
    skills: list[Skill]
    diagnostics: list[ResourceDiagnostic]


class LoadSkillsFromDirOptions(TypedDict):
    dir: str
    source: str


class LoadSkillsOptions(TypedDict):
    cwd: str
    agentDir: str
    skillPaths: list[str]
    includeDefaults: bool


class _IgnoreMatcher:
    def __init__(self) -> None:
        self._patterns: list[str] = []

    def add(self, patterns: list[str]) -> None:
        self._patterns.extend(patterns)

    def ignores(self, path: str) -> bool:
        if not self._patterns:
            return False
        return GitIgnoreSpec.from_lines(self._patterns).match_file(path)


def load_skills_from_dir(options: LoadSkillsFromDirOptions) -> LoadSkillsResult:
    return _load_skills_from_dir_internal(options["dir"], options["source"], True)


def load_skills(options: LoadSkillsOptions) -> LoadSkillsResult:
    resolved_cwd = resolve_path(options["cwd"])
    resolved_agent_dir = resolve_path(options.get("agentDir") or _default_agent_dir())
    include_defaults = bool(options.get("includeDefaults", True))
    skill_paths = list(options.get("skillPaths", []))

    skill_map: dict[str, Skill] = {}
    real_path_set: set[str] = set()
    diagnostics: list[ResourceDiagnostic] = []
    collision_diagnostics: list[ResourceDiagnostic] = []

    def add_skills(result: LoadSkillsResult) -> None:
        diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            real_path = canonicalize_path(skill.filePath)
            if real_path in real_path_set:
                continue
            existing = skill_map.get(skill.name)
            if existing is not None:
                collision_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.filePath,
                        collision=ResourceCollision(
                            resourceType="skill",
                            name=skill.name,
                            winnerPath=existing.filePath,
                            loserPath=skill.filePath,
                        ),
                    )
                )
                continue
            skill_map[skill.name] = skill
            real_path_set.add(real_path)

    user_skills_dir = os.path.join(resolved_agent_dir, "skills")
    project_skills_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "skills")

    if include_defaults:
        add_skills(_load_skills_from_dir_internal(user_skills_dir, "user", True))
        add_skills(_load_skills_from_dir_internal(project_skills_dir, "project", True))

    def is_under_path(target: str, root: str) -> bool:
        normalized_root = os.path.abspath(root)
        normalized_target = os.path.abspath(target)
        if normalized_target == normalized_root:
            return True
        prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
        return normalized_target.startswith(prefix)

    def get_source(resolved_path: str) -> str:
        if not include_defaults:
            if is_under_path(resolved_path, user_skills_dir):
                return "user"
            if is_under_path(resolved_path, project_skills_dir):
                return "project"
        return "path"

    for raw_path in skill_paths:
        resolved = resolve_path(raw_path, resolved_cwd, trim=True)
        if not os.path.exists(resolved):
            diagnostics.append(ResourceDiagnostic(type="warning", message="skill path does not exist", path=resolved))
            continue
        try:
            source = get_source(resolved)
            if os.path.isdir(resolved):
                add_skills(_load_skills_from_dir_internal(resolved, source, True))
            elif os.path.isfile(resolved) and resolved.endswith(".md"):
                result = _load_skill_from_file(resolved, source)
                if result.skill is not None:
                    add_skills(LoadSkillsResult(skills=[result.skill], diagnostics=result.diagnostics))
                else:
                    diagnostics.extend(result.diagnostics)
            else:
                diagnostics.append(
                    ResourceDiagnostic(
                        type="warning",
                        message="skill path is not a markdown file",
                        path=resolved,
                    )
                )
        except OSError as error:
            diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=resolved))

    return LoadSkillsResult(skills=list(skill_map.values()), diagnostics=[*diagnostics, *collision_diagnostics])


def format_skills_for_prompt(skills: list[Skill]) -> str:
    visible_skills = [skill for skill in skills if not skill.disableModelInvocation]
    if not visible_skills:
        return ""
    lines = [
        "",
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        (
            "When a skill file references a relative path, resolve it against the skill directory "
            "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands."
        ),
        "",
        "<available_skills>",
    ]
    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.filePath)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _load_skills_from_dir_internal(
    dir_path: str,
    source: str,
    include_root_files: bool,
    ignore_matcher: _IgnoreMatcher | None = None,
    root_dir: str | None = None,
) -> LoadSkillsResult:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []
    if not os.path.isdir(dir_path):
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    root = root_dir or dir_path
    matcher = ignore_matcher or _IgnoreMatcher()
    _add_ignore_rules(matcher, dir_path, root)

    try:
        entries = list(Path(dir_path).iterdir())
    except OSError:
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        entry_path = str(entry)
        if not _is_file(entry):
            continue
        rel_path = _to_posix_path(os.path.relpath(entry_path, root))
        if matcher.ignores(rel_path):
            continue
        result = _load_skill_from_file(entry_path, source)
        if result.skill is not None:
            skills.append(result.skill)
        diagnostics.extend(result.diagnostics)
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        entry_path = str(entry)
        is_directory = _is_dir(entry)
        is_file = _is_file(entry)
        rel_path = _to_posix_path(os.path.relpath(entry_path, root))
        ignore_path = f"{rel_path}/" if is_directory else rel_path
        if matcher.ignores(ignore_path):
            continue
        if is_directory:
            result = _load_skills_from_dir_internal(entry_path, source, False, matcher, root)
            skills.extend(result.skills)
            diagnostics.extend(result.diagnostics)
            continue
        if is_file and include_root_files and entry.name.endswith(".md"):
            result = _load_skill_from_file(entry_path, source)
            if result.skill is not None:
                skills.append(result.skill)
            diagnostics.extend(result.diagnostics)

    return LoadSkillsResult(skills=skills, diagnostics=diagnostics)


@dataclass(slots=True)
class _LoadSkillFromFileResult:
    skill: Skill | None
    diagnostics: list[ResourceDiagnostic]


def _load_skill_from_file(file_path: str, source: str) -> _LoadSkillFromFileResult:
    diagnostics: list[ResourceDiagnostic] = []
    try:
        raw_content = Path(file_path).read_text(encoding="utf-8")
    except OSError as error:
        diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=file_path))
        return _LoadSkillFromFileResult(skill=None, diagnostics=diagnostics)

    try:
        frontmatter, _body = _parse_frontmatter(raw_content)
    except Exception as error:
        diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=file_path))
        return _LoadSkillFromFileResult(skill=None, diagnostics=diagnostics)

    skill_dir = os.path.dirname(file_path)
    parent_dir_name = os.path.basename(skill_dir)
    description = _string_or_none(frontmatter.get("description"))
    for error in validate_description(description):
        diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

    name = _string_or_none(frontmatter.get("name")) or parent_dir_name
    for error in validate_name(name):
        diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

    if description is None or not description.strip():
        return _LoadSkillFromFileResult(skill=None, diagnostics=diagnostics)

    return _LoadSkillFromFileResult(
        skill=Skill(
            name=name,
            description=description,
            filePath=file_path,
            baseDir=skill_dir,
            sourceInfo=_create_skill_source_info(file_path, skill_dir, source),
            disableModelInvocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics=diagnostics,
    )


def validate_name(name: str) -> list[str]:
    errors: list[str] = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not all(char.islower() or char.isdigit() or char == "-" for char in name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if description is None or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


def _create_skill_source_info(file_path: str, base_dir: str, source: str) -> SourceInfo:
    if source == "user":
        return create_synthetic_source_info(file_path, {"source": "local", "scope": "user", "baseDir": base_dir})
    if source == "project":
        return create_synthetic_source_info(file_path, {"source": "local", "scope": "project", "baseDir": base_dir})
    if source == "path":
        return create_synthetic_source_info(file_path, {"source": "local", "baseDir": base_dir})
    return create_synthetic_source_info(file_path, {"source": source, "baseDir": base_dir})


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


def _add_ignore_rules(matcher: _IgnoreMatcher, dir_path: str, root_dir: str) -> None:
    relative_dir = os.path.relpath(dir_path, root_dir)
    prefix = f"{_to_posix_path(relative_dir)}/" if relative_dir != "." else ""
    for filename in IGNORE_FILE_NAMES:
        ignore_path = os.path.join(dir_path, filename)
        if not os.path.exists(ignore_path):
            continue
        try:
            content = Path(ignore_path).read_text(encoding="utf-8")
        except OSError:
            continue
        patterns = [
            prefixed
            for line in content.splitlines()
            if (prefixed := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            matcher.add(patterns)


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _to_posix_path(value: str) -> str:
    return value.replace(os.sep, "/")


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _yaml_load(content: str) -> Any:
    yaml = YAML(typ="safe")
    return yaml.load(content)


def _string_or_none(value: Any) -> str | None:
    return str(value) if isinstance(value, str) else None


def _default_agent_dir() -> str:
    return str(Path.home() / ".harnify" / "agent")


formatSkillsForPrompt = format_skills_for_prompt
loadSkills = load_skills
loadSkillsFromDir = load_skills_from_dir

__all__ = [
    "CONFIG_DIR_NAME",
    "IGNORE_FILE_NAMES",
    "LoadSkillsFromDirOptions",
    "LoadSkillsOptions",
    "LoadSkillsResult",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "Skill",
    "formatSkillsForPrompt",
    "format_skills_for_prompt",
    "loadSkills",
    "loadSkillsFromDir",
    "load_skills",
    "load_skills_from_dir",
    "validate_description",
    "validate_name",
]
