"""Resource loading for prompts, skills, context files, and extensions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

from harnify_coding_agent.core.diagnostics import ResourceCollision, ResourceDiagnostic
from harnify_coding_agent.core.event_bus import create_event_bus
from harnify_coding_agent.core.extensions.loader import (
    create_extension_runtime,
    load_extension_from_factory,
    load_extensions,
)
from harnify_coding_agent.core.extensions.types import (
    Extension,
    ExtensionFactory,
    ExtensionRuntime,
    LoadExtensionsResult,
)
from harnify_coding_agent.core.package_manager import DefaultPackageManager, ResolvedResource
from harnify_coding_agent.core.prompt_templates import CONFIG_DIR_NAME, PromptTemplate, load_prompt_templates
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.core.skills import LoadSkillsResult, Skill, load_skills
from harnify_coding_agent.core.source_info import PathMetadata, SourceInfo, create_source_info
from harnify_coding_agent.utils.paths import canonicalize_path, resolve_path


@dataclass(slots=True)
class ThemeResource:
    name: str
    data: dict[str, Any]
    sourcePath: str
    sourceInfo: SourceInfo | None = None


class ResourcePathEntry(TypedDict, total=False):
    path: str
    metadata: PathMetadata


class ResourceExtensionPaths(TypedDict, total=False):
    skillPaths: list[ResourcePathEntry]
    promptPaths: list[ResourcePathEntry]
    themePaths: list[ResourcePathEntry]


class DefaultResourceLoaderOptions(TypedDict, total=False):
    cwd: str
    agentDir: str
    settingsManager: SettingsManager
    eventBus: Any
    additionalExtensionPaths: list[str]
    additionalSkillPaths: list[str]
    additionalPromptTemplatePaths: list[str]
    additionalThemePaths: list[str]
    extensionFactories: list[ExtensionFactory]
    noExtensions: bool
    noSkills: bool
    noPromptTemplates: bool
    noThemes: bool
    noContextFiles: bool
    systemPrompt: str
    appendSystemPrompt: list[str]
    extensionsOverride: Any
    skillsOverride: Any
    promptsOverride: Any
    themesOverride: Any
    agentsFilesOverride: Any
    systemPromptOverride: Any
    appendSystemPromptOverride: Any


class ResourceLoaderLike(Protocol):
    def getExtensions(self) -> LoadExtensionsResult: ...

    def getSkills(self) -> dict[str, object]: ...

    def getPrompts(self) -> dict[str, object]: ...

    def getThemes(self) -> dict[str, object]: ...

    def getAgentsFiles(self) -> dict[str, object]: ...

    def getSystemPrompt(self) -> str | None: ...

    def getAppendSystemPrompt(self) -> list[str]: ...

    def extendResources(self, paths: ResourceExtensionPaths) -> None: ...

    async def reload(self) -> None: ...


def resolve_prompt_input(input_value: str | None, _description: str) -> str | None:
    if not input_value:
        return None
    if os.path.exists(input_value):
        try:
            return Path(input_value).read_text(encoding="utf-8")
        except OSError:
            return input_value
    return input_value


def load_project_context_files(options: dict[str, str]) -> list[dict[str, str]]:
    resolved_cwd = resolve_path(options["cwd"])
    resolved_agent_dir = resolve_path(options["agentDir"])
    context_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen_paths.add(global_context["path"])

    ancestor_context_files: list[dict[str, str]] = []
    current_dir = resolved_cwd
    root = os.path.abspath(os.sep)
    while True:
        context_file = _load_context_file_from_dir(current_dir)
        if context_file is not None and context_file["path"] not in seen_paths:
            ancestor_context_files.insert(0, context_file)
            seen_paths.add(context_file["path"])
        if current_dir == root:
            break
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    context_files.extend(ancestor_context_files)
    return context_files


@dataclass(slots=True)
class DefaultResourceLoader:
    cwd: str
    agentDir: str
    settingsManager: SettingsManager
    packageManager: DefaultPackageManager
    eventBus: Any | None = None
    additionalExtensionPaths: list[str] = field(default_factory=list)
    additionalSkillPaths: list[str] = field(default_factory=list)
    additionalPromptTemplatePaths: list[str] = field(default_factory=list)
    additionalThemePaths: list[str] = field(default_factory=list)
    extensionFactories: list[ExtensionFactory] = field(default_factory=list)
    noExtensions: bool = False
    noSkills: bool = False
    noPromptTemplates: bool = False
    noThemes: bool = False
    noContextFiles: bool = False
    systemPromptSource: str | None = None
    appendSystemPromptSource: list[str] = field(default_factory=list)
    extensionsOverride: Any = None
    skillsOverride: Any = None
    promptsOverride: Any = None
    themesOverride: Any = None
    agentsFilesOverride: Any = None
    systemPromptOverride: Any = None
    appendSystemPromptOverride: Any = None
    extensionsResult: LoadExtensionsResult = field(
        default_factory=lambda: LoadExtensionsResult(extensions=[], errors=[], runtime=create_extension_runtime())
    )
    skills: list[Skill] = field(default_factory=list)
    skillDiagnostics: list[ResourceDiagnostic] = field(default_factory=list)
    prompts: list[PromptTemplate] = field(default_factory=list)
    promptDiagnostics: list[ResourceDiagnostic] = field(default_factory=list)
    themes: list[ThemeResource] = field(default_factory=list)
    themeDiagnostics: list[ResourceDiagnostic] = field(default_factory=list)
    agentsFiles: list[dict[str, str]] = field(default_factory=list)
    systemPrompt: str | None = None
    appendSystemPrompt: list[str] = field(default_factory=list)
    lastSkillPaths: list[str] = field(default_factory=list)
    lastPromptPaths: list[str] = field(default_factory=list)
    lastThemePaths: list[str] = field(default_factory=list)
    extensionSkillSourceInfos: dict[str, SourceInfo] = field(default_factory=dict)
    extensionPromptSourceInfos: dict[str, SourceInfo] = field(default_factory=dict)
    extensionThemeSourceInfos: dict[str, SourceInfo] = field(default_factory=dict)

    def __init__(self, options: DefaultResourceLoaderOptions) -> None:
        self.cwd = resolve_path(options["cwd"])
        self.agentDir = resolve_path(options.get("agentDir") or _default_agent_dir())
        self.settingsManager = options.get("settingsManager") or SettingsManager.create(self.cwd, self.agentDir)
        self.packageManager = DefaultPackageManager(
            {"cwd": self.cwd, "agentDir": self.agentDir, "settingsManager": self.settingsManager}
        )
        self.eventBus = options.get("eventBus") or create_event_bus()
        self.additionalExtensionPaths = list(options.get("additionalExtensionPaths", []))
        self.additionalSkillPaths = list(options.get("additionalSkillPaths", []))
        self.additionalPromptTemplatePaths = list(options.get("additionalPromptTemplatePaths", []))
        self.additionalThemePaths = list(options.get("additionalThemePaths", []))
        self.extensionFactories = list(options.get("extensionFactories", []))
        self.noExtensions = bool(options.get("noExtensions", False))
        self.noSkills = bool(options.get("noSkills", False))
        self.noPromptTemplates = bool(options.get("noPromptTemplates", False))
        self.noThemes = bool(options.get("noThemes", False))
        self.noContextFiles = bool(options.get("noContextFiles", False))
        self.systemPromptSource = options.get("systemPrompt")
        self.appendSystemPromptSource = list(options.get("appendSystemPrompt", []))
        self.extensionsOverride = options.get("extensionsOverride")
        self.skillsOverride = options.get("skillsOverride")
        self.promptsOverride = options.get("promptsOverride")
        self.themesOverride = options.get("themesOverride")
        self.agentsFilesOverride = options.get("agentsFilesOverride")
        self.systemPromptOverride = options.get("systemPromptOverride")
        self.appendSystemPromptOverride = options.get("appendSystemPromptOverride")
        self.extensionsResult = LoadExtensionsResult(extensions=[], errors=[], runtime=create_extension_runtime())
        self.skills = []
        self.skillDiagnostics = []
        self.prompts = []
        self.promptDiagnostics = []
        self.themes = []
        self.themeDiagnostics = []
        self.agentsFiles = []
        self.systemPrompt = None
        self.appendSystemPrompt = []
        self.lastSkillPaths = []
        self.lastPromptPaths = []
        self.lastThemePaths = []
        self.extensionSkillSourceInfos = {}
        self.extensionPromptSourceInfos = {}
        self.extensionThemeSourceInfos = {}

    def getExtensions(self) -> LoadExtensionsResult:
        return self.extensionsResult

    def getSkills(self) -> dict[str, object]:
        return {"skills": self.skills, "diagnostics": self.skillDiagnostics}

    def getPrompts(self) -> dict[str, object]:
        return {"prompts": self.prompts, "diagnostics": self.promptDiagnostics}

    def getThemes(self) -> dict[str, object]:
        return {"themes": self.themes, "diagnostics": self.themeDiagnostics}

    def getAgentsFiles(self) -> dict[str, object]:
        return {"agentsFiles": self.agentsFiles}

    def getSystemPrompt(self) -> str | None:
        return self.systemPrompt

    def getAppendSystemPrompt(self) -> list[str]:
        return list(self.appendSystemPrompt)

    def extendResources(self, paths: ResourceExtensionPaths) -> None:
        skill_paths = self._normalize_extension_paths(paths.get("skillPaths", []))
        prompt_paths = self._normalize_extension_paths(paths.get("promptPaths", []))
        theme_paths = self._normalize_extension_paths(paths.get("themePaths", []))

        for entry in skill_paths:
            self.extensionSkillSourceInfos[entry["path"]] = create_source_info(entry["path"], entry["metadata"])
        for entry in prompt_paths:
            self.extensionPromptSourceInfos[entry["path"]] = create_source_info(entry["path"], entry["metadata"])
        for entry in theme_paths:
            self.extensionThemeSourceInfos[entry["path"]] = create_source_info(entry["path"], entry["metadata"])

        if skill_paths:
            self.lastSkillPaths = self._merge_paths(self.lastSkillPaths, [entry["path"] for entry in skill_paths])
            self.additionalSkillPaths = self._merge_paths(
                self.additionalSkillPaths,
                [entry["path"] for entry in skill_paths],
            )
            self._update_skills_from_paths(self.lastSkillPaths)

        if prompt_paths:
            self.lastPromptPaths = self._merge_paths(self.lastPromptPaths, [entry["path"] for entry in prompt_paths])
            self.additionalPromptTemplatePaths = self._merge_paths(
                self.additionalPromptTemplatePaths,
                [entry["path"] for entry in prompt_paths],
            )
            self._update_prompts_from_paths(self.lastPromptPaths)

        if theme_paths:
            self.lastThemePaths = self._merge_paths(self.lastThemePaths, [entry["path"] for entry in theme_paths])
            self.additionalThemePaths = self._merge_paths(
                self.additionalThemePaths,
                [entry["path"] for entry in theme_paths],
            )
            self._update_themes_from_paths(self.lastThemePaths)

    async def reload(self) -> None:
        await self.settingsManager.reload()
        resolved_paths = await self.packageManager.resolve()
        cli_extension_paths = await self.packageManager.resolveExtensionSources(
            self.additionalExtensionPaths,
            {"temporary": True},
        )

        metadata_by_path: dict[str, PathMetadata] = {}
        self.extensionSkillSourceInfos = {}
        self.extensionPromptSourceInfos = {}
        self.extensionThemeSourceInfos = {}

        def remember_enabled(resources: list[ResolvedResource]) -> list[ResolvedResource]:
            for resource in resources:
                metadata_by_path.setdefault(resource.path, resource.metadata)
            return [resource for resource in resources if resource.enabled]

        def enabled_paths(resources: list[ResolvedResource]) -> list[str]:
            return [resource.path for resource in remember_enabled(resources)]

        enabled_extensions = enabled_paths(resolved_paths.extensions)
        enabled_skill_resources = remember_enabled(resolved_paths.skills)
        enabled_prompts = enabled_paths(resolved_paths.prompts)
        enabled_themes = enabled_paths(resolved_paths.themes)

        def map_skill_path(resource: ResolvedResource) -> str:
            if resource.metadata.get("source") not in {"auto", "cli"} and resource.metadata.get("origin") != "package":
                return resource.path
            try:
                if not os.path.isdir(resource.path):
                    return resource.path
            except OSError:
                return resource.path
            skill_file = os.path.join(resource.path, "SKILL.md")
            if os.path.exists(skill_file):
                metadata_by_path.setdefault(skill_file, resource.metadata)
                return skill_file
            return resource.path

        enabled_skills = [map_skill_path(resource) for resource in enabled_skill_resources]

        for resource in cli_extension_paths.extensions:
            metadata_by_path.setdefault(
                resource.path,
                resource.metadata or {"source": "cli", "scope": "temporary", "origin": "top-level"},
            )
        for resource in cli_extension_paths.skills:
            metadata_by_path.setdefault(
                resource.path,
                resource.metadata or {"source": "cli", "scope": "temporary", "origin": "top-level"},
            )
        for resource in cli_extension_paths.prompts:
            metadata_by_path.setdefault(
                resource.path,
                resource.metadata or {"source": "cli", "scope": "temporary", "origin": "top-level"},
            )
        for resource in cli_extension_paths.themes:
            metadata_by_path.setdefault(
                resource.path,
                resource.metadata or {"source": "cli", "scope": "temporary", "origin": "top-level"},
            )

        cli_enabled_extensions = [resource.path for resource in cli_extension_paths.extensions if resource.enabled]
        cli_enabled_skills = [map_skill_path(resource) for resource in cli_extension_paths.skills if resource.enabled]
        cli_enabled_prompts = [resource.path for resource in cli_extension_paths.prompts if resource.enabled]
        cli_enabled_themes = [resource.path for resource in cli_extension_paths.themes if resource.enabled]

        extension_paths = (
            cli_enabled_extensions
            if self.noExtensions
            else self._merge_paths(cli_enabled_extensions, enabled_extensions)
        )

        if self.noExtensions and not extension_paths:
            extensions_result = LoadExtensionsResult(extensions=[], errors=[], runtime=create_extension_runtime())
        else:
            extensions_result = await load_extensions(
                extension_paths,
                self.cwd,
                self.eventBus,
            )
            inline_extensions = await self._load_extension_factories(extensions_result.runtime)
            extensions_result.extensions.extend(inline_extensions["extensions"])
            extensions_result.errors.extend(inline_extensions["errors"])
            extensions_result.errors.extend(self._detect_extension_conflicts(extensions_result.extensions))
            for raw_path in self.additionalExtensionPaths:
                if not os.path.exists(self._resolve_resource_path(raw_path)):
                    extensions_result.errors.append(
                        {
                            "path": self._resolve_resource_path(raw_path),
                            "error": f"Extension path does not exist: {self._resolve_resource_path(raw_path)}",
                        }
                    )

        metadata_source_infos = {
            path: create_source_info(path, metadata) for path, metadata in metadata_by_path.items()
        }

        self._apply_extension_source_info(extensions_result.extensions, metadata_source_infos)
        self.extensionsResult = (
            self.extensionsOverride(extensions_result)
            if callable(self.extensionsOverride)
            else extensions_result
        )

        self.extensionSkillSourceInfos = {}
        self.extensionPromptSourceInfos = {}
        self.extensionThemeSourceInfos = {}
        for extension in self.extensionsResult.extensions:
            for path in extension.skillPaths:
                self.extensionSkillSourceInfos[path] = extension.sourceInfo
            for path in extension.promptPaths:
                self.extensionPromptSourceInfos[path] = extension.sourceInfo
            for path in extension.themePaths:
                self.extensionThemeSourceInfos[path] = extension.sourceInfo

        extension_skill_paths = [
            path for extension in self.extensionsResult.extensions for path in extension.skillPaths
        ]
        extension_prompt_paths = [
            path for extension in self.extensionsResult.extensions for path in extension.promptPaths
        ]
        extension_theme_paths = [
            path for extension in self.extensionsResult.extensions for path in extension.themePaths
        ]

        skill_paths = (
            self._merge_paths(cli_enabled_skills, self.additionalSkillPaths)
            if self.noSkills
            else self._merge_paths([*cli_enabled_skills, *enabled_skills], self.additionalSkillPaths)
        )
        prompt_paths = (
            self._merge_paths(cli_enabled_prompts, self.additionalPromptTemplatePaths)
            if self.noPromptTemplates
            else self._merge_paths([*cli_enabled_prompts, *enabled_prompts], self.additionalPromptTemplatePaths)
        )
        theme_paths = (
            self._merge_paths(cli_enabled_themes, self.additionalThemePaths)
            if self.noThemes
            else self._merge_paths([*cli_enabled_themes, *enabled_themes], self.additionalThemePaths)
        )

        self.lastSkillPaths = self._merge_paths(skill_paths, extension_skill_paths)
        self.lastPromptPaths = self._merge_paths(extension_prompt_paths, prompt_paths)
        self.lastThemePaths = self._merge_paths(extension_theme_paths, theme_paths)

        self._update_skills_from_paths(self.lastSkillPaths, metadata_source_infos)
        self._update_prompts_from_paths(self.lastPromptPaths, metadata_source_infos)
        self._update_themes_from_paths(self.lastThemePaths, metadata_source_infos)

        agents_files = {
            "agentsFiles": (
                [] if self.noContextFiles else load_project_context_files({"cwd": self.cwd, "agentDir": self.agentDir})
            )
        }
        resolved_agents_files = (
            self.agentsFilesOverride(agents_files) if callable(self.agentsFilesOverride) else agents_files
        )
        self.agentsFiles = resolved_agents_files["agentsFiles"]

        base_system_prompt = resolve_prompt_input(
            self.systemPromptSource or self._discover_system_prompt_file(),
            "system prompt",
        )
        self.systemPrompt = (
            self.systemPromptOverride(base_system_prompt)
            if callable(self.systemPromptOverride)
            else base_system_prompt
        )

        append_sources = self.appendSystemPromptSource or (
            [self._discover_append_system_prompt_file()] if self._discover_append_system_prompt_file() else []
        )
        base_append = [
            content
            for source in append_sources
            if (content := resolve_prompt_input(source, "append system prompt")) is not None
        ]
        self.appendSystemPrompt = (
            self.appendSystemPromptOverride(base_append)
            if callable(self.appendSystemPromptOverride)
            else base_append
        )

    def _normalize_extension_paths(self, entries: list[ResourcePathEntry]) -> list[ResourcePathEntry]:
        normalized: list[ResourcePathEntry] = []
        for entry in entries:
            metadata = dict(entry.get("metadata", {}))
            if metadata.get("baseDir") is not None:
                metadata["baseDir"] = self._resolve_resource_path(str(metadata["baseDir"]))
            normalized.append(
                {
                    "path": self._resolve_resource_path(entry["path"]),
                    "metadata": metadata,  # type: ignore[typeddict-item]
                }
            )
        return normalized

    def _update_skills_from_paths(
        self,
        skill_paths: list[str],
        metadata_source_infos: dict[str, SourceInfo] | None = None,
    ) -> None:
        if self.noSkills and not skill_paths:
            skills_result = LoadSkillsResult(skills=[], diagnostics=[])
        else:
            skills_result = load_skills(
                {
                    "cwd": self.cwd,
                    "agentDir": self.agentDir,
                    "skillPaths": skill_paths,
                    "includeDefaults": False,
                }
            )

        resolved = self.skillsOverride(skills_result) if callable(self.skillsOverride) else skills_result
        combined_source_infos = {
            **(metadata_source_infos or {}),
            **self.extensionSkillSourceInfos,
        }
        self.skills = [
            Skill(
                name=skill.name,
                description=skill.description,
                filePath=skill.filePath,
                baseDir=skill.baseDir,
                sourceInfo=self._find_source_info_for_path(skill.filePath, combined_source_infos)
                or skill.sourceInfo
                or self._get_default_source_info_for_path(skill.filePath),
                disableModelInvocation=skill.disableModelInvocation,
            )
            for skill in resolved.skills
        ]
        self.skillDiagnostics = list(resolved.diagnostics)
        for path in skill_paths:
            resolved_path = self._resolve_resource_path(path)
            if not os.path.exists(resolved_path) and not any(
                diagnostic.path == resolved_path for diagnostic in self.skillDiagnostics
            ):
                self.skillDiagnostics.append(
                    ResourceDiagnostic(type="error", message="Skill path does not exist", path=resolved_path)
                )

    def _update_prompts_from_paths(
        self,
        prompt_paths: list[str],
        metadata_source_infos: dict[str, SourceInfo] | None = None,
    ) -> None:
        if self.noPromptTemplates and not prompt_paths:
            prompts_result = {"prompts": [], "diagnostics": []}
        else:
            prompts_result = self._dedupe_prompts(
                load_prompt_templates(
                    {
                        "cwd": self.cwd,
                        "agentDir": self.agentDir,
                        "promptPaths": prompt_paths,
                        "includeDefaults": False,
                    }
                )
            )

        resolved = self.promptsOverride(prompts_result) if callable(self.promptsOverride) else prompts_result
        combined_source_infos = {
            **(metadata_source_infos or {}),
            **self.extensionPromptSourceInfos,
        }
        self.prompts = [
            PromptTemplate(
                name=prompt.name,
                description=prompt.description,
                content=prompt.content,
                sourceInfo=self._find_source_info_for_path(prompt.filePath, combined_source_infos)
                or prompt.sourceInfo
                or self._get_default_source_info_for_path(prompt.filePath),
                filePath=prompt.filePath,
                argumentHint=prompt.argumentHint,
            )
            for prompt in resolved["prompts"]
        ]
        self.promptDiagnostics = list(resolved["diagnostics"])
        for path in prompt_paths:
            resolved_path = self._resolve_resource_path(path)
            if not os.path.exists(resolved_path) and not any(
                diagnostic.path == resolved_path for diagnostic in self.promptDiagnostics
            ):
                self.promptDiagnostics.append(
                    ResourceDiagnostic(
                        type="error",
                        message="Prompt template path does not exist",
                        path=resolved_path,
                    )
                )

    def _update_themes_from_paths(
        self,
        theme_paths: list[str],
        metadata_source_infos: dict[str, SourceInfo] | None = None,
    ) -> None:
        if self.noThemes and not theme_paths:
            themes_result = {"themes": [], "diagnostics": []}
        else:
            loaded = self._load_themes(theme_paths)
            deduped = self._dedupe_themes(loaded["themes"])
            themes_result = {
                "themes": deduped["themes"],
                "diagnostics": [*loaded["diagnostics"], *deduped["diagnostics"]],
            }

        resolved = self.themesOverride(themes_result) if callable(self.themesOverride) else themes_result
        combined_source_infos = {
            **(metadata_source_infos or {}),
            **self.extensionThemeSourceInfos,
        }
        self.themes = [
            ThemeResource(
                name=theme.name,
                data=theme.data,
                sourcePath=theme.sourcePath,
                sourceInfo=self._find_source_info_for_path(theme.sourcePath, combined_source_infos)
                or theme.sourceInfo
                or self._get_default_source_info_for_path(theme.sourcePath),
            )
            for theme in resolved["themes"]
        ]
        self.themeDiagnostics = list(resolved["diagnostics"])
        for path in theme_paths:
            resolved_path = self._resolve_resource_path(path)
            if not os.path.exists(resolved_path) and not any(
                diagnostic.path == resolved_path for diagnostic in self.themeDiagnostics
            ):
                self.themeDiagnostics.append(
                    ResourceDiagnostic(type="error", message="Theme path does not exist", path=resolved_path)
                )

    def _apply_extension_source_info(
        self,
        extensions: list[Extension],
        metadata_source_infos: dict[str, SourceInfo] | None = None,
    ) -> None:
        for extension in extensions:
            extension.sourceInfo = (
                self._find_source_info_for_path(extension.path, metadata_source_infos)
                or self._get_default_source_info_for_path(extension.path)
            )
            for command in extension.commands.values():
                command.sourceInfo = extension.sourceInfo
            for tool in extension.tools.values():
                tool.sourceInfo = extension.sourceInfo

    def _find_source_info_for_path(
        self,
        resource_path: str,
        extra_source_infos: dict[str, SourceInfo] | None = None,
    ) -> SourceInfo | None:
        if not resource_path:
            return None
        if resource_path.startswith("<"):
            return self._get_default_source_info_for_path(resource_path)

        normalized_resource_path = os.path.abspath(resource_path)
        for source_path, source_info in (extra_source_infos or {}).items():
            normalized_source_path = os.path.abspath(source_path)
            if normalized_resource_path == normalized_source_path or normalized_resource_path.startswith(
                f"{normalized_source_path}{os.sep}"
            ):
                return SourceInfo(
                    path=resource_path,
                    source=source_info.source,
                    scope=source_info.scope,
                    origin=source_info.origin,
                    baseDir=source_info.baseDir,
                )
        return None

    def _get_default_source_info_for_path(self, file_path: str) -> SourceInfo:
        if file_path.startswith("<") and file_path.endswith(">"):
            source = file_path[1:-1].split(":")[0] or "temporary"
            return SourceInfo(path=file_path, source=source, scope="temporary", origin="top-level", baseDir=None)

        normalized_path = os.path.abspath(file_path)
        agent_roots = [
            os.path.join(self.agentDir, "skills"),
            os.path.join(self.agentDir, "prompts"),
            os.path.join(self.agentDir, "themes"),
            os.path.join(self.agentDir, "extensions"),
        ]
        project_roots = [
            os.path.join(self.cwd, CONFIG_DIR_NAME, "skills"),
            os.path.join(self.cwd, CONFIG_DIR_NAME, "prompts"),
            os.path.join(self.cwd, CONFIG_DIR_NAME, "themes"),
            os.path.join(self.cwd, CONFIG_DIR_NAME, "extensions"),
        ]

        for root in agent_roots:
            if self._is_under_path(normalized_path, root):
                return SourceInfo(path=file_path, source="local", scope="user", origin="top-level", baseDir=root)

        for root in project_roots:
            if self._is_under_path(normalized_path, root):
                return SourceInfo(path=file_path, source="local", scope="project", origin="top-level", baseDir=root)

        base_dir = normalized_path if os.path.isdir(normalized_path) else os.path.dirname(normalized_path)
        return SourceInfo(
            path=file_path,
            source="local",
            scope="temporary",
            origin="top-level",
            baseDir=base_dir,
        )

    def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in [*primary, *additional]:
            resolved = self._resolve_resource_path(value)
            canonical = canonicalize_path(resolved)
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(resolved)
        return merged

    def _resolve_resource_path(self, path: str) -> str:
        return resolve_path(path, self.cwd, trim=True)

    def _load_themes(self, paths: list[str]) -> dict[str, list[Any]]:
        themes: list[ThemeResource] = []
        diagnostics: list[ResourceDiagnostic] = []

        for path in paths:
            resolved = self._resolve_resource_path(path)
            if not os.path.exists(resolved):
                diagnostics.append(
                    ResourceDiagnostic(type="warning", message="theme path does not exist", path=resolved)
                )
                continue
            try:
                if os.path.isdir(resolved):
                    self._load_themes_from_dir(resolved, themes, diagnostics)
                elif os.path.isfile(resolved) and resolved.endswith(".json"):
                    self._load_theme_from_file(resolved, themes, diagnostics)
                else:
                    diagnostics.append(
                        ResourceDiagnostic(type="warning", message="theme path is not a json file", path=resolved)
                    )
            except OSError as error:
                diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=resolved))
        return {"themes": themes, "diagnostics": diagnostics}

    def _load_themes_from_dir(
        self,
        dir_path: str,
        themes: list[ThemeResource],
        diagnostics: list[ResourceDiagnostic],
    ) -> None:
        if not os.path.isdir(dir_path):
            return
        try:
            for entry in sorted(Path(dir_path).iterdir(), key=lambda candidate: candidate.name):
                entry_path = str(entry)
                is_file = entry.is_file()
                if entry.is_symlink():
                    try:
                        is_file = Path(entry_path).is_file()
                    except OSError:
                        continue
                if is_file and entry.name.endswith(".json"):
                    self._load_theme_from_file(entry_path, themes, diagnostics)
        except OSError as error:
            diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=dir_path))

    def _load_theme_from_file(
        self,
        file_path: str,
        themes: list[ThemeResource],
        diagnostics: list[ResourceDiagnostic],
    ) -> None:
        try:
            payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("theme file must contain a JSON object")
            name = str(payload.get("name") or Path(file_path).stem)
            themes.append(
                ThemeResource(
                    name=name,
                    data=payload,
                    sourcePath=file_path,
                    sourceInfo=self._get_default_source_info_for_path(file_path),
                )
            )
        except Exception as error:
            diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=file_path))

    async def _load_extension_factories(self, runtime: ExtensionRuntime) -> dict[str, list[Any]]:
        extensions: list[Extension] = []
        errors: list[dict[str, str]] = []
        for index, factory in enumerate(self.extensionFactories, start=1):
            extension_path = f"<inline:{index}>"
            try:
                extension = await load_extension_from_factory(
                    factory,
                    self.cwd,
                    self.eventBus,
                    runtime,
                    extension_path,
                )
                extensions.append(extension)
            except Exception as error:
                errors.append({"path": extension_path, "error": str(error)})
        return {"extensions": extensions, "errors": errors}

    def _dedupe_prompts(self, prompts: list[PromptTemplate]) -> dict[str, list[Any]]:
        seen: dict[str, PromptTemplate] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for prompt in prompts:
            existing = seen.get(prompt.name)
            if existing is None:
                seen[prompt.name] = prompt
                continue
            diagnostics.append(
                ResourceDiagnostic(
                    type="collision",
                    message=f'name "/{prompt.name}" collision',
                    path=prompt.filePath,
                    collision=ResourceCollision(
                        resourceType="prompt",
                        name=prompt.name,
                        winnerPath=existing.filePath,
                        loserPath=prompt.filePath,
                    ),
                )
            )
        return {"prompts": list(seen.values()), "diagnostics": diagnostics}

    def _dedupe_themes(self, themes: list[ThemeResource]) -> dict[str, list[Any]]:
        seen: dict[str, ThemeResource] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for theme in themes:
            existing = seen.get(theme.name)
            if existing is None:
                seen[theme.name] = theme
                continue
            diagnostics.append(
                ResourceDiagnostic(
                    type="collision",
                    message=f'name "{theme.name}" collision',
                    path=theme.sourcePath,
                    collision=ResourceCollision(
                        resourceType="theme",
                        name=theme.name,
                        winnerPath=existing.sourcePath,
                        loserPath=theme.sourcePath,
                    ),
                )
            )
        return {"themes": list(seen.values()), "diagnostics": diagnostics}

    def _discover_system_prompt_file(self) -> str | None:
        project_path = os.path.join(self.cwd, CONFIG_DIR_NAME, "SYSTEM.md")
        if os.path.exists(project_path):
            return project_path
        global_path = os.path.join(self.agentDir, "SYSTEM.md")
        if os.path.exists(global_path):
            return global_path
        return None

    def _discover_append_system_prompt_file(self) -> str | None:
        project_path = os.path.join(self.cwd, CONFIG_DIR_NAME, "APPEND_SYSTEM.md")
        if os.path.exists(project_path):
            return project_path
        global_path = os.path.join(self.agentDir, "APPEND_SYSTEM.md")
        if os.path.exists(global_path):
            return global_path
        return None

    def _is_under_path(self, target: str, root: str) -> bool:
        normalized_root = os.path.abspath(root)
        normalized_target = os.path.abspath(target)
        if normalized_target == normalized_root:
            return True
        prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
        return normalized_target.startswith(prefix)

    def _detect_extension_conflicts(self, extensions: list[Extension]) -> list[dict[str, str]]:
        conflicts: list[dict[str, str]] = []
        tool_owners: dict[str, str] = {}
        flag_owners: dict[str, str] = {}
        for extension in extensions:
            for tool_name in extension.tools:
                existing_owner = tool_owners.get(tool_name)
                if existing_owner is not None and existing_owner != extension.path:
                    conflicts.append(
                        {"path": extension.path, "error": f'Tool "{tool_name}" conflicts with {existing_owner}'}
                    )
                else:
                    tool_owners[tool_name] = extension.path
            for flag_name in extension.flags:
                existing_owner = flag_owners.get(flag_name)
                if existing_owner is not None and existing_owner != extension.path:
                    conflicts.append(
                        {"path": extension.path, "error": f'Flag "--{flag_name}" conflicts with {existing_owner}'}
                    )
                else:
                    flag_owners[flag_name] = extension.path
        return conflicts


def _load_context_file_from_dir(dir_path: str) -> dict[str, str] | None:
    for filename in ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"):
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            continue
        try:
            return {"path": file_path, "content": Path(file_path).read_text(encoding="utf-8")}
        except OSError:
            continue
    return None


def _default_agent_dir() -> str:
    return str(Path.home() / ".harnify" / "agent")


DefaultResourceLoaderOptions = DefaultResourceLoaderOptions

__all__ = [
    "DefaultResourceLoader",
    "DefaultResourceLoaderOptions",
    "ResourceExtensionPaths",
    "ResourceLoaderLike",
    "ThemeResource",
    "load_project_context_files",
    "resolve_prompt_input",
]
