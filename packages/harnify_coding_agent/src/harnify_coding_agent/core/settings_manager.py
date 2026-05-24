"""Hierarchical settings management for coding-agent runtime behavior."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from harnify_ai.types import Transport

from harnify_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir
from harnify_coding_agent.core.http_dispatcher import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    parse_http_idle_timeout_ms,
)
from harnify_coding_agent.utils.paths import normalize_path, resolve_path

type Settings = dict[str, Any]
type PackageSource = str | dict[str, Any]
type SettingsScope = Literal["global", "project"]
type TransportSetting = Transport


def deep_merge_settings(base: Settings, overrides: Settings) -> Settings:
    result = copy.deepcopy(base)
    for key, override_value in overrides.items():
        base_value = base.get(key)
        if override_value is None:
            result[key] = None
            continue
        if (
            isinstance(override_value, dict)
            and isinstance(base_value, dict)
        ):
            result[key] = {**base_value, **override_value}
        else:
            result[key] = copy.deepcopy(override_value)
    return result


@dataclass(slots=True)
class SettingsError:
    scope: SettingsScope
    error: Exception


class SettingsStorage:
    def withLock(self, scope: SettingsScope, fn: Any) -> None:  # pragma: no cover - protocol-like
        raise NotImplementedError


class FileSettingsStorage(SettingsStorage):
    def __init__(self, cwd: str, agent_dir: str):
        resolved_cwd = resolve_path(cwd)
        resolved_agent_dir = resolve_path(agent_dir)
        self.globalSettingsPath = str(Path(resolved_agent_dir) / "settings.json")
        self.projectSettingsPath = str(Path(resolved_cwd) / CONFIG_DIR_NAME / "settings.json")

    @staticmethod
    def _lock_path(path: str) -> str:
        return f"{path}.lock"

    def withLock(self, scope: SettingsScope, fn: Any) -> None:
        path = self.globalSettingsPath if scope == "global" else self.projectSettingsPath
        directory = os.path.dirname(path)
        release_lock: FileLock | None = None

        try:
            file_exists = os.path.exists(path)
            current: str | None = None

            if file_exists:
                release_lock = FileLock(self._lock_path(path), timeout=10)
                release_lock.acquire()
                with open(path, encoding="utf-8") as handle:
                    current = handle.read()

            next_value = fn(current)
            if next_value is None:
                return

            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            if release_lock is None:
                release_lock = FileLock(self._lock_path(path), timeout=10)
                release_lock.acquire()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(next_value)
        finally:
            if release_lock is not None and release_lock.is_locked:
                release_lock.release()


class InMemorySettingsStorage(SettingsStorage):
    def __init__(self) -> None:
        self.global_value: str | None = None
        self.project_value: str | None = None

    def withLock(self, scope: SettingsScope, fn: Any) -> None:
        current = self.global_value if scope == "global" else self.project_value
        next_value = fn(current)
        if next_value is None:
            return
        if scope == "global":
            self.global_value = next_value
        else:
            self.project_value = next_value


class SettingsManager:
    def __init__(
        self,
        storage: SettingsStorage,
        initialGlobal: Settings,
        initialProject: Settings,
        globalLoadError: Exception | None = None,
        projectLoadError: Exception | None = None,
        initialErrors: list[SettingsError] | None = None,
    ) -> None:
        self.storage = storage
        self.globalSettings = copy.deepcopy(initialGlobal)
        self.projectSettings = copy.deepcopy(initialProject)
        self.settings = deep_merge_settings(self.globalSettings, self.projectSettings)
        self.modifiedFields: set[str] = set()
        self.modifiedNestedFields: dict[str, set[str]] = {}
        self.modifiedProjectFields: set[str] = set()
        self.modifiedProjectNestedFields: dict[str, set[str]] = {}
        self.globalSettingsLoadError = globalLoadError
        self.projectSettingsLoadError = projectLoadError
        self.errors: list[SettingsError] = list(initialErrors or [])

    @classmethod
    def create(cls, cwd: str, agentDir: str | None = None) -> SettingsManager:
        storage = FileSettingsStorage(cwd, agentDir or get_agent_dir())
        return cls.fromStorage(storage)

    @classmethod
    def fromStorage(cls, storage: SettingsStorage) -> SettingsManager:
        global_load = cls.tryLoadFromStorage(storage, "global")
        project_load = cls.tryLoadFromStorage(storage, "project")
        initial_errors: list[SettingsError] = []
        if global_load["error"] is not None:
            initial_errors.append(SettingsError(scope="global", error=global_load["error"]))
        if project_load["error"] is not None:
            initial_errors.append(SettingsError(scope="project", error=project_load["error"]))
        return cls(
            storage,
            global_load["settings"],
            project_load["settings"],
            global_load["error"],
            project_load["error"],
            initial_errors,
        )

    @classmethod
    def inMemory(cls, settings: dict[str, Any] | None = None) -> SettingsManager:
        storage = InMemorySettingsStorage()
        initial_settings = cls.migrateSettings(copy.deepcopy(settings or {}))
        storage.withLock("global", lambda _current: json.dumps(initial_settings, indent=2))
        return cls.fromStorage(storage)

    @classmethod
    def loadFromStorage(cls, storage: SettingsStorage, scope: SettingsScope) -> Settings:
        content: str | None = None

        def capture(current: str | None) -> None:
            nonlocal content
            content = current
            return None

        storage.withLock(scope, capture)
        if not content:
            return {}
        return cls.migrateSettings(json.loads(content))

    @classmethod
    def tryLoadFromStorage(cls, storage: SettingsStorage, scope: SettingsScope) -> dict[str, Any]:
        try:
            return {"settings": cls.loadFromStorage(storage, scope), "error": None}
        except Exception as error:  # noqa: BLE001
            return {"settings": {}, "error": error}

    @classmethod
    def migrateSettings(cls, settings: dict[str, Any]) -> Settings:
        migrated = copy.deepcopy(settings)
        if "queueMode" in migrated and "steeringMode" not in migrated:
            migrated["steeringMode"] = migrated.pop("queueMode")

        if "transport" not in migrated and isinstance(migrated.get("websockets"), bool):
            migrated["transport"] = "websocket" if migrated.pop("websockets") else "sse"

        skills = migrated.get("skills")
        if isinstance(skills, dict):
            if skills.get("enableSkillCommands") is not None and migrated.get("enableSkillCommands") is None:
                migrated["enableSkillCommands"] = skills["enableSkillCommands"]
            custom_directories = skills.get("customDirectories")
            if isinstance(custom_directories, list) and custom_directories:
                migrated["skills"] = custom_directories
            else:
                migrated.pop("skills", None)

        retry_settings = migrated.get("retry")
        if isinstance(retry_settings, dict):
            provider_settings = retry_settings.get("provider")
            if not isinstance(provider_settings, dict):
                provider_settings = {}
            max_delay = retry_settings.get("maxDelayMs")
            if isinstance(max_delay, int | float) and provider_settings.get("maxRetryDelayMs") is None:
                retry_settings["provider"] = {**provider_settings, "maxRetryDelayMs": max_delay}
            retry_settings.pop("maxDelayMs", None)

        return migrated

    def getGlobalSettings(self) -> Settings:
        return copy.deepcopy(self.globalSettings)

    def getProjectSettings(self) -> Settings:
        return copy.deepcopy(self.projectSettings)

    async def reload(self) -> None:
        global_load = self.tryLoadFromStorage(self.storage, "global")
        if global_load["error"] is None:
            self.globalSettings = global_load["settings"]
            self.globalSettingsLoadError = None
        else:
            self.globalSettingsLoadError = global_load["error"]
            self.recordError("global", global_load["error"])

        self.modifiedFields.clear()
        self.modifiedNestedFields.clear()
        self.modifiedProjectFields.clear()
        self.modifiedProjectNestedFields.clear()

        project_load = self.tryLoadFromStorage(self.storage, "project")
        if project_load["error"] is None:
            self.projectSettings = project_load["settings"]
            self.projectSettingsLoadError = None
        else:
            self.projectSettingsLoadError = project_load["error"]
            self.recordError("project", project_load["error"])

        self.settings = deep_merge_settings(self.globalSettings, self.projectSettings)

    def applyOverrides(self, overrides: Settings) -> None:
        self.settings = deep_merge_settings(self.settings, overrides)

    def markModified(self, field: str, nestedKey: str | None = None) -> None:
        self.modifiedFields.add(field)
        if nestedKey is not None:
            self.modifiedNestedFields.setdefault(field, set()).add(nestedKey)

    def markProjectModified(self, field: str, nestedKey: str | None = None) -> None:
        self.modifiedProjectFields.add(field)
        if nestedKey is not None:
            self.modifiedProjectNestedFields.setdefault(field, set()).add(nestedKey)

    def recordError(self, scope: SettingsScope, error: Exception | BaseException) -> None:
        normalized = error if isinstance(error, Exception) else Exception(str(error))
        self.errors.append(SettingsError(scope=scope, error=normalized))

    @staticmethod
    def _clone_modified_nested_fields(source: dict[str, set[str]]) -> dict[str, set[str]]:
        return {key: set(value) for key, value in source.items()}

    def _persistScopedSettings(
        self,
        scope: SettingsScope,
        snapshotSettings: Settings,
        modifiedFields: set[str],
        modifiedNestedFields: dict[str, set[str]],
    ) -> None:
        def persist(current: str | None) -> str:
            current_file_settings = (
                self.migrateSettings(json.loads(current))
                if current
                else {}
            )
            merged_settings: Settings = copy.deepcopy(current_file_settings)
            for field in modifiedFields:
                value = snapshotSettings.get(field)
                if field in modifiedNestedFields and isinstance(value, dict):
                    nested_modified = modifiedNestedFields[field]
                    base_nested = copy.deepcopy(current_file_settings.get(field) or {})
                    for nested_key in nested_modified:
                        base_nested[nested_key] = value.get(nested_key)
                    merged_settings[field] = base_nested
                else:
                    if value is None:
                        merged_settings.pop(field, None)
                    else:
                        merged_settings[field] = copy.deepcopy(value)
            return json.dumps(merged_settings, indent=2)

        self.storage.withLock(scope, persist)

    def save(self) -> None:
        self.settings = deep_merge_settings(self.globalSettings, self.projectSettings)
        if self.globalSettingsLoadError is not None:
            return

        snapshot_global_settings = copy.deepcopy(self.globalSettings)
        modified_fields = set(self.modifiedFields)
        modified_nested_fields = self._clone_modified_nested_fields(self.modifiedNestedFields)
        try:
            self._persistScopedSettings("global", snapshot_global_settings, modified_fields, modified_nested_fields)
            self.modifiedFields.clear()
            self.modifiedNestedFields.clear()
        except Exception as error:  # noqa: BLE001
            self.recordError("global", error)

    def saveProjectSettings(self, settings: Settings) -> None:
        self.projectSettings = copy.deepcopy(settings)
        self.settings = deep_merge_settings(self.globalSettings, self.projectSettings)
        if self.projectSettingsLoadError is not None:
            return

        snapshot_project_settings = copy.deepcopy(self.projectSettings)
        modified_fields = set(self.modifiedProjectFields)
        modified_nested_fields = self._clone_modified_nested_fields(self.modifiedProjectNestedFields)
        try:
            self._persistScopedSettings("project", snapshot_project_settings, modified_fields, modified_nested_fields)
            self.modifiedProjectFields.clear()
            self.modifiedProjectNestedFields.clear()
        except Exception as error:  # noqa: BLE001
            self.recordError("project", error)

    async def flush(self) -> None:
        return None

    def drainErrors(self) -> list[SettingsError]:
        drained = list(self.errors)
        self.errors = []
        return drained

    def _set_global_value(self, key: str, value: Any) -> None:
        if value is None:
            self.globalSettings.pop(key, None)
        else:
            self.globalSettings[key] = value
        self.markModified(key)
        self.save()

    def _ensure_global_nested(self, key: str) -> dict[str, Any]:
        value = self.globalSettings.get(key)
        if not isinstance(value, dict):
            value = {}
            self.globalSettings[key] = value
        return value

    def getLastChangelogVersion(self) -> str | None:
        return self.settings.get("lastChangelogVersion")

    def setLastChangelogVersion(self, version: str) -> None:
        self._set_global_value("lastChangelogVersion", version)

    def getSessionDir(self) -> str | None:
        session_dir = self.settings.get("sessionDir")
        return normalize_path(session_dir) if session_dir else session_dir

    def getDefaultProvider(self) -> str | None:
        return self.settings.get("defaultProvider")

    def getDefaultModel(self) -> str | None:
        return self.settings.get("defaultModel")

    def setDefaultProvider(self, provider: str) -> None:
        self._set_global_value("defaultProvider", provider)

    def setDefaultModel(self, modelId: str) -> None:
        self._set_global_value("defaultModel", modelId)

    def setDefaultModelAndProvider(self, provider: str, modelId: str) -> None:
        self.globalSettings["defaultProvider"] = provider
        self.globalSettings["defaultModel"] = modelId
        self.markModified("defaultProvider")
        self.markModified("defaultModel")
        self.save()

    def getSteeringMode(self) -> str:
        return self.settings.get("steeringMode") or "one-at-a-time"

    def setSteeringMode(self, mode: str) -> None:
        self._set_global_value("steeringMode", mode)

    def getFollowUpMode(self) -> str:
        return self.settings.get("followUpMode") or "one-at-a-time"

    def setFollowUpMode(self, mode: str) -> None:
        self._set_global_value("followUpMode", mode)

    def getTheme(self) -> str | None:
        return self.settings.get("theme")

    def setTheme(self, theme: str) -> None:
        self._set_global_value("theme", theme)

    def getDefaultThinkingLevel(self) -> str | None:
        return self.settings.get("defaultThinkingLevel")

    def setDefaultThinkingLevel(self, level: str) -> None:
        self._set_global_value("defaultThinkingLevel", level)

    def getTransport(self) -> TransportSetting:
        return self.settings.get("transport") or "auto"

    def setTransport(self, transport: TransportSetting) -> None:
        self._set_global_value("transport", transport)

    def getCompactionEnabled(self) -> bool:
        return bool((self.settings.get("compaction") or {}).get("enabled", True))

    def setCompactionEnabled(self, enabled: bool) -> None:
        compaction = self._ensure_global_nested("compaction")
        compaction["enabled"] = enabled
        self.markModified("compaction", "enabled")
        self.save()

    def getCompactionReserveTokens(self) -> int:
        return int((self.settings.get("compaction") or {}).get("reserveTokens", 16384))

    def getCompactionKeepRecentTokens(self) -> int:
        return int((self.settings.get("compaction") or {}).get("keepRecentTokens", 20000))

    def getCompactionSettings(self) -> dict[str, Any]:
        return {
            "enabled": self.getCompactionEnabled(),
            "reserveTokens": self.getCompactionReserveTokens(),
            "keepRecentTokens": self.getCompactionKeepRecentTokens(),
        }

    def getBranchSummarySettings(self) -> dict[str, Any]:
        branch_summary = self.settings.get("branchSummary") or {}
        return {
            "reserveTokens": branch_summary.get("reserveTokens", 16384),
            "skipPrompt": branch_summary.get("skipPrompt", False),
        }

    def getBranchSummarySkipPrompt(self) -> bool:
        return bool((self.settings.get("branchSummary") or {}).get("skipPrompt", False))

    def getRetryEnabled(self) -> bool:
        return bool((self.settings.get("retry") or {}).get("enabled", True))

    def setRetryEnabled(self, enabled: bool) -> None:
        retry_settings = self._ensure_global_nested("retry")
        retry_settings["enabled"] = enabled
        self.markModified("retry", "enabled")
        self.save()

    def getRetrySettings(self) -> dict[str, Any]:
        retry_settings = self.settings.get("retry") or {}
        return {
            "enabled": self.getRetryEnabled(),
            "maxRetries": retry_settings.get("maxRetries", 3),
            "baseDelayMs": retry_settings.get("baseDelayMs", 2000),
        }

    def getHttpIdleTimeoutMs(self) -> int:
        value = self.settings.get("httpIdleTimeoutMs")
        timeout_ms = parse_http_idle_timeout_ms(value)
        if timeout_ms is not None:
            return timeout_ms
        if value is not None:
            raise ValueError(f"Invalid httpIdleTimeoutMs setting: {value}")
        return DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def setHttpIdleTimeoutMs(self, timeoutMs: int) -> None:
        if not isinstance(timeoutMs, int | float) or not float(timeoutMs).is_integer() or timeoutMs < 0:
            raise ValueError(f"Invalid httpIdleTimeoutMs setting: {timeoutMs}")
        self._set_global_value("httpIdleTimeoutMs", int(timeoutMs))

    def getProviderRetrySettings(self) -> dict[str, Any]:
        provider = ((self.settings.get("retry") or {}).get("provider") or {})
        return {
            "timeoutMs": provider.get("timeoutMs"),
            "maxRetries": provider.get("maxRetries"),
            "maxRetryDelayMs": provider.get("maxRetryDelayMs", 60000),
        }

    def getHideThinkingBlock(self) -> bool:
        return bool(self.settings.get("hideThinkingBlock", False))

    def setHideThinkingBlock(self, hide: bool) -> None:
        self._set_global_value("hideThinkingBlock", hide)

    def getShellPath(self) -> str | None:
        return self.settings.get("shellPath")

    def setShellPath(self, path: str | None) -> None:
        self._set_global_value("shellPath", path)

    def getQuietStartup(self) -> bool:
        return bool(self.settings.get("quietStartup", False))

    def setQuietStartup(self, quiet: bool) -> None:
        self._set_global_value("quietStartup", quiet)

    def getShellCommandPrefix(self) -> str | None:
        return self.settings.get("shellCommandPrefix")

    def setShellCommandPrefix(self, prefix: str | None) -> None:
        self._set_global_value("shellCommandPrefix", prefix)

    def getNpmCommand(self) -> list[str] | None:
        npm_command = self.settings.get("npmCommand")
        return list(npm_command) if isinstance(npm_command, list) else None

    def setNpmCommand(self, command: list[str] | None) -> None:
        self._set_global_value("npmCommand", list(command) if command is not None else None)

    def getCollapseChangelog(self) -> bool:
        return bool(self.settings.get("collapseChangelog", False))

    def setCollapseChangelog(self, collapse: bool) -> None:
        self._set_global_value("collapseChangelog", collapse)

    def getEnableInstallTelemetry(self) -> bool:
        return bool(self.settings.get("enableInstallTelemetry", True))

    def setEnableInstallTelemetry(self, enabled: bool) -> None:
        self._set_global_value("enableInstallTelemetry", enabled)

    def getPackages(self) -> list[PackageSource]:
        return copy.deepcopy(self.settings.get("packages") or [])

    def setPackages(self, packages: list[PackageSource]) -> None:
        self._set_global_value("packages", copy.deepcopy(packages))

    def setProjectPackages(self, packages: list[PackageSource]) -> None:
        project_settings = copy.deepcopy(self.projectSettings)
        project_settings["packages"] = copy.deepcopy(packages)
        self.markProjectModified("packages")
        self.saveProjectSettings(project_settings)

    def getExtensionPaths(self) -> list[str]:
        return list(self.settings.get("extensions") or [])

    def setExtensionPaths(self, paths: list[str]) -> None:
        self._set_global_value("extensions", list(paths))

    def setProjectExtensionPaths(self, paths: list[str]) -> None:
        project_settings = copy.deepcopy(self.projectSettings)
        project_settings["extensions"] = list(paths)
        self.markProjectModified("extensions")
        self.saveProjectSettings(project_settings)

    def getSkillPaths(self) -> list[str]:
        return list(self.settings.get("skills") or [])

    def setSkillPaths(self, paths: list[str]) -> None:
        self._set_global_value("skills", list(paths))

    def setProjectSkillPaths(self, paths: list[str]) -> None:
        project_settings = copy.deepcopy(self.projectSettings)
        project_settings["skills"] = list(paths)
        self.markProjectModified("skills")
        self.saveProjectSettings(project_settings)

    def getPromptTemplatePaths(self) -> list[str]:
        return list(self.settings.get("prompts") or [])

    def setPromptTemplatePaths(self, paths: list[str]) -> None:
        self._set_global_value("prompts", list(paths))

    def setProjectPromptTemplatePaths(self, paths: list[str]) -> None:
        project_settings = copy.deepcopy(self.projectSettings)
        project_settings["prompts"] = list(paths)
        self.markProjectModified("prompts")
        self.saveProjectSettings(project_settings)

    def getThemePaths(self) -> list[str]:
        return list(self.settings.get("themes") or [])

    def setThemePaths(self, paths: list[str]) -> None:
        self._set_global_value("themes", list(paths))

    def setProjectThemePaths(self, paths: list[str]) -> None:
        project_settings = copy.deepcopy(self.projectSettings)
        project_settings["themes"] = list(paths)
        self.markProjectModified("themes")
        self.saveProjectSettings(project_settings)

    def getEnableSkillCommands(self) -> bool:
        return bool(self.settings.get("enableSkillCommands", True))

    def setEnableSkillCommands(self, enabled: bool) -> None:
        self._set_global_value("enableSkillCommands", enabled)

    def getThinkingBudgets(self) -> dict[str, Any] | None:
        budgets = self.settings.get("thinkingBudgets")
        return copy.deepcopy(budgets) if isinstance(budgets, dict) else budgets

    def getShowImages(self) -> bool:
        return bool((self.settings.get("terminal") or {}).get("showImages", True))

    def setShowImages(self, show: bool) -> None:
        terminal = self._ensure_global_nested("terminal")
        terminal["showImages"] = show
        self.markModified("terminal", "showImages")
        self.save()

    def getImageWidthCells(self) -> int:
        width = (self.settings.get("terminal") or {}).get("imageWidthCells")
        if not isinstance(width, int | float):
            return 60
        return max(1, int(width))

    def setImageWidthCells(self, width: int) -> None:
        terminal = self._ensure_global_nested("terminal")
        terminal["imageWidthCells"] = max(1, int(width))
        self.markModified("terminal", "imageWidthCells")
        self.save()

    def getClearOnShrink(self) -> bool:
        terminal = self.settings.get("terminal") or {}
        if terminal.get("clearOnShrink") is not None:
            return bool(terminal["clearOnShrink"])
        return os.environ.get("PI_CLEAR_ON_SHRINK") == "1"

    def setClearOnShrink(self, enabled: bool) -> None:
        terminal = self._ensure_global_nested("terminal")
        terminal["clearOnShrink"] = enabled
        self.markModified("terminal", "clearOnShrink")
        self.save()

    def getShowTerminalProgress(self) -> bool:
        return bool((self.settings.get("terminal") or {}).get("showTerminalProgress", False))

    def setShowTerminalProgress(self, enabled: bool) -> None:
        terminal = self._ensure_global_nested("terminal")
        terminal["showTerminalProgress"] = enabled
        self.markModified("terminal", "showTerminalProgress")
        self.save()

    def getImageAutoResize(self) -> bool:
        return bool((self.settings.get("images") or {}).get("autoResize", True))

    def setImageAutoResize(self, enabled: bool) -> None:
        images = self._ensure_global_nested("images")
        images["autoResize"] = enabled
        self.markModified("images", "autoResize")
        self.save()

    def getBlockImages(self) -> bool:
        return bool((self.settings.get("images") or {}).get("blockImages", False))

    def setBlockImages(self, blocked: bool) -> None:
        images = self._ensure_global_nested("images")
        images["blockImages"] = blocked
        self.markModified("images", "blockImages")
        self.save()

    def getEnabledModels(self) -> list[str] | None:
        enabled_models = self.settings.get("enabledModels")
        return list(enabled_models) if isinstance(enabled_models, list) else None

    def setEnabledModels(self, patterns: list[str] | None) -> None:
        self._set_global_value("enabledModels", list(patterns) if patterns is not None else None)

    def getDoubleEscapeAction(self) -> str:
        return self.settings.get("doubleEscapeAction") or "tree"

    def setDoubleEscapeAction(self, action: str) -> None:
        self._set_global_value("doubleEscapeAction", action)

    def getTreeFilterMode(self) -> str:
        mode = self.settings.get("treeFilterMode")
        valid = {"default", "no-tools", "user-only", "labeled-only", "all"}
        return mode if mode in valid else "default"

    def setTreeFilterMode(self, mode: str) -> None:
        self._set_global_value("treeFilterMode", mode)

    def getShowHardwareCursor(self) -> bool:
        if self.settings.get("showHardwareCursor") is not None:
            return bool(self.settings["showHardwareCursor"])
        return os.environ.get("PI_HARDWARE_CURSOR") == "1"

    def setShowHardwareCursor(self, enabled: bool) -> None:
        self._set_global_value("showHardwareCursor", enabled)

    def getEditorPaddingX(self) -> int:
        return int(self.settings.get("editorPaddingX", 0))

    def setEditorPaddingX(self, padding: int) -> None:
        self._set_global_value("editorPaddingX", max(0, min(3, int(padding))))

    def getAutocompleteMaxVisible(self) -> int:
        return int(self.settings.get("autocompleteMaxVisible", 5))

    def setAutocompleteMaxVisible(self, maxVisible: int) -> None:
        self._set_global_value("autocompleteMaxVisible", max(3, min(20, int(maxVisible))))

    def getCodeBlockIndent(self) -> str:
        return str((self.settings.get("markdown") or {}).get("codeBlockIndent", "  "))

    def getWarnings(self) -> dict[str, Any]:
        return copy.deepcopy(self.settings.get("warnings") or {})

    def setWarnings(self, warnings: dict[str, Any]) -> None:
        self._set_global_value("warnings", copy.deepcopy(warnings))


deepMergeSettings = deep_merge_settings

__all__ = [
    "FileSettingsStorage",
    "InMemorySettingsStorage",
    "PackageSource",
    "Settings",
    "SettingsError",
    "SettingsManager",
    "SettingsScope",
    "TransportSetting",
    "deepMergeSettings",
    "deep_merge_settings",
]
