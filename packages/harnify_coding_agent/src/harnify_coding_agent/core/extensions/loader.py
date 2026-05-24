"""Local-file Python extension discovery and loading."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harnify_coding_agent.core.event_bus import EventBusController, create_event_bus
from harnify_coding_agent.core.exec import exec_command
from harnify_coding_agent.core.extensions.types import (
    ExecOptions,
    ExecResult,
    Extension,
    ExtensionFactory,
    ExtensionFlag,
    ExtensionRuntime,
    ExtensionShortcut,
    LoadExtensionsResult,
    PendingProviderRegistration,
    ProviderConfig,
    RegisteredCommand,
    RegisteredTool,
    ToolDefinition,
    ToolInfo,
    _LoadedExtension,
)
from harnify_coding_agent.core.source_info import SourceInfo, create_synthetic_source_info
from harnify_coding_agent.utils.paths import resolve_path

CONFIG_DIR_NAME = ".harnify"


@dataclass(slots=True)
class _RuntimeState:
    staleMessage: str | None = None


@dataclass(slots=True)
class _ExtensionAPI:
    extension: Extension
    cwd: str
    runtime: ExtensionRuntime
    events: Any

    def on(self, event: str, handler: Any) -> None:
        self.runtime.assertActive()
        self.extension.handlers.setdefault(event, []).append(handler)

    def registerTool(self, definition: ToolDefinition[Any, Any]) -> None:
        self.runtime.assertActive()
        self.extension.tools[definition.name] = RegisteredTool(
            definition=definition,
            sourceInfo=self.extension.sourceInfo,
        )
        self.runtime.refreshTools()

    def registerCommand(self, name: str, options: dict[str, Any]) -> None:
        self.runtime.assertActive()
        self.extension.commands[name] = RegisteredCommand(
            name=name,
            sourceInfo=self.extension.sourceInfo,
            description=options.get("description"),
            getArgumentCompletions=options.get("getArgumentCompletions"),
            handler=options["handler"],
        )

    def registerShortcut(self, shortcut: str, options: dict[str, Any]) -> None:
        self.runtime.assertActive()
        self.extension.shortcuts[shortcut] = ExtensionShortcut(
            shortcut=shortcut,
            extensionPath=self.extension.path,
            description=options.get("description"),
            handler=options["handler"],
        )

    def registerFlag(self, name: str, options: dict[str, Any]) -> None:
        self.runtime.assertActive()
        self.extension.flags[name] = ExtensionFlag(
            name=name,
            extensionPath=self.extension.path,
            type=options["type"],
            description=options.get("description"),
            default=options.get("default"),
        )
        if options.get("default") is not None and name not in self.runtime.flagValues:
            self.runtime.flagValues[name] = options["default"]

    def registerMessageRenderer(self, customType: str, renderer: Any) -> None:
        self.runtime.assertActive()
        self.extension.messageRenderers[customType] = renderer

    def getFlag(self, name: str) -> bool | str | None:
        self.runtime.assertActive()
        if name not in self.extension.flags:
            return None
        return self.runtime.flagValues.get(name)

    def sendMessage(self, message: Any, options: dict[str, Any] | None = None) -> None:
        self.runtime.assertActive()
        self.runtime.sendMessage(message, options)

    def sendUserMessage(
        self,
        content: str | list[Any],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.runtime.assertActive()
        self.runtime.sendUserMessage(content, options)

    def appendEntry(self, customType: str, data: Any = None) -> None:
        self.runtime.assertActive()
        self.runtime.appendEntry(customType, data)

    def setSessionName(self, name: str) -> None:
        self.runtime.assertActive()
        self.runtime.setSessionName(name)

    def getSessionName(self) -> str | None:
        self.runtime.assertActive()
        return self.runtime.getSessionName()

    def setLabel(self, entryId: str, label: str | None) -> None:
        self.runtime.assertActive()
        self.runtime.setLabel(entryId, label)

    async def exec(self, command: str, args: list[str], options: ExecOptions | None = None) -> ExecResult:
        self.runtime.assertActive()
        resolved_options: ExecOptions = dict(options or {})
        resolved_cwd = str(resolved_options.get("cwd") or self.cwd)
        return await _exec_command(command, args, resolved_cwd, resolved_options)

    def getActiveTools(self) -> list[str]:
        self.runtime.assertActive()
        return self.runtime.getActiveTools()

    def getAllTools(self) -> list[ToolInfo]:
        self.runtime.assertActive()
        return self.runtime.getAllTools()

    def setActiveTools(self, toolNames: list[str]) -> None:
        self.runtime.assertActive()
        self.runtime.setActiveTools(toolNames)

    def getCommands(self) -> list[dict[str, Any]]:
        self.runtime.assertActive()
        return self.runtime.getCommands()

    async def setModel(self, model: Any) -> bool:
        self.runtime.assertActive()
        return await self.runtime.setModel(model)

    def getThinkingLevel(self) -> str:
        self.runtime.assertActive()
        return self.runtime.getThinkingLevel()

    def setThinkingLevel(self, level: str) -> None:
        self.runtime.assertActive()
        self.runtime.setThinkingLevel(level)

    def registerProvider(self, name: str, config: ProviderConfig) -> None:
        self.runtime.assertActive()
        self.runtime.registerProvider(name, config, self.extension.path)

    def unregisterProvider(self, name: str) -> None:
        self.runtime.assertActive()
        self.runtime.unregisterProvider(name, self.extension.path)


def _not_initialized(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("Extension runtime not initialized. Action methods cannot be called during extension loading.")


async def _not_initialized_async(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("Extension runtime not initialized. Action methods cannot be called during extension loading.")


async def _set_model_not_initialized(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("Extension runtime not initialized")


def create_extension_runtime() -> ExtensionRuntime:
    state = _RuntimeState()

    def assert_active() -> None:
        if state.staleMessage:
            raise RuntimeError(state.staleMessage)

    def invalidate(message: str | None = None) -> None:
        if state.staleMessage is not None:
            return
        state.staleMessage = (
            message
            or (
                "This extension ctx is stale after session replacement or reload. "
                "Do not use a captured context after replacement."
            )
        )

    runtime = ExtensionRuntime(
        sendMessage=_not_initialized,
        sendUserMessage=_not_initialized,
        appendEntry=_not_initialized,
        setSessionName=_not_initialized,
        getSessionName=_not_initialized,
        setLabel=_not_initialized,
        getActiveTools=_not_initialized,
        getAllTools=_not_initialized,
        setActiveTools=_not_initialized,
        refreshTools=lambda: None,
        getCommands=_not_initialized,
        setModel=_set_model_not_initialized,
        getThinkingLevel=_not_initialized,
        setThinkingLevel=_not_initialized,
        flagValues={},
        pendingProviderRegistrations=[],
        assertActive=assert_active,
        invalidate=invalidate,
        registerProvider=lambda name, config, extension_path=None: runtime.pendingProviderRegistrations.append(
            PendingProviderRegistration(
                name=name,
                config=config,
                extensionPath=extension_path or "<unknown>",
            )
        ),
        unregisterProvider=lambda name, _extension_path=None: runtime.pendingProviderRegistrations.__setitem__(
            slice(None),
            [entry for entry in runtime.pendingProviderRegistrations if entry.name != name],
        ),
    )
    return runtime


def _default_event_bus() -> EventBusController:
    return create_event_bus()


async def load_extension_from_factory(
    factory: ExtensionFactory,
    cwd: str,
    event_bus: Any,
    runtime: ExtensionRuntime,
    extension_path: str = "<inline>",
) -> Extension:
    extension = _create_extension(extension_path, extension_path)
    api = _ExtensionAPI(
        extension=extension,
        cwd=resolve_path(cwd),
        runtime=runtime,
        events=event_bus,
    )
    await _invoke_factory(factory, api)
    return extension


async def load_extensions(
    paths: list[str],
    cwd: str,
    event_bus: Any | None = None,
) -> LoadExtensionsResult:
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []
    resolved_cwd = resolve_path(cwd)
    resolved_event_bus = event_bus or _default_event_bus()
    runtime = create_extension_runtime()

    for ext_path in paths:
        extension, error = await _load_extension(ext_path, resolved_cwd, resolved_event_bus, runtime)
        if error is not None:
            errors.append({"path": ext_path, "error": error})
            continue
        if extension is not None:
            extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors, runtime=runtime)


def discover_extensions_in_dir(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    discovered: list[str] = []
    try:
        for entry in os.scandir(dir_path):
            entry_path = entry.path
            if (entry.is_file() or entry.is_symlink()) and is_extension_file(entry.name):
                discovered.append(entry_path)
                continue
            if entry.is_dir() or entry.is_symlink():
                entries = resolve_extension_entries(entry_path)
                if entries:
                    discovered.extend(entries)
    except OSError:
        return []
    return discovered


async def discover_and_load_extensions(
    configured_paths: list[str],
    cwd: str,
    agent_dir: str | None = None,
    event_bus: Any | None = None,
) -> LoadExtensionsResult:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir or _default_agent_dir())
    all_paths: list[str] = []
    seen: set[str] = set()

    def add_paths(paths: list[str]) -> None:
        for candidate in paths:
            resolved = os.path.abspath(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            all_paths.append(candidate)

    add_paths(discover_extensions_in_dir(os.path.join(resolved_cwd, CONFIG_DIR_NAME, "extensions")))
    add_paths(discover_extensions_in_dir(os.path.join(resolved_agent_dir, "extensions")))

    for raw_path in configured_paths:
        resolved = resolve_path(raw_path, resolved_cwd, trim=True, normalize_unicode_spaces=True)
        if os.path.isdir(resolved):
            entries = resolve_extension_entries(resolved)
            if entries:
                add_paths(entries)
            else:
                add_paths(discover_extensions_in_dir(resolved))
            continue
        add_paths([resolved])

    return await load_extensions(all_paths, resolved_cwd, event_bus)


def resolve_extension_entries(dir_path: str) -> list[str] | None:
    package_json_path = os.path.join(dir_path, "package.json")
    if os.path.exists(package_json_path):
        manifest = _read_pi_manifest(package_json_path)
        if manifest and manifest.get("extensions"):
            entries = [
                os.path.abspath(os.path.join(dir_path, candidate))
                for candidate in manifest["extensions"]
                if os.path.exists(os.path.join(dir_path, candidate))
            ]
            if entries:
                return entries
    index_py = os.path.join(dir_path, "index.py")
    if os.path.exists(index_py):
        return [index_py]
    return None


def is_extension_file(name: str) -> bool:
    return name.endswith(".py")


def _read_pi_manifest(package_json_path: str) -> dict[str, list[str]] | None:
    try:
        package = json.loads(Path(package_json_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    pi_section = package.get("pi")
    return pi_section if isinstance(pi_section, dict) else None


async def _load_extension(
    path: str,
    cwd: str,
    event_bus: Any,
    runtime: ExtensionRuntime,
) -> tuple[Extension | None, str | None]:
    resolved_path = resolve_path(path, cwd, trim=True, normalize_unicode_spaces=True)
    try:
        factory = _load_extension_module(resolved_path)
        if factory is None:
            return None, f"Extension does not export a valid factory function: {path}"
        extension = _create_extension(path, resolved_path)
        api = _ExtensionAPI(extension=extension, cwd=cwd, runtime=runtime, events=event_bus)
        await _invoke_factory(factory, api)
        return extension, None
    except Exception as error:
        return None, f"Failed to load extension: {error}"


def _load_extension_module(resolved_path: str) -> ExtensionFactory | None:
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Extension path does not exist: {resolved_path}")
    spec = importlib.util.spec_from_file_location(f"harnify_extension_{uuid.uuid4().hex}", resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "default", None)
    return candidate if callable(candidate) else None


async def _invoke_factory(factory: ExtensionFactory, api: _ExtensionAPI) -> None:
    result = factory(api)
    if inspect.isawaitable(result):
        await result


def _create_extension(path: str, resolved_path: str) -> Extension:
    source = path[1:-1].split(":")[0] if path.startswith("<") and path.endswith(">") else "local"
    base_dir = None if path.startswith("<") else os.path.dirname(resolved_path)
    source_info = create_synthetic_source_info(path, {"source": source or "temporary", "baseDir": base_dir})
    return _LoadedExtension(path=path, resolvedPath=resolved_path, sourceInfo=source_info)


async def _exec_command(
    command: str,
    args: list[str],
    cwd: str,
    options: ExecOptions,
) -> ExecResult:
    result = await exec_command(command, args, cwd, options)
    return ExecResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exitCode=result.code,
    )


def _default_agent_dir() -> str:
    return str(Path.home() / ".harnify" / "agent")


createExtensionRuntime = create_extension_runtime
discoverAndLoadExtensions = discover_and_load_extensions
loadExtensionFromFactory = load_extension_from_factory
loadExtensions = load_extensions

__all__ = [
    "createExtensionRuntime",
    "discoverAndLoadExtensions",
    "loadExtensionFromFactory",
    "loadExtensions",
]
