"""Cwd-bound session service construction for coding-agent runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from harnify_agent.types import ThinkingLevel
from harnify_ai.types import Model

from harnify_coding_agent.config import get_agent_dir
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.utils.paths import resolve_path

create_agent_session = None


class SessionManagerLike(Protocol):
    def getCwd(self) -> str: ...


class ResourceLoaderLike(Protocol):
    async def reload(self) -> None: ...

    def getExtensions(self) -> Any: ...


@dataclass(slots=True)
class AgentSessionRuntimeDiagnostic:
    type: str
    message: str


class CreateAgentSessionServicesOptions(TypedDict, total=False):
    cwd: str
    agentDir: str
    authStorage: AuthStorage
    settingsManager: SettingsManager
    modelRegistry: ModelRegistry
    extensionFlagValues: dict[str, bool | str]
    resourceLoaderOptions: DefaultResourceLoaderOptions


@dataclass(slots=True)
class AgentSessionServices:
    cwd: str
    agentDir: str
    authStorage: AuthStorage
    settingsManager: SettingsManager
    modelRegistry: ModelRegistry
    resourceLoader: ResourceLoaderLike
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


class CreateAgentSessionResultLike(TypedDict, total=False):
    session: Any
    extensionsResult: Any
    modelFallbackMessage: str


class CreateAgentSessionFromServicesOptions(TypedDict, total=False):
    services: AgentSessionServices
    sessionManager: SessionManagerLike
    sessionStartEvent: dict[str, Any]
    model: Model[Any]
    thinkingLevel: ThinkingLevel
    scopedModels: list[dict[str, Any]]
    tools: list[str]
    noTools: str
    customTools: list[Any]


def apply_extension_flag_values(
    resource_loader: ResourceLoaderLike,
    extension_flag_values: dict[str, bool | str] | None,
) -> list[AgentSessionRuntimeDiagnostic]:
    if not extension_flag_values:
        return []

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.getExtensions()
    registered_flags: dict[str, dict[str, str]] = {}
    for extension in extensions_result.extensions:
        for name, flag in extension.flags.items():
            registered_flags[name] = {"type": flag.type}

    unknown_flags: list[str] = []
    for name, value in extension_flag_values.items():
        flag = registered_flags.get(name)
        if flag is None:
            unknown_flags.append(name)
            continue
        if flag["type"] == "boolean":
            extensions_result.runtime.flagValues[name] = True
            continue
        if isinstance(value, str):
            extensions_result.runtime.flagValues[name] = value
            continue
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(
                type="error",
                message=f'Extension flag "--{name}" requires a value',
            )
        )

    if unknown_flags:
        suffix = "" if len(unknown_flags) == 1 else "s"
        rendered = ", ".join(f"--{name}" for name in unknown_flags)
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(
                type="error",
                message=f"Unknown option{suffix}: {rendered}",
            )
        )

    return diagnostics


async def create_agent_session_services(
    options: CreateAgentSessionServicesOptions,
) -> AgentSessionServices:
    cwd = resolve_path(options["cwd"])
    agent_dir = resolve_path(options["agentDir"]) if options.get("agentDir") else get_agent_dir()
    auth_storage = options.get("authStorage") or AuthStorage.create(os.path.join(agent_dir, "auth.json"))
    settings_manager = options.get("settingsManager") or SettingsManager.create(cwd, agent_dir)
    model_registry = options.get("modelRegistry") or ModelRegistry.create(
        auth_storage,
        os.path.join(agent_dir, "models.json"),
    )
    resource_loader = DefaultResourceLoader(
        {
            **(options.get("resourceLoaderOptions") or {}),
            "cwd": cwd,
            "agentDir": agent_dir,
            "settingsManager": settings_manager,
        }
    )
    await resource_loader.reload()

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.getExtensions()
    for registration in list(extensions_result.runtime.pendingProviderRegistrations):
        try:
            model_registry.registerProvider(registration.name, registration.config)
        except Exception as error:  # noqa: BLE001
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type="error",
                    message=f'Extension "{registration.extensionPath}" error: {error}',
                )
            )
    extensions_result.runtime.pendingProviderRegistrations = []
    diagnostics.extend(apply_extension_flag_values(resource_loader, options.get("extensionFlagValues")))

    return AgentSessionServices(
        cwd=cwd,
        agentDir=agent_dir,
        authStorage=auth_storage,
        settingsManager=settings_manager,
        modelRegistry=model_registry,
        resourceLoader=resource_loader,
        diagnostics=diagnostics,
    )


async def create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> CreateAgentSessionResultLike:
    create_session = _resolve_create_agent_session()
    if not callable(create_session):
        raise RuntimeError("create_agent_session is not available; sdk.py has not been ported yet")

    services = options["services"]
    return await create_session(
        {
            "cwd": services.cwd,
            "agentDir": services.agentDir,
            "authStorage": services.authStorage,
            "settingsManager": services.settingsManager,
            "modelRegistry": services.modelRegistry,
            "resourceLoader": services.resourceLoader,
            "sessionManager": options["sessionManager"],
            "model": options.get("model"),
            "thinkingLevel": options.get("thinkingLevel"),
            "scopedModels": options.get("scopedModels"),
            "tools": options.get("tools"),
            "noTools": options.get("noTools"),
            "customTools": options.get("customTools"),
            "sessionStartEvent": options.get("sessionStartEvent"),
        }
    )


def _resolve_create_agent_session() -> Any:
    global create_agent_session
    if callable(create_agent_session):
        return create_agent_session
    try:
        from harnify_coding_agent.core.sdk import create_agent_session as imported_create_agent_session
    except Exception:  # noqa: BLE001
        return None
    create_agent_session = imported_create_agent_session
    return create_agent_session


applyExtensionFlagValues = apply_extension_flag_values
createAgentSessionFromServices = create_agent_session_from_services
createAgentSessionServices = create_agent_session_services

__all__ = [
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionResultLike",
    "CreateAgentSessionServicesOptions",
    "ResourceLoaderLike",
    "SessionManagerLike",
    "applyExtensionFlagValues",
    "apply_extension_flag_values",
    "createAgentSessionFromServices",
    "createAgentSessionServices",
    "create_agent_session_from_services",
    "create_agent_session_services",
]
