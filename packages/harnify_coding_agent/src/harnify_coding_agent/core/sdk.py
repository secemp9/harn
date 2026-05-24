"""Public coding-agent SDK entrypoints."""

from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

from harnify_agent.agent import Agent
from harnify_agent.types import AgentMessage, ThinkingLevel
from harnify_ai.models import clamp_thinking_level
from harnify_ai.stream import stream_simple
from harnify_ai.types import Model, SimpleStreamOptions, TextContent, validate_message

from harnify_coding_agent.config import get_agent_dir
from harnify_coding_agent.core.agent_session import AgentSession
from harnify_coding_agent.core.auth_guidance import format_no_models_available_message
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.defaults import DEFAULT_THINKING_LEVEL
from harnify_coding_agent.core.extensions.types import ExtensionFactory, LoadExtensionsResult, ToolDefinition
from harnify_coding_agent.core.messages import convert_to_llm
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.model_resolver import findInitialModel
from harnify_coding_agent.core.resource_loader import DefaultResourceLoader, ResourceLoaderLike
from harnify_coding_agent.core.session_manager import SessionManager, get_default_session_dir
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.core.telemetry import is_install_telemetry_enabled
from harnify_coding_agent.core.timings import time
from harnify_coding_agent.core.tools import (
    Tool,
    ToolName,
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_only_tools,
    create_read_tool,
    create_write_tool,
    with_file_mutation_queue,
)
from harnify_coding_agent.utils.paths import resolve_path


class CreateAgentSessionOptions(TypedDict, total=False):
    cwd: str
    agentDir: str
    authStorage: AuthStorage
    modelRegistry: ModelRegistry
    model: Model[Any] | None
    thinkingLevel: ThinkingLevel
    scopedModels: list[dict[str, Any]]
    noTools: Literal["all", "builtin"]
    tools: list[str]
    customTools: list[ToolDefinition[Any, Any] | Any]
    resourceLoader: ResourceLoaderLike
    sessionManager: SessionManager
    settingsManager: SettingsManager
    sessionStartEvent: dict[str, Any]


class CreateAgentSessionResult(TypedDict, total=False):
    session: AgentSession
    extensionsResult: LoadExtensionsResult
    modelFallbackMessage: str


def get_default_agent_dir() -> str:
    return get_agent_dir()


def get_attribution_headers(
    model: Model[Any],
    settings_manager: SettingsManager,
) -> dict[str, str] | None:
    if not is_install_telemetry_enabled(settings_manager):
        return None

    if model.provider == "openrouter" or "openrouter.ai" in model.baseUrl:
        return {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "pi",
            "X-OpenRouter-Categories": "cli-agent",
        }

    if (
        model.provider in {"cloudflare-workers-ai", "cloudflare-ai-gateway"}
        or "api.cloudflare.com" in model.baseUrl
        or "gateway.ai.cloudflare.com" in model.baseUrl
    ):
        return {
            "User-Agent": "pi-coding-agent",
        }

    return None


async def create_agent_session(options: CreateAgentSessionOptions | None = None) -> CreateAgentSessionResult:
    resolved_options = dict(options or {})
    explicit_session_manager = resolved_options.get("sessionManager")
    cwd = resolve_path(
        resolved_options.get("cwd")
        or (explicit_session_manager.getCwd() if explicit_session_manager else os.getcwd())
    )
    agent_dir = (
        resolve_path(resolved_options["agentDir"])
        if resolved_options.get("agentDir")
        else get_default_agent_dir()
    )
    resource_loader = resolved_options.get("resourceLoader")

    auth_path = os.path.join(agent_dir, "auth.json") if resolved_options.get("agentDir") else None
    models_path = os.path.join(agent_dir, "models.json") if resolved_options.get("agentDir") else None
    auth_storage = resolved_options.get("authStorage") or AuthStorage.create(auth_path)
    model_registry = resolved_options.get("modelRegistry") or ModelRegistry.create(auth_storage, models_path)
    settings_manager = resolved_options.get("settingsManager") or SettingsManager.create(cwd, agent_dir)
    session_manager = explicit_session_manager or SessionManager.create(cwd, get_default_session_dir(cwd, agent_dir))

    if resource_loader is None:
        resource_loader = DefaultResourceLoader(
            {"cwd": cwd, "agentDir": agent_dir, "settingsManager": settings_manager}
        )
        await resource_loader.reload()
        time("resourceLoader.reload")

    existing_session = session_manager.buildSessionContext()
    has_existing_session = len(existing_session.messages) > 0
    has_thinking_entry = any(entry.get("type") == "thinking_level_change" for entry in session_manager.getBranch())

    model = resolved_options.get("model")
    model_fallback_message: str | None = None
    if model is None and has_existing_session and existing_session.model:
        restored_model = model_registry.find(existing_session.model["provider"], existing_session.model["modelId"])
        if restored_model is not None and model_registry.hasConfiguredAuth(restored_model):
            model = restored_model
        if model is None:
            model_fallback_message = (
                f'Could not restore model {existing_session.model["provider"]}/{existing_session.model["modelId"]}'
            )

    if model is None:
        result = await findInitialModel(
            {
                "scopedModels": [],
                "isContinuing": has_existing_session,
                "defaultProvider": settings_manager.getDefaultProvider(),
                "defaultModelId": settings_manager.getDefaultModel(),
                "defaultThinkingLevel": settings_manager.getDefaultThinkingLevel(),
                "modelRegistry": model_registry,
            }
        )
        model = result.model
        if model is None:
            model_fallback_message = format_no_models_available_message()
        elif model_fallback_message:
            model_fallback_message += f". Using {model.provider}/{model.id}"

    thinking_level = resolved_options.get("thinkingLevel")
    if thinking_level is None and has_existing_session:
        thinking_level = (
            existing_session.thinkingLevel
            if has_thinking_entry
            else settings_manager.getDefaultThinkingLevel() or DEFAULT_THINKING_LEVEL
        )
    if thinking_level is None:
        thinking_level = settings_manager.getDefaultThinkingLevel() or DEFAULT_THINKING_LEVEL
    thinking_level = "off" if model is None else clamp_thinking_level(model, thinking_level)

    default_active_tool_names: list[ToolName] = ["read", "bash", "edit", "write"]
    allowed_tool_names = resolved_options.get("tools")
    if allowed_tool_names is None and resolved_options.get("noTools") == "all":
        allowed_tool_names = []
    initial_active_tool_names = (
        list(resolved_options["tools"])
        if resolved_options.get("tools") is not None
        else ([] if resolved_options.get("noTools") else default_active_tool_names)
    )

    extension_runner_ref: dict[str, Any] = {}

    def convert_to_llm_with_block_images(messages: list[AgentMessage]) -> list[Any]:
        converted = convert_to_llm(messages)
        if not settings_manager.getBlockImages():
            return converted

        filtered_messages: list[Any] = []
        for message in converted:
            role = _message_field(message, "role")
            content = _message_field(message, "content")
            if role not in {"user", "toolResult"} or not isinstance(content, list):
                filtered_messages.append(message)
                continue
            if not any(_content_type(item) == "image" for item in content):
                filtered_messages.append(message)
                continue

            filtered_content: list[Any] = []
            previous_disabled = False
            for item in content:
                if _content_type(item) == "image":
                    if not previous_disabled:
                        filtered_content.append(TextContent(text="Image reading is disabled."))
                    previous_disabled = True
                    continue
                previous_disabled = (
                    _content_type(item) == "text"
                    and _content_text(item) == "Image reading is disabled."
                )
                filtered_content.append(item)

            payload = _message_dump(message)
            payload["content"] = [_message_dump(item) for item in filtered_content]
            filtered_messages.append(validate_message(payload))

        return filtered_messages

    async def stream_fn(model_value: Model[Any], context: Any, stream_options: Any = None) -> Any:
        auth = await model_registry.getApiKeyAndHeaders(model_value)
        if not auth.get("ok"):
            raise RuntimeError(auth["error"])

        provider_retry_settings = settings_manager.getProviderRetrySettings()
        attribution_headers = get_attribution_headers(model_value, settings_manager)
        resolved_stream_options = dict(stream_options or {})
        headers = _merge_headers(attribution_headers, auth.get("headers"), resolved_stream_options.get("headers"))
        final_options = dict(resolved_stream_options)
        final_options["apiKey"] = auth.get("apiKey")
        if final_options.get("timeoutMs") is None:
            final_options["timeoutMs"] = provider_retry_settings.get("timeoutMs")
        if final_options.get("maxRetries") is None:
            final_options["maxRetries"] = provider_retry_settings.get("maxRetries")
        if final_options.get("maxRetryDelayMs") is None:
            final_options["maxRetryDelayMs"] = provider_retry_settings.get("maxRetryDelayMs")
        if headers is not None:
            final_options["headers"] = headers
        return stream_simple(model_value, context, SimpleStreamOptions.model_validate(final_options))

    async def on_payload(payload: dict[str, Any], _model: Model[Any]) -> Any:
        runner = extension_runner_ref.get("current")
        if runner is None or not runner.has_handlers("before_provider_request"):
            return payload
        return await runner.emit_before_provider_request(payload)

    async def on_response(response: Any, _model: Model[Any]) -> None:
        runner = extension_runner_ref.get("current")
        if runner is None or not runner.has_handlers("after_provider_response"):
            return
        await runner.emit(
            {
                "type": "after_provider_response",
                "status": _message_field(response, "status"),
                "headers": _message_field(response, "headers"),
            }
        )

    async def transform_context(messages: list[AgentMessage], _signal: Any | None = None) -> list[AgentMessage]:
        runner = extension_runner_ref.get("current")
        if runner is None:
            return messages
        return await runner.emit_context(messages)

    initial_model = model or _unknown_model()
    agent = Agent(
        {
            "initialState": {
                "systemPrompt": "",
                "model": initial_model,
                "thinkingLevel": thinking_level,
                "tools": [],
            },
            "convertToLlm": convert_to_llm_with_block_images,
            "streamFn": stream_fn,
            "onPayload": on_payload,
            "onResponse": on_response,
            "sessionId": session_manager.getSessionId(),
            "transformContext": transform_context,
            "steeringMode": settings_manager.getSteeringMode(),
            "followUpMode": settings_manager.getFollowUpMode(),
            "transport": settings_manager.getTransport(),
            "thinkingBudgets": settings_manager.getThinkingBudgets(),
            "maxRetryDelayMs": settings_manager.getProviderRetrySettings().get("maxRetryDelayMs"),
        }
    )
    if model is None:
        agent.state.model = None

    if has_existing_session:
        agent.state.messages = existing_session.messages
        if not has_thinking_entry:
            session_manager.appendThinkingLevelChange(thinking_level)
    else:
        if model is not None:
            session_manager.appendModelChange(model.provider, model.id)
        session_manager.appendThinkingLevelChange(thinking_level)

    session = AgentSession(
        {
            "agent": agent,
            "sessionManager": session_manager,
            "settingsManager": settings_manager,
            "cwd": cwd,
            "scopedModels": resolved_options.get("scopedModels") or [],
            "resourceLoader": resource_loader,
            "customTools": resolved_options.get("customTools") or [],
            "modelRegistry": model_registry,
            "initialActiveToolNames": initial_active_tool_names,
            "allowedToolNames": allowed_tool_names,
            "extensionRunnerRef": extension_runner_ref,
            "sessionStartEvent": resolved_options.get("sessionStartEvent"),
        }
    )
    extensions_result = resource_loader.getExtensions()

    return {
        "session": session,
        "extensionsResult": extensions_result,
        "modelFallbackMessage": model_fallback_message,
    }


def _merge_headers(*header_groups: dict[str, str] | None) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    for group in header_groups:
        if group:
            merged.update(group)
    return merged or None


def _unknown_model() -> Model[Any]:
    return Model(
        id="unknown",
        name="unknown",
        api="unknown",
        provider="unknown",
        baseUrl="",
        reasoning=False,
        input=[],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=0,
        maxTokens=0,
    )


def _message_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _message_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _message_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_message_dump(item) for item in value]
    return value


def _content_type(block: Any) -> str | None:
    value = _message_field(block, "type")
    return value if isinstance(value, str) else None


def _content_text(block: Any) -> str | None:
    value = _message_field(block, "text")
    return value if isinstance(value, str) else None


createAgentSession = create_agent_session
createBashTool = create_bash_tool
createCodingTools = create_coding_tools
createEditTool = create_edit_tool
createFindTool = create_find_tool
createGrepTool = create_grep_tool
createLsTool = create_ls_tool
createReadOnlyTools = create_read_only_tools
createReadTool = create_read_tool
createWriteTool = create_write_tool
getAttributionHeaders = get_attribution_headers
getDefaultAgentDir = get_default_agent_dir
withFileMutationQueue = with_file_mutation_queue

__all__ = [
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "ExtensionFactory",
    "Tool",
    "ToolDefinition",
    "ToolName",
    "createAgentSession",
    "createBashTool",
    "createCodingTools",
    "createEditTool",
    "createFindTool",
    "createGrepTool",
    "createLsTool",
    "createReadOnlyTools",
    "createReadTool",
    "createWriteTool",
    "create_agent_session",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_only_tools",
    "create_read_tool",
    "create_write_tool",
    "getAttributionHeaders",
    "getDefaultAgentDir",
    "get_attribution_headers",
    "get_default_agent_dir",
    "withFileMutationQueue",
    "with_file_mutation_queue",
]
