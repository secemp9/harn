from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from harnify_agent.agent import Agent
from harnify_ai.providers.faux import faux_assistant_message, register_faux_provider
from harnify_ai.types import Usage
from harnify_coding_agent.core.agent_session import AgentSession, parseSkillBlock
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.compaction.branch_summarization import BranchSummaryResult
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.resource_loader import DefaultResourceLoader
from harnify_coding_agent.core.session_manager import SessionManager
from harnify_coding_agent.core.settings_manager import SettingsManager


def _event_type(event: object) -> str:
    return event["type"] if isinstance(event, dict) else str(event.type)


def _event_field(event: object, name: str, default: object | None = None) -> object | None:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _usage(total_tokens: int, *, input_tokens: int | None = None, output_tokens: int = 1) -> Usage:
    resolved_input = max(total_tokens - output_tokens, 0) if input_tokens is None else input_tokens
    return Usage(
        input=resolved_input,
        output=output_tokens,
        cacheRead=0,
        cacheWrite=0,
        totalTokens=total_tokens,
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    )


def _faux_message_for_model(message: object, model: object, *, usage: Usage | None = None) -> object:
    updates = {
        "provider": getattr(model, "provider"),
        "model": getattr(model, "id"),
        "api": getattr(model, "api"),
    }
    if usage is not None:
        updates["usage"] = usage
    return message.model_copy(update=updates)  # type: ignore[attr-defined]


def _create_session(
    tmp_path: Path,
    *,
    extension_factories: list[object] | None = None,
    custom_tools: list[object] | None = None,
    provider: str = "anthropic",
) -> AgentSession:
    cwd = str(tmp_path)
    agent_dir = str(tmp_path / "agent")
    os.makedirs(agent_dir, exist_ok=True)
    settings_manager = SettingsManager.create(cwd, agent_dir)
    auth_storage = AuthStorage.inMemory()
    auth_storage.setRuntimeApiKey(provider, "test-key")
    model_registry = ModelRegistry.inMemory(auth_storage)
    model = next((model for model in model_registry.getAll() if model.provider == provider), None)
    if model is None:
        model = model_registry.getAll()[0]
    resource_loader = DefaultResourceLoader(
        {
            "cwd": cwd,
            "agentDir": agent_dir,
            "settingsManager": settings_manager,
            "extensionFactories": list(extension_factories or []),
            "noSkills": True,
            "noPromptTemplates": True,
            "noThemes": True,
        }
    )

    session_manager = SessionManager.inMemory(cwd)
    agent = Agent(
        {
            "getApiKey": lambda _provider: "test-key",
            "initialState": {
                "model": model,
                "systemPrompt": "initial",
                "tools": [],
                "thinkingLevel": "high",
                "messages": [],
            },
        }
    )
    return AgentSession(
        {
            "agent": agent,
            "sessionManager": session_manager,
            "settingsManager": settings_manager,
            "cwd": cwd,
            "modelRegistry": model_registry,
            "resourceLoader": resource_loader,
            "customTools": list(custom_tools or []),
        }
    )


async def _create_loaded_session(
    tmp_path: Path,
    *,
    extension_factories: list[object] | None = None,
    custom_tools: list[object] | None = None,
    provider: str = "anthropic",
) -> AgentSession:
    cwd = str(tmp_path)
    agent_dir = str(tmp_path / "agent")
    os.makedirs(agent_dir, exist_ok=True)
    settings_manager = SettingsManager.create(cwd, agent_dir)
    auth_storage = AuthStorage.inMemory()
    auth_storage.setRuntimeApiKey(provider, "test-key")
    model_registry = ModelRegistry.inMemory(auth_storage)
    model = next((model for model in model_registry.getAll() if model.provider == provider), None)
    if model is None:
        model = model_registry.getAll()[0]
    resource_loader = DefaultResourceLoader(
        {
            "cwd": cwd,
            "agentDir": agent_dir,
            "settingsManager": settings_manager,
            "extensionFactories": list(extension_factories or []),
            "noSkills": True,
            "noPromptTemplates": True,
            "noThemes": True,
        }
    )
    await resource_loader.reload()

    session_manager = SessionManager.inMemory(cwd)
    agent = Agent(
        {
            "getApiKey": lambda _provider: "test-key",
            "initialState": {
                "model": model,
                "systemPrompt": "initial",
                "tools": [],
                "thinkingLevel": "high",
                "messages": [],
            },
        }
    )
    return AgentSession(
        {
            "agent": agent,
            "sessionManager": session_manager,
            "settingsManager": settings_manager,
            "cwd": cwd,
            "modelRegistry": model_registry,
            "resourceLoader": resource_loader,
            "customTools": list(custom_tools or []),
        }
    )


def test_parse_skill_block_round_trip() -> None:
    parsed = parseSkillBlock(
        '<skill name="db" location="/tmp/skill/SKILL.md">\nbody\n</skill>\n\nplease apply it'
    )
    assert parsed is not None
    assert parsed.name == "db"
    assert parsed.location == "/tmp/skill/SKILL.md"
    assert parsed.content == "body"
    assert parsed.userMessage == "please apply it"


@pytest.mark.asyncio
async def test_agent_session_reports_stats_and_forking_messages(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    try:
        session.sessionManager.appendMessage({"role": "user", "content": "hello", "timestamp": 1})
        session.sessionManager.appendMessage(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "api": "anthropic-messages",
                "provider": session.model.provider,
                "model": session.model.id,
                "usage": {
                    "input": 200,
                    "output": 1,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 201,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0.5},
                },
                "stopReason": "stop",
                "timestamp": 2,
            }
        )
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        stats = session.getSessionStats()
        assert stats.userMessages == 1
        assert stats.assistantMessages == 1
        assert stats.tokens.input == 200
        assert stats.tokens.total == 201
        assert stats.cost == 0.5
        assert stats.contextUsage is not None
        assert stats.contextUsage["tokens"] == 201

        fork_messages = session.getUserMessagesForForking()
        assert fork_messages == [{"entryId": session.sessionManager.getEntries()[0]["id"], "text": "hello"}]
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_refreshes_extension_and_custom_tools_on_bind(tmp_path: Path) -> None:
    class ExtensionTool:
        name = "dynamic_tool"
        label = "Dynamic Tool"
        description = "Tool registered from session_start"
        promptSnippet = "Run dynamic test behavior"
        promptGuidelines = ["Use dynamic_tool when the user asks for dynamic behavior tests."]
        parameters = {}

        async def execute(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"content": [{"type": "text", "text": "ok"}], "details": {}}

    async def sdk_execute(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"content": [{"type": "text", "text": "ok"}], "details": {}}

    custom_tool = SimpleNamespace(
        name="sdk_tool",
        label="SDK Tool",
        description="Tool registered through AgentSession",
        parameters={},
        execute=sdk_execute,
    )

    session = await _create_loaded_session(
        tmp_path,
        extension_factories=[
            lambda pi: pi.on(
                "session_start",
                lambda _event: pi.register_tool(ExtensionTool()),
            )
        ],
        custom_tools=[custom_tool],
    )

    assert "dynamic_tool" not in [tool.name for tool in session.getAllTools()]

    try:
        await session.bindExtensions({})

        all_tools = session.getAllTools()
        names = [tool.name for tool in all_tools]
        assert "read" in names
        assert "sdk_tool" in names
        assert "dynamic_tool" in names
        assert "dynamic_tool" in session.getActiveToolNames()
        assert "sdk_tool" in session.getActiveToolNames()
        assert "Run dynamic test behavior" in session.systemPrompt
        assert "Use dynamic_tool when the user asks for dynamic behavior tests." in session.systemPrompt

        dynamic = next(tool for tool in all_tools if tool.name == "dynamic_tool")
        sdk = next(tool for tool in all_tools if tool.name == "sdk_tool")
        read = next(tool for tool in all_tools if tool.name == "read")
        assert dynamic.sourceInfo.source == "inline"
        assert sdk.sourceInfo.path == "<sdk:sdk_tool>"
        assert read.sourceInfo.path == "<builtin:read>"
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_dispose_invalidates_replaced_context_and_queue_helpers(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    await session._resourceLoader.reload()
    await session.bindExtensions({})

    ctx = session.createReplacedSessionContext()
    await session.steer("steer this")
    await session.followUp("follow up")
    assert session.pendingMessageCount == 2

    session.setSessionName("Named")
    assert session.sessionName == "Named"

    session.dispose()

    with pytest.raises(RuntimeError, match="stale after session replacement or reload"):
        _ = ctx.cwd


@pytest.mark.asyncio
async def test_agent_session_extension_command_preserves_raw_argument_spacing(tmp_path: Path) -> None:
    seen_args: list[str] = []

    def extension_factory(pi: object) -> None:
        pi.registerCommand(  # type: ignore[attr-defined]
            "testcmd",
            {
                "description": "Test command",
                "handler": lambda args, _ctx: seen_args.append(args),
            },
        )

    session = await _create_loaded_session(tmp_path, extension_factories=[extension_factory])
    try:
        await session.prompt("/testcmd  spaced")

        assert seen_args == [" spaced"]
        assert session.messages == []
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_set_thinking_level_off_without_model_keeps_default_setting(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    try:
        session.settingsManager.setDefaultThinkingLevel("high")
        session.agent.state.model = None

        session.setThinkingLevel("off")

        assert session.settingsManager.getDefaultThinkingLevel() == "high"
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_cycles_models_records_bash_and_reads_last_assistant_text(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.modelRegistry.authStorage.setRuntimeApiKey("openai", "other-key")

    class Ops:
        async def exec(self, _command: str, _cwd: str, options: dict[str, object]) -> dict[str, int | None]:
            on_data = options["onData"]
            assert callable(on_data)
            on_data(b"hello from bash\n")
            return {"exitCode": 0}

    try:
        cycled = await session.cycleModel()
        assert cycled is not None
        assert session.sessionManager.getEntries()[-2]["type"] == "model_change"
        assert session.sessionManager.getEntries()[-1]["type"] == "thinking_level_change"

        result = await session.executeBash("echo hi", options={"operations": Ops()})
        assert result.output == "hello from bash\n"
        assert session.agent.state.messages[-1].role == "bashExecution"

        session.agent.state.messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "final"},
                    {"type": "text", "text": " answer"},
                ],
                "stopReason": "stop",
            }
        )
        assert session.getLastAssistantText() == "final answer"
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_exposes_slash_commands_and_exports_jsonl(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    await session._resourceLoader.reload()
    try:
        session.sessionManager.appendMessage({"role": "user", "content": "hello", "timestamp": 1})
        session.sessionManager.appendMessage(faux_assistant_message("world"))
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        assert session.getSlashCommands() == []

        exported = session.exportToJsonl(str(tmp_path / "exports" / "session.jsonl"))
        content = Path(exported).read_text(encoding="utf-8").splitlines()
        assert len(content) == 3
        assert '"type": "session"' in content[0]
        assert '"parentId": null' in content[1]
        assert '"parentId": "' in content[2]
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_reload_preserves_extension_flag_values_and_command_bindings(tmp_path: Path) -> None:
    def extension_factory(pi: object) -> None:
        pi.registerFlag("plan", {"type": "boolean", "description": "Enable planning"})  # type: ignore[attr-defined]
        pi.registerCommand(  # type: ignore[attr-defined]
            "review",
            {"description": "Run review", "handler": lambda _args, _ctx: None},
        )

    session = await _create_loaded_session(tmp_path, extension_factories=[extension_factory])
    try:
        await session.bindExtensions({})
        session.extensionRunner.set_flag_value("plan", True)

        await session.reload()

        assert session.extensionRunner.get_flag_values()["plan"] is True
        assert any(command["name"] == "review" for command in session.getSlashCommands())
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_reload_without_bindings_skips_session_start_and_resets_api_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_start_events: list[object] = []
    reset_calls: list[str] = []

    def extension_factory(pi: object) -> None:
        pi.on("session_start", lambda event: session_start_events.append(event))  # type: ignore[attr-defined]

    def fake_reset_api_providers() -> None:
        reset_calls.append("reset")

    monkeypatch.setattr(
        "harnify_coding_agent.core.agent_session.reset_api_providers",
        fake_reset_api_providers,
    )

    session = _create_session(tmp_path, extension_factories=[extension_factory])
    await session._resourceLoader.reload()
    try:
        await session.reload()

        assert session_start_events == []
        assert reset_calls == ["reset"]
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_prefers_sdk_tool_over_extension_tool_with_same_name(tmp_path: Path) -> None:
    class ExtensionTool:
        name = "shared_tool"
        label = "Extension Shared Tool"
        description = "Extension implementation"
        parameters = {}

        async def execute(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"content": [{"type": "text", "text": "extension"}], "details": {}}

    async def sdk_execute(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"content": [{"type": "text", "text": "sdk"}], "details": {}}

    custom_tool = SimpleNamespace(
        name="shared_tool",
        label="SDK Shared Tool",
        description="SDK implementation",
        parameters={},
        execute=sdk_execute,
    )

    def extension_factory(pi: object) -> None:
        pi.on("session_start", lambda _event: pi.register_tool(ExtensionTool()))  # type: ignore[attr-defined]

    session = await _create_loaded_session(
        tmp_path,
        extension_factories=[extension_factory],
        custom_tools=[custom_tool],
    )

    try:
        await session.bindExtensions({})

        shared_tools = [tool for tool in session.getAllTools() if tool.name == "shared_tool"]
        assert len(shared_tools) == 1
        assert shared_tools[0].description == "SDK implementation"
        assert shared_tools[0].sourceInfo.path == "<sdk:shared_tool>"
        assert session.getActiveToolNames().count("shared_tool") == 1
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_compact_runs_real_compaction_and_persists_entry(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "compact-model",
                    "reasoning": True,
                    "contextWindow": 200_000,
                    "maxTokens": 8_192,
                }
            ]
        }
    )
    registration.set_responses([faux_assistant_message("## Goal\nCompacted summary")])
    session = _create_session(tmp_path, provider="faux")
    session.agent.state.model = registration.get_model()
    try:
        assistant_one = faux_assistant_message("assistant one").model_copy(
            update={
                "usage": Usage(
                    input=200,
                    output=100,
                    cacheRead=0,
                    cacheWrite=0,
                    totalTokens=300,
                    cost={
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "total": 0,
                    },
                ),
            }
        )
        session.sessionManager.appendMessage(
            {"role": "user", "content": "hello", "timestamp": int(time.time() * 1000)}
        )
        session.sessionManager.appendMessage(assistant_one)
        session.sessionManager.appendMessage(
            {"role": "user", "content": "follow up", "timestamp": int(time.time() * 1000)}
        )
        session.sessionManager.appendMessage(faux_assistant_message("assistant two"))
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        result = await session.compact()
        entries = session.sessionManager.getEntries()

        assert result.summary.startswith("## Goal")
        assert any(entry.get("type") == "compaction" for entry in entries)
        assert session.agent.state.messages[0].role == "compactionSummary"
    finally:
        session.dispose()
        registration.unregister()


@pytest.mark.asyncio
async def test_agent_session_compact_errors_when_session_too_small(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Nothing to compact \\(session too small\\)"):
            await session.compact()
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_compact_allows_non_stream_simple_hook_without_api_key(tmp_path: Path) -> None:
    async def fake_get_api_key_and_headers(_model: object) -> dict[str, object]:
        return {"ok": True, "apiKey": None, "headers": {"x-test": "1"}}

    def extension_factory(pi: object) -> None:
        async def on_before_compact(event: object, _ctx: object) -> dict[str, object]:
            preparation = _event_field(event, "preparation")
            return {
                "compaction": {
                    "summary": "## Goal\nHook compacted",
                    "firstKeptEntryId": getattr(preparation, "firstKeptEntryId"),
                    "tokensBefore": getattr(preparation, "tokensBefore"),
                    "details": {"readFiles": [], "modifiedFiles": []},
                }
            }

        pi.on("session_before_compact", on_before_compact)  # type: ignore[attr-defined]

    session = await _create_loaded_session(tmp_path, provider="faux", extension_factories=[extension_factory])
    session.agent.streamFn = lambda *_args, **_kwargs: None
    session._modelRegistry.getApiKeyAndHeaders = fake_get_api_key_and_headers  # type: ignore[method-assign]
    await session.bindExtensions({})

    try:
        session.sessionManager.appendMessage({"role": "user", "content": "hello", "timestamp": int(time.time() * 1000)})
        session.sessionManager.appendMessage(faux_assistant_message("assistant one"))
        session.sessionManager.appendMessage(
            {"role": "user", "content": "follow up", "timestamp": int(time.time() * 1000)}
        )
        session.sessionManager.appendMessage(faux_assistant_message("assistant two"))
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        result = await session.compact()

        assert result.summary == "## Goal\nHook compacted"
        assert any(entry.get("type") == "compaction" for entry in session.sessionManager.getEntries())
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_navigate_tree_moves_to_user_target_and_restores_editor_text(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    try:
        root_id = session.sessionManager.appendMessage({"role": "user", "content": "root", "timestamp": 1})
        session.sessionManager.appendMessage({"role": "user", "content": "branch work", "timestamp": 2})
        old_leaf_id = session.sessionManager.appendMessage(faux_assistant_message("branch response"))
        session.sessionManager.branch(root_id)
        target_id = session.sessionManager.appendMessage({"role": "user", "content": "target draft", "timestamp": 3})
        session.sessionManager.branch(old_leaf_id)
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        result = await session.navigateTree(target_id, {"summarize": False})

        assert result == {"cancelled": False, "editorText": "target draft"}
        assert session.sessionManager.getLeafId() == root_id
        assert [message["role"] for message in session.agent.state.messages] == ["user"]
        assert session.agent.state.messages[0]["content"] == "root"
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_navigate_tree_can_summarize_abandoned_branch(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "branch-model",
                    "reasoning": True,
                    "contextWindow": 200_000,
                    "maxTokens": 8_192,
                }
            ]
        }
    )
    registration.set_responses([faux_assistant_message("## Goal\nCaptured branch summary")])
    session = _create_session(tmp_path, provider="faux")
    session.agent.state.model = registration.get_model()
    try:
        root_id = session.sessionManager.appendMessage({"role": "user", "content": "root", "timestamp": 1})
        session.sessionManager.appendMessage({"role": "user", "content": "branch work", "timestamp": 2})
        old_leaf_id = session.sessionManager.appendMessage(faux_assistant_message("branch response"))
        session.sessionManager.branch(root_id)
        target_id = session.sessionManager.appendMessage({"role": "user", "content": "target draft", "timestamp": 3})
        session.sessionManager.branch(old_leaf_id)
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        result = await session.navigateTree(
            target_id,
            {"summarize": True, "label": "return-here"},
        )

        summary_entry = result["summaryEntry"]
        assert result["cancelled"] is False
        assert result["editorText"] == "target draft"
        assert summary_entry["type"] == "branch_summary"
        assert session.sessionManager.getLabel(summary_entry["id"]) == "return-here"
        assert session.sessionManager.getLeafEntry()["type"] == "label"
        assert session.agent.state.messages[-1].role == "branchSummary"
    finally:
        session.dispose()
        registration.unregister()


@pytest.mark.asyncio
async def test_agent_session_abort_branch_summary_cancels_navigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate_branch_summary(_entries: list[object], options: object) -> BranchSummaryResult:
        await options.signal.wait()
        return BranchSummaryResult(aborted=True)

    monkeypatch.setattr(
        "harnify_coding_agent.core.agent_session.generate_branch_summary",
        fake_generate_branch_summary,
    )
    session = _create_session(tmp_path)
    try:
        root_id = session.sessionManager.appendMessage({"role": "user", "content": "root", "timestamp": 1})
        session.sessionManager.appendMessage({"role": "user", "content": "branch work", "timestamp": 2})
        old_leaf_id = session.sessionManager.appendMessage(faux_assistant_message("branch response"))
        session.sessionManager.branch(root_id)
        target_id = session.sessionManager.appendMessage({"role": "user", "content": "target draft", "timestamp": 3})
        session.sessionManager.branch(old_leaf_id)
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        task = asyncio.create_task(session.navigateTree(target_id, {"summarize": True}))
        for _ in range(200):
            if session._branchSummaryAbortController is not None:
                break
            await asyncio.sleep(0.005)
        assert session._branchSummaryAbortController is not None

        session.abortBranchSummary()
        result = await asyncio.wait_for(task, timeout=1)

        assert result == {"cancelled": True, "aborted": True}
        assert session.sessionManager.getLeafId() == old_leaf_id
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_auto_retry_retries_and_emits_lifecycle_events(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "retry-model",
                    "reasoning": True,
                    "contextWindow": 200_000,
                    "maxTokens": 8_192,
                }
            ]
        }
    )
    registration.set_responses(
        [
            faux_assistant_message("", stop_reason="error", error_message="503 service unavailable"),
            faux_assistant_message("recovered"),
        ]
    )
    session = _create_session(tmp_path, provider="faux")
    session.agent.state.model = registration.get_model()
    session.settingsManager.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 0}
    events: list[object] = []
    session.subscribe(events.append)

    try:
        await session.prompt("retry please")

        assert session.retryAttempt == 0
        assert session.isRetrying is False
        assert session.getLastAssistantText() == "recovered"

        agent_end_events = [event for event in events if _event_type(event) == "agent_end"]
        assert len(agent_end_events) == 2
        assert _event_field(agent_end_events[0], "willRetry") is True
        assert _event_field(agent_end_events[1], "willRetry") is False

        retry_start = next(event for event in events if _event_type(event) == "auto_retry_start")
        retry_end = next(event for event in events if _event_type(event) == "auto_retry_end")
        assert _event_field(retry_start, "attempt") == 1
        assert _event_field(retry_start, "maxAttempts") == 2
        assert _event_field(retry_start, "delayMs") == 0
        assert _event_field(retry_start, "errorMessage") == "503 service unavailable"
        assert _event_field(retry_end, "success") is True
        assert _event_field(retry_end, "attempt") == 1

        assistant_entries = [
            entry.get("message")
            for entry in session.sessionManager.getEntries()
            if entry.get("type") == "message"
            and (
                entry.get("message", {}).get("role")
                if isinstance(entry.get("message"), dict)
                else getattr(entry.get("message"), "role", None)
            )
            == "assistant"
        ]
        assert len(assistant_entries) == 2
    finally:
        session.dispose()
        registration.unregister()


@pytest.mark.asyncio
async def test_agent_session_auto_compaction_uses_threshold_hook_path(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "threshold-model",
                    "reasoning": True,
                    "contextWindow": 2_000,
                    "maxTokens": 8_192,
                }
            ]
        }
    )
    model = registration.get_model()
    registration.set_responses(
        [
            _faux_message_for_model(
                faux_assistant_message("threshold response"),
                model,
                usage=_usage(96, input_tokens=95),
            )
        ]
    )

    hook_events: list[object] = []

    def extension_factory(pi: object) -> None:
        async def on_before_compact(event: object, _ctx: object) -> dict[str, object]:
            hook_events.append(event)
            preparation = _event_field(event, "preparation")
            return {
                "compaction": {
                    "summary": "## Goal\nHook summary",
                    "firstKeptEntryId": getattr(preparation, "firstKeptEntryId"),
                    "tokensBefore": getattr(preparation, "tokensBefore"),
                    "details": {"readFiles": ["hook.txt"], "modifiedFiles": []},
                }
            }

        pi.on("session_before_compact", on_before_compact)  # type: ignore[attr-defined]

    session = await _create_loaded_session(tmp_path, provider="faux", extension_factories=[extension_factory])
    session.agent.state.model = model
    events: list[object] = []
    session.subscribe(events.append)
    session.settingsManager.settings["compaction"] = {"enabled": True, "reserveTokens": 1_000, "keepRecentTokens": 0}
    await session.bindExtensions({})

    try:
        await session.prompt("compact on threshold")

        compaction_entry = next(
            entry for entry in session.sessionManager.getEntries() if entry.get("type") == "compaction"
        )
        compaction_end = next(event for event in events if _event_type(event) == "compaction_end")

        assert hook_events
        assert compaction_entry["fromHook"] is True
        assert compaction_entry["summary"] == "## Goal\nHook summary"
        assert _event_field(compaction_end, "reason") == "threshold"
        assert _event_field(compaction_end, "aborted") is False
        assert _event_field(compaction_end, "willRetry") is False
        assert session.agent.state.messages[0].role == "compactionSummary"
    finally:
        session.dispose()
        registration.unregister()


@pytest.mark.asyncio
async def test_agent_session_check_compaction_does_not_retry_overflow_more_than_once(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    events: list[object] = []
    session.subscribe(events.append)
    calls: list[tuple[str, bool]] = []

    async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
        calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = fake_run_auto_compaction  # type: ignore[method-assign]
    model = session.model
    assert model is not None
    overflow_message = _faux_message_for_model(
        faux_assistant_message("", stop_reason="error", error_message="prompt is too long"),
        model,
        usage=_usage(0, input_tokens=0, output_tokens=0),
    )

    try:
        await session._check_compaction(overflow_message)  # type: ignore[attr-defined]
        await session._check_compaction(  # type: ignore[attr-defined]
            overflow_message.model_copy(update={"timestamp": overflow_message.timestamp + 1})
        )

        assert calls == [("overflow", True)]
        assert any(
            _event_type(event) == "compaction_end"
            and _event_field(event, "reason") == "overflow"
            and "failed after one compact-and-retry attempt" in str(_event_field(event, "errorMessage"))
            for event in events
        )
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_check_compaction_ignores_stale_pre_compaction_assistant_usage(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    calls: list[tuple[str, bool]] = []

    async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
        calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = fake_run_auto_compaction  # type: ignore[method-assign]
    model = session.model
    assert model is not None
    stale_timestamp = int(time.time() * 1000) - 10_000
    stale_assistant = _faux_message_for_model(
        faux_assistant_message("large response before compaction", timestamp=stale_timestamp),
        model,
        usage=_usage(610_000, input_tokens=610_000, output_tokens=0),
    )

    try:
        session.sessionManager.appendMessage(
            {"role": "user", "content": [{"type": "text", "text": "before compaction"}], "timestamp": stale_timestamp - 1000}
        )
        session.sessionManager.appendMessage(stale_assistant)
        first_kept_entry_id = session.sessionManager.getEntries()[0]["id"]
        session.sessionManager.appendCompaction("summary", first_kept_entry_id, stale_assistant.usage.totalTokens, None, False)
        session.sessionManager.appendMessage(
            {"role": "user", "content": [{"type": "text", "text": "after compaction"}], "timestamp": int(time.time() * 1000)}
        )
        session.agent.state.messages = session.sessionManager.buildSessionContext().messages

        await session._check_compaction(stale_assistant, False)  # type: ignore[attr-defined]

        assert calls == []
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_agent_session_abort_retry_cancels_backoff(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "retry-model",
                    "reasoning": True,
                    "contextWindow": 200_000,
                    "maxTokens": 8_192,
                }
            ]
        }
    )
    registration.set_responses([faux_assistant_message("", stop_reason="error", error_message="503 retry later")])
    session = _create_session(tmp_path, provider="faux")
    session.agent.state.model = registration.get_model()
    session.settingsManager.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 200}
    events: list[object] = []
    session.subscribe(events.append)

    try:
        prompt_task = asyncio.create_task(session.prompt("cancel retry"))
        for _ in range(200):
            if session.isRetrying:
                break
            await asyncio.sleep(0.005)
        assert session.isRetrying is True

        session.abortRetry()
        await asyncio.wait_for(prompt_task, timeout=1)

        retry_end = next(event for event in events if _event_type(event) == "auto_retry_end")
        assert _event_field(retry_end, "success") is False
        assert _event_field(retry_end, "attempt") == 1
        assert _event_field(retry_end, "finalError") == "Retry cancelled"
        assert session.retryAttempt == 0
        assert session.isRetrying is False
    finally:
        session.dispose()
        registration.unregister()


@pytest.mark.asyncio
async def test_agent_session_set_auto_retry_enabled_controls_live_behavior(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "retry-model",
                    "reasoning": True,
                    "contextWindow": 200_000,
                    "maxTokens": 8_192,
                }
            ]
        }
    )
    registration.set_responses([faux_assistant_message("", stop_reason="error", error_message="429 too many requests")])
    session = _create_session(tmp_path, provider="faux")
    session.agent.state.model = registration.get_model()
    session.settingsManager.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 0}
    session.setAutoRetryEnabled(False)
    events: list[object] = []
    session.subscribe(events.append)

    try:
        await session.prompt("do not retry")

        assert session.autoRetryEnabled is False
        assert all(_event_type(event) != "auto_retry_start" for event in events)
        agent_end = next(event for event in events if _event_type(event) == "agent_end")
        assert _event_field(agent_end, "willRetry") is False
    finally:
        session.dispose()
        registration.unregister()
