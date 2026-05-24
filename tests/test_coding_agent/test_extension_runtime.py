from __future__ import annotations

from typing import Any

import pytest
from harnify_agent.types import AgentToolResult
from harnify_ai.types import TextContent
from harnify_coding_agent.core import exec as exec_module
from harnify_coding_agent.core import extensions as extension_package
from harnify_coding_agent.core.event_bus import create_event_bus
from harnify_coding_agent.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from harnify_coding_agent.core.extensions.runner import ExtensionRunner
from harnify_coding_agent.core.extensions import types as extension_types
from harnify_coding_agent.core.extensions.types import ToolDefinition


def test_extension_types_exports_include_ts_surface_restorations() -> None:
    for name in [
        "AgentToolResult",
        "AgentToolUpdateCallback",
        "AppKeybinding",
        "AutocompleteProviderFactory",
        "EditorFactory",
        "ExecOptions",
        "ExecResult",
        "ExtensionRuntimeState",
        "ExtensionUIDialogOptions",
        "ExtensionWidgetOptions",
        "KeybindingsManager",
        "ReplacedSessionContext",
        "TerminalInputHandler",
        "WidgetPlacement",
        "WorkingIndicatorOptions",
        "defineTool",
        "isBashToolResult",
        "isEditToolResult",
        "isFindToolResult",
        "isGrepToolResult",
        "isLsToolResult",
        "isReadToolResult",
        "isToolCallEventType",
        "isWriteToolResult",
    ]:
        assert name in extension_types.__all__

    assert extension_types.ExecOptions is exec_module.ExecOptions
    assert extension_types.ExecResult is exec_module.ExecResult
    assert extension_types.defineTool({"demo": True}) == {"demo": True}
    assert extension_types.isToolCallEventType("bash", {"toolName": "bash"}) is True
    assert extension_types.isBashToolResult({"toolName": "bash"}) is True
    assert extension_types.isWriteToolResult({"toolName": "write"}) is True


def test_extension_package_exports_match_curated_ts_surface() -> None:
    for name in (
        "SlashCommandInfo",
        "SlashCommandSource",
        "SourceInfo",
        "createExtensionRuntime",
        "discoverAndLoadExtensions",
        "loadExtensionFromFactory",
        "loadExtensions",
        "ExtensionErrorListener",
        "ExtensionRunner",
        "ForkHandler",
        "NavigateTreeHandler",
        "NewSessionHandler",
        "ShutdownHandler",
        "SwitchSessionHandler",
        "defineTool",
        "wrapRegisteredTool",
        "wrapRegisteredTools",
    ):
        assert name in extension_package.__all__

    for name in (
        "ReloadHandler",
        "create_extension_runtime",
        "discover_and_load_extensions",
        "load_extension_from_factory",
        "load_extensions",
        "wrap_registered_tool",
        "wrap_registered_tools",
        "ExtensionRuntimeState",
        "ToolRenderContext",
    ):
        assert name not in extension_package.__all__


@pytest.mark.asyncio
async def test_extension_loader_and_runner_bind_real_runtime_surface() -> None:
    send_message_errors: list[str] = []
    flag_default: list[bool | str | None] = []

    async def factory(api: Any) -> None:
        try:
            api.sendMessage({"customType": "prebind", "content": "x", "display": True})
        except RuntimeError as error:
            send_message_errors.append(str(error))

        api.registerFlag(
            "demo-flag",
            {
                "type": "string",
                "description": "demo flag",
                "default": "flag-default",
            },
        )
        flag_default.append(api.getFlag("demo-flag"))
        api.registerCommand(
            "demo",
            {
                "description": "demo command",
                "handler": lambda args, ctx: None,
            },
        )
        api.registerShortcut(
            "ctrl+x",
            {
                "description": "demo shortcut",
                "handler": lambda ctx: None,
            },
        )
        api.registerMessageRenderer("demo", lambda message, options: f"{message}:{options}")
        api.registerProvider("demo-provider", {"baseUrl": "https://example.test"})

    runtime = create_extension_runtime()
    event_bus = create_event_bus()
    extension = await load_extension_from_factory(
        factory,
        "/tmp",
        event_bus,
        runtime,
        extension_path="<inline:demo>",
    )

    assert send_message_errors and "not initialized" in send_message_errors[0]
    assert flag_default == ["flag-default"]
    assert list(extension.commands) == ["demo"]
    assert list(extension.shortcuts) == ["ctrl+x"]
    assert list(extension.messageRenderers) == ["demo"]

    registered_providers: list[tuple[str, dict[str, Any]]] = []
    unregistered_providers: list[str] = []

    runner = ExtensionRunner(extensions=[extension], runtime=runtime)

    runner.bind_core(
        {
            "sendMessage": lambda message, options=None: None,
            "sendUserMessage": lambda content, options=None: None,
            "appendEntry": lambda custom_type, data=None: None,
            "setSessionName": lambda name: None,
            "getSessionName": lambda: "demo",
            "setLabel": lambda entry_id, label: None,
            "getActiveTools": lambda: ["demo"],
            "getAllTools": lambda: [],
            "setActiveTools": lambda names: None,
            "refreshTools": lambda: None,
            "getCommands": lambda: [],
            "setModel": _set_model_true,
            "getThinkingLevel": lambda: "off",
            "setThinkingLevel": lambda level: None,
        },
        {
            "getModel": lambda: None,
            "isIdle": lambda: True,
            "getSignal": lambda: None,
            "abort": lambda: None,
            "hasPendingMessages": lambda: False,
            "shutdown": lambda: None,
            "getContextUsage": lambda: None,
            "compact": lambda options=None: None,
            "getSystemPrompt": lambda: "prompt",
        },
        {
            "registerProvider": lambda name, config: registered_providers.append((name, dict(config))),
            "unregisterProvider": lambda name: unregistered_providers.append(name),
        },
    )

    assert registered_providers == [("demo-provider", {"baseUrl": "https://example.test"})]
    assert runner.runtime.pendingProviderRegistrations == []
    assert runner.get_flags()["demo-flag"].default == "flag-default"
    assert runner.get_flag_values()["demo-flag"] == "flag-default"

    runner.set_flag_value("demo-flag", "changed")
    assert runner.get_flag_values()["demo-flag"] == "changed"

    await runner.runtime.setModel(None)
    runner.runtime.registerProvider("late-provider", {"apiKey": "LATE"})
    runner.runtime.unregisterProvider("late-provider")

    assert registered_providers[-1] == ("late-provider", {"apiKey": "LATE"})
    assert unregistered_providers == ["late-provider"]
    assert runner.get_command("demo") is not None
    assert runner.get_message_renderer("demo")("m", "o") == "m:o"


@pytest.mark.asyncio
async def test_extension_api_exec_uses_shared_exec_result_and_per_call_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    exec_calls: list[tuple[str, list[str], str, dict[str, object]]] = []
    exec_result = exec_module.ExecResult(stdout="out", stderr="err", code=7, killed=True)
    seen_results: list[exec_module.ExecResult] = []

    async def fake_exec_command(
        command: str,
        args: list[str],
        cwd: str,
        options: dict[str, object] | None = None,
    ) -> exec_module.ExecResult:
        exec_calls.append((command, list(args), cwd, dict(options or {})))
        return exec_result

    monkeypatch.setattr("harnify_coding_agent.core.extensions.loader.exec_command", fake_exec_command)

    async def factory(api: Any) -> None:
        seen_results.append(await api.exec("demo", ["--flag"], {"cwd": "/override"}))
        seen_results.append(await api.exec("demo", ["--empty"], {"cwd": ""}))

    await load_extension_from_factory(
        factory,
        "/base",
        create_event_bus(),
        create_extension_runtime(),
        extension_path="<inline:exec>",
    )

    assert exec_calls == [
        ("demo", ["--flag"], "/override", {"cwd": "/override"}),
        ("demo", ["--empty"], "", {"cwd": ""}),
    ]
    assert seen_results == [exec_result, exec_result]
    assert seen_results[0].code == 7
    assert seen_results[0].killed is True


def test_create_extension_runtime_preserves_explicit_empty_provider_extension_path() -> None:
    runtime = create_extension_runtime()

    runtime.registerProvider("demo", {"apiKey": "key"}, "")

    assert len(runtime.pendingProviderRegistrations) == 1
    assert runtime.pendingProviderRegistrations[0].extensionPath == ""


@pytest.mark.asyncio
async def test_extension_runner_emits_events_and_invalidates_context() -> None:
    errors: list[Any] = []

    async def factory(api: Any) -> None:
        async def execute(tool_call_id: str, params: Any, signal: Any, on_update: Any, ctx: Any) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text=ctx["cwd"])], details={"tool": tool_call_id})

        api.registerTool(
            ToolDefinition(
                name="demo-tool",
                label="Demo Tool",
                description="demo",
                parameters={"type": "object"},
                execute=execute,
            )
        )
        api.on("context", lambda event, ctx: {"messages": event["messages"] + [{"role": "user", "content": "x"}]})
        api.on("before_provider_request", lambda event, ctx: {"wrapped": event["payload"]})
        api.on(
            "before_agent_start",
            lambda event, ctx: {
                "message": {"role": "custom", "content": "added"},
                "systemPrompt": event["systemPrompt"] + "::extra",
            },
        )
        api.on(
            "resources_discover",
            lambda event, ctx: {
                "skillPaths": ["/skills/demo"],
                "promptPaths": ["/prompts/demo"],
                "themePaths": ["/themes/demo"],
            },
        )
        api.on("input", lambda event, ctx: {"action": "transform", "text": event["text"].upper()})
        api.on("message_end", lambda event, ctx: {"message": {"role": "assistant", "text": "changed"}})
        api.on("tool_result", lambda event, ctx: {"content": ["done"], "details": {"changed": True}, "isError": True})
        api.on("user_bash", lambda event, ctx: {"block": True, "reason": "nope"})
        api.on("session_before_switch", lambda event, ctx: {"cancel": True, "reason": "stop"})
        api.on("message_end", lambda event, ctx: {"message": {"role": "user", "text": "bad-role"}})
        api.on("input", lambda event, ctx: (_ for _ in ()).throw(RuntimeError("boom")))

    extension = await load_extension_from_factory(
        factory,
        "/tmp/project",
        create_event_bus(),
        create_extension_runtime(),
        extension_path="<inline:eventful>",
    )
    runner = ExtensionRunner(
        extensions=[extension],
        cwd="runner-cwd",
    )
    runner.on_error(lambda error: errors.append(error))

    runner.bind_core(
        {
            "sendMessage": lambda message, options=None: None,
            "sendUserMessage": lambda content, options=None: None,
            "appendEntry": lambda custom_type, data=None: None,
            "setSessionName": lambda name: None,
            "getSessionName": lambda: "demo",
            "setLabel": lambda entry_id, label: None,
            "getActiveTools": lambda: ["demo-tool"],
            "getAllTools": lambda: [],
            "setActiveTools": lambda names: None,
            "refreshTools": lambda: None,
            "getCommands": lambda: [],
            "setModel": _set_model_true,
            "getThinkingLevel": lambda: "off",
            "setThinkingLevel": lambda level: None,
        },
        {
            "getModel": lambda: None,
            "isIdle": lambda: False,
            "getSignal": lambda: "signal",
            "abort": lambda: None,
            "hasPendingMessages": lambda: True,
            "shutdown": lambda: None,
            "getContextUsage": lambda: {"tokens": 1},
            "compact": lambda options=None: None,
            "getSystemPrompt": lambda: "base-prompt",
        },
    )

    ctx = runner.create_context()
    assert ctx["cwd"] == "runner-cwd"
    assert ctx.getSystemPrompt() == "base-prompt"
    assert ctx.signal == "signal"
    assert ctx.isIdle() is False
    assert ctx.hasPendingMessages() is True
    assert ctx.ui.theme is not None

    assert await runner.emit_context([{"role": "user"}]) == [{"role": "user"}, {"role": "user", "content": "x"}]
    assert await runner.emit_before_provider_request({"a": 1}) == {"wrapped": {"a": 1}}
    assert await runner.emit_before_agent_start("prompt", None, "system", {"cwd": "/tmp"}) == {
        "messages": [{"role": "custom", "content": "added"}],
        "systemPrompt": "system::extra",
    }
    assert await runner.emit_resources_discover("/tmp", "reload") == {
        "skillPaths": [{"path": "/skills/demo", "extensionPath": "<inline:eventful>"}],
        "promptPaths": [{"path": "/prompts/demo", "extensionPath": "<inline:eventful>"}],
        "themePaths": [{"path": "/themes/demo", "extensionPath": "<inline:eventful>"}],
    }
    assert await runner.emit_input("hello", None, "editor") == {
        "action": "transform",
        "text": "HELLO",
        "images": None,
    }
    assert await runner.emit_message_end(
        {"type": "message_end", "message": {"role": "assistant", "text": "start"}}
    ) == {
        "role": "assistant",
        "text": "changed",
    }
    assert await runner.emit_tool_result(
        {"type": "tool_result", "content": ["orig"], "details": {}, "isError": False}
    ) == {
        "content": ["done"],
        "details": {"changed": True},
        "isError": True,
    }
    assert await runner.emit_user_bash({"type": "user_bash", "command": "rm -rf /"}) == {
        "block": True,
        "reason": "nope",
    }
    assert await runner.emit({"type": "session_before_switch", "sessionPath": "next"}) == {
        "cancel": True,
        "reason": "stop",
    }

    runner.invalidate()
    with pytest.raises(RuntimeError):
        _ = ctx.cwd

    assert any(error.event == "message_end" for error in errors)
    assert any(error.event == "input" and error.error == "boom" for error in errors)
    assert any(error.event == "input" and error.stack for error in errors)


@pytest.mark.asyncio
async def test_extension_runner_before_agent_start_ctx_uses_latest_system_prompt() -> None:
    seen_prompts: list[str] = []

    async def factory(api: Any) -> None:
        def first_handler(event: dict[str, Any], _ctx: Any) -> dict[str, Any]:
            return {"systemPrompt": event["systemPrompt"] + "::one"}

        def second_handler(_event: dict[str, Any], ctx: Any) -> dict[str, Any]:
            prompt = ctx.getSystemPrompt()
            seen_prompts.append(prompt)
            return {"systemPrompt": prompt + "::two"}

        api.on("before_agent_start", first_handler)
        api.on("before_agent_start", second_handler)

    extension = await load_extension_from_factory(
        factory,
        "/tmp/project",
        create_event_bus(),
        create_extension_runtime(),
        extension_path="<inline:before-agent>",
    )
    runner = ExtensionRunner(extensions=[extension], cwd="runner-cwd")
    runner.bind_core(
        {
            "sendMessage": lambda message, options=None: None,
            "sendUserMessage": lambda content, options=None: None,
            "appendEntry": lambda custom_type, data=None: None,
            "setSessionName": lambda name: None,
            "getSessionName": lambda: "demo",
            "setLabel": lambda entry_id, label: None,
            "getActiveTools": lambda: [],
            "getAllTools": lambda: [],
            "setActiveTools": lambda names: None,
            "refreshTools": lambda: None,
            "getCommands": lambda: [],
            "setModel": _set_model_true,
            "getThinkingLevel": lambda: "off",
            "setThinkingLevel": lambda level: None,
        },
        {
            "getModel": lambda: None,
            "isIdle": lambda: True,
            "getSignal": lambda: None,
            "abort": lambda: None,
            "hasPendingMessages": lambda: False,
            "shutdown": lambda: None,
            "getContextUsage": lambda: None,
            "compact": lambda options=None: None,
            "getSystemPrompt": lambda: "stale-base",
        },
    )

    try:
        result = await runner.emit_before_agent_start("prompt", None, "system", {"cwd": "/tmp"})

        assert seen_prompts == ["system::one"]
        assert result == {"messages": None, "systemPrompt": "system::one::two"}
    finally:
        runner.invalidate("done")


def test_extension_runner_resolves_commands_and_shortcuts_conflicts() -> None:
    extension_one = _extension_with_command_and_shortcut("<ext:one>", "demo", "ctrl+x")
    extension_two = _extension_with_command_and_shortcut("<ext:two>", "demo", "ctrl+x")
    extension_reserved = _extension_with_command_and_shortcut("<ext:reserved>", "other", "ctrl+c")

    runner = ExtensionRunner(extensions=[extension_one, extension_two, extension_reserved])

    commands = runner.get_registered_commands()
    assert [command.invocationName for command in commands] == ["demo:1", "demo:2", "other"]

    shortcuts = runner.get_shortcuts(
        {
            "editor.other": "ctrl+x",
            "app.interrupt": "ctrl+c",
        }
    )
    assert list(shortcuts) == ["ctrl+x"]
    assert shortcuts["ctrl+x"].extensionPath == "<ext:two>"
    messages = [diagnostic.message for diagnostic in runner.get_shortcut_diagnostics()]
    assert any("built-in shortcut for editor.other" in message for message in messages)
    assert any("registered by both <ext:one> and <ext:two>" in message for message in messages)
    assert any("conflicts with built-in shortcut" in message for message in messages)


@pytest.mark.asyncio
async def test_extension_runner_matches_ts_nullish_and_warning_behaviour(capsys: pytest.CaptureFixture[str]) -> None:
    async def factory(api: Any) -> None:
        api.on("context", lambda event, ctx: {"messages": []})
        api.on("tool_result", lambda event, ctx: {"details": None})
        api.on("input", lambda event, ctx: {"action": "transform", "text": "changed", "images": None})

    extension = await load_extension_from_factory(
        factory,
        "/tmp/project",
        create_event_bus(),
        create_extension_runtime(),
        extension_path="<inline:nullish>",
    )
    shortcut_extension = _extension_with_command_and_shortcut("<ext:warn>", "demo", "ctrl+c")
    runner = ExtensionRunner(extensions=[extension, shortcut_extension], cwd="runner-cwd")

    runner.bind_core(
        {
            "sendMessage": lambda message, options=None: None,
            "sendUserMessage": lambda content, options=None: None,
            "appendEntry": lambda custom_type, data=None: None,
            "setSessionName": lambda name: None,
            "getSessionName": lambda: "demo",
            "setLabel": lambda entry_id, label: None,
            "getActiveTools": lambda: [],
            "getAllTools": lambda: [],
            "setActiveTools": lambda names: None,
            "refreshTools": lambda: None,
            "getCommands": lambda: [],
            "setModel": _set_model_true,
            "getThinkingLevel": lambda: "off",
            "setThinkingLevel": lambda level: None,
        },
        {
            "getModel": lambda: None,
            "isIdle": lambda: True,
            "getSignal": lambda: None,
            "abort": lambda: None,
            "hasPendingMessages": lambda: False,
            "shutdown": lambda: None,
            "getContextUsage": lambda: None,
            "compact": lambda options=None: None,
            "getSystemPrompt": lambda: "prompt",
        },
    )

    assert await runner.emit_context([{"role": "user", "content": "keep"}]) == [{"role": "user", "content": "keep"}]
    assert await runner.emit_tool_result(
        {"type": "tool_result", "content": ["orig"], "details": {"x": 1}, "isError": False}
    ) == {
        "content": ["orig"],
        "details": None,
        "isError": False,
    }
    images = [{"kind": "image"}]
    assert await runner.emit_input("hello", images, "interactive") == {
        "action": "transform",
        "text": "changed",
        "images": images,
    }

    runner.get_shortcuts({"app.interrupt": "ctrl+c"})
    warning_output = capsys.readouterr().err
    assert "conflicts with built-in shortcut" in warning_output

    runner.invalidate()
    with pytest.raises(RuntimeError, match=r"ctx\\.newSession\\(\\)") as excinfo:
        _ = runner.create_context().cwd
    assert "withSession" in str(excinfo.value)


async def _set_model_true(_model: Any) -> bool:
    return True


def _extension_with_command_and_shortcut(path: str, command_name: str, shortcut: str) -> Any:
    source_info = _source_info(path)
    return type(
        "Ext",
        (),
        {
            "path": path,
            "resolvedPath": path,
            "sourceInfo": source_info,
            "handlers": {},
            "tools": {},
            "messageRenderers": {},
            "commands": {
                command_name: type(
                    "Cmd",
                    (),
                    {
                        "name": command_name,
                        "sourceInfo": source_info,
                        "description": f"{command_name} command",
                        "getArgumentCompletions": None,
                        "handler": lambda args, ctx: None,
                    },
                )()
            },
            "flags": {},
            "shortcuts": {
                shortcut: type(
                    "Shortcut",
                    (),
                    {
                        "shortcut": shortcut,
                        "extensionPath": path,
                        "description": "shortcut",
                        "handler": lambda ctx: None,
                    },
                )()
            },
            "skillPaths": [],
            "promptPaths": [],
            "themePaths": [],
            "systemPrompt": None,
            "appendSystemPrompt": [],
        },
    )()


def _source_info(path: str) -> Any:
    return type(
        "SourceInfo",
        (),
        {
            "path": path,
            "source": "local",
            "scope": "temporary",
            "origin": "top-level",
            "baseDir": None,
        },
    )()
