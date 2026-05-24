from __future__ import annotations

from typing import Any

import pytest
from harnify_agent.types import AgentToolResult
from harnify_ai.types import TextContent
from harnify_coding_agent.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from harnify_coding_agent.core.extensions.runner import ExtensionRunner
from harnify_coding_agent.core.extensions.types import ToolDefinition


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
    extension = await load_extension_from_factory(
        factory,
        "/tmp",
        runtime=runtime,
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
async def test_extension_runner_emits_events_and_invalidates_context() -> None:
    errors: list[str] = []

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

    extension = await load_extension_from_factory(factory, "/tmp/project", extension_path="<inline:eventful>")
    runner = ExtensionRunner(
        extensions=[extension],
        contextFactory=lambda: {"cwd": "runner-cwd"},
    )
    runner.on_error(lambda error: errors.append(f"{error.event}:{error.error}"))

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

    assert any("message_end" in message for message in errors)
    assert any("input:boom" == message for message in errors)


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

    extension = await load_extension_from_factory(factory, "/tmp/project", extension_path="<inline:before-agent>")
    runner = ExtensionRunner(extensions=[extension], contextFactory=lambda: {"cwd": "runner-cwd"})
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
