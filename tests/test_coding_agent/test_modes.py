from __future__ import annotations

import asyncio
import io
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import pytest
from harnify_ai.types import Model
from harnify_coding_agent.core.agent_session import SessionStats, SessionTokenStats
from harnify_coding_agent.core.compaction import CompactionResult
from harnify_coding_agent.main import main
import harnify_coding_agent.modes as modes_package
import harnify_coding_agent.modes.print_mode as print_mode_module
import harnify_coding_agent.modes.rpc.jsonl as rpc_jsonl_module
import harnify_coding_agent.modes.rpc.rpc_mode as rpc_mode_module
from harnify_coding_agent.modes.print_mode import run_print_mode
from harnify_coding_agent.modes.rpc import JsonlLineBuffer, RpcClient, run_rpc_mode
import harnify_coding_agent.modes.rpc.rpc_client as rpc_client_module


@dataclass
class _FakeSkill:
    name: str
    description: str
    sourceInfo: dict[str, Any]


@dataclass
class _FakeTemplate:
    name: str
    description: str
    sourceInfo: dict[str, Any]


@dataclass
class _FakeCommand:
    invocationName: str
    description: str
    sourceInfo: dict[str, Any]


class _FakeExtensionRunner:
    def __init__(self) -> None:
        self._commands = [_FakeCommand("demo", "Demo command", {"path": "<extension:demo>"})]

    def get_registered_commands(self) -> list[_FakeCommand]:
        return list(self._commands)


def test_rpc_client_module_exports_match_ts_surface() -> None:
    assert rpc_client_module.__all__ == ["ModelInfo", "RpcClient", "RpcClientOptions", "RpcEventListener"]


def test_rpc_client_options_surface_matches_ts() -> None:
    assert list(rpc_client_module.RpcClientOptions.__dataclass_fields__) == [
        "cliPath",
        "cwd",
        "env",
        "provider",
        "model",
        "args",
    ]


def test_rpc_client_camel_method_surface_matches_ts() -> None:
    expected_aliases = {
        "onEvent": "on_event",
        "getStderr": "get_stderr",
        "followUp": "follow_up",
        "newSession": "new_session",
        "getState": "get_state",
        "setModel": "set_model",
        "cycleModel": "cycle_model",
        "getAvailableModels": "get_available_models",
        "setThinkingLevel": "set_thinking_level",
        "cycleThinkingLevel": "cycle_thinking_level",
        "setSteeringMode": "set_steering_mode",
        "setFollowUpMode": "set_follow_up_mode",
        "setAutoCompaction": "set_auto_compaction",
        "setAutoRetry": "set_auto_retry",
        "abortRetry": "abort_retry",
        "abortBash": "abort_bash",
        "getSessionStats": "get_session_stats",
        "exportHtml": "export_html",
        "switchSession": "switch_session",
        "getForkMessages": "get_fork_messages",
        "getLastAssistantText": "get_last_assistant_text",
        "setSessionName": "set_session_name",
        "getMessages": "get_messages",
        "getCommands": "get_commands",
        "waitForIdle": "wait_for_idle",
        "collectEvents": "collect_events",
        "promptAndWait": "prompt_and_wait",
    }
    for camel_name, snake_name in expected_aliases.items():
        assert getattr(RpcClient, camel_name) is getattr(RpcClient, snake_name)


def test_modes_package_exports_match_ts_surface() -> None:
    expected = [
        "InteractiveMode",
        "InteractiveModeOptions",
        "PrintModeOptions",
        "runPrintMode",
        "ModelInfo",
        "RpcClient",
        "RpcClientOptions",
        "RpcEventListener",
        "runRpcMode",
        "RpcCommand",
        "RpcResponse",
        "RpcSessionState",
    ]
    assert modes_package.__all__ == expected
    for name in expected:
        assert getattr(modes_package, name) is not None


def test_print_mode_module_exports_match_ts_surface() -> None:
    assert print_mode_module.__all__ == ["PrintModeOptions", "runPrintMode"]


def test_rpc_jsonl_module_exports_match_ts_surface() -> None:
    assert rpc_jsonl_module.__all__ == ["attachJsonlLineReader", "serializeJsonLine"]


def test_rpc_mode_module_exports_match_ts_surface() -> None:
    assert rpc_mode_module.__all__ == [
        "RpcCommand",
        "RpcExtensionUIRequest",
        "RpcExtensionUIResponse",
        "RpcResponse",
        "RpcSessionState",
        "runRpcMode",
    ]


def _fake_model(provider: str, model_id: str) -> Model[Any]:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages" if provider == "anthropic" else "openai-responses",
        provider=provider,
        baseUrl="https://example.test",
        reasoning=True,
        input=["text"],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=200_000,
        maxTokens=8_192,
    )


class _FakeSessionManager:
    def __init__(self) -> None:
        self._header = {"type": "session", "id": "session-1"}
        self._leaf_id = "leaf-1"

    def getHeader(self) -> dict[str, Any]:
        return dict(self._header)

    def getLeafId(self) -> str:
        return self._leaf_id


class _FakeSession:
    def __init__(self) -> None:
        self.sessionManager = _FakeSessionManager()
        self.state = type("State", (), {"messages": []})()
        self.agent = type("Agent", (), {"waitForIdle": _noop_async})()
        self.model = _fake_model("anthropic", "claude")
        self.thinkingLevel = "high"
        self.isStreaming = False
        self.isCompacting = False
        self.steeringMode = "one-at-a-time"
        self.followUpMode = "one-at-a-time"
        self.sessionFile = "/tmp/session.jsonl"
        self.sessionId = "session-1"
        self.sessionName = "Named"
        self.autoCompactionEnabled = True
        self.autoRetryEnabled = True
        self.isRetrying = False
        self.retryAttempt = 0
        self.pendingMessageCount = 0
        self.messages: list[dict[str, Any]] = []
        self.promptTemplates = [_FakeTemplate("summarize", "Summarize", {"path": "<prompt:summarize>"})]
        self.resourceLoader = type(
            "Loader",
            (),
            {"getSkills": lambda self: {"skills": [_FakeSkill("shell", "Shell skill", {"path": "<skill:shell>"})]}},
        )()
        self.extensionRunner = _FakeExtensionRunner()
        self.modelRegistry = type(
            "Registry",
            (),
            {
                "find": lambda self, provider, model_id: _fake_model(provider, model_id),
                "hasConfiguredAuth": lambda self, _model: True,
                "getAvailable": lambda self: [
                    _fake_model("anthropic", "claude"),
                    _fake_model("openai", "gpt"),
                ],
            },
        )()
        self._listener = None
        self._extension_bindings: dict[str, Any] | None = None
        self._navigate_calls: list[tuple[str, dict[str, Any]]] = []
        self._reload_calls = 0
        self._prompt_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def bindExtensions(self, bindings: dict[str, Any]) -> None:
        self._extension_bindings = bindings
        return None

    def subscribe(self, listener):
        self._listener = listener

        def unsubscribe() -> None:
            self._listener = None

        return unsubscribe

    async def prompt(self, text: str, options: dict[str, Any] | None = None) -> None:
        self._prompt_calls.append((text, options))
        if callable((options or {}).get("preflightResult")):
            options["preflightResult"](True)
        self.state.messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"answer:{text}"}],
                "stopReason": "stop",
            }
        )
        self.messages = list(self.state.messages)
        if self._listener is not None:
            self._listener({"type": "agent_end", "messages": list(self.messages)})

    async def navigateTree(self, target_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._navigate_calls.append((target_id, dict(options or {})))
        return {"cancelled": False}

    async def reload(self) -> None:
        self._reload_calls += 1

    async def steer(self, text: str, _images: Any = None) -> None:
        self.pendingMessageCount += 1
        self._last_steer = text

    async def followUp(self, text: str, _images: Any = None) -> None:
        self.pendingMessageCount += 1
        self._last_follow_up = text

    async def abort(self) -> None:
        return None

    async def setModel(self, model: Any) -> None:
        self.model = model

    async def cycleModel(self) -> dict[str, Any]:
        return {"model": {"provider": "openai", "id": "gpt"}, "thinkingLevel": "medium", "isScoped": False}

    def setThinkingLevel(self, level: str) -> None:
        self.thinkingLevel = level

    def cycleThinkingLevel(self) -> dict[str, Any] | None:
        self.thinkingLevel = "medium"
        return "medium"

    def setSteeringMode(self, mode: str) -> None:
        self.steeringMode = mode

    def setFollowUpMode(self, mode: str) -> None:
        self.followUpMode = mode

    def setAutoCompactionEnabled(self, enabled: bool) -> None:
        self.autoCompactionEnabled = enabled

    def setAutoRetryEnabled(self, enabled: bool) -> None:
        self.autoRetryEnabled = enabled
        self._auto_retry = enabled

    def abortRetry(self) -> None:
        self.isRetrying = False
        self._retry_aborted = True

    async def compact(self, _custom_instructions: str | None = None) -> CompactionResult:
        return CompactionResult(
            summary="## Goal\nCompacted",
            firstKeptEntryId="entry-1",
            tokensBefore=3,
        )

    async def executeBash(self, command: str) -> Any:
        return type(
            "BashResult",
            (),
            {
                "output": f"ran:{command}",
                "exitCode": 0,
                "cancelled": False,
                "truncated": False,
                "fullOutputPath": None,
            },
        )()

    def abortBash(self) -> None:
        self._bash_aborted = True

    def getSessionStats(self) -> SessionStats:
        return SessionStats(
            sessionFile=self.sessionFile,
            sessionId=self.sessionId,
            userMessages=1,
            assistantMessages=1,
            toolCalls=0,
            toolResults=0,
            totalMessages=2,
            tokens=SessionTokenStats(input=1, output=2, cacheRead=0, cacheWrite=0, total=3),
            cost=0.25,
            contextUsage={"tokens": 3, "percent": 0.1},
        )

    async def exportToHtml(self, outputPath: str | None = None) -> str:
        return outputPath or "/tmp/export.html"

    def getUserMessagesForForking(self) -> list[dict[str, str]]:
        return [{"entryId": "1", "text": "hello"}]

    def getLastAssistantText(self) -> str | None:
        if not self.state.messages:
            return None
        return self.state.messages[-1]["content"][0]["text"]

    def setSessionName(self, name: str) -> None:
        self.sessionName = name


class _FakeRuntime:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.disposed = False
        self.rebind = None
        self.services = type(
            "Services",
            (),
            {
                "settingsManager": type(
                    "Settings",
                    (),
                    {
                        "getTheme": lambda self: None,
                        "getImageAutoResize": lambda self: True,
                        "getHttpIdleTimeoutMs": lambda self: 30_000,
                    },
                )(),
                "resourceLoader": object(),
            },
        )()
        self.diagnostics = []
        self.modelFallbackMessage = None

    async def dispose(self) -> None:
        self.disposed = True

    def setRebindSession(self, fn) -> None:
        self.rebind = fn

    async def newSession(self, _options: dict[str, Any] | None = None) -> dict[str, bool]:
        return {"cancelled": False}

    async def switchSession(self, _session_path: str, _options: dict[str, Any] | None = None) -> dict[str, bool]:
        return {"cancelled": False}

    async def fork(self, _entry_id: str, _options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"cancelled": False, "selectedText": "forked"}


async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
    return None


class _FakeReadable:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Any]] = {"data": [], "end": []}

    def on(self, event: str, listener: Any) -> None:
        self._listeners.setdefault(event, []).append(listener)

    def off(self, event: str, listener: Any) -> None:
        listeners = self._listeners.get(event, [])
        if listener in listeners:
            listeners.remove(listener)

    def emit(self, event: str, *args: Any) -> None:
        for listener in list(self._listeners.get(event, [])):
            listener(*args)

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(event, []))


class _PausableInputStream:
    def __init__(self, chunks: list[str | bytes]) -> None:
        self._chunks = list(chunks)
        self.paused = False

    def read(self, _size: int) -> str | bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return ""

    def pause(self) -> None:
        self.paused = True


def test_jsonl_line_buffer_preserves_strict_lf_framing() -> None:
    reader = JsonlLineBuffer()
    lines = reader.feed(b'{"a":"one\xe2\x80\xa8two"}\r\n{"b":2}')
    assert lines == ['{"a":"one\u2028two"}']
    assert reader.end() == ['{"b":2}']


def test_rpc_jsonl_serialization_matches_ts_compact_unicode_behavior() -> None:
    line = rpc_jsonl_module.serializeJsonLine({"a": "é", "b": "one\u2028two", "c": "one\u2029two"})
    assert line == '{"a":"é","b":"one\u2028two","c":"one\u2029two"}\n'


def test_attach_jsonl_line_reader_matches_ts_listener_lifecycle() -> None:
    readable = _FakeReadable()
    lines: list[str] = []

    detach = rpc_jsonl_module.attachJsonlLineReader(readable, lines.append)
    assert readable.listener_count("data") == 1
    assert readable.listener_count("end") == 1

    readable.emit("data", b'{"a":"one\xe2\x80\xa8two"}\r\n{"b":2')
    assert lines == ['{"a":"one\u2028two"}']
    readable.emit("end")
    assert lines == ['{"a":"one\u2028two"}', '{"b":2']

    detach()
    assert readable.listener_count("data") == 0
    assert readable.listener_count("end") == 0


@pytest.mark.asyncio
async def test_print_mode_text_and_json(monkeypatch) -> None:
    runtime = _FakeRuntime()
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    exit_code = await run_print_mode(runtime, {"mode": "text", "initialMessage": "hello"})
    assert exit_code == 0
    assert "answer:hello" in stdout.getvalue()
    assert runtime.disposed is True

    runtime = _FakeRuntime()
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    exit_code = await run_print_mode(runtime, {"mode": "json", "initialMessage": "hello"})
    raw_lines = stdout.getvalue().splitlines()
    lines = [json.loads(line) for line in raw_lines]
    assert exit_code == 0
    assert raw_lines[0] == '{"type":"session","id":"session-1"}'
    assert raw_lines[1] == '{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"answer:hello"}],"stopReason":"stop"}]}'
    assert lines[0]["type"] == "session"
    assert lines[1]["type"] == "agent_end"


@pytest.mark.asyncio
async def test_print_mode_rebind_actions_match_ts(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    exit_code = await run_print_mode(runtime, {"mode": "text"})
    assert exit_code == 0

    bindings = runtime.session._extension_bindings
    assert bindings is not None
    actions = bindings["commandContextActions"]

    fork_result = await actions["fork"]("entry-1", {"label": "fork"})
    assert fork_result == {"cancelled": False}

    navigate_result = await actions["navigateTree"](
        "entry-2",
        {
            "summarize": True,
            "customInstructions": "custom",
            "replaceInstructions": "replace",
            "label": "branch",
        },
    )
    assert navigate_result == {"cancelled": False}
    assert runtime.session._navigate_calls == [
        (
            "entry-2",
            {
                "summarize": True,
                "customInstructions": "custom",
                "replaceInstructions": "replace",
                "label": "branch",
            },
        )
    ]

    await actions["reload"]()
    assert runtime.session._reload_calls == 1


@pytest.mark.asyncio
async def test_print_mode_registers_and_restores_signal_handlers(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    calls: list[tuple[int, Any]] = []

    def fake_getsignal(sig: int) -> str:
        return f"previous:{sig}"

    def fake_signal(sig: int, handler: Any) -> None:
        calls.append((sig, handler))

    monkeypatch.setattr(print_mode_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(print_mode_module.signal, "signal", fake_signal)

    exit_code = await run_print_mode(runtime, {"mode": "text"})
    assert exit_code == 0

    expected_signals = [signal.SIGTERM]
    sighup = getattr(signal, "SIGHUP", None)
    if os.name != "nt" and sighup is not None:
        expected_signals.append(sighup)

    registered = calls[: len(expected_signals)]
    restored = calls[len(expected_signals) :]
    assert [sig for sig, _handler in registered] == expected_signals
    assert all(callable(handler) for _sig, handler in registered)
    assert restored == [(sig, f"previous:{sig}") for sig in expected_signals]


@pytest.mark.asyncio
async def test_print_mode_preserves_initial_images_nullish_shape(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    exit_code = await run_print_mode(runtime, {"mode": "text", "initialMessage": "hello"})
    assert exit_code == 0
    assert runtime.session._prompt_calls[0] == ("hello", {"images": None})


@pytest.mark.asyncio
async def test_rpc_mode_handles_basic_commands(monkeypatch) -> None:
    runtime = _FakeRuntime()
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    input_stream = io.StringIO(
        "\n".join(
            [
                json.dumps({"id": "1", "type": "get_state"}),
                json.dumps({"id": "2", "type": "set_session_name", "name": "renamed"}),
                json.dumps({"id": "3", "type": "get_commands"}),
                json.dumps({"id": "4", "type": "get_last_assistant_text"}),
                json.dumps({"id": "5", "type": "get_session_stats"}),
                json.dumps({"id": "6", "type": "compact"}),
                json.dumps({"id": "7", "type": "set_auto_retry", "enabled": False}),
                json.dumps({"id": "8", "type": "abort_retry"}),
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: input_stream)
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert payloads[0]["command"] == "get_state"
    assert payloads[0]["data"]["sessionId"] == "session-1"
    assert payloads[0]["data"]["model"]["id"] == "claude"
    assert payloads[1]["command"] == "set_session_name"
    assert payloads[2]["command"] == "get_commands"
    assert payloads[2]["data"]["commands"][0]["source"] == "extension"
    assert payloads[3]["command"] == "get_last_assistant_text"
    assert payloads[4]["command"] == "get_session_stats"
    assert payloads[4]["data"]["tokens"]["total"] == 3
    assert payloads[5]["command"] == "compact"
    assert payloads[5]["data"]["firstKeptEntryId"] == "entry-1"
    assert payloads[6]["command"] == "set_auto_retry"
    assert payloads[7]["command"] == "abort_retry"
    assert runtime.session.autoRetryEnabled is False
    assert runtime.session._retry_aborted is True


@pytest.mark.asyncio
async def test_rpc_mode_rebind_actions_match_ts(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: io.StringIO(""))
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0

    bindings = runtime.session._extension_bindings
    assert bindings is not None
    actions = bindings["commandContextActions"]

    fork_result = await actions["fork"]("entry-1", {"position": "after"})
    assert fork_result == {"cancelled": False}

    navigate_result = await actions["navigateTree"](
        "entry-2",
        {
            "summarize": True,
            "customInstructions": "custom",
            "replaceInstructions": "replace",
            "label": "branch",
        },
    )
    assert navigate_result == {"cancelled": False}
    assert runtime.session._navigate_calls == [
        (
            "entry-2",
            {
                "summarize": True,
                "customInstructions": "custom",
                "replaceInstructions": "replace",
                "label": "branch",
            },
        )
    ]

    await actions["reload"]()
    assert runtime.session._reload_calls == 1


@pytest.mark.asyncio
async def test_rpc_mode_uses_ts_async_model_registry_path(monkeypatch) -> None:
    runtime = _FakeRuntime()
    runtime.session.modelRegistry = type(
        "Registry",
        (),
        {
            "getAvailable": lambda self: _registry_models(),
        },
    )()
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", io.StringIO())

    async def _registry_models() -> list[Model[Any]]:
        return [_fake_model("anthropic", "claude"), _fake_model("openai", "gpt")]

    input_stream = io.StringIO(
        "\n".join(
            [
                json.dumps({"id": "1", "type": "set_model", "provider": "openai", "modelId": "gpt"}),
                json.dumps({"id": "2", "type": "get_available_models"}),
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: input_stream)
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert payloads[0]["command"] == "set_model"
    assert payloads[0]["data"]["id"] == "gpt"
    assert payloads[1]["command"] == "get_available_models"
    assert [model["id"] for model in payloads[1]["data"]["models"]] == ["claude", "gpt"]


@pytest.mark.asyncio
async def test_rpc_ui_context_theme_and_signal_abort_match_ts() -> None:
    emitted: list[dict[str, Any]] = []
    pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
    ctx = rpc_mode_module._RpcUIContext(emitted.append, pending)

    class _Signal:
        def __init__(self) -> None:
            self.aborted = False
            self._listener = None

        def addEventListener(self, event: str, listener: Any, _opts: Any = None) -> None:
            assert event == "abort"
            self._listener = listener

        def removeEventListener(self, event: str, listener: Any) -> None:
            if event == "abort" and self._listener == listener:
                self._listener = None

        def trigger(self) -> None:
            self.aborted = True
            if self._listener is not None:
                self._listener()

    signal_obj = _Signal()
    opts = type("Opts", (), {"timeout": None, "signal": signal_obj})()

    task = asyncio.create_task(ctx.select("Pick", ["a"], opts))
    await asyncio.sleep(0)
    signal_obj.trigger()

    assert await task is None
    assert emitted[0]["method"] == "select"
    assert pending == {}
    assert ctx.theme is rpc_mode_module.theme


@pytest.mark.asyncio
async def test_rpc_mode_unknown_command_omits_id_like_ts(monkeypatch) -> None:
    runtime = _FakeRuntime()
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", io.StringIO())

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: io.StringIO('{"id":"x","type":"unknown"}\n'))
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0
    payload = json.loads(stdout.getvalue().splitlines()[0])
    assert payload["command"] == "unknown"
    assert "id" not in payload


@pytest.mark.asyncio
async def test_rpc_mode_non_object_json_uses_ts_raw_cast_path(monkeypatch) -> None:
    runtime = _FakeRuntime()
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", io.StringIO())

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: io.StringIO("[1,2,3]\n"))
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0
    payload = json.loads(stdout.getvalue().splitlines()[0])
    assert payload["error"] == "Unknown command: undefined"
    assert "id" not in payload
    assert "command" not in payload


@pytest.mark.asyncio
async def test_rpc_mode_registers_and_restores_signal_handlers(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    calls: list[tuple[int, Any]] = []

    def fake_getsignal(sig: int) -> str:
        return f"previous:{sig}"

    def fake_signal(sig: int, handler: Any) -> None:
        calls.append((sig, handler))

    monkeypatch.setattr(rpc_mode_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(rpc_mode_module.signal, "signal", fake_signal)

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: io.StringIO(""))
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0

    expected_signals = [signal.SIGTERM]
    sighup = getattr(signal, "SIGHUP", None)
    if os.name != "nt" and sighup is not None:
        expected_signals.append(sighup)

    registered = calls[: len(expected_signals)]
    restored = calls[len(expected_signals) :]
    assert [sig for sig, _handler in registered] == expected_signals
    assert all(callable(handler) for _sig, handler in registered)
    assert restored == [(sig, f"previous:{sig}") for sig in expected_signals]


@pytest.mark.asyncio
async def test_rpc_mode_cleanup_pauses_input_stream(monkeypatch) -> None:
    runtime = _FakeRuntime()
    stream = _PausableInputStream([""])
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: stream)
    exit_code = await run_rpc_mode(runtime)
    assert exit_code == 0
    assert stream.paused is True


@pytest.mark.asyncio
async def test_rpc_mode_shutdown_handler_triggers_immediate_cleanup(monkeypatch) -> None:
    runtime = _FakeRuntime()
    stream = _PausableInputStream(['{"id":"1","type":"get_state"}\n', ""])
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", io.StringIO())

    monkeypatch.setattr(rpc_mode_module, "_get_rpc_input_stream", lambda: stream)
    task = asyncio.create_task(run_rpc_mode(runtime))
    while runtime.session._extension_bindings is None:
        await asyncio.sleep(0)
    runtime.session._extension_bindings["shutdownHandler"]()

    exit_code = await task
    assert exit_code == 0
    assert stream.paused is True
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert payloads[0]["command"] == "get_state"


@pytest.mark.asyncio
async def test_main_dispatches_print_and_rpc(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("harnify_coding_agent.main.run_migrations", lambda _cwd: None)
    monkeypatch.setattr("harnify_coding_agent.main.create_session_manager", _fake_create_session_manager)
    monkeypatch.setattr("harnify_coding_agent.main.create_agent_session_runtime", _fake_create_runtime)

    calls: list[tuple[str, Any]] = []

    async def fake_print_mode(runtime: Any, options: dict[str, Any]) -> int:
        calls.append(("print", options))
        return 7

    async def fake_rpc_mode(runtime: Any) -> int:
        calls.append(("rpc", runtime))
        return 9

    monkeypatch.setattr("harnify_coding_agent.main.run_print_mode", fake_print_mode)
    monkeypatch.setattr("harnify_coding_agent.main.run_rpc_mode", fake_rpc_mode)
    monkeypatch.setattr("harnify_coding_agent.main.configureHttpDispatcher", lambda _ms: None)
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())

    assert await main(["-p", "hello"]) == 7
    assert calls[0][0] == "print"
    assert calls[0][1]["initialMessage"] == "hello"

    assert await main(["--mode", "rpc"]) == 9
    assert calls[1][0] == "rpc"


@pytest.mark.asyncio
async def test_main_initializes_theme_and_prints_timings_for_noninteractive_modes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("harnify_coding_agent.main.run_migrations", lambda _cwd: None)
    monkeypatch.setattr("harnify_coding_agent.main.create_session_manager", _fake_create_session_manager)
    monkeypatch.setattr("harnify_coding_agent.main.create_agent_session_runtime", _fake_create_runtime)
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())

    calls: list[tuple[str, Any]] = []

    async def fake_print_mode(runtime: Any, options: dict[str, Any]) -> int:
        calls.append(("print", options))
        return 0

    async def fake_rpc_mode(runtime: Any) -> int:
        calls.append(("rpc", runtime))
        return 0

    monkeypatch.setattr("harnify_coding_agent.main.run_print_mode", fake_print_mode)
    monkeypatch.setattr("harnify_coding_agent.main.run_rpc_mode", fake_rpc_mode)
    monkeypatch.setattr("harnify_coding_agent.main.configureHttpDispatcher", lambda _ms: calls.append(("dispatcher", _ms)))
    monkeypatch.setattr("harnify_coding_agent.main.init_theme", lambda name, enable=False: calls.append(("init", (name, enable))))
    monkeypatch.setattr("harnify_coding_agent.main.print_timings", lambda: calls.append(("timings", None)))
    monkeypatch.setattr("harnify_coding_agent.main.stop_theme_watcher", lambda: calls.append(("stop", None)))

    assert await main(["-p", "hello"]) == 0
    assert calls[:5] == [
        ("dispatcher", 30_000),
        ("init", (None, False)),
        ("timings", None),
        ("print", {"mode": "text", "messages": [], "initialMessage": "hello", "initialImages": None}),
        ("stop", None),
    ]

    calls.clear()
    assert await main(["--mode", "rpc"]) == 0
    assert calls == [
        ("dispatcher", 30_000),
        ("init", (None, False)),
        ("timings", None),
        ("rpc", ANY),
    ]


@pytest.mark.asyncio
async def test_main_dispatches_interactive(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("harnify_coding_agent.main.run_migrations", lambda _cwd: None)
    monkeypatch.setattr("harnify_coding_agent.main.create_session_manager", _fake_create_session_manager)
    monkeypatch.setattr("harnify_coding_agent.main.create_agent_session_runtime", _fake_create_runtime)
    monkeypatch.setattr("harnify_coding_agent.main.configureHttpDispatcher", lambda _ms: None)
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())

    captured: dict[str, Any] = {}

    class FakeInteractiveMode:
        def __init__(self, runtimeHost: Any = None, options: dict[str, Any] | None = None, **_kwargs: Any) -> None:
            captured["runtime"] = runtimeHost
            captured["options"] = options or {}

        async def run(self) -> int:
            captured["ran"] = True
            return 11

        def dispose(self) -> None:
            captured["disposed"] = True

    monkeypatch.setattr("harnify_coding_agent.main.InteractiveMode", FakeInteractiveMode)

    assert await main(["first", "second", "--verbose"]) == 11
    assert captured["ran"] is True
    assert "disposed" not in captured
    assert captured["runtime"].disposed is False
    assert captured["options"]["initialMessage"] == "first"
    assert captured["options"]["initialMessages"] == ["second"]
    assert captured["options"]["initialImages"] is None
    assert captured["options"]["modelFallbackMessage"] is None
    assert captured["options"]["verbose"] is True


@pytest.mark.asyncio
async def test_rpc_client_routes_responses_and_events() -> None:
    client = RpcClient()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    client._pending_requests["req_1"] = future

    events: list[dict[str, Any]] = []
    unsubscribe = client.onEvent(events.append)
    client._handle_line(json.dumps({"type": "response", "id": "req_1", "command": "get_state", "success": True}))
    assert await future == {"type": "response", "id": "req_1", "command": "get_state", "success": True}

    client._handle_line(json.dumps({"type": "agent_end", "messages": []}))
    client._handle_line(json.dumps(["non-object-event"]))
    unsubscribe()
    assert events == [{"type": "agent_end", "messages": []}, ["non-object-event"]]


@pytest.mark.asyncio
async def test_rpc_client_wait_helpers_match_ts_timeout_messages() -> None:
    client = RpcClient()
    client._stderr = "debug stderr"

    with pytest.raises(RuntimeError) as idle_error:
        await client.waitForIdle(timeout=0.001)
    assert str(idle_error.value) == "Timeout waiting for agent to become idle. Stderr: debug stderr"
    assert client._event_listeners == []

    with pytest.raises(RuntimeError) as collect_error:
        await client.collectEvents(timeout=0.001)
    assert str(collect_error.value) == "Timeout collecting events. Stderr: debug stderr"
    assert client._event_listeners == []


@pytest.mark.asyncio
async def test_rpc_client_stderr_passthrough_matches_ts(monkeypatch) -> None:
    client = RpcClient()

    class _Reader:
        def __init__(self) -> None:
            self._chunks = [b"one", b"two", b""]

        async def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    client.process = type("Proc", (), {"stderr": _Reader()})()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)

    await client._consume_stderr()

    assert client.getStderr() == "onetwo"
    assert stderr.getvalue() == "onetwo"


async def _fake_create_session_manager(*_args: Any, **_kwargs: Any) -> Any:
    class Manager:
        def getCwd(self) -> str:
            return os.getcwd()

    return Manager()


async def _fake_create_runtime(*_args: Any, **_kwargs: Any) -> Any:
    runtime = _FakeRuntime()
    runtime.services = type(
        "Services",
        (),
        {
            "settingsManager": type(
                "Settings",
                (),
                {
                    "getTheme": lambda self: None,
                    "getImageAutoResize": lambda self: True,
                    "getHttpIdleTimeoutMs": lambda self: 30_000,
                },
            )(),
            "resourceLoader": type(
                "Loader",
                (),
                {"getExtensions": lambda self: type("Ext", (), {"extensions": []})()},
            )(),
            "modelRegistry": object(),
        },
    )()
    return runtime
