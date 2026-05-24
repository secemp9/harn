from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from harnify_coding_agent.core.agent_session_runtime import (
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
    CreateAgentSessionRuntimeResult,
    SessionImportFileNotFoundError,
    createAgentSessionRuntime,
)
from harnify_coding_agent.core.agent_session_services import (
    createAgentSessionFromServices,
    createAgentSessionServices,
)
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.extensions.loader import create_extension_runtime
from harnify_coding_agent.core.extensions.types import (
    Extension,
    ExtensionFlag,
    LoadExtensionsResult,
    PendingProviderRegistration,
)
from harnify_coding_agent.core.resource_loader import DefaultResourceLoaderOptions
from harnify_coding_agent.core.session_manager import SessionManager
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.core.source_info import create_synthetic_source_info


@dataclass(slots=True)
class _StubAgentState:
    messages: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class _StubAgent:
    state: _StubAgentState = field(default_factory=_StubAgentState)


class _StubExtensionRunner:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}
        self.events: list[dict[str, Any]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def hasHandlers(self, event: str) -> bool:
        return bool(self.handlers.get(event))

    def has_handlers(self, event: str) -> bool:
        return self.hasHandlers(event)

    async def emit(self, event: dict[str, Any]) -> Any:
        self.events.append(dict(event))
        result = None
        for handler in self.handlers.get(event["type"], []):
            current = handler(event)
            if hasattr(current, "__await__"):
                current = await current
            if current is not None:
                result = current
                if current.get("cancel") is True:
                    return result
        return result


@dataclass(slots=True)
class _StubSession:
    sessionManager: SessionManager
    extensionRunner: _StubExtensionRunner
    sessionFile: str | None
    agent: _StubAgent = field(default_factory=_StubAgent)
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True

    def createReplacedSessionContext(self) -> dict[str, Any]:
        return {
            "cwd": self.sessionManager.getCwd(),
            "sessionFile": self.sessionFile,
        }


class _StubResourceLoader:
    def __init__(self, options: DefaultResourceLoaderOptions) -> None:
        self.options = options
        self.runtime = create_extension_runtime()
        self.runtime.pendingProviderRegistrations = [
            PendingProviderRegistration(
                name="ok-provider",
                config={"api": "responses"},
                extensionPath="/ext/ok.py",
            ),
            PendingProviderRegistration(
                name="broken-provider",
                config={"api": "responses"},
                extensionPath="/ext/broken.py",
            ),
        ]
        self.extensions = [
            Extension(
                path="/ext/flags.py",
                resolvedPath="/ext/flags.py",
                sourceInfo=create_synthetic_source_info(
                    "/ext/flags.py",
                    {"source": "test", "scope": "temporary", "origin": "top-level"},
                ),
                flags={
                    "feature": ExtensionFlag(
                        name="feature",
                        extensionPath="/ext/flags.py",
                        type="boolean",
                    ),
                    "mode": ExtensionFlag(
                        name="mode",
                        extensionPath="/ext/flags.py",
                        type="string",
                    ),
                },
            )
        ]
        self.reloaded = False

    async def reload(self) -> None:
        self.reloaded = True

    def getExtensions(self) -> LoadExtensionsResult:
        return LoadExtensionsResult(
            extensions=self.extensions,
            errors=[],
            runtime=self.runtime,
        )


class _StubModelRegistry:
    def __init__(self) -> None:
        self.registered: list[tuple[str, dict[str, Any]]] = []

    def registerProvider(self, name: str, config: dict[str, Any]) -> None:
        if name == "broken-provider":
            raise RuntimeError("boom")
        self.registered.append((name, config))


def _build_factory(
    seen: list[dict[str, Any]],
    *,
    event_handlers: dict[str, list[Any]] | None = None,
) -> Any:
    async def _factory(options: dict[str, Any]) -> CreateAgentSessionRuntimeResult:
        seen.append(dict(options))
        runner = _StubExtensionRunner()
        for event_name, handlers in (event_handlers or {}).items():
            for handler in handlers:
                runner.on(event_name, handler)
        if options.get("sessionStartEvent"):
            await runner.emit(dict(options["sessionStartEvent"]))
        session = _StubSession(
            sessionManager=options["sessionManager"],
            sessionFile=options["sessionManager"].getSessionFile(),
            extensionRunner=runner,
        )
        session.agent.state.messages = options["sessionManager"].buildSessionContext().messages
        return CreateAgentSessionRuntimeResult(
            session=session,
            services=AgentSessionServices(
                cwd=options["cwd"],
                agentDir=options["agentDir"],
                authStorage=AuthStorage.inMemory(),
                settingsManager=SettingsManager.inMemory(),
                modelRegistry=_StubModelRegistry(),  # type: ignore[arg-type]
                resourceLoader=_StubResourceLoader({"cwd": options["cwd"], "agentDir": options["agentDir"]}),
                diagnostics=[AgentSessionRuntimeDiagnostic(type="info", message="created")],
            ),
            diagnostics=[AgentSessionRuntimeDiagnostic(type="info", message="created")],
        )

    return _factory


@pytest.mark.asyncio
async def test_create_agent_session_services_applies_pending_providers_and_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = _StubModelRegistry()
    monkeypatch.setattr(
        "harnify_coding_agent.core.agent_session_services.DefaultResourceLoader",
        _StubResourceLoader,
    )

    services = await createAgentSessionServices(
        {
            "cwd": str(tmp_path),
            "agentDir": str(tmp_path / "agent"),
            "authStorage": AuthStorage.inMemory(),
            "settingsManager": SettingsManager.inMemory(),
            "modelRegistry": registry,  # type: ignore[typeddict-item]
            "extensionFlagValues": {"feature": True, "mode": "fast", "unknown": True},
        }
    )

    assert services.resourceLoader.reloaded is True  # type: ignore[attr-defined]
    assert registry.registered == [("ok-provider", {"api": "responses"})]
    assert services.resourceLoader.runtime.flagValues == {"feature": True, "mode": "fast"}  # type: ignore[attr-defined]
    assert any("Extension \"/ext/broken.py\" error: boom" == diagnostic.message for diagnostic in services.diagnostics)
    assert any("Unknown option: --unknown" == diagnostic.message for diagnostic in services.diagnostics)


@pytest.mark.asyncio
async def test_create_agent_session_from_services_uses_lazy_sdk_callable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_agent_session(options: dict[str, Any]) -> dict[str, Any]:
        captured.update(options)
        return {"session": "ok", "extensionsResult": None}

    monkeypatch.setattr(
        "harnify_coding_agent.core.agent_session_services.create_agent_session",
        fake_create_agent_session,
    )
    services = AgentSessionServices(
        cwd=str(tmp_path),
        agentDir=str(tmp_path / "agent"),
        authStorage=AuthStorage.inMemory(),
        settingsManager=SettingsManager.inMemory(),
        modelRegistry=_StubModelRegistry(),  # type: ignore[arg-type]
        resourceLoader=_StubResourceLoader({"cwd": str(tmp_path), "agentDir": str(tmp_path / "agent")}),
    )
    session_manager = SessionManager.inMemory(str(tmp_path))

    result = await createAgentSessionFromServices(
        {
            "services": services,
            "sessionManager": session_manager,
            "noTools": "builtin",
            "tools": ["read"],
        }
    )

    assert result["session"] == "ok"
    assert captured["cwd"] == str(tmp_path)
    assert captured["sessionManager"] is session_manager
    assert captured["noTools"] == "builtin"
    assert captured["tools"] == ["read"]


@pytest.mark.asyncio
async def test_agent_session_runtime_switch_new_fork_import_and_dispose(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    initial_manager = SessionManager.create(str(root))
    initial_manager.appendMessage({"role": "user", "content": "hello", "timestamp": 1})
    initial_manager.appendMessage(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "world"}],
            "api": "responses",
            "provider": "faux",
            "model": "faux-1",
            "usage": {
                "input": 1,
                "output": 1,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 2,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 2,
        }
    )

    switch_events: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        switch_events.append(dict(event))

    runtime = await createAgentSessionRuntime(
        _build_factory(
            seen,
            event_handlers={
                "session_before_switch": [record],
                "session_before_fork": [record],
                "session_shutdown": [record],
                "session_start": [record],
            },
        ),
        {
            "cwd": str(root),
            "agentDir": str(root),
            "sessionManager": initial_manager,
            "sessionStartEvent": {"type": "session_start", "reason": "startup"},
        },
    )

    assert switch_events == [{"type": "session_start", "reason": "startup"}]
    switch_events.clear()

    rebound: list[str] = []
    with_session_values: list[dict[str, Any]] = []
    runtime.setRebindSession(lambda session: _async_append(rebound, session.sessionManager.getCwd()))
    runtime.setBeforeSessionInvalidate(lambda: rebound.append("invalidate"))

    new_result = await runtime.newSession(
        {
            "withSession": lambda ctx: _async_append(with_session_values, dict(ctx)),
        }
    )
    assert new_result == {"cancelled": False}
    assert rebound[0] == "invalidate"
    assert rebound[1] == str(root)
    assert with_session_values[0]["cwd"] == str(root)
    assert switch_events[0] == {
        "type": "session_before_switch",
        "reason": "new",
        "targetSessionFile": None,
    }
    assert switch_events[1]["type"] == "session_shutdown"
    assert switch_events[2]["type"] == "session_start"
    new_session_file = runtime.session.sessionFile
    assert new_session_file is not None
    switch_events.clear()
    rebound.clear()

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_manager = SessionManager.create(str(other_dir))
    other_manager.appendMessage({"role": "user", "content": "other", "timestamp": 3})
    other_manager.appendMessage(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "reply"}],
            "api": "responses",
            "provider": "faux",
            "model": "faux-2",
            "usage": {
                "input": 1,
                "output": 1,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 2,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 4,
        }
    )

    switch_result = await runtime.switchSession(other_manager.getSessionFile())
    assert switch_result == {"cancelled": False}
    assert os.path.realpath(runtime.cwd) == os.path.realpath(str(other_dir))
    assert switch_events[0]["type"] == "session_before_switch"
    assert switch_events[1]["type"] == "session_shutdown"
    assert switch_events[2]["type"] == "session_start"
    switch_events.clear()

    user_entry_id = next(
        entry["id"]
        for entry in runtime.session.sessionManager.getEntries()
        if entry["type"] == "message" and entry["message"]["role"] == "user"
    )
    fork_result = await runtime.fork(user_entry_id)
    assert fork_result == {"cancelled": False, "selectedText": "other"}
    assert switch_events[0] == {
        "type": "session_before_fork",
        "entryId": user_entry_id,
        "position": "before",
    }
    assert switch_events[1]["type"] == "session_shutdown"
    assert switch_events[2]["type"] == "session_start"
    switch_events.clear()

    imported_source = tmp_path / "import.jsonl"
    imported_source.write_text(Path(other_manager.getSessionFile()).read_text(encoding="utf-8"), encoding="utf-8")
    import_result = await runtime.importFromJsonl(str(imported_source))
    assert import_result == {"cancelled": False}
    assert runtime.session.sessionFile is not None
    assert Path(runtime.session.sessionFile).name == imported_source.name

    await runtime.dispose()
    assert runtime.session.disposed is True


@pytest.mark.asyncio
async def test_agent_session_runtime_honors_cancellation_and_missing_import(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    initial_manager = SessionManager.create(str(root))
    initial_manager.appendMessage({"role": "user", "content": "hello", "timestamp": 1})
    initial_manager.appendMessage(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "world"}],
            "api": "responses",
            "provider": "faux",
            "model": "faux-1",
            "usage": {
                "input": 1,
                "output": 1,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 2,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 2,
        }
    )

    def cancel(event: dict[str, Any]) -> dict[str, Any]:
        return {"cancel": True}

    runtime = await createAgentSessionRuntime(
        _build_factory(
            [],
            event_handlers={
                "session_before_switch": [cancel],
                "session_before_fork": [cancel],
            },
        ),
        {
            "cwd": str(root),
            "agentDir": str(root),
            "sessionManager": initial_manager,
        },
    )

    current_file = runtime.session.sessionFile
    assert await runtime.newSession() == {"cancelled": True}
    assert runtime.session.sessionFile == current_file
    assert await runtime.fork("missing-entry", {"position": "at"}) == {"cancelled": True}
    with pytest.raises(SessionImportFileNotFoundError):
        await runtime.importFromJsonl(str(tmp_path / "missing.jsonl"))


async def _async_append(target: list[Any], value: Any) -> None:
    target.append(value)
