from __future__ import annotations

import time
from pathlib import Path

import pytest
from harnify_ai.types import AssistantMessage, Model, SimpleStreamOptions, TextContent
from harnify_ai.utils.event_stream import AssistantMessageEventStream
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.resource_loader import DefaultResourceLoader
from harnify_coding_agent.core.sdk import create_agent_session
from harnify_coding_agent.core.session_manager import SessionManager
from harnify_coding_agent.core.settings_manager import SettingsManager


def _built_in_model(provider: str = "anthropic") -> Model:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())
    return next(model for model in registry.getAll() if model.provider == provider)


def _safe_session_path(cwd: str) -> str:
    trimmed = cwd.lstrip("/\\")
    return f"--{trimmed.replace('/', '-').replace('\\\\', '-').replace(':', '-')}--"


def _done_stream(provider: str = "capture-provider", model_id: str = "capture-model") -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.end(
        AssistantMessage(
            role="assistant",
            content=[TextContent(text="ok")],
            api="openai-completions",
            provider=provider,
            model=model_id,
            usage={
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            stopReason="stop",
            timestamp=int(time.time() * 1000),
        )
    )
    return stream


@pytest.mark.asyncio
async def test_create_agent_session_uses_agent_dir_for_default_persisted_session_path(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir(parents=True)
    agent_dir.mkdir(parents=True)

    result = await create_agent_session(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "model": _built_in_model(),
        }
    )
    session = result["session"]
    try:
        expected_dir = agent_dir / "sessions" / _safe_session_path(str(cwd))
        session_dir = Path(session.sessionManager.getSessionDir())
        session_file = session.sessionManager.getSessionFile()

        assert session_dir == expected_dir
        assert session_file is not None
        assert str(session_file).startswith(f"{expected_dir}/")
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_create_agent_session_keeps_explicit_session_manager_override(tmp_path: Path) -> None:
    cwd = str(tmp_path / "project")
    agent_dir = str(tmp_path / "agent")
    session_manager = SessionManager.inMemory(cwd)

    result = await create_agent_session(
        {
            "cwd": cwd,
            "agentDir": agent_dir,
            "model": _built_in_model(),
            "sessionManager": session_manager,
        }
    )
    session = result["session"]
    try:
        assert session.sessionManager is session_manager
        assert session.sessionManager.isPersisted() is False
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_create_agent_session_derives_cwd_from_session_manager_when_omitted(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session-project"
    session_cwd.mkdir(parents=True)
    session_manager = SessionManager.inMemory(str(session_cwd))

    result = await create_agent_session(
        {
            "agentDir": str(tmp_path / "agent"),
            "model": _built_in_model(),
            "sessionManager": session_manager,
        }
    )
    session = result["session"]
    try:
        assert session.sessionManager is session_manager
        assert f"Current working directory: {session_cwd}" in session.systemPrompt

        bash_tool = next(tool for tool in session.agent.state.tools if tool.name == "bash")
        tool_result = await bash_tool.execute("test", {"command": "pwd"}, None, None)
        output = "".join(block.text for block in tool_result.content if getattr(block, "type", None) == "text")

        assert Path(output.strip()).resolve() == session_cwd.resolve()
    finally:
        session.dispose()


@pytest.mark.asyncio
async def test_create_agent_session_respects_no_tools_all_and_explicit_allowlist(tmp_path: Path) -> None:
    base_options = {
        "cwd": str(tmp_path / "project"),
        "agentDir": str(tmp_path / "agent"),
        "model": _built_in_model(),
        "sessionManager": SessionManager.inMemory(str(tmp_path / "project")),
    }

    no_tools_session = (await create_agent_session({**base_options, "noTools": "all"}))["session"]
    try:
        assert no_tools_session.getAllTools() == []
        assert no_tools_session.getActiveToolNames() == []
    finally:
        no_tools_session.dispose()

    allowlisted_session = (await create_agent_session({**base_options, "tools": ["read"]}))["session"]
    try:
        assert [tool.name for tool in allowlisted_session.getAllTools()] == ["read"]
        assert allowlisted_session.getActiveToolNames() == ["read"]
    finally:
        allowlisted_session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "base_url", "telemetry_enabled", "provider_headers", "request_headers", "expected"),
    [
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            True,
            None,
            None,
            {
                "HTTP-Referer": "https://pi.dev",
                "X-OpenRouter-Title": "pi",
                "X-OpenRouter-Categories": "cli-agent",
            },
        ),
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            False,
            None,
            None,
            {},
        ),
        (
            "custom-openrouter",
            "https://openrouter.ai/api/v1",
            True,
            None,
            None,
            {
                "HTTP-Referer": "https://pi.dev",
                "X-OpenRouter-Title": "pi",
                "X-OpenRouter-Categories": "cli-agent",
            },
        ),
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            True,
            {
                "HTTP-Referer": "https://provider.example",
                "X-OpenRouter-Categories": "provider-category",
            },
            {
                "X-OpenRouter-Title": "request-title",
            },
            {
                "HTTP-Referer": "https://provider.example",
                "X-OpenRouter-Title": "request-title",
                "X-OpenRouter-Categories": "provider-category",
            },
        ),
    ],
)
async def test_create_agent_session_openrouter_attribution_headers(
    tmp_path: Path,
    provider: str,
    base_url: str,
    telemetry_enabled: bool,
    provider_headers: dict[str, str] | None,
    request_headers: dict[str, str] | None,
    expected: dict[str, str],
) -> None:
    cwd = str(tmp_path / "project")
    agent_dir = str(tmp_path / "agent")
    settings_manager = SettingsManager.create(cwd, agent_dir)
    if not telemetry_enabled:
        settings_manager.setEnableInstallTelemetry(False)

    auth_storage = AuthStorage.create(str(Path(agent_dir) / "auth.json"))
    auth_storage.setRuntimeApiKey(provider, "test-api-key")
    model_registry = ModelRegistry.create(auth_storage, str(Path(agent_dir) / "models.json"))

    captured_options: list[SimpleStreamOptions | None] = []

    def capture_stream(
        _model: Model,
        _context: object,
        provider_options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        captured_options.append(provider_options)
        return _done_stream()

    model_registry.registerProvider("capture-provider", {"api": "openai-completions", "streamSimple": capture_stream})
    registered_providers = ["capture-provider"]
    if provider_headers:
        model_registry.registerProvider(provider, {"headers": provider_headers})
        registered_providers.append(provider)

    model = Model(
        id=f"{provider}-test-model",
        name=f"{provider} Test Model",
        api="openai-completions",
        provider=provider,
        baseUrl=base_url,
        reasoning=False,
        input=["text"],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=128000,
        maxTokens=4096,
    )

    session = (
        await create_agent_session(
            {
                "cwd": cwd,
                "agentDir": agent_dir,
                "model": model,
                "authStorage": auth_storage,
                "modelRegistry": model_registry,
                "settingsManager": settings_manager,
                "sessionManager": SessionManager.inMemory(cwd),
            }
        )
    )["session"]
    try:
        await session.agent.streamFn(
            model,
            {"messages": []},
            {"headers": request_headers} if request_headers else None,
        )
        headers = captured_options[-1].headers if captured_options[-1] is not None else None
        if expected:
            assert headers is not None
            for key, value in expected.items():
                assert headers.get(key) == value
        else:
            assert headers is None or all(headers.get(key) is None for key in expected)
            assert headers is None or headers.get("HTTP-Referer") is None
            assert headers is None or headers.get("X-OpenRouter-Title") is None
            assert headers is None or headers.get("X-OpenRouter-Categories") is None
    finally:
        session.dispose()
        for registered_provider in reversed(registered_providers):
            model_registry.unregisterProvider(registered_provider)


@pytest.mark.asyncio
async def test_create_agent_session_applies_dynamic_provider_overrides(tmp_path: Path) -> None:
    cwd = str(tmp_path)
    agent_dir = str(tmp_path / "agent")
    auth_storage = AuthStorage.create(str(Path(agent_dir) / "auth.json"))
    auth_storage.setRuntimeApiKey("anthropic", "test-key")

    async def build_session(extension_factories: list[ExtensionFactoryLike]) -> object:
        settings_manager = SettingsManager.create(cwd, agent_dir)
        resource_loader = DefaultResourceLoader(
            {
                "cwd": cwd,
                "agentDir": agent_dir,
                "settingsManager": settings_manager,
                "extensionFactories": extension_factories,
            }
        )
        await resource_loader.reload()
        return (
            await create_agent_session(
                {
                    "cwd": cwd,
                    "agentDir": agent_dir,
                    "model": _built_in_model(),
                    "settingsManager": settings_manager,
                    "sessionManager": SessionManager.inMemory(cwd),
                    "authStorage": auth_storage,
                    "resourceLoader": resource_loader,
                }
            )
        )["session"]

    top_level = await build_session(
        [
            lambda pi: pi.registerProvider("anthropic", {"baseUrl": "http://localhost:8080/top-level"}),
        ]
    )
    try:
        assert top_level.model is not None
        assert top_level.model.baseUrl == "http://localhost:8080/top-level"
    finally:
        top_level.dispose()

    session_start = await build_session(
        [
            lambda pi: pi.on(
                "session_start",
                lambda _event: pi.registerProvider("anthropic", {"baseUrl": "http://localhost:8080/session-start"}),
            ),
        ]
    )
    try:
        await session_start.bindExtensions({})
        assert session_start.model is not None
        assert session_start.model.baseUrl == "http://localhost:8080/session-start"
    finally:
        session_start.dispose()


class ExtensionFactoryLike:
    def __call__(self, pi: object) -> object: ...
