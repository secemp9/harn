from __future__ import annotations

import asyncio
import re

import pytest
from harnify_ai.utils.oauth import OAuthDeviceCodeInfo
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.modes.interactive.components.login_dialog import LoginDialogComponent
import harnify_coding_agent.modes.interactive.components.login_dialog as login_dialog_module
from harnify_coding_agent.modes.interactive.components.model_selector import ModelSelectorComponent, ScopedModelItem
from harnify_coding_agent.modes.interactive.components.oauth_selector import (
    AuthSelectorProvider,
    OAuthSelectorComponent,
)
from harnify_coding_agent.modes.interactive.components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
)
from harnify_coding_agent.modes.interactive.theme.theme import init_theme
from harnify_tui import setKeybindings

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    init_theme("dark")


class FakeTui:
    def __init__(self) -> None:
        self.render_calls = 0

    def requestRender(self) -> None:
        self.render_calls += 1


def _register_demo_models() -> tuple[ModelRegistry, SettingsManager]:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())
    registry.registerProvider(
        "demo",
        {
            "baseUrl": "https://demo.example.com/v1",
            "apiKey": "DEMO_KEY",
            "api": "openai-completions",
            "models": [
                {
                    "id": "alpha",
                    "name": "Alpha",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                },
                {
                    "id": "beta",
                    "name": "Beta",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                },
                {
                    "id": "gamma",
                    "name": "Gamma",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                },
            ],
        },
    )
    return registry, SettingsManager.inMemory()


def test_model_selector_preserves_scoped_order_and_persists_selection() -> None:
    registry, settings = _register_demo_models()
    alpha = registry.find("demo", "alpha")
    beta = registry.find("demo", "beta")
    gamma = registry.find("demo", "gamma")
    assert alpha is not None and beta is not None and gamma is not None

    selected: list[str] = []
    component = ModelSelectorComponent(
        FakeTui(),
        alpha,
        settings,
        registry,
        [ScopedModelItem(model=beta), ScopedModelItem(model=alpha), ScopedModelItem(model=gamma)],
        lambda model: selected.append(f"{model.provider}/{model.id}"),
        lambda: None,
    )

    rendered = _strip_ansi("\n".join(component.render(120)))
    scoped_lines = [line.strip() for line in rendered.splitlines() if "[demo]" in line][:3]
    assert [line.removeprefix("→ ").split(" [", 1)[0] for line in scoped_lines] == ["beta", "alpha", "gamma"]

    component.handleInput("\r")
    assert selected == ["demo/alpha"]
    assert settings.getDefaultProvider() == "demo"
    assert settings.getDefaultModel() == "alpha"


def test_scoped_models_selector_reorders_and_persists() -> None:
    registry, _settings = _register_demo_models()
    models = [model for model in registry.getAll() if model.provider == "demo"]
    ordered_ids = [f"{model.provider}/{model.id}" for model in models]
    changes: list[list[str] | None] = []
    persisted: list[list[str] | None] = []
    selector = ScopedModelsSelectorComponent(
        ModelsConfig(allModels=models, enabledModelIds=ordered_ids),
        ModelsCallbacks(
            onChange=lambda enabled: changes.append(enabled),
            onPersist=lambda enabled: persisted.append(enabled),
            onCancel=lambda: None,
        ),
    )

    selector.handleInput("\x1b[1;3B")
    assert changes == [[ordered_ids[1], ordered_ids[0], ordered_ids[2]]]

    selector.handleInput("\x13")
    assert persisted == [[ordered_ids[1], ordered_ids[0], ordered_ids[2]]]


def test_oauth_selector_renders_environment_and_stored_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    auth_storage = AuthStorage.inMemory(
        {
            "anthropic": {
                "type": "oauth",
                "access": "access-token",
                "refresh": "refresh-token",
                "expires": 9999999999999,
            }
        }
    )
    selector = OAuthSelectorComponent(
        "login",
        auth_storage,
        [
            AuthSelectorProvider(id="anthropic", name="Anthropic", authType="api_key"),
            AuthSelectorProvider(id="openai", name="OpenAI", authType="api_key"),
        ],
        lambda _provider_id: None,
        lambda: None,
    )

    output = _strip_ansi("\n".join(selector.render(120)))
    assert "Anthropic" in output
    assert "subscription configured" in output
    assert "OpenAI" in output
    assert "✓ env: OPENAI_API_KEY" in output


@pytest.mark.asyncio
async def test_login_dialog_prompt_and_cancel_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    completed: list[tuple[bool, str | None]] = []
    opened: list[str] = []
    tui = FakeTui()
    dialog = LoginDialogComponent(tui, "anthropic", lambda success, message=None: completed.append((success, message)))
    monkeypatch.setattr(dialog, "openUrl", opened.append)

    dialog.showAuth("https://example.com/auth", "Open the page")
    assert opened == ["https://example.com/auth"]

    prompt_task = dialog.showPrompt("Enter code", "abc123")
    dialog.handleInput("token-value")
    dialog.handleInput("\r")
    assert await prompt_task == "token-value"

    manual_task = dialog.showManualInput("Paste callback URL")
    dialog.handleInput("\x1b")
    with pytest.raises(RuntimeError, match="Login cancelled"):
        await manual_task

    assert dialog.signal.aborted is True
    assert completed == [(False, "Login cancelled")]
    assert tui.render_calls > 0


def test_login_dialog_device_code_renders_and_opens_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    dialog = LoginDialogComponent(FakeTui(), "anthropic", lambda _success, _message=None: None)
    monkeypatch.setattr(dialog, "openUrl", opened.append)

    dialog.showDeviceCode(
        OAuthDeviceCodeInfo(
            userCode="ABCD-1234",
            verificationUri="https://example.com/device",
            intervalSeconds=5,
            expiresInSeconds=300,
        )
    )

    output = _strip_ansi("\n".join(dialog.render(120)))
    assert "ABCD-1234" in output
    assert "https://example.com/device" in output
    assert opened == ["https://example.com/device"]


def test_login_dialog_module_exports_match_ts_surface() -> None:
    assert login_dialog_module.__all__ == ["LoginDialogComponent"]
