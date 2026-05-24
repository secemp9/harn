from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, get_origin, get_type_hints

import pytest
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.model_registry import ModelRegistry


def _write_models_json(path: Path, providers: dict[str, object]) -> None:
    path.write_text(json.dumps({"providers": providers}), encoding="utf-8")


def _provider_config(base_url: str, models: list[dict[str, str]], api: str = "anthropic-messages") -> dict[str, object]:
    return {
        "baseUrl": base_url,
        "apiKey": "TEST_KEY",
        "api": api,
        "models": [
            {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 100000,
                "maxTokens": 8000,
            }
            for model in models
        ],
    }


def test_base_url_override_keeps_all_builtin_models(tmp_path: Path) -> None:
    models_json_path = tmp_path / "models.json"
    _write_models_json(models_json_path, {"anthropic": {"baseUrl": "https://proxy.example.com/v1"}})

    registry = ModelRegistry.create(AuthStorage.inMemory(), str(models_json_path))
    anthropic_models = [model for model in registry.getAll() if model.provider == "anthropic"]

    assert len(anthropic_models) > 1
    assert all(model.baseUrl == "https://proxy.example.com/v1" for model in anthropic_models)


def test_builtin_provider_custom_models_merge_with_builtin_models(tmp_path: Path) -> None:
    models_json_path = tmp_path / "models.json"
    _write_models_json(
        models_json_path,
        {"anthropic": _provider_config("https://proxy.example.com/v1", [{"id": "claude-custom"}])},
    )

    registry = ModelRegistry.create(AuthStorage.inMemory(), str(models_json_path))
    anthropic_models = [model for model in registry.getAll() if model.provider == "anthropic"]

    assert any(model.id == "claude-custom" for model in anthropic_models)
    assert any("claude" in model.id for model in anthropic_models)


@pytest.mark.asyncio
async def test_request_auth_resolution_merges_headers_and_auth_header(tmp_path: Path) -> None:
    models_json_path = tmp_path / "models.json"
    _write_models_json(
        models_json_path,
        {
            "demo": {
                "baseUrl": "https://demo.example.com/v1",
                "apiKey": "literal-secret",
                "authHeader": True,
                "headers": {"X-Custom-Header": "custom-value"},
                "api": "openai-completions",
                "models": [
                    {
                        "id": "demo-model",
                        "name": "Demo Model",
                        "reasoning": False,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": 1000,
                        "maxTokens": 100,
                    }
                ],
            }
        },
    )

    registry = ModelRegistry.create(AuthStorage.inMemory(), str(models_json_path))
    model = registry.find("demo", "demo-model")
    assert model is not None

    auth = await registry.getApiKeyAndHeaders(model)
    assert auth["ok"] is True
    assert auth["apiKey"] == "literal-secret"
    assert auth["headers"]["X-Custom-Header"] == "custom-value"
    assert auth["headers"]["Authorization"] == "Bearer literal-secret"


def test_provider_auth_status_reports_models_json_command_without_executing_it(tmp_path: Path) -> None:
    models_json_path = tmp_path / "models.json"
    _write_models_json(
        models_json_path,
        {
            "demo": {
                "baseUrl": "https://demo.example.com/v1",
                "apiKey": "!echo secret",
                "api": "openai-completions",
                "models": [],
                "headers": {"X-Custom-Header": "custom-value"},
            }
        },
    )

    registry = ModelRegistry.create(AuthStorage.inMemory(), str(models_json_path))
    status = registry.getProviderAuthStatus("demo")

    assert status.configured is True
    assert status.source == "models_json_command"


def test_models_json_schema_validation_reports_nested_path(tmp_path: Path) -> None:
    models_json_path = tmp_path / "models.json"
    _write_models_json(
        models_json_path,
        {
            "demo": {
                "headers": {
                    "Authorization": 123,
                }
            }
        },
    )

    registry = ModelRegistry.create(AuthStorage.inMemory(), str(models_json_path))

    assert registry.getError() == (
        "Invalid models.json schema:\n"
        "  - providers.demo.headers.Authorization: Input should be a valid string\n\n"
        f"File: {models_json_path}"
    )


def test_empty_headers_override_is_accepted(tmp_path: Path) -> None:
    models_json_path = tmp_path / "models.json"
    _write_models_json(models_json_path, {"demo": {"headers": {}}})

    registry = ModelRegistry.create(AuthStorage.inMemory(), str(models_json_path))

    assert registry.getError() is None


def test_model_registry_module_exports_match_ts_surface() -> None:
    from harnify_coding_agent.core import model_registry

    assert model_registry.__all__ == [
        "ModelRegistry",
        "ProviderConfigInput",
        "ResolvedRequestAuth",
        "clearApiKeyCache",
    ]


def test_provider_config_input_model_shape_matches_ts_required_keys() -> None:
    from harnify_coding_agent.core import model_registry

    hints = get_type_hints(model_registry._ProviderModelInput, include_extras=True)

    for key in ("id", "name", "reasoning", "input", "cost", "contextWindow", "maxTokens"):
        assert get_origin(hints[key]) is not NotRequired

    for key in ("api", "baseUrl", "thinkingLevelMap", "headers", "compat"):
        assert get_origin(hints[key]) is NotRequired


def test_model_registry_private_helpers_are_not_public() -> None:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())

    assert not hasattr(ModelRegistry, "loadModels")
    assert not hasattr(ModelRegistry, "loadCustomModels")
    assert not hasattr(ModelRegistry, "parseModels")
    assert not hasattr(ModelRegistry, "getModelRequestKey")
    assert not hasattr(ModelRegistry, "applyProviderConfig")
    assert hasattr(ModelRegistry, "_loadModels")
    assert hasattr(ModelRegistry, "_loadCustomModels")
    assert hasattr(ModelRegistry, "_parseModels")
    assert hasattr(ModelRegistry, "_getModelRequestKey")
    assert hasattr(ModelRegistry, "_applyProviderConfig")
    assert not hasattr(registry, "models")
    assert not hasattr(registry, "providerRequestConfigs")
    assert not hasattr(registry, "modelRequestHeaders")
    assert not hasattr(registry, "registeredProviders")
    assert not hasattr(registry, "loadError")
    assert not hasattr(registry, "modelsJsonPath")
    assert hasattr(registry, "_models")
    assert hasattr(registry, "_providerRequestConfigs")
    assert hasattr(registry, "_modelRequestHeaders")
    assert hasattr(registry, "_registeredProviders")
    assert hasattr(registry, "_loadError")
    assert hasattr(registry, "_modelsJsonPath")


@pytest.mark.asyncio
async def test_register_provider_accepts_oauth_objects_and_overrides_id() -> None:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())

    class DemoOAuthProvider:
        id = "wrong-id"
        name = "Dynamic OAuth"
        usesCallbackServer = None
        modifyModels = None

        async def login(self, callbacks):  # pragma: no cover - not exercised
            raise RuntimeError("unused")

        async def refreshToken(self, credentials):  # pragma: no cover - not exercised
            return credentials

        def getApiKey(self, credentials):  # pragma: no cover - not exercised
            return credentials.access

    registry.registerProvider("dynamic-oauth", {"oauth": DemoOAuthProvider()})
    try:
        assert registry.getProviderDisplayName("dynamic-oauth") == "Dynamic OAuth"
        assert any(provider.id == "dynamic-oauth" for provider in registry.authStorage.getOAuthProviders())
    finally:
        registry.unregisterProvider("dynamic-oauth")


def test_dynamic_provider_registration_and_unregister_restores_state() -> None:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())
    registry.registerProvider(
        "dynamic-demo",
        {
            "baseUrl": "https://dynamic.example.com/v1",
            "apiKey": "DYNAMIC_KEY",
            "api": "openai-completions",
            "models": [
                {
                    "id": "dynamic-model",
                    "name": "Dynamic Model",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                }
            ],
        },
    )

    assert registry.find("dynamic-demo", "dynamic-model") is not None

    registry.unregisterProvider("dynamic-demo")
    assert registry.find("dynamic-demo", "dynamic-model") is None


@pytest.mark.asyncio
async def test_register_provider_keeps_live_config_references_like_ts() -> None:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())
    config: dict[str, object] = {
        "baseUrl": "https://dynamic.example.com/v1",
        "apiKey": "DYNAMIC_KEY",
        "api": "openai-completions",
        "headers": {"X-Demo": "one"},
        "models": [
            {
                "id": "dynamic-model",
                "name": "Dynamic Model",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 1000,
                "maxTokens": 100,
            }
        ],
    }
    registry.registerProvider("dynamic-demo", config)
    try:
        model = registry.find("dynamic-demo", "dynamic-model")
        assert model is not None

        auth = await registry.getApiKeyAndHeaders(model)
        assert auth["ok"] is True
        assert auth["headers"] == {"X-Demo": "one"}

        config["baseUrl"] = "https://changed.example.com/v1"
        config["headers"]["X-Demo"] = "two"  # type: ignore[index]

        auth = await registry.getApiKeyAndHeaders(model)
        assert auth["ok"] is True
        assert auth["headers"] == {"X-Demo": "two"}

        registry.refresh()
        refreshed_model = registry.find("dynamic-demo", "dynamic-model")
        assert refreshed_model is not None
        assert refreshed_model.baseUrl == "https://changed.example.com/v1"
    finally:
        registry.unregisterProvider("dynamic-demo")


def test_dynamic_provider_registration_does_not_apply_python_only_defaults() -> None:
    registry = ModelRegistry.inMemory(AuthStorage.inMemory())
    registry.registerProvider(
        "dynamic-demo",
        {
            "baseUrl": "https://dynamic.example.com/v1",
            "apiKey": "DYNAMIC_KEY",
            "api": "openai-completions",
            "models": [
                {
                    "id": "dynamic-model",
                }
            ],
        },
    )
    try:
        model = registry.find("dynamic-demo", "dynamic-model")
        assert model is not None
        assert model.name is None
        assert model.reasoning is None
        assert model.input is None
        assert model.cost is None
        assert model.contextWindow is None
        assert model.maxTokens is None
    finally:
        registry.unregisterProvider("dynamic-demo")
