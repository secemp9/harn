from __future__ import annotations

import json
from pathlib import Path

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
