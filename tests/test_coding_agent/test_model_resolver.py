from __future__ import annotations

from dataclasses import dataclass

import pytest
from harnify_ai.types import Model
from harnify_coding_agent.core.model_resolver import (
    defaultModelPerProvider,
    findInitialModel,
    parseModelPattern,
    resolveCliModel,
)


def _model(provider: str, model_id: str, *, name: str | None = None) -> Model:
    return Model(
        id=model_id,
        name=name or model_id,
        api="anthropic-messages",
        provider=provider,
        baseUrl=f"https://{provider}.example.com",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2},
        contextWindow=128000,
        maxTokens=8192,
    )


ALL_MODELS = [
    _model("anthropic", "claude-sonnet-4-5", name="Claude Sonnet 4.5"),
    _model("openai", "gpt-4o", name="GPT-4o"),
    _model("openrouter", "qwen/qwen3-coder:exacto", name="Qwen Exacto"),
    _model("openrouter", "openai/gpt-4o:extended", name="GPT-4o Extended"),
]


def test_parse_model_pattern_handles_models_with_colons_and_thinking_levels() -> None:
    result = parseModelPattern("openrouter/qwen/qwen3-coder:exacto:high", ALL_MODELS)

    assert result.model is not None
    assert result.model.provider == "openrouter"
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinkingLevel == "high"
    assert result.warning is None


def test_parse_model_pattern_warns_on_invalid_thinking_suffix() -> None:
    result = parseModelPattern("sonnet:random", ALL_MODELS)

    assert result.model is not None
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinkingLevel is None
    assert result.warning is not None
    assert "Invalid thinking level" in result.warning


def test_resolve_cli_model_prefers_exact_gateway_style_ids() -> None:
    registry = type("Registry", (), {"getAll": lambda self: ALL_MODELS})()

    result = resolveCliModel({"cliModel": "openai/gpt-4o:extended", "modelRegistry": registry})

    assert result.error is None
    assert result.model is not None
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/gpt-4o:extended"


def test_resolve_cli_model_allows_custom_ids_for_explicit_provider() -> None:
    registry = type("Registry", (), {"getAll": lambda self: ALL_MODELS})()

    result = resolveCliModel(
        {
            "cliProvider": "openrouter",
            "cliModel": "openrouter/openai/ghost-model",
            "modelRegistry": registry,
        }
    )

    assert result.error is None
    assert result.model is not None
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/ghost-model"


@pytest.mark.asyncio
async def test_find_initial_model_prefers_known_provider_defaults() -> None:
    available = [
        _model("vercel-ai-gateway", "zai/glm-5.1"),
        _model("anthropic", "claude-sonnet-4-5"),
    ]

    @dataclass
    class Registry:
        def getAvailable(self):
            return available

    result = await findInitialModel(
        {
            "scopedModels": [],
            "isContinuing": False,
            "modelRegistry": Registry(),
        }
    )

    assert result.model is not None
    assert result.model.provider == "vercel-ai-gateway"
    assert result.model.id == defaultModelPerProvider["vercel-ai-gateway"]
