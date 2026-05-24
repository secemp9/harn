from __future__ import annotations

from harnify_ai.models import (
    calculate_cost,
    clamp_thinking_level,
    get_model,
    get_models,
    get_providers,
    get_supported_thinking_levels,
    models_are_equal,
)
from harnify_ai.types import Usage, UsageCost


def test_generated_model_registry_loads_real_catalog_entries() -> None:
    providers = get_providers()

    assert "openai" in providers
    assert "anthropic" in providers
    assert "amazon-bedrock" in providers

    openai_model = get_model("openai", "gpt-5")
    assert openai_model is not None
    assert openai_model.api == "openai-responses"
    assert openai_model.provider == "openai"

    anthropic_models = get_models("anthropic")
    assert any(model.id == "claude-sonnet-4-5" for model in anthropic_models)


def test_calculate_cost_mutates_usage_cost_from_model_pricing() -> None:
    model = get_model("openai", "gpt-5")
    assert model is not None

    usage = Usage(
        input=1_000_000,
        output=500_000,
        cacheRead=250_000,
        cacheWrite=125_000,
        totalTokens=1_875_000,
        cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
    )

    cost = calculate_cost(model, usage)

    assert cost.input == model.cost.input
    assert cost.output == model.cost.output * 0.5
    assert cost.cacheRead == model.cost.cacheRead * 0.25
    assert cost.cacheWrite == model.cost.cacheWrite * 0.125
    assert cost.total == cost.input + cost.output + cost.cacheRead + cost.cacheWrite


def test_thinking_level_helpers_match_registry_capabilities() -> None:
    gpt5 = get_model("openai", "gpt-5")
    bedrock_model = get_model("amazon-bedrock", "eu.anthropic.claude-opus-4-6-v1")

    assert gpt5 is not None
    assert bedrock_model is not None

    assert get_supported_thinking_levels(gpt5) == ["minimal", "low", "medium", "high"]
    assert clamp_thinking_level(gpt5, "xhigh") == "high"

    assert get_supported_thinking_levels(bedrock_model)[-1] == "xhigh"
    assert clamp_thinking_level(bedrock_model, "xhigh") == "xhigh"


def test_models_are_equal_compares_provider_and_id() -> None:
    first = get_model("openai", "gpt-5")
    second = get_model("openai", "gpt-5")
    third = get_model("openai", "gpt-5-mini")

    assert first is not None
    assert second is not None
    assert third is not None

    assert models_are_equal(first, second) is True
    assert models_are_equal(first, third) is False
    assert models_are_equal(first, None) is False
