from __future__ import annotations

import pytest

from harnify_ai.providers import cloudflare as cloudflare_provider
from harnify_ai.providers.cloudflare import is_cloudflare_provider, resolve_cloudflare_base_url
from harnify_ai.providers.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
    infer_copilot_initiator,
)
from harnify_ai.types import Context, Model, ModelCost


def _model(provider: str, base_url: str) -> Model:
    return Model(
        id="helper-model",
        name="helper-model",
        api="openai-responses",
        provider=provider,
        baseUrl=base_url,
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=128_000,
        maxTokens=8_192,
    )


def test_cloudflare_helpers_expand_placeholders_and_identify_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gateway-id")
    model = _model(
        "cloudflare-ai-gateway",
        "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat",
    )

    assert is_cloudflare_provider("cloudflare-ai-gateway") is True
    assert is_cloudflare_provider("openai") is False
    assert (
        resolve_cloudflare_base_url(model)
        == "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/compat"
    )


def test_cloudflare_helpers_raise_for_missing_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    model = _model(
        "cloudflare-workers-ai",
        "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
    )

    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        resolve_cloudflare_base_url(model)


def test_cloudflare_module_exports_expected_names() -> None:
    assert cloudflare_provider.__all__ == [
        "CLOUDFLARE_WORKERS_AI_BASE_URL",
        "CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL",
        "CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL",
        "CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL",
        "isCloudflareProvider",
        "resolveCloudflareBaseUrl",
    ]


def test_copilot_header_helpers_detect_initiator_and_vision() -> None:
    user_messages = Context.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": 1},
                {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "vision",
                    "content": [{"type": "image", "data": "abc", "mimeType": "image/png"}],
                    "isError": False,
                    "timestamp": 2,
                },
            ]
        }
    ).messages
    agent_messages = Context.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": 1},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Working"}],
                    "api": "openai-responses",
                    "provider": "github-copilot",
                    "model": "gpt-4.1",
                    "usage": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 0,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                    },
                    "stopReason": "stop",
                    "timestamp": 3,
                },
            ]
        }
    ).messages

    assert infer_copilot_initiator(user_messages) == "agent"
    assert infer_copilot_initiator(agent_messages) == "agent"
    assert has_copilot_vision_input(user_messages) is True
    assert has_copilot_vision_input(agent_messages) is False
    assert build_copilot_dynamic_headers(messages=user_messages, hasImages=True) == {
        "X-Initiator": "agent",
        "Openai-Intent": "conversation-edits",
        "Copilot-Vision-Request": "true",
    }
