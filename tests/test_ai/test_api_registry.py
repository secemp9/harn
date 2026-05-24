from __future__ import annotations

import pytest

from harnify_ai.api_registry import (
    ApiProvider,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)
from harnify_ai.models import get_model
from harnify_ai.types import validate_assistant_message_event
from harnify_ai.utils.event_stream import AssistantMessageEventStream


def _assistant_message_payload(stop_reason: str = "stop") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": "registry output"}],
        "api": "openai-responses",
        "provider": "openai",
        "model": "gpt-5",
        "usage": {
            "input": 1,
            "output": 1,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 2,
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "total": 0,
            },
        },
        "stopReason": stop_reason,
        "timestamp": 1_715_000_000_200,
    }


def _make_stream() -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(
        validate_assistant_message_event(
            {
                "type": "done",
                "reason": "stop",
                "message": _assistant_message_payload(),
            }
        )
    )
    return stream


def test_api_registry_registers_and_unregisters_providers() -> None:
    clear_api_providers()

    provider = ApiProvider(
        api="openai-responses",
        stream=lambda model, context, options=None: _make_stream(),
        streamSimple=lambda model, context, options=None: _make_stream(),
    )

    register_api_provider(provider, source_id="test-suite")

    registered = get_api_provider("openai-responses")
    assert registered is not None
    assert registered.api == "openai-responses"
    assert len(get_api_providers()) == 1

    unregister_api_providers("test-suite")

    assert get_api_provider("openai-responses") is None
    assert get_api_providers() == []


def test_api_registry_wrapper_rejects_mismatched_models() -> None:
    clear_api_providers()

    provider = ApiProvider(
        api="openai-responses",
        stream=lambda model, context, options=None: _make_stream(),
        streamSimple=lambda model, context, options=None: _make_stream(),
    )
    register_api_provider(provider)

    wrapped = get_api_provider("openai-responses")
    assert wrapped is not None

    anthropic_model = get_model("anthropic", "claude-sonnet-4-5")
    openai_model = get_model("openai", "gpt-5")
    assert anthropic_model is not None
    assert openai_model is not None

    with pytest.raises(ValueError, match="Mismatched api"):
        wrapped.stream(anthropic_model, {"messages": []})  # type: ignore[arg-type]

    stream = wrapped.stream(openai_model, {"messages": []})  # type: ignore[arg-type]
    result = stream.result()

    assert result.done() is True
    assert result.result().model == "gpt-5"
