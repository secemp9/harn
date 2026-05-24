from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnify_ai.models import get_model
from harnify_ai.providers.openai_prompt_cache import clamp_openai_prompt_cache_key
import harnify_ai.providers.simple_options as simple_options_provider
import harnify_ai.providers.transform_messages as transform_messages_provider
import harnify_ai.stream as stream_module
from harnify_ai.api_registry import clear_api_providers
from harnify_ai.providers.simple_options import adjust_max_tokens_for_thinking, build_base_options, clamp_reasoning
from harnify_ai.providers.transform_messages import (
    transform_messages,
)
from harnify_ai.stream import complete_simple
from harnify_ai.types import SimpleStreamOptions, validate_assistant_message_event, validate_message
import harnify_ai.providers.register_builtins as register_builtins

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"


def _usage_payload() -> dict[str, object]:
    return {
        "input": 10,
        "output": 5,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 15,
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "total": 0,
        },
    }


def _assistant_message_payload(*, provider: str, api: str, model: str, stop_reason: str = "stop") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": "shared output"}],
        "api": api,
        "provider": provider,
        "model": model,
        "usage": _usage_payload(),
        "stopReason": stop_reason,
        "timestamp": 1_715_000_000_400,
    }


@pytest.mark.asyncio
async def test_complete_simple_uses_lazy_registered_provider_module(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_model("openai", "gpt-5")
    assert model is not None

    async def stream_openai_responses(model, context, options=None):
        yield validate_assistant_message_event(
            {
                "type": "done",
                "reason": "stop",
                "message": _assistant_message_payload(
                    provider="openai",
                    api="openai-responses",
                    model=model.id,
                ),
            }
        )

    original_import_module = register_builtins.importlib.import_module

    def fake_import_module(name: str):
        if name == "harnify_ai.providers.openai_responses":
            return SimpleNamespace(
                streamOpenAIResponses=stream_openai_responses,
                streamSimpleOpenAIResponses=stream_openai_responses,
            )
        return original_import_module(name)

    register_builtins._module_tasks.clear()
    register_builtins.reset_api_providers()
    monkeypatch.setattr(register_builtins.importlib, "import_module", fake_import_module)

    message = await complete_simple(model, {"messages": []})  # type: ignore[arg-type]

    assert message.provider == "openai"
    assert message.model == "gpt-5"
    assert message.stopReason == "stop"


def test_simple_option_helpers_and_prompt_cache_clamp_match_upstream_behavior() -> None:
    model = get_model("openai", "gpt-5")
    assert model is not None

    options = SimpleStreamOptions(maxTokens=2048, temperature=0.2, apiKey="from-options")
    base = build_base_options(model, options, api_key="override-key")
    base_without_options = build_base_options(model, None, api_key="")
    adjusted = adjust_max_tokens_for_thinking(2048, model.maxTokens, "xhigh")

    assert base.maxTokens == 2048
    assert base.apiKey == "override-key"
    assert base_without_options.apiKey is None
    assert clamp_reasoning("xhigh") == "high"
    assert adjusted.maxTokens == 2048 + adjusted.thinkingBudget
    assert adjusted.thinkingBudget == 16384
    assert clamp_openai_prompt_cache_key("x" * 80) == "x" * 64


def test_simple_options_module_exports_expected_names() -> None:
    assert simple_options_provider.__all__ == [
        "adjustMaxTokensForThinking",
        "buildBaseOptions",
        "clampReasoning",
    ]


def test_transform_messages_downgrades_images_and_normalizes_cross_model_tool_calls() -> None:
    target_model = get_model("mistral", "codestral-latest")
    assert target_model is not None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "data": "aaa", "mimeType": "image/png"},
                {"type": "image", "data": "bbb", "mimeType": "image/png"},
                {"type": "text", "text": "describe this"},
            ],
            "timestamp": 1_715_000_000_401,
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": {"city": "Paris"},
                    "thoughtSignature": "secret",
                },
            ],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "usage": _usage_payload(),
            "stopReason": "stop",
            "timestamp": 1_715_000_000_402,
        },
        {
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "lookup",
            "content": [{"type": "text", "text": "18 C"}],
            "isError": False,
            "timestamp": 1_715_000_000_403,
        },
    ]

    transformed = transform_messages(
        [validate_message(message) for message in messages],
        target_model,
        normalize_tool_call_id=lambda tool_call_id, model, source: f"norm-{tool_call_id}",
    )

    assert transformed[0].content[0].text == NON_VISION_USER_IMAGE_PLACEHOLDER
    assert transformed[0].content[1].text == "describe this"
    assert transformed[1].content[0].type == "text"
    assert transformed[1].content[0].text == "private reasoning"
    assert transformed[1].content[1].id == "norm-call-1"
    assert transformed[1].content[1].thoughtSignature is None
    assert transformed[2].toolCallId == "norm-call-1"


def test_transform_messages_synthesizes_missing_tool_results_and_skips_errored_assistants() -> None:
    target_model = get_model("openai", "gpt-5")
    assert target_model is not None

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "call-2",
                    "name": "lookup",
                    "arguments": {"city": "Paris"},
                }
            ],
            "api": "openai-responses",
            "provider": "openai",
            "model": "gpt-5",
            "usage": _usage_payload(),
            "stopReason": "stop",
            "timestamp": 1_715_000_000_404,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "partial"}],
            "api": "openai-responses",
            "provider": "openai",
            "model": "gpt-5",
            "usage": _usage_payload(),
            "stopReason": "error",
            "errorMessage": "provider failed",
            "timestamp": 1_715_000_000_405,
        },
        {
            "role": "user",
            "content": "continue",
            "timestamp": 1_715_000_000_406,
        },
    ]

    transformed = transform_messages([validate_message(message) for message in messages], target_model)

    assert transformed[0].role == "assistant"
    assert transformed[1].role == "toolResult"
    assert transformed[1].toolCallId == "call-2"
    assert transformed[1].isError is True
    assert transformed[1].content[0].text == "No result provided"
    assert transformed[2].role == "user"


def test_transform_messages_module_exports_expected_names() -> None:
    assert transform_messages_provider.__all__ == ["transformMessages"]


def test_stream_module_exports_expected_names() -> None:
    assert stream_module.__all__ == [
        "getEnvApiKey",
        "stream",
        "complete",
        "streamSimple",
        "completeSimple",
    ]


def test_stream_module_raises_ts_message_for_unregistered_api() -> None:
    model = get_model("openai", "gpt-5")
    assert model is not None

    clear_api_providers()
    try:
        with pytest.raises(RuntimeError, match="No API provider registered for api: openai-responses"):
            stream_module.stream(model, {"messages": []})  # type: ignore[arg-type]
    finally:
        register_builtins.reset_api_providers()
