from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from harnify_ai.models import get_model
from harnify_ai.providers.openai_completions import (
    build_params,
    convert_messages,
    create_client,
    get_compat,
    stream_openai_completions,
)
from harnify_ai.types import AssistantMessage, Context, Model, ModelCost, ToolResultMessage, Usage, UsageCost


def _openai_model() -> Model:
    model = get_model("openai", "gpt-4o-mini")
    assert model is not None
    return model.model_copy(update={"api": "openai-completions"})


def _openrouter_auto_model() -> Model:
    return Model(
        id="openrouter/auto",
        name="OpenRouter Auto",
        api="openai-completions",
        provider="openrouter",
        baseUrl="https://openrouter.ai/api/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=200_000,
        maxTokens=8_192,
    )


def _usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cacheRead=0,
        cacheWrite=0,
        totalTokens=0,
        cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
    )


class _FakeCompletionStream:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        for chunk in self._chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self._chunks = chunks
        self.params: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> _FakeCompletionStream:
        self.params = dict(kwargs)
        return _FakeCompletionStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self.completions = _FakeCompletions(chunks)


class _FakeClient:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self.chat = _FakeChat(chunks)


def test_build_params_omits_empty_tools_and_keeps_tool_history_stub() -> None:
    model = _openai_model()

    params = build_params(
        model,
        Context(messages=[{"role": "user", "content": "hi", "timestamp": 1}]),
    )
    assert "tools" not in params
    assert "max_tokens" not in params
    assert "max_completion_tokens" not in params

    params_with_limit = build_params(
        model,
        Context(messages=[{"role": "user", "content": "hi", "timestamp": 1}], tools=[]),
        {"maxTokens": 1234},
    )
    assert "tools" not in params_with_limit
    assert params_with_limit["max_completion_tokens"] == 1234

    assistant = AssistantMessage(
        content=[{"type": "toolCall", "id": "t1", "name": "noop", "arguments": {}}],
        api="openai-completions",
        provider="openai",
        model="gpt-4o-mini",
        usage=_usage(),
        stopReason="toolUse",
        timestamp=2,
    )
    tool_result = ToolResultMessage(
        toolCallId="t1",
        toolName="noop",
        content=[{"type": "text", "text": "done"}],
        isError=False,
        timestamp=3,
    )
    params_with_history = build_params(
        model,
        Context(
            messages=[
                {"role": "user", "content": "use the tool", "timestamp": 1},
                assistant,
                tool_result,
            ],
            tools=[],
        ),
    )
    assert params_with_history["tools"] == []


def test_create_client_cloudflare_gateway_uses_compat_base_url_and_affinity_headers(monkeypatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gateway-id")

    model = get_model("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6")
    assert model is not None

    client = create_client(
        model,
        Context(messages=[]),
        "test-key",
        session_id="session-1",
        compat=get_compat(model),
    )

    assert str(client.base_url) == "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/compat/"
    assert client.default_headers["Authorization"] is None
    assert client.default_headers["cf-aig-authorization"] == "Bearer test-key"
    assert client.default_headers["session_id"] == "session-1"
    assert client.default_headers["x-client-request-id"] == "session-1"
    assert client.default_headers["x-session-affinity"] == "session-1"


def test_convert_messages_batches_tool_result_images_after_consecutive_tool_results() -> None:
    model = _openai_model().model_copy(update={"input": ["text", "image"]})
    compat = get_compat(model)
    now = 1_715_000_000_100
    assistant = AssistantMessage(
        content=[
            {"type": "toolCall", "id": "tool-1", "name": "read", "arguments": {"path": "img-1.png"}},
            {"type": "toolCall", "id": "tool-2", "name": "read", "arguments": {"path": "img-2.png"}},
        ],
        api="openai-completions",
        provider=model.provider,
        model=model.id,
        usage=_usage(),
        stopReason="toolUse",
        timestamp=now,
    )

    def build_tool_result(tool_call_id: str, timestamp: int) -> ToolResultMessage:
        return ToolResultMessage(
            toolCallId=tool_call_id,
            toolName="read",
            content=[
                {"type": "text", "text": "Read image file [image/png]"},
                {"type": "image", "data": "ZmFrZQ==", "mimeType": "image/png"},
            ],
            isError=False,
            timestamp=timestamp,
        )

    messages = convert_messages(
        model,
        Context(
            messages=[
                {"role": "user", "content": "Read the images", "timestamp": now - 2},
                assistant,
                build_tool_result("tool-1", now + 1),
                build_tool_result("tool-2", now + 2),
            ]
        ),
        compat,
    )

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "user"]
    image_message = messages[-1]
    assert image_message["role"] == "user"
    image_parts = [part for part in image_message["content"] if part.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_convert_messages_serializes_same_model_thinking_as_text_when_required() -> None:
    model = Model(
        id="repro-model",
        name="Repro Model",
        api="openai-completions",
        provider="repro-provider",
        baseUrl="http://127.0.0.1:1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=128_000,
        maxTokens=4_096,
        compat={"requiresThinkingAsText": True},
    )
    compat = get_compat(model)
    assistant = AssistantMessage(
        content=[
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "text", "text": "visible answer"},
        ],
        api="openai-completions",
        provider=model.provider,
        model=model.id,
        usage=_usage(),
        stopReason="stop",
        timestamp=2,
    )

    messages = convert_messages(
        model,
        Context(
            messages=[
                {"role": "user", "content": "hello", "timestamp": 1},
                assistant,
                {"role": "user", "content": "continue", "timestamp": 3},
            ]
        ),
        compat,
    )

    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "internal reasoning"},
            {"type": "text", "text": "visible answer"},
        ],
    }


@pytest.mark.asyncio
async def test_stream_openai_completions_surfaces_response_model_and_streamed_blocks() -> None:
    model = _openrouter_auto_model()
    fake_client = _FakeClient(
        [
            {
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.7",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.7",
                "choices": [{"index": 0, "delta": {"reasoning_content": "think"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.7",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": '{"city":"Par'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.7",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'is"}'}}]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 1},
                },
            },
        ]
    )

    stream = stream_openai_completions(
        model,
        Context(messages=[{"role": "user", "content": "hi", "timestamp": 1}]),
        {"client": fake_client},
    )
    result = await stream.result()

    assert result.model == "openrouter/auto"
    assert result.responseModel == "anthropic/claude-opus-4.7"
    assert result.stopReason == "toolUse"
    assert result.usage.input == 9
    assert result.usage.output == 5
    assert [block.type for block in result.content] == ["text", "thinking", "toolCall"]
    assert result.content[0].text == "hi"
    assert result.content[1].thinking == "think"
    assert result.content[2].arguments == {"city": "Paris"}
