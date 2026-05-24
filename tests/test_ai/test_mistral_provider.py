from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

import harnify_ai.providers.mistral as mistral_provider
from harnify_ai.types import AssistantMessage, Context, Model, ModelCost, SimpleStreamOptions, Tool, ToolResultMessage, Usage, UsageCost


@dataclass(slots=True)
class _FakeDelta:
    content: Any = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class _FakeChoice:
    delta: _FakeDelta
    finish_reason: str | None = None


@dataclass(slots=True)
class _FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class _FakeChunk:
    id: str
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = None


@dataclass(slots=True)
class _FakeEvent:
    data: _FakeChunk


class _FakeChat:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events
        self.calls: list[dict[str, Any]] = []

    async def stream_async(self, **kwargs: Any):
        self.calls.append(kwargs)

        async def _iterate():
            for event in self._events:
                yield event

        return _iterate()


class _FakeClient:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self.chat = _FakeChat(events)


def _make_model(model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="mistral-conversations",
        provider="mistral",
        baseUrl="https://api.mistral.ai",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=128_000,
        maxTokens=32_000,
        thinkingLevelMap={"minimal": "none", "low": "high", "medium": "high", "high": "high"},
    )


def _make_context(tools: list[Tool] | None = None) -> Context:
    return Context(
        messages=[{"role": "user", "content": "Hello", "timestamp": 1}],
        tools=tools,
    )


@pytest.mark.asyncio
async def test_stream_simple_mistral_uses_reasoning_effort_for_small_models(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, Any] | None = None
    fake_client = _FakeClient([_FakeEvent(_FakeChunk(id="resp_1", choices=[_FakeChoice(_FakeDelta(), "stop")]))])

    def fake_create_client(*_args: Any, **_kwargs: Any) -> _FakeClient:
        return fake_client

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    monkeypatch.setattr(mistral_provider, "create_client", fake_create_client)

    result = await mistral_provider.stream_simple_mistral(
        _make_model("mistral-small-2603"),
        _make_context(),
        SimpleStreamOptions(apiKey="test-key", reasoning="medium", onPayload=on_payload),
    ).result()

    assert result.stopReason == "stop"
    assert captured_payload is not None
    assert captured_payload["reasoningEffort"] == "high"
    assert "promptMode" not in captured_payload
    assert fake_client.chat.calls[0]["reasoning_effort"] == "high"
    assert "prompt_mode" not in fake_client.chat.calls[0]


@pytest.mark.asyncio
async def test_stream_simple_mistral_uses_prompt_mode_for_magistral_models(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, Any] | None = None
    fake_client = _FakeClient([_FakeEvent(_FakeChunk(id="resp_2", choices=[_FakeChoice(_FakeDelta(), "stop")]))])

    def fake_create_client(*_args: Any, **_kwargs: Any) -> _FakeClient:
        return fake_client

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    monkeypatch.setattr(mistral_provider, "create_client", fake_create_client)

    result = await mistral_provider.stream_simple_mistral(
        _make_model("magistral-medium-latest"),
        _make_context(),
        SimpleStreamOptions(apiKey="test-key", reasoning="medium", onPayload=on_payload),
    ).result()

    assert result.stopReason == "stop"
    assert captured_payload is not None
    assert captured_payload["promptMode"] == "reasoning"
    assert "reasoningEffort" not in captured_payload
    assert fake_client.chat.calls[0]["prompt_mode"] == "reasoning"
    assert "reasoning_effort" not in fake_client.chat.calls[0]


def test_build_request_options_matches_upstream_surface() -> None:
    signal = object()
    model = _make_model("mistral-small-2603")
    model.headers = {"x-model": "1"}

    request_options = mistral_provider.build_request_kwargs(
        model,
        {"signal": signal, "headers": {"x-option": "2"}, "sessionId": "session-1"},
    )

    assert request_options == {
        "signal": signal,
        "retries": {"strategy": "none"},
        "headers": {"x-model": "1", "x-option": "2", "x-affinity": "session-1"},
    }


def test_to_chat_messages_uses_upstream_key_casing() -> None:
    context = Context(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "mimeType": "image/png", "data": "abc"},
                ],
                "timestamp": 1,
            },
            AssistantMessage(
                content=[{"type": "toolCall", "id": "call-1", "name": "calc", "arguments": {"a": 1}}],
                api="mistral-conversations",
                provider="mistral",
                model="mistral-small-2603",
                usage=Usage(input=0, output=0, cacheRead=0, cacheWrite=0, totalTokens=0, cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0)),
                stopReason="toolUse",
                timestamp=2,
            ),
            ToolResultMessage(
                toolCallId="call-1",
                toolName="calc",
                content=[{"type": "image", "mimeType": "image/png", "data": "xyz"}],
                isError=False,
                timestamp=3,
            ),
        ]
    )

    chat_messages = mistral_provider.to_chat_messages(context.messages, supports_images=True)

    assert chat_messages[0]["content"][1] == {"type": "image_url", "imageUrl": "data:image/png;base64,abc"}
    assert chat_messages[1]["toolCalls"][0]["function"]["arguments"] == '{"a": 1}'
    assert chat_messages[2]["toolCallId"] == "call-1"
    assert chat_messages[2]["content"][1] == {"type": "image_url", "imageUrl": "data:image/png;base64,xyz"}


def test_to_function_tools_serializes_parameters_schema() -> None:
    tools = [
        Tool(
            name="inspect_schema",
            description="Inspect the schema",
            parameters={
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
            },
        )
    ]

    payload_tools = mistral_provider.to_function_tools(tools)

    assert payload_tools == [
        {
            "type": "function",
            "function": {
                "name": "inspect_schema",
                "description": "Inspect the schema",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        }
                    },
                },
                "strict": False,
            },
        }
    ]


@pytest.mark.asyncio
async def test_stream_mistral_maps_text_and_tool_call_deltas() -> None:
    fake_client = _FakeClient(
        [
            _FakeEvent(_FakeChunk(id="resp_stream", choices=[_FakeChoice(_FakeDelta(content="Hello "))])),
            _FakeEvent(
                _FakeChunk(
                    id="resp_stream",
                    choices=[
                        _FakeChoice(
                            _FakeDelta(
                                tool_calls=[
                                    {"id": "abc123xyz", "index": 0, "function": {"name": "calc", "arguments": '{"a":'}}
                                ]
                            )
                        )
                    ],
                )
            ),
            _FakeEvent(
                _FakeChunk(
                    id="resp_stream",
                    choices=[
                        _FakeChoice(
                            _FakeDelta(
                                tool_calls=[
                                    {"id": "abc123xyz", "index": 0, "function": {"name": "calc", "arguments": "1}"}}
                                ]
                            ),
                            "tool_calls",
                        )
                    ],
                    usage=_FakeUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                )
            ),
        ]
    )

    stream = mistral_provider.stream_mistral(
        _make_model("mistral-small-2603"),
        _make_context(),
        {"apiKey": "test-key", "client": fake_client},
    )

    event_types: list[str] = []
    async for event in stream:
        event_types.append(event.type)

    result = await stream.result()
    text_block = result.content[0]
    tool_block = result.content[1]

    assert event_types == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert result.responseId == "resp_stream"
    assert result.stopReason == "toolUse"
    assert text_block.type == "text"
    assert text_block.text == "Hello "
    assert tool_block.type == "toolCall"
    assert tool_block.id == "abc123xyz"
    assert tool_block.name == "calc"
    assert tool_block.arguments == {"a": 1}
    assert result.usage.input == 5
    assert result.usage.output == 3
    assert result.usage.totalTokens == 8


@pytest.mark.asyncio
async def test_stream_mistral_returns_aborted_result_for_preaborted_signal() -> None:
    fake_client = _FakeClient([_FakeEvent(_FakeChunk(id="resp_abort", choices=[_FakeChoice(_FakeDelta(), "stop")]))])
    signal = asyncio.Event()
    signal.set()

    stream = mistral_provider.stream_mistral(
        _make_model("mistral-small-2603"),
        _make_context(),
        {"apiKey": "test-key", "client": fake_client, "signal": signal},
    )

    result = await stream.result()

    assert result.stopReason == "aborted"
    assert result.errorMessage == "Request was aborted"
    assert fake_client.chat.calls == []
