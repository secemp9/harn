from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import harnify_ai.providers.mistral as mistral_provider
from harnify_ai.types import Context, Model, ModelCost, SimpleStreamOptions, Tool


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
    assert captured_payload["reasoning_effort"] == "high"
    assert "prompt_mode" not in captured_payload


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
    assert captured_payload["prompt_mode"] == "reasoning"
    assert "reasoning_effort" not in captured_payload


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
