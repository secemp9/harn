from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from harnify_ai.providers.google import build_params, stream_google, stream_simple_google
from harnify_ai.types import Context, Model, ModelCost, SimpleStreamOptions, ThinkingBudgets


@dataclass(slots=True)
class _FakeFunctionCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass(slots=True)
class _FakePart:
    text: str | None = None
    thought: bool | None = None
    thought_signature: str | None = None
    function_call: _FakeFunctionCall | None = None


@dataclass(slots=True)
class _FakeContent:
    parts: list[_FakePart]


@dataclass(slots=True)
class _FakeFinishReason:
    name: str


@dataclass(slots=True)
class _FakeCandidate:
    content: _FakeContent | None = None
    finish_reason: _FakeFinishReason | None = None


@dataclass(slots=True)
class _FakeUsageMetadata:
    prompt_token_count: int = 0
    cached_content_token_count: int = 0
    response_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int = 0


@dataclass(slots=True)
class _FakeChunk:
    candidates: list[_FakeCandidate]
    response_id: str | None = None
    usage_metadata: _FakeUsageMetadata | None = None


class _FakeModels:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any):
        self.calls.append(kwargs)

        async def _iterate():
            for chunk in self._chunks:
                yield chunk

        return _iterate()


class _FakeAsyncClient:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.models = _FakeModels(chunks)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.aio = _FakeAsyncClient(chunks)


def _make_model(model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-generative-ai",
        provider="google",
        baseUrl="https://generativelanguage.googleapis.com/v1beta",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=1_000_000,
        maxTokens=65_536,
    )


def _make_context() -> Context:
    return Context(
        systemPrompt="Think carefully.",
        messages=[{"role": "user", "content": "Hello", "timestamp": 1}],
    )


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("gemini-2.5-flash", {"thinkingBudget": 0}),
        ("gemini-3-flash-preview", {"thinkingLevel": "MINIMAL"}),
        ("gemini-3.1-pro-preview", {"thinkingLevel": "LOW"}),
    ],
)
def test_build_params_maps_disabled_thinking_by_google_model_family(
    model_id: str,
    expected: dict[str, Any],
) -> None:
    params = build_params(
        _make_model(model_id),
        _make_context(),
        {"thinking": {"enabled": False}},
    )

    assert params["config"]["thinkingConfig"] == expected


@pytest.mark.asyncio
async def test_stream_simple_google_uses_thinking_level_for_gemini3(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, Any] | None = None
    fake_client = _FakeClient(
        [
            _FakeChunk(
                candidates=[_FakeCandidate(finish_reason=_FakeFinishReason("STOP"))],
                response_id="resp_simple_level",
            )
        ]
    )

    def fake_create_client(*_args: Any, **_kwargs: Any) -> _FakeClient:
        return fake_client

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    monkeypatch.setattr("harnify_ai.providers.google.create_client", fake_create_client)

    result = await stream_simple_google(
        _make_model("gemini-3-flash-preview"),
        _make_context(),
        SimpleStreamOptions(apiKey="test-key", reasoning="low", onPayload=on_payload),
    ).result()

    assert result.stopReason == "stop"
    assert captured_payload is not None
    assert captured_payload["config"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "LOW",
    }


@pytest.mark.asyncio
async def test_stream_simple_google_uses_custom_budget_for_gemini25(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, Any] | None = None
    fake_client = _FakeClient(
        [
            _FakeChunk(
                candidates=[_FakeCandidate(finish_reason=_FakeFinishReason("STOP"))],
                response_id="resp_simple_budget",
            )
        ]
    )

    def fake_create_client(*_args: Any, **_kwargs: Any) -> _FakeClient:
        return fake_client

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    monkeypatch.setattr("harnify_ai.providers.google.create_client", fake_create_client)

    result = await stream_simple_google(
        _make_model("gemini-2.5-flash"),
        _make_context(),
        SimpleStreamOptions(
            apiKey="test-key",
            reasoning="medium",
            thinkingBudgets=ThinkingBudgets(medium=1234),
            onPayload=on_payload,
        ),
    ).result()

    assert result.stopReason == "stop"
    assert captured_payload is not None
    assert captured_payload["config"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 1234,
    }


@pytest.mark.asyncio
async def test_stream_google_maps_thinking_text_tool_calls_and_usage() -> None:
    fake_client = _FakeClient(
        [
            _FakeChunk(
                candidates=[
                    _FakeCandidate(
                        content=_FakeContent(
                            parts=[
                                _FakePart(text="step 1", thought=True, thought_signature="sig_think"),
                                _FakePart(text="answer", thought=False, thought_signature="sig_text"),
                                _FakePart(
                                    function_call=_FakeFunctionCall(name="read", args={"path": "a.txt"}),
                                    thought_signature="sig_tool",
                                ),
                            ]
                        ),
                        finish_reason=_FakeFinishReason("STOP"),
                    )
                ],
                response_id="resp_stream_1",
                usage_metadata=_FakeUsageMetadata(
                    prompt_token_count=12,
                    cached_content_token_count=2,
                    response_token_count=4,
                    thoughts_token_count=3,
                    total_token_count=17,
                ),
            )
        ]
    )

    stream = stream_google(
        _make_model("gemini-3-flash-preview"),
        _make_context(),
        {"apiKey": "test-key", "client": fake_client},
    )

    event_types: list[str] = []
    async for event in stream:
        event_types.append(event.type)

    result = await stream.result()
    thinking_block = result.content[0]
    text_block = result.content[1]
    tool_call = result.content[2]

    assert event_types == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert result.responseId == "resp_stream_1"
    assert result.stopReason == "toolUse"
    assert thinking_block.type == "thinking"
    assert thinking_block.thinking == "step 1"
    assert thinking_block.thinkingSignature == "sig_think"
    assert text_block.type == "text"
    assert text_block.text == "answer"
    assert text_block.textSignature == "sig_text"
    assert tool_call.type == "toolCall"
    assert tool_call.name == "read"
    assert tool_call.arguments == {"path": "a.txt"}
    assert tool_call.thoughtSignature == "sig_tool"
    assert tool_call.id.startswith("read_")
    assert result.usage.input == 10
    assert result.usage.output == 7
    assert result.usage.cacheRead == 2
    assert result.usage.totalTokens == 17
    assert fake_client.aio.closed is True
