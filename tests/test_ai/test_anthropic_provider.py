from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from anthropic import omit

import harnify_ai.providers.anthropic as anthropic_provider
from harnify_ai.models import get_model
from harnify_ai.providers.anthropic import (
    convert_tools,
    create_client,
    from_claude_code_name,
    iterate_anthropic_events,
    should_use_fine_grained_tool_streaming_beta,
    stream_anthropic,
    stream_simple_anthropic,
    to_claude_code_name,
)
from harnify_ai.types import Context, Model, ModelCost, SimpleStreamOptions, Tool


class _FakeResponse:
    def __init__(self, events: list[dict[str, str]]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._lines: list[str] = []
        for event in events:
            self._lines.append(f"event: {event['event']}")
            self._lines.append(f"data: {event['data']}")
            self._lines.append("")

    async def iter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeCreateResult:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def asResponse(self) -> _FakeResponse:
        return self._response


class _FakeMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.payloads: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeCreateResult:
        self.payloads.append(kwargs)
        return _FakeCreateResult(self._response)


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessages(response)


class _FakeRawMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.payloads: list[dict[str, object]] = []

    @property
    def with_raw_response(self) -> _FakeRawMessages:
        return self

    async def create(self, **kwargs: object) -> _FakeResponse:
        self.payloads.append(kwargs)
        return self._response


class _FakeClientWithOptions:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeRawMessages(response)
        self.with_options_calls: list[dict[str, object]] = []

    def with_options(self, **kwargs: object) -> _FakeClientWithOptions:
        self.with_options_calls.append(kwargs)
        return self


def _make_context(tools: list[Tool] | None = None) -> Context:
    return Context(
        messages=[{"role": "user", "content": "Use the tool.", "timestamp": 1_715_000_000_001}],
        tools=tools,
    )


def _make_custom_model(
    force_adaptive_thinking: bool | None = None,
    *,
    supports_eager_tool_input_streaming: bool | None = None,
) -> Model:
    compat: dict[str, object] = {}
    if force_adaptive_thinking is not None:
        compat["forceAdaptiveThinking"] = force_adaptive_thinking
    if supports_eager_tool_input_streaming is not None:
        compat["supportsEagerToolInputStreaming"] = supports_eager_tool_input_streaming
    return Model(
        id="vendor--claude-opus-latest",
        name="Vendor Proxy Opus Latest",
        api="anthropic-messages",
        provider="vendor-proxy",
        baseUrl="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=200_000,
        maxTokens=32_000,
        compat=compat or None,
    )


def _tool() -> Tool:
    return Tool(
        name="edit",
        description="Edit a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
    )


@pytest.mark.asyncio
async def test_stream_anthropic_repairs_malformed_tool_json() -> None:
    model = get_model("anthropic", "claude-haiku-4-5")
    assert model is not None

    malformed_tool_json_delta = (
        r'{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
        r'"partial_json":"{\"path\":\"A\H\",\"text\":\"col1\tcol2\"}"}}'
    )
    response = _FakeResponse(
        [
            {
                "event": "message_start",
                "data": (
                    '{"type":"message_start","message":{"id":"msg_test","usage":{"input_tokens":12,'
                    '"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}'
                ),
            },
            {
                "event": "content_block_start",
                "data": (
                    '{"type":"content_block_start","index":0,"content_block":{"type":"tool_use",'
                    '"id":"toolu_test","name":"edit","input":{}}}'
                ),
            },
            {"event": "content_block_delta", "data": malformed_tool_json_delta},
            {"event": "content_block_stop", "data": '{"type":"content_block_stop","index":0}'},
            {
                "event": "message_delta",
                "data": (
                    '{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":12,'
                    '"output_tokens":5,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}'
                ),
            },
            {"event": "message_stop", "data": '{"type":"message_stop"}'},
            {"event": "done", "data": "[DONE]"},
        ]
    )

    stream = stream_anthropic(model, _make_context([_tool()]), {"client": _FakeClient(response)})
    result = await stream.result()

    assert result.stopReason == "toolUse"
    assert result.errorMessage is None
    tool_call = next(block for block in result.content if block.type == "toolCall")
    assert tool_call.arguments == {"path": "A\\H", "text": "col1\tcol2"}


@pytest.mark.asyncio
async def test_iterate_anthropic_events_requires_message_stop() -> None:
    response = _FakeResponse(
        [
            {
                "event": "message_start",
                "data": '{"type":"message_start","message":{"id":"msg_test","usage":{"input_tokens":1,"output_tokens":0}}}',
            },
            {
                "event": "content_block_start",
                "data": '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            },
        ]
    )

    with pytest.raises(RuntimeError, match="message_stop"):
        async for _ in iterate_anthropic_events(response):
            pass


def test_anthropic_tool_compatibility_sets_expected_flags_and_headers() -> None:
    context = _make_context([_tool()])
    legacy_model = _make_custom_model(supports_eager_tool_input_streaming=False)

    assert should_use_fine_grained_tool_streaming_beta(legacy_model, context) is True
    converted_tools = convert_tools(context.tools or [], False, False)
    assert "eager_input_streaming" not in converted_tools[0]

    client, _ = create_client(legacy_model, "test-key", True, True)
    assert client.default_headers["anthropic-beta"] == (
        "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"
    )

    eager_tools = convert_tools(context.tools or [], False, True)
    assert eager_tools[0]["eager_input_streaming"] is True


def test_cloudflare_client_omits_sdk_auth_headers() -> None:
    model = Model(
        id="vendor--claude-opus-latest",
        name="Vendor Proxy Opus Latest",
        api="anthropic-messages",
        provider="cloudflare-ai-gateway",
        baseUrl="https://gateway.example/anthropic",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=200_000,
        maxTokens=32_000,
    )

    client, is_oauth = create_client(model, "test-key", True, False)

    assert is_oauth is False
    assert client.default_headers["cf-aig-authorization"] == "Bearer test-key"
    assert client.default_headers["X-Api-Key"] is omit
    assert client.default_headers["Authorization"] is omit


@pytest.mark.asyncio
async def test_stream_simple_anthropic_emits_disabled_thinking_payload_when_reasoning_is_off() -> None:
    model = get_model("anthropic", "claude-opus-4-7")
    assert model is not None
    captured_payload: dict[str, object] | None = None

    def on_payload(payload: dict[str, object], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload
        raise RuntimeError("payload captured")

    stream = stream_simple_anthropic(
        model,
        Context(messages=[{"role": "user", "content": "Hello", "timestamp": 1_715_000_000_002}]),
        SimpleStreamOptions(apiKey="fake-key", onPayload=on_payload),
    )
    result = await stream.result()

    assert result.stopReason == "error"
    assert captured_payload is not None
    assert captured_payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in captured_payload


@pytest.mark.asyncio
async def test_stream_simple_anthropic_emits_adaptive_payload_when_force_adaptive_thinking_is_enabled() -> None:
    captured_payload: dict[str, object] | None = None

    def on_payload(payload: dict[str, object], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload
        raise RuntimeError("payload captured")

    stream = stream_simple_anthropic(
        _make_custom_model(force_adaptive_thinking=True),
        Context(messages=[{"role": "user", "content": "Hello", "timestamp": 1_715_000_000_003}]),
        SimpleStreamOptions(apiKey="fake-key", reasoning="medium", onPayload=on_payload),
    )
    await stream.result()

    assert captured_payload is not None
    assert captured_payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert captured_payload["output_config"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_stream_anthropic_applies_timeout_and_retry_via_with_options() -> None:
    model = get_model("anthropic", "claude-haiku-4-5")
    assert model is not None
    response = _FakeResponse(
        [
            {
                "event": "message_start",
                "data": (
                    '{"type":"message_start","message":{"id":"msg_test","usage":{"input_tokens":1,'
                    '"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}'
                ),
            },
            {
                "event": "content_block_start",
                "data": '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            },
            {
                "event": "content_block_delta",
                "data": '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}',
            },
            {"event": "content_block_stop", "data": '{"type":"content_block_stop","index":0}'},
            {
                "event": "message_delta",
                "data": (
                    '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,'
                    '"output_tokens":2,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}'
                ),
            },
            {"event": "message_stop", "data": '{"type":"message_stop"}'},
        ]
    )
    client = _FakeClientWithOptions(response)

    stream = stream_anthropic(
        model,
        _make_context(),
        {"client": client, "timeoutMs": 1500, "maxRetries": 4},
    )
    result = await stream.result()

    assert result.stopReason == "stop"
    assert client.with_options_calls == [{"timeout": 1.5, "max_retries": 4}]
    assert client.messages.payloads == [
        {
            "model": model.id,
            "messages": [{"role": "user", "content": "Use the tool."}],
            "max_tokens": model.maxTokens,
            "stream": True,
        }
    ]


def test_anthropic_module_exports_expected_names() -> None:
    assert anthropic_provider.__all__ == [
        "AnthropicEffort",
        "AnthropicOptions",
        "AnthropicThinkingDisplay",
        "streamAnthropic",
        "streamSimpleAnthropic",
    ]


def test_claude_code_tool_name_normalization_round_trips_matching_tools_only() -> None:
    tools = [
        Tool(name="todowrite", description="Write a todo item.", parameters={"type": "object", "properties": {}}),
        Tool(name="find", description="Find a file.", parameters={"type": "object", "properties": {}}),
    ]

    assert to_claude_code_name("todowrite") == "TodoWrite"
    assert to_claude_code_name("find") == "find"
    assert from_claude_code_name("TodoWrite", tools) == "todowrite"
    assert from_claude_code_name("Glob", tools) == "Glob"
