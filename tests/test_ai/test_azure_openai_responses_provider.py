from __future__ import annotations

import asyncio

import httpx
import pytest
from harnify_ai.providers import azure_openai_responses as azure_provider
from harnify_ai.types import Context, Model, ModelCost


def _azure_model() -> Model:
    return Model(
        id="gpt-4.1",
        name="gpt-4.1",
        api="azure-openai-responses",
        provider="azure-openai-responses",
        baseUrl="https://example.openai.azure.com/openai/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=128_000,
        maxTokens=16_384,
    )


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "Hello", "timestamp": 1}])


class _RawResponse:
    def __init__(self, parsed: object) -> None:
        self.http_response = httpx.Response(
            202,
            headers={"x-ms-request-id": "req-123"},
            request=httpx.Request("POST", "https://example.openai.azure.com/openai/v1/responses"),
        )
        self._parsed = parsed

    async def parse(self) -> object:
        return self._parsed


class _ListStream:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = list(events)
        self.closed = False

    def __aiter__(self) -> _ListStream:
        return self

    async def __anext__(self) -> dict[str, object]:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def close(self) -> None:
        self.closed = True


class _BlockingStream:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False
        self.cancelled = False

    def __aiter__(self) -> _BlockingStream:
        return self

    async def __anext__(self) -> dict[str, object]:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_create_responses_stream_uses_raw_response_metadata_for_on_response() -> None:
    parsed_stream = object()
    raw_response = _RawResponse(parsed_stream)
    captured: dict[str, object] = {}

    class _WithRawResponse:
        async def create(self, **params):
            captured["params"] = params
            return raw_response

    class _Responses:
        with_raw_response = _WithRawResponse()

    class _Client:
        responses = _Responses()

    async def on_response(metadata: dict[str, object], model: Model) -> None:
        captured["metadata"] = metadata
        captured["model"] = model

    result = await azure_provider._create_responses_stream(
        _Client(),
        {"model": "deployment-1", "input": [], "stream": True},
        {"onResponse": on_response},
        _azure_model(),
    )

    assert result is parsed_stream
    assert captured["params"] == {"model": "deployment-1", "input": [], "stream": True}
    assert captured["metadata"] == {
        "status": 202,
        "headers": {"x-ms-request-id": "req-123"},
    }
    assert captured["model"].provider == "azure-openai-responses"


@pytest.mark.asyncio
async def test_stream_azure_openai_responses_emits_events_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_stream = _ListStream(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []},
            },
            {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
            {"type": "response.output_text.delta", "delta": "Hello"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
        ]
    )
    response_metadata: dict[str, object] | None = None

    class _WithRawResponse:
        async def create(self, **params):
            assert params["model"] == "gpt-4.1"
            return _RawResponse(response_stream)

    class _Responses:
        with_raw_response = _WithRawResponse()

    class _Client:
        responses = _Responses()

    async def on_response(metadata: dict[str, object], _model: Model) -> None:
        nonlocal response_metadata
        response_metadata = metadata

    monkeypatch.setattr(azure_provider, "create_client", lambda model, api_key, options=None: _Client())

    stream = azure_provider.stream_azure_openai_responses(
        _azure_model(),
        _context(),
        {"apiKey": "test-key", "onResponse": on_response},
    )

    event_types: list[str] = []
    async for event in stream:
        event_types.append(event.type)

    result = await stream.result()

    assert event_types == ["start", "text_start", "text_delta", "text_end", "done"]
    assert result.responseId == "resp_1"
    assert result.stopReason == "stop"
    assert result.content[0].text == "Hello"
    assert result.usage.input == 5
    assert result.usage.output == 3
    assert result.usage.totalTokens == 8
    assert response_metadata == {"status": 202, "headers": {"x-ms-request-id": "req-123"}}


@pytest.mark.asyncio
async def test_stream_azure_openai_responses_short_circuits_preaborted_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    class _WithRawResponse:
        async def create(self, **params):
            nonlocal request_count
            request_count += 1
            return _RawResponse(_ListStream([]))

    class _Responses:
        with_raw_response = _WithRawResponse()

    class _Client:
        responses = _Responses()

    signal = asyncio.Event()
    signal.set()
    monkeypatch.setattr(azure_provider, "create_client", lambda model, api_key, options=None: _Client())

    result = await azure_provider.stream_azure_openai_responses(
        _azure_model(),
        _context(),
        {"apiKey": "test-key", "signal": signal},
    ).result()

    assert request_count == 0
    assert result.stopReason == "aborted"
    assert result.errorMessage == "Request was aborted"


@pytest.mark.asyncio
async def test_stream_azure_openai_responses_aborts_while_waiting_for_stream_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking_stream = _BlockingStream()

    class _WithRawResponse:
        async def create(self, **params):
            return _RawResponse(blocking_stream)

    class _Responses:
        with_raw_response = _WithRawResponse()

    class _Client:
        responses = _Responses()

    signal = asyncio.Event()
    monkeypatch.setattr(azure_provider, "create_client", lambda model, api_key, options=None: _Client())

    stream = azure_provider.stream_azure_openai_responses(
        _azure_model(),
        _context(),
        {"apiKey": "test-key", "signal": signal},
    )

    await asyncio.wait_for(blocking_stream.entered.wait(), timeout=1)
    signal.set()
    result = await asyncio.wait_for(stream.result(), timeout=1)

    assert result.stopReason == "aborted"
    assert result.errorMessage == "Request was aborted"
    assert blocking_stream.cancelled is True
    assert blocking_stream.closed is True
