from __future__ import annotations

import asyncio
import base64
import json
from email.utils import formatdate

import httpx
import pytest
from harnify_ai.providers import openai_codex_responses as codex_provider
from harnify_ai.providers.openai_codex_responses import (
    build_request_body,
    build_sse_headers,
    close_openai_codex_websocket_sessions,
    extract_account_id,
    get_openai_codex_websocket_debug_stats,
    parse_error_response,
    parse_sse,
    reset_openai_codex_websocket_debug_stats,
    resolve_codex_url,
    stream_openai_codex_responses,
    stream_simple_openai_codex_responses,
)
from harnify_ai.types import Context, Model, ModelCost, SimpleStreamOptions


def _mock_token() -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acc_test"}}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"aaa.{payload}.bbb"


def _codex_model(model_id: str = "gpt-5.1-codex") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-codex-responses",
        provider="openai-codex",
        baseUrl="https://chatgpt.com/backend-api",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=400_000,
        maxTokens=128_000,
        thinkingLevelMap={"xhigh": "xhigh", "minimal": "low"},
    )


def _context() -> Context:
    return Context(
        systemPrompt="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Say hello", "timestamp": 1}],
    )


def _sse_payload(status: str = "completed") -> str:
    terminal_type = "response.incomplete" if status == "incomplete" else "response.completed"
    incomplete_details = {"reason": "max_output_tokens"} if status == "incomplete" else None
    events = [
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
            "type": terminal_type,
            "response": {
                "status": status,
                "incomplete_details": incomplete_details,
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        },
        "[DONE]",
    ]
    payload = "\n\n".join(
        f"data: {json.dumps(event)}" if isinstance(event, dict) else f"data: {event}"
        for event in events
    )
    return payload + "\n\n"


def _websocket_events(response_id: str, message_id: str, text: str) -> list[dict[str, object]]:
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": message_id, "role": "assistant", "status": "in_progress", "content": []},
        },
        {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
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


class _MockWebSocket:
    def __init__(self, responses: list[list[dict[str, object]]], sent_bodies: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self._sent_bodies = sent_bodies
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False
        self.state = 1

    async def send(self, data: str) -> None:
        self._sent_bodies.append(json.loads(data))
        if not self._responses:
            raise RuntimeError("unexpected websocket request")
        events = self._responses.pop(0)
        for event in events:
            await self._queue.put(json.dumps(event))

    async def recv(self) -> str:
        return await self._queue.get()

    def close(self, code: int = 1000, reason: str = "done") -> None:
        self.closed = True
        self.state = 3


class _InvalidJsonWebSocket:
    closed = False
    state = 1

    async def send(self, data: str) -> None:
        return None

    async def recv(self) -> str:
        return "{invalid json"

    def close(self, code: int = 1000, reason: str = "done") -> None:
        self.closed = True
        self.state = 3


class _ListByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> _ListByteStream:
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class _BlockingByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False
        self.cancelled = False

    def __aiter__(self) -> _BlockingByteStream:
        return self

    async def __anext__(self) -> bytes:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_codex_websocket_state():
    close_openai_codex_websocket_sessions()
    reset_openai_codex_websocket_debug_stats()
    yield
    close_openai_codex_websocket_sessions()
    reset_openai_codex_websocket_debug_stats()


def test_build_request_body_and_headers_include_codex_specific_fields() -> None:
    session_id = "x" * 67
    model = _codex_model("gpt-5.5")
    context = _context()

    body = build_request_body(
        model,
        context,
        {"sessionId": session_id, "reasoningEffort": "xhigh", "textVerbosity": "high"},
    )
    headers = build_sse_headers(None, None, "acc_test", _mock_token(), "session-1")

    assert body["instructions"] == "You are a helpful assistant."
    assert body["text"] == {"verbosity": "high"}
    assert body["prompt_cache_key"] == "x" * 64
    assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["input"][0]["role"] == "user"

    assert headers["authorization"].startswith("Bearer ")
    assert headers["chatgpt-account-id"] == "acc_test"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["originator"] == "pi"
    assert headers["session_id"] == "session-1"
    assert headers["x-client-request-id"] == "session-1"


def test_build_request_body_uses_nullish_defaults_like_upstream() -> None:
    body = build_request_body(
        _codex_model(),
        _context(),
        {"textVerbosity": None, "reasoningEffort": "xhigh", "reasoningSummary": None},
    )

    assert body["text"] == {"verbosity": "low"}
    assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}


def test_build_sse_headers_preserve_custom_headers_but_override_reserved_values() -> None:
    headers = build_sse_headers(
        {"authorization": "bad", "originator": "bad", "x-base": "1", "accept": "bad"},
        {"authorization": "worse", "OpenAI-Beta": "bad", "User-Agent": "bad", "x-extra": "2"},
        "acc_test",
        _mock_token(),
        "session-1",
    )

    assert headers["authorization"].startswith("Bearer ")
    assert headers["chatgpt-account-id"] == "acc_test"
    assert headers["originator"] == "pi"
    assert headers["accept"] == "text/event-stream"
    assert headers["content-type"] == "application/json"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["User-Agent"].startswith("pi (")
    assert headers["x-base"] == "1"
    assert headers["x-extra"] == "2"


def test_extract_account_id_and_resolve_codex_url() -> None:
    token = _mock_token()

    assert extract_account_id(token) == "acc_test"
    assert resolve_codex_url() == "https://chatgpt.com/backend-api/codex/responses"
    assert resolve_codex_url("https://chatgpt.com/backend-api") == "https://chatgpt.com/backend-api/codex/responses"
    assert resolve_codex_url("https://chatgpt.com/backend-api/codex") == "https://chatgpt.com/backend-api/codex/responses"
    assert resolve_codex_url("https://chatgpt.com/backend-api/codex/responses") == "https://chatgpt.com/backend-api/codex/responses"


def test_extract_account_id_raises_for_invalid_token() -> None:
    with pytest.raises(RuntimeError, match="Failed to extract accountId from token"):
        extract_account_id("not-a-jwt")


@pytest.mark.asyncio
async def test_stream_openai_codex_responses_sends_expected_request_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _mock_token()
    captured: dict[str, object] = {}

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload("completed"),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": token, "transport": "sse", "sessionId": "test-session-123"},
    ).result()

    headers = captured["headers"]
    body = captured["body"]

    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert headers["authorization"] == f"Bearer {token}"
    assert headers["chatgpt-account-id"] == "acc_test"
    assert headers["openai-beta"] == "responses=experimental"
    assert headers["originator"] == "pi"
    assert headers["session_id"] == "test-session-123"
    assert headers["x-client-request-id"] == "test-session-123"
    assert body["prompt_cache_key"] == "test-session-123"
    assert result.stopReason == "stop"
    assert result.content[0].text == "Hello"


@pytest.mark.asyncio
async def test_stream_openai_codex_responses_maps_incomplete_to_length(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload("incomplete"),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "transport": "sse"},
    ).result()

    assert result.stopReason == "length"
    assert result.content[0].text == "Hello"


@pytest.mark.asyncio
async def test_stream_openai_codex_responses_does_not_use_openai_env_fallback_for_openai_codex_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _mock_token()

    monkeypatch.setenv("OPENAI_API_KEY", token)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"transport": "sse"},
    ).result()

    assert result.stopReason == "error"
    assert result.errorMessage == "No API key for provider: openai-codex"


@pytest.mark.asyncio
async def test_stream_openai_codex_responses_short_circuits_preaborted_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = asyncio.Event()
    signal.set()
    send_calls = 0

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        nonlocal send_calls
        send_calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload("completed"),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "transport": "sse", "signal": signal},
    ).result()

    assert send_calls == 0
    assert result.stopReason == "aborted"
    assert result.errorMessage == "Request was aborted"


def test_parse_retry_after_delay_ms_supports_milliseconds_header() -> None:
    response = httpx.Response(
        429,
        headers={"retry-after-ms": "2500"},
        request=httpx.Request("POST", "https://example.com"),
    )

    assert codex_provider._parse_retry_after_delay_ms(response, 1000) == 2500


def test_parse_retry_after_delay_ms_supports_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_700_000_000
    response = httpx.Response(
        429,
        headers={"retry-after": formatdate(now + 3, usegmt=True)},
        request=httpx.Request("POST", "https://example.com"),
    )
    monkeypatch.setattr(codex_provider.time, "time", lambda: now)

    assert codex_provider._parse_retry_after_delay_ms(response, 1000) == 3000


@pytest.mark.asyncio
async def test_parse_error_response_formats_usage_limit_message(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_700_000_000
    monkeypatch.setattr(codex_provider.time, "time", lambda: now)
    response = httpx.Response(
        429,
        text=json.dumps(
            {
                "error": {
                    "code": "usage_limit_reached",
                    "plan_type": "PLUS",
                    "resets_at": now + 180,
                }
            }
        ),
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )

    info = await parse_error_response(response)

    assert info == {
        "message": "You have hit your ChatGPT usage limit (plus plan). Try again in ~3 min.",
        "friendlyMessage": "You have hit your ChatGPT usage limit (plus plan). Try again in ~3 min.",
    }


@pytest.mark.asyncio
async def test_parse_sse_ignores_trailing_chunk_without_separator() -> None:
    stream = _ListByteStream([b'data: {"type":"response.completed"}'])
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )

    events = [event async for event in parse_sse(response)]

    assert events == []


@pytest.mark.asyncio
async def test_stream_simple_openai_codex_responses_preserves_xhigh_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, object] | None = None

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload("completed"),
            request=request,
        )

    def on_payload(payload: dict[str, object], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    await stream_simple_openai_codex_responses(
        _codex_model("gpt-5.5"),
        _context(),
        SimpleStreamOptions(apiKey=_mock_token(), reasoning="xhigh", transport="sse", onPayload=on_payload),
    ).result()

    assert captured_payload is not None
    assert captured_payload["reasoning"] == {"effort": "xhigh", "summary": "auto"}


@pytest.mark.asyncio
async def test_parse_sse_aborts_waiting_for_next_chunk() -> None:
    signal = asyncio.Event()
    stream = _BlockingByteStream()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )

    async def consume() -> None:
        async for _event in parse_sse(response, signal):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(stream.entered.wait(), timeout=1)
    signal.set()

    with pytest.raises(RuntimeError, match="Request was aborted"):
        await asyncio.wait_for(task, timeout=1)

    assert stream.closed is True
    assert stream.cancelled is True


@pytest.mark.asyncio
async def test_stream_openai_codex_responses_clears_partial_json_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload("completed"),
            request=request,
        )

    async def fake_process_stream(*args, **kwargs) -> None:
        output = args[1]
        output.content = [{"type": "reasoning", "partialJson": '{"a":1}'}]
        raise RuntimeError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setattr(codex_provider, "process_stream", fake_process_stream)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "transport": "sse"},
    ).result()

    assert result.stopReason == "error"
    assert result.errorMessage == "boom"
    assert result.content == [{"type": "reasoning"}]


@pytest.mark.asyncio
async def test_stream_openai_codex_responses_uses_websocket_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_bodies: list[dict[str, object]] = []
    socket = _MockWebSocket([_websocket_events("resp_ws_1", "msg_ws_1", "Hello")], sent_bodies)

    async def fake_connect(
        _url: str,
        _headers: dict[str, str],
        _signal=None,
        _timeout_ms: int | None = None,
    ) -> _MockWebSocket:
        return socket

    async def fail_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        raise AssertionError("SSE transport should not be used when websocket succeeds")

    monkeypatch.setattr(codex_provider, "_connect_websocket", fake_connect)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail_send)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "transport": "websocket", "sessionId": "session-ws"},
    ).result()

    assert result.content[0].text == "Hello"
    assert sent_bodies[0]["type"] == "response.create"
    assert sent_bodies[0]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Say hello"}]}]
    assert get_openai_codex_websocket_debug_stats("session-ws") == {
        "requests": 1,
        "connectionsCreated": 1,
        "connectionsReused": 0,
        "cachedContextRequests": 0,
        "storeTrueRequests": 0,
        "fullContextRequests": 1,
        "deltaRequests": 0,
        "lastInputItems": 1,
        "websocketFailures": 0,
        "sseFallbacks": 0,
        "lastDeltaInputItems": None,
        "lastPreviousResponseId": None,
    }


@pytest.mark.asyncio
async def test_stream_simple_openai_codex_responses_auto_uses_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_bodies: list[dict[str, object]] = []
    socket = _MockWebSocket([_websocket_events("resp_auto_1", "msg_auto_1", "Hello")], sent_bodies)

    async def fake_connect(
        _url: str,
        _headers: dict[str, str],
        _signal=None,
        _timeout_ms: int | None = None,
    ) -> _MockWebSocket:
        return socket

    async def fail_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        raise AssertionError("SSE transport should not be used in successful auto websocket mode")

    monkeypatch.setattr(codex_provider, "_connect_websocket", fake_connect)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail_send)

    result = await stream_simple_openai_codex_responses(
        _codex_model(),
        _context(),
        SimpleStreamOptions(apiKey=_mock_token(), sessionId="session-auto", transport="auto"),
    ).result()

    assert result.content[0].text == "Hello"
    assert len(sent_bodies) == 1
    assert get_openai_codex_websocket_debug_stats("session-auto") == {
        "requests": 1,
        "connectionsCreated": 1,
        "connectionsReused": 0,
        "cachedContextRequests": 1,
        "storeTrueRequests": 0,
        "fullContextRequests": 1,
        "deltaRequests": 0,
        "lastInputItems": 1,
        "websocketFailures": 0,
        "sseFallbacks": 0,
        "lastDeltaInputItems": None,
        "lastPreviousResponseId": None,
    }


@pytest.mark.asyncio
async def test_websocket_cached_mode_sends_only_delta_input_and_reuses_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_bodies: list[dict[str, object]] = []
    socket = _MockWebSocket(
        [
            _websocket_events("resp_1", "msg_1", "Hello"),
            _websocket_events("resp_2", "msg_2", "Done"),
        ],
        sent_bodies,
    )
    connect_calls = 0

    async def fake_connect(
        _url: str,
        _headers: dict[str, str],
        _signal=None,
        _timeout_ms: int | None = None,
    ) -> _MockWebSocket:
        nonlocal connect_calls
        connect_calls += 1
        return socket

    async def fail_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        raise AssertionError("SSE transport should not be used when websocket-cached succeeds")

    monkeypatch.setattr(codex_provider, "_connect_websocket", fake_connect)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail_send)

    first_context = _context()
    first = await stream_openai_codex_responses(
        _codex_model(),
        first_context,
        {"apiKey": _mock_token(), "sessionId": "session-1", "transport": "websocket-cached"},
    ).result()

    second_context = Context(
        systemPrompt="You are a helpful assistant.",
        messages=[
            *first_context.messages,
            first,
            {"role": "user", "content": "Now finish", "timestamp": 2},
        ],
    )
    await stream_openai_codex_responses(
        _codex_model(),
        second_context,
        {"apiKey": _mock_token(), "sessionId": "session-1", "transport": "websocket-cached"},
    ).result()

    assert connect_calls == 1
    assert len(sent_bodies) == 2
    first_body = sent_bodies[0]
    second_body = sent_bodies[1]
    assert first_body["store"] is False
    assert first_body.get("previous_response_id") is None
    assert first_body["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Say hello"}]}]
    assert second_body["store"] is False
    assert second_body["previous_response_id"] == "resp_1"
    assert second_body["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]}]
    assert get_openai_codex_websocket_debug_stats("session-1") == {
        "requests": 2,
        "connectionsCreated": 1,
        "connectionsReused": 1,
        "cachedContextRequests": 2,
        "storeTrueRequests": 0,
        "fullContextRequests": 1,
        "deltaRequests": 1,
        "lastInputItems": 1,
        "lastDeltaInputItems": 1,
        "lastPreviousResponseId": "resp_1",
        "websocketFailures": 0,
        "sseFallbacks": 0,
    }


@pytest.mark.asyncio
async def test_websocket_transport_falls_back_to_sse_and_disables_future_session_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_calls = 0

    async def failing_connect(
        _url: str,
        _headers: dict[str, str],
        _signal=None,
        _timeout_ms: int | None = None,
    ) -> _MockWebSocket:
        nonlocal connect_calls
        connect_calls += 1
        raise RuntimeError("websocket unavailable")

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload("completed"),
            request=request,
        )

    monkeypatch.setattr(codex_provider, "_connect_websocket", failing_connect)
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    first = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "sessionId": "fallback-session", "transport": "websocket"},
    ).result()
    second = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "sessionId": "fallback-session", "transport": "auto"},
    ).result()

    assert first.content[0].text == "Hello"
    assert second.content[0].text == "Hello"
    assert connect_calls == 1
    assert first.diagnostics is not None
    assert first.diagnostics[0].type == "provider_transport_failure"
    assert get_openai_codex_websocket_debug_stats("fallback-session") == {
        "requests": 0,
        "connectionsCreated": 0,
        "connectionsReused": 0,
        "cachedContextRequests": 0,
        "storeTrueRequests": 0,
        "fullContextRequests": 0,
        "deltaRequests": 0,
        "lastInputItems": 0,
        "websocketFailures": 1,
        "sseFallbacks": 2,
        "websocketFallbackActive": True,
        "lastWebSocketError": "websocket unavailable",
    }


@pytest.mark.asyncio
async def test_close_and_reset_openai_codex_websocket_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_bodies: list[dict[str, object]] = []
    socket = _MockWebSocket([_websocket_events("resp_close_1", "msg_close_1", "Hello")], sent_bodies)

    async def fake_connect(
        _url: str,
        _headers: dict[str, str],
        _signal=None,
        _timeout_ms: int | None = None,
    ) -> _MockWebSocket:
        return socket

    async def fail_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        raise AssertionError("SSE transport should not be used when websocket-cached succeeds")

    monkeypatch.setattr(codex_provider, "_connect_websocket", fake_connect)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail_send)

    await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "sessionId": "session-close", "transport": "websocket-cached"},
    ).result()

    assert socket.closed is False
    assert get_openai_codex_websocket_debug_stats("session-close") is not None

    close_openai_codex_websocket_sessions("session-close")
    assert socket.closed is True

    reset_openai_codex_websocket_debug_stats("session-close")
    assert get_openai_codex_websocket_debug_stats("session-close") is None


@pytest.mark.asyncio
async def test_websocket_protocol_error_surfaces_without_sse_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_connect(
        _url: str,
        _headers: dict[str, str],
        _signal=None,
        _timeout_ms: int | None = None,
    ) -> _InvalidJsonWebSocket:
        return _InvalidJsonWebSocket()

    async def fail_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        raise AssertionError("SSE fallback should not run for websocket protocol errors")

    monkeypatch.setattr(codex_provider, "_connect_websocket", fake_connect)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail_send)

    result = await stream_openai_codex_responses(
        _codex_model(),
        _context(),
        {"apiKey": _mock_token(), "sessionId": "proto-session", "transport": "websocket"},
    ).result()

    assert result.stopReason == "error"
    assert result.errorMessage is not None
    assert "Invalid Codex WebSocket JSON" in result.errorMessage
