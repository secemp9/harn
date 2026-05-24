from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from harnify_agent import AbortController, Agent, ProxyStreamOptions, process_proxy_event, stream_proxy
from harnify_ai.types import AssistantMessage, Context, Model, Usage


def _model() -> Model:
    return Model(
        id="proxy-model",
        name="Proxy Model",
        api="openai-responses",
        provider="openai",
        baseUrl="https://proxy.example.invalid",
        reasoning=True,
        input=["text"],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=128_000,
        maxTokens=8_192,
    )


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hello", "timestamp": 1}])


def _usage() -> dict[str, Any]:
    return {
        "input": 1,
        "output": 2,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 3,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def _sse_payload(events: list[dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(event)}\n" for event in events)


def test_process_proxy_event_reconstructs_tool_call_arguments() -> None:
    partial = AssistantMessage(
        role="assistant",
        content=[],
        api="openai-responses",
        provider="openai",
        model="proxy-model",
        usage=Usage.model_validate(_usage()),
        stopReason="stop",
        timestamp=1,
    )
    buffers: dict[int, str] = {}

    assert process_proxy_event({"type": "start"}, partial, buffers) is not None
    assert process_proxy_event(
        {"type": "toolcall_start", "contentIndex": 0, "id": "tool-1", "toolName": "echo"},
        partial,
        buffers,
    ) is not None
    process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": '{"value": "'}, partial, buffers)
    event = process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": 'hello"}'}, partial, buffers)

    assert event is not None
    assert partial.content[0].type == "toolCall"
    assert partial.content[0].arguments == {"value": "hello"}
    assert process_proxy_event({"type": "toolcall_end", "contentIndex": 1}, partial, buffers) is None


@pytest.mark.asyncio
async def test_stream_proxy_posts_expected_payload_and_reconstructs_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = [
        {"type": "start"},
        {"type": "thinking_start", "contentIndex": 0},
        {"type": "thinking_delta", "contentIndex": 0, "delta": "plan"},
        {"type": "thinking_end", "contentIndex": 0, "contentSignature": "sig-think"},
        {"type": "text_start", "contentIndex": 1},
        {"type": "text_delta", "contentIndex": 1, "delta": "Hello"},
        {"type": "text_end", "contentIndex": 1, "contentSignature": "sig-text"},
        {"type": "done", "reason": "stop", "usage": _usage()},
    ]

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload(events),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = await stream_proxy(
        _model(),
        _context(),
        ProxyStreamOptions(authToken="token-123", proxyUrl="https://proxy.example.invalid", reasoning="high"),
    ).result()

    assert captured["url"] == "https://proxy.example.invalid/api/stream"
    assert captured["headers"]["authorization"] == "Bearer token-123"
    assert captured["body"]["options"]["reasoning"] == "high"
    assert result.stopReason == "stop"
    assert [block.type for block in result.content] == ["thinking", "text"]
    assert result.content[0].thinking == "plan"
    assert result.content[0].thinkingSignature == "sig-think"
    assert result.content[1].text == "Hello"
    assert result.content[1].textSignature == "sig-text"


@pytest.mark.asyncio
async def test_agent_prompt_with_proxy_stream_uses_normal_agent_event_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _sse_payload(
        [
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "Hello"},
            {"type": "text_end", "contentIndex": 0, "contentSignature": "sig"},
            {"type": "done", "reason": "stop", "usage": _usage()},
        ]
    )

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=payload,
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    agent = Agent(
        initialState={"model": _model()},
        streamFn=lambda model, context, options=None: stream_proxy(
            model,
            context,
            {
                **(options.model_dump(exclude_none=True) if options is not None else {}),
                "authToken": "token-123",
                "proxyUrl": "https://proxy.example.invalid",
            },
        ),
    )
    event_types: list[str] = []
    agent.subscribe(lambda event, _signal: event_types.append(event.type))

    await agent.prompt("hello")

    assert event_types[:4] == ["agent_start", "turn_start", "message_start", "message_end"]
    assert "message_update" in event_types
    assert event_types[-3:] == ["message_end", "turn_end", "agent_end"]
    assert agent.state.messages[-1].role == "assistant"
    assert agent.state.messages[-1].content[0].text == "Hello"


@pytest.mark.asyncio
async def test_stream_proxy_reports_abort_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = AbortController()

    async def fake_send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        await asyncio.sleep(0.1)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_payload(
                [
                    {"type": "start"},
                    {"type": "done", "reason": "stop", "usage": _usage()},
                ]
            ),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    stream = stream_proxy(
        _model(),
        _context(),
        {
            "authToken": "token-123",
            "proxyUrl": "https://proxy.example.invalid",
            "signal": controller.signal,
        },
    )
    await asyncio.sleep(0.02)
    controller.abort()
    result = await stream.result()

    assert result.stopReason == "aborted"
    assert result.errorMessage == "Request aborted by user"


def test_root_exports_include_phase3_proxy_and_loop_surface() -> None:
    import harnify_agent

    assert callable(harnify_agent.stream_proxy)
    assert callable(harnify_agent.run_agent_loop)
    assert callable(harnify_agent.run_agent_loop_continue)
    assert harnify_agent.ProxyStreamOptions is ProxyStreamOptions
    assert callable(harnify_agent.collectEntriesForBranchSummary)
    assert callable(harnify_agent.compact)
    assert callable(harnify_agent.load_prompt_templates)
    assert harnify_agent.JsonlSessionRepo.__name__ == "JsonlSessionRepo"
    assert callable(harnify_agent.execute_shell_with_capture)
    assert callable(harnify_agent.truncate_tail)
