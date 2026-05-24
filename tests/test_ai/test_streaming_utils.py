from __future__ import annotations

import asyncio

import pytest

from harnify_ai.types import validate_assistant_message_event
from harnify_ai.utils.event_stream import AssistantMessageEventStream, EventStream
from harnify_ai.utils.json_parse import parse_json_with_repair, parse_streaming_json


def _assistant_message_payload(stop_reason: str = "stop") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": "stream output"}],
        "api": "openai-responses",
        "provider": "openai",
        "model": "gpt-5",
        "usage": {
            "input": 10,
            "output": 5,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 15,
            "cost": {
                "input": 10,
                "output": 20,
                "cacheRead": 0,
                "cacheWrite": 0,
                "total": 30,
            },
        },
        "stopReason": stop_reason,
        "timestamp": 1_715_000_000_100,
    }


@pytest.mark.asyncio
async def test_event_stream_delivers_waiting_consumer_and_final_result() -> None:
    stream = EventStream[str, str](lambda event: event == "done", lambda event: event.upper())
    collected: list[str] = []

    async def consume() -> None:
        async for event in stream:
            collected.append(event)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    stream.push("chunk")
    stream.push("done")
    stream.end("DONE")
    stream.push("ignored")

    await consumer

    assert collected == ["chunk", "done"]
    assert await stream.result() == "DONE"


@pytest.mark.asyncio
async def test_event_stream_requires_explicit_end_after_complete_event() -> None:
    stream = EventStream[str, str](lambda event: event == "done", lambda event: event.upper())
    consumer = asyncio.create_task(anext(stream.__aiter__()))

    stream.push("done")
    assert await consumer == "done"

    drain = asyncio.create_task(anext(stream.__aiter__(), "sentinel"))
    await asyncio.sleep(0)
    assert not drain.done()

    stream.end("DONE")
    assert await drain == "sentinel"
    assert await stream.result() == "DONE"


@pytest.mark.asyncio
async def test_assistant_message_event_stream_uses_done_and_error_payloads_as_results() -> None:
    stream = AssistantMessageEventStream()
    event = validate_assistant_message_event(
        {
            "type": "error",
            "reason": "error",
            "error": {
                **_assistant_message_payload(stop_reason="error"),
                "errorMessage": "provider failed",
            },
        }
    )

    stream.push(event)
    stream.end(event.error)
    events = [item async for item in stream]
    result = await stream.result()

    assert len(events) == 1
    assert result.stopReason == "error"
    assert result.errorMessage == "provider failed"


def test_parse_json_with_repair_handles_invalid_escapes_and_control_characters() -> None:
    invalid_escape = parse_json_with_repair(r'{"value":"bad\q"}')
    control_character = parse_json_with_repair('{"value":"line\nbreak"}')

    assert invalid_escape == {"value": r"bad\q"}
    assert control_character == {"value": "line\nbreak"}


def test_parse_streaming_json_recovers_partial_nested_objects() -> None:
    assert parse_streaming_json('{"tool":{"x":1') == {"tool": {"x": 1}}
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
