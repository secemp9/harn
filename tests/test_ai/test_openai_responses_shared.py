from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from harnify_ai.models import get_model
import harnify_ai.providers.openai_responses_shared as shared
from harnify_ai.providers.openai_responses_shared import (
    convert_responses_messages,
    process_responses_stream,
)
from harnify_ai.types import AssistantMessage, Context, Model, Usage, UsageCost


def _responses_model() -> Model:
    model = get_model("openai", "gpt-4o-mini")
    assert model is not None
    return model.model_copy(update={"api": "openai-responses"})


def _usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cacheRead=0,
        cacheWrite=0,
        totalTokens=0,
        cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
    )


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[object] = []

    def push(self, event: object) -> None:
        self.events.append(event)


async def _stream(events: list[dict[str, object]]) -> AsyncIterator[dict[str, object]]:
    for event in events:
        yield event


def test_openai_responses_shared_exports_option_types() -> None:
    assert "OpenAIResponsesStreamOptions" in shared.__all__
    assert "ConvertResponsesMessagesOptions" in shared.__all__
    assert "ConvertResponsesToolsOptions" in shared.__all__
    assert hasattr(shared, "OpenAIResponsesStreamOptions")
    assert hasattr(shared, "ConvertResponsesMessagesOptions")
    assert hasattr(shared, "ConvertResponsesToolsOptions")


def test_convert_responses_messages_uses_ts_style_tool_call_id_splitting() -> None:
    model = _responses_model()
    assistant = AssistantMessage(
        content=[{"type": "toolCall", "id": "call|item|extra", "name": "lookup", "arguments": {"q": "x"}}],
        api="openai-responses",
        provider="openai",
        model=model.id,
        usage=_usage(),
        stopReason="toolUse",
        timestamp=2,
    )

    converted = convert_responses_messages(
        model,
        Context(messages=[assistant]),
        {"openai"},
    )

    function_calls = [item for item in converted if item.get("type") == "function_call"]
    assert function_calls == [
        {
            "type": "function_call",
            "id": "item",
            "call_id": "call",
            "name": "lookup",
            "arguments": '{"q": "x"}',
        }
    ]


def test_convert_responses_messages_raises_on_invalid_thinking_signature() -> None:
    model = _responses_model()
    assistant = AssistantMessage(
        content=[{"type": "thinking", "thinking": "draft", "thinkingSignature": "{"}],
        api="openai-responses",
        provider="openai",
        model=model.id,
        usage=_usage(),
        stopReason="stop",
        timestamp=2,
    )

    with pytest.raises(ValueError):
        convert_responses_messages(
            model,
            Context(messages=[assistant]),
            {"openai"},
        )


@pytest.mark.asyncio
async def test_process_responses_stream_ignores_summary_delta_without_summary_part() -> None:
    model = _responses_model()
    output = AssistantMessage(
        content=[],
        api="openai-responses",
        provider="openai",
        model=model.id,
        usage=_usage(),
        stopReason="stop",
        timestamp=1,
    )
    collector = _EventCollector()

    await process_responses_stream(
        _stream(
            [
                {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
                {"type": "response.reasoning_summary_text.delta", "delta": "hidden"},
                {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs_1", "summary": []}},
                {"type": "response.completed", "response": {"status": "completed"}},
            ]
        ),
        output,
        collector,
        model,
    )

    assert [event.type for event in collector.events] == ["thinking_start", "thinking_end"]
    assert output.content[0].thinking == ""


@pytest.mark.asyncio
async def test_process_responses_stream_error_event_preserves_ts_format_when_message_missing() -> None:
    model = _responses_model()
    output = AssistantMessage(
        content=[],
        api="openai-responses",
        provider="openai",
        model=model.id,
        usage=_usage(),
        stopReason="stop",
        timestamp=1,
    )

    with pytest.raises(RuntimeError, match=r"^Error Code None: None$"):
        await process_responses_stream(
            _stream([{"type": "error"}]),
            output,
            _EventCollector(),
            model,
        )
