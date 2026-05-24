from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from harnify_ai.types import Context, Tool, validate_assistant_message_event, validate_message


def _usage_payload() -> dict[str, object]:
    return {
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
    }


def _assistant_message_payload(stop_reason: str = "stop") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello from the model."},
            {"type": "thinking", "thinking": "Reasoning trace."},
            {
                "type": "toolCall",
                "id": "call_123",
                "name": "lookup_weather",
                "arguments": {"location": "Paris"},
            },
        ],
        "api": "openai-responses",
        "provider": "openai",
        "model": "gpt-5",
        "usage": _usage_payload(),
        "stopReason": stop_reason,
        "timestamp": 1_715_000_000_000,
    }


def test_context_validates_mixed_message_history() -> None:
    context = Context.model_validate(
        {
            "systemPrompt": "Be concise.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "data": "YmFzZTY0", "mimeType": "image/png"},
                    ],
                    "timestamp": 1_715_000_000_001,
                },
                _assistant_message_payload(),
                {
                    "role": "toolResult",
                    "toolCallId": "call_123",
                    "toolName": "lookup_weather",
                    "content": [{"type": "text", "text": "18 C and clear."}],
                    "isError": False,
                    "timestamp": 1_715_000_000_002,
                },
            ],
        }
    )

    assert context.systemPrompt == "Be concise."
    assert context.messages[0].role == "user"
    assert context.messages[0].content[1].type == "image"
    assert context.messages[1].role == "assistant"
    assert context.messages[1].content[1].type == "thinking"
    assert context.messages[1].content[2].arguments == {"location": "Paris"}
    assert context.messages[2].role == "toolResult"


def test_validate_message_parses_tool_result_payload() -> None:
    message = validate_message(
        {
            "role": "toolResult",
            "toolCallId": "call_123",
            "toolName": "lookup_weather",
            "content": [
                {"type": "text", "text": "18 C and clear."},
                {"type": "image", "data": "c3Vuc2V0", "mimeType": "image/png"},
            ],
            "details": {"provider": "mock"},
            "isError": False,
            "timestamp": 1_715_000_000_003,
        }
    )

    assert message.role == "toolResult"
    assert message.content[1].mimeType == "image/png"
    assert message.details == {"provider": "mock"}


def test_assistant_message_event_union_validates_done_payload() -> None:
    event = validate_assistant_message_event(
        {
            "type": "done",
            "reason": "toolUse",
            "message": _assistant_message_payload(stop_reason="toolUse"),
        }
    )

    assert event.type == "done"
    assert event.reason == "toolUse"
    assert event.message.content[0].text == "Hello from the model."


def test_assistant_message_event_rejects_incompatible_done_reason() -> None:
    with pytest.raises(ValidationError):
        validate_assistant_message_event(
            {
                "type": "done",
                "reason": "error",
                "message": _assistant_message_payload(stop_reason="error"),
            }
        )


def test_tool_accepts_pydantic_parameter_models_and_emits_json_schema() -> None:
    class WeatherArgs(BaseModel):
        location: str
        units: Literal["c", "f"] = "c"

    tool = Tool(
        name="lookup_weather",
        description="Look up current weather.",
        parameters=WeatherArgs,
    )

    schema = tool.parameters_json_schema()

    assert schema["properties"]["location"]["type"] == "string"
    assert schema["properties"]["units"]["enum"] == ["c", "f"]
    assert schema["required"] == ["location"]


def test_tool_accepts_raw_json_schema_mappings() -> None:
    raw_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    tool = Tool(name="lookup_city", description="Resolve city metadata.", parameters=raw_schema)

    schema = tool.parameters_json_schema()
    schema["properties"]["country"] = {"type": "string"}

    assert tool.parameters == raw_schema
    assert "country" not in raw_schema["properties"]
