from __future__ import annotations

from harnify_ai.providers.google_shared import convert_messages, convert_tools
from harnify_ai.types import Context, Model, ModelCost, Tool


def _make_tool(parameters: dict[str, object]) -> Tool:
    return Tool(name="test_tool", description="A test tool", parameters=parameters)


def _make_model(api: str, provider: str, model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        baseUrl="https://example.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=128_000,
        maxTokens=8_192,
    )


def _usage_payload() -> dict[str, object]:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def test_convert_tools_strips_json_schema_meta_keys_for_parameters_mode() -> None:
    original_parameters = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "urn:bash-tool",
        "$comment": "A bash tool for demonstration",
        "$defs": {"commandDef": {"type": "string"}},
        "definitions": {"legacyDef": {"type": "number"}},
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    tools = [_make_tool(original_parameters)]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parameters"] == {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    assert "$schema" not in decl["parameters"]
    assert original_parameters["$schema"] == "http://json-schema.org/draft-07/schema#"


def test_convert_tools_preserves_schema_when_parameters_json_schema_mode_is_used() -> None:
    tools = [
        _make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        )
    ]

    result = convert_tools(tools, False)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parametersJsonSchema"] == {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    assert convert_tools([], True) is None


def test_convert_messages_keeps_separate_image_turn_for_gemini_2_models() -> None:
    model = _make_model("google-generative-ai", "google", "gemini-2.5-flash")
    now = 1_715_000_000_100
    context = Context.model_validate(
        {
            "messages": [
                {"role": "user", "content": "read the files", "timestamp": now},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call_a", "name": "read", "arguments": {"path": "a.txt"}},
                        {"type": "toolCall", "id": "call_img", "name": "read", "arguments": {"path": "image.png"}},
                        {"type": "toolCall", "id": "call_b", "name": "read", "arguments": {"path": "b.txt"}},
                    ],
                    "api": model.api,
                    "provider": model.provider,
                    "model": model.id,
                    "usage": _usage_payload(),
                    "stopReason": "toolUse",
                    "timestamp": now,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call_a",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "alpha text"}],
                    "isError": False,
                    "timestamp": now,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call_img",
                    "toolName": "read",
                    "content": [{"type": "image", "data": "abc", "mimeType": "image/png"}],
                    "isError": False,
                    "timestamp": now,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call_b",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "beta text"}],
                    "isError": False,
                    "timestamp": now,
                },
            ]
        }
    )

    contents = convert_messages(model, context)

    assert len(contents) == 5
    assert all("functionResponse" in part for part in contents[2]["parts"])
    assert contents[3]["parts"][0]["text"] == "Tool result image:"
    assert contents[3]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert "functionResponse" in contents[4]["parts"][0]


def test_convert_messages_nests_image_tool_results_for_gemini_3_models() -> None:
    model = _make_model("google-generative-ai", "google", "gemini-3-pro-preview")
    now = 1_715_000_000_100
    context = Context.model_validate(
        {
            "messages": [
                {"role": "user", "content": "read the files", "timestamp": now},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call_a", "name": "read", "arguments": {"path": "a.txt"}},
                        {"type": "toolCall", "id": "call_img", "name": "read", "arguments": {"path": "image.png"}},
                        {"type": "toolCall", "id": "call_b", "name": "read", "arguments": {"path": "b.txt"}},
                    ],
                    "api": model.api,
                    "provider": model.provider,
                    "model": model.id,
                    "usage": _usage_payload(),
                    "stopReason": "toolUse",
                    "timestamp": now,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call_a",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "alpha text"}],
                    "isError": False,
                    "timestamp": now,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call_img",
                    "toolName": "read",
                    "content": [{"type": "image", "data": "abc", "mimeType": "image/png"}],
                    "isError": False,
                    "timestamp": now,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call_b",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "beta text"}],
                    "isError": False,
                    "timestamp": now,
                },
            ]
        }
    )

    contents = convert_messages(model, context)

    assert len(contents) == 3
    tool_result_turn = contents[2]
    assert len(tool_result_turn["parts"]) == 3
    image_response = tool_result_turn["parts"][1]["functionResponse"]
    assert len(image_response["parts"]) == 1
    assert image_response["parts"][0]["inlineData"]["mimeType"] == "image/png"


def test_convert_messages_preserves_valid_tool_call_thought_signature_only_for_same_provider_and_model() -> None:
    model = _make_model("google-generative-ai", "google", "gemini-3-pro-preview")
    valid_signature = "AAAAAAAAAAAAAAAAAAAAAA=="
    context = Context.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hi", "timestamp": 1},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "echo hi"},
                            "thoughtSignature": valid_signature,
                        },
                        {
                            "type": "toolCall",
                            "id": "call_2",
                            "name": "bash",
                            "arguments": {"command": "ls -la"},
                        },
                    ],
                    "api": model.api,
                    "provider": model.provider,
                    "model": model.id,
                    "usage": _usage_payload(),
                    "stopReason": "toolUse",
                    "timestamp": 1,
                },
            ]
        }
    )

    contents = convert_messages(model, context)
    model_turn = next(content for content in contents if content["role"] == "model")
    function_call_parts = [part for part in model_turn["parts"] if "functionCall" in part]

    assert len(function_call_parts) == 2
    assert function_call_parts[0]["thoughtSignature"] == valid_signature
    assert "thoughtSignature" not in function_call_parts[1]

    foreign_context = Context.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hi", "timestamp": 1},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "echo hi"},
                            "thoughtSignature": valid_signature,
                        }
                    ],
                    "api": model.api,
                    "provider": model.provider,
                    "model": "other-model",
                    "usage": _usage_payload(),
                    "stopReason": "toolUse",
                    "timestamp": 1,
                },
            ]
        }
    )
    foreign_contents = convert_messages(model, foreign_context)
    foreign_turn = next(content for content in foreign_contents if content["role"] == "model")
    foreign_part = next(part for part in foreign_turn["parts"] if "functionCall" in part)
    assert "thoughtSignature" not in foreign_part
