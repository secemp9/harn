from __future__ import annotations

from pathlib import Path
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel

import harnify_ai.env_api_keys as env_api_keys
import harnify_ai.session_resources as session_resources_module
import harnify_ai.utils.validation as validation_utils
from harnify_ai.session_resources import cleanup_session_resources, register_session_resource_cleanup
from harnify_ai.types import AssistantMessage, Tool, ToolCall, Usage, UsageCost
from harnify_ai.utils.hash import short_hash
from harnify_ai.utils.headers import headers_to_record
from harnify_ai.utils.overflow import get_overflow_patterns, is_context_overflow
from harnify_ai.utils.sanitize_unicode import sanitize_surrogates
from harnify_ai.utils.validation import validate_tool_arguments, validate_tool_call


def _assistant_message(
    *,
    stop_reason: str = "error",
    error_message: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> AssistantMessage:
    return AssistantMessage(
        content=[{"type": "text", "text": "result"}],
        api="openai-responses",
        provider="openai",
        model="gpt-5",
        usage=Usage(
            input=input_tokens,
            output=output_tokens,
            cacheRead=0,
            cacheWrite=0,
            totalTokens=input_tokens + output_tokens,
            cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
        ),
        stopReason=stop_reason,
        errorMessage=error_message,
        timestamp=1_715_000_000_300,
    )


def test_short_hash_matches_upstream_js_outputs() -> None:
    assert short_hash("hello") == "1h6qa0qrowduu"
    assert short_hash("Hello 🙈 World") == "11begrz17n9aby"
    assert short_hash("Text \ud83d here") == "cu711p32odqm"


def test_headers_to_record_and_sanitize_surrogates_preserve_expected_content() -> None:
    headers = httpx.Headers([("x-one", "1"), ("x-two", "2")])

    assert headers_to_record(headers) == {"x-one": "1", "x-two": "2"}
    assert sanitize_surrogates("Hello 🙈 World") == "Hello 🙈 World"
    assert sanitize_surrogates("Text \ud83d here") == "Text  here"
    assert sanitize_surrogates("Pair \ud83d\ude48 ok") == "Pair \ud83d\ude48 ok"


def test_overflow_detection_handles_error_and_silent_overflow_cases() -> None:
    error_message = "Requested token count exceeds the model's maximum context length of 131072 tokens"
    assert is_context_overflow(_assistant_message(error_message=error_message)) is True
    assert is_context_overflow(_assistant_message(error_message="Rate limit exceeded")) is False
    assert is_context_overflow(_assistant_message(stop_reason="stop", error_message=None, input_tokens=205_000), 200_000)
    assert is_context_overflow(_assistant_message(stop_reason="length", error_message=None, input_tokens=199_000, output_tokens=0), 200_000)
    assert get_overflow_patterns()


def test_env_api_key_lookup_covers_priority_and_ambient_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in [
        "ANTHROPIC_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "AWS_PROFILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-token")
    assert env_api_keys.find_env_keys("anthropic") == ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]
    assert env_api_keys.get_env_api_key("anthropic") == "oauth-token"

    monkeypatch.setenv("AWS_PROFILE", "default")
    assert env_api_keys.get_env_api_key("amazon-bedrock") == "<authenticated>"

    credentials_file = tmp_path / "adc.json"
    credentials_file.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    env_api_keys._has_vertex_adc_credentials.cache_clear()
    assert env_api_keys.get_env_api_key("google-vertex") == "<authenticated>"


def test_session_resource_cleanup_runs_all_handlers_and_aggregates_failures() -> None:
    seen: list[tuple[str, str | None]] = []

    def ok_cleanup(session_id: str | None = None) -> None:
        seen.append(("ok", session_id))

    def bad_cleanup(session_id: str | None = None) -> None:
        seen.append(("bad", session_id))
        raise RuntimeError("boom")

    unregister_ok = register_session_resource_cleanup(ok_cleanup)
    unregister_bad = register_session_resource_cleanup(bad_cleanup)
    try:
        with pytest.raises(ExceptionGroup):
            cleanup_session_resources("session-123")
    finally:
        unregister_ok()
        unregister_bad()

    assert seen == [("ok", "session-123"), ("bad", "session-123")]


def test_session_resources_module_exports_expected_names() -> None:
    assert session_resources_module.__all__ == [
        "SessionResourceCleanup",
        "registerSessionResourceCleanup",
        "cleanupSessionResources",
    ]


def test_validate_tool_arguments_supports_pydantic_models_and_raw_json_schema() -> None:
    class WeatherArgs(BaseModel):
        city: str
        units: Literal["c", "f"] = "c"

    pydantic_tool = Tool(name="weather", description="Weather lookup", parameters=WeatherArgs)
    raw_tool = Tool(
        name="counter",
        description="Counter tool",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
            },
            "required": ["count", "enabled"],
        },
    )

    weather_args = validate_tool_arguments(
        pydantic_tool,
        ToolCall(id="1", name="weather", arguments={"city": "Paris", "units": "f"}),
    )
    counter_args = validate_tool_arguments(
        raw_tool,
        ToolCall(id="2", name="counter", arguments={"count": "5", "enabled": "true"}),
    )

    assert weather_args == {"city": "Paris", "units": "f"}
    assert counter_args == {"count": 5, "enabled": True}
    assert validate_tool_call([raw_tool], ToolCall(id="3", name="counter", arguments={"count": 1, "enabled": False})) == {
        "count": 1,
        "enabled": False,
    }

    with pytest.raises(ValueError, match='Tool "missing" not found'):
        validate_tool_call([], ToolCall(id="4", name="missing", arguments={}))

    with pytest.raises(ValueError, match='Validation failed for tool "counter"'):
        validate_tool_arguments(raw_tool, ToolCall(id="5", name="counter", arguments={"count": "nope"}))


def test_validate_tool_arguments_matches_ts_coercion_edges() -> None:
    tool = Tool(
        name="coerce",
        description="Coercion tool",
        parameters={
            "type": "object",
            "properties": {
                "flag": {"type": "string"},
                "count": {"type": ["integer", "string"]},
            },
            "required": ["flag", "count"],
        },
    )

    arguments = validate_tool_arguments(
        tool,
        ToolCall(id="6", name="coerce", arguments={"flag": True, "count": 1.0}),
    )

    assert arguments["flag"] == "true"
    assert isinstance(arguments["count"], float)
    assert arguments["count"] == 1.0


def test_validate_tool_arguments_formats_received_arguments_as_pretty_json() -> None:
    tool = Tool(
        name="counter",
        description="Counter tool",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
            "required": ["count"],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        validate_tool_arguments(tool, ToolCall(id="7", name="counter", arguments={"count": "nope"}))

    message = str(exc_info.value)
    assert 'Validation failed for tool "counter"' in message
    assert 'Received arguments:\n{\n  "count": "nope"\n}' in message
    assert "{'count': 'nope'}" not in message


def test_validation_module_exports_expected_names() -> None:
    assert validation_utils.__all__ == [
        "validateToolArguments",
        "validateToolCall",
        "validate_tool_arguments",
        "validate_tool_call",
    ]
