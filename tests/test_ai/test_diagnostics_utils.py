from __future__ import annotations

from types import SimpleNamespace

from harnify_ai.utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)


class _NamedError(RuntimeError):
    def __init__(self, message: str, *, name: str, code: str | int | None = None, stack: str | None = None) -> None:
        super().__init__(message)
        self.name = name
        self.code = code
        self.stack = stack


def test_format_thrown_value_matches_error_message_or_name() -> None:
    assert format_thrown_value(_NamedError("", name="CustomError")) == "CustomError"
    assert format_thrown_value("plain text") == "plain text"
    assert format_thrown_value(123) == "123"


def test_extract_diagnostic_error_prefers_runtime_name_stack_and_code() -> None:
    diagnostic = extract_diagnostic_error(
        _NamedError("broken", name="CustomError", code="E_CUSTOM", stack="stack trace")
    )

    assert diagnostic.name == "CustomError"
    assert diagnostic.message == "broken"
    assert diagnostic.stack == "stack trace"
    assert diagnostic.code == "E_CUSTOM"


def test_create_and_append_assistant_message_diagnostic() -> None:
    diagnostic = create_assistant_message_diagnostic("provider_error", RuntimeError("boom"), {"provider": "openai"})
    message = SimpleNamespace(diagnostics=None)

    append_assistant_message_diagnostic(message, diagnostic)

    assert len(message.diagnostics) == 1
    assert message.diagnostics[0].type == "provider_error"
    assert message.diagnostics[0].details == {"provider": "openai"}
