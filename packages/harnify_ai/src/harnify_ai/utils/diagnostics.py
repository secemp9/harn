"""Diagnostic helpers shared by AI providers and stream orchestration."""

from __future__ import annotations

import time
import traceback
from typing import Any

from pydantic import BaseModel, ConfigDict


class _DiagnosticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiagnosticErrorInfo(_DiagnosticsModel):
    name: str | None = None
    message: str
    stack: str | None = None
    code: str | int | None = None


class AssistantMessageDiagnostic(_DiagnosticsModel):
    type: str
    timestamp: int
    error: DiagnosticErrorInfo | None = None
    details: dict[str, Any] | None = None


def format_thrown_value(value: Any) -> str:
    if isinstance(value, BaseException):
        return str(value) or value.__class__.__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))

    code = getattr(error, "code", None)
    return DiagnosticErrorInfo(
        name=error.__class__.__name__ or None,
        message=str(error) or error.__class__.__name__,
        stack="".join(traceback.format_exception(error)).rstrip() or None,
        code=code if isinstance(code, (str, int)) else None,
    )


def create_assistant_message_diagnostic(
    type: str,
    error: Any,
    details: dict[str, Any] | None = None,
) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=type,
        timestamp=int(time.time() * 1000),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(message: Any, diagnostic: AssistantMessageDiagnostic) -> None:
    diagnostics = list(getattr(message, "diagnostics", None) or [])
    diagnostics.append(diagnostic)
    setattr(message, "diagnostics", diagnostics)


formatThrownValue = format_thrown_value
extractDiagnosticError = extract_diagnostic_error
createAssistantMessageDiagnostic = create_assistant_message_diagnostic
appendAssistantMessageDiagnostic = append_assistant_message_diagnostic
