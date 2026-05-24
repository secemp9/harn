"""ANSI-stripping helpers derived from the upstream runtime."""

from __future__ import annotations

import re


def ansi_regex(*, only_first: bool = False) -> re.Pattern[str]:
    st = r"(?:\u0007|\u001B\u005C|\u009C)"
    osc = rf"(?:\u001B\][\s\S]*?{st})"
    csi = r"[\u001B\u009B][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
    pattern = f"{osc}|{csi}"
    return re.compile(pattern, 0 if only_first else re.MULTILINE)


_REGEX = ansi_regex()


def strip_ansi(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected a `string`, got `{type(value).__name__}`")
    if "\u001B" not in value and "\u009B" not in value:
        return value
    return _REGEX.sub("", value)


stripAnsi = strip_ansi

__all__ = ["stripAnsi", "strip_ansi"]
