"""Path resolution helpers for file-oriented tools."""

from __future__ import annotations

import os

from harnify_coding_agent.utils.paths import normalize_path, resolve_path

NARROW_NO_BREAK_SPACE = "\u202F"


def try_macos_screenshot_path(file_path: str) -> str:
    return file_path.replace(" AM.", f"{NARROW_NO_BREAK_SPACE}AM.").replace(" PM.", f"{NARROW_NO_BREAK_SPACE}PM.")


def try_nfd_variant(file_path: str) -> str:
    return file_path.normalize("NFD") if hasattr(file_path, "normalize") else file_path


def try_curly_quote_variant(file_path: str) -> str:
    return file_path.replace("'", "\u2019")


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def expand_path(file_path: str) -> str:
    return normalize_path(file_path, normalize_unicode_spaces=True, strip_at_prefix=True)


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    return resolve_path(file_path, cwd, normalize_unicode_spaces=True, strip_at_prefix=True)


def resolve_read_path(file_path: str, cwd: str) -> str:
    resolved = resolve_to_cwd(file_path, cwd)
    if file_exists(resolved):
        return resolved

    am_pm_variant = try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and file_exists(am_pm_variant):
        return am_pm_variant

    nfd_variant = unicodedata_nfd(resolved)
    if nfd_variant != resolved and file_exists(nfd_variant):
        return nfd_variant

    curly_variant = try_curly_quote_variant(resolved)
    if curly_variant != resolved and file_exists(curly_variant):
        return curly_variant

    nfd_curly_variant = try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and file_exists(nfd_curly_variant):
        return nfd_curly_variant

    return resolved


def unicodedata_nfd(value: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFD", value)


expandPath = expand_path
resolveToCwd = resolve_to_cwd
resolveReadPath = resolve_read_path

__all__ = ["expandPath", "expand_path", "resolveReadPath", "resolveToCwd", "resolve_read_path", "resolve_to_cwd"]
