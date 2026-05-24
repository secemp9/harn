"""Ring buffer for Emacs-style kill and yank behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class KillRingPushOptions:
    prepend: bool
    accumulate: bool = False


class KillRing:
    def __init__(self) -> None:
        self._ring: list[str] = []

    def push(self, text: str, opts: KillRingPushOptions | dict[str, bool]) -> None:
        if not text:
            return

        options = opts if isinstance(opts, KillRingPushOptions) else KillRingPushOptions(**opts)
        if options.accumulate and self._ring:
            last = self._ring.pop()
            self._ring.append(text + last if options.prepend else last + text)
            return

        self._ring.append(text)

    def peek(self) -> str | None:
        return self._ring[-1] if self._ring else None

    def rotate(self) -> None:
        if len(self._ring) > 1:
            last = self._ring.pop()
            self._ring.insert(0, last)

    @property
    def length(self) -> int:
        return len(self._ring)

    def __len__(self) -> int:
        return len(self._ring)


__all__ = ["KillRing", "KillRingPushOptions"]
