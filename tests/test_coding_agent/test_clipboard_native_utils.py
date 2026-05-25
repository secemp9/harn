from __future__ import annotations

from io import BytesIO

import pytest
import harnify_coding_agent.utils.clipboard_native as clipboard_native_module


class _FakePyperclip:
    def __init__(self) -> None:
        self.copied: list[str] = []

    def copy(self, text: str) -> None:
        self.copied.append(text)


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def save(self, buffer: BytesIO, format: str) -> None:
        assert format == "PNG"
        buffer.write(self.payload)

    def close(self) -> None:
        self.closed = True


class _FakeImageGrab:
    def __init__(self, image) -> None:
        self.image = image

    def grabclipboard(self):
        return self.image


def test_clipboard_native_module_exports_match_ts_surface() -> None:
    assert clipboard_native_module.__all__ == ["clipboard"]


def test_load_clipboard_returns_none_without_display_or_on_termux() -> None:
    fake_pyperclip = _FakePyperclip()
    fake_image_grab = _FakeImageGrab(_FakeImage(b"png"))

    assert (
        clipboard_native_module._load_clipboard(
            env={},
            platform_name="linux",
            pyperclip_module=fake_pyperclip,
            image_grab_module=fake_image_grab,
        )
        is None
    )
    assert (
        clipboard_native_module._load_clipboard(
            env={"TERMUX_VERSION": "1", "DISPLAY": ":0"},
            platform_name="linux",
            pyperclip_module=fake_pyperclip,
            image_grab_module=fake_image_grab,
        )
        is None
    )


@pytest.mark.asyncio
async def test_load_clipboard_builds_native_bridge_when_display_available() -> None:
    fake_pyperclip = _FakePyperclip()
    fake_image = _FakeImage(b"png-bytes")
    fake_image_grab = _FakeImageGrab(fake_image)

    clipboard = clipboard_native_module._load_clipboard(
        env={"DISPLAY": ":0"},
        platform_name="linux",
        pyperclip_module=fake_pyperclip,
        image_grab_module=fake_image_grab,
    )

    assert clipboard is not None
    assert clipboard.has_image() is True
    await clipboard.set_text("hello")
    assert fake_pyperclip.copied == ["hello"]
    assert await clipboard.get_image_binary() == list(b"png-bytes")
    assert fake_image.closed is True
