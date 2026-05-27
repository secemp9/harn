from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from harnify_coding_agent.utils import clipboard_image as clipboard_image_utils
from PIL import Image


class _NativeClipboard:
    def __init__(self, *, has_image: bool, data: bytes | None) -> None:
        self._has_image = has_image
        self._data = data

    def has_image(self) -> bool:
        return self._has_image

    async def get_image_binary(self) -> bytes | None:
        return self._data


def _spawn_ok(stdout: bytes) -> clipboard_image_utils._CommandResult:
    return clipboard_image_utils._CommandResult(stdout=stdout, ok=True)


def _spawn_error() -> clipboard_image_utils._CommandResult:
    return clipboard_image_utils._CommandResult(stdout=b"", ok=False)


def _create_tiny_bmp_1x1_red_24bpp() -> bytes:
    buffer = bytearray(58)
    buffer[0:2] = b"BM"
    _write_u32_le(buffer, 2, len(buffer))
    _write_u32_le(buffer, 10, 54)
    _write_u32_le(buffer, 14, 40)
    _write_i32_le(buffer, 18, 1)
    _write_i32_le(buffer, 22, 1)
    _write_u16_le(buffer, 26, 1)
    _write_u16_le(buffer, 28, 24)
    _write_u32_le(buffer, 34, 4)
    buffer[54] = 0x00
    buffer[55] = 0x00
    buffer[56] = 0xFF
    buffer[57] = 0x00
    return bytes(buffer)


def _write_u16_le(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 2] = value.to_bytes(2, "little", signed=False)


def _write_u32_le(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 4] = value.to_bytes(4, "little", signed=False)


def _write_i32_le(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 4] = value.to_bytes(4, "little", signed=True)


def test_clipboard_image_module_exports_match_ts_surface() -> None:
    assert clipboard_image_utils.__all__ == [
        "ClipboardImage",
        "extensionForImageMimeType",
        "isWaylandSession",
        "readClipboardImage",
    ]


@pytest.mark.asyncio
async def test_read_clipboard_image_wayland_uses_wl_paste(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: str, args: list[str], **_kwargs):
        if command == "wl-paste" and args[0] == "--list-types":
            return _spawn_ok(b"text/plain\nimage/png\n")
        if command == "wl-paste" and args[0] == "--type":
            return _spawn_ok(bytes([1, 2, 3]))
        raise AssertionError(f"unexpected run_command call: {command} {args}")

    def unexpected_native():
        raise AssertionError("native clipboard should not be used on Wayland")

    monkeypatch.setattr(clipboard_image_utils, "run_command", fake_run)
    monkeypatch.setattr(clipboard_image_utils, "_get_native_clipboard", unexpected_native)

    result = await clipboard_image_utils.read_clipboard_image({"platform": "linux", "env": {"WAYLAND_DISPLAY": "1"}})
    assert result is not None
    assert result.mimeType == "image/png"
    assert list(result.bytes) == [1, 2, 3]


@pytest.mark.asyncio
async def test_read_clipboard_image_wayland_falls_back_to_xclip(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: str, args: list[str], **_kwargs):
        if command == "wl-paste":
            return _spawn_error()
        if command == "xclip" and "TARGETS" in args:
            return _spawn_ok(b"image/png\n")
        if command == "xclip" and "image/png" in args:
            return _spawn_ok(bytes([9, 8]))
        return _spawn_ok(b"")

    monkeypatch.setattr(clipboard_image_utils, "run_command", fake_run)

    result = await clipboard_image_utils.read_clipboard_image(
        {"platform": "linux", "env": {"XDG_SESSION_TYPE": "wayland"}}
    )
    assert result is not None
    assert result.mimeType == "image/png"
    assert list(result.bytes) == [9, 8]


@pytest.mark.asyncio
async def test_read_clipboard_image_wsl_uses_powershell_path_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmp_file: Path | None = None

    def fake_tmpdir() -> str:
        return str(tmp_path)

    def fake_uuid4() -> str:
        return "clip-uuid"

    def fake_run(command: str, args: list[str], **kwargs):
        nonlocal tmp_file
        if command in {"wl-paste", "xclip"}:
            return _spawn_ok(b"")
        if command == "wslpath":
            tmp_file = Path(args[1])
            return _spawn_ok(b"C:\\Users\\O'Hare\\clip.png\n")
        if command == "powershell.exe":
            assert kwargs.get("env", {}).get("HARNIFY_WSL_CLIPBOARD_IMAGE_PATH") is None
            assert "$path = 'C:\\Users\\O''Hare\\clip.png'" in args[2]
            assert tmp_file is not None
            tmp_file.write_bytes(bytes([4, 5, 6]))
            return _spawn_ok(b"ok\n")
        raise AssertionError(f"unexpected run_command call: {command} {args}")

    monkeypatch.setattr(clipboard_image_utils, "run_command", fake_run)
    monkeypatch.setattr(clipboard_image_utils.tempfile, "gettempdir", fake_tmpdir)
    monkeypatch.setattr(clipboard_image_utils, "uuid4", fake_uuid4)

    result = await clipboard_image_utils.read_clipboard_image(
        {"platform": "linux", "env": {"WSL_DISTRO_NAME": "Ubuntu"}}
    )
    assert result is not None
    assert result.mimeType == "image/png"
    assert list(result.bytes) == [4, 5, 6]


@pytest.mark.asyncio
async def test_read_clipboard_image_non_wayland_uses_native_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clipboard_image_utils,
        "_get_native_clipboard",
        lambda: _NativeClipboard(has_image=True, data=bytes([7])),
    )

    result = await clipboard_image_utils.read_clipboard_image({"platform": "linux", "env": {}})
    assert result is not None
    assert result.mimeType == "image/png"
    assert list(result.bytes) == [7]


@pytest.mark.asyncio
async def test_read_clipboard_image_non_wayland_returns_none_when_no_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clipboard_image_utils,
        "_get_native_clipboard",
        lambda: _NativeClipboard(has_image=False, data=None),
    )
    result = await clipboard_image_utils.read_clipboard_image({"platform": "linux", "env": {}})
    assert result is None


@pytest.mark.asyncio
async def test_read_clipboard_image_converts_bmp_to_png(monkeypatch: pytest.MonkeyPatch) -> None:
    bmp_bytes = _create_tiny_bmp_1x1_red_24bpp()

    def fake_run(command: str, args: list[str], **_kwargs):
        if command == "wl-paste" and "--list-types" in args:
            return _spawn_ok(b"image/bmp\n")
        if command == "wl-paste" and "image/bmp" in args:
            return _spawn_ok(bmp_bytes)
        return _spawn_error()

    monkeypatch.setattr(clipboard_image_utils, "run_command", fake_run)

    image = await clipboard_image_utils.read_clipboard_image(
        {"platform": "linux", "env": {"WAYLAND_DISPLAY": "wayland-0"}}
    )
    assert image is not None
    assert image.mimeType == "image/png"
    assert image.bytes[:4] == b"\x89PNG"

    with Image.open(BytesIO(image.bytes)) as rendered:
        assert rendered.size == (1, 1)
