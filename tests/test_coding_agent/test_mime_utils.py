from __future__ import annotations

import pytest

import harnify_coding_agent.utils.mime as mime_module
from harnify_coding_agent.utils.mime import (
    detect_supported_image_mime_type,
    detect_supported_image_mime_type_from_file,
)


def test_mime_module_exports_match_ts_surface() -> None:
    assert mime_module.__all__ == [
        "detectSupportedImageMimeType",
        "detectSupportedImageMimeTypeFromFile",
    ]


def test_detect_supported_image_mime_type_matches_ts_cases() -> None:
    assert detect_supported_image_mime_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert detect_supported_image_mime_type(b"\xff\xd8\xff\xf7rest") is None

    png = b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + b"\x00" * 20
    assert detect_supported_image_mime_type(png) == "image/png"

    apng = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + b"\x00" * 17
        + (0).to_bytes(4, "big")
        + b"acTL"
        + b"\x00" * 4
    )
    assert detect_supported_image_mime_type(apng) is None

    assert detect_supported_image_mime_type(b"GIF89a") == "image/gif"
    assert detect_supported_image_mime_type(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"
    assert detect_supported_image_mime_type(b"not-an-image") is None


@pytest.mark.asyncio
async def test_detect_supported_image_mime_type_from_file(tmp_path) -> None:
    path = tmp_path / "tiny.gif"
    path.write_bytes(b"GIF89a")

    assert await detect_supported_image_mime_type_from_file(str(path)) == "image/gif"
