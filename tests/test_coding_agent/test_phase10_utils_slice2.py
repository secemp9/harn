from __future__ import annotations

import base64
from io import BytesIO

import pytest
import harnify_coding_agent.utils.exif_orientation as exif_orientation_module
import harnify_coding_agent.utils.image_convert as image_convert_module
import harnify_coding_agent.utils.image_resize as image_resize_module
import harnify_coding_agent.utils.harnify_user_agent as harnify_user_agent_module
import harnify_coding_agent.utils.photon as photon_module
import harnify_coding_agent.utils.syntax_highlight as syntax_highlight_module
import harnify_coding_agent.utils.version_check as version_check_module
from harnify_coding_agent.utils.exif_orientation import get_exif_orientation
from harnify_coding_agent.utils.image_convert import convert_to_png
from harnify_coding_agent.utils.harnify_user_agent import get_harnify_user_agent
from harnify_coding_agent.utils.syntax_highlight import highlight, render_highlighted_html, supports_language
from harnify_coding_agent.utils.version_check import (
    check_for_new_harnify_version,
    compare_package_versions,
    get_latest_harnify_release,
    get_latest_harnify_version,
    is_newer_package_version,
)
from PIL import Image


def test_render_highlighted_html_and_highlight() -> None:
    rendered = render_highlighted_html(
        '<span class="hljs-keyword">const</span> value',
        {"keyword": lambda text: f"[keyword:{text}]"},
    )
    assert rendered == "[keyword:const] value"

    decoded = render_highlighted_html("&lt;tag attr=&quot;value&quot;&gt;&amp;#x41;&#65;&lt;/tag&gt;")
    assert decoded == '<tag attr="value">&#x41;A</tag>'

    nested = render_highlighted_html(
        '<span class="hljs-string">a<span class="language-xml">b</span>c</span>',
        {"string": lambda text: f"[string:{text}]"},
    )
    assert nested == "[string:a][string:b][string:c]"

    assert supports_language("typescript") is True
    highlighted = highlight(
        "const value = 1",
        {
            "language": "typescript",
            "ignoreIllegals": True,
            "theme": {
                "keyword": lambda text: f"[keyword:{text}]",
                "number": lambda text: f"[number:{text}]",
            },
        },
    )
    assert "[keyword:const]" in highlighted
    assert "[number:1]" in highlighted

    auto_highlighted = highlight(
        "const value = 1",
        {
            "languageSubset": ["json", "typescript"],
            "theme": {
                "keyword": lambda text: f"[keyword:{text}]",
                "number": lambda text: f"[number:{text}]",
            },
        },
    )
    assert "[keyword:const]" in auto_highlighted


def test_exif_orientation_module_exports_match_ts_surface() -> None:
    assert exif_orientation_module.__all__ == ["applyExifOrientation"]


def test_image_convert_module_exports_match_ts_surface() -> None:
    assert image_convert_module.__all__ == ["convertToPng"]


def test_image_resize_module_exports_match_ts_surface() -> None:
    assert image_resize_module.__all__ == [
        "DEFAULT_MAX_BYTES",
        "ImageResizeOptions",
        "ResizedImage",
        "formatDimensionNote",
        "resizeImage",
    ]


def test_syntax_highlight_module_exports_match_ts_surface() -> None:
    assert syntax_highlight_module.__all__ == [
        "HighlightFormatter",
        "HighlightOptions",
        "HighlightTheme",
        "highlight",
        "renderHighlightedHtml",
        "supportsLanguage",
    ]


def test_harnify_user_agent_module_exports_match_ts_surface() -> None:
    assert harnify_user_agent_module.__all__ == ["getHarnifyUserAgent", "get_harnify_user_agent"]
    user_agent = get_harnify_user_agent("1.2.3")
    assert user_agent.startswith("harnify/1.2.3 (")
    assert "; python/" in user_agent
    assert user_agent.endswith(")")


def test_harnify_user_agent_normalizes_arch_like_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harnify_user_agent_module.platform, "machine", lambda: "x86_64")
    assert get_harnify_user_agent("1.2.3").endswith("; x64)")


def test_version_check_module_exports_match_ts_surface() -> None:
    assert version_check_module.__all__ == [
        "LatestHarnifyRelease",
        "comparePackageVersions",
        "isNewerPackageVersion",
        "getLatestHarnifyRelease",
        "getLatestHarnifyVersion",
        "checkForNewHarnifyVersion",
    ]


@pytest.mark.asyncio
async def test_photon_module_exports_and_load_caches() -> None:
    assert photon_module.__all__ == ["loadPhoton"]

    first = await photon_module.load_photon()
    second = await photon_module.load_photon()

    assert first is not None
    assert first is second


@pytest.mark.asyncio
async def test_version_check_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert compare_package_versions("0.70.6", "0.70.5") > 0
    assert compare_package_versions("0.70.5", "0.70.5") == 0
    assert compare_package_versions("0.70.4", "0.70.5") < 0
    assert is_newer_package_version("0.70.5", "0.70.5") is False
    assert is_newer_package_version("0.70.6", "0.70.5") is True

    calls: list[tuple[str, dict[str, str], int]] = []

    async def fake_fetch(url: str, headers: dict[str, str], timeout_ms: int):
        calls.append((url, headers, timeout_ms))
        return {"packageName": "@new-scope/harnify", "version": "1.2.4", "note": " **Read this** "}

    monkeypatch.setattr("harnify_coding_agent.utils.version_check._fetch_latest_release_json", fake_fetch)

    release = await get_latest_harnify_release("1.2.3")
    assert release is not None
    assert release.packageName == "@new-scope/harnify"
    assert release.version == "1.2.4"
    assert release.note == "**Read this**"
    assert calls[0][0] == "https://harnify.dev/api/latest-version"
    assert calls[0][1]["accept"] == "application/json"
    assert calls[0][1]["User-Agent"].startswith("harnify/1.2.3 ")

    assert await get_latest_harnify_version("1.2.3") == "1.2.4"
    assert await check_for_new_harnify_version("1.2.3") == release
    assert await check_for_new_harnify_version("1.2.4") is None

    monkeypatch.setenv("HARNIFY_SKIP_VERSION_CHECK", "1")
    assert await get_latest_harnify_version("1.2.3") is None
    monkeypatch.delenv("HARNIFY_SKIP_VERSION_CHECK", raising=False)


@pytest.mark.asyncio
async def test_convert_to_png_preserves_png_and_applies_exif_orientation() -> None:
    png_image = Image.new("RGB", (2, 2), (255, 0, 0))
    png_buffer = BytesIO()
    png_image.save(png_buffer, format="PNG")
    png_base64 = base64.b64encode(png_buffer.getvalue()).decode("ascii")

    original = await convert_to_png(png_base64, "image/png")
    assert original is not None
    assert original.data == png_base64
    assert original.mimeType == "image/png"

    jpeg_image = Image.new("RGB", (2, 3), (0, 0, 255))
    exif = Image.Exif()
    exif[274] = 6
    jpeg_buffer = BytesIO()
    jpeg_image.save(jpeg_buffer, format="JPEG", exif=exif)
    jpeg_bytes = jpeg_buffer.getvalue()
    assert get_exif_orientation(jpeg_bytes) == 6

    converted = await convert_to_png(base64.b64encode(jpeg_bytes).decode("ascii"), "image/jpeg")
    assert converted is not None
    assert converted.mimeType == "image/png"
    converted_bytes = base64.b64decode(converted.data)
    assert converted_bytes[:4] == b"\x89PNG"

    with Image.open(BytesIO(converted_bytes)) as normalized:
        assert normalized.size == (3, 2)
