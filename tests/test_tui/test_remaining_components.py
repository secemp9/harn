from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import harnify_tui.components.cancellable_loader as cancellable_loader_module
import harnify_tui.components.image as image_module
import harnify_tui.components.loader as loader_module
from harnify_tui import (
    AbortSignal,
    CancellableLoader,
    Image,
    ImageDimensions,
    ImageOptions,
    ImageTheme,
    Loader,
    LoaderIndicatorOptions,
    SettingItem,
    SettingsList,
    SettingsListOptions,
    SettingsListTheme,
    Spacer,
    getCellDimensions,
    resetCapabilitiesCache,
    setCapabilities,
    setCellDimensions,
)

TEST_SETTINGS_THEME = SettingsListTheme(
    label=lambda text, selected: f"[{text}]" if selected else text,
    value=lambda text, selected: f"<{text}>" if selected else text,
    description=lambda text: text,
    cursor="> ",
    hint=lambda text: text,
)


class FakeUI:
    def __init__(self) -> None:
        self.render_requests = 0

    def requestRender(self) -> None:
        self.render_requests += 1


@contextmanager
def patched_image_env(images: str | None, width_px: int = 9, height_px: int = 18) -> Iterator[None]:
    previous_dims = getCellDimensions()
    try:
        setCapabilities({"images": images, "trueColor": True, "hyperlinks": True})
        setCellDimensions({"widthPx": width_px, "heightPx": height_px})
        yield
    finally:
        resetCapabilitiesCache()
        setCellDimensions(previous_dims)


def test_spacer_renders_requested_blank_lines() -> None:
    spacer = Spacer(2)
    assert spacer.render(10) == ["", ""]

    spacer.setLines(3)
    assert spacer.render(10) == ["", "", ""]


def test_loader_renders_message_and_updates_ui() -> None:
    ui = FakeUI()
    loader = Loader(
        ui,
        spinnerColorFn=lambda text: f"[spin:{text}]",
        messageColorFn=lambda text: f"[msg:{text}]",
        message="Working...",
        indicator=LoaderIndicatorOptions(frames=["*"], intervalMs=1),
    )

    try:
        lines = loader.render(24)
        assert lines[0] == ""
        assert "* [msg:Working...]" in lines[1]
        assert ui.render_requests >= 1

        loader.setMessage("Done")
        assert ui.render_requests >= 2
        assert "[msg:Done]" in loader.render(24)[1]
    finally:
        loader.stop()


def test_loader_module_exports_match_ts_surface() -> None:
    assert loader_module.__all__ == ["Loader", "LoaderIndicatorOptions"]


def test_cancellable_loader_aborts_on_escape() -> None:
    ui = FakeUI()
    loader = CancellableLoader(
        ui,
        spinnerColorFn=lambda text: text,
        messageColorFn=lambda text: text,
        indicator=LoaderIndicatorOptions(frames=["!"]),
    )
    aborted = False

    def on_abort() -> None:
        nonlocal aborted
        aborted = True

    loader.onAbort = on_abort

    try:
        loader.handleInput("\x1b")
        assert isinstance(loader.signal, AbortSignal)
        assert loader.aborted is True
        assert loader.signal.aborted is True
        assert aborted is True
    finally:
        loader.dispose()


def test_cancellable_loader_module_exports_match_ts_surface() -> None:
    assert cancellable_loader_module.__all__ == ["CancellableLoader"]


def test_settings_list_renders_empty_state_and_search_hint() -> None:
    settings = SettingsList(
        [],
        5,
        TEST_SETTINGS_THEME,
        onChange=lambda _id, _value: None,
        onCancel=lambda: None,
        options=SettingsListOptions(enableSearch=True),
    )

    lines = settings.render(60)

    assert any("No settings available" in line for line in lines)
    assert any("Type to search" in line for line in lines)


def test_settings_list_cycles_values_with_enter_and_space() -> None:
    changes: list[tuple[str, str]] = []
    items = [SettingItem(id="theme", label="Theme", currentValue="light", values=["light", "dark"])]
    settings = SettingsList(
        items,
        5,
        TEST_SETTINGS_THEME,
        onChange=lambda item_id, value: changes.append((item_id, value)),
        onCancel=lambda: None,
    )

    settings.handleInput("\r")
    assert items[0].currentValue == "dark"
    assert changes == [("theme", "dark")]

    settings.handleInput(" ")
    assert items[0].currentValue == "light"
    assert changes[-1] == ("theme", "light")


def test_settings_list_filters_search_and_shows_selected_description() -> None:
    items = [
        SettingItem(id="alpha", label="Alpha", currentValue="off"),
        SettingItem(id="beta", label="Beta Mode", currentValue="on", description="Detailed beta description"),
        SettingItem(id="gamma", label="Gamma", currentValue="off"),
    ]
    settings = SettingsList(
        items,
        5,
        TEST_SETTINGS_THEME,
        onChange=lambda _id, _value: None,
        onCancel=lambda: None,
        options=SettingsListOptions(enableSearch=True),
    )

    settings.handleInput("b")
    settings.handleInput("e")
    lines = settings.render(50)
    rendered = "\n".join(lines)

    assert "Beta Mode" in rendered
    assert "Alpha" not in rendered
    assert "Gamma" not in rendered
    assert "Detailed beta description" in rendered


def test_settings_list_opens_and_closes_submenu_with_selection_restore() -> None:
    changes: list[tuple[str, str]] = []
    submenu_inputs: list[str] = []
    received_current_values: list[str] = []

    class DummySubmenu:
        def __init__(self, done: Callable[[str | None], None]) -> None:
            self._done = done
            self.invalidated = False

        def render(self, width: int) -> list[str]:
            return [f"submenu:{width}"]

        def handleInput(self, data: str) -> None:
            submenu_inputs.append(data)
            if data == "\r":
                self._done("manual")

        def invalidate(self) -> None:
            self.invalidated = True

    def make_submenu(current_value: str, done: Callable[[str | None], None]) -> DummySubmenu:
        received_current_values.append(current_value)
        return DummySubmenu(done)

    items = [
        SettingItem(id="theme", label="Theme", currentValue="light", values=["light", "dark"]),
        SettingItem(id="mode", label="Mode", currentValue="auto", submenu=make_submenu),
    ]
    settings = SettingsList(
        items,
        5,
        TEST_SETTINGS_THEME,
        onChange=lambda item_id, value: changes.append((item_id, value)),
        onCancel=lambda: None,
    )

    settings.handleInput("\x1b[B")
    settings.handleInput("\r")

    assert settings.render(20) == ["submenu:20"]
    assert received_current_values == ["auto"]

    settings.handleInput("\r")

    assert submenu_inputs == ["\r"]
    assert items[1].currentValue == "manual"
    assert changes == [("mode", "manual")]
    assert settings.selectedIndex == 1
    assert settings.submenuComponent is None


def test_image_component_uses_fallback_when_terminal_has_no_image_support() -> None:
    image = Image(
        "AAAA",
        "image/png",
        ImageTheme(fallbackColor=lambda text: f"<{text}>"),
        ImageOptions(filename="diagram.png"),
        ImageDimensions(widthPx=20, heightPx=10),
    )

    with patched_image_env(images=None):
        lines = image.render(40)

    assert lines == ["<[Image: diagram.png [image/png] 20x10]>"]


def test_image_component_places_kitty_sequence_on_first_line_with_padding_rows() -> None:
    image = Image(
        "AAAA",
        "image/png",
        ImageTheme(fallbackColor=lambda text: text),
        ImageOptions(maxWidthCells=2),
        ImageDimensions(widthPx=20, heightPx=20),
    )

    with patched_image_env(images="kitty", width_px=10, height_px=10):
        lines = image.render(4)
        image_id = image.getImageId()

    assert isinstance(image_id, int)
    assert lines[0].startswith("\x1b_G")
    assert ",C=1," in lines[0]
    assert f",i={image_id}" in lines[0]
    assert lines[0].endswith("\x1b\\")
    assert lines[1:] == [""]


def test_image_component_uses_ts_nullish_width_and_height_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(image_module, "getCapabilities", lambda: type("Caps", (), {"images": "kitty"})())
    monkeypatch.setattr(image_module, "getCellDimensions", lambda: ImageDimensions(widthPx=10, heightPx=10))
    monkeypatch.setattr(image_module, "allocateImageId", lambda: 7)

    def fake_render_image(base64_data: str, dimensions: ImageDimensions, options: dict[str, object]) -> object:
        captured["base64"] = base64_data
        captured["dimensions"] = dimensions
        captured["options"] = options
        return type("Rendered", (), {"sequence": "\x1b_Gfake\x1b\\", "rows": 1, "imageId": 7})()

    monkeypatch.setattr(image_module, "renderImage", fake_render_image)

    image = Image(
        "AAAA",
        "image/png",
        ImageTheme(fallbackColor=lambda text: text),
        ImageOptions(maxWidthCells=0, maxHeightCells=0),
        ImageDimensions(widthPx=20, heightPx=20),
    )

    assert image.render(4) == ["\x1b_Gfake\x1b\\"]
    assert captured["options"] == {
        "maxWidthCells": 1,
        "maxHeightCells": 0,
        "imageId": 7,
        "moveCursor": False,
    }
