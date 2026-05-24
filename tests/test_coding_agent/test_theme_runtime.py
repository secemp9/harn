from __future__ import annotations

import copy
import importlib
import json
import time
from pathlib import Path

interactive_theme_module = importlib.import_module("harnify_coding_agent.modes.interactive.theme.theme")


def test_load_theme_falls_back_to_256color_when_truecolor_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        interactive_theme_module,
        "getCapabilities",
        lambda: type("Caps", (), {"trueColor": False})(),
    )

    loaded = interactive_theme_module.load_theme("dark")

    assert loaded.getColorMode() == "256color"
    assert loaded.getFgAnsi("accent").startswith("\x1b[38;5;")
    assert loaded.getBgAnsi("userMessageBg").startswith("\x1b[48;5;")


def test_theme_syntax_highlighting_and_language_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        interactive_theme_module,
        "getCapabilities",
        lambda: type("Caps", (), {"trueColor": True})(),
    )
    loaded = interactive_theme_module.load_theme("dark")

    highlighted_lines = interactive_theme_module.highlight_code("const value = 1", "typescript")
    highlighted_text = loaded.syntax_highlight("const value = 1", "typescript")

    assert interactive_theme_module.get_language_from_path("src/example.tsx") == "typescript"
    assert interactive_theme_module.get_language_from_path("Dockerfile") == "dockerfile"
    assert any("\x1b[" in line for line in highlighted_lines)
    assert "\x1b[" in highlighted_text


def test_theme_watcher_reloads_custom_theme_file(monkeypatch, tmp_path: Path) -> None:
    custom_themes_dir = tmp_path / "themes"
    custom_themes_dir.mkdir()
    payload = copy.deepcopy(interactive_theme_module.load_theme_json("dark"))
    payload["name"] = "watch-test"
    payload["colors"]["accent"] = "#112233"
    theme_path = custom_themes_dir / "watch-test.json"
    theme_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        interactive_theme_module,
        "getCapabilities",
        lambda: type("Caps", (), {"trueColor": True})(),
    )
    monkeypatch.setattr(interactive_theme_module, "get_custom_themes_dir", lambda: str(custom_themes_dir))

    changes: list[str] = []
    interactive_theme_module.on_theme_change(lambda: changes.append(interactive_theme_module.theme.getFgAnsi("accent")))
    interactive_theme_module.init_theme("watch-test", True)

    try:
        assert interactive_theme_module.theme.getFgAnsi("accent") == "\x1b[38;2;17;34;51m"

        payload["colors"]["accent"] = "#334455"
        theme_path.write_text(json.dumps(payload), encoding="utf-8")

        deadline = time.time() + 3
        while time.time() < deadline:
            if interactive_theme_module.theme.getFgAnsi("accent") == "\x1b[38;2;51;68;85m":
                break
            time.sleep(0.05)

        assert interactive_theme_module.theme.getFgAnsi("accent") == "\x1b[38;2;51;68;85m"
        assert changes
    finally:
        interactive_theme_module.stop_theme_watcher()
        interactive_theme_module.init_theme("dark")
