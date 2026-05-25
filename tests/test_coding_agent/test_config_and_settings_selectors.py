from __future__ import annotations

import importlib
from pathlib import Path

from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.package_manager import ResolvedPaths, ResolvedResource
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.modes.interactive.components.config_selector import ConfigSelectorComponent
from harnify_coding_agent.modes.interactive.components.settings_selector import (
    SettingsCallbacks,
    SettingsConfig,
    SettingsSelectorComponent,
)
from harnify_coding_agent.modes.interactive.theme.theme import init_theme
from harnify_tui import setCapabilities, setKeybindings

config_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.config_selector"
)
settings_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.settings_selector"
)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    init_theme("dark")
    setCapabilities({"images": None, "trueColor": True, "hyperlinks": True})


def test_config_selector_toggles_package_and_top_level_resources(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    agent_dir = tmp_path / "agent"
    package_root = tmp_path / "pkg-root"
    (agent_dir / "extensions").mkdir(parents=True)
    (package_root / "skills" / "demo-skill").mkdir(parents=True)

    settings = SettingsManager.inMemory()
    settings.setProjectPackages(["demo-package"])

    resolved = ResolvedPaths(
        extensions=[
            ResolvedResource(
                path=str(agent_dir / "extensions" / "demo.py"),
                enabled=False,
                metadata={
                    "source": "local",
                    "scope": "user",
                    "origin": "top-level",
                    "baseDir": str(agent_dir),
                },
            )
        ],
        skills=[
            ResolvedResource(
                path=str(package_root / "skills" / "demo-skill" / "SKILL.md"),
                enabled=True,
                metadata={
                    "source": "demo-package",
                    "scope": "project",
                    "origin": "package",
                    "baseDir": str(package_root),
                },
            )
        ],
    )

    render_calls: list[bool] = []
    close_calls: list[bool] = []
    exit_calls: list[bool] = []
    component = ConfigSelectorComponent(
        resolved,
        settings,
        str(cwd),
        str(agent_dir),
        lambda: close_calls.append(True),
        lambda: exit_calls.append(True),
        lambda: render_calls.append(True),
        terminalHeight=24,
    )

    resource_list = component.getResourceList()
    resource_list.handleInput(" ")
    project_packages = settings.getProjectSettings()["packages"]
    assert isinstance(project_packages[0], dict)
    assert project_packages[0]["source"] == "demo-package"
    assert project_packages[0]["skills"] == ["-skills/demo-skill/SKILL.md"]

    resource_list.handleInput("\x1b[B")
    resource_list.handleInput(" ")
    assert settings.getExtensionPaths() == ["+extensions/demo.py"]
    assert render_calls == [True, True]

    resource_list.handleInput("\x03")
    assert close_calls == [True]
    assert exit_calls == []


def test_config_selector_module_exports_match_ts_surface() -> None:
    assert config_selector_module.__all__ == ["ConfigSelectorComponent"]


def test_settings_selector_module_exports_match_ts_surface() -> None:
    assert settings_selector_module.__all__ == [
        "SettingsCallbacks",
        "SettingsConfig",
        "SettingsSelectorComponent",
    ]


def test_settings_selector_supports_theme_preview_and_image_settings() -> None:
    setCapabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})

    auto_compact: list[bool] = []
    selected_themes: list[str] = []
    previewed_themes: list[str] = []

    component = SettingsSelectorComponent(
        SettingsConfig(
            autoCompact=True,
            showImages=True,
            imageWidthCells=80,
            autoResizeImages=True,
            blockImages=False,
            enableSkillCommands=True,
            steeringMode="one-at-a-time",
            followUpMode="one-at-a-time",
            transport="auto",
            httpIdleTimeoutMs=300_000,
            thinkingLevel="medium",
            availableThinkingLevels=["minimal", "medium", "high"],
            currentTheme="dark",
            availableThemes=["dark", "light"],
            hideThinkingBlock=False,
            collapseChangelog=False,
            enableInstallTelemetry=True,
            doubleEscapeAction="tree",
            treeFilterMode="default",
            showHardwareCursor=False,
            editorPaddingX=0,
            autocompleteMaxVisible=5,
            quietStartup=False,
            clearOnShrink=False,
            showTerminalProgress=False,
            warnings={"anthropicExtraUsage": True},
        ),
        SettingsCallbacks(
            onAutoCompactChange=auto_compact.append,
            onShowImagesChange=lambda _value: None,
            onImageWidthCellsChange=lambda _value: None,
            onAutoResizeImagesChange=lambda _value: None,
            onBlockImagesChange=lambda _value: None,
            onEnableSkillCommandsChange=lambda _value: None,
            onSteeringModeChange=lambda _value: None,
            onFollowUpModeChange=lambda _value: None,
            onTransportChange=lambda _value: None,
            onHttpIdleTimeoutMsChange=lambda _value: None,
            onThinkingLevelChange=lambda _value: None,
            onThemeChange=selected_themes.append,
            onThemePreview=previewed_themes.append,
            onHideThinkingBlockChange=lambda _value: None,
            onCollapseChangelogChange=lambda _value: None,
            onEnableInstallTelemetryChange=lambda _value: None,
            onDoubleEscapeActionChange=lambda _value: None,
            onTreeFilterModeChange=lambda _value: None,
            onShowHardwareCursorChange=lambda _value: None,
            onEditorPaddingXChange=lambda _value: None,
            onAutocompleteMaxVisibleChange=lambda _value: None,
            onQuietStartupChange=lambda _value: None,
            onClearOnShrinkChange=lambda _value: None,
            onShowTerminalProgressChange=lambda _value: None,
            onWarningsChange=lambda _value: None,
            onCancel=lambda: None,
        ),
    )

    settings_list = component.getSettingsList()
    item_ids = [item.id for item in settings_list.items]
    assert "show-images" in item_ids
    assert "image-width-cells" in item_ids

    settings_list.handleInput("\r")
    assert auto_compact == [False]

    theme_index = next(index for index, item in enumerate(settings_list.items) if item.id == "theme")
    settings_list.selectedIndex = theme_index
    settings_list.activateItem()
    submenu = settings_list.submenuComponent
    assert submenu is not None

    submenu.handleInput("\x1b[B")
    assert previewed_themes == ["light"]
    submenu.handleInput("\x1b")
    assert previewed_themes[-1] == "dark"
    assert settings_list.submenuComponent is None

    settings_list.activateItem()
    submenu = settings_list.submenuComponent
    assert submenu is not None
    submenu.handleInput("\x1b[B")
    submenu.handleInput("\r")
    assert selected_themes == ["light"]


def test_settings_select_submenu_uses_ts_title_style_order() -> None:
    submenu = settings_selector_module.SelectSubmenu(
        "Theme",
        "Select color theme",
        [settings_selector_module.SelectItem(value="dark", label="dark")],
        "dark",
        lambda _value: None,
        lambda: None,
    )

    title = submenu.children[0]
    assert title.text == settings_selector_module.theme.bold(
        settings_selector_module.theme.fg("accent", "Theme")
    )
