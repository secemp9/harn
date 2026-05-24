"""Interactive theme helpers used by coding-agent selectors and HTML export."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, TypedDict, cast

from harnify_tui import MarkdownTheme, SelectListTheme, SettingsListTheme, getCapabilities

from harnify_coding_agent.config import get_custom_themes_dir, get_themes_dir
from harnify_coding_agent.utils.syntax_highlight import highlight, supports_language

type ColorValue = str | int
type ThemeColor = str
type ThemeBg = Literal[
    "selectedBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
]

_BACKGROUND_COLOR_KEYS: set[str] = {
    "selectedBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
}
_ANSI_BASE_COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#c0c0c0",
    "#808080",
    "#ff0000",
    "#00ff00",
    "#ffff00",
    "#0000ff",
    "#ff00ff",
    "#00ffff",
    "#ffffff",
]


class ThemeExportSection(TypedDict, total=False):
    pageBg: ColorValue
    cardBg: ColorValue
    infoBg: ColorValue


class ThemeJson(TypedDict, total=False):
    name: str
    vars: dict[str, ColorValue]
    colors: dict[str, ColorValue]
    export: ThemeExportSection


class ThemeExportColors(TypedDict, total=False):
    pageBg: str
    cardBg: str
    infoBg: str


class ThemeInfo(TypedDict):
    name: str
    path: str | None


@dataclass(slots=True)
class _RegisteredThemeRecord:
    path: str | None = None
    data: ThemeJson | None = None
    sourceInfo: Any | None = None


class Theme:
    def __init__(
        self,
        fgColors: dict[ThemeColor, ColorValue],
        bgColors: dict[ThemeBg, ColorValue],
        *,
        name: str | None = None,
        sourcePath: str | None = None,
        sourceInfo: Any | None = None,
        colorMode: str = "truecolor",
    ) -> None:
        self.name = name
        self.sourcePath = sourcePath
        self.sourceInfo = sourceInfo
        self._colorMode = colorMode
        self._fgColors = {key: _fg_ansi(value, colorMode) for key, value in fgColors.items()}
        self._bgColors = {key: _bg_ansi(value, colorMode) for key, value in bgColors.items()}

    def fg(self, color: ThemeColor, text: str) -> str:
        ansi = self._fgColors.get(color)
        if ansi is None:
            raise KeyError(f"Unknown theme color: {color}")
        return f"{ansi}{text}\x1b[39m"

    def bg(self, color: ThemeBg, text: str) -> str:
        ansi = self._bgColors.get(color)
        if ansi is None:
            raise KeyError(f"Unknown theme background color: {color}")
        return f"{ansi}{text}\x1b[49m"

    def bold(self, text: str) -> str:
        return f"\x1b[1m{text}\x1b[22m"

    def italic(self, text: str) -> str:
        return f"\x1b[3m{text}\x1b[23m"

    def underline(self, text: str) -> str:
        return f"\x1b[4m{text}\x1b[24m"

    def inverse(self, text: str) -> str:
        return f"\x1b[7m{text}\x1b[27m"

    def strikethrough(self, text: str) -> str:
        return f"\x1b[9m{text}\x1b[29m"

    def getFgAnsi(self, color: ThemeColor) -> str:
        ansi = self._fgColors.get(color)
        if ansi is None:
            raise KeyError(f"Unknown theme color: {color}")
        return ansi

    def getBgAnsi(self, color: ThemeBg) -> str:
        ansi = self._bgColors.get(color)
        if ansi is None:
            raise KeyError(f"Unknown theme background color: {color}")
        return ansi

    def getColorMode(self) -> str:
        return self._colorMode

    def getThinkingBorderColor(self, level: str) -> Any:
        mapping = {
            "off": "thinkingOff",
            "minimal": "thinkingMinimal",
            "low": "thinkingLow",
            "medium": "thinkingMedium",
            "high": "thinkingHigh",
            "xhigh": "thinkingXhigh",
        }
        color = mapping.get(level, "thinkingOff")
        return lambda text: self.fg(color, text)

    def getBashModeBorderColor(self) -> Any:
        return lambda text: self.fg("bashMode", text)

    def syntax_highlight(self, code: str, language: str | None = None) -> str:
        return "\n".join(_highlight_code_with_theme(self, code, language))


_BUILTIN_THEMES: dict[str, ThemeJson] | None = None
_REGISTERED_THEMES: dict[str, _RegisteredThemeRecord] = {}
_CURRENT_THEME_NAME: str | None = None
_ON_THEME_CHANGE: Any | None = None
_THEME_WATCHER_THREAD: threading.Thread | None = None
_THEME_WATCHER_STOP: threading.Event | None = None
_CACHED_HIGHLIGHT_THEME_FOR: Theme | None = None
_CACHED_CLI_HIGHLIGHT_THEME: dict[str, Callable[[str], str]] | None = None


def _load_json(path: str) -> ThemeJson:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Theme file must contain a JSON object: {path}")
    return payload  # type: ignore[return-value]


def _color256_to_hex(index: int) -> str:
    if index < 16:
        return _ANSI_BASE_COLORS[index]
    if index < 232:
        cube_index = index - 16
        red = cube_index // 36
        green = (cube_index % 36) // 6
        blue = cube_index % 6

        def to_component(value: int) -> int:
            return 0 if value == 0 else 55 + value * 40

        return f"#{to_component(red):02x}{to_component(green):02x}{to_component(blue):02x}"

    gray = 8 + (index - 232) * 10
    return f"#{gray:02x}{gray:02x}{gray:02x}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {color}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


_CUBE_VALUES = (0, 95, 135, 175, 215, 255)
_GRAY_VALUES = tuple(8 + index * 10 for index in range(24))


def _find_closest_index(value: int, candidates: tuple[int, ...]) -> int:
    min_distance = float("inf")
    min_index = 0
    for index, candidate in enumerate(candidates):
        distance = abs(value - candidate)
        if distance < min_distance:
            min_distance = distance
            min_index = index
    return min_index


def _color_distance(r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> float:
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    return dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114


def _rgb_to_256(red: int, green: int, blue: int) -> int:
    red_index = _find_closest_index(red, _CUBE_VALUES)
    green_index = _find_closest_index(green, _CUBE_VALUES)
    blue_index = _find_closest_index(blue, _CUBE_VALUES)
    cube_red = _CUBE_VALUES[red_index]
    cube_green = _CUBE_VALUES[green_index]
    cube_blue = _CUBE_VALUES[blue_index]
    cube_value = 16 + 36 * red_index + 6 * green_index + blue_index
    cube_distance = _color_distance(red, green, blue, cube_red, cube_green, cube_blue)

    gray = round(0.299 * red + 0.587 * green + 0.114 * blue)
    gray_index = _find_closest_index(gray, _GRAY_VALUES)
    gray_value = _GRAY_VALUES[gray_index]
    gray_distance = _color_distance(red, green, blue, gray_value, gray_value, gray_value)

    spread = max(red, green, blue) - min(red, green, blue)
    if spread < 10 and gray_distance < cube_distance:
        return 232 + gray_index
    return cube_value


def _hex_to_256(color: str) -> int:
    red, green, blue = _hex_to_rgb(color)
    return _rgb_to_256(red, green, blue)


def _fg_ansi(color: ColorValue, mode: str = "truecolor") -> str:
    if isinstance(color, int):
        return f"\x1b[38;5;{color}m"
    if color == "":
        return "\x1b[39m"
    if color.startswith("#"):
        if mode == "256color":
            return f"\x1b[38;5;{_hex_to_256(color)}m"
        red, green, blue = _hex_to_rgb(color)
        return f"\x1b[38;2;{red};{green};{blue}m"
    raise ValueError(f"Invalid theme color value: {color}")


def _bg_ansi(color: ColorValue, mode: str = "truecolor") -> str:
    if isinstance(color, int):
        return f"\x1b[48;5;{color}m"
    if color == "":
        return "\x1b[49m"
    if color.startswith("#"):
        if mode == "256color":
            return f"\x1b[48;5;{_hex_to_256(color)}m"
        red, green, blue = _hex_to_rgb(color)
        return f"\x1b[48;2;{red};{green};{blue}m"
    raise ValueError(f"Invalid theme background value: {color}")


def _detect_color_mode() -> Literal["truecolor", "256color"]:
    capabilities = getCapabilities()
    return "truecolor" if getattr(capabilities, "trueColor", False) else "256color"


def get_builtin_themes() -> dict[str, ThemeJson]:
    global _BUILTIN_THEMES
    if _BUILTIN_THEMES is None:
        themes_dir = Path(get_themes_dir())
        _BUILTIN_THEMES = {
            "dark": _load_json(str(themes_dir / "dark.json")),
            "light": _load_json(str(themes_dir / "light.json")),
        }
    return _BUILTIN_THEMES


def _ansi_luminance(index: int) -> float:
    color = _color256_to_hex(index).lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)

    def to_linear(component: int) -> float:
        scaled = component / 255.0
        return scaled / 12.92 if scaled <= 0.03928 else ((scaled + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(red) + 0.7152 * to_linear(green) + 0.0722 * to_linear(blue)


def get_default_theme() -> str:
    colorfgbg = os.environ.get("COLORFGBG")
    if colorfgbg:
        try:
            background = int(colorfgbg.split(";")[-1])
            return "light" if _ansi_luminance(background) >= 0.5 else "dark"
        except (TypeError, ValueError):
            pass
    return "dark"


def set_registered_themes(themes: list[Any]) -> None:
    _REGISTERED_THEMES.clear()
    for item in themes:
        if isinstance(item, dict):
            name = item.get("name")
            source_path = item.get("sourcePath")
            data = item.get("data")
            source_info = item.get("sourceInfo")
        else:
            name = getattr(item, "name", None)
            source_path = getattr(item, "sourcePath", None)
            data = getattr(item, "data", None)
            source_info = getattr(item, "sourceInfo", None)
        if isinstance(name, str) and name:
            _REGISTERED_THEMES[name] = _RegisteredThemeRecord(
                path=str(source_path) if source_path else None,
                data=data if isinstance(data, dict) else None,
                sourceInfo=source_info,
            )


def set_current_theme_name(theme_name: str | None) -> None:
    global _CURRENT_THEME_NAME
    _CURRENT_THEME_NAME = theme_name


def _resolve_var_refs(
    value: ColorValue | None,
    variables: dict[str, ColorValue],
    visited: set[str] | None = None,
) -> ColorValue | None:
    if value is None or isinstance(value, int):
        return value
    if not isinstance(value, str):
        return str(value)
    if value not in variables:
        return value
    trail = set(visited or set())
    if value in trail:
        raise ValueError(f"Cyclic theme variable reference: {value}")
    trail.add(value)
    return _resolve_var_refs(variables[value], variables, trail)


def resolve_theme_colors(
    colors: dict[str, ColorValue] | None,
    variables: dict[str, ColorValue] | None = None,
) -> dict[str, ColorValue]:
    resolved: dict[str, ColorValue] = {}
    for key, value in (colors or {}).items():
        resolved_value = _resolve_var_refs(value, variables or {})
        if resolved_value is not None:
            resolved[key] = resolved_value
    return resolved


def load_theme_json(name: str) -> ThemeJson:
    builtin_themes = get_builtin_themes()
    if name in builtin_themes:
        return builtin_themes[name]

    registered = _REGISTERED_THEMES.get(name)
    if registered is not None:
        if registered.data is not None:
            return dict(registered.data)
        if registered.path:
            return _load_json(registered.path)
        raise ValueError(f'Theme "{name}" does not have a source path for export')

    custom_path = Path(get_custom_themes_dir()) / f"{name}.json"
    if custom_path.exists():
        return _load_json(str(custom_path))
    raise FileNotFoundError(f"Theme not found: {name}")


def get_available_themes() -> list[str]:
    custom_dir = Path(get_custom_themes_dir())
    names = set(get_builtin_themes())
    if custom_dir.exists():
        names.update(path.stem for path in custom_dir.glob("*.json"))
    names.update(_REGISTERED_THEMES)
    return sorted(names)


def get_available_themes_with_paths() -> list[ThemeInfo]:
    result: list[ThemeInfo] = []
    seen: set[str] = set()

    def add(name: str, path: str | None) -> None:
        if name in seen:
            return
        seen.add(name)
        result.append({"name": name, "path": path})

    themes_dir = Path(get_themes_dir())
    for name in get_builtin_themes():
        add(name, str(themes_dir / f"{name}.json"))

    custom_dir = Path(get_custom_themes_dir())
    if custom_dir.exists():
        for path in sorted(custom_dir.glob("*.json")):
            try:
                payload = _load_json(str(path))
            except Exception:
                continue
            theme_name = payload.get("name") if isinstance(payload.get("name"), str) else path.stem
            add(theme_name, str(path))

    for name, record in sorted(_REGISTERED_THEMES.items()):
        add(name, record.path)

    return result


def load_theme(name: str) -> Theme:
    theme_json = load_theme_json(name)
    resolved = resolve_theme_colors(theme_json.get("colors"), theme_json.get("vars"))
    fg_colors = {
        key: value
        for key, value in resolved.items()
        if key not in _BACKGROUND_COLOR_KEYS
    }
    bg_colors = {
        cast(ThemeBg, key): value
        for key, value in resolved.items()
        if key in _BACKGROUND_COLOR_KEYS
    }
    registered = _REGISTERED_THEMES.get(name)
    source_path: str | None = None
    if registered is not None:
        source_path = registered.path
    else:
        custom_path = Path(get_custom_themes_dir()) / f"{name}.json"
        if custom_path.exists():
            source_path = str(custom_path)
        elif name in get_builtin_themes():
            source_path = str(Path(get_themes_dir()) / f"{name}.json")
    return Theme(
        fg_colors,
        bg_colors,
        name=name,
        sourcePath=source_path,
        sourceInfo=registered.sourceInfo if registered is not None else None,
        colorMode=_detect_color_mode(),
    )


def set_global_theme(theme_instance: Theme) -> None:
    global theme
    theme = theme_instance


def init_theme(theme_name: str | None = None, enableWatcher: bool = False) -> None:
    stop_theme_watcher()
    name = theme_name or get_default_theme()
    global _CURRENT_THEME_NAME
    _CURRENT_THEME_NAME = name
    try:
        set_global_theme(load_theme(name))
        if enableWatcher:
            _start_theme_watcher()
    except Exception:
        _CURRENT_THEME_NAME = "dark"
        set_global_theme(load_theme("dark"))


def set_theme(name: str, enableWatcher: bool = False) -> dict[str, Any]:
    stop_theme_watcher()
    global _CURRENT_THEME_NAME
    _CURRENT_THEME_NAME = name
    try:
        set_global_theme(load_theme(name))
        if enableWatcher:
            _start_theme_watcher()
        if callable(_ON_THEME_CHANGE):
            _ON_THEME_CHANGE()
        return {"success": True}
    except Exception as error:
        _CURRENT_THEME_NAME = "dark"
        set_global_theme(load_theme("dark"))
        return {"success": False, "error": str(error)}


def set_theme_instance(theme_instance: Theme) -> None:
    global _CURRENT_THEME_NAME
    set_global_theme(theme_instance)
    _CURRENT_THEME_NAME = theme_instance.name or "<in-memory>"
    stop_theme_watcher()
    if callable(_ON_THEME_CHANGE):
        _ON_THEME_CHANGE()


def on_theme_change(callback: Any) -> None:
    global _ON_THEME_CHANGE
    _ON_THEME_CHANGE = callback


def stop_theme_watcher() -> None:
    global _THEME_WATCHER_STOP, _THEME_WATCHER_THREAD
    stop_event = _THEME_WATCHER_STOP
    watcher_thread = _THEME_WATCHER_THREAD
    _THEME_WATCHER_STOP = None
    _THEME_WATCHER_THREAD = None
    if stop_event is not None:
        stop_event.set()
    if watcher_thread is not None and watcher_thread.is_alive() and watcher_thread is not threading.current_thread():
        watcher_thread.join(timeout=0.5)


def get_theme_by_name(name: str | None = None) -> Theme:
    return load_theme(name or _CURRENT_THEME_NAME or get_default_theme())


def get_resolved_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    name = theme_name or _CURRENT_THEME_NAME or get_default_theme()
    theme_json = load_theme_json(name)
    resolved = resolve_theme_colors(theme_json.get("colors"), theme_json.get("vars"))
    default_text = "#000000" if name == "light" else "#e5e5e7"

    css_colors: dict[str, str] = {}
    for key, value in resolved.items():
        if isinstance(value, int):
            css_colors[key] = _color256_to_hex(value)
        elif value == "":
            css_colors[key] = default_text
        else:
            css_colors[key] = str(value)
    return css_colors


def get_theme_export_colors(theme_name: str | None = None) -> ThemeExportColors:
    name = theme_name or _CURRENT_THEME_NAME or get_default_theme()
    try:
        theme_json = load_theme_json(name)
        export_section = theme_json.get("export") or {}
        variables = theme_json.get("vars") or {}

        def resolve(value: ColorValue | None) -> str | None:
            resolved = _resolve_var_refs(value, variables)
            if resolved is None or resolved == "":
                return None
            if isinstance(resolved, int):
                return _color256_to_hex(resolved)
            return str(resolved)

        result: ThemeExportColors = {}
        for key in ("pageBg", "cardBg", "infoBg"):
            resolved_value = resolve(export_section.get(key))  # type: ignore[arg-type]
            if resolved_value is not None:
                result[key] = resolved_value  # type: ignore[literal-required]
        return result
    except Exception:
        return {}


def get_markdown_theme() -> MarkdownTheme:
    return MarkdownTheme(
        heading=lambda text: theme.fg("mdHeading", text),
        link=lambda text: theme.fg("mdLink", text),
        linkUrl=lambda text: theme.fg("mdLinkUrl", text),
        code=lambda text: theme.fg("mdCode", text),
        codeBlock=lambda text: theme.fg("mdCodeBlock", text),
        codeBlockBorder=lambda text: theme.fg("mdCodeBlockBorder", text),
        quote=lambda text: theme.fg("mdQuote", text),
        quoteBorder=lambda text: theme.fg("mdQuoteBorder", text),
        hr=lambda text: theme.fg("mdHr", text),
        listBullet=lambda text: theme.fg("mdListBullet", text),
        bold=theme.bold,
        italic=theme.italic,
        strikethrough=theme.strikethrough,
        underline=theme.underline,
        highlightCode=highlight_code,
    )


def get_select_list_theme() -> SelectListTheme:
    return SelectListTheme(
        selectedPrefix=lambda text: theme.fg("accent", text),
        selectedText=lambda text: theme.fg("accent", text),
        description=lambda text: theme.fg("muted", text),
        scrollInfo=lambda text: theme.fg("muted", text),
        noMatch=lambda text: theme.fg("muted", text),
    )


def get_settings_list_theme() -> SettingsListTheme:
    return SettingsListTheme(
        label=lambda text, selected: theme.fg("accent", text) if selected else text,
        value=lambda text, selected: theme.fg("accent", text) if selected else theme.fg("muted", text),
        description=lambda text: theme.fg("dim", text),
        cursor=theme.fg("accent", "-> " if sys.platform == "win32" else "→ "),
        hint=lambda text: theme.fg("dim", text),
    )


def get_editor_theme() -> Any:
    return SimpleNamespace(
        borderColor=lambda text: theme.fg("borderMuted", text),
        selectList=get_select_list_theme(),
    )


def _build_cli_highlight_theme(theme_instance: Theme) -> dict[str, Callable[[str], str]]:
    return {
        "keyword": lambda text: theme_instance.fg("syntaxKeyword", text),
        "built_in": lambda text: theme_instance.fg("syntaxType", text),
        "literal": lambda text: theme_instance.fg("syntaxNumber", text),
        "number": lambda text: theme_instance.fg("syntaxNumber", text),
        "string": lambda text: theme_instance.fg("syntaxString", text),
        "comment": lambda text: theme_instance.fg("syntaxComment", text),
        "function": lambda text: theme_instance.fg("syntaxFunction", text),
        "title": lambda text: theme_instance.fg("syntaxFunction", text),
        "class": lambda text: theme_instance.fg("syntaxType", text),
        "type": lambda text: theme_instance.fg("syntaxType", text),
        "attr": lambda text: theme_instance.fg("syntaxVariable", text),
        "variable": lambda text: theme_instance.fg("syntaxVariable", text),
        "params": lambda text: theme_instance.fg("syntaxVariable", text),
        "operator": lambda text: theme_instance.fg("syntaxOperator", text),
        "punctuation": lambda text: theme_instance.fg("syntaxPunctuation", text),
    }


def _get_cli_highlight_theme(theme_instance: Theme) -> dict[str, Callable[[str], str]]:
    global _CACHED_HIGHLIGHT_THEME_FOR, _CACHED_CLI_HIGHLIGHT_THEME
    if _CACHED_HIGHLIGHT_THEME_FOR is not theme_instance or _CACHED_CLI_HIGHLIGHT_THEME is None:
        _CACHED_HIGHLIGHT_THEME_FOR = theme_instance
        _CACHED_CLI_HIGHLIGHT_THEME = _build_cli_highlight_theme(theme_instance)
    return _CACHED_CLI_HIGHLIGHT_THEME


def _highlight_code_with_theme(theme_instance: Theme, code: str, language: str | None = None) -> list[str]:
    valid_language = language if language and supports_language(language) else None
    if not valid_language:
        return [theme_instance.fg("mdCodeBlock", line) for line in code.split("\n")]
    try:
        return highlight(
            code,
            {
                "language": valid_language,
                "ignoreIllegals": True,
                "theme": _get_cli_highlight_theme(theme_instance),
            },
        ).split("\n")
    except Exception:
        return code.split("\n")


def highlight_code(code: str, lang: str | None = None) -> list[str]:
    return _highlight_code_with_theme(theme, code, lang)


def get_language_from_path(file_path: str) -> str | None:
    suffix = Path(file_path).suffix.lower().lstrip(".")
    if not suffix:
        basename = Path(file_path).name.lower()
        if basename == "dockerfile":
            return "dockerfile"
        if basename == "makefile":
            return "makefile"
        return None

    return {
        "ts": "typescript",
        "tsx": "typescript",
        "js": "javascript",
        "jsx": "javascript",
        "mjs": "javascript",
        "cjs": "javascript",
        "py": "python",
        "rb": "ruby",
        "rs": "rust",
        "go": "go",
        "java": "java",
        "kt": "kotlin",
        "swift": "swift",
        "c": "c",
        "h": "c",
        "cpp": "cpp",
        "cc": "cpp",
        "cxx": "cpp",
        "hpp": "cpp",
        "cs": "csharp",
        "php": "php",
        "sh": "bash",
        "bash": "bash",
        "zsh": "bash",
        "fish": "fish",
        "ps1": "powershell",
        "sql": "sql",
        "html": "html",
        "htm": "html",
        "css": "css",
        "scss": "scss",
        "sass": "sass",
        "less": "less",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
        "xml": "xml",
        "md": "markdown",
        "markdown": "markdown",
        "cmake": "cmake",
        "lua": "lua",
        "perl": "perl",
        "r": "r",
        "scala": "scala",
        "clj": "clojure",
        "ex": "elixir",
        "exs": "elixir",
        "erl": "erlang",
        "hs": "haskell",
        "ml": "ocaml",
        "vim": "vim",
        "graphql": "graphql",
        "proto": "protobuf",
        "tf": "hcl",
        "hcl": "hcl",
    }.get(suffix)


def _theme_file_signature(path: str) -> tuple[int, int] | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _start_theme_watcher() -> None:
    global _THEME_WATCHER_STOP, _THEME_WATCHER_THREAD
    stop_theme_watcher()
    watched_theme_name = _CURRENT_THEME_NAME
    if not watched_theme_name or watched_theme_name in {"dark", "light"}:
        return
    current_source_path = getattr(theme, "sourcePath", None)
    if not isinstance(current_source_path, str) or not current_source_path:
        return

    watched_path = current_source_path
    stop_event = threading.Event()
    last_signature = _theme_file_signature(watched_path)

    def watch_loop() -> None:
        nonlocal last_signature
        while not stop_event.wait(0.1):
            if _CURRENT_THEME_NAME != watched_theme_name:
                return
            next_signature = _theme_file_signature(watched_path)
            if next_signature == last_signature:
                continue
            last_signature = next_signature
            if next_signature is None:
                continue
            try:
                reloaded = load_theme(watched_theme_name)
            except Exception:
                continue
            set_global_theme(reloaded)
            if callable(_ON_THEME_CHANGE):
                _ON_THEME_CHANGE()

    _THEME_WATCHER_STOP = stop_event
    _THEME_WATCHER_THREAD = threading.Thread(
        target=watch_loop,
        name="harnify-theme-watcher",
        daemon=True,
    )
    _THEME_WATCHER_THREAD.start()


try:
    theme = load_theme(get_default_theme())
    _CURRENT_THEME_NAME = theme.name
except Exception:
    theme = Theme({}, {}, name="uninitialized")


getAvailableThemes = get_available_themes
getAvailableThemesWithPaths = get_available_themes_with_paths
getBuiltinThemes = get_builtin_themes
getDefaultTheme = get_default_theme
getEditorTheme = get_editor_theme
getMarkdownTheme = get_markdown_theme
getResolvedThemeColors = get_resolved_theme_colors
getSelectListTheme = get_select_list_theme
getSettingsListTheme = get_settings_list_theme
getThemeByName = get_theme_by_name
getThemeExportColors = get_theme_export_colors
initTheme = init_theme
loadTheme = load_theme
loadThemeJson = load_theme_json
onThemeChange = on_theme_change
resolveThemeColors = resolve_theme_colors
setCurrentThemeName = set_current_theme_name
setGlobalTheme = set_global_theme
setRegisteredThemes = set_registered_themes
setTheme = set_theme
setThemeInstance = set_theme_instance
stopThemeWatcher = stop_theme_watcher
highlightCode = highlight_code
getLanguageFromPath = get_language_from_path

__all__ = [
    "Theme",
    "ThemeBg",
    "ThemeColor",
    "ThemeExportColors",
    "ThemeExportSection",
    "ThemeInfo",
    "ThemeJson",
    "getAvailableThemes",
    "getAvailableThemesWithPaths",
    "getBuiltinThemes",
    "getDefaultTheme",
    "getEditorTheme",
    "getLanguageFromPath",
    "getMarkdownTheme",
    "getResolvedThemeColors",
    "getSelectListTheme",
    "getSettingsListTheme",
    "getThemeByName",
    "getThemeExportColors",
    "get_available_themes",
    "get_available_themes_with_paths",
    "get_builtin_themes",
    "get_default_theme",
    "get_editor_theme",
    "get_language_from_path",
    "get_markdown_theme",
    "get_resolved_theme_colors",
    "get_select_list_theme",
    "get_settings_list_theme",
    "get_theme_by_name",
    "get_theme_export_colors",
    "highlightCode",
    "highlight_code",
    "initTheme",
    "init_theme",
    "loadTheme",
    "loadThemeJson",
    "load_theme",
    "load_theme_json",
    "onThemeChange",
    "on_theme_change",
    "resolveThemeColors",
    "resolve_theme_colors",
    "setCurrentThemeName",
    "setGlobalTheme",
    "setRegisteredThemes",
    "setTheme",
    "setThemeInstance",
    "set_current_theme_name",
    "set_global_theme",
    "set_registered_themes",
    "set_theme",
    "set_theme_instance",
    "stopThemeWatcher",
    "stop_theme_watcher",
    "theme",
]
