> harn can create TUI components. Ask it to build one for your use case.

# TUI Components

Extensions and custom tools can render custom TUI components for interactive user interfaces. This page covers the component system and available building blocks.

**Source:** [`harn-tui`](https://pypi.org/project/harn-tui/)

## Component Interface

All components implement:

```python
from typing import Protocol, Optional

class Component(Protocol):
    def render(self, width: int) -> list[str]:
        """Return list of strings (one per line). Each line must not exceed width."""
        ...

    def handle_input(self, data: str) -> None:
        """Receive keyboard input when component has focus."""
        ...

    wants_key_release: bool = False
    """If true, component receives key release events (Kitty protocol)."""

    def invalidate(self) -> None:
        """Clear cached render state. Called on theme changes."""
        ...
```

| Method | Description |
|--------|-------------|
| `render(width)` | Return list of strings (one per line). Each line **must not exceed `width`**. |
| `handle_input(data)` | Receive keyboard input when component has focus. |
| `wants_key_release` | If true, component receives key release events (Kitty protocol). Default: False. |
| `invalidate()` | Clear cached render state. Called on theme changes. |

The TUI appends a full SGR reset and OSC 8 reset at the end of each rendered line. Styles do not carry across lines. If you emit multi-line text with styling, reapply styles per line or use `wrap_text_with_ansi()` so styles are preserved for each wrapped line.

## Focusable Interface (IME Support)

Components that display a text cursor and need IME (Input Method Editor) support should implement the `Focusable` interface:

```python
from harn_tui import CURSOR_MARKER, Component, Focusable

class MyInput(Component, Focusable):
    focused: bool = False  # Set by TUI when focus changes

    def render(self, width: int) -> list[str]:
        marker = CURSOR_MARKER if self.focused else ""
        # Emit marker right before the fake cursor
        return [f"> {before_cursor}{marker}\x1b[7m{at_cursor}\x1b[27m{after_cursor}"]
```

When a `Focusable` component has focus, TUI:
1. Sets `focused = True` on the component
2. Scans rendered output for `CURSOR_MARKER` (a zero-width APC escape sequence)
3. Positions the hardware terminal cursor at that location
4. Shows the hardware cursor

This enables IME candidate windows to appear at the correct position for CJK input methods. The `Editor` and `Input` built-in components already implement this interface.

### Container Components with Embedded Inputs

When a container component (dialog, selector, etc.) contains an `Input` or `Editor` child, the container must implement `Focusable` and propagate the focus state to the child. Otherwise, the hardware cursor won't be positioned correctly for IME input.

```python
from harn_tui import Container, Focusable, Input

class SearchDialog(Container, Focusable):
    def __init__(self):
        super().__init__()
        self._search_input = Input()
        self._focused = False
        self.add_child(self._search_input)

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value
```

Without this propagation, typing with an IME (Chinese, Japanese, Korean, etc.) will show the candidate window in the wrong position on screen.

## Using Components

**In extensions** via `ctx.ui.custom()`:

```python
@harn.on("session_start")
async def on_session_start(event, ctx):
    handle = ctx.ui.custom(my_component)
    # handle.request_render() - trigger re-render
    # handle.close() - restore normal UI
```

**In custom tools** via `harn.ui.custom()`:

```python
async def execute(self, tool_call_id, params, on_update, ctx, signal):
    handle = harn.ui.custom(my_component)
    # ...
    handle.close()
```

## Overlays

Overlays render components on top of existing content without clearing the screen. Pass `overlay=True` to `ctx.ui.custom()`:

```python
result = await ctx.ui.custom(
    lambda tui, theme, keybindings, done: MyDialog(on_close=done),
    overlay=True,
)
```

For positioning and sizing, use `overlay_options`:

```python
result = await ctx.ui.custom(
    lambda tui, theme, keybindings, done: SidePanel(on_close=done),
    overlay=True,
    overlay_options={
        # Size: number or percentage string
        "width": "50%",          # 50% of terminal width
        "min_width": 40,         # minimum 40 columns
        "max_height": "80%",     # max 80% of terminal height

        # Position: anchor-based (default: "center")
        "anchor": "right-center",  # 9 positions: center, top-left, top-center, etc.
        "offset_x": -2,           # offset from anchor
        "offset_y": 0,

        # Or percentage/absolute positioning
        "row": "25%",            # 25% from top
        "col": 10,              # column 10

        # Margins
        "margin": 2,            # all sides, or {"top", "right", "bottom", "left"}

        # Responsive: hide on narrow terminals
        "visible": lambda term_width, term_height: term_width >= 80,
    },
    # Get handle for programmatic visibility control
    on_handle=lambda handle: None,
    # handle.set_hidden(True/False) - toggle visibility
    # handle.hide() - permanently remove
)
```

### Overlay Lifecycle

Overlay components are disposed when closed. Don't reuse references - create fresh instances:

```python
# Wrong - stale reference
menu = None

async def show():
    nonlocal menu
    menu = await ctx.ui.custom(
        lambda _, __, ___, done: MenuComponent(done),
        overlay=True,
    )

# Correct - re-call to re-show
async def show_menu():
    return await ctx.ui.custom(
        lambda _, __, ___, done: MenuComponent(done),
        overlay=True,
    )

await show_menu()  # First show
await show_menu()  # "Back" = just call again
```

## Built-in Components

Import from `harn_tui`:

```python
from harn_tui import Text, Box, Container, Spacer, Markdown
```

### Text

Multi-line text with word wrapping.

```python
text = Text(
    "Hello World",    # content
    padding_x=1,     # default: 1
    padding_y=1,     # default: 1
    bg_fn=bg_gray,   # optional background function
)
text.set_text("Updated")
```

### Box

Container with padding and background color.

```python
box = Box(
    padding_x=1,
    padding_y=1,
    bg_fn=bg_gray,   # background function
)
box.add_child(Text("Content", padding_x=0, padding_y=0))
box.set_bg_fn(bg_blue)
```

### Container

Groups child components vertically.

```python
container = Container()
container.add_child(component1)
container.add_child(component2)
container.remove_child(component1)
```

### Spacer

Empty vertical space.

```python
spacer = Spacer(2)  # 2 empty lines
```

### Markdown

Renders markdown with syntax highlighting.

```python
md = Markdown(
    "# Title\n\nSome **bold** text",
    padding_x=1,
    padding_y=1,
    theme=theme,     # MarkdownTheme (see below)
)
md.set_text("Updated markdown")
```

### Image

Renders images in supported terminals (Kitty, iTerm2, Ghostty, WezTerm).

```python
image = Image(
    base64_data,      # base64-encoded image
    "image/png",      # MIME type
    theme,            # ImageTheme
    max_width_cells=80,
    max_height_cells=24,
)
```

## Keyboard Input

Use `matches_key()` for key detection:

```python
from harn_tui import matches_key, Key

def handle_input(self, data: str) -> None:
    if matches_key(data, Key.UP):
        self.selected_index -= 1
    elif matches_key(data, Key.ENTER):
        if self.on_select:
            self.on_select(self.selected_index)
    elif matches_key(data, Key.ESCAPE):
        if self.on_cancel:
            self.on_cancel()
    elif matches_key(data, Key.ctrl("c")):
        pass  # Ctrl+C
```

**Key identifiers** (use `Key.*` for autocomplete, or string literals):
- Basic keys: `Key.ENTER`, `Key.ESCAPE`, `Key.TAB`, `Key.SPACE`, `Key.BACKSPACE`, `Key.DELETE`, `Key.HOME`, `Key.END`
- Arrow keys: `Key.UP`, `Key.DOWN`, `Key.LEFT`, `Key.RIGHT`
- With modifiers: `Key.ctrl("c")`, `Key.shift("tab")`, `Key.alt("left")`, `Key.ctrl_shift("p")`
- String format also works: `"enter"`, `"ctrl+c"`, `"shift+tab"`, `"ctrl+shift+p"`

## Line Width

**Critical:** Each line from `render()` must not exceed the `width` parameter.

```python
from harn_tui import visible_width, truncate_to_width

def render(self, width: int) -> list[str]:
    # Truncate long lines
    return [truncate_to_width(self.text, width)]
```

Utilities:
- `visible_width(s)` - Get display width (ignores ANSI codes)
- `truncate_to_width(s, width, ellipsis=None)` - Truncate with optional ellipsis
- `wrap_text_with_ansi(s, width)` - Word wrap preserving ANSI codes

## Creating Custom Components

Example: Interactive selector

```python
from harn_tui import matches_key, Key, truncate_to_width, visible_width

class MySelector:
    def __init__(self, items: list[str]):
        self.items = items
        self.selected = 0
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None
        self.on_select: callable | None = None
        self.on_cancel: callable | None = None

    def handle_input(self, data: str) -> None:
        if matches_key(data, Key.UP) and self.selected > 0:
            self.selected -= 1
            self.invalidate()
        elif matches_key(data, Key.DOWN) and self.selected < len(self.items) - 1:
            self.selected += 1
            self.invalidate()
        elif matches_key(data, Key.ENTER):
            if self.on_select:
                self.on_select(self.items[self.selected])
        elif matches_key(data, Key.ESCAPE):
            if self.on_cancel:
                self.on_cancel()

    def render(self, width: int) -> list[str]:
        if self._cached_lines and self._cached_width == width:
            return self._cached_lines

        self._cached_lines = []
        for i, item in enumerate(self.items):
            prefix = "> " if i == self.selected else "  "
            self._cached_lines.append(truncate_to_width(prefix + item, width))
        self._cached_width = width
        return self._cached_lines

    def invalidate(self) -> None:
        self._cached_width = None
        self._cached_lines = None
```

Usage in an extension:

```python
@harn.register_command("pick", description="Pick an item")
async def pick_command(args, ctx):
    items = ["Option A", "Option B", "Option C"]
    selector = MySelector(items)

    import asyncio
    future = asyncio.get_event_loop().create_future()

    def on_select(item):
        ctx.ui.notify(f"Selected: {item}", "info")
        handle.close()
        future.set_result(None)

    def on_cancel():
        handle.close()
        future.set_result(None)

    selector.on_select = on_select
    selector.on_cancel = on_cancel
    handle = ctx.ui.custom(selector)

    await future
```

## Theming

Components accept theme objects for styling.

**In `render_call`/`render_result`**, use the `theme` parameter:

```python
def render_result(self, result, options, theme, context):
    # Use theme.fg() for foreground colors
    return Text(theme.fg("success", "Done!"), padding_x=0, padding_y=0)

    # Use theme.bg() for background colors
    styled = theme.bg("toolPendingBg", theme.fg("accent", "text"))
```

**Foreground colors** (`theme.fg(color, text)`):

| Category | Colors |
|----------|--------|
| General | `text`, `accent`, `muted`, `dim` |
| Status | `success`, `error`, `warning` |
| Borders | `border`, `borderAccent`, `borderMuted` |
| Messages | `userMessageText`, `customMessageText`, `customMessageLabel` |
| Tools | `toolTitle`, `toolOutput` |
| Diffs | `toolDiffAdded`, `toolDiffRemoved`, `toolDiffContext` |
| Markdown | `mdHeading`, `mdLink`, `mdLinkUrl`, `mdCode`, `mdCodeBlock`, `mdCodeBlockBorder`, `mdQuote`, `mdQuoteBorder`, `mdHr`, `mdListBullet` |
| Syntax | `syntaxComment`, `syntaxKeyword`, `syntaxFunction`, `syntaxVariable`, `syntaxString`, `syntaxNumber`, `syntaxType`, `syntaxOperator`, `syntaxPunctuation` |
| Thinking | `thinkingOff`, `thinkingMinimal`, `thinkingLow`, `thinkingMedium`, `thinkingHigh`, `thinkingXhigh` |
| Modes | `bashMode` |

**Background colors** (`theme.bg(color, text)`):

`selectedBg`, `userMessageBg`, `customMessageBg`, `toolPendingBg`, `toolSuccessBg`, `toolErrorBg`

**For Markdown**, use `get_markdown_theme()`:

```python
from harn_coding_agent import get_markdown_theme
from harn_tui import Markdown

def render_result(self, result, options, theme, context):
    md_theme = get_markdown_theme()
    return Markdown(result.details.markdown, padding_x=0, padding_y=0, theme=md_theme)
```

**For custom components**, define your own theme interface:

```python
from typing import Protocol

class MyTheme(Protocol):
    def selected(self, s: str) -> str: ...
    def normal(self, s: str) -> str: ...
```

## Debug logging

Set `HARN_TUI_WRITE_LOG` to capture the raw ANSI stream written to stdout.

```bash
HARN_TUI_WRITE_LOG=/tmp/tui-ansi.log python -m harn_tui.test.chat_simple
```

## Performance

Cache rendered output when possible:

```python
class CachedComponent:
    def __init__(self):
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines and self._cached_width == width:
            return self._cached_lines
        # ... compute lines ...
        self._cached_width = width
        self._cached_lines = lines
        return lines

    def invalidate(self) -> None:
        self._cached_width = None
        self._cached_lines = None
```

Call `invalidate()` when state changes, then `handle.request_render()` to trigger re-render.

## Invalidation and Theme Changes

When the theme changes, the TUI calls `invalidate()` on all components to clear their caches. Components must properly implement `invalidate()` to ensure theme changes take effect.

### The Problem

If a component pre-bakes theme colors into strings (via `theme.fg()`, `theme.bg()`, etc.) and caches them, the cached strings contain ANSI escape codes from the old theme. Simply clearing the render cache isn't enough if the component stores the themed content separately.

**Wrong approach** (theme colors won't update):

```python
class BadComponent(Container):
    def __init__(self, message: str, theme):
        super().__init__()
        # Pre-baked theme colors stored in Text component
        self.content = Text(theme.fg("accent", message), padding_x=1, padding_y=0)
        self.add_child(self.content)
    # No invalidate override - parent's invalidate only clears
    # child render caches, not the pre-baked content
```

### The Solution

Components that build content with theme colors must rebuild that content when `invalidate()` is called:

```python
class GoodComponent(Container):
    def __init__(self, message: str):
        super().__init__()
        self._message = message
        self.content = Text("", padding_x=1, padding_y=0)
        self.add_child(self.content)
        self._update_display()

    def _update_display(self) -> None:
        # Rebuild content with current theme
        self.content.set_text(theme.fg("accent", self._message))

    def invalidate(self) -> None:
        super().invalidate()  # Clear child caches
        self._update_display()  # Rebuild with new theme
```

### Pattern: Rebuild on Invalidate

For components with complex content:

```python
class ComplexComponent(Container):
    def __init__(self, data):
        super().__init__()
        self._data = data
        self._rebuild()

    def _rebuild(self) -> None:
        self.clear()  # Remove all children

        # Build UI with current theme
        self.add_child(Text(theme.fg("accent", theme.bold("Title")), padding_x=1, padding_y=0))
        self.add_child(Spacer(1))

        for item in self._data.items:
            color = "success" if item.active else "muted"
            self.add_child(Text(theme.fg(color, item.label), padding_x=1, padding_y=0))

    def invalidate(self) -> None:
        super().invalidate()
        self._rebuild()
```

### When This Matters

This pattern is needed when:

1. **Pre-baking theme colors** - Using `theme.fg()` or `theme.bg()` to create styled strings stored in child components
2. **Syntax highlighting** - Using `highlight_code()` which applies theme-based syntax colors
3. **Complex layouts** - Building child component trees that embed theme colors

This pattern is NOT needed when:

1. **Using theme callbacks** - Passing functions like `lambda text: theme.fg("accent", text)` that are called during render
2. **Simple containers** - Just grouping other components without adding themed content
3. **Stateless render** - Computing themed output fresh in every `render()` call (no caching)

## Common Patterns

These patterns cover the most common UI needs in extensions. **Copy these patterns instead of building from scratch.**

### Pattern 1: Selection Dialog (SelectList)

For letting users pick from a list of options. Use `SelectList` from `harn_tui` with `DynamicBorder` for framing.

```python
from harn_coding_agent import DynamicBorder
from harn_tui import Container, SelectItem, SelectList, Text

@harn.register_command("pick")
async def pick_handler(args, ctx):
    items = [
        SelectItem(value="opt1", label="Option 1", description="First option"),
        SelectItem(value="opt2", label="Option 2", description="Second option"),
        SelectItem(value="opt3", label="Option 3"),  # description is optional
    ]

    def factory(tui, theme, _kb, done):
        container = Container()

        # Top border
        container.add_child(DynamicBorder(lambda s: theme.fg("accent", s)))

        # Title
        container.add_child(Text(theme.fg("accent", theme.bold("Pick an Option")), padding_x=1, padding_y=0))

        # SelectList with theme
        select_list = SelectList(items, min(len(items), 10), {
            "selected_prefix": lambda t: theme.fg("accent", t),
            "selected_text": lambda t: theme.fg("accent", t),
            "description": lambda t: theme.fg("muted", t),
            "scroll_info": lambda t: theme.fg("dim", t),
            "no_match": lambda t: theme.fg("warning", t),
        })
        select_list.on_select = lambda item: done(item.value)
        select_list.on_cancel = lambda: done(None)
        container.add_child(select_list)

        # Help text
        container.add_child(Text(theme.fg("dim", "up/down navigate | enter select | esc cancel"), padding_x=1, padding_y=0))

        # Bottom border
        container.add_child(DynamicBorder(lambda s: theme.fg("accent", s)))

        return {
            "render": lambda w: container.render(w),
            "invalidate": lambda: container.invalidate(),
            "handle_input": lambda data: (select_list.handle_input(data), tui.request_render()),
        }

    result = await ctx.ui.custom(factory)

    if result:
        ctx.ui.notify(f"Selected: {result}", "info")
```

### Pattern 2: Async Operation with Cancel (BorderedLoader)

For operations that take time and should be cancellable. `BorderedLoader` shows a spinner and handles escape to cancel.

```python
from harn_coding_agent import BorderedLoader

@harn.register_command("fetch")
async def fetch_handler(args, ctx):
    def factory(tui, theme, _kb, done):
        loader = BorderedLoader(tui, theme, "Fetching data...")
        loader.on_abort = lambda: done(None)

        # Do async work
        async def do_fetch():
            try:
                data = await fetch_data(loader.signal)
                done(data)
            except Exception:
                done(None)

        import asyncio
        asyncio.create_task(do_fetch())
        return loader

    result = await ctx.ui.custom(factory)

    if result is None:
        ctx.ui.notify("Cancelled", "info")
    else:
        ctx.ui.set_editor_text(result)
```

### Pattern 3: Settings/Toggles (SettingsList)

For toggling multiple settings. Use `SettingsList` from `harn_tui` with `get_settings_list_theme()`.

```python
from harn_coding_agent import get_settings_list_theme
from harn_tui import Container, SettingItem, SettingsList, Text

@harn.register_command("settings")
async def settings_handler(args, ctx):
    items = [
        SettingItem(id="verbose", label="Verbose mode", current_value="off", values=["on", "off"]),
        SettingItem(id="color", label="Color output", current_value="on", values=["on", "off"]),
    ]

    def factory(_tui, theme, _kb, done):
        container = Container()
        container.add_child(Text(theme.fg("accent", theme.bold("Settings")), padding_x=1, padding_y=1))

        settings_list = SettingsList(
            items,
            min(len(items) + 2, 15),
            get_settings_list_theme(),
            on_change=lambda id, new_value: ctx.ui.notify(f"{id} = {new_value}", "info"),
            on_close=lambda: done(None),
            enable_search=True,
        )
        container.add_child(settings_list)

        return {
            "render": lambda w: container.render(w),
            "invalidate": lambda: container.invalidate(),
            "handle_input": lambda data: settings_list.handle_input(data),
        }

    await ctx.ui.custom(factory)
```

### Pattern 4: Persistent Status Indicator

Show status in the footer that persists across renders. Good for mode indicators.

```python
# Set status (shown in footer)
ctx.ui.set_status("my-ext", ctx.ui.theme.fg("accent", "● active"))

# Clear status
ctx.ui.set_status("my-ext", None)
```

### Pattern 4b: Working Indicator Customization

Customize the inline working indicator shown while harn is streaming a response.

```python
# Static indicator
ctx.ui.set_working_indicator(frames=[ctx.ui.theme.fg("accent", "●")])

# Custom animated indicator
ctx.ui.set_working_indicator(
    frames=[
        ctx.ui.theme.fg("dim", "·"),
        ctx.ui.theme.fg("muted", "•"),
        ctx.ui.theme.fg("accent", "●"),
        ctx.ui.theme.fg("muted", "•"),
    ],
    interval_ms=120,
)

# Hide the indicator entirely
ctx.ui.set_working_indicator(frames=[])

# Restore harn's default spinner
ctx.ui.set_working_indicator()
```

This only affects the normal streaming working indicator. Compaction and retry loaders keep their built-in styling. Custom frames are rendered verbatim, so extensions must add their own colors when needed.

### Pattern 5: Widgets Above/Below Editor

Show persistent content above or below the input editor. Good for todo lists, progress.

```python
# Simple string array (above editor by default)
ctx.ui.set_widget("my-widget", ["Line 1", "Line 2"])

# Render below the editor
ctx.ui.set_widget("my-widget", ["Line 1", "Line 2"], placement="below_editor")

# Or with theme
def widget_factory(_tui, theme):
    lines = []
    for item in items:
        if item.done:
            lines.append(theme.fg("success", "✓ ") + theme.fg("muted", item.text))
        else:
            lines.append(theme.fg("dim", "○ ") + item.text)
    return {
        "render": lambda: lines,
        "invalidate": lambda: None,
    }

ctx.ui.set_widget("my-widget", widget_factory)

# Clear
ctx.ui.set_widget("my-widget", None)
```

### Pattern 6: Custom Footer

Replace the footer. `footer_data` exposes data not otherwise accessible to extensions.

```python
def footer_factory(tui, theme, footer_data):
    def render(width: int) -> list[str]:
        branch = footer_data.get_git_branch() or "no git"
        return [f"{ctx.model.id} ({branch})"]

    dispose = footer_data.on_branch_change(lambda: tui.request_render())

    return {
        "invalidate": lambda: None,
        "render": render,
        "dispose": dispose,
    }

ctx.ui.set_footer(footer_factory)

ctx.ui.set_footer(None)  # restore default
```

Token stats available via `ctx.session_manager.get_branch()` and `ctx.model`.

### Pattern 7: Custom Editor (vim mode, etc.)

Replace the main input editor with a custom implementation. Useful for modal editing (vim), different keybindings (emacs), or specialized input handling.

```python
from harn_coding_agent import CustomEditor, ExtensionAPI
from harn_tui import matches_key, truncate_to_width

class VimEditor(CustomEditor):
    def __init__(self, theme, keybindings):
        super().__init__(theme, keybindings)
        self._mode = "insert"

    def handle_input(self, data: str) -> None:
        # Escape: switch to normal mode, or pass through for app handling
        if matches_key(data, "escape"):
            if self._mode == "insert":
                self._mode = "normal"
                return
            # In normal mode, escape aborts agent (handled by CustomEditor)
            super().handle_input(data)
            return

        # Insert mode: pass everything to CustomEditor
        if self._mode == "insert":
            super().handle_input(data)
            return

        # Normal mode: vim-style navigation
        if data == "i":
            self._mode = "insert"
        elif data == "h":
            super().handle_input("\x1b[D")  # Left
        elif data == "j":
            super().handle_input("\x1b[B")  # Down
        elif data == "k":
            super().handle_input("\x1b[A")  # Up
        elif data == "l":
            super().handle_input("\x1b[C")  # Right
        else:
            # Pass unhandled keys to super (ctrl+c, etc.), but filter printable chars
            if len(data) == 1 and ord(data) >= 32:
                return
            super().handle_input(data)

    def render(self, width: int) -> list[str]:
        lines = super().render(width)
        # Add mode indicator to bottom border
        if lines:
            label = " NORMAL " if self._mode == "normal" else " INSERT "
            last_line = lines[-1]
            lines[-1] = truncate_to_width(last_line, width - len(label), "") + label
        return lines


def extension_factory(harn: ExtensionAPI):
    @harn.on("session_start")
    def on_session_start(event, ctx):
        # Factory receives theme and keybindings from the app
        ctx.ui.set_editor_component(
            lambda tui, theme, keybindings: VimEditor(theme, keybindings)
        )
```

**Key points:**

- **Extend `CustomEditor`** (not base `Editor`) to get app keybindings (escape to abort, ctrl+d to exit, model switching, etc.)
- **Call `super().handle_input(data)`** for keys you don't handle
- **Factory pattern**: `set_editor_component` receives a factory function that gets `tui`, `theme`, and `keybindings`
- **Pass `None`** to restore the default editor: `ctx.ui.set_editor_component(None)`

## Key Rules

1. **Always use theme from callback** - Don't import theme directly. Use `theme` from the `ctx.ui.custom(lambda tui, theme, keybindings, done: ...)` callback.

2. **Always type DynamicBorder color param** - Write `lambda s: theme.fg("accent", s)`.

3. **Call tui.request_render() after state changes** - In `handle_input`, call `tui.request_render()` after updating state.

4. **Return the three-method dict** - Custom components need `{"render", "invalidate", "handle_input"}`.

5. **Use existing components** - `SelectList`, `SettingsList`, `BorderedLoader` cover 90% of cases. Don't rebuild them.
