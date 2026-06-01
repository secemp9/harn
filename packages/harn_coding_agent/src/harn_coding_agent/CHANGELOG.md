# Changelog

This project is a Python port of [pi coding-agent](https://github.com/earendil-works/pi-mono/tree/main/packages/coding-agent). The changelog below covers migration notes relevant to the harn Python port.

---

### Extensions Migration

Hooks and custom tools are now unified as **extensions**. Both were Python modules exporting a factory function that receives an API object. Now there's one concept, one discovery location, one CLI flag, one settings.json entry.

**Automatic migration:**

- `commands/` directories are automatically renamed to `prompts/` on startup (both `~/.harn/agent/commands/` and `.harn/commands/`)

**Manual migration required:**

1. Move files from `hooks/` and `tools/` directories to `extensions/` (deprecation warnings shown on startup)
2. Update imports and type names in your extension code
3. Update `settings.json` if you have explicit hook and custom tool paths configured

**Directory changes:**

```
# Before
~/.harn/agent/hooks/*.py       ->  ~/.harn/agent/extensions/*.py
~/.harn/agent/tools/*.py       ->  ~/.harn/agent/extensions/*.py
.harn/hooks/*.py               ->  .harn/extensions/*.py
.harn/tools/*.py               ->  .harn/extensions/*.py
```

**Extension discovery rules** (in `extensions/` directories):

1. **Direct files:** `extensions/*.py` -> loaded directly
2. **Subdirectory with index:** `extensions/myext/index.py` -> loaded as single extension
3. **Subdirectory with pyproject.toml:** `extensions/myext/pyproject.toml` with `[tool.harn]` section -> loads declared paths

```toml
# extensions/my-package/pyproject.toml
[project]
name = "my-extension-package"
dependencies = ["pydantic>=2.0.0"]

[tool.harn]
extensions = ["./src/main.py", "./src/tools.py"]
```

No recursion beyond one level. Complex packages must use the `pyproject.toml` manifest. Dependencies are resolved via importlib, and extensions can be published to and installed from PyPI.

**Type renames:**

- `HookAPI` -> `ExtensionAPI`
- `HookContext` -> `ExtensionContext`
- `HookCommandContext` -> `ExtensionCommandContext`
- `HookUIContext` -> `ExtensionUIContext`
- `CustomToolAPI` -> `ExtensionAPI` (merged)
- `CustomToolContext` -> `ExtensionContext` (merged)
- `CustomToolUIContext` -> `ExtensionUIContext`
- `CustomTool` -> `ToolDefinition`
- `CustomToolFactory` -> `ExtensionFactory`
- `HookMessage` -> `CustomMessage`

**Import changes:**

```python
# Before (hook)
from harn_coding_agent import HookAPI, HookContext
def extension_factory(harn: HookAPI): ...

# Before (custom tool)
from harn_coding_agent import CustomToolFactory
def factory(harn): return {"name": "my_tool", ...}

# After (both are now extensions)
from harn_coding_agent import ExtensionAPI
def extension_factory(harn: ExtensionAPI):
    @harn.on("tool_call")
    async def on_tool_call(event, ctx): ...
    harn.register_tool(name="my_tool", ...)
```

**Custom tools now have full context access.** Tools registered via `harn.register_tool()` now receive the same `ctx` object that event handlers receive. Previously, custom tools had limited context. Now all extension code shares the same capabilities:

- `harn.register_tool()` - Register tools the LLM can call
- `harn.register_command()` - Register commands like `/mycommand`
- `harn.register_shortcut()` - Register keyboard shortcuts (shown in `/hotkeys`)
- `harn.register_flag()` - Register CLI flags (shown in `--help`)
- `harn.register_message_renderer()` - Custom TUI rendering for message types
- `harn.on()` - Subscribe to lifecycle events (tool_call, session_start, etc.)
- `harn.send_message()` - Inject messages into the conversation
- `harn.append_entry()` - Persist custom data in session (survives restart/branch)
- `harn.exec()` - Run shell commands
- `harn.get_active_tools()` / `harn.set_active_tools()` - Dynamic tool enable/disable
- `harn.get_all_tools()` - List all available tools
- `harn.events` - Event bus for cross-extension communication
- `ctx.ui.confirm()` / `select()` / `input()` - User prompts
- `ctx.ui.notify()` - Toast notifications
- `ctx.ui.set_status()` - Persistent status in footer (multiple extensions can set their own)
- `ctx.ui.set_widget()` - Widget display above editor
- `ctx.ui.set_title()` - Set terminal window title
- `ctx.ui.custom()` - Full TUI component with keyboard handling
- `ctx.ui.editor()` - Multi-line text editor with external editor support
- `ctx.session_manager` - Read session entries, get branch history

**Settings changes:**

```json
// Before
{
  "hooks": ["./my_hook.py"],
  "customTools": ["./my_tool.py"]
}

// After
{
  "extensions": ["./my_extension.py"]
}
```

**CLI changes:**

```bash
# Before
harn --hook ./safety.py --tool ./todo.py

# After
harn --extension ./safety.py -e ./todo.py
```

### Prompt Templates Migration

"Slash commands" (markdown files defining reusable prompts invoked via `/name`) are renamed to "prompt templates" to avoid confusion with extension-registered commands.

**Automatic migration:** The `commands/` directory is automatically renamed to `prompts/` on startup (if `prompts/` doesn't exist). Works for both regular directories and symlinks.

**Directory changes:**

```
~/.harn/agent/commands/*.md    ->  ~/.harn/agent/prompts/*.md
.harn/commands/*.md            ->  .harn/prompts/*.md
```

**Type renames:**

- `FileSlashCommand` -> `PromptTemplate`
- `LoadSlashCommandsOptions` -> `LoadPromptTemplatesOptions`

**Function renames:**

- `discover_slash_commands()` -> `discover_prompt_templates()`
- `load_slash_commands()` -> `load_prompt_templates()`
- `expand_slash_command()` -> `expand_prompt_template()`
- `get_commands_dir()` -> `get_prompts_dir()`

**Option renames:**

- `CreateAgentSessionOptions.slash_commands` -> `.prompt_templates`
- `AgentSession.file_commands` -> `.prompt_templates`
- `PromptOptions.expand_slash_commands` -> `.expand_prompt_templates`
