# Quickstart

This page gets you from install to a useful first harn session.

## Install

Harn is distributed as a PyPI package:

```bash
pip install harn
```

Or with uv:

```bash
uv add harn
```

### Uninstall

Use the package manager that installed harn:

```bash
# pip
pip uninstall harn

# uv
uv remove harn

# pipx
pipx uninstall harn
```

Uninstalling harn leaves settings, credentials, sessions, and installed harn packages in `~/.harn/agent/`.

Then start harn in the project directory you want it to work on:

```bash
cd /path/to/project
harn
```

## Authenticate

Harn can use subscription providers through `/login`, or API-key providers through environment variables or the auth file.

### Option 1: subscription login

Start harn and run:

```text
/login
```

Then select a provider. Built-in subscription logins include Claude Pro/Max, ChatGPT Plus/Pro (Codex), and GitHub Copilot.

### Option 2: API key

Set an API key before launching harn:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
harn
```

You can also run `/login` and select an API-key provider to store the key in `~/.harn/agent/auth.json`.

See [Providers](providers.md) for all supported providers, environment variables, and cloud-provider setup.

## First session

Once harn starts, type a request and press Enter:

```text
Summarize this repository and tell me how to run its checks.
```

By default, harn gives the model four tools:

- `read` - read files
- `write` - create or overwrite files
- `edit` - patch files
- `bash` - run shell commands

Additional built-in read-only tools (`grep`, `find`, `ls`) are available through tool options. Harn runs in your current working directory and can modify files there. Use git or another checkpointing workflow if you want easy rollback.

## Give harn project instructions

Harn loads context files at startup. Add an `AGENTS.md` file to tell it how to work in a project:

```markdown
# Project Instructions

- Run `pytest` after code changes.
- Do not run production migrations locally.
- Keep responses concise.
```

Harn loads:

- `~/.harn/agent/AGENTS.md` for global instructions
- `AGENTS.md` or `CLAUDE.md` from parent directories and the current directory

Restart harn, or run `/reload`, after changing context files.

## Common things to try

### Reference files

Type `@` in the editor to fuzzy-search files, or pass files on the command line:

```bash
harn @README.md "Summarize this"
harn @src/app.py @src/test_app.py "Review these together"
```

Images can be pasted with Ctrl+V (Alt+V on Windows) or dragged into supported terminals.

### Run shell commands

In interactive mode:

```text
!pytest
```

The command output is sent to the model. Use `!!command` to run a command without adding its output to the model context.

### Switch models

Use `/model` or Ctrl+L to choose a model. Use Shift+Tab to cycle thinking level. Use Ctrl+P / Shift+Ctrl+P to cycle through scoped models.

### Continue later

Sessions are saved automatically:

```bash
harn -c                  # Continue most recent session
harn -r                  # Browse previous sessions
harn --session <path|id> # Open a specific session
```

Inside harn, use `/resume`, `/new`, `/tree`, `/fork`, and `/clone` to manage sessions.

### Non-interactive mode

For one-shot prompts:

```bash
harn -p "Summarize this codebase"
cat README.md | harn -p "Summarize this text"
harn -p @screenshot.png "What's in this image?"
```

Use `--mode json` for JSON event output or `--mode rpc` for process integration.

## Next steps

- [Using Harn](usage.md) - interactive mode, slash commands, sessions, context files, and CLI reference.
- [Providers](providers.md) - authentication and model setup.
- [Settings](settings.md) - global and project configuration.
- [Keybindings](keybindings.md) - shortcuts and customization.
- [Harn Packages](packages.md) - install shared extensions, skills, prompts, and themes.

Platform notes: [Windows](windows.md), [Termux](termux.md), [tmux](tmux.md), [Terminal setup](terminal-setup.md), [Shell aliases](shell-aliases.md).
