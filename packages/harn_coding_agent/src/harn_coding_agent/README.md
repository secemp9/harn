# Harn

Harn is an AI coding assistant for the terminal. It is a full Python port of
[earendil-pi](https://github.com/earendil-works/pi-mono), a TypeScript-based
agent harness, rebuilt from the ground up with a native Python stack. Harn
ships a custom TUI, supports 30+ LLM providers out of the box, and gives models
access to your filesystem through read, edit, write, bash, grep, find, and ls
tools.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### From PyPI (recommended)

```bash
pip install harn
```

Or with uv:

```bash
uv tool install harn
```

Or with pipx:

```bash
pipx install harn
```

Then just run:

```bash
harn
```

### From source

Clone the repository and let uv handle the rest.

#### With uv (recommended)

```bash
git clone https://github.com/secemp9/harn.git
cd harn
uv sync
```

This installs all four workspace packages and their dependencies into a local
virtual environment. The `harn` CLI entry point is immediately available:

```bash
uv run harn
```

#### With pip

```bash
git clone https://github.com/secemp9/harn.git
cd harn
python -m venv .venv
source .venv/bin/activate
pip install -e packages/harn_ai \
            -e packages/harn_agent \
            -e packages/harn_tui \
            -e packages/harn_coding_agent
```

#### With pipx (isolated install)

```bash
git clone https://github.com/secemp9/harn.git
cd harn
pipx install --editable packages/harn_coding_agent \
    --pip-args="-e packages/harn_ai -e packages/harn_agent -e packages/harn_tui"
```

## Quick Start

Set an API key for at least one provider, then launch the interactive TUI:

```bash
export ANTHROPIC_API_KEY="sk-..."
harn
```

Or use uv run if you have not activated the virtual environment:

```bash
uv run harn
```

Send a one-shot prompt without entering interactive mode:

```bash
harn -p "List all Python files in src/"
```

Start with a specific provider and model:

```bash
harn --provider openai --model gpt-4o "Refactor the database module"
```

Use the shorthand provider/model syntax (no `--provider` flag needed):

```bash
harn --model anthropic/claude-sonnet-4-20250514 "Review this code"
```

Continue a previous session:

```bash
harn --continue
```

Attach files to the initial message:

```bash
harn @prompt.md @screenshot.png "What is shown in this image?"
```

Control the thinking/reasoning level:

```bash
harn --thinking high "Solve this complex architecture problem"
```

## Supported Providers

Harn supports a broad set of LLM providers. Set the corresponding
environment variable and optionally pass `--provider <name>`:

| Provider | Environment Variable |
|---|---|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| xAI (Grok) | `XAI_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| Amazon Bedrock | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| Google Vertex AI | (Application Default Credentials) |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` / `CLOUDFLARE_ACCOUNT_ID` |
| GitHub Copilot | (OAuth) |
| MiniMax | `MINIMAX_API_KEY` |
| Moonshot AI | `MOONSHOT_API_KEY` |
| Kimi | `KIMI_API_KEY` |
| Xiaomi MiMo | `XIAOMI_API_KEY` |

List available models for a provider:

```bash
harn --list-models
harn --list-models openai
```

## Configuration

Harn stores its configuration under `~/.harn/agent/` by default. The
location can be overridden with the `HARN_CODING_AGENT_DIR` environment
variable.

Key files and directories:

| Path | Purpose |
|---|---|
| `~/.harn/agent/settings.json` | Global settings (default provider, model, theme, etc.) |
| `~/.harn/agent/models.json` | Custom model definitions and overrides |
| `~/.harn/agent/sessions/` | Saved conversation sessions |
| `.harn/` (project root) | Per-project settings and context |
| `AGENTS.md` | Project-level instructions loaded into the system prompt |

### Useful environment variables

| Variable | Description |
|---|---|
| `HARN_CODING_AGENT_DIR` | Override the config directory |
| `HARN_CODING_AGENT_SESSION_DIR` | Override the session storage directory |
| `HARN_OFFLINE` | Set to `1` to disable startup network operations |

## Built-in Tools

| Tool | Description |
|---|---|
| `read` | Read file contents |
| `bash` | Execute bash commands |
| `edit` | Edit files with find/replace |
| `write` | Write files (create or overwrite) |
| `grep` | Search file contents (off by default) |
| `find` | Find files by glob pattern (off by default) |
| `ls` | List directory contents (off by default) |

Restrict the active tool set with `--tools`:

```bash
harn --tools read,grep,find,ls -p "Review the code in src/"
```

## CLI Reference

```
harn [options] [@files...] [messages...]
```

Common flags:

| Flag | Description |
|---|---|
| `--provider <name>` | Provider name (default: google) |
| `--model <pattern>` | Model pattern or ID; supports `provider/id` and `:<thinking>` suffix |
| `--api-key <key>` | API key (overrides environment variable) |
| `--thinking <level>` | Thinking level: off, minimal, low, medium, high, xhigh |
| `--print`, `-p` | Non-interactive mode: process the prompt and exit |
| `--continue`, `-c` | Continue the most recent session |
| `--resume`, `-r` | Select a session to resume |
| `--session <id>` | Use a specific session file or partial UUID |
| `--no-session` | Ephemeral mode, do not save the session |
| `--models <list>` | Comma-separated model patterns for Ctrl+P cycling |
| `--tools`, `-t <list>` | Comma-separated allowlist of tool names |
| `--no-tools`, `-nt` | Disable all tools |
| `--extension`, `-e <path>` | Load an extension file |
| `--verbose` | Force verbose startup output |
| `--offline` | Disable startup network operations |
| `--list-models [search]` | List available models with optional fuzzy search |
| `--export <file>` | Export a session to HTML |
| `--help`, `-h` | Show full help text |
| `--version`, `-v` | Show version |

Run `harn --help` for the complete list of options, environment variables,
and examples.

## Monorepo Structure

Harn is organized as a uv workspace with four packages:

```
harn/
  packages/
    harn_ai/             Unified multi-provider LLM API
    harn_agent/          Agent runtime with tool calling and state management
    harn_tui/            Terminal UI library with differential rendering
    harn_coding_agent/   Interactive coding agent CLI (the main entry point)
```

| Package | Description |
|---|---|
| **harn-ai** | Streaming LLM client supporting Anthropic, OpenAI, Google, Mistral, Bedrock, and more |
| **harn-agent** | Agent loop, tool dispatch, session persistence, and context management |
| **harn-tui** | Terminal rendering, input handling, markdown display, and image support |
| **harn** | CLI entry point, built-in tools (read/edit/write/bash), extensions, and skills |

## Development

```bash
git clone https://github.com/secemp9/harn.git
cd harn
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

Run harn from source:

```bash
uv run harn
```

## Roadmap

- [ ] Port Pi extensions from the upstream ecosystem to Python (extensions live
  outside the pi-mono monorepo at various `earendil-works/*` repos and
  community packages)
- [ ] Publish individual extension packages to PyPI as `harn-ext-*`
- [ ] Track upstream earendil-pi releases and keep harn at feature parity

## License

This project does not yet specify a license. See the individual source files
for details.
