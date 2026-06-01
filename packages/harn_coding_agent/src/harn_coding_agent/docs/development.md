# Development

See [AGENTS.md](../../../AGENTS.md) for additional guidelines.

## Setup

```bash
git clone https://github.com/secemp9/harnify
cd harnify
pip install -e .
```

Run from source:

```bash
python -m harn_coding_agent
```

The command can be run from any directory. Harn keeps the caller's current working directory.

## Forking / Rebranding

Configure via `pyproject.toml`:

```toml
[tool.harn]
name = "harn"
config_dir = ".harn"
```

Change `name`, `config_dir`, and console script entry point for your fork. Affects CLI banner, config paths, and environment variable names.

## Path Resolution

Three execution modes: pip install, standalone binary, running from source.

**Always use `config.py`** for package assets:

```python
from harn_coding_agent.config import get_package_dir, get_theme_dir
```

Never use `__file__` directly for package assets.

## Debug Command

`/debug` (hidden) writes to `~/.harn/agent/harn-debug.log`:
- Rendered TUI lines with ANSI codes
- Last messages sent to the LLM

## Testing

```bash
./test.sh                         # Run non-LLM tests (no API keys needed)
pytest                            # Run all tests
pytest tests/test_specific.py     # Run specific test
```

## Project Structure

```
packages/
  harn_ai/           # LLM provider abstraction
  harn_agent/        # Agent loop and message types
  harn_tui/          # Terminal UI components
  harn_coding_agent/ # CLI and interactive mode
```
