> harn can help you create harn packages. Ask it to bundle your extensions, skills, prompt templates, or themes.

# Harn Packages

Harn packages bundle extensions, skills, prompt templates, and themes so you can share them through PyPI or git. A package can declare resources in `pyproject.toml` under the `[tool.harn]` key, or use conventional directories.

## Table of Contents

- [Install and Manage](#install-and-manage)
- [Package Sources](#package-sources)
- [Creating a Harn Package](#creating-a-harn-package)
- [Package Structure](#package-structure)
- [Dependencies](#dependencies)
- [Package Filtering](#package-filtering)
- [Enable and Disable Resources](#enable-and-disable-resources)
- [Scope and Deduplication](#scope-and-deduplication)

## Install and Manage

> **Security:** Harn packages run with full system access. Extensions execute arbitrary code, and skills can instruct the model to perform any action including running executables. Review source code before installing third-party packages.

```bash
harn install pypi:foo-bar==1.0.0
harn install git:github.com/user/repo@v1
harn install https://github.com/user/repo  # raw URLs work too
harn install /absolute/path/to/package
harn install ./relative/path/to/package

harn remove pypi:foo-bar
harn list                     # show installed packages from settings
harn update                   # update harn and all non-pinned packages
harn update --extensions      # update all non-pinned packages only
harn update --self            # update harn only
harn update --self --force    # reinstall harn even if current
harn update pypi:foo-bar      # update one package
harn update --extension pypi:foo-bar
```

These commands manage harn packages, not the harn CLI installation. To uninstall harn itself, see [Quickstart](quickstart.md#uninstall).

By default, `install` and `remove` write to user settings (`~/.harn/agent/settings.json`). Use `-l` to write to project settings (`.harn/settings.json`) instead. Project settings can be shared with your team, and harn installs any missing packages automatically on startup.

To try a package without installing it, use `--extension` or `-e`. This installs to a temporary directory for the current run only:

```bash
harn -e pypi:foo-bar
harn -e git:github.com/user/repo
```

## Package Sources

Harn accepts three source types in settings and `harn install`.

### PyPI

```
pypi:package-name==1.2.3
pypi:package-name
```

- Versioned specs are pinned and skipped by package updates (`harn update`, `harn update --extensions`).
- User installs go under `~/.harn/agent/packages/`.
- Project installs go under `.harn/packages/`.
- Set `pipCommand` in `settings.json` to pin pip package lookup and install operations to a specific wrapper command.

Example:

```json
{
  "pipCommand": ["uv", "pip"]
}
```

### git

```
git:github.com/user/repo@v1
git:git@github.com:user/repo@v1
https://github.com/user/repo@v1
ssh://git@github.com/user/repo@v1
```

- Without `git:` prefix, only protocol URLs are accepted (`https://`, `http://`, `ssh://`, `git://`).
- With `git:` prefix, shorthand formats are accepted, including `github.com/user/repo` and `git@github.com:user/repo`.
- HTTPS and SSH URLs are both supported.
- SSH URLs use your configured SSH keys automatically (respects `~/.ssh/config`).
- For non-interactive runs (for example CI), you can set `GIT_TERMINAL_PROMPT=0` to disable credential prompts and set `GIT_SSH_COMMAND` (for example `ssh -o BatchMode=yes -o ConnectTimeout=5`) to fail fast.
- Refs are pinned tags or commits and skip package updates (`harn update`, `harn update --extensions`). Use `harn install git:host/user/repo@new-ref` to move an existing package to a new pinned ref.
- Cloned to `~/.harn/agent/git/<host>/<path>` (global) or `.harn/git/<host>/<path>` (project).
- Runs `pip install -e .` after clone, pull, or pinned ref change if `pyproject.toml` exists.

**SSH examples:**
```bash
# git@host:path shorthand (requires git: prefix)
harn install git:git@github.com:user/repo

# ssh:// protocol format
harn install ssh://git@github.com/user/repo

# With version ref
harn install git:git@github.com:user/repo@v1.0.0
```

### Local Paths

```
/absolute/path/to/package
./relative/path/to/package
```

Local paths point to files or directories on disk and are added to settings without copying. Relative paths are resolved against the settings file they appear in. If the path is a file, it loads as a single extension. If it is a directory, harn loads resources using package rules.

## Creating a Harn Package

Add a `[tool.harn]` manifest to `pyproject.toml` or use conventional directories. Include the `harn-package` keyword for discoverability.

```toml
[project]
name = "my-package"
keywords = ["harn-package"]

[tool.harn]
extensions = ["./extensions"]
skills = ["./skills"]
prompts = ["./prompts"]
themes = ["./themes"]
```

Paths are relative to the package root. Arrays support glob patterns and `!exclusions`.

### Gallery Metadata

The [package gallery](https://harn.dev/packages) displays packages tagged with `harn-package`. Add `video` or `image` fields to show a preview:

```toml
[tool.harn]
extensions = ["./extensions"]
video = "https://example.com/demo.mp4"
image = "https://example.com/screenshot.png"
```

- **video**: MP4 only. On desktop, autoplays on hover. Clicking opens a fullscreen player.
- **image**: PNG, JPEG, GIF, or WebP. Displayed as a static preview.

If both are set, video takes precedence.

## Package Structure

### Convention Directories

If no `[tool.harn]` manifest is present, harn auto-discovers resources from these directories:

- `extensions/` loads `.py` files
- `skills/` recursively finds `SKILL.md` folders and loads top-level `.md` files as skills
- `prompts/` loads `.md` files
- `themes/` loads `.json` files

## Dependencies

Third party runtime dependencies belong in `[project.dependencies]` in `pyproject.toml`. When harn installs a package from PyPI or git, it runs `pip install`, so those dependencies are installed automatically.

Harn bundles core packages for extensions and skills. If you import any of these, list them as optional or dev dependencies and do not bundle them: `harn-ai`, `harn-agent`, `harn`, `harn-tui`.

## Package Filtering

Filter what a package loads using the object form in settings:

```json
{
  "packages": [
    "pypi:simple-pkg",
    {
      "source": "pypi:my-package",
      "extensions": ["extensions/*.py", "!extensions/legacy.py"],
      "skills": [],
      "prompts": ["prompts/review.md"],
      "themes": ["+themes/legacy.json"]
    }
  ]
}
```

`+path` and `-path` are exact paths relative to the package root.

- Omit a key to load all of that type.
- Use `[]` to load none of that type.
- `!pattern` excludes matches.
- `+path` force-includes an exact path.
- `-path` force-excludes an exact path.
- Filters layer on top of the manifest. They narrow down what is already allowed.

## Enable and Disable Resources

Use `harn config` to enable or disable extensions, skills, prompt templates, and themes from installed packages and local directories. Works for both global (`~/.harn/agent`) and project (`.harn/`) scopes.

## Scope and Deduplication

Packages can appear in both global and project settings. If the same package appears in both, the project entry wins. Identity is determined by:

- PyPI: package name
- git: repository URL without ref
- local: resolved absolute path
