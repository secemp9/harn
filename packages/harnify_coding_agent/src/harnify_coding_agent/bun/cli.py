"""Bun-style wrapper entrypoint for the coding-agent CLI."""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from collections.abc import Callable

from harnify_coding_agent.bun.restore_sandbox_env import restore_sandbox_env
from harnify_coding_agent.config import APP_NAME


def _set_process_title(title: str) -> None:
    # Keep argv visible to Python tooling even if no native process-title setter is installed.
    sys.argv[0] = title

    try:
        import setproctitle  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        setproctitle.setproctitle(title)
    except Exception:
        return


def _suppress_runtime_warnings() -> None:
    warnings.showwarning = lambda *args, **kwargs: None


def _import_register_bedrock_module() -> None:
    importlib.import_module("harnify_coding_agent.bun.register_bedrock")


def _load_cli_main() -> Callable[[list[str] | None], int]:
    from harnify_coding_agent.cli import main as cli_main

    return cli_main


def main(argv: list[str] | None = None) -> int:
    _set_process_title(APP_NAME)
    _suppress_runtime_warnings()
    restore_sandbox_env()
    _import_register_bedrock_module()

    os.environ["PI_CODING_AGENT"] = "true"
    return _load_cli_main()(argv)


run = main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__: list[str] = []

