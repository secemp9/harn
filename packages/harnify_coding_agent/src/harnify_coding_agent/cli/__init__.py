"""CLI helpers and the installed console entry point for the coding-agent package."""

from __future__ import annotations

import asyncio
import os
import sys
import warnings

from harnify_coding_agent.cli.args import ArgDiagnostic, Args, parse_args, print_help
from harnify_coding_agent.cli.file_processor import (
    ProcessedFiles,
    ProcessFileOptions,
    process_file_arguments,
)
from harnify_coding_agent.cli.initial_message import InitialMessageResult, build_initial_message
from harnify_coding_agent.cli.list_models import list_models
from harnify_coding_agent.config import APP_NAME
from harnify_coding_agent.core.http_dispatcher import configure_http_dispatcher


def _set_process_title(title: str) -> None:
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


async def _invoke_main(argv: list[str]) -> int:
    from harnify_coding_agent.main import main as async_main

    return await async_main(argv)


def main(argv: list[str] | None = None) -> int:
    _set_process_title(APP_NAME)
    os.environ["PI_CODING_AGENT"] = "true"
    _suppress_runtime_warnings()
    configure_http_dispatcher()
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    return asyncio.run(_invoke_main(resolved_argv))


run = main

__all__ = [
    "ArgDiagnostic",
    "Args",
    "InitialMessageResult",
    "ProcessFileOptions",
    "ProcessedFiles",
    "build_initial_message",
    "list_models",
    "main",
    "parse_args",
    "print_help",
    "process_file_arguments",
    "run",
]
