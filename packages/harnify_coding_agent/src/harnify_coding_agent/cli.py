"""Top-level CLI entry point."""

from __future__ import annotations

import asyncio
import os
import sys


async def _invoke_main(argv: list[str]) -> int:
    from harnify_coding_agent.main import main as async_main

    return await async_main(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ["HARNIFY_CODING_AGENT"] = "true"
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    return asyncio.run(_invoke_main(resolved_argv))


run = main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run"]
