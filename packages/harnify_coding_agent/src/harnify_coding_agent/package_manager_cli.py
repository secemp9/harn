"""CLI front door for package-backed configuration commands."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

from harnify_coding_agent.cli.config_selector import select_config
from harnify_coding_agent.config import PACKAGE_NAME, get_agent_dir, get_update_instruction
from harnify_coding_agent.core.package_manager import DefaultPackageManager
from harnify_coding_agent.core.settings_manager import SettingsManager


PackageCommand = Literal["install", "remove", "update", "list"]


@dataclass(slots=True)
class PackageCommandOptions:
    command: PackageCommand
    source: str | None = None
    local: bool = False
    help: bool = False
    updateSelf: bool = False
    updateExtensions: bool = False


def _parse_package_command(args: list[str]) -> PackageCommandOptions | None:
    if not args:
        return None

    raw_command = args[0]
    command: PackageCommand | None = None
    if raw_command == "uninstall":
        command = "remove"
    elif raw_command in {"install", "remove", "update", "list"}:
        command = raw_command
    if command is None:
        return None

    options = PackageCommandOptions(command=command)
    remainder = args[1:]
    index = 0
    while index < len(remainder):
        arg = remainder[index]
        if arg in {"-h", "--help"}:
            options.help = True
        elif arg in {"-l", "--local"}:
            options.local = True
        elif arg == "--self":
            options.updateSelf = True
        elif arg == "--extensions":
            options.updateExtensions = True
        elif arg.startswith("-"):
            raise ValueError(f"Unknown option: {arg}")
        elif options.source is None:
            options.source = arg
        else:
            raise ValueError(f"Unexpected argument: {arg}")
        index += 1

    return options


def _print_package_command_help(command: PackageCommand) -> None:
    app_name = PACKAGE_NAME.replace("-coding-agent", "")
    if command == "install":
        print(f"Usage: {app_name} install <source> [-l|--local]")
        return
    if command == "remove":
        print(f"Usage: {app_name} remove <source> [-l|--local]")
        print(f"Alias: {app_name} uninstall <source> [-l|--local]")
        return
    if command == "update":
        print(f"Usage: {app_name} update [source|self|pi] [--self] [--extensions]")
        return
    print(f"Usage: {app_name} list")


async def handle_config_command(args: list[str]) -> int | None:
    if not args or args[0] != "config":
        return None

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    settings_manager = SettingsManager.create(cwd, agent_dir)
    package_manager = DefaultPackageManager(
        {
            "cwd": cwd,
            "agentDir": agent_dir,
            "settingsManager": settings_manager,
        }
    )
    resolved_paths = await package_manager.resolve()
    await select_config(
        {
            "resolvedPaths": resolved_paths,
            "settingsManager": settings_manager,
            "cwd": cwd,
            "agentDir": agent_dir,
        }
    )
    return 0


async def handle_package_command(args: list[str]) -> int | None:
    try:
        parsed = _parse_package_command(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if parsed is None:
        return None

    if parsed.help:
        _print_package_command_help(parsed.command)
        return 0

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    settings_manager = SettingsManager.create(cwd, agent_dir)
    package_manager = DefaultPackageManager(
        {
            "cwd": cwd,
            "agentDir": agent_dir,
            "settingsManager": settings_manager,
        }
    )

    try:
        if parsed.command == "list":
            packages = package_manager.listConfiguredPackages()
            if not packages:
                print("No packages configured.")
                return 0
            for package in packages:
                scope = "local" if package.scope == "project" else package.scope
                installed = package.installedPath or "<not installed>"
                print(f"{package.source}\t{scope}\t{installed}")
            return 0

        if parsed.command == "install":
            if not parsed.source:
                print("Error: install requires a source argument", file=sys.stderr)
                return 1
            await package_manager.installAndPersist(parsed.source, {"local": parsed.local})
            return 0

        if parsed.command == "remove":
            if not parsed.source:
                print("Error: remove requires a source argument", file=sys.stderr)
                return 1
            await package_manager.removeAndPersist(parsed.source, {"local": parsed.local})
            return 0

        if parsed.command == "update":
            target = parsed.source
            update_self = parsed.updateSelf or target in {"self", "pi"} or not parsed.updateExtensions
            update_extensions = parsed.updateExtensions or (target not in {"self", "pi"})
            if target is None:
                update_extensions = True

            if update_self:
                print(get_update_instruction(PACKAGE_NAME))
            if update_extensions:
                extension_target = None if target in {None, "self", "pi"} else target
                await package_manager.update(extension_target)
            return 0
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


handleConfigCommand = handle_config_command
handlePackageCommand = handle_package_command

__all__ = [
    "PackageCommand",
    "PackageCommandOptions",
    "handleConfigCommand",
    "handlePackageCommand",
    "handle_config_command",
    "handle_package_command",
]
