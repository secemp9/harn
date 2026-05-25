"""CLI front door for package-backed configuration commands."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import Literal

from harnify_coding_agent.cli.config_selector import select_config
from harnify_coding_agent.config import (
    APP_NAME,
    PACKAGE_NAME,
    VERSION,
    SelfUpdateCommand,
    get_agent_dir,
    get_package_dir,
    get_self_update_command,
    get_self_update_unavailable_instruction,
)
from harnify_coding_agent.core.package_manager import DefaultPackageManager
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.utils.child_process import spawn_process
from harnify_coding_agent.utils.version_check import get_latest_pi_release, is_newer_package_version
from harnify_coding_agent.utils.windows_self_update import (
    cleanup_windows_self_update_quarantine,
    quarantine_windows_native_dependencies,
)


PackageCommand = Literal["install", "remove", "update", "list"]
UpdateTargetType = Literal["all", "self", "extensions"]


@dataclass(slots=True)
class UpdateTarget:
    type: UpdateTargetType
    source: str | None = None


@dataclass(slots=True)
class PackageCommandOptions:
    command: PackageCommand
    source: str | None = None
    updateTarget: UpdateTarget | None = None
    local: bool = False
    force: bool = False
    help: bool = False
    invalidOption: str | None = None
    invalidArgument: str | None = None
    missingOptionValue: str | None = None
    conflictingOptions: str | None = None


@dataclass(slots=True)
class _SelfUpdatePlan:
    packageName: str
    shouldRun: bool
    note: str | None = None


_command_exit_code = 0


def _set_command_exit_code(code: int) -> None:
    global _command_exit_code
    _command_exit_code = code


def _take_command_exit_code() -> int:
    global _command_exit_code
    code = _command_exit_code
    _command_exit_code = 0
    return code


def _get_package_command_usage(command: PackageCommand) -> str:
    match command:
        case "install":
            return f"{APP_NAME} install <source> [-l]"
        case "remove":
            return f"{APP_NAME} remove <source> [-l]"
        case "update":
            return f"{APP_NAME} update [source|self|pi] [--self] [--extensions] [--extension <source>] [--force]"
        case "list":
            return f"{APP_NAME} list"


def _report_settings_errors(settings_manager: SettingsManager, context: str) -> None:
    for settings_error in settings_manager.drainErrors():
        error = settings_error.error
        print(
            f"Warning ({context}, {settings_error.scope} settings): {error}",
            file=sys.stderr,
        )
        if error.__traceback__ is not None:
            print(
                "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip("\n"),
                file=sys.stderr,
            )


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
    self_flag = False
    extensions_flag = False
    extension_flag_source: str | None = None
    index = 0
    while index < len(remainder):
        arg = remainder[index]
        if arg in {"-h", "--help"}:
            options.help = True
        elif arg in {"-l", "--local"}:
            if command in {"install", "remove"}:
                options.local = True
            elif options.invalidOption is None:
                options.invalidOption = arg
        elif arg == "--self":
            if command == "update":
                self_flag = True
            elif options.invalidOption is None:
                options.invalidOption = arg
        elif arg == "--extensions":
            if command == "update":
                extensions_flag = True
            elif options.invalidOption is None:
                options.invalidOption = arg
        elif arg == "--force":
            if command == "update":
                options.force = True
            elif options.invalidOption is None:
                options.invalidOption = arg
        elif arg == "--extension":
            if command != "update":
                if options.invalidOption is None:
                    options.invalidOption = arg
            else:
                value = remainder[index + 1] if index + 1 < len(remainder) else None
                if value is None or value.startswith("-"):
                    if options.missingOptionValue is None:
                        options.missingOptionValue = arg
                elif extension_flag_source is not None:
                    if options.conflictingOptions is None:
                        options.conflictingOptions = "--extension can only be provided once"
                    index += 1
                else:
                    extension_flag_source = value
                    index += 1
        elif arg.startswith("-"):
            if options.invalidOption is None:
                options.invalidOption = arg
        elif options.source is None:
            options.source = arg
        elif options.invalidArgument is None:
            options.invalidArgument = arg
        index += 1

    if command == "update":
        if extension_flag_source is not None:
            if self_flag or extensions_flag:
                options.conflictingOptions = (
                    options.conflictingOptions
                    or "--extension cannot be combined with --self or --extensions"
                )
            if options.source:
                options.conflictingOptions = (
                    options.conflictingOptions
                    or "--extension cannot be combined with a positional source"
                )
            options.updateTarget = UpdateTarget(type="extensions", source=extension_flag_source)
        elif options.source:
            source_is_self = options.source in {"self", "pi"}
            if source_is_self:
                options.updateTarget = UpdateTarget(type="all" if extensions_flag else "self")
            else:
                if extensions_flag or self_flag:
                    options.conflictingOptions = (
                        options.conflictingOptions
                        or "positional update targets cannot be combined with --self or --extensions"
                    )
                options.updateTarget = UpdateTarget(type="extensions", source=options.source)
        elif self_flag and extensions_flag:
            options.updateTarget = UpdateTarget(type="all")
        elif self_flag:
            options.updateTarget = UpdateTarget(type="self")
        elif extensions_flag:
            options.updateTarget = UpdateTarget(type="extensions")
        else:
            options.updateTarget = UpdateTarget(type="all")

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


async def handle_config_command(args: list[str]) -> bool | None:
    if not args or args[0] != "config":
        return None
    _set_command_exit_code(0)

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    settings_manager = SettingsManager.create(cwd, agent_dir)
    _report_settings_errors(settings_manager, "config command")
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
    return True


async def handle_package_command(args: list[str]) -> bool | None:
    parsed = _parse_package_command(args)
    if parsed is None:
        return None
    _set_command_exit_code(0)

    if parsed.help:
        _print_package_command_help(parsed.command)
        return True

    if parsed.invalidOption:
        print(f'Unknown option {parsed.invalidOption} for "{parsed.command}".', file=sys.stderr)
        print(f'Use "{APP_NAME} --help" or "{_get_package_command_usage(parsed.command)}".', file=sys.stderr)
        _set_command_exit_code(1)
        return True

    if parsed.missingOptionValue:
        print(f"Missing value for {parsed.missingOptionValue}.", file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(parsed.command)}", file=sys.stderr)
        _set_command_exit_code(1)
        return True

    if parsed.invalidArgument:
        print(f"Unexpected argument {parsed.invalidArgument}.", file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(parsed.command)}", file=sys.stderr)
        _set_command_exit_code(1)
        return True

    if parsed.conflictingOptions:
        print(parsed.conflictingOptions, file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(parsed.command)}", file=sys.stderr)
        _set_command_exit_code(1)
        return True

    if parsed.command in {"install", "remove"} and not parsed.source:
        print(f"Missing {parsed.command} source.", file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(parsed.command)}", file=sys.stderr)
        _set_command_exit_code(1)
        return True

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    settings_manager = SettingsManager.create(cwd, agent_dir)
    _report_settings_errors(settings_manager, "package command")
    package_manager = DefaultPackageManager(
        {
            "cwd": cwd,
            "agentDir": agent_dir,
            "settingsManager": settings_manager,
        }
    )
    set_progress_callback = getattr(package_manager, "setProgressCallback", None)
    if callable(set_progress_callback):
        set_progress_callback(
            lambda event: (
                print(event.message)
                if getattr(event, "type", None) == "start" and getattr(event, "message", None)
                else None
            )
        )

    try:
        if parsed.command == "list":
            configured_packages = package_manager.listConfiguredPackages()
            user_packages = [package for package in configured_packages if package.scope == "user"]
            project_packages = [package for package in configured_packages if package.scope == "project"]

            if not configured_packages:
                print("No packages installed.")
                return True

            def format_package(package: object) -> None:
                display = f"{package.source} (filtered)" if package.filtered else package.source
                print(f"  {display}")
                if package.installedPath:
                    print(f"    {package.installedPath}")

            if user_packages:
                print("User packages:")
                for package in user_packages:
                    format_package(package)

            if project_packages:
                if user_packages:
                    print()
                print("Project packages:")
                for package in project_packages:
                    format_package(package)
            return True

        if parsed.command == "install":
            await package_manager.installAndPersist(parsed.source, {"local": parsed.local})
            print(f"Installed {parsed.source}")
            return True

        if parsed.command == "remove":
            removed = await package_manager.removeAndPersist(parsed.source, {"local": parsed.local})
            if not removed:
                print(f"No matching package found for {parsed.source}", file=sys.stderr)
                _set_command_exit_code(1)
                return True
            print(f"Removed {parsed.source}")
            return True

        if parsed.command == "update":
            target = parsed.updateTarget or UpdateTarget(type="all")
            update_self = target.type in {"all", "self"}
            update_extensions = target.type in {"all", "extensions"}

            if update_self:
                print(get_update_instruction(PACKAGE_NAME))
            if update_extensions:
                extension_target = target.source if target.type == "extensions" else None
                await package_manager.update(extension_target)
            return True
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        _set_command_exit_code(1)
        return True

    return True


handleConfigCommand = handle_config_command
handlePackageCommand = handle_package_command

__all__ = [
    "PackageCommand",
    "handleConfigCommand",
    "handlePackageCommand",
]
