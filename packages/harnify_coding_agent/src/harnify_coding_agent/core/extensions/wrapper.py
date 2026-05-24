"""Wrappers for extension-registered tools."""

from __future__ import annotations

from harnify_agent.types import AgentTool

from harnify_coding_agent.core.extensions.runner import ExtensionRunner
from harnify_coding_agent.core.extensions.types import RegisteredTool
from harnify_coding_agent.core.tools.tool_definition_wrapper import wrap_tool_definition, wrap_tool_definitions


def wrap_registered_tool(registered_tool: RegisteredTool, runner: ExtensionRunner) -> AgentTool:
    return wrap_tool_definition(registered_tool.definition, lambda: runner.create_context())


def wrap_registered_tools(registered_tools: list[RegisteredTool], runner: ExtensionRunner) -> list[AgentTool]:
    return wrap_tool_definitions(
        [registered_tool.definition for registered_tool in registered_tools],
        lambda: runner.create_context(),
    )


wrapRegisteredTool = wrap_registered_tool
wrapRegisteredTools = wrap_registered_tools

__all__ = [
    "wrapRegisteredTool",
    "wrapRegisteredTools",
    "wrap_registered_tool",
    "wrap_registered_tools",
]
