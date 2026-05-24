"""Eagerly register the Bedrock provider module for the Bun-style wrapper."""

from __future__ import annotations

from harnify_ai.bedrock_provider import bedrockProviderModule
from harnify_ai.providers.register_builtins import setBedrockProviderModule


def register_bedrock() -> None:
    setBedrockProviderModule(bedrockProviderModule)


registerBedrock = register_bedrock
register_bedrock()

__all__: list[str] = []

