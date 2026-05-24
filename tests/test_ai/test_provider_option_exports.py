from __future__ import annotations

from typing import get_args

from harnify_ai import (
    AnthropicEffort,
    AnthropicOptions,
    AnthropicThinkingDisplay,
    AzureOpenAIResponsesOptions,
    BedrockOptions,
    BedrockThinkingDisplay,
    GoogleOptions,
    GoogleThinkingLevel,
    GoogleVertexOptions,
    MistralOptions,
    OpenAICompletionsOptions,
    OpenAICodexResponsesOptions,
    OpenAICodexWebSocketDebugStats,
    OpenAIResponsesOptions,
)


def test_harnify_ai_exports_provider_option_types() -> None:
    assert AnthropicOptions(apiKey="test", thinkingEnabled=True)["thinkingEnabled"] is True
    assert BedrockOptions(region="us-east-1", bearerToken="token")["region"] == "us-east-1"
    assert AzureOpenAIResponsesOptions(azureDeploymentName="dep")["azureDeploymentName"] == "dep"
    assert GoogleOptions(toolChoice="auto")["toolChoice"] == "auto"
    assert GoogleVertexOptions(project="demo-project", location="europe-west1")["project"] == "demo-project"
    assert MistralOptions(promptMode="reasoning")["promptMode"] == "reasoning"
    assert OpenAICompletionsOptions(reasoningEffort="high")["reasoningEffort"] == "high"
    assert OpenAIResponsesOptions(serviceTier="priority")["serviceTier"] == "priority"
    assert OpenAICodexResponsesOptions(textVerbosity="high")["textVerbosity"] == "high"
    assert OpenAICodexWebSocketDebugStats(requests=1)["requests"] == 1

    assert get_args(AnthropicEffort) == ("low", "medium", "high", "xhigh", "max")
    assert get_args(AnthropicThinkingDisplay) == ("summarized", "omitted")
    assert get_args(BedrockThinkingDisplay) == ("summarized", "omitted")
    assert get_args(GoogleThinkingLevel) == (
        "THINKING_LEVEL_UNSPECIFIED",
        "MINIMAL",
        "LOW",
        "MEDIUM",
        "HIGH",
    )
