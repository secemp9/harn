# Custom Providers

Extensions can register custom model providers via `harn.register_provider()`. This enables:

- **Proxies** - Route requests through corporate proxies or API gateways
- **Custom endpoints** - Use self-hosted or private model deployments
- **OAuth/SSO** - Add authentication flows for enterprise providers
- **Custom APIs** - Implement streaming for non-standard LLM APIs

## Table of Contents

- [Quick Reference](#quick-reference)
- [Override Existing Provider](#override-existing-provider)
- [Register New Provider](#register-new-provider)
- [Unregister Provider](#unregister-provider)
- [OAuth Support](#oauth-support)
- [Custom Streaming API](#custom-streaming-api)
- [Context Overflow Errors](#context-overflow-errors)
- [Testing Your Implementation](#testing-your-implementation)
- [Config Reference](#config-reference)
- [Model Definition Reference](#model-definition-reference)

## Quick Reference

```python
from harn_coding_agent import ExtensionAPI

def extension_factory(harn: ExtensionAPI):
    # Override baseUrl for existing provider
    harn.register_provider("anthropic", {
        "base_url": "https://proxy.example.com"
    })

    # Register new provider with models
    harn.register_provider("my-provider", {
        "name": "My Provider",
        "base_url": "https://api.example.com",
        "api_key": "MY_API_KEY",
        "api": "openai-completions",
        "models": [
            {
                "id": "my-model",
                "name": "My Model",
                "reasoning": False,
                "input": ["text", "image"],
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "context_window": 128000,
                "max_tokens": 4096,
            }
        ],
    })
```

The extension factory can also be `async`. For dynamic model discovery, fetch and register models in the factory instead of `session_start`. Harn waits for the factory before startup continues, so the provider is available during interactive startup and to `harn --list-models`.

## Override Existing Provider

The simplest use case: redirect an existing provider through a proxy.

```python
# All Anthropic requests now go through your proxy
harn.register_provider("anthropic", {
    "base_url": "https://proxy.example.com"
})

# Add custom headers to OpenAI requests
harn.register_provider("openai", {
    "headers": {
        "X-Custom-Header": "value"
    }
})

# Both base_url and headers
harn.register_provider("google", {
    "base_url": "https://ai-gateway.corp.com/google",
    "headers": {
        "X-Corp-Auth": "CORP_AUTH_TOKEN"  # env var or literal
    }
})
```

When only `base_url` and/or `headers` are provided (no `models`), all existing models for that provider are preserved with the new endpoint.

## Register New Provider

To add a completely new provider, specify `models` along with the required configuration.

If the model list comes from a remote endpoint, use an async extension factory:

```python
import httpx
from harn_coding_agent import ExtensionAPI

async def extension_factory(harn: ExtensionAPI):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models")
        payload = response.json()

    harn.register_provider("local-openai", {
        "base_url": "http://localhost:1234/v1",
        "api_key": "LOCAL_OPENAI_API_KEY",
        "api": "openai-completions",
        "models": [
            {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "context_window": model.get("context_window", 128000),
                "max_tokens": model.get("max_tokens", 4096),
            }
            for model in payload["data"]
        ],
    })
```

This registers the fetched models before startup finishes.

```python
harn.register_provider("my-llm", {
    "base_url": "https://api.my-llm.com/v1",
    "api_key": "MY_LLM_API_KEY",  # env var name or literal value
    "api": "openai-completions",   # which streaming API to use
    "models": [
        {
            "id": "my-llm-large",
            "name": "My LLM Large",
            "reasoning": True,        # supports extended thinking
            "input": ["text", "image"],
            "cost": {
                "input": 3.0,         # $/million tokens
                "output": 15.0,
                "cache_read": 0.3,
                "cache_write": 3.75,
            },
            "context_window": 200000,
            "max_tokens": 16384,
        }
    ],
})
```

When `models` is provided, it **replaces** all existing models for that provider.

## Unregister Provider

Use `harn.unregister_provider(name)` to remove a provider that was previously registered via `harn.register_provider(name, ...)`:

```python
# Register
harn.register_provider("my-llm", {
    "base_url": "https://api.my-llm.com/v1",
    "api_key": "MY_LLM_API_KEY",
    "api": "openai-completions",
    "models": [
        {
            "id": "my-llm-large",
            "name": "My LLM Large",
            "reasoning": True,
            "input": ["text", "image"],
            "cost": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
            "context_window": 200000,
            "max_tokens": 16384,
        }
    ],
})

# Later, remove it
harn.unregister_provider("my-llm")
```

Unregistering removes that provider's dynamic models, API key fallback, OAuth provider registration, and custom stream handler registrations. Any built-in models or provider behavior that were overridden are restored.

Calls made after the initial extension load phase are applied immediately, so no `/reload` is required.

### API Types

The `api` field determines which streaming implementation is used:

| API | Use for |
|-----|---------|
| `anthropic-messages` | Anthropic Claude API and compatibles |
| `openai-completions` | OpenAI Chat Completions API and compatibles |
| `openai-responses` | OpenAI Responses API |
| `azure-openai-responses` | Azure OpenAI Responses API |
| `openai-codex-responses` | OpenAI Codex Responses API |
| `mistral-conversations` | Mistral SDK Conversations/Chat streaming |
| `google-generative-ai` | Google Generative AI API |
| `google-vertex` | Google Vertex AI API |
| `bedrock-converse-stream` | Amazon Bedrock Converse API |

Most OpenAI-compatible providers work with `openai-completions`. Use model-level `thinking_level_map` for model-specific thinking levels, and `compat` for provider quirks:

```python
"models": [{
    "id": "custom-model",
    # ...
    "reasoning": True,
    "thinking_level_map": {              # map harn levels to provider values; None hides unsupported levels
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "default",
        "xhigh": "max",
    },
    "compat": {
        "supports_developer_role": False,   # use "system" instead of "developer"
        "supports_reasoning_effort": True,
        "max_tokens_field": "max_tokens",   # instead of "max_completion_tokens"
        "requires_tool_result_name": True,  # tool results need name field
        "thinking_format": "qwen",         # top-level enable_thinking: true
        "cache_control_format": "anthropic", # Anthropic-style cache_control markers
    },
}]
```

### Auth Header

If your provider expects `Authorization: Bearer <key>` but doesn't use a standard API, set `auth_header: True`:

```python
harn.register_provider("custom-api", {
    "base_url": "https://api.example.com",
    "api_key": "MY_API_KEY",
    "auth_header": True,  # adds Authorization: Bearer header
    "api": "openai-completions",
    "models": [...],
})
```

## OAuth Support

Add OAuth/SSO authentication that integrates with `/login`:

```python
from harn_ai import OAuthCredentials, OAuthLoginCallbacks

harn.register_provider("corporate-ai", {
    "base_url": "https://ai.corp.com/v1",
    "api": "openai-responses",
    "models": [...],
    "oauth": {
        "name": "Corporate AI (SSO)",

        "login": async_login_function,
        "refresh_token": async_refresh_function,
        "get_api_key": get_api_key_function,
        "modify_models": modify_models_function,  # optional
    },
})
```

After registration, users can authenticate via `/login corporate-ai`.

### OAuthLoginCallbacks

The `callbacks` object provides three ways to authenticate:

```python
class OAuthLoginCallbacks:
    def on_auth(self, url: str) -> None:
        """Open URL in browser (for OAuth redirects)."""
        ...

    def on_device_code(self, user_code: str, verification_uri: str,
                       interval_seconds: int = None, expires_in_seconds: int = None) -> None:
        """Show device code (for device authorization flow)."""
        ...

    async def on_prompt(self, message: str) -> str:
        """Prompt user for input (for manual token entry)."""
        ...

    async def on_select(self, message: str, options: list[dict]) -> str | None:
        """Show an interactive selector."""
        ...
```

### OAuthCredentials

Credentials are persisted in `~/.harn/agent/auth.json`:

```python
@dataclass
class OAuthCredentials:
    refresh: str    # Refresh token (for refresh_token())
    access: str     # Access token (returned by get_api_key())
    expires: int    # Expiration timestamp in milliseconds
```

## Custom Streaming API

For providers with non-standard APIs, implement `stream_simple`.

```python
from harn_ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    Context,
    Model,
    SimpleStreamOptions,
    calculate_cost,
    create_assistant_message_event_stream,
)
import time

def stream_my_provider(
    model: Model,
    context: Context,
    options: SimpleStreamOptions = None,
) -> AssistantMessageEventStream:
    stream = create_assistant_message_event_stream()

    async def run():
        # Initialize output message
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
                   "total_tokens": 0, "cost": {"input": 0, "output": 0,
                   "cache_read": 0, "cache_write": 0, "total": 0}},
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            # Push start event
            stream.push({"type": "start", "partial": output})

            # Make API request and process response...
            # Push content events as they arrive...

            # Push done event
            stream.push({"type": "done", "reason": output.stop_reason, "message": output})
            stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = str(error)
            stream.push({"type": "error", "reason": output.stop_reason, "error": output})
            stream.end()

    import asyncio
    asyncio.create_task(run())
    return stream
```

### Registration

Register your stream function:

```python
harn.register_provider("my-provider", {
    "base_url": "https://api.example.com",
    "api_key": "MY_API_KEY",
    "api": "my-custom-api",
    "models": [...],
    "stream_simple": stream_my_provider,
})
```

## Context Overflow Errors

When a request exceeds the model's context window, harn can recover automatically by compacting the conversation and retrying. This recovery only kicks in if harn recognizes the failure as an overflow.

Detection runs on the finalized assistant message:

- `stop_reason == "error"`
- `error_message` matches one of harn's known overflow patterns

If your provider returns overflow errors with a message harn does not recognize, normalize the error from the same extension that registers the provider. Use a `message_end` handler to rewrite the assistant message so its `error_message` starts with a phrase harn recognizes. The generic fallback `context_length_exceeded` is the safest choice.

```python
import re

MY_PROVIDER_OVERFLOW_PATTERN = re.compile(r"your provider's overflow phrase", re.IGNORECASE)

def extension_factory(harn: ExtensionAPI):
    harn.register_provider("my-provider", { ... })

    @harn.on("message_end")
    def on_message_end(event, ctx):
        message = event.message
        if message.role != "assistant":
            return
        if message.stop_reason != "error":
            return
        if message.provider != "my-provider" and (not ctx.model or ctx.model.provider != "my-provider"):
            return

        error_message = message.error_message or ""
        if "context_length_exceeded" in error_message:
            return
        if not MY_PROVIDER_OVERFLOW_PATTERN.search(error_message):
            return

        return {
            "message": {
                **message,
                "error_message": f"context_length_exceeded: {error_message}",
            }
        }
```

## Testing Your Implementation

Test your provider against the same test suites used by built-in providers:

| Test | Purpose |
|------|---------|
| `test_stream.py` | Basic streaming, text output |
| `test_tokens.py` | Token counting and usage |
| `test_abort.py` | AbortSignal handling |
| `test_empty.py` | Empty/minimal responses |
| `test_context_overflow.py` | Context window limits |
| `test_image_limits.py` | Image input handling |
| `test_unicode_surrogate.py` | Unicode edge cases |
| `test_tool_call_without_result.py` | Tool call edge cases |
| `test_image_tool_result.py` | Images in tool results |
| `test_total_tokens.py` | Total token calculation |
| `test_cross_provider_handoff.py` | Context handoff between providers |

Run tests with your provider/model pairs to verify compatibility.

## Config Reference

```python
@dataclass
class ProviderConfig:
    name: str = None          # Display name for the provider in UI such as /login
    base_url: str = None      # API endpoint URL. Required when defining models
    api_key: str = None       # API key or environment variable name
    api: str = None           # API type for streaming
    stream_simple: Callable = None  # Custom streaming implementation
    headers: dict[str, str] = None  # Custom headers to include in requests
    auth_header: bool = False       # If true, adds Authorization: Bearer header
    models: list = None             # Models to register
    oauth: dict = None              # OAuth provider for /login support
```

## Model Definition Reference

```python
@dataclass
class ProviderModelConfig:
    id: str                          # Model ID (e.g., "claude-sonnet-4-20250514")
    name: str                        # Display name (e.g., "Claude 4 Sonnet")
    reasoning: bool                  # Whether the model supports extended thinking
    input: list[str]                 # Supported input types: ["text"] or ["text", "image"]
    cost: dict                       # Cost per million tokens
    context_window: int              # Maximum context window size in tokens
    max_tokens: int                  # Maximum output tokens
    api: str = None                  # API type override for this specific model
    base_url: str = None             # API endpoint URL override for this specific model
    thinking_level_map: dict = None  # Maps harn thinking levels to provider values
    headers: dict[str, str] = None   # Custom headers for this specific model
    compat: dict = None              # Compatibility settings for the selected API
```
