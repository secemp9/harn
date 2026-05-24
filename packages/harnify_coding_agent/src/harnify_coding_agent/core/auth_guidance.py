"""Authentication guidance helpers for coding-agent errors."""

from __future__ import annotations

from pathlib import Path

UNKNOWN_PROVIDER = "unknown"


def get_provider_login_help() -> str:
    docs_path = _get_docs_path()
    return "\n".join(
        [
            "Use /login to log into a provider via OAuth or API key. See:",
            f"  {docs_path / 'providers.md'}",
            f"  {docs_path / 'models.md'}",
        ]
    )


def format_no_models_available_message() -> str:
    return f"No models available. {get_provider_login_help()}"


def format_no_model_selected_message() -> str:
    return f"No model selected.\n\n{get_provider_login_help()}\n\nThen use /model to select a model."


def format_no_api_key_found_message(provider: str) -> str:
    provider_display = "the selected model" if provider == UNKNOWN_PROVIDER else provider
    return f"No API key found for {provider_display}.\n\n{get_provider_login_help()}"


def _get_docs_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[5]
        / "important_repository_of_the_dependencies_source_code_to_read_better_than_online_docs"
        / "earendil-pi"
        / "docs",
        module_path.parents[3] / "docs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("docs")


formatNoApiKeyFoundMessage = format_no_api_key_found_message
formatNoModelSelectedMessage = format_no_model_selected_message
formatNoModelsAvailableMessage = format_no_models_available_message
getProviderLoginHelp = get_provider_login_help

__all__ = [
    "UNKNOWN_PROVIDER",
    "formatNoApiKeyFoundMessage",
    "formatNoModelSelectedMessage",
    "formatNoModelsAvailableMessage",
    "format_no_api_key_found_message",
    "format_no_model_selected_message",
    "format_no_models_available_message",
    "getProviderLoginHelp",
    "get_provider_login_help",
]
