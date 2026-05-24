"""Environment-based credential discovery for provider auth."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshotai-cn": "MOONSHOT_API_KEY",
    "huggingface": "HF_TOKEN",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
    "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
    "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
    "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
}


@lru_cache(maxsize=1)
def _has_vertex_adc_credentials() -> bool:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        return Path(credentials_path).exists()

    return Path.home().joinpath(".config", "gcloud", "application_default_credentials.json").exists()


def _get_api_key_env_vars(provider: str) -> tuple[str, ...] | None:
    if provider == "github-copilot":
        return ("COPILOT_GITHUB_TOKEN",)

    if provider == "anthropic":
        return ("ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY")

    env_var = _ENV_MAP.get(provider)
    return (env_var,) if env_var else None


def find_env_keys(provider: str) -> list[str] | None:
    env_vars = _get_api_key_env_vars(provider)
    if not env_vars:
        return None

    found = [env_var for env_var in env_vars if os.environ.get(env_var)]
    return found or None


def get_env_api_key(provider: str) -> str | None:
    env_keys = find_env_keys(provider)
    if env_keys and env_keys[0]:
        return os.environ.get(env_keys[0])

    if provider == "google-vertex":
        has_project = bool(os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT"))
        has_location = bool(os.environ.get("GOOGLE_CLOUD_LOCATION"))
        if _has_vertex_adc_credentials() and has_project and has_location:
            return "<authenticated>"

    if provider == "amazon-bedrock":
        if (
            os.environ.get("AWS_PROFILE")
            or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
        ):
            return "<authenticated>"

    return None


findEnvKeys = find_env_keys
getEnvApiKey = get_env_api_key
