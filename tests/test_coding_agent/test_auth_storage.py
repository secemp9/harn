from __future__ import annotations

from pathlib import Path

import pytest
from harnify_ai.utils.oauth import registerOAuthProvider, resetOAuthProviders
from harnify_ai.utils.oauth.types import OAuthCredentials
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.resolve_config_value import clearConfigValueCache


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    clearConfigValueCache()
    resetOAuthProviders()


@pytest.mark.asyncio
async def test_runtime_override_takes_priority() -> None:
    storage = AuthStorage.inMemory({"anthropic": {"type": "api_key", "key": "stored-value"}})
    storage.setRuntimeApiKey("anthropic", "runtime-value")

    assert await storage.getApiKey("anthropic") == "runtime-value"


@pytest.mark.asyncio
async def test_environment_named_api_keys_are_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AUTH_KEY_CACHE", "first-value")
    storage = AuthStorage.inMemory({"anthropic": {"type": "api_key", "key": "TEST_AUTH_KEY_CACHE"}})

    assert await storage.getApiKey("anthropic") == "first-value"

    monkeypatch.setenv("TEST_AUTH_KEY_CACHE", "second-value")
    assert await storage.getApiKey("anthropic") == "second-value"


@pytest.mark.asyncio
async def test_command_api_keys_are_cached_until_cleared(tmp_path: Path) -> None:
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0", encoding="utf-8")
    command = (
        "!sh -c 'count=$(cat "
        f"\"{counter_file.as_posix()}\""
        "); echo $((count + 1)) > "
        f"\"{counter_file.as_posix()}\""
        "; echo key-value'"
    )
    storage = AuthStorage.inMemory({"anthropic": {"type": "api_key", "key": command}})

    assert await storage.getApiKey("anthropic") == "key-value"
    assert await storage.getApiKey("anthropic") == "key-value"
    assert counter_file.read_text(encoding="utf-8").strip() == "1"

    clearConfigValueCache()
    assert await storage.getApiKey("anthropic") == "key-value"
    assert counter_file.read_text(encoding="utf-8").strip() == "2"


@pytest.mark.asyncio
async def test_fallback_resolver_and_include_fallback_flag() -> None:
    storage = AuthStorage.inMemory()
    storage.setFallbackResolver(lambda provider: "fallback-key" if provider == "demo" else None)

    assert await storage.getApiKey("demo") == "fallback-key"
    assert await storage.getApiKey("demo", {"includeFallback": False}) is None
    assert storage.getAuthStatus("demo").source == "fallback"


@pytest.mark.asyncio
async def test_expired_oauth_credentials_are_refreshed_and_persisted() -> None:
    provider_id = "test-oauth-provider"

    class FakeOAuthProvider:
        id = provider_id
        name = "Test OAuth Provider"
        usesCallbackServer = False

        async def login(self, callbacks):  # pragma: no cover - not used in this test
            raise RuntimeError("not used")

        async def refreshToken(self, credentials: OAuthCredentials) -> OAuthCredentials:
            return OAuthCredentials(refresh=credentials.refresh, access="refreshed-token", expires=9_999_999_999_999)

        def getApiKey(self, credentials: OAuthCredentials) -> str:
            return f"Bearer {credentials.access}"

        def modifyModels(self, models, credentials):  # pragma: no cover - not used in this test
            return models

    registerOAuthProvider(FakeOAuthProvider())
    storage = AuthStorage.inMemory(
        {
            provider_id: {
                "type": "oauth",
                "refresh": "refresh-token",
                "access": "expired-token",
                "expires": 0,
            }
        }
    )

    assert await storage.getApiKey(provider_id) == "Bearer refreshed-token"
    stored = storage.get(provider_id)
    assert stored is not None
    assert stored["access"] == "refreshed-token"
