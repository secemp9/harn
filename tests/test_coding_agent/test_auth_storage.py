from __future__ import annotations

import json
from pathlib import Path

import pytest
from harnify_ai.utils.oauth import registerOAuthProvider, resetOAuthProviders
from harnify_ai.utils.oauth.types import OAuthCredentials
from harnify_coding_agent.core.auth_storage import AuthStorage, InMemoryAuthStorageBackend
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


def test_reload_records_parse_errors_and_drain_errors_clears_buffer(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"anthropic": {"type": "api_key", "key": "anthropic-key"}}', encoding="utf-8")

    storage = AuthStorage.create(str(auth_path))
    auth_path.write_text("{invalid-json", encoding="utf-8")

    storage.reload()

    assert storage.get("anthropic") == {"type": "api_key", "key": "anthropic-key"}

    first_drain = storage.drainErrors()
    assert first_drain
    assert isinstance(first_drain[0], Exception)
    assert storage.drainErrors() == []


def test_set_and_remove_preserve_unrelated_external_edits(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        '{"anthropic": {"type": "api_key", "key": "old-anthropic"}, "openai": {"type": "api_key", "key": "openai-key"}}',
        encoding="utf-8",
    )
    storage = AuthStorage.create(str(auth_path))

    auth_path.write_text(
        '{"anthropic": {"type": "api_key", "key": "old-anthropic"}, "openai": {"type": "api_key", "key": "openai-key"}, "google": {"type": "api_key", "key": "google-key"}}',
        encoding="utf-8",
    )
    storage.set("anthropic", {"type": "api_key", "key": "new-anthropic"})

    updated = json.loads(auth_path.read_text(encoding="utf-8"))
    assert updated["anthropic"]["key"] == "new-anthropic"
    assert updated["openai"]["key"] == "openai-key"
    assert updated["google"]["key"] == "google-key"

    auth_path.write_text(
        '{"anthropic": {"type": "api_key", "key": "new-anthropic"}, "openai": {"type": "api_key", "key": "openai-key"}, "google": {"type": "api_key", "key": "google-key"}}',
        encoding="utf-8",
    )
    storage.remove("anthropic")

    updated = json.loads(auth_path.read_text(encoding="utf-8"))
    assert "anthropic" not in updated
    assert updated["openai"]["key"] == "openai-key"
    assert updated["google"]["key"] == "google-key"


def test_non_object_json_storage_uses_ts_style_object_coercion(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('["zero", {"type": "api_key", "key": "array-key"}]', encoding="utf-8")

    storage = AuthStorage.create(str(auth_path))

    assert storage.list() == ["0", "1"]
    assert storage.has("1") is True
    assert storage.get("1") == {"type": "api_key", "key": "array-key"}
    assert storage.getAll() == {
        "0": "zero",
        "1": {"type": "api_key", "key": "array-key"},
    }

    storage.set("anthropic", {"type": "api_key", "key": "new-anthropic"})

    assert json.loads(auth_path.read_text(encoding="utf-8")) == {
        "0": "zero",
        "1": {"type": "api_key", "key": "array-key"},
        "anthropic": {"type": "api_key", "key": "new-anthropic"},
    }


@pytest.mark.asyncio
async def test_get_and_get_all_match_ts_reference_semantics() -> None:
    storage = AuthStorage.inMemory({"anthropic": {"type": "api_key", "key": "stored-value"}})

    credential = storage.get("anthropic")
    assert credential is not None
    credential["key"] = "mutated-via-get"
    assert await storage.getApiKey("anthropic") == "mutated-via-get"

    all_credentials = storage.getAll()
    all_credentials["anthropic"]["key"] = "mutated-via-get-all"
    assert await storage.getApiKey("anthropic") == "mutated-via-get-all"


@pytest.mark.asyncio
async def test_refresh_failure_returns_none_then_allows_later_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_id = "retry-oauth-provider"

    class FakeOAuthProvider:
        id = provider_id
        name = "Retry OAuth Provider"
        usesCallbackServer = False

        async def login(self, callbacks):  # pragma: no cover - not used in this test
            raise RuntimeError("not used")

        async def refreshToken(self, credentials: OAuthCredentials) -> OAuthCredentials:
            return OAuthCredentials(refresh=credentials.refresh, access="refreshed-token", expires=9_999_999_999_999)

        def getApiKey(self, credentials: OAuthCredentials) -> str:
            return f"Bearer {credentials.access}"

    registerOAuthProvider(FakeOAuthProvider())
    backend = InMemoryAuthStorageBackend()
    backend.value = (
        '{\n'
        f'  "{provider_id}": {{\n'
        '    "type": "oauth",\n'
        '    "refresh": "refresh-token",\n'
        '    "access": "expired-token",\n'
        '    "expires": 0\n'
        "  }\n"
        "}"
    )
    storage = AuthStorage.fromStorage(backend)
    real_refresh = storage.refreshOAuthTokenWithLock
    attempts = 0

    async def flaky_refresh(provider: str) -> dict[str, object] | None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("lock compromised")
        return await real_refresh(provider)

    monkeypatch.setattr(storage, "refreshOAuthTokenWithLock", flaky_refresh)

    assert await storage.getApiKey(provider_id) is None
    assert await storage.getApiKey(provider_id) == "Bearer refreshed-token"


@pytest.mark.asyncio
async def test_public_exports_match_ts_surface() -> None:
    from harnify_coding_agent.core import auth_storage

    assert auth_storage.__all__ == [
        "ApiKeyCredential",
        "AuthCredential",
        "AuthStatus",
        "AuthStorage",
        "AuthStorageBackend",
        "AuthStorageData",
        "FileAuthStorageBackend",
        "InMemoryAuthStorageBackend",
        "OAuthCredential",
    ]
