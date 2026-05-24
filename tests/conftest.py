"""Shared pytest fixtures for the harnify workspace test matrix."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from harnify_agent.harness.env.local import NodeExecutionEnv
from harnify_agent.harness.session.jsonl_repo import JsonlSessionRepo
from harnify_ai.providers.faux import FauxProviderRegistration, register_faux_provider


@dataclass(slots=True)
class SessionScaffold:
    cwd: Path
    env: NodeExecutionEnv
    sessions_root: Path
    repo: JsonlSessionRepo


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def node_env(workspace_root: Path) -> NodeExecutionEnv:
    return NodeExecutionEnv({"cwd": str(workspace_root)})


@pytest.fixture
def session_scaffold(workspace_root: Path, node_env: NodeExecutionEnv) -> SessionScaffold:
    sessions_root = workspace_root / ".sessions"
    return SessionScaffold(
        cwd=workspace_root,
        env=node_env,
        sessions_root=sessions_root,
        repo=JsonlSessionRepo({"fs": node_env, "sessionsRoot": str(sessions_root)}),
    )


@pytest.fixture
def jsonl_session_factory(session_scaffold: SessionScaffold) -> Callable[..., Awaitable[Any]]:
    async def factory(**overrides: Any) -> Any:
        options = {"cwd": str(session_scaffold.cwd)}
        options.update(overrides)
        return await session_scaffold.repo.create(options)

    return factory


@pytest.fixture
def faux_provider_factory() -> Iterator[Callable[[dict[str, Any] | None], FauxProviderRegistration]]:
    registrations: list[FauxProviderRegistration] = []

    def factory(options: dict[str, Any] | None = None) -> FauxProviderRegistration:
        registration = register_faux_provider(dict(options or {}))
        registrations.append(registration)
        return registration

    yield factory

    while registrations:
        registrations.pop().unregister()
