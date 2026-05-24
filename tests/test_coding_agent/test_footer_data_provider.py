from __future__ import annotations

import time
from pathlib import Path

import pytest
from harnify_coding_agent.core.footer_data_provider import FooterDataProvider


def create_plain_repo(temp_dir: Path) -> Path:
    repo_dir = temp_dir / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return repo_dir


def create_plain_reftable_repo(temp_dir: Path) -> Path:
    repo_dir = temp_dir / "repo"
    (repo_dir / ".git" / "reftable").mkdir(parents=True)
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/.invalid\n", encoding="utf-8")
    return repo_dir


def create_reftable_worktree(temp_dir: Path) -> tuple[Path, Path]:
    repo_dir = temp_dir / "repo"
    common_git_dir = repo_dir / ".git"
    git_dir = common_git_dir / "worktrees" / "src"
    worktree_dir = temp_dir / "worktree"
    reftable_dir = common_git_dir / "reftable"

    git_dir.mkdir(parents=True)
    reftable_dir.mkdir(parents=True)
    worktree_dir.mkdir(parents=True)

    (worktree_dir / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/.invalid\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (reftable_dir / "tables.list").write_text("0\n", encoding="utf-8")

    return worktree_dir, reftable_dir


def wait_for(condition, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("Timed out waiting for condition")
        time.sleep(0.01)


def test_footer_data_provider_uses_head_directly_in_regular_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = create_plain_repo(tmp_path)
    nested_dir = repo_dir / "src" / "nested"
    nested_dir.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(
        "harnify_coding_agent.core.footer_data_provider.resolve_branch_with_git_sync",
        lambda repo_dir: calls.append(repo_dir) or "main",
    )

    provider = FooterDataProvider(str(nested_dir))
    try:
        assert provider.getGitBranch() == "main"
        assert calls == []
    finally:
        provider.dispose()


def test_footer_data_provider_resolves_invalid_reftable_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = create_plain_reftable_repo(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        "harnify_coding_agent.core.footer_data_provider.resolve_branch_with_git_sync",
        lambda repo_dir: calls.append(repo_dir) or "main",
    )

    provider = FooterDataProvider(str(repo_dir))
    try:
        assert provider.getGitBranch() == "main"
        assert calls == [str(repo_dir)]
    finally:
        provider.dispose()


def test_footer_data_provider_resolves_invalid_reftable_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree_dir, _ = create_reftable_worktree(tmp_path)
    monkeypatch.setattr(
        "harnify_coding_agent.core.footer_data_provider.resolve_branch_with_git_sync",
        lambda _repo_dir: "feature/foo",
    )

    provider = FooterDataProvider(str(worktree_dir))
    try:
        assert provider.getGitBranch() == "feature/foo"
    finally:
        provider.dispose()


def test_footer_data_provider_uses_detached_for_unresolved_invalid_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = create_plain_reftable_repo(tmp_path)
    monkeypatch.setattr(
        "harnify_coding_agent.core.footer_data_provider.resolve_branch_with_git_sync",
        lambda _repo_dir: None,
    )

    provider = FooterDataProvider(str(repo_dir))
    try:
        assert provider.getGitBranch() == "detached"
    finally:
        provider.dispose()


def test_footer_data_provider_refreshes_on_reftable_changes_without_false_notifications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree_dir, reftable_dir = create_reftable_worktree(tmp_path)
    branches = iter(["main", "main", "feature/bar"])
    monkeypatch.setattr(
        FooterDataProvider,
        "WATCH_DEBOUNCE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        FooterDataProvider,
        "WATCH_POLL_INTERVAL_SECONDS",
        0.02,
    )
    monkeypatch.setattr(
        "harnify_coding_agent.core.footer_data_provider.resolve_branch_with_git_sync",
        lambda _repo_dir: next(branches),
    )

    provider = FooterDataProvider(str(worktree_dir))
    try:
        assert provider.getGitBranch() == "main"
        calls: list[str] = []
        provider.onBranchChange(lambda: calls.append("changed"))

        (reftable_dir / "tables.list").write_text("1\n", encoding="utf-8")
        wait_for(lambda: provider.getGitBranch() == "main")
        time.sleep(0.15)
        assert calls == []

        (reftable_dir / "tables.list").write_text("2\n", encoding="utf-8")
        wait_for(lambda: provider.getGitBranch() == "feature/bar")
        wait_for(lambda: len(calls) == 1)
    finally:
        provider.dispose()


def test_footer_data_provider_tracks_extension_statuses_and_provider_count(tmp_path: Path) -> None:
    repo_dir = create_plain_repo(tmp_path)
    provider = FooterDataProvider(str(repo_dir))
    try:
        provider.setExtensionStatus("lint", "ok")
        provider.setExtensionStatus("plan", "running")
        provider.setAvailableProviderCount(3)
        assert dict(provider.getExtensionStatuses()) == {"lint": "ok", "plan": "running"}
        assert provider.getAvailableProviderCount() == 3

        provider.setExtensionStatus("lint", None)
        provider.clearExtensionStatuses()
        assert dict(provider.getExtensionStatuses()) == {}
    finally:
        provider.dispose()
