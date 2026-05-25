from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest
from harnify_agent.agent import AbortController
import harnify_coding_agent.utils.changelog as changelog_module
import harnify_coding_agent.utils.child_process as child_process_module
import harnify_coding_agent.utils.frontmatter as frontmatter_module
from harnify_coding_agent.utils.changelog import ChangelogEntry, compare_versions, get_new_entries, parse_changelog
from harnify_coding_agent.utils.child_process import spawn_process, spawn_process_sync, wait_for_child_process
from harnify_coding_agent.utils.frontmatter import parse_frontmatter, strip_frontmatter
from harnify_coding_agent.utils.fs_watch import close_watcher, watch_with_error_handler
from harnify_coding_agent.utils.git import parse_git_url
from harnify_coding_agent.utils.html import decode_html_entity, decode_html_entity_at
from harnify_coding_agent.utils.shell import get_shell_env
from harnify_coding_agent.utils.sleep import sleep
from ruamel.yaml import YAMLError


def test_parse_changelog_and_get_new_entries(tmp_path: Path) -> None:
    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(
        "# Changelog\n\n"
        "## [1.2.0] - 2026-01-01\n"
        "- newer\n\n"
        "## [1.0.5] - 2025-12-31\n"
        "- older\n",
        encoding="utf-8",
    )

    entries = parse_changelog(str(changelog_path))

    assert entries == [
        ChangelogEntry(major=1, minor=2, patch=0, content="## [1.2.0] - 2026-01-01\n- newer"),
        ChangelogEntry(major=1, minor=0, patch=5, content="## [1.0.5] - 2025-12-31\n- older"),
    ]
    assert compare_versions(entries[0], entries[1]) > 0
    assert get_new_entries(entries, "1.1.9") == [entries[0]]
    assert parse_changelog(str(tmp_path / "missing.md")) == []


def test_changelog_module_exports_match_ts_surface() -> None:
    assert changelog_module.__all__ == [
        "ChangelogEntry",
        "compareVersions",
        "getChangelogPath",
        "getNewEntries",
        "parseChangelog",
    ]


def test_child_process_module_exports_match_ts_surface() -> None:
    assert child_process_module.__all__ == [
        "spawnProcess",
        "spawnProcessSync",
        "waitForChildProcess",
    ]


def test_parse_frontmatter_matches_upstream_contract() -> None:
    parsed = parse_frontmatter('---\nname: "skill-name"\ndescription: \'A desc\'\nfoo-bar: value\n---\n\nBody text')
    assert parsed.frontmatter["name"] == "skill-name"
    assert parsed.frontmatter["description"] == "A desc"
    assert parsed.frontmatter["foo-bar"] == "value"
    assert parsed.body == "Body text"

    assert parse_frontmatter("---\r\nname: test\r\n---\r\nLine one\r\nLine two").body == "Line one\nLine two"

    multiline = parse_frontmatter("---\ndescription: |\n  Line one\n  Line two\n---\n\nBody")
    assert multiline.frontmatter["description"] == "Line one\nLine two\n"
    assert multiline.body == "Body"

    no_frontmatter = parse_frontmatter("Just text\nsecond line")
    assert no_frontmatter.frontmatter == {}
    assert no_frontmatter.body == "Just text\nsecond line"

    missing_end = parse_frontmatter("---\nname: test\nBody without terminator")
    assert missing_end.body == "---\nname: test\nBody without terminator"

    assert parse_frontmatter("---\n# just a comment\n---\nBody").frontmatter == {}
    assert strip_frontmatter("---\nkey: value\n---\n\nBody\n") == "Body"

    with pytest.raises(YAMLError):
        parse_frontmatter("---\nfoo: [bar\n---\nBody")


def test_frontmatter_module_exports_match_ts_surface() -> None:
    assert frontmatter_module.__all__ == ["parseFrontmatter", "stripFrontmatter"]


def test_parse_git_url_matches_upstream_contract() -> None:
    https_result = parse_git_url("https://github.com/user/repo")
    assert https_result is not None
    assert https_result.host == "github.com"
    assert https_result.path == "user/repo"
    assert https_result.repo == "https://github.com/user/repo"

    ssh_result = parse_git_url("ssh://git@github.com/user/repo")
    assert ssh_result is not None
    assert ssh_result.host == "github.com"
    assert ssh_result.path == "user/repo"
    assert ssh_result.repo == "ssh://git@github.com/user/repo"

    tagged = parse_git_url("https://github.com/user/repo@v1.0.0")
    assert tagged is not None
    assert tagged.ref == "v1.0.0"
    assert tagged.repo == "https://github.com/user/repo"

    scp = parse_git_url("git:git@github.com:user/repo")
    assert scp is not None
    assert scp.host == "github.com"
    assert scp.path == "user/repo"
    assert scp.repo == "git@github.com:user/repo"

    shorthand = parse_git_url("git:github.com/user/repo")
    assert shorthand is not None
    assert shorthand.repo == "https://github.com/user/repo"
    assert shorthand.host == "github.com"

    hosted_shorthand = parse_git_url("git:github:user/repo#main")
    assert hosted_shorthand is not None
    assert hosted_shorthand.repo == "https://github.com/user/repo"
    assert hosted_shorthand.host == "github.com"
    assert hosted_shorthand.path == "user/repo"
    assert hosted_shorthand.ref == "main"

    with_ref = parse_git_url("git:git@github.com:user/repo@v1.0.0")
    assert with_ref is not None
    assert with_ref.ref == "v1.0.0"
    assert with_ref.pinned is True

    assert parse_git_url("git@github.com:user/repo") is None
    assert parse_git_url("github.com/user/repo") is None
    assert parse_git_url("user/repo") is None


def test_decode_html_entity_helpers() -> None:
    assert decode_html_entity("amp") == "&"
    assert decode_html_entity("#35") == "#"
    assert decode_html_entity("#x41") == "A"
    assert decode_html_entity("bogus") is None
    assert decode_html_entity("#x110000") is None

    decoded = decode_html_entity_at("before &quot; after", 7)
    assert decoded is not None
    assert decoded.text == '"'
    assert decoded.length == 6
    assert decode_html_entity_at("before &this_entity_name_is_too_long; after", 7) is None


def test_get_shell_env_prepends_bin_dir_without_duplicating_and_preserves_path_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("path", "/usr/bin")
    monkeypatch.setattr("harnify_coding_agent.config.get_bin_dir", lambda: "/tmp/harnify-bin")

    env = get_shell_env()
    assert env["path"] == "/tmp/harnify-bin:/usr/bin"

    monkeypatch.setenv("path", "/tmp/harnify-bin:/usr/bin")
    env_with_existing_bin = get_shell_env()
    assert env_with_existing_bin["path"] == "/tmp/harnify-bin:/usr/bin"


@pytest.mark.asyncio
async def test_sleep_respects_abort_signal() -> None:
    controller = AbortController()
    with pytest.raises(RuntimeError, match="Aborted"):
        controller.abort()
        await sleep(1, controller.signal)

    controller = AbortController()
    task = asyncio.create_task(sleep(500, controller.signal))
    await asyncio.sleep(0.05)
    controller.abort()
    with pytest.raises(RuntimeError, match="Aborted"):
        await task


def test_watch_with_error_handler_handles_missing_paths_and_changes(tmp_path: Path) -> None:
    missing_errors: list[str] = []
    watcher = watch_with_error_handler(
        str(tmp_path / "missing.txt"),
        lambda *_args: None,
        lambda: missing_errors.append("missing"),
    )
    assert watcher is None
    assert missing_errors == ["missing"]

    file_path = tmp_path / "watched.txt"
    file_path.write_text("a", encoding="utf-8")
    changes: list[tuple[str, str | None]] = []
    event = threading.Event()
    watcher = watch_with_error_handler(
        str(file_path),
        lambda event_type, filename=None: (changes.append((event_type, filename)), event.set()),
        lambda: None,
    )
    assert watcher is not None
    file_path.write_text("b", encoding="utf-8")
    assert event.wait(2.0)
    close_watcher(watcher)
    assert changes[0][0] == "change"
    assert changes[0][1] == "watched.txt"


@pytest.mark.asyncio
async def test_child_process_helpers_cover_sync_spawn_and_async_wait(tmp_path: Path) -> None:
    sync_result = spawn_process_sync(
        sys.executable,
        ["-c", "print('hello')"],
        cwd=str(tmp_path),
    )
    assert sync_result.status == 0
    assert sync_result.stdout == "hello\n"
    assert sync_result.stderr == ""

    child = spawn_process(
        sys.executable,
        ["-c", "print('world')"],
        cwd=str(tmp_path),
    )
    assert await wait_for_child_process(child) == 0
    assert child.stdout is None or child.stdout.closed
