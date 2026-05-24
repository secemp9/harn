"""Footer data provider for git branch and extension footer state."""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

FS_WATCH_RETRY_DELAY_SECONDS = 5.0
_UNSET = object()


@dataclass(slots=True)
class GitPaths:
    repoDir: str
    commonGitDir: str
    headPath: str


def find_git_paths(cwd: str) -> GitPaths | None:
    dir_path = os.path.abspath(cwd)
    while True:
        git_path = os.path.join(dir_path, ".git")
        if os.path.exists(git_path):
            try:
                if os.path.isfile(git_path):
                    content = Path(git_path).read_text(encoding="utf-8").strip()
                    if content.startswith("gitdir: "):
                        git_dir = os.path.abspath(os.path.join(dir_path, content[8:].strip()))
                        head_path = os.path.join(git_dir, "HEAD")
                        if not os.path.exists(head_path):
                            return None
                        common_dir_path = os.path.join(git_dir, "commondir")
                        common_dir = (
                            Path(common_dir_path).read_text(encoding="utf-8").strip()
                            if os.path.exists(common_dir_path)
                            else None
                        )
                        common_git_dir = (
                            os.path.abspath(os.path.join(git_dir, common_dir))
                            if common_dir is not None
                            else git_dir
                        )
                        return GitPaths(repoDir=dir_path, commonGitDir=common_git_dir, headPath=head_path)
                elif os.path.isdir(git_path):
                    head_path = os.path.join(git_path, "HEAD")
                    if not os.path.exists(head_path):
                        return None
                    return GitPaths(repoDir=dir_path, commonGitDir=git_path, headPath=head_path)
            except OSError:
                return None
        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            return None
        dir_path = parent


def resolve_branch_with_git_sync(repo_dir: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo_dir,
            encoding="utf-8",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    branch = result.stdout.strip() if result.returncode == 0 else ""
    return branch or None


class ReadonlyFooterDataProvider(Protocol):
    def getGitBranch(self) -> str | None: ...

    def getExtensionStatuses(self) -> Mapping[str, str]: ...

    def getAvailableProviderCount(self) -> int: ...

    def onBranchChange(self, callback: Callable[[], None]) -> Callable[[], None]: ...


class FooterDataProvider:
    WATCH_DEBOUNCE_SECONDS = 0.5
    WATCH_POLL_INTERVAL_SECONDS = 0.1

    def __init__(self, cwd: str) -> None:
        self.cwd = os.path.abspath(cwd)
        self.extensionStatuses: dict[str, str] = {}
        self.cachedBranch: object | str | None = _UNSET
        self.gitPaths: GitPaths | None = find_git_paths(self.cwd)
        self.branchChangeCallbacks: set[Callable[[], None]] = set()
        self.availableProviderCount = 0
        self.refreshTimer: threading.Timer | None = None
        self.gitWatcherRetryTimer: threading.Timer | None = None
        self.refreshInFlight = False
        self.refreshPending = False
        self.disposed = False
        self._lock = threading.RLock()
        self._watch_thread: threading.Thread | None = None
        self._watch_stop = threading.Event()
        self._watch_started = False
        self.setupGitWatcher()

    def getGitBranch(self) -> str | None:
        with self._lock:
            if self.cachedBranch is _UNSET:
                self.cachedBranch = self.resolveGitBranchSync()
            return None if self.cachedBranch is _UNSET else self.cachedBranch

    def getExtensionStatuses(self) -> Mapping[str, str]:
        return self.extensionStatuses

    def onBranchChange(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            self.branchChangeCallbacks.add(callback)

        def unsubscribe() -> None:
            with self._lock:
                self.branchChangeCallbacks.discard(callback)

        return unsubscribe

    def setExtensionStatus(self, key: str, text: str | None) -> None:
        if text is None:
            self.extensionStatuses.pop(key, None)
        else:
            self.extensionStatuses[key] = text

    def clearExtensionStatuses(self) -> None:
        self.extensionStatuses.clear()

    def getAvailableProviderCount(self) -> int:
        return self.availableProviderCount

    def setAvailableProviderCount(self, count: int) -> None:
        self.availableProviderCount = count

    def setCwd(self, cwd: str) -> None:
        normalized = os.path.abspath(cwd)
        if self.cwd == normalized:
            return
        self.cwd = normalized
        with self._lock:
            if self.refreshTimer is not None:
                self.refreshTimer.cancel()
                self.refreshTimer = None
            self.clearGitWatchers()
            self.cachedBranch = _UNSET
            self.gitPaths = find_git_paths(self.cwd)
            self.setupGitWatcher()
        self.notifyBranchChange()

    def dispose(self) -> None:
        with self._lock:
            self.disposed = True
            if self.refreshTimer is not None:
                self.refreshTimer.cancel()
                self.refreshTimer = None
            self.clearGitWatchers()
            self.branchChangeCallbacks.clear()

    def notifyBranchChange(self) -> None:
        callbacks = list(self.branchChangeCallbacks)
        for callback in callbacks:
            callback()

    def scheduleRefresh(self) -> None:
        with self._lock:
            if self.disposed:
                return
            if self.refreshTimer is not None:
                self.refreshPending = True
                return
            if self.refreshInFlight:
                self.refreshPending = True
                return
            self.refreshTimer = threading.Timer(self.WATCH_DEBOUNCE_SECONDS, self.refreshGitBranchAsync)
            self.refreshTimer.daemon = True
            self.refreshTimer.start()

    def refreshGitBranchAsync(self) -> None:
        with self._lock:
            if self.disposed:
                return
            self.refreshTimer = None
            if self.refreshInFlight:
                self.refreshPending = True
                return
            self.refreshInFlight = True
        try:
            next_branch = self.resolveGitBranchAsync()
            with self._lock:
                if self.disposed:
                    return
                if self.cachedBranch is not _UNSET and self.cachedBranch != next_branch:
                    self.cachedBranch = next_branch
                    self.notifyBranchChange()
                    return
                self.cachedBranch = next_branch
        finally:
            with self._lock:
                self.refreshInFlight = False
                if self.refreshPending and not self.disposed:
                    self.refreshPending = False
                    self.scheduleRefresh()

    def resolveGitBranchSync(self) -> str | None:
        try:
            if self.gitPaths is None:
                return None
            content = Path(self.gitPaths.headPath).read_text(encoding="utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                branch = content[16:]
                if branch == ".invalid":
                    return resolve_branch_with_git_sync(self.gitPaths.repoDir) or "detached"
                return branch
            return "detached"
        except OSError:
            return None

    def resolveGitBranchAsync(self) -> str | None:
        return self.resolveGitBranchSync()

    def clearGitWatchers(self) -> None:
        self._watch_stop.set()
        if self._watch_thread is not None and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=0.5)
        self._watch_thread = None
        self._watch_stop = threading.Event()
        self._watch_started = False
        if self.gitWatcherRetryTimer is not None:
            self.gitWatcherRetryTimer.cancel()
            self.gitWatcherRetryTimer = None

    def scheduleGitWatcherRetry(self) -> None:
        with self._lock:
            if self.disposed or self.gitWatcherRetryTimer is not None:
                return
            self.gitWatcherRetryTimer = threading.Timer(FS_WATCH_RETRY_DELAY_SECONDS, self._retryGitWatcher)
            self.gitWatcherRetryTimer.daemon = True
            self.gitWatcherRetryTimer.start()

    def _retryGitWatcher(self) -> None:
        with self._lock:
            self.gitWatcherRetryTimer = None
        self.setupGitWatcher()

    def handleGitWatcherError(self) -> None:
        self.clearGitWatchers()
        self.scheduleGitWatcherRetry()

    def setupGitWatcher(self) -> None:
        self.clearGitWatchers()
        if self.gitPaths is None or self.disposed:
            return
        try:
            reftable_dir = os.path.join(self.gitPaths.commonGitDir, "reftable")
            tables_list_path = os.path.join(reftable_dir, "tables.list")
            head_sig = self._path_signature(self.gitPaths.headPath)
            reftable_sig = self._dir_signature(reftable_dir)
            tables_sig = self._path_signature(tables_list_path)
            self._watch_thread = threading.Thread(
                target=self._watch_loop,
                args=(head_sig, reftable_sig, tables_sig),
                name="footer-git-watch",
                daemon=True,
            )
            self._watch_thread.start()
            self._watch_started = True
        except Exception:
            self._watch_thread = None
            self.handleGitWatcherError()

    def _watch_loop(
        self,
        head_sig: tuple[Literal["missing"], None, None, None] | tuple[int, int, int, str | None],
        reftable_sig: tuple[Literal["missing"], None, None] | tuple[tuple[str, int, int], ...],
        tables_sig: tuple[Literal["missing"], None, None, None] | tuple[int, int, int, str | None],
    ) -> None:
        if self.gitPaths is None:
            return
        reftable_dir = os.path.join(self.gitPaths.commonGitDir, "reftable")
        tables_list_path = os.path.join(reftable_dir, "tables.list")
        while not self._watch_stop.wait(self.WATCH_POLL_INTERVAL_SECONDS):
            try:
                next_head_sig = self._path_signature(self.gitPaths.headPath)
                next_reftable_sig = self._dir_signature(reftable_dir)
                next_tables_sig = self._path_signature(tables_list_path)
            except Exception:
                self.handleGitWatcherError()
                return
            if next_head_sig != head_sig or next_reftable_sig != reftable_sig or next_tables_sig != tables_sig:
                head_sig = next_head_sig
                reftable_sig = next_reftable_sig
                tables_sig = next_tables_sig
                self.scheduleRefresh()

    def _path_signature(
        self,
        path: str,
    ) -> tuple[Literal["missing"], None, None, None] | tuple[int, int, int, str | None]:
        try:
            stat_result = os.stat(path)
        except OSError:
            return ("missing", None, None, None)

        content_fingerprint: str | None = None
        if os.path.isfile(path):
            try:
                content_fingerprint = hashlib.blake2b(
                    Path(path).read_bytes(),
                    digest_size=8,
                ).hexdigest()
            except OSError:
                content_fingerprint = None
        return (
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
            stat_result.st_size,
            content_fingerprint,
        )

    def _dir_signature(self, path: str) -> tuple[Literal["missing"], None, None] | tuple[tuple[str, int, int], ...]:
        if not os.path.isdir(path):
            return ("missing", None, None)
        try:
            items: list[tuple[str, int, int]] = []
            for entry in os.scandir(path):
                try:
                    stat_result = entry.stat()
                except OSError:
                    continue
                items.append((entry.name, stat_result.st_mtime_ns, stat_result.st_size))
        except OSError:
            return ("missing", None, None)
        items.sort()
        return tuple(items)


findGitPaths = find_git_paths
resolveBranchWithGitSync = resolve_branch_with_git_sync

__all__ = [
    "FS_WATCH_RETRY_DELAY_SECONDS",
    "FooterDataProvider",
    "GitPaths",
    "ReadonlyFooterDataProvider",
    "findGitPaths",
    "find_git_paths",
    "resolveBranchWithGitSync",
    "resolve_branch_with_git_sync",
]
