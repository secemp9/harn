from __future__ import annotations

from pathlib import Path

import pytest
import harnify_coding_agent.utils.windows_self_update as windows_self_update_module


def test_windows_self_update_module_exports_match_ts_surface() -> None:
    assert windows_self_update_module.__all__ == [
        "cleanupWindowsSelfUpdateQuarantine",
        "quarantineWindowsNativeDependencies",
    ]


def test_cleanup_windows_self_update_quarantine_removes_node_modules_quarantine(tmp_path: Path) -> None:
    package_dir = tmp_path / "node_modules" / "pkg" / "dist"
    quarantine_root = tmp_path / "node_modules" / ".harnify-native-quarantine"
    package_dir.mkdir(parents=True)
    (quarantine_root / "stale").mkdir(parents=True)

    windows_self_update_module.cleanup_windows_self_update_quarantine(str(package_dir))

    assert not quarantine_root.exists()


def test_quarantine_windows_native_dependencies_preserves_loaded_binary_and_quarantines_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "node_modules" / "pkg"
    loaded_file = package_dir / "native" / "addon.pyd"
    loaded_file.parent.mkdir(parents=True)
    loaded_file.write_text("native-binary", encoding="utf-8")

    monkeypatch.setattr(
        windows_self_update_module,
        "_get_loaded_shared_objects_in_package_dir",
        lambda _package_dir: [loaded_file],
    )
    monkeypatch.setattr(windows_self_update_module, "uuid4", lambda: "fixed-uuid")
    monkeypatch.setattr(windows_self_update_module.os, "getpid", lambda: 4321)
    monkeypatch.setattr(windows_self_update_module.time, "time", lambda: 1.234)

    windows_self_update_module.quarantine_windows_native_dependencies(str(package_dir))

    quarantine_root = tmp_path / "node_modules" / ".harnify-native-quarantine"
    run_dirs = list(quarantine_root.iterdir())
    assert [path.name for path in run_dirs] == ["1234-4321-fixed-uuid"]
    assert loaded_file.read_text(encoding="utf-8") == "native-binary"
    assert (run_dirs[0] / "native" / "addon.pyd").read_text(encoding="utf-8") == "native-binary"
