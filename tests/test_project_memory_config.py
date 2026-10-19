from __future__ import annotations

from pathlib import Path
import pytest

import codex_memory.project_memory_config as config_module
from codex_memory.project_memory_config import (
    ProjectMemoryConfigStatus,
    load_project_memory_config,
)


def test_missing_project_memory_config_disables_workspace_recall(tmp_path: Path) -> None:
    config = load_project_memory_config(tmp_path)

    assert config.workspace_scope_keys == ()
    assert config.diagnostic is None
    assert config.status is ProjectMemoryConfigStatus.ABSENT
    assert config.diagnostic_code is None


def test_project_memory_config_loads_workspace_keys_and_deduplicates(
    tmp_path: Path,
) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text(
        '[scope]\nworkspace_keys = ["team-notes", "team-notes", "other"]\n',
        encoding="utf-8",
    )

    config = load_project_memory_config(tmp_path)

    assert config.workspace_scope_keys == ("team-notes", "other")
    assert config.diagnostic is None
    assert config.status is ProjectMemoryConfigStatus.VALID
    assert config.diagnostic_code is None


def test_empty_project_memory_config_is_valid_not_absent(tmp_path: Path) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text("", encoding="utf-8")

    config = load_project_memory_config(tmp_path)

    assert config.status is ProjectMemoryConfigStatus.VALID
    assert config.workspace_scope_keys == ()


def test_malformed_project_memory_config_fails_closed_with_diagnostic(
    tmp_path: Path,
) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text("[scope\ninvalid", encoding="utf-8")

    config = load_project_memory_config(tmp_path)

    assert config.workspace_scope_keys == ()
    assert config.diagnostic
    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "invalid_toml"
    assert "invalid" in config.diagnostic


def test_unknown_scope_key_is_invalid_but_unrelated_top_level_table_is_allowed(
    tmp_path: Path,
) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    path = codex / "memory.toml"
    path.write_text(
        "[future]\nvalue = true\n[scope]\nworkspace_keys = ['ok']\n",
        encoding="utf-8",
    )
    assert load_project_memory_config(tmp_path).status is ProjectMemoryConfigStatus.VALID

    path.write_text("[scope]\nworkspace_keys = ['ok']\nunexpected = true\n", encoding="utf-8")
    config = load_project_memory_config(tmp_path)
    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "invalid_schema"
    assert config.workspace_scope_keys == ()


def test_project_memory_config_enforces_hard_byte_cap(tmp_path: Path) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_bytes(b"#" + b"x" * 65_536)

    config = load_project_memory_config(tmp_path)

    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "too_large"
    assert config.workspace_scope_keys == ()


def test_project_memory_config_rejects_growth_after_initial_size_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    path = codex / "memory.toml"
    prefix = b"[scope]\nworkspace_keys = []\n"
    path.write_bytes(prefix + b"#" * (config_module.MAX_PROJECT_MEMORY_CONFIG_BYTES - len(prefix)))
    original_open = Path.open

    def grow_before_read(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if self == path and mode == "rb":
            with original_open(self, "ab") as stream:
                stream.write(b"#")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow_before_read)

    config = load_project_memory_config(tmp_path)

    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "too_large"
    assert config.workspace_scope_keys == ()


def test_project_memory_config_diagnostics_are_bounded_and_do_not_leak_path(
    tmp_path: Path,
) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_bytes(b"\xff")

    config = load_project_memory_config(tmp_path)

    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "invalid_toml"
    assert config.diagnostic_code in {"invalid_toml", "read_error"}
    assert str(tmp_path) not in (config.diagnostic or "")


def test_redirected_memory_config_fails_closed(monkeypatch, tmp_path: Path) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text("[scope]\nworkspace_keys=['secret']\n")
    original = config_module.path_is_redirect
    monkeypatch.setattr(
        config_module,
        "path_is_redirect",
        lambda path: path.name == "memory.toml" or original(path),
    )

    config = load_project_memory_config(tmp_path)

    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "unsafe_path"
    assert config.workspace_scope_keys == ()


def test_memory_config_read_error_is_bounded(monkeypatch, tmp_path: Path) -> None:
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text("[scope]\nworkspace_keys=['ok']\n")

    def fail_open(*args: object, **kwargs: object) -> object:
        raise OSError("private filesystem detail")

    monkeypatch.setattr(Path, "open", fail_open)
    config = load_project_memory_config(tmp_path)

    assert config.status is ProjectMemoryConfigStatus.INVALID
    assert config.diagnostic_code == "read_error"
    assert "private filesystem detail" not in (config.diagnostic or "")
