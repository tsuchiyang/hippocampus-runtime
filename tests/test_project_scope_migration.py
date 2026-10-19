"""A physical move must not rename an explicitly preserved project identity."""
from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from codex_memory import cli
from codex_memory.database import connect_database, migrate_database
from codex_memory.memory_cli import run_memory_command
from codex_memory.paths import RuntimePaths
from codex_memory.project_memory_config import load_project_memory_config
from codex_memory.structured_ledger import apply_action, propose_record


OLD_KEY = r"C:\Old Project\memory"


def _configure(root: Path, value: str) -> None:
    (root / ".codex").mkdir(exist_ok=True)
    (root / ".codex" / "memory.toml").write_text(
        f"[scope]\nproject_key = {value}\n", encoding="utf-8"
    )


def _fixture(root: Path) -> RuntimePaths:
    paths = RuntimePaths.for_repo(root)
    paths.ensure()
    connection = connect_database(paths.db_path)
    try:
        migrate_database(connection)
        for key, scope, statement in (
            ("old", OLD_KEY, "KEEP_LEGACY_PROJECT"),
            ("foreign", "foreign-project", "EXCLUDE_FOREIGN_PROJECT"),
            ("physical", str(root), "EXCLUDE_NEW_PHYSICAL_IDENTITY"),
        ):
            record = propose_record(
                connection, record_type="decision", scope_kind="project",
                scope_key=scope, topic_key="migration.identity", statement=statement,
                idempotency_key=key,
            )
            apply_action(connection, record_id=record.record_id, action="activate",
                         idempotency_key=f"{key}-activate")
    finally:
        connection.close()
    return paths


def _recall(root: Path) -> tuple[int, str, str]:
    out, err = StringIO(), StringIO()
    code = run_memory_command(["memory", "recall", "--session-id", "migration-session"],
                              out, err, lambda _: root)
    return code, out.getvalue(), err.getvalue()


def test_explicit_project_key_restores_permanent_recall_without_ledger_rewrite(tmp_path):
    paths = _fixture(tmp_path)
    before = hashlib.sha256(paths.db_path.read_bytes()).hexdigest()
    _configure(tmp_path, json.dumps(OLD_KEY))
    code, output, error = _recall(tmp_path)
    assert (code, error) == (0, "")
    assert "KEEP_LEGACY_PROJECT" in output
    assert "EXCLUDE_FOREIGN_PROJECT" not in output
    assert "EXCLUDE_NEW_PHYSICAL_IDENTITY" not in output
    assert hashlib.sha256(paths.db_path.read_bytes()).hexdigest() == before


def test_absent_identity_override_preserves_existing_physical_scope(tmp_path):
    _fixture(tmp_path)
    code, output, error = _recall(tmp_path)
    assert (code, error) == (0, "")
    assert "EXCLUDE_NEW_PHYSICAL_IDENTITY" in output
    assert "KEEP_LEGACY_PROJECT" not in output
    assert "EXCLUDE_FOREIGN_PROJECT" not in output


@pytest.mark.parametrize("value", ['""', '"   "', 'true', '123', '[]', '"' + '界' * 86 + '"'])
def test_invalid_project_key_never_broadens_recall(tmp_path, value):
    _fixture(tmp_path)
    _configure(tmp_path, value)
    config = load_project_memory_config(tmp_path)
    assert config.status.value == "invalid"
    code, output, error = _recall(tmp_path)
    assert code == 0
    assert error == config.diagnostic
    assert "EXCLUDE_NEW_PHYSICAL_IDENTITY" in output
    assert "KEEP_LEGACY_PROJECT" not in output
    assert "EXCLUDE_FOREIGN_PROJECT" not in output


def test_capture_uses_preserved_key_and_rejects_unconfigured_project(tmp_path):
    _fixture(tmp_path)
    _configure(tmp_path, json.dumps(OLD_KEY))
    common = ["memory", "capture", "--session-id", "migration-session",
              "--turn-id", "migration-turn", "--context-id", "startup",
              "--type", "decision", "--scope", "project", "--topic-key", "migration.capture",
              "--statement", "TEMPORARY_MIGRATED_RECORD", "--idempotency-key", "migrated-capture"]
    out, err = StringIO(), StringIO()
    code = run_memory_command(common + ["--scope-key", OLD_KEY], out, err, lambda _: tmp_path)
    assert (code, err.getvalue()) == (0, "")
    assert json.loads(out.getvalue())["scope_key"] == OLD_KEY
    assert "TEMPORARY_MIGRATED_RECORD" in _recall(tmp_path)[1]
    other_out, other_err = StringIO(), StringIO()
    assert run_memory_command(["memory", "recall", "--session-id", "other-session"],
                              other_out, other_err, lambda _: tmp_path) == 0
    assert "TEMPORARY_MIGRATED_RECORD" not in other_out.getvalue()
    out, err = StringIO(), StringIO()
    assert run_memory_command(common + ["--scope-key", "foreign-project"],
                              out, err, lambda _: tmp_path) != 0


def test_real_prompt_hook_uses_preserved_key_in_packet_and_protocol(tmp_path, monkeypatch):
    _fixture(tmp_path)
    _configure(tmp_path, json.dumps(OLD_KEY))
    monkeypatch.setattr(cli, "_resolve_repo_root", lambda _: tmp_path)
    out, err = StringIO(), StringIO()
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "migration-session",
               "turn_id": "migration-turn", "cwd": str(tmp_path), "prompt": "migration"}
    code = cli.main(["user-prompt-submit"], StringIO(json.dumps(payload)), out, err)
    assert (code, err.getvalue()) == (0, "")
    packet = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
    assert "KEEP_LEGACY_PROJECT" in packet
    assert f"scope={json.dumps(OLD_KEY)}" in packet
    assert "EXCLUDE_FOREIGN_PROJECT" not in packet
    assert "EXCLUDE_NEW_PHYSICAL_IDENTITY" not in packet
