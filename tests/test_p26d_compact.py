from __future__ import annotations

import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from codex_memory import cli
from codex_memory.ledger_common import canonical_json
from codex_memory.memory_context import current_context_id
from codex_memory.retrieval.paths import RetrievalPaths
from test_p26c_selected_host import selected_host_lab  # shared real-storage fixture


def exact_hooks(entry):
    hooks = {}
    for event, command, matcher in (
        ("UserPromptSubmit", "user-prompt-submit", None),
        ("PreCompact", "pre-compact", "manual"),
        ("PostCompact", "post-compact", "manual"),
        ("SessionStart", "session-start", "compact"),
    ):
        windows = str(entry).replace("'", "''")
        group = {"hooks": [{"type": "command",
            "command": f'"{entry}" {command} --selected-compact',
            "commandWindows": f"& '{windows}' {command} --selected-compact",
            "timeout": 2}]}
        if event in {"UserPromptSubmit", "SessionStart"}:
            group["hooks"][0]["additionalContextLimit"] = 0
        if matcher:
            group["matcher"] = matcher
        hooks[event] = [group]
    return {"hooks": hooks}


@pytest.fixture
def compact_lab(selected_host_lab):
    runtime, marker, _hooks = selected_host_lab
    root = runtime.paths.repo_root
    hooks = exact_hooks(Path(marker["runtime_entry"]))
    raw = canonical_json(hooks).encode()
    (root / ".codex/hooks.json").write_bytes(raw)
    marker = {**marker, "format": "codex-memory.p26d-selected-compact.v1",
        "hook_policy": "selected_compact", "hooks_sha256": hashlib.sha256(raw).hexdigest()}
    (runtime.paths.runtime_root / "p26d-compact-lab.json").write_text(canonical_json(marker), encoding="utf-8")
    (root / ".gitignore").write_text(".codex/\n.codex-memory/\nrollout.jsonl\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Codex Synthetic Lab",
        "-c", "user.email=codex-memory@local.invalid", "commit", "-qm", "synthetic fixture"],
        check=True, capture_output=True)
    (root / "rollout.jsonl").write_text('{"type":"synthetic"}\n', encoding="utf-8")
    return runtime, marker, hooks


def invoke(lab, command, event, **extras):
    runtime, _marker, _hooks = lab
    payload = {"hook_event_name": event, "session_id": "selected-session",
        "cwd": str(runtime.paths.repo_root), "transcript_path": str(runtime.paths.repo_root / "rollout.jsonl"),
        **extras}
    out, err = StringIO(), StringIO()
    rc = cli.main([command, "--selected-compact"], stdin=StringIO(canonical_json(payload)),
        stdout=out, stderr=err, _process_started_ns=time.monotonic_ns())
    return rc, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("cache_absent", [False, True])
def test_compact_profile_real_checkpoint_recovery_and_fresh_window(compact_lab, cache_absent):
    runtime, marker, hooks = compact_lab
    from codex_memory.retrieval.selected_host import hook_definition
    assert hook_definition(Path(marker["runtime_entry"]), compact_cycle=True) == hooks
    rc, out, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="before", prompt="What is the selected host verification code?")
    assert rc == 0, err
    assert json.loads(err)["delivery"] == "emitted"
    assert current_context_id(runtime.connection, session_id="selected-session") == "startup"
    for command, event in (("pre-compact", "PreCompact"), ("post-compact", "PostCompact")):
        rc, out, err = invoke(compact_lab, command, event, turn_id="compact-turn", trigger="manual")
        assert (rc, json.loads(out), err) == (0, {"continue": True}, "")
    rc, recovered, err = invoke(compact_lab, "session-start", "SessionStart", source="compact")
    assert rc == 0 and not err
    checkpoint = current_context_id(runtime.connection, session_id="selected-session")
    assert checkpoint.startswith("cp_") and checkpoint in recovered
    if cache_absent:
        RetrievalPaths.for_repo(runtime.paths.repo_root).database_path.unlink()
    rc, fresh, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="after", prompt="What is the selected host verification code?")
    assert rc == 0, err
    assert json.loads(err)["delivery"] == ("legacy_fallback" if cache_absent else "emitted")
    assert checkpoint in fresh and 'HOST-P26C-EXAMPLE' in fresh
    rows = runtime.connection.execute("SELECT event_name FROM events ORDER BY event_id").fetchall()
    assert [r[0] for r in rows] == ["UserPromptSubmit", "PreCompact", "PostCompact", "SessionStart", "UserPromptSubmit"]
    row = runtime.connection.execute("SELECT status, recovery_required, recovery_delivered_at FROM checkpoints").fetchone()
    assert row[0] == "COMMITTED" and row[1] == 0 and row[2]


def test_compact_recovery_with_entire_retrieval_absent_keeps_fresh_legacy(compact_lab, monkeypatch):
    """A missing optional index must not become a core recovery dependency."""
    from codex_memory.retrieval import config

    runtime, marker, _hooks = compact_lab
    assert marker["provider_policy"] == "disabled"

    def forbidden_provider(*args, **kwargs):
        pytest.fail("selected compact recovery must not construct a provider")

    monkeypatch.setattr(config, "provider_from_config", forbidden_provider)
    rc, out, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="before", prompt="verification code")
    assert rc == 0 and json.loads(err)["delivery"] == "emitted"
    tables = ("memory_records", "memory_events", "memory_evidence", "memory_relations")
    before = {table: [tuple(row) for row in runtime.connection.execute(f"SELECT * FROM {table}")]
        for table in tables}
    manifests = runtime.paths.runtime_root / "injections"
    saved_manifests = {path.relative_to(manifests): path.read_bytes()
        for path in manifests.glob("inj_*/manifest.json")}
    assert len(saved_manifests) == 1

    # Move only this newly-created synthetic fixture's disposable directory.
    # Keep the displaced files for inspection; never touch a real project.
    retrieval = RetrievalPaths.for_repo(runtime.paths.repo_root).retrieval_root
    assert retrieval.resolve().is_relative_to(runtime.paths.runtime_root.resolve())
    retrieval.rename(runtime.paths.runtime_root / "retrieval-held-outside-active-path")
    assert not retrieval.exists()
    for command, event in (("pre-compact", "PreCompact"), ("post-compact", "PostCompact")):
        rc, out, err = invoke(compact_lab, command, event, turn_id="compact-turn", trigger="manual")
        assert (rc, json.loads(out), err) == (0, {"continue": True}, "")
        assert not retrieval.exists()

    rc, recovered, err = invoke(compact_lab, "session-start", "SessionStart", source="compact")
    assert rc == 0 and not err
    checkpoint = current_context_id(runtime.connection, session_id="selected-session")
    assert checkpoint.startswith("cp_")
    assert checkpoint in json.loads(recovered)["hookSpecificOutput"]["additionalContext"]
    assert not retrieval.exists()
    rc, out, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="after", prompt="verification code")
    assert rc == 0
    metadata = json.loads(err)
    assert metadata["delivery"] == "legacy_fallback" and metadata["injection_id"] is None
    packet = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert checkpoint in packet and "HOST-P26C-EXAMPLE" in packet
    assert len(packet) <= 4000
    assert not retrieval.exists()
    assert {path.relative_to(manifests): path.read_bytes()
        for path in manifests.glob("inj_*/manifest.json")} == saved_manifests
    assert {table: [tuple(row) for row in runtime.connection.execute(f"SELECT * FROM {table}")]
        for table in tables} == before
    row = runtime.connection.execute("SELECT status, recovery_required, recovery_delivered_at FROM checkpoints").fetchone()
    assert row[0] == "COMMITTED" and row[1] == 0 and row[2]
    assert [row[0] for row in runtime.connection.execute("SELECT event_name FROM events ORDER BY event_id")] == [
        "UserPromptSubmit", "PreCompact", "PostCompact", "SessionStart", "UserPromptSubmit"]


@pytest.mark.parametrize("command,event,extra", [
    ("user-prompt-submit", "UserPromptSubmit", {"turn_id": "t", "prompt": "code", "session_id": "other"}),
    ("pre-compact", "PreCompact", {"turn_id": "t", "trigger": "auto"}),
    ("post-compact", "PostCompact", {"turn_id": "t", "trigger": "auto"}),
    ("session-start", "SessionStart", {"source": "startup"}),
])
def test_compact_profile_rejects_other_scope_before_runtime(compact_lab, monkeypatch, command, event, extra):
    calls = []
    monkeypatch.setattr(cli, "_construct_runtime", lambda *a, **k: calls.append(True))
    rc, out, err = invoke(compact_lab, command, event, **extra)
    assert rc == 2 and not out and err and calls == []
    assert compact_lab[0].connection.execute("SELECT count(*) FROM events").fetchone()[0] == 0


@pytest.mark.parametrize("damage", ["wrong_session", "extra_hook", "runtime_config", "old_profile"])
def test_compact_profile_rejects_unapproved_bindings_before_runtime(compact_lab, monkeypatch, damage):
    runtime, marker, hooks = compact_lab
    if damage == "wrong_session":
        marker["session_id"] = "other"
        (runtime.paths.runtime_root / "p26d-compact-lab.json").write_text(canonical_json(marker), encoding="utf-8")
    elif damage == "extra_hook":
        hooks["hooks"]["Stop"] = hooks["hooks"]["UserPromptSubmit"]
        raw = canonical_json(hooks).encode()
        (runtime.paths.repo_root / ".codex/hooks.json").write_bytes(raw)
        marker["hooks_sha256"] = hashlib.sha256(raw).hexdigest()
        (runtime.paths.runtime_root / "p26d-compact-lab.json").write_text(canonical_json(marker), encoding="utf-8")
    elif damage == "runtime_config":
        (runtime.paths.repo_root / ".codex/memory.toml").write_text("[retrieval]\nmode='shadow'\n", encoding="utf-8")
    else:
        (runtime.paths.runtime_root / "p26d-compact-lab.json").unlink()
    calls = []
    monkeypatch.setattr(cli, "_construct_runtime", lambda *a, **k: calls.append(True))
    rc, out, err = invoke(compact_lab, "pre-compact", "PreCompact", turn_id="t", trigger="manual")
    assert rc == 2 and not out and calls == []


def test_compact_profile_cannot_unlock_selected_host_guard(compact_lab):
    from test_p26c_selected_host import invoke as invoke_original
    rc, out, err, _payload = invoke_original(compact_lab)
    assert rc == 2 and not out and 'selected_host_not_authorized' in err


def test_compact_profile_cleanup_failure_is_not_a_success_or_raw_trace(compact_lab, monkeypatch):
    class Connection:
        def close(self):
            raise OSError("private cleanup detail")
    monkeypatch.setattr(cli, "_construct_runtime", lambda *a, **k: SimpleNamespace(connection=Connection()))
    monkeypatch.setattr(cli, "_dispatch", lambda *a, **k: 0)
    rc, out, err = invoke(compact_lab, "pre-compact", "PreCompact", turn_id="t", trigger="manual")
    assert rc == 1 and not out and err
    assert "private cleanup detail" not in err


def test_compact_expired_budget_never_constructs_runtime(compact_lab, monkeypatch):
    runtime = compact_lab[0]
    calls = []
    monkeypatch.setattr(cli, "_construct_runtime", lambda *a, **k: calls.append(True))
    out, err = StringIO(), StringIO()
    rc = cli.main(["pre-compact", "--selected-compact"], stdin=StringIO(canonical_json({
        "hook_event_name": "PreCompact", "cwd": str(runtime.paths.repo_root), "session_id": "selected-session",
        "turn_id": "t", "trigger": "manual"})), stdout=out, stderr=err,
        _process_started_ns=time.monotonic_ns() - 2_000_000_000)
    assert rc != 0 and not out.getvalue() and calls == []
