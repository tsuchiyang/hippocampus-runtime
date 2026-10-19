from __future__ import annotations

import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from codex_memory import cli
from codex_memory.blob_store import BlobStore
from codex_memory.injection_store import InjectionPaths, InjectionStore
from codex_memory.ledger_common import canonical_json
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from codex_memory.structured_ledger import apply_action, propose_record


@pytest.fixture
def selected_host_lab(runtime, monkeypatch):
    root = runtime.paths.repo_root.resolve()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    home = root.parent / (root.name + "-private-home")
    home.mkdir()
    entry = Path(sys.executable).parent / "codex-memory-hook.exe"
    command = f'"{entry}" user-prompt-submit --selected-host'
    windows_path = str(entry).replace("'", "''")
    hooks = {"hooks": {"UserPromptSubmit": [{"hooks": [{
        "type": "command", "command": command,
        "commandWindows": f"& '{windows_path}' user-prompt-submit --selected-host",
        "timeout": 2, "additionalContextLimit": 0,
    }]}]}}
    (root / ".codex").mkdir()
    raw_hooks = canonical_json(hooks).encode("utf-8")
    (root / ".codex/hooks.json").write_bytes(raw_hooks)
    marker = {
        "format": "codex-memory.p26c-selected-host.v1", "project_root": str(root),
        "data_origin": "synthetic", "provider_policy": "disabled",
        "hook_policy": "selected_host", "session_id": "selected-session",
        "codex_home": str(home.resolve()), "runtime_entry": str(entry.resolve()),
        "runtime_entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        "hooks_sha256": hashlib.sha256(raw_hooks).hexdigest(),
    }
    (runtime.paths.runtime_root / "p26c-host-lab.json").write_text(
        canonical_json(marker), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setattr(sys, "argv", [str(entry)])
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project",
        scope_key=str(root), topic_key="p26c.host-code",
        statement="The synthetic selected host verification code is HOST-P26C-EXAMPLE.",
        value=None, evidence_uris=(), idempotency_key="host-record")
    apply_action(runtime.connection, record_id=record.record_id, action="activate",
        idempotency_key="host-active")
    assert RetrievalService(RetrievalPaths.for_repo(root)).rebuild().status.value == "current"
    return runtime, marker, hooks


def invoke(lab, *, session_id="selected-session", output=None, flag="--selected-host"):
    runtime, _marker, _hooks = lab
    output = StringIO() if output is None else output
    errors = StringIO()
    payload = {"hook_event_name": "UserPromptSubmit", "cwd": str(runtime.paths.repo_root),
        "session_id": session_id, "turn_id": "selected-turn",
        "prompt": "What is the selected host verification code?"}
    code = cli.main(["user-prompt-submit", flag], stdin=StringIO(canonical_json(payload)),
        stdout=output, stderr=errors, _process_started_ns=time.monotonic_ns())
    return code, output.getvalue(), errors.getvalue(), payload


def test_selected_host_commits_source_and_exact_wire_before_output(selected_host_lab, monkeypatch):
    runtime, _marker, _hooks = selected_host_lab
    actual_load = InjectionStore.load
    reopened = []

    def load(self, injection_id, **kwargs):
        verified = actual_load(self, injection_id, **kwargs)
        reopened.append(verified)
        return verified

    monkeypatch.setattr(InjectionStore, "load", load)

    class Output(StringIO):
        def write(self, value):
            assert reopened, "real manifest must already be reopened before output"
            assert reopened[-1].serialized_response_bytes == value.encode("utf-8")
            return super().write(value)

    code, stdout, stderr, payload = invoke(selected_host_lab, output=Output())
    assert code == 0, stderr
    metadata = json.loads(stderr)
    assert metadata["output_mode"] == "selected_host"
    assert metadata["source_origin"] == "hook"
    assert metadata["delivery"] == "emitted"
    assert metadata["host_acknowledgement"] == "unknown"
    packet = json.loads(stdout)["hookSpecificOutput"]["additionalContext"]
    assert "HOST-P26C-EXAMPLE" in packet and len(packet) <= 4000
    verified = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root),
        BlobStore(runtime.paths.blobs_dir)).load(metadata["injection_id"])
    assert verified.serialized_response_bytes == stdout.encode("utf-8")
    assert verified.prepared_packet_bytes == packet.encode("utf-8")
    assert verified.manifest.body["source"]["origin"] == "hook"
    assert verified.manifest.body["source"]["session_id"] == "selected-session"
    assert verified.manifest.body["source"]["turn_id"] == "selected-turn"
    assert verified.manifest.body["source"]["event_blob_sha256"] == hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()
    assert verified.manifest.body["delivery"]["v2"] == "prepared"
    assert runtime.connection.execute("SELECT count(*) FROM events").fetchone()[0] == 1


@pytest.mark.parametrize("damage", [
    "missing", "extra_field", "private_origin", "wrong_root", "wrong_home",
    "wrong_session", "wrong_entry_hash", "changed_hook", "second_hook",
    "project_config", "old_marker_only", "hardlinked_marker", "hardlinked_hook", "bare_windows_command",
])
def test_selected_host_rejects_unapproved_binding_before_runtime_or_writes(
        selected_host_lab, monkeypatch, damage):
    runtime, marker, hooks = selected_host_lab
    root = runtime.paths.repo_root
    marker_path = runtime.paths.runtime_root / "p26c-host-lab.json"
    hook_path = root / ".codex/hooks.json"
    session_id = "selected-session"
    if damage == "missing":
        marker_path.unlink()
    elif damage == "old_marker_only":
        marker_path.unlink()
        (runtime.paths.runtime_root / "p26b-lab.json").write_text(canonical_json({
            "format": "codex-memory.p26b-prompt-lab.v1", "project_root": str(root),
            "data_origin": "synthetic", "provider_policy": "disabled",
            "hook_policy": "isolated_command", "session_id": "selected-session",
        }), encoding="utf-8")
    elif damage == "wrong_session":
        session_id = "unselected-session"
    elif damage == "wrong_home":
        monkeypatch.setenv("CODEX_HOME", str(root))
    elif damage == "changed_hook":
        hook_path.write_bytes(b"{}")
    elif damage == "second_hook":
        hooks["hooks"]["Stop"] = [{"hooks": [{"type": "command", "command": "not-approved"}]}]
        raw = canonical_json(hooks).encode("utf-8")
        hook_path.write_bytes(raw)
        marker["hooks_sha256"] = hashlib.sha256(raw).hexdigest()
        marker_path.write_text(canonical_json(marker), encoding="utf-8")
    elif damage == "bare_windows_command":
        handler = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        handler["commandWindows"] = handler["command"]
        raw = canonical_json(hooks).encode("utf-8")
        hook_path.write_bytes(raw)
        marker["hooks_sha256"] = hashlib.sha256(raw).hexdigest()
        marker_path.write_text(canonical_json(marker), encoding="utf-8")
    elif damage == "project_config":
        (root / ".codex/config.toml").write_text("[hooks]\n", encoding="utf-8")
    elif damage in {"hardlinked_marker", "hardlinked_hook"}:
        path = marker_path if damage == "hardlinked_marker" else hook_path
        os.link(path, root / (damage + ".alias"))
    else:
        if damage == "extra_field":
            marker["approve_everything"] = True
        elif damage == "private_origin":
            marker["data_origin"] = "private"
        elif damage == "wrong_root":
            marker["project_root"] += "-other"
        else:
            marker["runtime_entry_sha256"] = "0" * 64
        marker_path.write_text(canonical_json(marker), encoding="utf-8")
    before = tuple(runtime.connection.iterdump())

    def forbidden_runtime(*args, **kwargs):
        pytest.fail("authorization must precede Runtime creation")

    monkeypatch.setattr(cli, "_construct_runtime", forbidden_runtime)
    code, stdout, stderr, _payload = invoke(selected_host_lab, session_id=session_id)
    assert code == 2 and stdout == ""
    assert stderr.startswith("{"), "must reach the new route's real authorization boundary"
    assert json.loads(stderr)["reason"] == "selected_host_not_authorized"
    assert tuple(runtime.connection.iterdump()) == before


def test_selected_host_cache_absent_returns_fresh_legacy(selected_host_lab):
    runtime, _marker, _hooks = selected_host_lab
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    paths.database_path.unlink()
    code, stdout, stderr, _payload = invoke(selected_host_lab)
    assert code == 0, stderr
    assert json.loads(stderr)["delivery"] == "legacy_fallback"
    assert json.loads(stderr)["injection_id"] is None
    assert "HOST-P26C-EXAMPLE" in json.loads(stdout)["hookSpecificOutput"]["additionalContext"]


def test_selected_host_canonical_lock_refuses_output(selected_host_lab):
    runtime, _marker, _hooks = selected_host_lab
    locker = sqlite3.connect(runtime.paths.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        code, stdout, stderr, _payload = invoke(selected_host_lab)
    finally:
        locker.rollback()
        locker.close()
    assert code == 1 and stdout == ""
    assert json.loads(stderr)["delivery"] == "not_emitted"


def test_selected_host_does_not_unlock_the_p26b_command_guard(selected_host_lab):
    runtime, _marker, _hooks = selected_host_lab
    (runtime.paths.runtime_root / "p26b-lab.json").write_text(canonical_json({
        "format": "codex-memory.p26b-prompt-lab.v1", "project_root": str(runtime.paths.repo_root),
        "data_origin": "synthetic", "provider_policy": "disabled",
        "hook_policy": "isolated_command", "session_id": "selected-session",
    }), encoding="utf-8")
    code, stdout, stderr, _payload = invoke(selected_host_lab, flag="--isolated-active")
    assert code == 2 and stdout == ""
    assert json.loads(stderr)["reason"] == "lab_not_authorized"
