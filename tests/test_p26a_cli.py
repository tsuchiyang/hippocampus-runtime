from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from io import BytesIO, StringIO

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.hook_input import parse_hook_payload
from codex_memory.injection_store import InjectionPaths, InjectionStore
from codex_memory.ledger_common import canonical_json
from codex_memory.memory_evidence import event_uri
from codex_memory.retrieval.cli import run_retrieval_command
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from codex_memory.structured_ledger import apply_action, propose_record


@pytest.fixture
def lab(runtime):
    root = runtime.paths.repo_root
    marker = {
        "format": "codex-memory.p26a-cli-lab.v1", "project_root": str(root.resolve()),
        "data_origin": "synthetic", "provider_policy": "disabled", "hook_policy": "disabled",
    }
    (runtime.paths.runtime_root / "p26a-lab.json").write_text(canonical_json(marker), encoding="utf-8")
    record = propose_record(
        runtime.connection, record_type="decision", scope_kind="project", scope_key=str(root),
        topic_key="lab.storage", statement="Use WAL in the isolated sandbox.", value=None,
        evidence_uris=(), idempotency_key="p26a-propose",
    )
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="p26a-activate")
    event = runtime.events.append(parse_hook_payload({
        "hook_event_name": "UserPromptSubmit", "session_id": "p26a_session", "turn_id": "p26a_turn",
        "cwd": str(root), "prompt": "Which storage should the isolated sandbox use?",
    }))
    assert RetrievalService(RetrievalPaths.for_repo(root)).rebuild().status.value == "current"
    return runtime, record.record_id, event_uri("p26a_session", event.event_id)


def _recall(lab, *, overrides=(), stdout=None):
    runtime, _record, uri = lab
    output, errors = stdout if stdout is not None else StringIO(), StringIO()
    values = {
        "--session-id": "p26a_session", "--turn-id": "p26a_turn", "--context-id": "p26a_cli",
        "--query": "isolated sandbox WAL", "--prompt-event-uri": uri,
    }
    values.update(dict(overrides))
    tokens = ["retrieval", "recall", "--isolated-cli"]
    for key, value in values.items():
        tokens.extend((key, value))
    code = run_retrieval_command(tokens, output, errors, lambda _cwd: runtime.paths.repo_root)
    return code, output.getvalue(), errors.getvalue()


def _history(runtime):
    return tuple(runtime.connection.iterdump())


def test_isolated_cli_commits_and_reopens_exact_output_before_emission(lab):
    runtime, record_id, _uri = lab
    before = _history(runtime)
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    document, metadata = json.loads(stdout), json.loads(stderr)
    assert document["format"] == "codex-memory.p26a-cli-packet.v1"
    assert document["mode"] == "active_cli"
    assert document["status"] == "phase2"
    assert "Use WAL in the isolated sandbox." in document["packet"]
    assert record_id in document["selected_ids"]
    assert document["packet_chars"] == len(document["packet"]) <= 4000
    assert document["packet_sha256"] == hashlib.sha256(document["packet"].encode()).hexdigest()
    assert stdout == canonical_json(document) + "\n"
    assert metadata["delivery"] == "emitted"
    assert metadata["host_acknowledgement"] == "unknown"
    verified = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), BlobStore(runtime.paths.blobs_dir)).load(metadata["injection_id"])
    assert verified.serialized_response_bytes == stdout.encode("utf-8")
    assert verified.prepared_packet_bytes == document["packet"].encode("utf-8")
    assert verified.manifest.body["source"]["origin"] == "offline"
    assert verified.manifest.body["delivery"]["v2"] == "prepared"
    assert _history(runtime) == before


@pytest.mark.parametrize("damage", ["missing", "wrong_root", "unknown_field", "not_synthetic", "hook_enabled"])
def test_isolated_cli_rejects_missing_or_unapproved_lab_without_side_effects(lab, damage):
    runtime, _record, _uri = lab
    marker_path = runtime.paths.runtime_root / "p26a-lab.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if damage == "missing":
        marker_path.unlink()
    else:
        if damage == "wrong_root": marker["project_root"] += "-different"
        elif damage == "unknown_field": marker["anything"] = True
        elif damage == "not_synthetic": marker["data_origin"] = "private"
        else: marker["hook_policy"] = "enabled"
        marker_path.write_text(canonical_json(marker), encoding="utf-8")
    before = _history(runtime)
    code, stdout, stderr = _recall(lab)
    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["reason"] == "lab_not_authorized"
    assert not (runtime.paths.runtime_root / "injections").exists()
    assert _history(runtime) == before


@pytest.mark.parametrize("option,value", [("--session-id", "other_session"), ("--turn-id", "other_turn"), ("--prompt-event-uri", "codex-event://p26a_session/999999")])
def test_isolated_cli_wrong_source_fails_closed(lab, option, value):
    runtime, _record, _uri = lab
    before = _history(runtime)
    code, stdout, stderr = _recall(lab, overrides=((option, value),))
    assert code == 1
    assert stdout == ""
    assert json.loads(stderr)["reason"] == "source_mismatch"
    assert _history(runtime) == before


def test_manifest_commit_failure_emits_only_verified_legacy(lab, monkeypatch):
    runtime, _record, _uri = lab
    def fail_commit(self, manifest, *, deadline):
        raise PermissionError("private absolute path must not be disclosed")
    monkeypatch.setattr(InjectionStore, "commit", fail_commit)
    before = _history(runtime)
    code, stdout, stderr = _recall(lab)
    assert code == 0
    document = json.loads(stdout)
    assert document["status"] == "legacy_fallback"
    assert document["mode"] == "legacy"
    assert "CODEX MEMORY PROTOCOL v1" in document["packet"]
    assert document["selected_ids"] == []
    assert json.loads(stderr)["injection_id"] is None
    assert "private absolute path" not in stdout + stderr
    assert not list((runtime.paths.runtime_root / "injections").glob("inj_*/manifest.json"))
    assert _history(runtime) == before


def test_isolated_cli_never_constructs_provider(lab, monkeypatch):
    from codex_memory.retrieval import config
    def forbidden(*args, **kwargs):
        pytest.fail("isolated deterministic recall must never construct a provider")
    monkeypatch.setattr(config, "provider_from_config", forbidden)
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    assert json.loads(stdout)["status"] == "phase2"


@pytest.mark.parametrize("damage", ["absent", "corrupt"])
def test_unavailable_cache_has_explicit_legacy_fallback_and_no_new_manifest(lab, damage):
    runtime, _record, _uri = lab
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    if damage == "absent":
        shutil.rmtree(paths.retrieval_root)
    else:
        paths.database_path.write_bytes(b"not a SQLite database")
    before = _history(runtime)
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    body = json.loads(stdout)
    assert body["status"] == "legacy_fallback"
    assert body["degraded_reasons"] == ["retrieval_degraded"]
    assert json.loads(stderr)["injection_id"] is None
    assert not list((runtime.paths.runtime_root / "injections").glob("inj_*/manifest.json"))
    assert _history(runtime) == before


def test_canonical_schema_mismatch_cannot_emit_any_packet(lab):
    runtime, _record, _uri = lab
    trigger = runtime.connection.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'trigger' AND tbl_name = 'memory_records' ORDER BY name LIMIT 1"
    ).fetchone()[0]
    runtime.connection.execute('DROP TRIGGER "' + trigger + '"')
    runtime.connection.commit()
    before = _history(runtime)
    code, stdout, stderr = _recall(lab)
    assert (code, stdout) == (1, "")
    assert json.loads(stderr)["reason"] == "canonical_invalid"
    assert _history(runtime) == before


def test_post_validation_source_read_io_error_is_canonical_unavailable(lab, monkeypatch):
    from codex_memory.retrieval import isolated_cli
    def broken_read(*args, **kwargs):
        error = sqlite3.OperationalError("private storage location")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        raise error
    monkeypatch.setattr(isolated_cli, "resolve_event_evidence", broken_read)
    code, stdout, stderr = _recall(lab)
    assert (code, stdout) == (1, "")
    assert json.loads(stderr)["reason"] == "canonical_unavailable"
    assert "private storage" not in stderr


def test_failed_manifest_reopen_never_emits_new_packet(lab, monkeypatch):
    from codex_memory.injection_manifest import InjectionAuditError
    def broken_load(*args, **kwargs):
        raise InjectionAuditError("tampered")
    monkeypatch.setattr(InjectionStore, "load", broken_load)
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    assert json.loads(stdout)["status"] == "legacy_fallback"
    assert json.loads(stderr)["injection_id"] is None


def test_blob_write_failure_never_emits_new_packet(lab, monkeypatch):
    from codex_memory.retrieval import injection_audit
    def broken_blob(*args, **kwargs):
        raise PermissionError("private blob location")
    monkeypatch.setattr(injection_audit, "put_audit_blob", broken_blob)
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    assert json.loads(stdout)["status"] == "legacy_fallback"
    assert json.loads(stderr)["injection_id"] is None


def test_lifecycle_change_after_query_discards_old_selection(lab, monkeypatch):
    runtime, record_id, _uri = lab
    original = RetrievalService.query
    def query_then_invalidate(self, request, **kwargs):
        result = original(self, request, **kwargs)
        apply_action(runtime.connection, record_id=record_id, action="invalidate", idempotency_key="p26a-race-invalidate")
        runtime.connection.commit()
        return result
    monkeypatch.setattr(RetrievalService, "query", query_then_invalidate)
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    body = json.loads(stdout)
    assert body["status"] == "legacy_fallback"
    assert "Use WAL in the isolated sandbox." not in body["packet"]
    assert json.loads(stderr)["injection_id"] is None


def test_cli_hard_character_cap_holds(lab):
    code, stdout, stderr = _recall(lab, overrides=(("--max-chars", "1800"),))
    assert code == 0, stderr
    assert len(json.loads(stdout)["packet"]) <= 1800


def test_expired_deadline_never_starts_storage_or_emits(lab, monkeypatch):
    from codex_memory.retrieval import isolated_cli
    values = iter((10_000_000_000, 16_000_000_000))
    monkeypatch.setattr(isolated_cli.time, "monotonic_ns", lambda: next(values, 16_000_000_000))
    code, stdout, stderr = _recall(lab)
    assert (code, stdout) == (1, "")
    assert json.loads(stderr)["reason"] == "deadline_exhausted"


def test_manifest_load_accepts_caller_deadline_without_restarting(lab):
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.retrieval.deadline import DeadlineBudget
    runtime, _record, _uri = lab
    code, _stdout, stderr = _recall(lab)
    assert code == 0
    store = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), BlobStore(runtime.paths.blobs_dir))
    with pytest.raises(InjectionAuditError) as failed:
        store.load(json.loads(stderr)["injection_id"], deadline=DeadlineBudget(1, 1))
    assert failed.value.reason.value == "deadline_exhausted"


def test_stdout_failure_cannot_claim_emission(lab):
    class FailedOutput(StringIO):
        def write(self, text):
            raise BrokenPipeError("client disconnected")
    code, stdout, stderr = _recall(lab, stdout=FailedOutput())
    assert (code, stdout) == (1, "")
    # An exception from write cannot prove whether a different real sink had
    # already accepted bytes. Do not claim zero delivery or erase its manifest.
    assert json.loads(stderr)["delivery"] == "delivery_unknown"
    assert json.loads(stderr)["injection_id"].startswith("inj_")
    assert json.loads(stderr)["host_acknowledgement"] == "unknown"
    assert json.loads(stderr)["reason"] == "output_unavailable"


def test_installed_cli_returns_exact_wire_and_manifest_without_hook_activation(lab):
    runtime, _record, uri = lab
    root = runtime.paths.repo_root
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    executable = shutil.which("codex-memory-hook")
    assert executable is not None
    before = _history(runtime)
    child = subprocess.run([executable, "retrieval", "recall", "--isolated-cli", "--session-id", "p26a_session",
        "--turn-id", "p26a_turn", "--context-id", "installed_cli", "--query", "isolated WAL",
        "--prompt-event-uri", uri], cwd=root, capture_output=True, timeout=10, check=False)
    assert child.returncode == 0, child.stderr
    body, metadata = json.loads(child.stdout), json.loads(child.stderr)
    assert child.stdout == (canonical_json(body) + "\n").encode("utf-8")
    verified = InjectionStore(InjectionPaths.for_repo(root), BlobStore(runtime.paths.blobs_dir)).load(metadata["injection_id"])
    assert verified.serialized_response_bytes == child.stdout
    assert body["status"] == "phase2"
    assert not (root / ".codex" / "hooks.json").exists()
    assert _history(runtime) == before


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_redirected_lab_marker_is_rejected(lab, tmp_path, link_kind):
    runtime, _record, _uri = lab
    marker = runtime.paths.runtime_root / "p26a-lab.json"
    target = tmp_path / "outside-marker.json"
    target.write_bytes(marker.read_bytes())
    marker.unlink()
    try:
        if link_kind == "symlink": os.symlink(target, marker)
        else: os.link(target, marker)
    except OSError as error:
        pytest.skip(f"marker fixture {link_kind} creation unavailable: {error}")
    code, stdout, stderr = _recall(lab)
    assert (code, stdout) == (2, "")
    assert json.loads(stderr)["reason"] == "lab_not_authorized"


def test_cross_project_and_cross_session_memories_are_not_returned(lab):
    from codex_memory.candidate_memory import capture_candidate
    runtime, _record, _uri = lab
    foreign = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key="another-project",
        topic_key="lab.foreign", statement="FOREIGN_PROJECT_ONLY", value=None, evidence_uris=(), idempotency_key="foreign-record")
    apply_action(runtime.connection, record_id=foreign.record_id, action="activate", idempotency_key="foreign-activate")
    capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key=str(runtime.paths.repo_root),
        record_type="assumption", scope_kind="project", scope_key=str(runtime.paths.repo_root),
        topic_key="lab.temporary", statement="FOREIGN_SESSION_ONLY", value=None, evidence_uris=(),
        session_id="other_session", turn_id="other_turn", context_id="other_context", idempotency_key="foreign-temporary")
    assert RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)).rebuild().status.value == "current"
    code, stdout, stderr = _recall(lab)
    assert code == 0, stderr
    assert json.loads(stdout)["status"] == "phase2"
    assert "FOREIGN_PROJECT_ONLY" not in stdout
    assert "FOREIGN_SESSION_ONLY" not in stdout


def test_changed_source_blob_cannot_be_used_as_verified_provenance(lab):
    runtime, _record, _uri = lab
    digest = runtime.connection.execute("SELECT payload_blob_sha256 FROM events WHERE event_id=1").fetchone()[0]
    (runtime.paths.blobs_dir / "sha256" / digest[:2] / (digest + ".bin")).write_bytes(b"tampered prompt")
    code, stdout, stderr = _recall(lab)
    assert (code, stdout) == (1, "")
    assert json.loads(stderr)["reason"] == "source_mismatch"


def test_lab_read_deadline_keeps_timeout_classification(lab, monkeypatch):
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.retrieval import isolated_cli
    def late_read(*args, **kwargs):
        raise InjectionAuditError("deadline_exhausted")
    monkeypatch.setattr(isolated_cli, "_read", late_read)
    code, stdout, stderr = _recall(lab)
    assert (code, stdout) == (1, "")
    assert json.loads(stderr)["reason"] == "deadline_exhausted"


def test_stderr_failure_cannot_certify_cli_success(lab):
    runtime, _record, uri = lab
    class FailedMetadata(StringIO):
        def write(self, text):
            raise BrokenPipeError("metadata receiver closed")
    output = StringIO()
    code = run_retrieval_command(["retrieval", "recall", "--isolated-cli", "--session-id", "p26a_session",
        "--turn-id", "p26a_turn", "--context-id", "stderr-failure", "--query", "WAL",
        "--prompt-event-uri", uri], output, FailedMetadata(), lambda _cwd: runtime.paths.repo_root)
    assert code == 1
    assert json.loads(output.getvalue())["status"] == "phase2"


def test_manifest_configuration_identity_binds_effective_workspace_scope(lab):
    runtime, _record, _uri = lab
    code, _stdout, metadata = _recall(lab)
    assert code == 0
    store = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), BlobStore(runtime.paths.blobs_dir))
    first = store.load(json.loads(metadata)["injection_id"]).manifest
    codex = runtime.paths.repo_root / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text('[scope]\nworkspace_keys = ["permitted-workspace"]\n', encoding="utf-8")
    code, _stdout, metadata = _recall(lab)
    assert code == 0
    second = store.load(json.loads(metadata)["injection_id"]).manifest
    assert second.body["request"]["workspace_scope_keys"] == ("permitted-workspace",)
    assert first.body["identities"]["config_sha256"] != second.body["identities"]["config_sha256"]


def test_isolated_cli_does_not_silently_ignore_query_explain_option(lab):
    runtime, _record, uri = lab
    stdout, stderr = StringIO(), StringIO()
    code = run_retrieval_command(["retrieval", "recall", "--isolated-cli", "--explain", "--session-id", "p26a_session",
        "--turn-id", "p26a_turn", "--context-id", "cli", "--query", "WAL", "--prompt-event-uri", uri],
        stdout, stderr, lambda _cwd: runtime.paths.repo_root)
    assert (code, stdout.getvalue()) == (2, "")
    assert json.loads(stderr.getvalue())["reason"] == "invalid_input"


def test_caller_deadline_cannot_expand_existing_manifest_read_cap(lab, monkeypatch):
    from contextlib import contextmanager
    from codex_memory import injection_store
    from codex_memory.retrieval.deadline import DeadlineBudget
    runtime, _record, _uri = lab
    code, _stdout, metadata = _recall(lab)
    assert code == 0
    store = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), BlobStore(runtime.paths.blobs_dir))
    original = injection_store._serialized
    seen = []
    @contextmanager
    def observed(paths, deadline):
        seen.append(deadline.deadline_ns)
        with original(paths, deadline):
            yield
    monkeypatch.setattr(injection_store, "_serialized", observed)
    monkeypatch.setattr(injection_store.time, "monotonic_ns", lambda: 10_000_000_000)
    store.load(json.loads(metadata)["injection_id"], deadline=DeadlineBudget(10_000_000_000, 70_000_000_000))
    assert seen == [15_000_000_000]


def test_post_emission_guard_failure_keeps_delivery_fact_but_fails_completion(lab, monkeypatch):
    from contextlib import contextmanager
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.retrieval import isolated_cli
    original = isolated_cli._directories
    @contextmanager
    def failed_exit(*args, **kwargs):
        with original(*args, **kwargs) as held:
            yield held
        raise InjectionAuditError("deadline_exhausted")
    monkeypatch.setattr(isolated_cli, "_directories", failed_exit)
    code, stdout, stderr = _recall(lab)
    assert code == 1
    assert json.loads(stdout)["status"] == "phase2"
    metadata = json.loads(stderr)
    assert metadata["delivery"] == "emitted"
    assert metadata["reason"] == "deadline_exhausted"
    assert metadata["injection_id"].startswith("inj_")


@pytest.mark.parametrize("fault", ["text_short", "binary_short", "text_flush", "binary_flush", "write_then_raise"])
def test_output_fault_retains_uncertain_delivery_and_committed_identity(lab, fault):
    class FaultyText(StringIO):
        def write(self, value):
            if fault == "text_short":
                return super().write(value[:len(value) // 2])
            if fault == "write_then_raise":
                super().write(value)
                raise OSError("write completed locally before failure")
            return super().write(value)

        def flush(self):
            if fault == "text_flush":
                raise OSError("flush failed after writing")
            return super().flush()

    class FaultyBinary(BytesIO):
        def write(self, value):
            if fault == "binary_short":
                return super().write(value[:len(value) // 2])
            return super().write(value)

        def flush(self):
            if fault == "binary_flush":
                raise OSError("flush failed after writing")
            return super().flush()

    class BinaryOutput:
        def __init__(self):
            self.buffer = FaultyBinary()

        def getvalue(self):
            return self.buffer.getvalue().decode("utf-8")

    output = BinaryOutput() if fault.startswith("binary") else FaultyText()
    runtime, _record, _uri = lab
    before = _history(runtime)
    code, stdout, stderr = _recall(lab, stdout=output)
    assert code == 1
    assert stdout, "the failure seam must actually write bytes before failing"
    metadata = json.loads(stderr)
    assert metadata["delivery"] == "delivery_unknown"
    assert metadata["reason"] == "output_unavailable"
    assert metadata["host_acknowledgement"] == "unknown"
    assert metadata["injection_id"].startswith("inj_")
    verified = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), BlobStore(runtime.paths.blobs_dir)).load(metadata["injection_id"])
    assert verified.serialized_response_bytes.startswith(stdout.encode("utf-8"))
    if fault.endswith("short"):
        assert len(stdout.encode("utf-8")) < len(verified.serialized_response_bytes)
    else:
        assert stdout.encode("utf-8") == verified.serialized_response_bytes
    assert verified.manifest.body["delivery"]["v2"] == "prepared"
    assert _history(runtime) == before


@pytest.mark.parametrize("fault", ["short", "flush"])
def test_legacy_output_fault_stays_uncertain_without_inventing_manifest(lab, monkeypatch, fault):
    def failed_commit(*args, **kwargs):
        raise PermissionError("fixture manifest unavailable")
    monkeypatch.setattr(InjectionStore, "commit", failed_commit)

    class FaultyLegacy(StringIO):
        def write(self, value):
            return super().write(value[:len(value) // 2] if fault == "short" else value)

        def flush(self):
            if fault == "flush":
                raise OSError("flush failed after legacy bytes were written")
            return super().flush()

    runtime, _record, _uri = lab
    before = _history(runtime)
    code, stdout, stderr = _recall(lab, stdout=FaultyLegacy())
    assert code == 1 and stdout
    metadata = json.loads(stderr)
    assert metadata["delivery"] == "delivery_unknown"
    assert metadata["reason"] == "output_unavailable"
    assert metadata["injection_id"] is None
    assert metadata["host_acknowledgement"] == "unknown"
    if fault == "flush":
        assert json.loads(stdout)["status"] == "legacy_fallback"
    assert not list((runtime.paths.runtime_root / "injections").glob("inj_*/manifest.json"))
    assert _history(runtime) == before
