from __future__ import annotations

import hashlib
from io import StringIO
import json
import importlib.util
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import pytest

from codex_memory import cli
from codex_memory.blob_store import BlobStore
from codex_memory.injection_store import InjectionPaths, InjectionStore
from codex_memory.ledger_common import canonical_json
from codex_memory.memory_context import build_memory_packet, current_context_id
from codex_memory.memory_evidence import event_uri
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from codex_memory.structured_ledger import apply_action, propose_record


@pytest.fixture
def prompt_lab(runtime):
    root = runtime.paths.repo_root
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    marker = {
        "format": "codex-memory.p26b-prompt-lab.v1", "project_root": str(root.resolve()),
        "data_origin": "synthetic", "provider_policy": "disabled", "hook_policy": "isolated_command",
        "session_id": "p26b_session",
    }
    (runtime.paths.runtime_root / "p26b-lab.json").write_text(canonical_json(marker), encoding="utf-8")
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project",
        scope_key=str(root), topic_key="p26b.storage", statement="沙箱采用 WAL，保留原子事务。", value=None,
        evidence_uris=(), idempotency_key="p26b-storage")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="p26b-active")
    assert RetrievalService(RetrievalPaths.for_repo(root)).rebuild().status.value == "current"
    return runtime, record.record_id


def _payload(prompt_lab, **changes):
    runtime, _record_id = prompt_lab
    value = {"hook_event_name": "UserPromptSubmit", "cwd": str(runtime.paths.repo_root),
        "session_id": "p26b_session", "turn_id": "p26b_turn", "prompt": "沙箱 WAL 事务如何处理？"}
    value.update(changes)
    return value


def _invoke(prompt_lab, *, payload=None, stdout=None, started_ns=None, arguments=None):
    output, errors = stdout if stdout is not None else StringIO(), StringIO()
    code = cli.main(arguments or ["user-prompt-submit", "--isolated-active"],
        stdin=StringIO(json.dumps(payload or _payload(prompt_lab), ensure_ascii=False)),
        stdout=output, stderr=errors,
        _process_started_ns=time.monotonic_ns() if started_ns is None else started_ns)
    return code, output.getvalue(), errors.getvalue()


def _metadata(stderr):
    assert stderr.startswith("{"), stderr
    return json.loads(stderr)


def _memory_contents(runtime):
    tables = [row[0] for row in runtime.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memory_%' ORDER BY name"
    )]
    return {table: tuple(tuple(row) for row in runtime.connection.execute(f'SELECT * FROM "{table}"'))
        for table in tables}


def _verified(prompt_lab, injection_id):
    runtime, _record_id = prompt_lab
    return InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), BlobStore(runtime.paths.blobs_dir)).load(injection_id)


def test_isolated_prompt_reopens_exact_hook_wire_before_first_output(prompt_lab, monkeypatch):
    runtime, record_id = prompt_lab
    before = _memory_contents(runtime)
    actual_load = InjectionStore.load
    opened = []

    def load(self, injection_id, **kwargs):
        result = actual_load(self, injection_id, **kwargs)
        opened.append(result)
        return result

    class CheckedOutput(StringIO):
        def write(self, value):
            assert opened, "no active bytes before actual manifest reopen"
            assert opened[-1].serialized_response_bytes == value.encode("utf-8")
            return super().write(value)

    monkeypatch.setattr(InjectionStore, "load", load)
    code, stdout, stderr = _invoke(prompt_lab, stdout=CheckedOutput())
    assert code == 0, stderr
    metadata = _metadata(stderr)
    assert metadata["delivery"] == "emitted"
    assert metadata["output_mode"] == "isolated_prompt"
    assert metadata["source_origin"] == "offline"
    assert metadata["host_acknowledgement"] == "unknown"
    document = json.loads(stdout)
    assert set(document) == {"hookSpecificOutput"}
    assert document["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    packet = document["hookSpecificOutput"]["additionalContext"]
    assert "沙箱采用 WAL，保留原子事务。" in packet
    assert len(packet) <= 4000
    verified = _verified(prompt_lab, metadata["injection_id"])
    assert verified.serialized_response_bytes == stdout.encode("utf-8")
    assert verified.prepared_packet_bytes == packet.encode("utf-8")
    assert verified.manifest.body["delivery"]["v2"] == "prepared"
    assert verified.manifest.body["source"]["origin"] == "offline"
    assert verified.manifest.body["request"]["mode"] == "shadow"
    assert record_id in [item["candidate_id"] for item in verified.manifest.body["candidates"]["final_selected"]]
    rows = runtime.connection.execute("SELECT event_id,session_id,turn_id,payload_blob_sha256 FROM events").fetchall()
    assert len(rows) == 1
    assert (rows[0][1], rows[0][2]) == ("p26b_session", "p26b_turn")
    assert verified.manifest.body["source"]["event_id"] == rows[0][0]
    assert verified.manifest.body["source"]["event_blob_sha256"] == rows[0][3]
    assert _memory_contents(runtime) == before


@pytest.mark.parametrize("damage", ["missing", "wrong_root", "wrong_session", "unknown_field", "private", "registered_hook"])
def test_isolated_prompt_authorization_precedes_runtime_and_any_write(prompt_lab, monkeypatch, damage):
    runtime, _record_id = prompt_lab
    marker_path = runtime.paths.runtime_root / "p26b-lab.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if damage == "missing":
        marker_path.unlink()
    elif damage == "registered_hook":
        (runtime.paths.repo_root / ".codex").mkdir()
        (runtime.paths.repo_root / ".codex" / "hooks.json").write_text("{}", encoding="utf-8")
    else:
        if damage == "wrong_root": marker["project_root"] += "-other"
        elif damage == "wrong_session": marker["session_id"] = "other-session"
        elif damage == "unknown_field": marker["any"] = True
        else: marker["data_origin"] = "private"
        marker_path.write_text(canonical_json(marker), encoding="utf-8")
    before = tuple(runtime.connection.iterdump())

    def forbidden_runtime(_root):
        pytest.fail("authorization must precede real runtime initialization")

    monkeypatch.setattr(cli, "_construct_runtime", forbidden_runtime)
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 2 and stdout == ""
    assert _metadata(stderr)["reason"] == "lab_not_authorized"
    assert tuple(runtime.connection.iterdump()) == before
    assert not InjectionPaths.for_repo(runtime.paths.repo_root).injections_dir.exists()


def test_isolated_prompt_does_not_use_cli_or_new_phase_budget(prompt_lab, monkeypatch):
    runtime, _record_id = prompt_lab
    start = time.monotonic_ns()
    deadlines = []
    actual_load = InjectionStore.load

    def load(self, injection_id, **kwargs):
        if "deadline" in kwargs:
            deadlines.append(kwargs["deadline"])
        return actual_load(self, injection_id, **kwargs)

    monkeypatch.setattr(InjectionStore, "load", load)
    code, _stdout, stderr = _invoke(prompt_lab, started_ns=start)
    assert code == 0, stderr
    assert deadlines and deadlines[0].started_ns == start
    assert deadlines[0].deadline_ns == start + 1000 * 1_000_000
    assert _verified(prompt_lab, _metadata(stderr)["injection_id"]).manifest.body["timing"]["budget_ms"] == 1000


def test_expired_isolated_prompt_does_not_initialize_or_append(prompt_lab, monkeypatch):
    runtime, _record_id = prompt_lab
    before = tuple(runtime.connection.iterdump())

    def forbidden_runtime(_root):
        pytest.fail("expired attempt must not dispatch runtime")

    monkeypatch.setattr(cli, "_construct_runtime", forbidden_runtime)
    code, stdout, stderr = _invoke(prompt_lab, started_ns=time.monotonic_ns() - 1600 * 1_000_000)
    assert code != 0 and stdout == ""
    assert _metadata(stderr)["reason"] == "deadline_exhausted"
    assert tuple(runtime.connection.iterdump()) == before


def test_unflagged_hook_ignores_sandbox_marker_and_emits_original_legacy(prompt_lab):
    runtime, _record_id = prompt_lab
    code, stdout, stderr = _invoke(prompt_lab, arguments=["user-prompt-submit"])
    assert code == 0 and stderr == ""
    context = current_context_id(runtime.connection, session_id="p26b_session")
    event_id = runtime.events.highwater("p26b_session")
    legacy = build_memory_packet(runtime.connection, BlobStore(runtime.paths.blobs_dir),
        session_id="p26b_session", turn_id="p26b_turn", context_id=context,
        scope_key=str(runtime.paths.repo_root),
        prompt_event_uri=event_uri("p26b_session", event_id), query=_payload(prompt_lab)["prompt"], max_chars=4000)
    expected = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": legacy}}
    assert stdout == json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert not InjectionPaths.for_repo(runtime.paths.repo_root).injections_dir.exists()


def test_locked_canonical_writer_does_not_spend_five_seconds_in_runtime_initialization(prompt_lab):
    runtime, _record_id = prompt_lab
    before = tuple(runtime.connection.iterdump())
    locker = sqlite3.connect(runtime.paths.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        code, stdout, stderr = _invoke(prompt_lab)
        elapsed = time.monotonic() - started
    finally:
        locker.rollback()
        locker.close()
    assert elapsed < 1.5, f"shared Hook process budget exceeded: {elapsed}"
    assert code == 1 and stdout == ""
    assert _metadata(stderr)["reason"] == "runtime_unavailable"
    assert tuple(runtime.connection.iterdump()) == before


def test_exclusive_retrieval_lock_returns_verified_legacy_inside_hook_budget(prompt_lab):
    runtime, _record_id = prompt_lab
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    before = _memory_contents(runtime)
    locker = sqlite3.connect(paths.database_path, isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        code, stdout, stderr = _invoke(prompt_lab)
        elapsed = time.monotonic() - started
    finally:
        locker.rollback()
        locker.close()
    assert elapsed < 1.5, f"retrieval lock escaped Hook budget: {elapsed}"
    assert code == 0, stderr
    metadata = _metadata(stderr)
    assert metadata["delivery"] == "legacy_fallback" and metadata["injection_id"] is None
    assert "沙箱采用 WAL，保留原子事务。" in json.loads(stdout)["hookSpecificOutput"]["additionalContext"]
    assert _memory_contents(runtime) == before


def test_runtime_close_failure_cannot_escape_or_certify_completion(prompt_lab, monkeypatch):
    original = cli._construct_runtime

    class ClosingConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def close(self):
            self.connection.close()
            raise OSError("PRIVATE_CLOSE_ERROR")

    def runtime_with_failing_close(*args, **kwargs):
        value = original(*args, **kwargs)
        from dataclasses import replace
        return replace(value, connection=ClosingConnection(value.connection))

    monkeypatch.setattr(cli, "_construct_runtime", runtime_with_failing_close)
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 1 and stdout
    metadata = _metadata(stderr)
    assert metadata["delivery"] == "emitted"
    assert metadata["reason"] == "runtime_close_unavailable"
    assert metadata["injection_id"].startswith("inj_")
    assert "PRIVATE_CLOSE_ERROR" not in stderr
    assert _verified(prompt_lab, metadata["injection_id"]).serialized_response_bytes == stdout.encode("utf-8")


@pytest.mark.parametrize("fault", ["commit", "reopen", "blob"])
def test_isolated_prompt_storage_failure_emits_only_fresh_legacy(prompt_lab, monkeypatch, fault):
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.retrieval import injection_audit
    original = InjectionStore.commit if fault == "commit" else InjectionStore.load

    def failed(*args, **kwargs):
        raise InjectionAuditError("unavailable")

    if fault == "blob":
        monkeypatch.setattr(injection_audit, "put_audit_blob", failed)
    else:
        monkeypatch.setattr(InjectionStore, "commit" if fault == "commit" else "load", failed)
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 0, stderr
    metadata = _metadata(stderr)
    assert metadata["delivery"] == "legacy_fallback" and metadata["injection_id"] is None
    assert "UNVERIFIED TEMPORARY" not in json.loads(stdout)["hookSpecificOutput"]["additionalContext"]
    runtime, _record_id = prompt_lab
    legacy = build_memory_packet(runtime.connection, BlobStore(runtime.paths.blobs_dir),
        session_id="p26b_session", turn_id="p26b_turn", context_id=current_context_id(runtime.connection, session_id="p26b_session"),
        scope_key=str(runtime.paths.repo_root), prompt_event_uri=event_uri("p26b_session", runtime.events.highwater("p26b_session")),
        query=_payload(prompt_lab)["prompt"], max_chars=4000)
    assert json.loads(stdout)["hookSpecificOutput"]["additionalContext"] == legacy


def test_isolated_prompt_lifecycle_race_cannot_emit_invalidated_memory(prompt_lab, monkeypatch):
    runtime, record_id = prompt_lab
    original = RetrievalService.query

    def query_then_invalidate(self, request, **kwargs):
        result = original(self, request, **kwargs)
        apply_action(runtime.connection, record_id=record_id, action="invalidate", idempotency_key="p26b-race")
        return result

    monkeypatch.setattr(RetrievalService, "query", query_then_invalidate)
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 0, stderr
    assert _metadata(stderr)["delivery"] == "legacy_fallback"
    assert _metadata(stderr)["injection_id"] is None
    assert "沙箱采用 WAL，保留原子事务。" not in json.loads(stdout)["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("logical_elapsed_ms", [1100, 1460])
def test_enhancement_expiry_keeps_only_original_fresh_legacy_window(prompt_lab, monkeypatch, logical_elapsed_ms):
    from codex_memory.retrieval import isolated_hook
    runtime, record_id = prompt_lab
    started_ns = time.monotonic_ns()
    clock = [started_ns]
    actual_query = RetrievalService.query
    actual_legacy = isolated_hook._legacy_and_source
    query_deadlines, legacy_deadlines = [], []

    def query_then_expire(self, request, **kwargs):
        result = actual_query(self, request, **kwargs)
        query_deadlines.append(kwargs["deadline"])
        # Use the real lifecycle API so a cached pre-query Legacy string is
        # observably wrong. This is a deterministic state-machine budget test,
        # not a wall-clock/process performance claim or a production clock change.
        apply_action(runtime.connection, record_id=record_id, action="invalidate",
            idempotency_key=f"p26b-expiry-{logical_elapsed_ms}")
        clock[0] = started_ns + logical_elapsed_ms * 1_000_000
        return result

    def fresh_legacy(*args, **kwargs):
        legacy_deadlines.append(kwargs["deadline"])
        return actual_legacy(*args, **kwargs)

    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(RetrievalService, "query", query_then_expire)
    monkeypatch.setattr(isolated_hook, "_legacy_and_source", fresh_legacy)
    code, stdout, stderr = _invoke(prompt_lab, started_ns=started_ns)
    assert len(query_deadlines) == 1
    assert query_deadlines[0].started_ns == started_ns
    assert query_deadlines[0].deadline_ns == started_ns + 1000 * 1_000_000
    metadata = _metadata(stderr)
    assert metadata["reason"] == "deadline_exhausted"
    assert metadata["injection_id"] is None
    assert metadata["cleanup_status"] == "complete"
    if logical_elapsed_ms == 1100:
        assert code == 0, stderr
        assert metadata["delivery"] == "legacy_fallback"
        assert [(item.started_ns, item.deadline_ns) for item in legacy_deadlines] == [
            (started_ns, started_ns + 1000 * 1_000_000),
            (started_ns, started_ns + 1450 * 1_000_000),
        ]
        assert "沙箱采用 WAL，保留原子事务。" not in json.loads(stdout)["hookSpecificOutput"]["additionalContext"]
    else:
        assert code == 1 and stdout == ""
        assert metadata["delivery"] == "not_emitted"
        assert len(legacy_deadlines) == 1


def test_unmodified_directory_guard_keeps_its_original_expiry_check(prompt_lab, monkeypatch):
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.injection_store import _directories, _read
    from codex_memory.retrieval.deadline import DeadlineBudget
    runtime, _record_id = prompt_lab
    started_ns = time.monotonic_ns()
    clock = [started_ns]
    deadline = DeadlineBudget(started_ns, started_ns + 1000 * 1_000_000)
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    with pytest.raises(InjectionAuditError) as failure:
        with _directories(runtime.paths.runtime_root, create=False, deadline=deadline) as held:
            assert _read(held, runtime.paths.runtime_root / "p26b-lab.json", limit=4096, deadline=deadline)
            clock[0] = started_ns + 1100 * 1_000_000
    assert failure.value.reason.value == "deadline_exhausted"


@pytest.mark.parametrize("violation", ["reset_start", "shorter", "creation", "install", "acquisition_expired"])
def test_read_only_lease_cannot_reset_or_extend_authorization_work(prompt_lab, monkeypatch, violation):
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.injection_store import _directories
    from codex_memory.retrieval.deadline import DeadlineBudget
    runtime, _record_id = prompt_lab
    started_ns = time.monotonic_ns()
    deadline = DeadlineBudget(started_ns, started_ns + 1000 * 1_000_000)
    release = DeadlineBudget(started_ns, started_ns + 1450 * 1_000_000)
    if violation == "reset_start":
        release = DeadlineBudget(started_ns + 1_000_000, release.deadline_ns)
    elif violation == "shorter":
        release = DeadlineBudget(started_ns, deadline.deadline_ns - 1_000_000)
    elif violation == "acquisition_expired":
        monkeypatch.setattr(time, "monotonic_ns", lambda: started_ns + 1100 * 1_000_000)
    with pytest.raises(InjectionAuditError) as failure:
        with _directories(runtime.paths.runtime_root, create=violation == "creation", deadline=deadline,
            install=runtime.paths.runtime_root if violation == "install" else None, release_deadline=release):
            pytest.fail("invalid or expired lease must not acquire authorization handles")
    assert failure.value.reason.value == ("deadline_exhausted" if violation == "acquisition_expired" else "invalid_input")


def test_post_validation_source_io_failure_is_unavailable_not_invalid(prompt_lab, monkeypatch):
    from codex_memory.retrieval import isolated_cli

    def failed(*args, **kwargs):
        error = sqlite3.OperationalError("PRIVATE_IO_SOURCE")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        raise error

    monkeypatch.setattr(isolated_cli, "resolve_event_evidence", failed)
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 1 and stdout == ""
    assert _metadata(stderr)["reason"] == "canonical_unavailable"
    assert "PRIVATE_IO_SOURCE" not in stderr


@pytest.mark.parametrize("fault", ["short", "flush", "write_then_raise"])
def test_isolated_prompt_partial_output_cannot_be_replaced_with_legacy(prompt_lab, fault):
    class FaultyOutput(StringIO):
        def write(self, value):
            if fault == "short":
                return super().write(value[:len(value) // 2])
            written = super().write(value)
            if fault == "write_then_raise":
                raise OSError("private output")
            return written

        def flush(self):
            if fault == "flush":
                raise OSError("private flush")

    output = FaultyOutput()
    code, stdout, stderr = _invoke(prompt_lab, stdout=output)
    assert code == 1 and stdout
    metadata = _metadata(stderr)
    assert metadata["delivery"] == "delivery_unknown"
    assert metadata["reason"] == "output_unavailable"
    assert metadata["injection_id"].startswith("inj_")
    verified = _verified(prompt_lab, metadata["injection_id"])
    assert verified.serialized_response_bytes.startswith(stdout.encode("utf-8"))
    assert len(stdout) <= len(verified.serialized_response_bytes.decode("utf-8"))


def test_assembly_read_uses_existing_budget_not_default_five_second_lock_wait(prompt_lab, monkeypatch):
    from contextlib import contextmanager
    from codex_memory.retrieval import assembler
    original = assembler.open_verified_canonical_snapshot
    observed = []

    @contextmanager
    def observe(*args, **kwargs):
        with original(*args, **kwargs) as (connection, snapshot):
            observed.append(connection.execute("PRAGMA busy_timeout").fetchone()[0])
            yield connection, snapshot

    monkeypatch.setattr(assembler, "open_verified_canonical_snapshot", observe)
    code, _stdout, stderr = _invoke(prompt_lab)
    assert code == 0, stderr
    assert _metadata(stderr)["delivery"] == "emitted"
    assert observed == [0], "assembly must not reacquire the default 5000ms lock wait"


def test_provider_configuration_cannot_construct_optional_models_in_prompt_lab(prompt_lab, monkeypatch, p26_phase_trace):
    from codex_memory.retrieval import config

    def forbidden(*args, **kwargs):
        pytest.fail("isolated prompt must not construct providers")

    monkeypatch.setattr(config, "provider_from_config", forbidden)
    code, _stdout, stderr = _invoke(prompt_lab)
    assert code == 0, (stderr, p26_phase_trace)
    assert _metadata(stderr)["delivery"] == "emitted", (stderr, p26_phase_trace)


def test_prompt_scope_and_temporary_session_filters_survive_actual_emission(prompt_lab):
    from codex_memory.candidate_memory import capture_candidate
    runtime, _record_id = prompt_lab
    root = str(runtime.paths.repo_root)
    foreign = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key="foreign-project",
        topic_key="p26b.foreign", statement="FOREIGN_PROJECT_WAL", value=None, evidence_uris=(), idempotency_key="p26b-foreign")
    apply_action(runtime.connection, record_id=foreign.record_id, action="activate", idempotency_key="p26b-foreign-active")
    for session, text in [("p26b_session", "SAME_SESSION_WAL"), ("other_session", "FOREIGN_SESSION_WAL")]:
        capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key=root,
            record_type="assumption", scope_kind="project", scope_key=root, topic_key=f"p26b.{session}",
            statement=text, session_id=session, turn_id="origin_turn", context_id="origin_context", idempotency_key=session)
    assert RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)).rebuild().status.value == "current"
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 0 and _metadata(stderr)["delivery"] == "emitted"
    assert "SAME_SESSION_WAL" in stdout and "UNVERIFIED TEMPORARY" in stdout
    assert "FOREIGN_PROJECT_WAL" not in stdout and "FOREIGN_SESSION_WAL" not in stdout
    assert len(json.loads(stdout)["hookSpecificOutput"]["additionalContext"]) <= 4000


def test_invalid_canonical_cannot_be_migrated_or_used_as_legacy(prompt_lab, monkeypatch):
    runtime, _record_id = prompt_lab
    trigger = runtime.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='memory_records' ORDER BY name"
    ).fetchone()[0]
    runtime.connection.execute(f'DROP TRIGGER "{trigger}"')
    before = tuple(runtime.connection.iterdump())

    def forbidden(*args, **kwargs):
        pytest.fail("invalid canonical must not reach migration")

    monkeypatch.setattr(cli, "_construct_runtime", forbidden)
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 1 and stdout == ""
    assert _metadata(stderr)["reason"] == "canonical_invalid"
    assert tuple(runtime.connection.iterdump()) == before


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_redirected_prompt_marker_cannot_authorize_initialization(prompt_lab, kind):
    runtime, _record_id = prompt_lab
    marker = runtime.paths.runtime_root / "p26b-lab.json"
    target = runtime.paths.repo_root / "untrusted-marker.json"
    marker.rename(target)
    try:
        if kind == "symlink":
            marker.symlink_to(target)
        else:
            os.link(target, marker)
    except OSError as exc:
        pytest.skip(f"{kind} creation unavailable: {type(exc).__name__},winerror={getattr(exc, 'winerror', None)}")
    before = tuple(runtime.connection.iterdump())
    code, stdout, stderr = _invoke(prompt_lab)
    assert code == 2 and stdout == ""
    assert _metadata(stderr)["reason"] == "lab_not_authorized"
    assert tuple(runtime.connection.iterdump()) == before


@pytest.mark.parametrize("arguments", [["session-start", "--isolated-active"], ["pre-compact", "--isolated-active"],
    ["user-prompt-submit", "--isolated-active", "--isolated-active"]])
def test_isolated_flag_never_unlocks_other_hook_or_ambiguous_routes(prompt_lab, arguments):
    runtime, _record_id = prompt_lab
    before = tuple(runtime.connection.iterdump())
    code, stdout, stderr = _invoke(prompt_lab, arguments=arguments)
    assert code == 2 and stdout == "" and "unsupported command" in stderr
    assert tuple(runtime.connection.iterdump()) == before






def test_ordinary_query_keeps_existing_observer_only_deadline_semantics(prompt_lab):
    from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.deadline import DeadlineBudget
    runtime, _record_id = prompt_lab
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root))
    request = RetrievalRequest("WAL", "p26b_session", "turn", "context", None,
        RetrievalScope(str(runtime.paths.repo_root)))
    ordinary = service.query(request)
    # Before this feature, deadline without an Observer was only metadata;
    # the selected sandbox must not silently change existing callers.
    observed = service.query(request, deadline=DeadlineBudget(0, 0))
    assert observed.to_dict() == ordinary.to_dict()
