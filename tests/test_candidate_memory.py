from __future__ import annotations

from dataclasses import fields
import importlib.util
from importlib import import_module
import re
import sqlite3
from threading import Barrier, Thread

import pytest

from codex_memory.blob_store import BlobIntegrityError, BlobStore
from codex_memory.ledger_common import (
    IdempotencyConflict,
    LedgerError,
    RecordNotFound,
    ValidationError,
    VerificationError,
    _hash_preimage,
    canonical_json,
)
from codex_memory.memory_evidence import event_uri
from codex_memory.memory_security import SecretRejected
from codex_memory.paths import RuntimePaths
from codex_memory.structured_ledger import apply_action, propose_record, verify_ledger


def _candidate():
    spec = importlib.util.find_spec("codex_memory.candidate_memory")
    assert spec is not None, "candidate capture module is required"
    module = import_module("codex_memory.candidate_memory")
    for name in ("CandidateView", "CandidateActionResult", "capture_candidate"):
        assert hasattr(module, name), f"candidate capture API is missing {name}"
    return module


def _arguments(**changes):
    arguments = {
        "project_scope_key": r"C:\Work\Repo",
        "record_type": "decision",
        "scope_kind": "project",
        "scope_key": r"c:\work\repo",
        "topic_key": "storage.engine",
        "statement": "Use SQLite for the durable ledger.",
        "session_id": "session_a",
        "turn_id": "turn_1",
        "context_id": "startup",
        "idempotency_key": "capture-1",
        "value": {"engine": "sqlite3"},
        "evidence_uris": ("https://example.test/design",),
    }
    arguments.update(changes)
    return arguments


def _capture(runtime, **changes):
    return _candidate().capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        **_arguments(**changes),
    )


def _count(connection, table: str) -> int:
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _permanent(
    runtime,
    *,
    key: str = "permanent-1",
    activate_key: str | None = None,
    topic: str = "storage.engine",
    statement: str = "Use SQLite for the durable ledger.",
    value=None,
    evidence=("https://example.test/design",),
):
    if value is None:
        value = {"engine": "sqlite3"}
    record = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=r"C:\Work\Repo",
        topic_key=topic,
        statement=statement,
        value=value,
        evidence_uris=evidence,
        idempotency_key=key,
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key=activate_key or f"{key}-activate",
    )
    return record


def _review_payload(
    *,
    action: str,
    requested_outcome: str | None = None,
    policy_code: str | None = None,
    reason: str = "Reviewed evidence.",
    evidence_uris: list[str] | None = None,
    permanent_record_id: str | None = None,
    permanent_record_created: bool | None = None,
    superseded_record_id: str | None = None,
) -> dict[str, object]:
    return {
        "evidence_uris": [] if evidence_uris is None else evidence_uris,
        "permanent_record_created": permanent_record_created,
        "permanent_record_id": permanent_record_id,
        "policy_code": policy_code
        or {
            "defer": "review_deferred",
            "reject": "review_rejected",
            "expire": "review_expired",
        }.get(action, "linked_duplicate"),
        "policy_version": "phase1b-v1",
        "reason": reason,
        "requested_outcome": requested_outcome or action,
        "superseded_record_id": superseded_record_id,
    }


def _insert_review_event(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    action: str,
    payload: dict[str, object],
    key: str,
    event_hash: str | None = None,
) -> int:
    event_id = f"cevt_test_{key}"
    cursor = connection.execute(
        """
        INSERT INTO memory_candidate_events (
            event_id, candidate_id, action, payload_json, idempotency_key,
            event_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, '2026-09-01T00:00:00.000000Z')
        """,
        (
            event_id,
            candidate_id,
            action,
            canonical_json(payload),
            key,
            event_hash
            or _hash_preimage(
                {"action": action, "candidate_id": candidate_id, "payload": payload}
            ),
        ),
    )
    return cursor.lastrowid


def _apply_internal_review(runtime, captured, **changes):
    arguments = {
        "candidate_id": captured.candidate_id,
        "action": "defer",
        "requested_outcome": "defer",
        "reason": "Reviewed evidence.",
        "idempotency_key": "review-1",
        "evidence_uris": (),
        "policy_code": "review_deferred",
        "permanent_record_id": None,
        "permanent_record_created": None,
        "superseded_record_id": None,
    }
    arguments.update(changes)
    runtime.connection.execute("BEGIN IMMEDIATE")
    try:
        result = _candidate().apply_candidate_action_in_transaction(
            runtime.connection, **arguments
        )
        runtime.connection.commit()
        return result
    except BaseException:
        if runtime.connection.in_transaction:
            runtime.connection.rollback()
        raise


def test_candidate_capture_public_result_shapes_and_initial_projection(runtime) -> None:
    candidate = _candidate()
    assert tuple(field.name for field in fields(candidate.CandidateView)) == (
        "candidate_id",
        "record_type",
        "scope_kind",
        "scope_key",
        "topic_key",
        "statement",
        "value",
        "session_id",
        "turn_id",
        "context_id",
        "evidence_uris",
        "state",
        "permanent_record_id",
        "content_sha256",
        "created_at",
        "updated_at",
    )
    assert tuple(field.name for field in fields(candidate.CandidateActionResult)) == (
        "candidate_id",
        "event_id",
        "action",
        "requested_outcome",
        "policy_code",
        "state",
        "permanent_record_id",
        "permanent_record_created",
        "superseded_record_id",
        "created_at",
    )

    result = _capture(runtime)

    assert re.fullmatch(r"cand_[0-9a-f]{32}", result.candidate_id)
    assert result.state == "temporary"
    assert result.permanent_record_id is None
    assert result.scope_key == r"C:\Work\Repo"
    assert result.evidence_uris == ("https://example.test/design",)
    assert re.fullmatch(r".+\.\d{6}Z", result.created_at)
    assert result.updated_at == result.created_at
    capture_event = runtime.connection.execute(
        "SELECT action, payload_json, idempotency_key FROM memory_candidate_events "
        "WHERE candidate_id = ?",
        (result.candidate_id,),
    ).fetchone()
    assert tuple(capture_event) == ("capture", "{}", "capture-1")


def test_project_scope_uses_windows_casefold_but_workspace_preserves_caller_spelling(runtime) -> None:
    project = _capture(runtime)
    workspace = _capture(
        runtime,
        scope_kind="workspace",
        scope_key="Workspace Caller Spelling",
        idempotency_key="workspace-capture",
    )

    assert project.scope_key == r"C:\Work\Repo"
    assert workspace.scope_key == "Workspace Caller Spelling"
    with pytest.raises(ValidationError):
        _capture(runtime, scope_key=r"D:\Different\Repo", idempotency_key="different-project")


@pytest.mark.parametrize(
    "changes",
    [
        {"record_type": "unknown"},
        {"scope_kind": "unknown"},
        {"topic_key": "UPPER"},
        {"statement": "x" * 8193},
        {"value": "x" * 32769},
        {"idempotency_key": "x" * 129},
        {"idempotency_key": "has whitespace"},
        {"session_id": " "},
        {"turn_id": "x" * 257},
        {"context_id": "\t"},
        {"evidence_uris": tuple(f"https://example.test/{index}" for index in range(17))},
        {"evidence_uris": ("not-a-uri",)},
    ],
    ids=(
        "record-type",
        "scope-kind",
        "topic",
        "statement-size",
        "value-size",
        "idempotency-size",
        "idempotency-printable",
        "session",
        "turn-size",
        "context",
        "evidence-limit",
        "external-uri",
    ),
)
def test_capture_rejects_invalid_content_before_rows_exist(runtime, changes) -> None:
    with pytest.raises(ValidationError):
        _capture(runtime, **changes)

    assert _count(runtime.connection, "memory_candidates") == 0
    assert _count(runtime.connection, "memory_candidate_events") == 0
    assert _count(runtime.connection, "memory_candidate_evidence") == 0


def test_capture_rejects_synthetic_secret_before_rows_or_transaction(runtime, capsys) -> None:
    candidate = _candidate()
    statements: list[str] = []
    runtime.connection.set_trace_callback(statements.append)
    with pytest.raises(SecretRejected) as raised:
        candidate.capture_candidate(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            **_arguments(statement="token = synthetic-token-value"),
        )

    runtime.connection.set_trace_callback(None)
    assert "BEGIN IMMEDIATE" not in statements
    assert "synthetic-token-value" not in str(raised.value)
    assert capsys.readouterr().out == ""
    assert _count(runtime.connection, "memory_candidates") == 0


def test_capture_rejects_placeholder_prefix_secret_before_rows(runtime) -> None:
    with pytest.raises(SecretRejected):
        _capture(
            runtime,
            statement="token = ${TOKEN}synthetic-real-secret",
            idempotency_key="placeholder-prefix-secret",
        )

    assert _count(runtime.connection, "memory_candidates") == 0
    assert _count(runtime.connection, "memory_candidate_events") == 0
    assert _count(runtime.connection, "memory_candidate_evidence") == 0


def test_capture_resolves_and_revalidates_local_event_evidence_before_rows(runtime, prompt_envelope) -> None:
    event = runtime.events.append(prompt_envelope)
    uri = event_uri("session_a", event.event_id)
    result = _capture(runtime, evidence_uris=(uri,), idempotency_key="local-event")

    assert result.evidence_uris == (uri,)
    digest = runtime.connection.execute(
        "SELECT payload_blob_sha256 FROM events WHERE event_id = ?", (event.event_id,)
    ).fetchone()["payload_blob_sha256"]
    (runtime.paths.blobs_dir / "sha256" / digest[:2] / f"{digest}.bin").write_bytes(b"corrupt")
    with pytest.raises(BlobIntegrityError):
        _capture(runtime, evidence_uris=(uri,), idempotency_key="corrupt-event")
    assert _count(runtime.connection, "memory_candidates") == 1


def test_capture_rejects_cross_session_local_evidence_before_rows(
    runtime, prompt_envelope_factory
) -> None:
    event = runtime.events.append(
        prompt_envelope_factory("session_b", "cross-session evidence")
    )
    uri = event_uri("session_b", event.event_id)

    with pytest.raises(ValidationError, match="another session"):
        _capture(
            runtime,
            session_id="session_a",
            evidence_uris=(uri,),
            idempotency_key="cross-session-local-event",
        )

    assert _count(runtime.connection, "memory_candidates") == 0
    assert _count(runtime.connection, "memory_candidate_events") == 0
    assert _count(runtime.connection, "memory_candidate_evidence") == 0


def test_candidate_action_rejects_cross_session_local_evidence(
    runtime, prompt_envelope_factory
) -> None:
    captured = _capture(runtime)
    event = runtime.events.append(
        prompt_envelope_factory("session_b", "cross-session review evidence")
    )
    uri = event_uri("session_b", event.event_id)

    with pytest.raises(ValidationError, match="another session"):
        _candidate().apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="defer",
            reason="Evidence must remain session-bound.",
            idempotency_key="cross-session-review-event",
            evidence_uris=(uri,),
        )

    assert _count(runtime.connection, "memory_candidate_events") == 1
    assert runtime.connection.execute(
        "SELECT state FROM memory_candidate_current WHERE candidate_id = ?",
        (captured.candidate_id,),
    ).fetchone()[0] == "temporary"


def test_candidate_and_equivalent_permanent_record_share_content_identity(runtime) -> None:
    candidate = _capture(runtime)
    permanent_arguments = {
        key: value
        for key, value in _arguments().items()
        if key not in {"project_scope_key", "session_id", "turn_id", "context_id"}
    }
    permanent_arguments["scope_key"] = candidate.scope_key
    permanent = propose_record(runtime.connection, **permanent_arguments)

    assert candidate.content_sha256 == permanent.content_sha256


@pytest.mark.parametrize(
    "changes",
    [
        {"statement": "Changed candidate body.", "idempotency_key": "capture-1"},
        {"context_id": "recovered", "idempotency_key": "capture-1"},
        {
            "evidence_uris": ("https://example.test/two", "https://example.test/one"),
            "idempotency_key": "ordered-capture",
        },
    ],
    ids=("body", "context", "evidence-order"),
)
def test_capture_idempotency_conflicts_when_complete_request_differs(runtime, changes) -> None:
    if changes["idempotency_key"] == "ordered-capture":
        _capture(
            runtime,
            idempotency_key="ordered-capture",
            evidence_uris=("https://example.test/one", "https://example.test/two"),
        )
    else:
        _capture(runtime)

    with pytest.raises(IdempotencyConflict):
        _capture(runtime, **changes)


def test_exact_capture_retry_returns_original_rows(runtime) -> None:
    first = _capture(runtime)
    second = _capture(runtime)

    assert second == first
    assert _count(runtime.connection, "memory_candidates") == 1
    assert _count(runtime.connection, "memory_candidate_events") == 1
    assert _count(runtime.connection, "memory_candidate_evidence") == 1


def test_first_capture_and_retry_return_canonical_persisted_value(runtime) -> None:
    first = _capture(runtime, value=("a", "b"), idempotency_key="tuple-value")
    second = _capture(runtime, value=("a", "b"), idempotency_key="tuple-value")

    assert first.value == ["a", "b"]
    assert second.value == ["a", "b"]
    assert first == second


@pytest.mark.parametrize("corruption", ("ordinal", "hash"))
def test_exact_retry_conflicts_on_corrupt_ordered_evidence(runtime, corruption) -> None:
    captured = _capture(
        runtime,
        idempotency_key=f"evidence-{corruption}",
        evidence_uris=("https://example.test/one", "https://example.test/two"),
    )
    runtime.connection.execute("DROP TRIGGER memory_candidate_evidence_reject_update")
    if corruption == "ordinal":
        runtime.connection.execute(
            "UPDATE memory_candidate_evidence SET ordinal = 5 "
            "WHERE candidate_id = ? AND ordinal = 1",
            (captured.candidate_id,),
        )
    else:
        runtime.connection.execute(
            "UPDATE memory_candidate_evidence SET evidence_sha256 = ? "
            "WHERE candidate_id = ? AND ordinal = 1",
            ("0" * 64, captured.candidate_id),
        )

    with pytest.raises(IdempotencyConflict):
        _capture(
            runtime,
            idempotency_key=f"evidence-{corruption}",
            evidence_uris=("https://example.test/one", "https://example.test/two"),
        )


@pytest.mark.parametrize("corruption", ("extra", "missing"))
def test_exact_retry_conflicts_on_extra_or_missing_evidence(runtime, corruption) -> None:
    captured = _capture(
        runtime,
        idempotency_key=f"evidence-{corruption}",
        evidence_uris=("https://example.test/one", "https://example.test/two"),
    )
    if corruption == "extra":
        runtime.connection.execute(
            """
            INSERT INTO memory_candidate_evidence (
                evidence_id, candidate_id, ordinal, uri, evidence_sha256, created_at
            ) VALUES (?, ?, 2, ?, ?, ?)
            """,
            (
                "cevi_extra",
                captured.candidate_id,
                "https://example.test/extra",
                "0" * 64,
                captured.created_at,
            ),
        )
    else:
        runtime.connection.execute("DROP TRIGGER memory_candidate_evidence_reject_delete")
        runtime.connection.execute(
            "DELETE FROM memory_candidate_evidence WHERE candidate_id = ? AND ordinal = 1",
            (captured.candidate_id,),
        )

    with pytest.raises(IdempotencyConflict):
        _capture(
            runtime,
            idempotency_key=f"evidence-{corruption}",
            evidence_uris=("https://example.test/one", "https://example.test/two"),
        )


def test_capture_rolls_back_if_capture_event_insert_fails(runtime) -> None:
    def deny_capture_event(action, table, _column, _database, _trigger):
        if action == sqlite3.SQLITE_INSERT and table == "memory_candidate_events":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    runtime.connection.set_authorizer(deny_capture_event)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            _capture(runtime)
    finally:
        runtime.connection.set_authorizer(None)

    assert runtime.connection.in_transaction is False
    assert _count(runtime.connection, "memory_candidates") == 0
    assert _count(runtime.connection, "memory_candidate_events") == 0


def test_concurrent_same_key_capture_leaves_one_immutable_request(tmp_path) -> None:
    database = import_module("codex_memory.database")
    paths = RuntimePaths.for_repo(tmp_path)
    setup = database.connect_database(paths.db_path)
    database.migrate_database(setup)
    setup.close()
    barrier = Barrier(2)
    results = []
    failures = []

    def worker() -> None:
        connection = database.connect_database(paths.db_path)
        try:
            barrier.wait()
            results.append(
                _candidate().capture_candidate(
                    connection,
                    BlobStore(paths.blobs_dir),
                    **_arguments(idempotency_key="concurrent-capture"),
                )
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            connection.close()

    threads = [Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len({result.candidate_id for result in results}) == 1
    check = database.connect_database(paths.db_path)
    try:
        assert _count(check, "memory_candidates") == 1
        assert _count(check, "memory_candidate_events") == 1
        assert _count(check, "memory_candidate_evidence") == 1
    finally:
        check.close()


def test_candidate_lifecycle_and_exact_defer_snapshot_retry(runtime) -> None:
    module = _candidate()
    captured = _capture(runtime)
    deferred = module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action="defer",
        reason="Need more evidence.",
        idempotency_key="defer-1",
    )
    permanent = _permanent(runtime)
    promoted = _apply_internal_review(
        runtime,
        captured,
        action="promote",
        requested_outcome="promote",
        reason="Evidence confirms the durable decision.",
        idempotency_key="promote-1",
        policy_code="linked_duplicate",
        permanent_record_id=permanent.record_id,
        permanent_record_created=False,
    )
    retried = module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action="defer",
        reason="Need more evidence.",
        idempotency_key="defer-1",
    )

    assert deferred == retried
    assert deferred.state == "temporary"
    assert promoted.state == "promoted"
    assert promoted.permanent_record_id == permanent.record_id
    assert _count(runtime.connection, "memory_candidate_events") == 3


@pytest.mark.parametrize("action", ("reject", "expire"))
def test_terminal_candidate_actions_and_conflicts(runtime, action) -> None:
    module = _candidate()
    captured = _capture(runtime, idempotency_key=f"capture-{action}")
    result = module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action=action,
        reason="Reviewed outcome.",
        idempotency_key=f"{action}-1",
    )
    assert result.state == {"reject": "rejected", "expire": "expired"}[action]
    assert module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action=action,
        reason="Reviewed outcome.",
        idempotency_key=f"{action}-1",
    ) == result
    with pytest.raises(IdempotencyConflict):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="expire" if action == "reject" else "reject",
            reason="Other outcome.",
            idempotency_key="other-terminal",
        )
    with pytest.raises(IdempotencyConflict):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action=action,
            reason="Changed same-key rationale.",
            idempotency_key=f"{action}-1",
        )


def test_candidate_action_validates_limits_domain_errors_and_internal_transaction(
    runtime,
) -> None:
    module = _candidate()
    captured = _capture(runtime)
    with pytest.raises(ValidationError):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="defer",
            reason=" ",
            idempotency_key="bad",
        )
    with pytest.raises(ValidationError):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="defer",
            reason="é" * 2049,
            idempotency_key="bad-long",
        )
    with pytest.raises(ValidationError):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="promote",
            reason="Not a public non-promotion action.",
            idempotency_key="bad-action",
        )
    with pytest.raises(ValidationError):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action=[],
            reason="Malformed action type.",
            idempotency_key="bad-action-type",
        )
    with pytest.raises(ValidationError):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="defer",
            reason="Too much evidence.",
            idempotency_key="too-much-evidence",
            evidence_uris=tuple(
                f"https://example.test/{index}" for index in range(17)
            ),
        )
    with pytest.raises(RecordNotFound):
        module.apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id="cand_missing",
            action="defer",
            reason="Missing candidate.",
            idempotency_key="missing-candidate",
        )
    with pytest.raises(LedgerError, match="transaction"):
        module.apply_candidate_action_in_transaction(
            runtime.connection,
            candidate_id=captured.candidate_id,
            action="defer",
            requested_outcome="defer",
            reason="No transaction.",
            idempotency_key="no-transaction",
            evidence_uris=(),
            policy_code="review_deferred",
            permanent_record_id=None,
            permanent_record_created=None,
            superseded_record_id=None,
        )
    runtime.connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ValidationError):
            module.apply_candidate_action_in_transaction(
                runtime.connection,
                candidate_id=captured.candidate_id,
                action=[],
                reason="Malformed internal action type.",
                idempotency_key="bad-internal-action",
            )
        with pytest.raises(ValidationError):
            module.apply_candidate_action_in_transaction(
                runtime.connection,
                candidate_id=captured.candidate_id,
                action="defer",
                reason="Malformed internal evidence.",
                idempotency_key="bad-internal-evidence",
                evidence_uris=None,
            )
    finally:
        runtime.connection.rollback()

    accepted = module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action="defer",
        reason="é" * 2048,
        idempotency_key="exact-byte-limit",
    )
    assert accepted.state == "temporary"


def test_candidate_action_resolves_ordered_local_evidence_before_append(
    runtime, prompt_envelope_factory
) -> None:
    captured = _capture(runtime)
    first = runtime.events.append(prompt_envelope_factory("session_a", "first"))
    second = runtime.events.append(prompt_envelope_factory("session_a", "second"))
    evidence = (event_uri("session_a", first.event_id), event_uri("session_a", second.event_id))

    result = _candidate().apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action="defer",
        reason="Both events were reviewed.",
        idempotency_key="local-review",
        evidence_uris=evidence,
    )
    payload = runtime.connection.execute(
        "SELECT payload_json FROM memory_candidate_events WHERE event_id = ?",
        (result.event_id,),
    ).fetchone()[0]
    assert payload == canonical_json(
        _review_payload(
            action="defer",
            reason="Both events were reviewed.",
            evidence_uris=list(evidence),
        )
    )

    digest = runtime.connection.execute(
        "SELECT payload_blob_sha256 FROM events WHERE event_id = ?", (second.event_id,)
    ).fetchone()[0]
    (runtime.paths.blobs_dir / "sha256" / digest[:2] / f"{digest}.bin").write_bytes(
        b"corrupt"
    )
    before = _count(runtime.connection, "memory_candidate_events")
    assert _candidate().apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=captured.candidate_id,
        action="defer",
        reason="Both events were reviewed.",
        idempotency_key="local-review",
        evidence_uris=evidence,
    ) == result
    assert _count(runtime.connection, "memory_candidate_events") == before
    with pytest.raises(BlobIntegrityError):
        _candidate().apply_candidate_action(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            candidate_id=captured.candidate_id,
            action="defer",
            reason="Corrupt evidence must not append.",
            idempotency_key="corrupt-local-review",
            evidence_uris=(event_uri("session_a", second.event_id),),
        )
    assert _count(runtime.connection, "memory_candidate_events") == before


def test_pending_candidates_are_session_scoped_oldest_first_and_bounded(runtime) -> None:
    module = _candidate()
    first = _capture(runtime, idempotency_key="first")
    other_session = _capture(
        runtime, idempotency_key="other", session_id="session_b"
    )
    second = _capture(runtime, idempotency_key="second", topic_key="storage.second")
    third = _capture(runtime, idempotency_key="third", topic_key="storage.third")
    module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=first.candidate_id,
        action="defer",
        reason="Still pending.",
        idempotency_key="first-defer",
    )
    module.apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=second.candidate_id,
        action="reject",
        reason="No longer pending.",
        idempotency_key="second-reject",
    )

    assert [
        item.candidate_id
        for item in module.list_pending_candidates(
            runtime.connection, session_id="session_a", limit=100
        )
    ] == [first.candidate_id, third.candidate_id]
    assert [
        item.candidate_id
        for item in module.list_pending_candidates(
            runtime.connection, session_id="session_a", limit=1
        )
    ] == [first.candidate_id]
    assert [
        item.candidate_id
        for item in module.list_pending_candidates(
            runtime.connection, session_id="session_b", limit=1
        )
    ] == [other_session.candidate_id]
    for invalid in (0, 101, True):
        with pytest.raises(ValidationError):
            module.list_pending_candidates(
                runtime.connection, session_id="session_a", limit=invalid
            )


@pytest.mark.parametrize(
    ("action", "requested", "policy", "created", "superseded"),
    [
        ("defer", "defer", "review_deferred", None, None),
        ("defer", "promote", "decision_conflict", None, None),
        ("defer", "promote", "constitution_conflict", None, None),
        ("defer", "promote", "newer_evidence_required", None, None),
        ("reject", "reject", "review_rejected", None, None),
        ("expire", "expire", "review_expired", None, None),
        ("promote", "promote", "linked_duplicate", False, None),
        ("promote", "promote", "promoted_new", True, None),
    ],
)
def test_exact_review_policy_combinations_replay(
    runtime, action, requested, policy, created, superseded
) -> None:
    captured = _capture(runtime, idempotency_key=f"capture-{policy}")
    permanent = None
    if action == "promote":
        if policy == "promoted_new":
            permanent = _permanent(
                runtime,
                key=f"candidate:{captured.candidate_id}:propose",
                activate_key=f"candidate:{captured.candidate_id}:activate",
            )
        else:
            permanent = _permanent(runtime, key=f"permanent-{policy}")

    result = _apply_internal_review(
        runtime,
        captured,
        action=action,
        requested_outcome=requested,
        policy_code=policy,
        idempotency_key=f"review-{policy}",
        permanent_record_id=None if permanent is None else permanent.record_id,
        permanent_record_created=created,
        superseded_record_id=superseded,
    )

    assert result.action == action
    assert result.requested_outcome == requested
    assert result.policy_code == policy
    assert result.state == {
        "defer": "temporary",
        "reject": "rejected",
        "expire": "expired",
        "promote": "promoted",
    }[action]
    assert _candidate().verify_candidate_snapshot(runtime.connection)[
        "candidate_current"
    ] == 1


def test_promoted_superseding_policy_replays_with_both_permanent_links(runtime) -> None:
    captured = _capture(runtime)
    old = _permanent(
        runtime,
        key="old-permanent",
        statement="Use the legacy durable ledger.",
        value={"engine": "legacy"},
    )
    replacement = _permanent(
        runtime,
        key=f"candidate:{captured.candidate_id}:propose",
        activate_key=f"candidate:{captured.candidate_id}:activate",
    )
    apply_action(
        runtime.connection,
        record_id=old.record_id,
        action="supersede",
        idempotency_key=f"candidate:{captured.candidate_id}:supersede",
        replacement_record_id=replacement.record_id,
    )

    result = _apply_internal_review(
        runtime,
        captured,
        action="promote",
        requested_outcome="promote",
        policy_code="promoted_superseding",
        idempotency_key="promote-superseding",
        permanent_record_id=replacement.record_id,
        permanent_record_created=True,
        superseded_record_id=old.record_id,
    )

    assert result.state == "promoted"
    assert result.permanent_record_id == replacement.record_id
    assert result.superseded_record_id == old.record_id
    _candidate().verify_candidate_snapshot(runtime.connection)


def test_candidate_replay_rejects_linked_duplicate_with_different_content(runtime) -> None:
    captured = _capture(runtime)
    unrelated = _permanent(
        runtime,
        key="unrelated-linked-record",
        topic="storage.unrelated",
        statement="Use an unrelated ledger.",
        value={"engine": "unrelated"},
    )
    with pytest.raises(VerificationError, match="candidate|promotion|link"):
        _apply_internal_review(
            runtime,
            captured,
            action="promote",
            requested_outcome="promote",
            policy_code="linked_duplicate",
            idempotency_key="forged-linked-review",
            permanent_record_id=unrelated.record_id,
            permanent_record_created=False,
        )
    verify_ledger(runtime.connection)


def test_ledger_verify_rejects_a_persisted_forged_promotion_link(runtime) -> None:
    captured = _capture(runtime)
    unrelated = _permanent(
        runtime,
        key="persisted-unrelated-link",
        topic="storage.persisted-unrelated",
        statement="Use a persisted unrelated ledger.",
        value={"engine": "persisted-unrelated"},
    )
    key = "persisted-forged-promotion"
    event_seq = _insert_review_event(
        runtime.connection,
        candidate_id=captured.candidate_id,
        action="promote",
        payload=_review_payload(
            action="promote",
            policy_code="linked_duplicate",
            permanent_record_id=unrelated.record_id,
            permanent_record_created=False,
        ),
        key=key,
    )
    runtime.connection.execute(
        "UPDATE memory_candidate_current "
        "SET state = 'promoted', last_event_seq = ?, last_event_id = ?, "
        "permanent_record_id = ?, updated_at = ? WHERE candidate_id = ?",
        (
            event_seq,
            f"cevt_test_{key}",
            unrelated.record_id,
            "2026-09-01T00:00:00.000000Z",
            captured.candidate_id,
        ),
    )

    with pytest.raises(VerificationError, match="candidate|promotion|link"):
        verify_ledger(runtime.connection)


def test_candidate_replay_rejects_promoted_new_without_deterministic_creation(
    runtime,
) -> None:
    captured = _capture(runtime)
    manually_created = _permanent(runtime, key="manual-promoted-new")
    with pytest.raises(VerificationError, match="candidate|promotion|link"):
        _apply_internal_review(
            runtime,
            captured,
            action="promote",
            requested_outcome="promote",
            policy_code="promoted_new",
            idempotency_key="forged-created-review",
            permanent_record_id=manually_created.record_id,
            permanent_record_created=True,
        )
    verify_ledger(runtime.connection)


def test_candidate_replay_requires_permanent_record_active_at_promotion(runtime) -> None:
    captured = _capture(runtime)
    permanent = _permanent(runtime, key="closed-before-promotion")
    apply_action(
        runtime.connection,
        record_id=permanent.record_id,
        action="close",
        idempotency_key="closed-before-promotion-close",
    )

    with pytest.raises(VerificationError, match="active link"):
        _apply_internal_review(
            runtime,
            captured,
            action="promote",
            requested_outcome="promote",
            policy_code="linked_duplicate",
            idempotency_key="closed-before-promotion-review",
            permanent_record_id=permanent.record_id,
            permanent_record_created=False,
        )


def test_later_permanent_lifecycle_preserves_a_valid_prior_promotion(runtime) -> None:
    captured = _capture(runtime)
    permanent = _permanent(runtime, key="closed-after-promotion")
    _apply_internal_review(
        runtime,
        captured,
        action="promote",
        requested_outcome="promote",
        policy_code="linked_duplicate",
        idempotency_key="closed-after-promotion-review",
        permanent_record_id=permanent.record_id,
        permanent_record_created=False,
    )
    apply_action(
        runtime.connection,
        record_id=permanent.record_id,
        action="close",
        idempotency_key="closed-after-promotion-close",
    )

    activation_time = runtime.connection.execute(
        "SELECT created_at FROM memory_events "
        "WHERE record_id = ? AND action = 'activate'",
        (permanent.record_id,),
    ).fetchone()[0]
    promotion_time = runtime.connection.execute(
        "SELECT created_at FROM memory_candidate_events "
        "WHERE candidate_id = ? AND action = 'promote'",
        (captured.candidate_id,),
    ).fetchone()[0]
    close_time = runtime.connection.execute(
        "SELECT created_at FROM memory_events "
        "WHERE record_id = ? AND action = 'close'",
        (permanent.record_id,),
    ).fetchone()[0]
    assert activation_time < promotion_time < close_time

    verify_ledger(runtime.connection)


def test_candidate_replay_rejects_promoted_new_without_copied_capture_evidence(
    runtime,
) -> None:
    captured = _capture(runtime)
    permanent = _permanent(
        runtime,
        key=f"candidate:{captured.candidate_id}:propose",
        activate_key=f"candidate:{captured.candidate_id}:activate",
        evidence=("https://example.test/not-the-capture-evidence",),
    )
    with pytest.raises(VerificationError, match="candidate|promotion|link"):
        _apply_internal_review(
            runtime,
            captured,
            action="promote",
            requested_outcome="promote",
            policy_code="promoted_new",
            idempotency_key="forged-evidence-review",
            permanent_record_id=permanent.record_id,
            permanent_record_created=True,
        )
    verify_ledger(runtime.connection)


def test_candidate_replay_rejects_supersession_not_created_by_candidate(runtime) -> None:
    captured = _capture(runtime)
    old = _permanent(
        runtime,
        key="forged-old-permanent",
        statement="Use the old durable ledger.",
        value={"engine": "old"},
    )
    replacement = _permanent(
        runtime,
        key=f"candidate:{captured.candidate_id}:propose",
        activate_key=f"candidate:{captured.candidate_id}:activate",
    )
    apply_action(
        runtime.connection,
        record_id=old.record_id,
        action="supersede",
        idempotency_key="unrelated-supersede-action",
        replacement_record_id=replacement.record_id,
    )
    with pytest.raises(VerificationError, match="candidate|promotion|link"):
        _apply_internal_review(
            runtime,
            captured,
            action="promote",
            requested_outcome="promote",
            policy_code="promoted_superseding",
            idempotency_key="forged-superseding-review",
            permanent_record_id=replacement.record_id,
            permanent_record_created=True,
            superseded_record_id=old.record_id,
        )
    verify_ledger(runtime.connection)


def test_rebuild_restores_every_candidate_projection_column_from_one_replay(
    runtime,
) -> None:
    first = _capture(runtime, idempotency_key="capture-first")
    _candidate().apply_candidate_action(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=first.candidate_id,
        action="defer",
        reason="Keep pending.",
        idempotency_key="defer-first",
    )
    second = _capture(
        runtime,
        idempotency_key="capture-second",
        topic_key="storage.second",
    )
    permanent = _permanent(runtime, topic="storage.second")
    _apply_internal_review(
        runtime,
        second,
        action="promote",
        requested_outcome="promote",
        policy_code="linked_duplicate",
        idempotency_key="promote-second",
        permanent_record_id=permanent.record_id,
        permanent_record_created=False,
    )
    expected = [
        dict(row)
        for row in runtime.connection.execute(
            "SELECT * FROM memory_candidate_current ORDER BY candidate_id"
        )
    ]
    immutable_before = {
        table: [dict(row) for row in runtime.connection.execute(f"SELECT * FROM {table}")]
        for table in (
            "memory_candidates",
            "memory_candidate_events",
            "memory_candidate_evidence",
        )
    }
    runtime.connection.execute("DELETE FROM memory_candidate_current")

    rebuilt = _candidate().rebuild_candidate_projection(runtime.connection)

    assert rebuilt == 2
    assert [
        dict(row)
        for row in runtime.connection.execute(
            "SELECT * FROM memory_candidate_current ORDER BY candidate_id"
        )
    ] == expected
    assert {
        table: [dict(row) for row in runtime.connection.execute(f"SELECT * FROM {table}")]
        for table in immutable_before
    } == immutable_before


@pytest.mark.parametrize(
    "damage",
    (
        "noncanonical_body",
        "invalid_shared_body",
        "content_hash",
        "evidence_ordinal",
        "evidence_hash",
        "capture_payload",
        "event_hash",
        "terminal_transition",
        "permanent_link",
    ),
)
def test_rebuild_rolls_back_projection_on_any_immutable_corruption(runtime, damage) -> None:
    captured = _capture(
        runtime,
        evidence_uris=("https://example.test/one", "https://example.test/two"),
    )
    before = [
        dict(row)
        for row in runtime.connection.execute(
            "SELECT * FROM memory_candidate_current ORDER BY candidate_id"
        )
    ]
    if damage in {"noncanonical_body", "invalid_shared_body", "content_hash"}:
        runtime.connection.execute("DROP TRIGGER memory_candidates_reject_update")
        if damage == "noncanonical_body":
            runtime.connection.execute(
                "UPDATE memory_candidates SET value_json = ? WHERE candidate_id = ?",
                ('{ "engine": "sqlite3" }', captured.candidate_id),
            )
        elif damage == "invalid_shared_body":
            runtime.connection.execute(
                "UPDATE memory_candidates SET statement = ' ' WHERE candidate_id = ?",
                (captured.candidate_id,),
            )
        else:
            runtime.connection.execute(
                "UPDATE memory_candidates SET content_sha256 = ? WHERE candidate_id = ?",
                ("0" * 64, captured.candidate_id),
            )
    elif damage in {"evidence_ordinal", "evidence_hash"}:
        runtime.connection.execute(
            "DROP TRIGGER memory_candidate_evidence_reject_update"
        )
        if damage == "evidence_ordinal":
            runtime.connection.execute(
                "UPDATE memory_candidate_evidence SET ordinal = 7 "
                "WHERE candidate_id = ? AND ordinal = 1",
                (captured.candidate_id,),
            )
        else:
            runtime.connection.execute(
                "UPDATE memory_candidate_evidence SET evidence_sha256 = ? "
                "WHERE candidate_id = ? AND ordinal = 1",
                ("0" * 64, captured.candidate_id),
            )
    elif damage in {"capture_payload", "event_hash"}:
        runtime.connection.execute("DROP TRIGGER memory_candidate_events_reject_update")
        if damage == "capture_payload":
            runtime.connection.execute(
                "UPDATE memory_candidate_events SET payload_json = '{ }' "
                "WHERE candidate_id = ? AND action = 'capture'",
                (captured.candidate_id,),
            )
        else:
            runtime.connection.execute(
                "UPDATE memory_candidate_events SET event_sha256 = ? "
                "WHERE candidate_id = ? AND action = 'capture'",
                ("0" * 64, captured.candidate_id),
            )
    elif damage == "terminal_transition":
        _insert_review_event(
            runtime.connection,
            candidate_id=captured.candidate_id,
            action="reject",
            payload=_review_payload(action="reject"),
            key="reject-terminal",
        )
        _insert_review_event(
            runtime.connection,
            candidate_id=captured.candidate_id,
            action="expire",
            payload=_review_payload(action="expire"),
            key="expire-after-terminal",
        )
    else:
        _insert_review_event(
            runtime.connection,
            candidate_id=captured.candidate_id,
            action="promote",
            payload=_review_payload(
                action="promote",
                policy_code="linked_duplicate",
                permanent_record_id="mem_missing",
                permanent_record_created=False,
            ),
            key="missing-permanent-link",
        )

    with pytest.raises(VerificationError):
        _candidate().rebuild_candidate_projection(runtime.connection)

    assert runtime.connection.in_transaction is False
    assert [
        dict(row)
        for row in runtime.connection.execute(
            "SELECT * FROM memory_candidate_current ORDER BY candidate_id"
        )
    ] == before


@pytest.mark.parametrize(
    "damage",
    (
        "extra_key",
        "missing_key",
        "evidence_not_list",
        "evidence_item_not_text",
        "evidence_over_limit",
        "created_not_bool",
        "blank_reason",
        "wrong_policy_version",
        "reject_requested_defer",
        "reject_wrong_policy",
        "nonpromote_permanent_fields",
        "blocked_promote_wrong_policy",
        "promote_missing_record",
        "linked_duplicate_created",
        "promoted_new_not_created",
        "promoted_superseding_without_target",
    ),
)
def test_replay_rejects_non_exact_review_payload_keys_types_and_policy(runtime, damage) -> None:
    captured = _capture(runtime)
    action = "defer" if damage.startswith("blocked_") else "reject"
    payload = _review_payload(action=action)
    if damage == "extra_key":
        payload["extra"] = None
    elif damage == "missing_key":
        del payload["reason"]
    elif damage == "evidence_not_list":
        payload["evidence_uris"] = "https://example.test/evidence"
    elif damage == "evidence_item_not_text":
        payload["evidence_uris"] = [7]
    elif damage == "evidence_over_limit":
        payload["evidence_uris"] = [
            f"https://example.test/{index}" for index in range(17)
        ]
    elif damage == "created_not_bool":
        payload["permanent_record_created"] = 0
    elif damage == "blank_reason":
        payload["reason"] = " "
    elif damage == "wrong_policy_version":
        payload["policy_version"] = "phase1b-v2"
    elif damage == "reject_requested_defer":
        payload["requested_outcome"] = "defer"
    elif damage == "reject_wrong_policy":
        payload["policy_code"] = "review_expired"
    elif damage == "nonpromote_permanent_fields":
        payload["permanent_record_id"] = "mem_unexpected"
    elif damage == "blocked_promote_wrong_policy":
        payload["requested_outcome"] = "promote"
        payload["policy_code"] = "review_deferred"
    else:
        action = "promote"
        permanent = _permanent(runtime, key=f"permanent-{damage}")
        payload = _review_payload(
            action="promote",
            permanent_record_id=permanent.record_id,
            permanent_record_created=False,
        )
        if damage == "promote_missing_record":
            payload["permanent_record_id"] = "mem_missing"
        elif damage == "linked_duplicate_created":
            payload["permanent_record_created"] = True
        elif damage == "promoted_new_not_created":
            payload["policy_code"] = "promoted_new"
            payload["permanent_record_created"] = False
        else:
            payload["policy_code"] = "promoted_superseding"
            payload["permanent_record_created"] = True
            payload["superseded_record_id"] = None
    _insert_review_event(
        runtime.connection,
        candidate_id=captured.candidate_id,
        action=action,
        payload=payload,
        key=f"invalid-{damage}",
    )

    with pytest.raises(VerificationError):
        _candidate().verify_candidate_snapshot(runtime.connection)


@pytest.mark.parametrize(
    "damage",
    (
        "missing_row",
        "state",
        "last_event_seq",
        "last_event_id",
        "permanent_record_id",
        "updated_at",
    ),
)
def test_verify_compares_every_candidate_projection_column_byte_for_byte(
    runtime, damage
) -> None:
    captured = _capture(runtime)
    permanent = _permanent(runtime)
    other = _permanent(runtime, key="permanent-other", topic="storage.other")
    _apply_internal_review(
        runtime,
        captured,
        action="promote",
        requested_outcome="promote",
        policy_code="linked_duplicate",
        idempotency_key="promote-candidate",
        permanent_record_id=permanent.record_id,
        permanent_record_created=False,
    )
    if damage == "missing_row":
        runtime.connection.execute(
            "DELETE FROM memory_candidate_current WHERE candidate_id = ?",
            (captured.candidate_id,),
        )
    elif damage == "state":
        runtime.connection.execute(
            "UPDATE memory_candidate_current SET state = 'rejected' WHERE candidate_id = ?",
            (captured.candidate_id,),
        )
    elif damage == "last_event_seq":
        runtime.connection.execute(
            "UPDATE memory_candidate_current SET last_event_seq = last_event_seq + 10 "
            "WHERE candidate_id = ?",
            (captured.candidate_id,),
        )
    elif damage == "last_event_id":
        capture_event = runtime.connection.execute(
            "SELECT event_id FROM memory_candidate_events "
            "WHERE candidate_id = ? AND action = 'capture'",
            (captured.candidate_id,),
        ).fetchone()[0]
        runtime.connection.execute(
            "UPDATE memory_candidate_current SET last_event_id = ? WHERE candidate_id = ?",
            (capture_event, captured.candidate_id),
        )
    elif damage == "permanent_record_id":
        runtime.connection.execute(
            "UPDATE memory_candidate_current SET permanent_record_id = ? "
            "WHERE candidate_id = ?",
            (other.record_id, captured.candidate_id),
        )
    else:
        runtime.connection.execute(
            "UPDATE memory_candidate_current SET updated_at = 'tampered' "
            "WHERE candidate_id = ?",
            (captured.candidate_id,),
        )

    with pytest.raises(VerificationError):
        _candidate().verify_candidate_snapshot(runtime.connection)
