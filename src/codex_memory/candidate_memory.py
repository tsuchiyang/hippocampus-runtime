from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit
import uuid

from codex_memory.blob_store import BlobStore
from codex_memory.database import next_memory_timestamp
from codex_memory.ledger_common import (
    IdempotencyConflict,
    LedgerError,
    RecordNotFound,
    ValidatedRecordContent,
    ValidationError,
    VerificationError,
    _hash_preimage,
    _require_bounded_text,
    _validate_idempotency_key,
    canonical_json,
    parse_json_value,
    scope_keys_equal,
    validate_evidence_uris,
    validate_new_evidence_uris,
    validate_record_content,
)
from codex_memory.memory_evidence import parse_event_uri, resolve_event_evidence
from codex_memory.memory_security import reject_secrets


_REVIEW_PAYLOAD_KEYS = frozenset(
    {
        "evidence_uris",
        "permanent_record_created",
        "permanent_record_id",
        "policy_code",
        "policy_version",
        "reason",
        "requested_outcome",
        "superseded_record_id",
    }
)
_BLOCKED_PROMOTION_POLICIES = frozenset(
    {"decision_conflict", "constitution_conflict", "newer_evidence_required"}
)
_ACTION_STATES = {
    "defer": "temporary",
    "reject": "rejected",
    "expire": "expired",
    "promote": "promoted",
}


@dataclass(frozen=True, slots=True)
class CandidateView:
    candidate_id: str
    record_type: str
    scope_kind: str
    scope_key: str
    topic_key: str
    statement: str
    value: Any
    session_id: str
    turn_id: str
    context_id: str
    evidence_uris: tuple[str, ...]
    state: str
    permanent_record_id: str | None
    content_sha256: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CandidateActionResult:
    candidate_id: str
    event_id: str
    action: str
    requested_outcome: str
    policy_code: str
    state: str
    permanent_record_id: str | None
    permanent_record_created: bool | None
    superseded_record_id: str | None
    created_at: str


def capture_candidate(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    project_scope_key: str,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    session_id: str,
    turn_id: str,
    context_id: str,
    idempotency_key: str,
    value: Any = None,
    evidence_uris: Sequence[str] = (),
) -> CandidateView:
    """Capture one immutable temporary candidate under one immediate transaction."""
    content = validate_record_content(
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value=value,
    )
    _require_bounded_text(project_scope_key, "project scope key", 256, nonblank=True)
    if scope_kind == "project":
        if not scope_keys_equal(project_scope_key, scope_key):
            raise ValidationError("project scope key mismatch")
        stored_scope_key = project_scope_key
    else:
        stored_scope_key = scope_key
    if stored_scope_key != scope_key:
        content = validate_record_content(
            record_type=record_type,
            scope_kind=scope_kind,
            scope_key=stored_scope_key,
            topic_key=topic_key,
            statement=statement,
            value=value,
        )
    _require_bounded_text(session_id, "session ID", 256, nonblank=True)
    _require_bounded_text(turn_id, "turn ID", 256, nonblank=True)
    _require_bounded_text(context_id, "context ID", 256, nonblank=True)
    _validate_idempotency_key(idempotency_key)
    evidence = validate_new_evidence_uris(evidence_uris, maximum=16)
    reject_secrets(statement=statement, value=value)
    for uri in evidence:
        if urlsplit(uri).scheme.lower() == "codex-event":
            event = resolve_event_evidence(connection, blobs, uri)
            if event.session_id != session_id:
                raise ValidationError(
                    "local candidate evidence belongs to another session"
                )

    connection.execute("BEGIN IMMEDIATE")
    try:
        prior_candidate = connection.execute(
            "SELECT * FROM memory_candidates WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        prior_event = connection.execute(
            "SELECT * FROM memory_candidate_events WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if prior_candidate is not None:
            if prior_event is None or not _capture_matches(
                connection,
                prior_candidate,
                prior_event,
                content=content,
                record_type=record_type,
                scope_kind=scope_kind,
                scope_key=stored_scope_key,
                topic_key=topic_key,
                statement=statement,
                session_id=session_id,
                turn_id=turn_id,
                context_id=context_id,
                idempotency_key=idempotency_key,
                evidence_uris=evidence,
            ):
                raise IdempotencyConflict("candidate idempotency key conflict")
            result = _candidate_view(connection, prior_candidate)
            connection.commit()
            return result
        if prior_event is not None:
            raise IdempotencyConflict("candidate idempotency key conflict")

        created_at = next_memory_timestamp(connection)
        candidate_id = _opaque_id("cand")
        event_id = _opaque_id("cevt")
        connection.execute(
            """
            INSERT INTO memory_candidates (
                candidate_id, record_type, scope_kind, scope_key, topic_key,
                statement, value_json, session_id, turn_id, context_id,
                idempotency_key, content_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                record_type,
                scope_kind,
                stored_scope_key,
                topic_key,
                statement,
                content.value_json,
                session_id,
                turn_id,
                context_id,
                idempotency_key,
                content.content_sha256,
                created_at,
            ),
        )
        for ordinal, uri in enumerate(evidence):
            connection.execute(
                """
                INSERT INTO memory_candidate_evidence (
                    evidence_id, candidate_id, ordinal, uri, evidence_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _opaque_id("cevi"),
                    candidate_id,
                    ordinal,
                    uri,
                    _hash_preimage({"candidate_id": candidate_id, "uri": uri}),
                    created_at,
                ),
            )
        event_cursor = connection.execute(
            """
            INSERT INTO memory_candidate_events (
                event_id, candidate_id, action, payload_json, idempotency_key,
                event_sha256, created_at
            ) VALUES (?, ?, 'capture', '{}', ?, ?, ?)
            """,
            (
                event_id,
                candidate_id,
                idempotency_key,
                _hash_preimage(
                    {"action": "capture", "candidate_id": candidate_id, "payload": {}}
                ),
                created_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_candidate_current (
                candidate_id, state, last_event_seq, last_event_id, permanent_record_id,
                updated_at
            ) VALUES (?, 'temporary', ?, ?, NULL, ?)
            """,
            (candidate_id, event_cursor.lastrowid, event_id, created_at),
        )
        connection.commit()
        return CandidateView(
            candidate_id=candidate_id,
            record_type=record_type,
            scope_kind=scope_kind,
            scope_key=stored_scope_key,
            topic_key=topic_key,
            statement=statement,
            value=parse_json_value(content.value_json),
            session_id=session_id,
            turn_id=turn_id,
            context_id=context_id,
            evidence_uris=evidence,
            state="temporary",
            permanent_record_id=None,
            content_sha256=content.content_sha256,
            created_at=created_at,
            updated_at=created_at,
        )
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _capture_matches(
    connection: sqlite3.Connection,
    candidate: sqlite3.Row,
    event: sqlite3.Row,
    *,
    content: ValidatedRecordContent,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    session_id: str,
    turn_id: str,
    context_id: str,
    idempotency_key: str,
    evidence_uris: tuple[str, ...],
) -> bool:
    stored_evidence = tuple(
        (row["ordinal"], row["uri"], row["evidence_sha256"])
        for row in connection.execute(
            """
            SELECT ordinal, uri, evidence_sha256
            FROM memory_candidate_evidence
            WHERE candidate_id = ?
            ORDER BY ordinal
            """,
            (candidate["candidate_id"],),
        )
    )
    expected_evidence = tuple(
        (
            ordinal,
            uri,
            _hash_preimage({"candidate_id": candidate["candidate_id"], "uri": uri}),
        )
        for ordinal, uri in enumerate(evidence_uris)
    )
    return (
        candidate["record_type"],
        candidate["scope_kind"],
        candidate["scope_key"],
        candidate["topic_key"],
        candidate["statement"],
        candidate["value_json"],
        candidate["session_id"],
        candidate["turn_id"],
        candidate["context_id"],
        candidate["idempotency_key"],
        candidate["content_sha256"],
        stored_evidence,
        event["candidate_id"],
        event["action"],
        event["payload_json"],
        event["idempotency_key"],
        event["event_sha256"],
    ) == (
        record_type,
        scope_kind,
        scope_key,
        topic_key,
        statement,
        content.value_json,
        session_id,
        turn_id,
        context_id,
        idempotency_key,
        content.content_sha256,
        expected_evidence,
        candidate["candidate_id"],
        "capture",
        "{}",
        idempotency_key,
        _hash_preimage(
            {"action": "capture", "candidate_id": candidate["candidate_id"], "payload": {}}
        ),
    )


def _candidate_view(connection: sqlite3.Connection, candidate: sqlite3.Row) -> CandidateView:
    current = connection.execute(
        "SELECT state, permanent_record_id, updated_at FROM memory_candidate_current "
        "WHERE candidate_id = ?",
        (candidate["candidate_id"],),
    ).fetchone()
    if current is None:
        raise IdempotencyConflict("candidate idempotency key conflict")
    evidence = tuple(
        row[0]
        for row in connection.execute(
            "SELECT uri FROM memory_candidate_evidence WHERE candidate_id = ? ORDER BY ordinal",
            (candidate["candidate_id"],),
        )
    )
    return CandidateView(
        candidate_id=candidate["candidate_id"],
        record_type=candidate["record_type"],
        scope_kind=candidate["scope_kind"],
        scope_key=candidate["scope_key"],
        topic_key=candidate["topic_key"],
        statement=candidate["statement"],
        value=parse_json_value(candidate["value_json"]),
        session_id=candidate["session_id"],
        turn_id=candidate["turn_id"],
        context_id=candidate["context_id"],
        evidence_uris=evidence,
        state=current["state"],
        permanent_record_id=current["permanent_record_id"],
        content_sha256=candidate["content_sha256"],
        created_at=candidate["created_at"],
        updated_at=current["updated_at"],
    )


def _opaque_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def apply_candidate_action(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    candidate_id: str,
    action: Literal["defer", "reject", "expire"],
    reason: str,
    idempotency_key: str,
    evidence_uris: Sequence[str] = (),
) -> CandidateActionResult:
    """Append one non-promotion review under an immediate transaction."""
    _require_bounded_text(candidate_id, "candidate ID", 256, nonblank=True)
    if not isinstance(action, str) or action not in {"defer", "reject", "expire"}:
        raise ValidationError("invalid candidate action")
    _require_bounded_text(reason, "reason", 4096, nonblank=True)
    _validate_idempotency_key(idempotency_key)
    evidence = validate_new_evidence_uris(evidence_uris, maximum=16)
    connection.execute("BEGIN IMMEDIATE")
    try:
        prior = connection.execute(
            "SELECT 1 FROM memory_candidate_events WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if prior is None:
            candidate = connection.execute(
                "SELECT session_id FROM memory_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            for uri in evidence:
                if urlsplit(uri).scheme.lower() == "codex-event":
                    event = resolve_event_evidence(connection, blobs, uri)
                    if (
                        candidate is not None
                        and event.session_id != candidate["session_id"]
                    ):
                        raise ValidationError(
                            "local candidate evidence belongs to another session"
                        )
        result = apply_candidate_action_in_transaction(
            connection,
            candidate_id=candidate_id,
            action=action,
            requested_outcome=action,
            reason=reason,
            idempotency_key=idempotency_key,
            evidence_uris=evidence,
            policy_code={
                "defer": "review_deferred",
                "reject": "review_rejected",
                "expire": "review_expired",
            }[action],
            permanent_record_id=None,
            permanent_record_created=None,
            superseded_record_id=None,
        )
        connection.commit()
        return result
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def apply_candidate_action_in_transaction(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    action: str,
    reason: str,
    idempotency_key: str,
    evidence_uris: Sequence[str] = (),
    requested_outcome: str | None = None,
    policy_code: str | None = None,
    permanent_record_id: str | None = None,
    permanent_record_created: bool | None = None,
    superseded_record_id: str | None = None,
) -> CandidateActionResult:
    """Append one already-authorized candidate review inside the caller's transaction."""
    if not connection.in_transaction:
        raise LedgerError("candidate action requires an active transaction")
    _require_bounded_text(candidate_id, "candidate ID", 256, nonblank=True)
    _validate_idempotency_key(idempotency_key)
    if not isinstance(action, str) or action not in _ACTION_STATES:
        raise ValidationError("invalid candidate action")
    if requested_outcome is None:
        requested_outcome = action
    if policy_code is None:
        policy_code = {
            "defer": "review_deferred",
            "reject": "review_rejected",
            "expire": "review_expired",
        }.get(action)
    evidence = validate_new_evidence_uris(evidence_uris, maximum=16)
    payload = {
        "evidence_uris": list(evidence),
        "permanent_record_created": permanent_record_created,
        "permanent_record_id": permanent_record_id,
        "policy_code": policy_code,
        "policy_version": "phase1b-v1",
        "reason": reason,
        "requested_outcome": requested_outcome,
        "superseded_record_id": superseded_record_id,
    }
    _validate_review_payload(action, payload)
    payload_json = canonical_json(payload)

    projection = _validated_candidate_replay(connection)
    _require_candidate_projection(connection, projection)
    prior = connection.execute(
        "SELECT * FROM memory_candidate_events WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    if prior is not None:
        if (
            prior["candidate_id"],
            prior["action"],
            prior["payload_json"],
        ) != (candidate_id, action, payload_json):
            raise IdempotencyConflict("candidate idempotency key conflict")
        return _action_result_from_event(prior)

    current = projection.get(candidate_id)
    if current is None:
        exists = connection.execute(
            "SELECT 1 FROM memory_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if exists is None:
            raise RecordNotFound("candidate not found")
        raise VerificationError("candidate projection is missing")
    if current[0] != "temporary":
        raise IdempotencyConflict("candidate terminal state conflict")
    transition = _ACTION_STATES[action]
    created_at = next_memory_timestamp(connection)
    event_id = _opaque_id("cevt")
    cursor = connection.execute(
        """
        INSERT INTO memory_candidate_events (
            event_id, candidate_id, action, payload_json, idempotency_key,
            event_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            candidate_id,
            action,
            payload_json,
            idempotency_key,
            _hash_preimage(
                {"action": action, "candidate_id": candidate_id, "payload": payload}
            ),
            created_at,
        ),
    )
    updated = connection.execute(
        """
        UPDATE memory_candidate_current
        SET state = ?, last_event_seq = ?, last_event_id = ?,
            permanent_record_id = ?, updated_at = ?
        WHERE candidate_id = ?
        """,
        (
            transition,
            cursor.lastrowid,
            event_id,
            permanent_record_id if action == "promote" else None,
            created_at,
            candidate_id,
        ),
    )
    if updated.rowcount != 1:
        raise VerificationError("candidate projection update failed")

    replayed = _validated_candidate_replay(connection)
    _require_candidate_projection(connection, replayed)
    event = connection.execute(
        "SELECT * FROM memory_candidate_events WHERE event_id = ?", (event_id,)
    ).fetchone()
    return _action_result_from_event(event)


def _action_result_from_event(event: sqlite3.Row) -> CandidateActionResult:
    payload = _load_candidate_canonical(event["payload_json"], "candidate event payload")
    action = event["action"]
    try:
        _validate_review_payload(action, payload)
    except ValidationError as exc:
        raise VerificationError("invalid candidate review payload") from exc
    return CandidateActionResult(
        candidate_id=event["candidate_id"],
        event_id=event["event_id"],
        action=action,
        requested_outcome=payload["requested_outcome"],
        policy_code=payload["policy_code"],
        state=_ACTION_STATES[action],
        permanent_record_id=payload["permanent_record_id"],
        permanent_record_created=payload["permanent_record_created"],
        superseded_record_id=payload["superseded_record_id"],
        created_at=event["created_at"],
    )


def list_pending_candidates(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    limit: int = 100,
) -> tuple[CandidateView, ...]:
    _require_bounded_text(session_id, "session ID", 256, nonblank=True)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValidationError("pending limit must be between 1 and 100")
    rows = connection.execute(
        """
        SELECT c.*
        FROM memory_candidates AS c
        JOIN memory_candidate_current AS p ON p.candidate_id = c.candidate_id
        WHERE c.session_id = ? AND p.state = 'temporary'
        ORDER BY c.candidate_seq ASC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return tuple(_candidate_view(connection, row) for row in rows)


def rebuild_candidate_projection(connection: sqlite3.Connection) -> int:
    """Atomically rebuild the disposable candidate projection from strict replay."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        projection = _validated_candidate_replay(connection)
        connection.execute("DELETE FROM memory_candidate_current")
        for candidate_id in sorted(projection):
            state, event_seq, event_id, permanent_id, updated_at = projection[candidate_id]
            connection.execute(
                """
                INSERT INTO memory_candidate_current (
                    candidate_id, state, last_event_seq, last_event_id,
                    permanent_record_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (candidate_id, state, event_seq, event_id, permanent_id, updated_at),
            )
        connection.commit()
        return len(projection)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def verify_candidate_snapshot(connection: sqlite3.Connection) -> dict[str, int]:
    """Verify candidate immutables and every current-projection column."""
    projection = _validated_candidate_replay(connection)
    actual = _require_candidate_projection(connection, projection)
    return {
        "candidates": connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0],
        "candidate_events": connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_events"
        ).fetchone()[0],
        "candidate_evidence": connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_evidence"
        ).fetchone()[0],
        "candidate_current": len(actual),
    }


def _validated_candidate_replay(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, int, str, str | None, str]]:
    candidates: dict[str, sqlite3.Row] = {}
    for candidate in connection.execute(
        "SELECT * FROM memory_candidates ORDER BY candidate_seq"
    ):
        candidate_id = candidate["candidate_id"]
        try:
            _require_bounded_text(candidate_id, "candidate ID", 256, nonblank=True)
            _require_bounded_text(candidate["session_id"], "session ID", 256, nonblank=True)
            _require_bounded_text(candidate["turn_id"], "turn ID", 256, nonblank=True)
            _require_bounded_text(candidate["context_id"], "context ID", 256, nonblank=True)
            _require_bounded_text(candidate["created_at"], "candidate timestamp", 128, nonblank=True)
            _validate_idempotency_key(candidate["idempotency_key"])
            value = _load_candidate_canonical(candidate["value_json"], "candidate value")
            content = validate_record_content(
                record_type=candidate["record_type"],
                scope_kind=candidate["scope_kind"],
                scope_key=candidate["scope_key"],
                topic_key=candidate["topic_key"],
                statement=candidate["statement"],
                value=value,
            )
        except ValidationError as exc:
            raise VerificationError("invalid candidate body") from exc
        if (
            content.value_json != candidate["value_json"]
            or content.content_sha256 != candidate["content_sha256"]
            or candidate_id in candidates
        ):
            raise VerificationError("candidate content mismatch")
        candidates[candidate_id] = candidate

    next_ordinal: dict[str, int] = {}
    evidence_counts: dict[str, int] = {}
    for evidence in connection.execute(
        "SELECT * FROM memory_candidate_evidence ORDER BY candidate_id, ordinal"
    ):
        candidate_id = evidence["candidate_id"]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise VerificationError("candidate evidence references a missing candidate")
        ordinal = next_ordinal.get(candidate_id, 0)
        if evidence["ordinal"] != ordinal:
            raise VerificationError("candidate evidence order is not contiguous")
        count = evidence_counts.get(candidate_id, 0) + 1
        if count > 16:
            raise VerificationError("candidate evidence limit exceeded")
        try:
            validate_evidence_uris((evidence["uri"],), maximum=16)
            _verify_candidate_local_evidence(
                connection, candidate, evidence["uri"]
            )
            _require_bounded_text(
                evidence["evidence_id"], "candidate evidence ID", 256, nonblank=True
            )
            _require_bounded_text(
                evidence["created_at"],
                "candidate evidence timestamp",
                128,
                nonblank=True,
            )
        except ValidationError as exc:
            raise VerificationError("invalid candidate evidence") from exc
        if evidence["created_at"] != candidate["created_at"]:
            raise VerificationError("candidate evidence timestamp mismatch")
        if evidence["evidence_sha256"] != _hash_preimage(
            {"candidate_id": candidate_id, "uri": evidence["uri"]}
        ):
            raise VerificationError("candidate evidence hash mismatch")
        next_ordinal[candidate_id] = ordinal + 1
        evidence_counts[candidate_id] = count

    projection: dict[str, tuple[str, int, str, str | None, str]] = {}
    for event in connection.execute(
        "SELECT * FROM memory_candidate_events ORDER BY event_seq"
    ):
        candidate_id = event["candidate_id"]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise VerificationError("candidate event references a missing candidate")
        try:
            _require_bounded_text(event["event_id"], "candidate event ID", 256, nonblank=True)
            _require_bounded_text(
                event["created_at"], "candidate event timestamp", 128, nonblank=True
            )
            _validate_idempotency_key(event["idempotency_key"])
        except ValidationError as exc:
            raise VerificationError("invalid candidate event metadata") from exc
        payload = _load_candidate_canonical(
            event["payload_json"], "candidate event payload"
        )
        if event["event_sha256"] != _hash_preimage(
            {"action": event["action"], "candidate_id": candidate_id, "payload": payload}
        ):
            raise VerificationError("candidate event hash mismatch")

        prior = projection.get(candidate_id)
        if event["action"] == "capture":
            if (
                prior is not None
                or payload != {}
                or event["idempotency_key"] != candidate["idempotency_key"]
                or event["created_at"] != candidate["created_at"]
            ):
                raise VerificationError("invalid candidate capture event")
            state = "temporary"
            permanent_id = None
        else:
            if prior is None:
                raise VerificationError("candidate lifecycle event precedes capture")
            if prior[0] != "temporary":
                raise VerificationError("invalid candidate terminal transition")
            try:
                _validate_review_payload(event["action"], payload)
                for uri in payload["evidence_uris"]:
                    _verify_candidate_local_evidence(connection, candidate, uri)
            except ValidationError as exc:
                raise VerificationError("invalid candidate review payload") from exc
            if event["action"] == "promote":
                _verify_promotion_relationship(
                    connection, candidate, payload, event["created_at"]
                )
            state = _ACTION_STATES[event["action"]]
            permanent_id = payload["permanent_record_id"]
            if permanent_id is not None:
                permanent = connection.execute(
                    "SELECT 1 FROM memory_records WHERE record_id = ?", (permanent_id,)
                ).fetchone()
                if permanent is None:
                    raise VerificationError("candidate permanent link is missing")
            superseded_id = payload["superseded_record_id"]
            if superseded_id is not None:
                superseded = connection.execute(
                    "SELECT 1 FROM memory_records WHERE record_id = ?", (superseded_id,)
                ).fetchone()
                if superseded is None:
                    raise VerificationError("candidate supersession link is missing")
        projection[candidate_id] = (
            state,
            event["event_seq"],
            event["event_id"],
            permanent_id,
            event["created_at"],
        )

    if set(projection) != set(candidates):
        raise VerificationError("candidate is missing its capture event")
    return projection


def _require_candidate_projection(
    connection: sqlite3.Connection,
    expected: dict[str, tuple[str, int, str, str | None, str]],
) -> dict[str, tuple[str, int, str, str | None, str]]:
    actual = {
        row["candidate_id"]: (
            row["state"],
            row["last_event_seq"],
            row["last_event_id"],
            row["permanent_record_id"],
            row["updated_at"],
        )
        for row in connection.execute("SELECT * FROM memory_candidate_current")
    }
    if actual != expected:
        raise VerificationError("candidate projection does not match event replay")
    return actual


def _validate_review_payload(action: str, payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != _REVIEW_PAYLOAD_KEYS:
        raise ValidationError("invalid candidate review payload")
    if not isinstance(action, str) or action not in _ACTION_STATES:
        raise ValidationError("invalid candidate action")
    evidence = payload["evidence_uris"]
    if not isinstance(evidence, list):
        raise ValidationError("invalid candidate review evidence")
    validate_evidence_uris(evidence, maximum=16)
    _require_bounded_text(payload["reason"], "reason", 4096, nonblank=True)
    if payload["policy_version"] != "phase1b-v1":
        raise ValidationError("invalid candidate review policy version")
    requested = payload["requested_outcome"]
    policy = payload["policy_code"]
    if not isinstance(requested, str) or not isinstance(policy, str):
        raise ValidationError("invalid candidate review policy")

    permanent_id = payload["permanent_record_id"]
    permanent_created = payload["permanent_record_created"]
    superseded_id = payload["superseded_record_id"]
    if permanent_id is not None:
        _require_bounded_text(permanent_id, "permanent record ID", 256, nonblank=True)
    if superseded_id is not None:
        _require_bounded_text(
            superseded_id, "superseded record ID", 256, nonblank=True
        )
    if permanent_created is not None and type(permanent_created) is not bool:
        raise ValidationError("invalid permanent record creation flag")

    if action == "defer":
        valid = (
            requested == "defer" and policy == "review_deferred"
        ) or (requested == "promote" and policy in _BLOCKED_PROMOTION_POLICIES)
        if not valid or any(
            value is not None
            for value in (permanent_id, permanent_created, superseded_id)
        ):
            raise ValidationError("invalid defer review policy")
        return
    if action == "reject":
        valid = requested == "reject" and policy == "review_rejected"
        if not valid or any(
            value is not None
            for value in (permanent_id, permanent_created, superseded_id)
        ):
            raise ValidationError("invalid reject review policy")
        return
    if action == "expire":
        valid = requested == "expire" and policy == "review_expired"
        if not valid or any(
            value is not None
            for value in (permanent_id, permanent_created, superseded_id)
        ):
            raise ValidationError("invalid expire review policy")
        return

    if requested != "promote" or permanent_id is None:
        raise ValidationError("invalid promote review policy")
    if policy == "linked_duplicate":
        valid = permanent_created is False and superseded_id is None
    elif policy == "promoted_new":
        valid = permanent_created is True and superseded_id is None
    elif policy == "promoted_superseding":
        valid = (
            permanent_created is True
            and superseded_id is not None
            and superseded_id != permanent_id
        )
    else:
        valid = False
    if not valid:
        raise ValidationError("invalid promote review policy")


def _verify_candidate_local_evidence(
    connection: sqlite3.Connection, candidate: sqlite3.Row, uri: str
) -> None:
    if urlsplit(uri).scheme.lower() != "codex-event":
        return
    try:
        session_id, event_id = parse_event_uri(uri)
    except ValueError as exc:
        raise ValidationError("invalid local candidate evidence") from exc
    if session_id != candidate["session_id"]:
        raise ValidationError("local candidate evidence belongs to another session")
    if connection.execute(
        "SELECT 1 FROM events WHERE session_id = ? AND event_id = ?",
        (session_id, event_id),
    ).fetchone() is None:
        raise ValidationError("local candidate evidence is missing")


def _verify_promotion_relationship(
    connection: sqlite3.Connection,
    candidate: sqlite3.Row,
    payload: dict[str, Any],
    promotion_created_at: str,
) -> None:
    permanent_id = payload["permanent_record_id"]
    permanent = connection.execute(
        "SELECT * FROM memory_records WHERE record_id = ?", (permanent_id,)
    ).fetchone()
    if permanent is None:
        raise VerificationError("candidate promotion link is missing")
    if (
        permanent["record_type"],
        permanent["scope_kind"],
        permanent["scope_key"],
        permanent["topic_key"],
        permanent["statement"],
        permanent["value_json"],
        permanent["content_sha256"],
    ) != (
        candidate["record_type"],
        candidate["scope_kind"],
        candidate["scope_key"],
        candidate["topic_key"],
        candidate["statement"],
        candidate["value_json"],
        candidate["content_sha256"],
    ):
        raise VerificationError("candidate promotion content link mismatch")
    if _permanent_state_at(
        connection, permanent_id, through_created_at=promotion_created_at
    ) != "active":
        raise VerificationError("candidate promotion active link mismatch")

    candidate_id = candidate["candidate_id"]
    created = payload["permanent_record_created"]
    if created:
        propose_key = f"candidate:{candidate_id}:propose"
        activate_key = f"candidate:{candidate_id}:activate"
        if permanent["idempotency_key"] != propose_key:
            raise VerificationError("candidate promotion creation link mismatch")
        proposal = connection.execute(
            "SELECT 1 FROM memory_events "
            "WHERE record_id = ? AND action = 'propose' AND idempotency_key = ?",
            (permanent_id, propose_key),
        ).fetchone()
        activation = connection.execute(
            "SELECT 1 FROM memory_events "
            "WHERE record_id = ? AND action = 'activate' AND idempotency_key = ?",
            (permanent_id, activate_key),
        ).fetchone()
        if proposal is None or activation is None:
            raise VerificationError("candidate promotion lifecycle link mismatch")
        candidate_evidence = tuple(
            row["uri"]
            for row in connection.execute(
                "SELECT uri FROM memory_candidate_evidence "
                "WHERE candidate_id = ? ORDER BY ordinal",
                (candidate_id,),
            )
        )
        permanent_evidence = tuple(
            row["uri"]
            for row in connection.execute(
                "SELECT uri FROM memory_evidence "
                "WHERE record_id = ? ORDER BY ordinal",
                (permanent_id,),
            )
        )
        if permanent_evidence != candidate_evidence:
            raise VerificationError("candidate promotion evidence link mismatch")
    elif connection.execute(
        "SELECT 1 FROM memory_events WHERE record_id = ? AND action = 'activate'",
        (permanent_id,),
    ).fetchone() is None:
        raise VerificationError("candidate promotion active link mismatch")

    superseded_id = payload["superseded_record_id"]
    if superseded_id is None:
        return
    target = connection.execute(
        "SELECT * FROM memory_records WHERE record_id = ?", (superseded_id,)
    ).fetchone()
    if target is None:
        raise VerificationError("candidate promotion supersession link is missing")
    if (
        target["record_type"] != candidate["record_type"]
        or target["scope_kind"] != candidate["scope_kind"]
        or not scope_keys_equal(target["scope_key"], candidate["scope_key"])
        or target["topic_key"] != candidate["topic_key"]
        or target["content_sha256"] == candidate["content_sha256"]
    ):
        raise VerificationError("candidate promotion supersession target mismatch")
    supersede_event = connection.execute(
        "SELECT event_id, payload_json FROM memory_events "
        "WHERE record_id = ? AND action = 'supersede' AND idempotency_key = ?",
        (superseded_id, f"candidate:{candidate_id}:supersede"),
    ).fetchone()
    if supersede_event is None:
        raise VerificationError("candidate promotion supersession event mismatch")
    try:
        supersede_payload = parse_json_value(supersede_event["payload_json"])
    except ValidationError as exc:
        raise VerificationError("candidate promotion supersession payload mismatch") from exc
    if supersede_payload != {"replacement_record_id": permanent_id}:
        raise VerificationError("candidate promotion supersession replacement mismatch")
    if _permanent_state_at(
        connection, superseded_id, through_created_at=promotion_created_at
    ) != "superseded":
        raise VerificationError("candidate promotion supersession state mismatch")
    relations = connection.execute(
        "SELECT relation_type, source_record_id, target_record_id "
        "FROM memory_relations WHERE event_id = ?",
        (supersede_event["event_id"],),
    ).fetchall()
    if len(relations) != 1 or tuple(relations[0]) != (
        "superseded_by",
        superseded_id,
        permanent_id,
    ):
        raise VerificationError("candidate promotion supersession relation mismatch")


def _permanent_state_at(
    connection: sqlite3.Connection,
    record_id: str,
    *,
    through_created_at: str,
) -> str | None:
    state: str | None = None
    for event in connection.execute(
        "SELECT action FROM memory_events "
        "WHERE record_id = ? AND created_at <= ? ORDER BY event_seq",
        (record_id, through_created_at),
    ):
        action = event["action"]
        if action == "propose":
            state = "proposed"
        elif action in {"activate", "reopen"}:
            state = "active"
        elif action == "supersede":
            state = "superseded"
        elif action == "invalidate":
            state = "invalidated"
        elif action == "close":
            state = "closed"
    return state


def _load_candidate_canonical(raw: str, label: str) -> Any:
    try:
        value = parse_json_value(raw)
        if canonical_json(value) != raw:
            raise VerificationError(f"{label} is not canonical")
        return value
    except VerificationError:
        raise
    except (ValidationError, UnicodeError) as exc:
        raise VerificationError(f"invalid {label}") from exc
