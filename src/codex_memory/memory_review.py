from __future__ import annotations

import sqlite3
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit

from codex_memory.blob_store import BlobIntegrityError, BlobStore
from codex_memory.candidate_memory import (
    CandidateActionResult,
    _action_result_from_event,
    apply_candidate_action_in_transaction,
)
from codex_memory.ledger_common import (
    IdempotencyConflict,
    LedgerError,
    RecordNotFound,
    ValidationError,
    VerificationError,
    _require_bounded_text,
    _validate_idempotency_key,
    parse_json_value,
    scope_keys_equal,
    validate_new_evidence_uris,
)
from codex_memory.memory_evidence import EventEvidence, resolve_event_evidence
from codex_memory.memory_security import SecretRejected, reject_secrets
from codex_memory.structured_ledger import (
    _validated_replay,
    apply_action_in_transaction,
    propose_record_in_transaction,
    verify_ledger,
)


_OUTCOMES = frozenset({"promote", "reject", "defer", "expire"})
_AUTHORITATIVE_TYPES = frozenset({"decision", "constitution"})
_NON_PROMOTION_POLICIES = {
    "defer": "review_deferred",
    "reject": "review_rejected",
    "expire": "review_expired",
}
_CONFLICT_POLICIES = {
    "decision": "decision_conflict",
    "constitution": "constitution_conflict",
}
_AMBIGUOUS_TARGET = object()


def review_candidate(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    candidate_id: str,
    outcome: Literal["promote", "reject", "defer", "expire"],
    reason: str,
    idempotency_key: str,
    evidence_uris: Sequence[str] = (),
    supersedes_record_id: str | None = None,
) -> CandidateActionResult:
    """Review one candidate and atomically apply the resulting ledger changes."""
    _require_bounded_text(candidate_id, "candidate ID", 256, nonblank=True)
    if not isinstance(outcome, str) or outcome not in _OUTCOMES:
        raise ValidationError("invalid review outcome")
    _require_bounded_text(reason, "reason", 4096, nonblank=True)
    _validate_idempotency_key(idempotency_key)
    evidence = validate_new_evidence_uris(evidence_uris, maximum=16)
    if supersedes_record_id is not None:
        _require_bounded_text(
            supersedes_record_id,
            "superseded record ID",
            256,
            nonblank=True,
        )
    if connection.in_transaction:
        raise LedgerError("candidate review requires transaction ownership")

    try:
        connection.execute("BEGIN IMMEDIATE")
        verify_ledger(connection)

        prior = connection.execute(
            "SELECT * FROM memory_candidate_events WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if prior is not None:
            if not _prior_review_matches(
                connection,
                prior,
                candidate_id=candidate_id,
                outcome=outcome,
                reason=reason,
                evidence=evidence,
                supersedes_record_id=supersedes_record_id,
            ):
                raise IdempotencyConflict("candidate review idempotency key conflict")
            result = _action_result_from_event(prior)
            connection.commit()
            return result

        candidate = connection.execute(
            """
            SELECT c.*, p.state, p.permanent_record_id
            FROM memory_candidates AS c
            JOIN memory_candidate_current AS p
              ON p.candidate_id = c.candidate_id
            WHERE c.candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            exists = connection.execute(
                "SELECT 1 FROM memory_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if exists is None:
                raise RecordNotFound("candidate not found")
            raise VerificationError("candidate projection is missing")
        if candidate["state"] != "temporary":
            raise IdempotencyConflict("candidate terminal state conflict")

        value = parse_json_value(candidate["value_json"])
        try:
            reject_secrets(statement=candidate["statement"], value=value)
        except SecretRejected as exc:
            raise ValidationError("candidate content failed the review scanner") from exc

        history_uris = _candidate_history_evidence(connection, candidate_id)
        history_events = _resolve_local_events(
            connection,
            blobs,
            history_uris,
            expected_session=candidate["session_id"],
        )
        current_events = _resolve_local_events(
            connection,
            blobs,
            evidence,
            expected_session=candidate["session_id"],
        )
        all_candidate_events = (*history_events, *current_events)

        if outcome != "promote":
            if supersedes_record_id is not None:
                raise ValidationError(
                    "supersession target is valid only for promotion"
                )
            result = apply_candidate_action_in_transaction(
                connection,
                candidate_id=candidate_id,
                action=outcome,
                requested_outcome=outcome,
                reason=reason,
                idempotency_key=idempotency_key,
                evidence_uris=evidence,
                policy_code=_NON_PROMOTION_POLICIES[outcome],
                permanent_record_id=None,
                permanent_record_created=None,
                superseded_record_id=None,
            )
        else:
            result = _promote_or_defer(
                connection,
                blobs,
                candidate=candidate,
                value=value,
                reason=reason,
                idempotency_key=idempotency_key,
                review_evidence=evidence,
                current_review_events=current_events,
                all_candidate_events=all_candidate_events,
                supersedes_record_id=supersedes_record_id,
            )

        verify_ledger(connection)
        _verify_result_link(connection, result)
        connection.commit()
        return result
    except BaseException as exc:
        if connection.in_transaction:
            connection.rollback()
        if isinstance(exc, LedgerError):
            raise
        raise VerificationError("candidate review transaction failed") from exc


def _prior_review_matches(
    connection: sqlite3.Connection,
    prior: sqlite3.Row,
    *,
    candidate_id: str,
    outcome: str,
    reason: str,
    evidence: tuple[str, ...],
    supersedes_record_id: str | None,
) -> bool:
    try:
        payload = parse_json_value(prior["payload_json"])
    except ValidationError as exc:
        raise VerificationError("stored candidate review payload is invalid") from exc
    if not isinstance(payload, dict):
        raise VerificationError("stored candidate review payload is invalid")
    if (
        prior["candidate_id"] != candidate_id
        or payload.get("requested_outcome") != outcome
        or payload.get("reason") != reason
        or payload.get("evidence_uris") != list(evidence)
    ):
        return False

    if outcome != "promote":
        return (
            prior["action"] == outcome
            and payload.get("policy_code") == _NON_PROMOTION_POLICIES[outcome]
            and supersedes_record_id is None
        )
    if prior["action"] == "promote":
        return payload.get("superseded_record_id") == supersedes_record_id
    if prior["action"] != "defer":
        return False
    policy = payload.get("policy_code")
    if policy in {"decision_conflict", "constitution_conflict"}:
        return supersedes_record_id is None
    if policy != "newer_evidence_required":
        return False
    return _inferred_deferred_target(
        connection,
        candidate_id,
        review_created_at=prior["created_at"],
    ) == supersedes_record_id


def _inferred_deferred_target(
    connection: sqlite3.Connection,
    candidate_id: str,
    *,
    review_created_at: str,
) -> str | None | object:
    candidate = connection.execute(
        "SELECT * FROM memory_candidates WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    if candidate is None:
        return _AMBIGUOUS_TARGET
    cutoff = connection.execute(
        "SELECT MAX(event_seq) FROM memory_events WHERE created_at <= ?",
        (review_created_at,),
    ).fetchone()[0]
    historical = _validated_replay(
        connection,
        through_event_seq=0 if cutoff is None else cutoff,
    )
    matches = [
        row
        for row in connection.execute(
            """
            SELECT *
            FROM memory_records
            WHERE scope_kind = ? AND topic_key = ? AND record_type = ?
            ORDER BY record_seq ASC, record_id ASC
            """,
            (
                candidate["scope_kind"],
                candidate["topic_key"],
                candidate["record_type"],
            ),
        )
        if historical.get(row["record_id"], (None,))[0] == "active"
        and scope_keys_equal(row["scope_key"], candidate["scope_key"])
    ]
    if any(
        row["content_sha256"] == candidate["content_sha256"] for row in matches
    ):
        return None
    differing = [
        row for row in matches if row["content_sha256"] != candidate["content_sha256"]
    ]
    if candidate["record_type"] in _AUTHORITATIVE_TYPES and len(differing) == 1:
        return differing[0]["record_id"]
    if not differing:
        return None
    return _AMBIGUOUS_TARGET


def _promote_or_defer(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    candidate: sqlite3.Row,
    value: Any,
    reason: str,
    idempotency_key: str,
    review_evidence: tuple[str, ...],
    current_review_events: tuple[EventEvidence, ...],
    all_candidate_events: tuple[EventEvidence, ...],
    supersedes_record_id: str | None,
) -> CandidateActionResult:
    matches = _active_matching_records(connection, candidate)
    exact = [
        row for row in matches if row["content_sha256"] == candidate["content_sha256"]
    ]
    differing = [
        row for row in matches if row["content_sha256"] != candidate["content_sha256"]
    ]
    record_type = candidate["record_type"]

    if record_type not in _AUTHORITATIVE_TYPES and supersedes_record_id is not None:
        raise ValidationError("this record type cannot supersede automatically")
    if record_type in _AUTHORITATIVE_TYPES and exact and differing:
        if supersedes_record_id is not None:
            raise ValidationError(
                "an exact authoritative duplicate cannot resolve another active conflict"
            )
        return _defer_promotion(
            connection,
            candidate_id=candidate["candidate_id"],
            reason=reason,
            idempotency_key=idempotency_key,
            evidence=review_evidence,
            policy_code=_CONFLICT_POLICIES[record_type],
        )
    if exact:
        if supersedes_record_id is not None:
            raise ValidationError("an exact duplicate cannot supersede another record")
        if record_type == "constitution" and not _has_direct_user_event(
            all_candidate_events
        ):
            return _defer_promotion(
                connection,
                candidate_id=candidate["candidate_id"],
                reason=reason,
                idempotency_key=idempotency_key,
                evidence=review_evidence,
                policy_code="newer_evidence_required",
            )
        return apply_candidate_action_in_transaction(
            connection,
            candidate_id=candidate["candidate_id"],
            action="promote",
            requested_outcome="promote",
            reason=reason,
            idempotency_key=idempotency_key,
            evidence_uris=review_evidence,
            policy_code="linked_duplicate",
            permanent_record_id=exact[0]["record_id"],
            permanent_record_created=False,
            superseded_record_id=None,
        )

    target: sqlite3.Row | None = None
    if record_type in _AUTHORITATIVE_TYPES and differing:
        if supersedes_record_id is None:
            return _defer_promotion(
                connection,
                candidate_id=candidate["candidate_id"],
                reason=reason,
                idempotency_key=idempotency_key,
                evidence=review_evidence,
                policy_code=_CONFLICT_POLICIES[record_type],
            )
        if len(differing) != 1 or differing[0]["record_id"] != supersedes_record_id:
            raise ValidationError(
                "supersession target must be the sole differing active record"
            )
        target = differing[0]
        if not _has_newer_evidence(
            connection,
            blobs,
            target=target,
            current_review_events=current_review_events,
            all_candidate_events=all_candidate_events,
            require_user_prompt=record_type == "constitution",
        ):
            return _defer_promotion(
                connection,
                candidate_id=candidate["candidate_id"],
                reason=reason,
                idempotency_key=idempotency_key,
                evidence=review_evidence,
                policy_code="newer_evidence_required",
            )
    elif supersedes_record_id is not None:
        raise ValidationError("supersession target is not an active conflict")

    if record_type == "constitution" and not _has_direct_user_event(
        all_candidate_events
    ):
        return _defer_promotion(
            connection,
            candidate_id=candidate["candidate_id"],
            reason=reason,
            idempotency_key=idempotency_key,
            evidence=review_evidence,
            policy_code="newer_evidence_required",
        )

    capture_evidence = tuple(
        row["uri"]
        for row in connection.execute(
            """
            SELECT uri
            FROM memory_candidate_evidence
            WHERE candidate_id = ?
            ORDER BY ordinal
            """,
            (candidate["candidate_id"],),
        )
    )
    permanent = propose_record_in_transaction(
        connection,
        record_type=record_type,
        scope_kind=candidate["scope_kind"],
        scope_key=candidate["scope_key"],
        topic_key=candidate["topic_key"],
        statement=candidate["statement"],
        value=value,
        evidence_uris=capture_evidence,
        idempotency_key=f"candidate:{candidate['candidate_id']}:propose",
    )
    apply_action_in_transaction(
        connection,
        record_id=permanent.record_id,
        action="activate",
        idempotency_key=f"candidate:{candidate['candidate_id']}:activate",
    )
    if target is not None:
        apply_action_in_transaction(
            connection,
            record_id=target["record_id"],
            action="supersede",
            replacement_record_id=permanent.record_id,
            idempotency_key=f"candidate:{candidate['candidate_id']}:supersede",
        )
    return apply_candidate_action_in_transaction(
        connection,
        candidate_id=candidate["candidate_id"],
        action="promote",
        requested_outcome="promote",
        reason=reason,
        idempotency_key=idempotency_key,
        evidence_uris=review_evidence,
        policy_code="promoted_superseding" if target is not None else "promoted_new",
        permanent_record_id=permanent.record_id,
        permanent_record_created=True,
        superseded_record_id=target["record_id"] if target is not None else None,
    )


def _defer_promotion(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    reason: str,
    idempotency_key: str,
    evidence: tuple[str, ...],
    policy_code: str,
) -> CandidateActionResult:
    return apply_candidate_action_in_transaction(
        connection,
        candidate_id=candidate_id,
        action="defer",
        requested_outcome="promote",
        reason=reason,
        idempotency_key=idempotency_key,
        evidence_uris=evidence,
        policy_code=policy_code,
        permanent_record_id=None,
        permanent_record_created=None,
        superseded_record_id=None,
    )


def _active_matching_records(
    connection: sqlite3.Connection, candidate: sqlite3.Row
) -> list[sqlite3.Row]:
    return [
        row
        for row in connection.execute(
            """
            SELECT r.*
            FROM memory_records AS r
            JOIN memory_current AS p ON p.record_id = r.record_id
            WHERE p.state = 'active'
              AND r.scope_kind = ?
              AND r.topic_key = ?
              AND r.record_type = ?
            ORDER BY r.record_seq ASC, r.record_id ASC
            """,
            (
                candidate["scope_kind"],
                candidate["topic_key"],
                candidate["record_type"],
            ),
        )
        if scope_keys_equal(row["scope_key"], candidate["scope_key"])
    ]


def _candidate_history_evidence(
    connection: sqlite3.Connection, candidate_id: str
) -> tuple[str, ...]:
    evidence = [
        row["uri"]
        for row in connection.execute(
            """
            SELECT uri
            FROM memory_candidate_evidence
            WHERE candidate_id = ?
            ORDER BY ordinal
            """,
            (candidate_id,),
        )
    ]
    for row in connection.execute(
        """
        SELECT payload_json
        FROM memory_candidate_events
        WHERE candidate_id = ? AND action <> 'capture'
        ORDER BY event_seq
        """,
        (candidate_id,),
    ):
        payload = parse_json_value(row["payload_json"])
        if not isinstance(payload, dict) or not isinstance(
            payload.get("evidence_uris"), list
        ):
            raise VerificationError("candidate review evidence payload is invalid")
        evidence.extend(payload["evidence_uris"])
    return tuple(evidence)


def _resolve_local_events(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    uris: Sequence[str],
    *,
    expected_session: str | None,
) -> tuple[EventEvidence, ...]:
    events: list[EventEvidence] = []
    for uri in uris:
        if urlsplit(uri).scheme.lower() != "codex-event":
            continue
        try:
            event = resolve_event_evidence(connection, blobs, uri)
        except (BlobIntegrityError, OSError, RecordNotFound, ValueError) as exc:
            raise VerificationError("local review evidence verification failed") from exc
        if expected_session is not None and event.session_id != expected_session:
            raise VerificationError("local review evidence belongs to another session")
        events.append(event)
    return tuple(events)


def _has_direct_user_event(events: Sequence[EventEvidence]) -> bool:
    return any(event.event_name == "UserPromptSubmit" for event in events)


def _has_newer_evidence(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    target: sqlite3.Row,
    current_review_events: tuple[EventEvidence, ...],
    all_candidate_events: tuple[EventEvidence, ...],
    require_user_prompt: bool,
) -> bool:
    target_events = list(
        _resolve_local_events(
            connection,
            blobs,
            tuple(
                row["uri"]
                for row in connection.execute(
                    """
                    SELECT uri
                    FROM memory_evidence
                    WHERE record_id = ?
                    ORDER BY ordinal
                    """,
                    (target["record_id"],),
                )
            ),
            expected_session=None,
        )
    )
    linked_candidates = connection.execute(
        """
        SELECT c.candidate_id, c.session_id
        FROM memory_candidates AS c
        JOIN memory_candidate_current AS p
          ON p.candidate_id = c.candidate_id
        WHERE p.state = 'promoted' AND p.permanent_record_id = ?
        ORDER BY c.candidate_seq ASC, c.candidate_id ASC
        """,
        (target["record_id"],),
    ).fetchall()
    for linked in linked_candidates:
        target_events.extend(
            _resolve_local_events(
                connection,
                blobs,
                _candidate_history_evidence(connection, linked["candidate_id"]),
                expected_session=linked["session_id"],
            )
        )

    if target_events:
        highwater = max(event.event_id for event in target_events)
        newer = tuple(
            event for event in all_candidate_events if event.event_id > highwater
        )
    else:
        newer = current_review_events
    if require_user_prompt:
        newer = tuple(
            event for event in newer if event.event_name == "UserPromptSubmit"
        )
    return bool(newer)


def _verify_result_link(
    connection: sqlite3.Connection, result: CandidateActionResult
) -> None:
    candidate = connection.execute(
        """
        SELECT state, permanent_record_id
        FROM memory_candidate_current
        WHERE candidate_id = ?
        """,
        (result.candidate_id,),
    ).fetchone()
    if candidate is None or (
        candidate["state"],
        candidate["permanent_record_id"],
    ) != (result.state, result.permanent_record_id):
        raise VerificationError("candidate review result link is inconsistent")
    if result.action != "promote":
        return
    permanent = connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?",
        (result.permanent_record_id,),
    ).fetchone()
    if permanent is None or permanent["state"] != "active":
        raise VerificationError("promoted permanent record is not active")
    if result.superseded_record_id is not None:
        superseded = connection.execute(
            "SELECT state FROM memory_current WHERE record_id = ?",
            (result.superseded_record_id,),
        ).fetchone()
        if superseded is None or superseded["state"] != "superseded":
            raise VerificationError("superseded permanent record is not superseded")
