from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Callable, Sequence
import uuid

from codex_memory.database import (
    APPEND_ONLY_UNIQUE_KEYS,
    STRUCTURED_TABLE_STATEMENTS,
    _VERSION_THREE_STATEMENTS,
    append_only_insert_guard_statement,
    append_only_trigger_statement,
    next_memory_timestamp,
)
from codex_memory.ledger_common import (
    IdempotencyConflict,
    InvalidTransition,
    LedgerError,
    RecordNotFound,
    ValidatedRecordContent,
    ValidationError,
    VerificationError,
    _hash_preimage,
    _RECORD_TYPES,
    _SCOPE_KINDS,
    _TOPIC_PATTERN,
    _require_bounded_text,
    _validate_idempotency_key,
    canonical_json,
    parse_json_value,
    validate_evidence_uris,
    validate_new_evidence_uris,
    validate_record_content,
)


_ACTIONS = frozenset({"activate", "supersede", "invalidate", "close", "reopen"})
_STATES = frozenset({"proposed", "active", "superseded", "invalidated", "closed"})
_SQLITE_TOKENIZER_WHITESPACE = frozenset({"\u0009", "\u000a", "\u000c", "\u000d", "\u0020"})


def _is_sqlite_tokenizer_whitespace(character: str) -> bool:
    return character in _SQLITE_TOKENIZER_WHITESPACE


_TRANSITIONS = {
    ("activate", "proposed"): "active",
    ("supersede", "active"): "superseded",
    ("invalidate", "proposed"): "invalidated",
    ("invalidate", "active"): "invalidated",
    ("close", "active"): "closed",
    ("reopen", "closed"): "active",
}
_REQUIRED_COLUMN_SHAPES = {
    "memory_records": (
        ("record_seq", "INTEGER", 0, None, 1),
        ("record_id", "TEXT", 1, None, 0),
        ("record_type", "TEXT", 1, None, 0),
        ("scope_kind", "TEXT", 1, None, 0),
        ("scope_key", "TEXT", 1, None, 0),
        ("topic_key", "TEXT", 1, None, 0),
        ("statement", "TEXT", 1, None, 0),
        ("value_json", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("content_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_events": (
        ("event_seq", "INTEGER", 0, None, 1),
        ("event_id", "TEXT", 1, None, 0),
        ("record_id", "TEXT", 1, None, 0),
        ("action", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("event_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_evidence": (
        ("evidence_seq", "INTEGER", 0, None, 1),
        ("evidence_id", "TEXT", 1, None, 0),
        ("record_id", "TEXT", 1, None, 0),
        ("ordinal", "INTEGER", 1, None, 0),
        ("uri", "TEXT", 1, None, 0),
        ("evidence_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_relations": (
        ("relation_seq", "INTEGER", 0, None, 1),
        ("relation_id", "TEXT", 1, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("relation_type", "TEXT", 1, None, 0),
        ("source_record_id", "TEXT", 1, None, 0),
        ("target_record_id", "TEXT", 1, None, 0),
        ("relation_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_current": (
        ("record_id", "TEXT", 0, None, 1),
        ("state", "TEXT", 1, None, 0),
        ("last_event_seq", "INTEGER", 1, None, 0),
        ("last_event_id", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ),
    "memory_candidates": (
        ("candidate_seq", "INTEGER", 0, None, 1),
        ("candidate_id", "TEXT", 1, None, 0),
        ("record_type", "TEXT", 1, None, 0),
        ("scope_kind", "TEXT", 1, None, 0),
        ("scope_key", "TEXT", 1, None, 0),
        ("topic_key", "TEXT", 1, None, 0),
        ("statement", "TEXT", 1, None, 0),
        ("value_json", "TEXT", 1, None, 0),
        ("session_id", "TEXT", 1, None, 0),
        ("turn_id", "TEXT", 1, None, 0),
        ("context_id", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("content_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_candidate_evidence": (
        ("evidence_seq", "INTEGER", 0, None, 1),
        ("evidence_id", "TEXT", 1, None, 0),
        ("candidate_id", "TEXT", 1, None, 0),
        ("ordinal", "INTEGER", 1, None, 0),
        ("uri", "TEXT", 1, None, 0),
        ("evidence_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_candidate_events": (
        ("event_seq", "INTEGER", 0, None, 1),
        ("event_id", "TEXT", 1, None, 0),
        ("candidate_id", "TEXT", 1, None, 0),
        ("action", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("event_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_candidate_current": (
        ("candidate_id", "TEXT", 0, None, 1),
        ("state", "TEXT", 1, None, 0),
        ("last_event_seq", "INTEGER", 1, None, 0),
        ("last_event_id", "TEXT", 1, None, 0),
        ("permanent_record_id", "TEXT", 0, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ),
}
_REQUIRED_INDEXES = {
    "memory_records_scope_topic_type",
    "memory_events_record_event_seq",
    "memory_evidence_record_ordinal",
    "memory_relations_source",
    "memory_relations_target",
    "memory_current_state_record_id",
    "memory_candidates_scope_topic_type",
    "memory_candidates_session_seq",
    "memory_candidate_events_candidate_event_seq",
    "memory_candidate_evidence_candidate_ordinal",
    "memory_candidate_current_state_candidate_id",
}
_REQUIRED_INDEX_COLUMNS = {
    "memory_records_scope_topic_type": (
        "scope_kind",
        "scope_key",
        "topic_key",
        "record_type",
        "record_seq",
    ),
    "memory_events_record_event_seq": ("record_id", "event_seq"),
    "memory_evidence_record_ordinal": ("record_id", "ordinal"),
    "memory_relations_source": ("source_record_id", "relation_seq"),
    "memory_relations_target": ("target_record_id", "relation_seq"),
    "memory_current_state_record_id": ("state", "record_id"),
    "memory_candidates_scope_topic_type": (
        "scope_kind",
        "scope_key",
        "topic_key",
        "record_type",
        "candidate_seq",
    ),
    "memory_candidates_session_seq": ("session_id", "candidate_seq"),
    "memory_candidate_events_candidate_event_seq": ("candidate_id", "event_seq"),
    "memory_candidate_evidence_candidate_ordinal": ("candidate_id", "ordinal"),
    "memory_candidate_current_state_candidate_id": ("state", "candidate_id"),
}
_REQUIRED_INDEX_TABLES = {
    "memory_records_scope_topic_type": "memory_records",
    "memory_events_record_event_seq": "memory_events",
    "memory_evidence_record_ordinal": "memory_evidence",
    "memory_relations_source": "memory_relations",
    "memory_relations_target": "memory_relations",
    "memory_current_state_record_id": "memory_current",
    "memory_candidates_scope_topic_type": "memory_candidates",
    "memory_candidates_session_seq": "memory_candidates",
    "memory_candidate_events_candidate_event_seq": "memory_candidate_events",
    "memory_candidate_evidence_candidate_ordinal": "memory_candidate_evidence",
    "memory_candidate_current_state_candidate_id": "memory_candidate_current",
}
_EXPECTED_UNIQUE_COLUMNS = {
    "memory_records": frozenset({("record_id",), ("idempotency_key",)}),
    "memory_events": frozenset({("event_id",), ("idempotency_key",)}),
    "memory_evidence": frozenset({("evidence_id",), ("record_id", "ordinal")}),
    "memory_relations": frozenset({("relation_id",)}),
    "memory_current": frozenset({("record_id",)}),
    "memory_candidates": frozenset({("candidate_id",), ("idempotency_key",)}),
    "memory_candidate_evidence": frozenset(
        {("evidence_id",), ("candidate_id", "ordinal")}
    ),
    "memory_candidate_events": frozenset({("event_id",), ("idempotency_key",)}),
    "memory_candidate_current": frozenset({("candidate_id",)}),
}
_EXPECTED_FOREIGN_KEYS = {
    "memory_records": frozenset(),
    "memory_events": frozenset(
        {("record_id", "memory_records", "record_id", "NO ACTION", "NO ACTION", "NONE")}
    ),
    "memory_evidence": frozenset(
        {("record_id", "memory_records", "record_id", "NO ACTION", "NO ACTION", "NONE")}
    ),
    "memory_relations": frozenset(
        {
            ("event_id", "memory_events", "event_id", "NO ACTION", "NO ACTION", "NONE"),
            (
                "source_record_id",
                "memory_records",
                "record_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
            (
                "target_record_id",
                "memory_records",
                "record_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        }
    ),
    "memory_current": frozenset(
        {
            ("record_id", "memory_records", "record_id", "NO ACTION", "NO ACTION", "NONE"),
            ("last_event_id", "memory_events", "event_id", "NO ACTION", "NO ACTION", "NONE"),
        }
    ),
    "memory_candidates": frozenset(),
    "memory_candidate_evidence": frozenset(
        {
            (
                "candidate_id",
                "memory_candidates",
                "candidate_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            )
        }
    ),
    "memory_candidate_events": frozenset(
        {
            (
                "candidate_id",
                "memory_candidates",
                "candidate_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            )
        }
    ),
    "memory_candidate_current": frozenset(
        {
            (
                "candidate_id",
                "memory_candidates",
                "candidate_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
            (
                "last_event_id",
                "memory_candidate_events",
                "event_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
            (
                "permanent_record_id",
                "memory_records",
                "record_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        }
    ),
}
_EXPECTED_TABLE_SQL = dict(
    zip(tuple(_REQUIRED_COLUMN_SHAPES)[:5], STRUCTURED_TABLE_STATEMENTS, strict=True)
)
_EXPECTED_TABLE_SQL.update(
    zip(tuple(_REQUIRED_COLUMN_SHAPES)[5:], _VERSION_THREE_STATEMENTS[:4], strict=True)
)
_EXPECTED_CANDIDATE_INDEX_SQL = dict(
    zip(
        (
            "memory_candidates_scope_topic_type",
            "memory_candidates_session_seq",
            "memory_candidate_events_candidate_event_seq",
            "memory_candidate_evidence_candidate_ordinal",
            "memory_candidate_current_state_candidate_id",
        ),
        _VERSION_THREE_STATEMENTS[4:],
        strict=True,
    )
)
_APPEND_ONLY_TABLES = (
    "memory_records",
    "memory_events",
    "memory_evidence",
    "memory_relations",
    "memory_candidates",
    "memory_candidate_evidence",
    "memory_candidate_events",
)


@dataclass(frozen=True, slots=True)
class RecordResult:
    record_id: str
    event_id: str
    record_type: str
    scope_kind: str
    scope_key: str
    topic_key: str
    statement: str
    value: Any
    evidence_uris: tuple[str, ...]
    state: str
    content_sha256: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ActionResult:
    record_id: str
    event_id: str
    action: str
    state: str
    replacement_record_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class RecordView:
    record_id: str
    record_type: str
    scope_kind: str
    scope_key: str
    topic_key: str
    statement: str
    value: Any
    evidence_uris: tuple[str, ...]
    state: str
    content_sha256: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    schema_version: int
    records: int
    events: int
    evidence: int
    relations: int
    current: int
    candidates: int
    candidate_events: int
    candidate_evidence: int
    candidate_current: int


def propose_record(
    connection: sqlite3.Connection,
    *,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    idempotency_key: str,
    value: Any = None,
    evidence_uris: Sequence[str] = (),
) -> RecordResult:
    """Atomically append one immutable record, proposal event, and evidence set."""
    content, evidence = _validate_proposal(
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value=value,
        evidence_uris=evidence_uris,
        idempotency_key=idempotency_key,
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = _propose_record_validated_in_transaction(
            connection,
            content=content,
            record_type=record_type,
            scope_kind=scope_kind,
            scope_key=scope_key,
            topic_key=topic_key,
            statement=statement,
            value=value,
            evidence=evidence,
            idempotency_key=idempotency_key,
        )
        connection.commit()
        return result
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def propose_record_in_transaction(
    connection: sqlite3.Connection,
    *,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    idempotency_key: str,
    value: Any = None,
    evidence_uris: Sequence[str] = (),
) -> RecordResult:
    """Append one proposal while leaving transaction ownership to the caller."""
    if not connection.in_transaction:
        raise LedgerError("record proposal requires an active transaction")
    content, evidence = _validate_proposal(
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value=value,
        evidence_uris=evidence_uris,
        idempotency_key=idempotency_key,
    )
    return _propose_record_validated_in_transaction(
        connection,
        content=content,
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value=value,
        evidence=evidence,
        idempotency_key=idempotency_key,
    )


def _propose_record_validated_in_transaction(
    connection: sqlite3.Connection,
    *,
    content: ValidatedRecordContent,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    value: Any,
    evidence: tuple[str, ...],
    idempotency_key: str,
) -> RecordResult:
    prior_record = connection.execute(
        "SELECT * FROM memory_records WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    prior_event = connection.execute(
        "SELECT * FROM memory_events WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    if prior_record is not None:
        if (
            prior_event is None
            or prior_event["record_id"] != prior_record["record_id"]
            or prior_event["action"] != "propose"
        ):
            raise IdempotencyConflict("proposal idempotency key conflict")
        if not _proposal_matches(
            connection,
            prior_record,
            record_type=record_type,
            scope_kind=scope_kind,
            scope_key=scope_key,
            topic_key=topic_key,
            statement=statement,
            value_json=content.value_json,
            evidence_uris=evidence,
            content_sha256=content.content_sha256,
        ):
            raise IdempotencyConflict("proposal idempotency key conflict")
        return _proposal_result(connection, prior_record)
    if prior_event is not None:
        raise IdempotencyConflict("proposal idempotency key conflict")

    created_at = next_memory_timestamp(connection)
    record_id = _opaque_id("mem")
    event_id = _opaque_id("evt")
    connection.execute(
        """
        INSERT INTO memory_records (
            record_id, record_type, scope_kind, scope_key, topic_key,
            statement, value_json, idempotency_key, content_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            record_type,
            scope_kind,
            scope_key,
            topic_key,
            statement,
            content.value_json,
            idempotency_key,
            content.content_sha256,
            created_at,
        ),
    )
    event_cursor = connection.execute(
        """
        INSERT INTO memory_events (
            event_id, record_id, action, payload_json, idempotency_key,
            event_sha256, created_at
        ) VALUES (?, ?, 'propose', '{}', ?, ?, ?)
        """,
        (
            event_id,
            record_id,
            idempotency_key,
            _hash_preimage(
                {"action": "propose", "payload": {}, "record_id": record_id}
            ),
            created_at,
        ),
    )
    for ordinal, uri in enumerate(evidence):
        connection.execute(
            """
            INSERT INTO memory_evidence (
                evidence_id, record_id, ordinal, uri, evidence_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _opaque_id("evi"),
                record_id,
                ordinal,
                uri,
                _hash_preimage({"record_id": record_id, "uri": uri}),
                created_at,
            ),
        )
    connection.execute(
        """
        INSERT INTO memory_current (
            record_id, state, last_event_seq, last_event_id, updated_at
        ) VALUES (?, 'proposed', ?, ?, ?)
        """,
        (record_id, event_cursor.lastrowid, event_id, created_at),
    )
    return RecordResult(
        record_id=record_id,
        event_id=event_id,
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value=value,
        evidence_uris=evidence,
        state="proposed",
        content_sha256=content.content_sha256,
        created_at=created_at,
    )


def apply_action(
    connection: sqlite3.Connection,
    *,
    record_id: str,
    action: str,
    idempotency_key: str,
    replacement_record_id: str | None = None,
) -> ActionResult:
    """Atomically append a validated lifecycle event and update its projection."""
    payload, payload_json = _validate_action_request(
        record_id=record_id,
        action=action,
        idempotency_key=idempotency_key,
        replacement_record_id=replacement_record_id,
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = _apply_action_validated_in_transaction(
            connection,
            record_id=record_id,
            action=action,
            idempotency_key=idempotency_key,
            replacement_record_id=replacement_record_id,
            payload=payload,
            payload_json=payload_json,
        )
        connection.commit()
        return result
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def apply_action_in_transaction(
    connection: sqlite3.Connection,
    *,
    record_id: str,
    action: str,
    idempotency_key: str,
    replacement_record_id: str | None = None,
) -> ActionResult:
    """Append one action while leaving transaction ownership to the caller."""
    if not connection.in_transaction:
        raise LedgerError("record action requires an active transaction")
    payload, payload_json = _validate_action_request(
        record_id=record_id,
        action=action,
        idempotency_key=idempotency_key,
        replacement_record_id=replacement_record_id,
    )
    return _apply_action_validated_in_transaction(
        connection,
        record_id=record_id,
        action=action,
        idempotency_key=idempotency_key,
        replacement_record_id=replacement_record_id,
        payload=payload,
        payload_json=payload_json,
    )


def _validate_action_request(
    *,
    record_id: str,
    action: str,
    idempotency_key: str,
    replacement_record_id: str | None,
) -> tuple[dict[str, str], str]:
    _require_bounded_text(record_id, "record ID", 256, nonblank=True)
    if not isinstance(action, str) or action not in _ACTIONS:
        raise ValidationError("invalid lifecycle action")
    _validate_idempotency_key(idempotency_key)
    if action == "supersede":
        _require_bounded_text(
            replacement_record_id, "replacement record ID", 256, nonblank=True
        )
        payload = {"replacement_record_id": replacement_record_id}
    else:
        if replacement_record_id is not None:
            raise ValidationError("replacement is valid only for supersede")
        payload = {}
    return payload, canonical_json(payload)


def _apply_action_validated_in_transaction(
    connection: sqlite3.Connection,
    *,
    record_id: str,
    action: str,
    idempotency_key: str,
    replacement_record_id: str | None,
    payload: dict[str, str],
    payload_json: str,
) -> ActionResult:
    prior_event = connection.execute(
        "SELECT * FROM memory_events WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    prior_record = connection.execute(
        "SELECT record_id FROM memory_records WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    if prior_record is not None:
        raise IdempotencyConflict("action idempotency key conflict")
    if prior_event is not None:
        if (
            prior_event["record_id"],
            prior_event["action"],
            prior_event["payload_json"],
        ) != (record_id, action, payload_json):
            raise IdempotencyConflict("action idempotency key conflict")
        return _action_result_for_event(connection, prior_event)

    record = connection.execute(
        "SELECT record_id FROM memory_records WHERE record_id = ?", (record_id,)
    ).fetchone()
    if record is None:
        raise RecordNotFound("record not found")
    current = connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?", (record_id,)
    ).fetchone()
    if current is None:
        raise VerificationError("record projection is missing")

    if action == "supersede":
        if replacement_record_id == record_id:
            raise InvalidTransition("a record cannot supersede itself")
        replacement = connection.execute(
            "SELECT record_id FROM memory_records WHERE record_id = ?",
            (replacement_record_id,),
        ).fetchone()
        if replacement is None:
            raise RecordNotFound("replacement record not found")
    new_state = _TRANSITIONS.get((action, current["state"]))
    if new_state is None:
        raise InvalidTransition("invalid lifecycle transition")

    created_at = next_memory_timestamp(connection)
    event_id = _opaque_id("evt")
    event_cursor = connection.execute(
        """
        INSERT INTO memory_events (
            event_id, record_id, action, payload_json, idempotency_key,
            event_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            record_id,
            action,
            payload_json,
            idempotency_key,
            _hash_preimage(
                {"action": action, "payload": payload, "record_id": record_id}
            ),
            created_at,
        ),
    )
    if action == "supersede":
        relation = {
            "event_id": event_id,
            "relation_type": "superseded_by",
            "source_record_id": record_id,
            "target_record_id": replacement_record_id,
        }
        connection.execute(
            """
            INSERT INTO memory_relations (
                relation_id, event_id, relation_type, source_record_id,
                target_record_id, relation_sha256, created_at
            ) VALUES (?, ?, 'superseded_by', ?, ?, ?, ?)
            """,
            (
                _opaque_id("rel"),
                event_id,
                record_id,
                replacement_record_id,
                _hash_preimage(relation),
                created_at,
            ),
        )
    updated = connection.execute(
        """
        UPDATE memory_current
        SET state = ?, last_event_seq = ?, last_event_id = ?, updated_at = ?
        WHERE record_id = ?
        """,
        (new_state, event_cursor.lastrowid, event_id, created_at, record_id),
    )
    if updated.rowcount != 1:
        raise VerificationError("record projection update failed")
    return ActionResult(
        record_id=record_id,
        event_id=event_id,
        action=action,
        state=new_state,
        replacement_record_id=replacement_record_id,
        created_at=created_at,
    )


def show_record(connection: sqlite3.Connection, record_id: str) -> RecordView:
    _require_bounded_text(record_id, "record ID", 256, nonblank=True)
    row = connection.execute(
        """
        SELECT r.*, c.state, c.updated_at
        FROM memory_records AS r
        JOIN memory_current AS c ON c.record_id = r.record_id
        WHERE r.record_id = ?
        """,
        (record_id,),
    ).fetchone()
    if row is None:
        exists = connection.execute(
            "SELECT 1 FROM memory_records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if exists is None:
            raise RecordNotFound("record not found")
        raise VerificationError("record projection is missing")
    return _record_view(connection, row)


def list_records(
    connection: sqlite3.Connection,
    *,
    scope_kind: str | None = None,
    scope_key: str | None = None,
    topic_key: str | None = None,
    record_type: str | None = None,
    state: str | None = None,
    limit: int = 100,
) -> tuple[RecordView, ...]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValidationError("list limit must be between 1 and 100")
    clauses: list[str] = []
    parameters: list[Any] = []
    filters = {
        "r.scope_kind": scope_kind,
        "r.scope_key": scope_key,
        "r.topic_key": topic_key,
        "r.record_type": record_type,
        "c.state": state,
    }
    if scope_kind is not None and (
        not isinstance(scope_kind, str) or scope_kind not in _SCOPE_KINDS
    ):
        raise ValidationError("invalid scope kind")
    if record_type is not None and (
        not isinstance(record_type, str) or record_type not in _RECORD_TYPES
    ):
        raise ValidationError("invalid record type")
    if state is not None and (not isinstance(state, str) or state not in _STATES):
        raise ValidationError("invalid state")
    if topic_key is not None and (
        not isinstance(topic_key, str) or _TOPIC_PATTERN.fullmatch(topic_key) is None
    ):
        raise ValidationError("invalid topic key")
    if scope_key is not None:
        _require_bounded_text(scope_key, "scope key", 256, nonblank=True)
    for column, value in filters.items():
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    parameters.append(limit)
    rows = connection.execute(
        "SELECT r.*, c.state, c.updated_at "
        "FROM memory_records AS r JOIN memory_current AS c ON c.record_id = r.record_id"
        + where
        + " ORDER BY r.record_seq LIMIT ?",
        parameters,
    ).fetchall()
    return tuple(_record_view(connection, row) for row in rows)


def rebuild_projection(connection: sqlite3.Connection) -> int:
    """Atomically replace only the disposable projection from immutable sources."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("DELETE FROM memory_current")
        projection = _validated_replay(connection)
        for record_id in sorted(projection):
            state, event_seq, event_id, updated_at = projection[record_id]
            connection.execute(
                """
                INSERT INTO memory_current (
                    record_id, state, last_event_seq, last_event_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, state, event_seq, event_id, updated_at),
            )
        connection.commit()
        return len(projection)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def verify_ledger(
    connection: sqlite3.Connection,
    *,
    _after_first_snapshot_read: Callable[[], None] | None = None,
    _before_first_snapshot_read: Callable[[], None] | None = None,
    _validation_stage: Callable[[str], None] | None = None,
) -> VerificationResult:
    """Verify one snapshot of schema, immutable streams, and current projection."""
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN")
    observer_failed = False

    def before_read() -> None:
        nonlocal observer_failed
        try:
            _before_first_snapshot_read()
        except BaseException:
            observer_failed = True
            raise

    try:
        return _verify_snapshot(
            connection,
            _after_first_snapshot_read=_after_first_snapshot_read,
            _validation_stage=_validation_stage,
            _before_first_snapshot_read=before_read if _before_first_snapshot_read is not None else None,
        )
    except VerificationError:
        raise
    except BaseException as exc:
        if observer_failed:
            raise
        raise VerificationError("ledger verification failed") from exc
    finally:
        if owns_transaction and connection.in_transaction:
            connection.rollback()


def connect_readonly_database(path: Path) -> sqlite3.Connection:
    """Open one existing SQLite file with URI mode=ro and query-only enabled."""
    connection: sqlite3.Connection | None = None
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("database is not a regular file")
        connection = sqlite3.connect(
            resolved.as_uri() + "?mode=ro",
            uri=True,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA recursive_triggers=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA query_only=ON")
        return connection
    except (OSError, sqlite3.Error) as exc:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        raise VerificationError("ledger database is unavailable") from exc


def verify_database(path: Path) -> VerificationResult:
    connection = connect_readonly_database(path)
    try:
        connection.execute("BEGIN")
        return verify_ledger(connection)
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _validate_proposal(
    *,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    value: Any,
    evidence_uris: Sequence[str],
    idempotency_key: str,
) -> tuple[ValidatedRecordContent, tuple[str, ...]]:
    _validate_idempotency_key(idempotency_key)
    content = validate_record_content(
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value=value,
    )
    return content, validate_new_evidence_uris(evidence_uris)


def _proposal_matches(
    connection: sqlite3.Connection,
    record: sqlite3.Row,
    *,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    value_json: str,
    evidence_uris: tuple[str, ...],
    content_sha256: str,
) -> bool:
    stored_evidence = tuple(
        row[0]
        for row in connection.execute(
            "SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal",
            (record["record_id"],),
        )
    )
    return (
        record["record_type"],
        record["scope_kind"],
        record["scope_key"],
        record["topic_key"],
        record["statement"],
        record["value_json"],
        record["content_sha256"],
        stored_evidence,
    ) == (
        record_type,
        scope_kind,
        scope_key,
        topic_key,
        statement,
        value_json,
        content_sha256,
        evidence_uris,
    )


def _proposal_result(connection: sqlite3.Connection, record: sqlite3.Row) -> RecordResult:
    event = connection.execute(
        "SELECT event_id FROM memory_events WHERE record_id = ? AND action = 'propose'",
        (record["record_id"],),
    ).fetchone()
    evidence = tuple(
        row[0]
        for row in connection.execute(
            "SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal",
            (record["record_id"],),
        )
    )
    return RecordResult(
        record_id=record["record_id"],
        event_id=event["event_id"],
        record_type=record["record_type"],
        scope_kind=record["scope_kind"],
        scope_key=record["scope_key"],
        topic_key=record["topic_key"],
        statement=record["statement"],
        value=_load_canonical(record["value_json"], "record value"),
        evidence_uris=evidence,
        state="proposed",
        content_sha256=record["content_sha256"],
        created_at=record["created_at"],
    )


def _action_result_for_event(
    connection: sqlite3.Connection, event: sqlite3.Row
) -> ActionResult:
    projection = _validated_replay(connection, through_event_seq=event["event_seq"])
    state = projection[event["record_id"]][0]
    payload = _load_canonical(event["payload_json"], "event payload")
    replacement = payload.get("replacement_record_id") if isinstance(payload, dict) else None
    return ActionResult(
        record_id=event["record_id"],
        event_id=event["event_id"],
        action=event["action"],
        state=state,
        replacement_record_id=replacement,
        created_at=event["created_at"],
    )


def _record_view(connection: sqlite3.Connection, row: sqlite3.Row) -> RecordView:
    evidence = tuple(
        item[0]
        for item in connection.execute(
            "SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal",
            (row["record_id"],),
        )
    )
    return RecordView(
        record_id=row["record_id"],
        record_type=row["record_type"],
        scope_kind=row["scope_kind"],
        scope_key=row["scope_key"],
        topic_key=row["topic_key"],
        statement=row["statement"],
        value=_load_canonical(row["value_json"], "record value"),
        evidence_uris=evidence,
        state=row["state"],
        content_sha256=row["content_sha256"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _verify_snapshot(
    connection: sqlite3.Connection,
    *,
    _after_first_snapshot_read: Callable[[], None] | None = None,
    _validation_stage: Callable[[str], None] | None = None,
    _before_first_snapshot_read: Callable[[], None] | None = None,
) -> VerificationResult:
    if _validation_stage is not None:
        _validation_stage("invariant_validation")
    if _before_first_snapshot_read is not None:
        _before_first_snapshot_read()
    integrity = connection.execute("PRAGMA integrity_check").fetchall()
    if _after_first_snapshot_read is not None:
        _after_first_snapshot_read()
    if [row[0] for row in integrity] != ["ok"]:
        raise VerificationError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise VerificationError("foreign key check failed")
    _verify_schema(connection, _validation_stage=_validation_stage)
    if _validation_stage is not None:
        _validation_stage("invariant_validation")
    projection = _validated_replay(connection)
    actual = {
        row["record_id"]: (
            row["state"],
            row["last_event_seq"],
            row["last_event_id"],
            row["updated_at"],
        )
        for row in connection.execute("SELECT * FROM memory_current")
    }
    if actual != projection:
        raise VerificationError("current projection does not match event replay")
    from codex_memory.candidate_memory import verify_candidate_snapshot
    try:
        candidate_counts = verify_candidate_snapshot(connection)
    except Exception as exc:
        raise VerificationError("candidate verification failed") from exc
    counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "memory_records",
            "memory_events",
            "memory_evidence",
            "memory_relations",
            "memory_current",
        )
    }
    return VerificationResult(
        schema_version=3,
        records=counts["memory_records"],
        events=counts["memory_events"],
        evidence=counts["memory_evidence"],
        relations=counts["memory_relations"],
        current=counts["memory_current"],
        candidates=candidate_counts["candidates"],
        candidate_events=candidate_counts["candidate_events"],
        candidate_evidence=candidate_counts["candidate_evidence"],
        candidate_current=candidate_counts["candidate_current"],
    )


def _verify_schema(
    connection: sqlite3.Connection,
    *,
    _validation_stage: Callable[[str], None] | None = None,
) -> None:
    if _validation_stage is not None:
        _validation_stage("schema_query")
    if connection.execute("PRAGMA recursive_triggers").fetchone()[0] != 1:
        raise VerificationError("recursive trigger protection is disabled")
    try:
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error as exc:
        raise VerificationError("schema metadata is missing") from exc
    if version is None or version[0] != "3":
        raise VerificationError("unsupported ledger schema version")
    if _validation_stage is not None:
        _validation_stage("ddl_read")
    objects = connection.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL"
    ).fetchall()
    by_type = {
        object_type: {row["name"]: row["sql"] for row in objects if row["type"] == object_type}
        for object_type in ("table", "index", "trigger")
    }
    for table, expected_columns in _REQUIRED_COLUMN_SHAPES.items():
        if table not in by_type["table"]:
            raise VerificationError("required ledger table is missing")
        columns = tuple(
            (
                row["name"],
                row["type"].upper(),
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if columns != expected_columns:
            raise VerificationError("ledger table shape mismatch")
        if _normalize_schema_sql(by_type["table"][table]) != _normalize_schema_sql(
            _EXPECTED_TABLE_SQL[table]
        ):
            raise VerificationError("ledger table constraint definition mismatch")
        index_rows = connection.execute(f"PRAGMA index_list({table})").fetchall()
        unique_columns = frozenset(
            tuple(
                item["name"]
                for item in connection.execute(f"PRAGMA index_info({row['name']})")
            )
            for row in index_rows
            if row["unique"] == 1
        )
        if unique_columns != _EXPECTED_UNIQUE_COLUMNS[table]:
            raise VerificationError("ledger unique constraint mismatch")
        foreign_keys = frozenset(
            (
                row["from"],
                row["table"],
                row["to"],
                row["on_update"],
                row["on_delete"],
                row["match"],
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        if foreign_keys != _EXPECTED_FOREIGN_KEYS[table]:
            raise VerificationError("ledger foreign key definition mismatch")
    if not _REQUIRED_INDEXES <= set(by_type["index"]):
        raise VerificationError("required ledger index is missing")
    for index, expected_columns in _REQUIRED_INDEX_COLUMNS.items():
        table = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        if table is None:
            raise VerificationError("required ledger index is missing")
        expected_table = _REQUIRED_INDEX_TABLES[index]
        if table["tbl_name"] != expected_table:
            raise VerificationError("ledger index owner mismatch")
        index_definition = next(
            (
                row
                for row in connection.execute(f"PRAGMA index_list({expected_table})")
                if row["name"] == index
            ),
            None,
        )
        if (
            index_definition is None
            or index_definition["unique"] != 0
            or index_definition["origin"] != "c"
            or index_definition["partial"] != 0
        ):
            raise VerificationError("ledger index definition mismatch")
        columns = tuple(
            row["name"] for row in connection.execute(f"PRAGMA index_info({index})")
        )
        if columns != expected_columns:
            raise VerificationError("ledger index shape mismatch")
        expected_sql = _EXPECTED_CANDIDATE_INDEX_SQL.get(index)
        if expected_sql is not None and _normalize_schema_sql(
            by_type["index"][index]
        ) != _normalize_schema_sql(expected_sql):
            raise VerificationError("candidate index definition mismatch")
    for table in _APPEND_ONLY_TABLES:
        for operation in ("update", "delete"):
            name = f"{table}_reject_{operation}"
            sql = by_type["trigger"].get(name)
            if sql is None:
                raise VerificationError("append-only trigger is missing")
            expected = append_only_trigger_statement(table, operation)
            if _normalize_schema_sql(sql) != _normalize_schema_sql(expected):
                raise VerificationError("append-only trigger shape mismatch")
        guard_name = f"{table}_guard_insert_conflict"
        guard_sql = by_type["trigger"].get(guard_name)
        if guard_sql is None:
            raise VerificationError("append-only insert guard is missing")
        expected_guard = append_only_insert_guard_statement(
            table, APPEND_ONLY_UNIQUE_KEYS[table]
        )
        if _normalize_schema_sql(guard_sql) != _normalize_schema_sql(
            expected_guard
        ):
            raise VerificationError("append-only insert guard shape mismatch")


def _normalize_schema_sql(statement: str) -> str:
    normalized: list[str] = []
    quote: str | None = None
    pending_space = False
    index = 0
    while index < len(statement):
        character = statement[index]
        if quote is not None:
            normalized.append(character)
            if character == quote:
                if index + 1 < len(statement) and statement[index + 1] == quote:
                    normalized.append(statement[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if character in {"'", '"', "`"}:
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            quote = character
            normalized.append(character)
        elif character == "[":
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            quote = "]"
            normalized.append(character)
        elif _is_sqlite_tokenizer_whitespace(character):
            pending_space = True
        else:
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            normalized.append(character.lower())
        index += 1
    value = "".join(normalized)
    return value[:-1] if value.endswith(";") else value


def _validated_replay(
    connection: sqlite3.Connection, *, through_event_seq: int | None = None
) -> dict[str, tuple[str, int, str, str]]:
    records: dict[str, sqlite3.Row] = {}
    for record in connection.execute("SELECT * FROM memory_records ORDER BY record_seq"):
        value = _load_canonical(record["value_json"], "record value")
        try:
            content = validate_record_content(
                record_type=record["record_type"],
                scope_kind=record["scope_kind"],
                scope_key=record["scope_key"],
                topic_key=record["topic_key"],
                statement=record["statement"],
                value=value,
            )
        except ValidationError as exc:
            raise VerificationError("invalid record content") from exc
        if content.value_json != record["value_json"]:
            raise VerificationError("record value is not canonical")
        if content.content_sha256 != record["content_sha256"]:
            raise VerificationError("record content hash mismatch")
        records[record["record_id"]] = record

    evidence_ordinals: dict[str, int] = {}
    for evidence in connection.execute(
        "SELECT * FROM memory_evidence ORDER BY record_id, ordinal"
    ):
        record_id = evidence["record_id"]
        if record_id not in records:
            raise VerificationError("evidence references a missing record")
        expected_ordinal = evidence_ordinals.get(record_id, 0)
        if evidence["ordinal"] != expected_ordinal:
            raise VerificationError("evidence order is not contiguous")
        evidence_ordinals[record_id] = expected_ordinal + 1
        try:
            validate_evidence_uris((evidence["uri"],))
        except ValidationError as exc:
            raise VerificationError("invalid evidence URI") from exc
        if evidence["evidence_sha256"] != _hash_preimage(
            {"record_id": record_id, "uri": evidence["uri"]}
        ):
            raise VerificationError("evidence hash mismatch")

    states: dict[str, tuple[str, int, str, str]] = {}
    supersede_events: dict[str, tuple[str, str]] = {}
    query = "SELECT * FROM memory_events"
    parameters: tuple[Any, ...] = ()
    if through_event_seq is not None:
        query += " WHERE event_seq <= ?"
        parameters = (through_event_seq,)
    query += " ORDER BY event_seq"
    for event in connection.execute(query, parameters):
        record_id = event["record_id"]
        if record_id not in records:
            raise VerificationError("event references a missing record")
        payload = _load_canonical(event["payload_json"], "event payload")
        if not isinstance(payload, dict):
            raise VerificationError("event payload is not an object")
        if event["event_sha256"] != _hash_preimage(
            {"action": event["action"], "payload": payload, "record_id": record_id}
        ):
            raise VerificationError("event hash mismatch")
        prior = states.get(record_id)
        if event["action"] == "propose":
            if prior is not None or payload != {}:
                raise VerificationError("invalid proposal event")
            new_state = "proposed"
        else:
            if prior is None:
                raise VerificationError("lifecycle event precedes proposal")
            new_state = _TRANSITIONS.get((event["action"], prior[0]))
            if new_state is None:
                raise VerificationError("invalid lifecycle transition in stream")
            if event["action"] == "supersede":
                if set(payload) != {"replacement_record_id"}:
                    raise VerificationError("invalid supersede payload")
                target = payload["replacement_record_id"]
                if not isinstance(target, str) or target == record_id or target not in records:
                    raise VerificationError("invalid supersede target")
                supersede_events[event["event_id"]] = (record_id, target)
            elif payload != {}:
                raise VerificationError("unexpected lifecycle payload")
        states[record_id] = (
            new_state,
            event["event_seq"],
            event["event_id"],
            event["created_at"],
        )

    if through_event_seq is not None:
        return states
    if set(states) != set(records):
        raise VerificationError("record is missing its proposal event")

    seen_relations: set[str] = set()
    for relation in connection.execute("SELECT * FROM memory_relations ORDER BY relation_seq"):
        event_id = relation["event_id"]
        expected = supersede_events.get(event_id)
        if expected is None or event_id in seen_relations:
            raise VerificationError("relation does not match a supersede event")
        if (
            relation["relation_type"],
            relation["source_record_id"],
            relation["target_record_id"],
        ) != ("superseded_by", expected[0], expected[1]):
            raise VerificationError("supersede relation mismatch")
        if relation["relation_sha256"] != _hash_preimage(
            {
                "event_id": event_id,
                "relation_type": relation["relation_type"],
                "source_record_id": relation["source_record_id"],
                "target_record_id": relation["target_record_id"],
            }
        ):
            raise VerificationError("relation hash mismatch")
        seen_relations.add(event_id)
    if seen_relations != set(supersede_events):
        raise VerificationError("supersede event is missing its relation")
    return states


def _load_canonical(raw: str, label: str) -> Any:
    try:
        value = parse_json_value(raw)
        if canonical_json(value) != raw:
            raise VerificationError(f"{label} is not canonical")
        return value
    except (ValidationError, UnicodeError) as exc:
        raise VerificationError(f"invalid {label}") from exc


def _opaque_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
