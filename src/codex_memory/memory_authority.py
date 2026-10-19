"""Shared, fail-closed Phase 1B memory authority selection."""
from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
from typing import Any, Iterable
import unicodedata
from urllib.parse import urlsplit

from codex_memory.blob_store import BlobIntegrityError, BlobStore
from codex_memory.ledger_common import RecordNotFound, VerificationError, canonical_json, parse_json_value, scope_keys_equal
from codex_memory.memory_evidence import EventEvidence, resolve_event_evidence, safe_evidence_reference
from codex_memory.memory_scope import ScopeContext
from codex_memory.memory_security import SecretRejected, reject_secrets

_QUERY_TERM = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class MemoryItem:
    identifier: str; record_type: str; label: str; scope_kind: str; scope_key: str
    topic_key: str; statement: str; value: Any; state: str; evidence: tuple[str, ...]
    bucket: int; sequence: int; certified: bool

    def line(self) -> str:
        reject_secrets(statement=self.statement, value=self.value)
        return canonical_json({"evidence": list(self.evidence), "id": self.identifier,
            "kind": self.record_type, "label": self.label,
            "scope": {"key": self.scope_key, "kind": self.scope_kind}, "state": self.state,
            "statement": self.statement, "topic": self.topic_key, "value": self.value})


def select_authorized_items(connection: sqlite3.Connection, blobs: BlobStore, *, session_id: str,
                            scope_context: ScopeContext, query: str | None) -> tuple[tuple[MemoryItem, ...], int]:
    """Return exactly the pre-render Phase 1B authority set from one snapshot."""
    permanent_rows = tuple(row for row in connection.execute("""SELECT r.*, p.state FROM memory_records AS r
        JOIN memory_current AS p ON p.record_id = r.record_id WHERE p.state = 'active' ORDER BY r.record_seq""")
        if scope_context.contains(scope_kind=row["scope_kind"], scope_key=row["scope_key"]))
    candidate_rows = tuple(row for row in connection.execute("""SELECT c.*, p.state FROM memory_candidates AS c
        JOIN memory_candidate_current AS p ON p.candidate_id = c.candidate_id
        WHERE p.state = 'temporary' AND c.session_id = ? ORDER BY c.candidate_seq""", (session_id,))
        if scope_context.contains(scope_kind=row["scope_kind"], scope_key=row["scope_key"]))
    items: list[MemoryItem] = []; unsafe_count = 0
    for row in permanent_rows:
        value = parse_json_value(row["value_json"])
        if not _content_is_safe(row["statement"], value): unsafe_count += 1; continue
        evidence = _verified_evidence(connection, blobs, _permanent_evidence_uris(connection, row["record_id"]))
        certified = False; certification_evidence: tuple[str, ...] = ()
        if row["record_type"] == "constitution": certified, certification_evidence = _constitution_certification(connection, blobs, record=row)
        bucket, label = _permanent_bucket_and_label(row["record_type"], certified=certified)
        items.append(MemoryItem(row["record_id"], row["record_type"], label, row["scope_kind"], row["scope_key"], row["topic_key"], row["statement"], value, row["state"], _deduplicate((*evidence, *certification_evidence)), bucket, row["record_seq"], certified))
    for row in candidate_rows:
        value = parse_json_value(row["value_json"])
        if not _content_is_safe(row["statement"], value): unsafe_count += 1; continue
        evidence = _verified_evidence(connection, blobs, _candidate_evidence_uris(connection, row["candidate_id"]), expected_session=row["session_id"])
        items.append(MemoryItem(row["candidate_id"], row["record_type"], "UNVERIFIED TEMPORARY", row["scope_kind"], row["scope_key"], row["topic_key"], row["statement"], value, row["state"], evidence, 2, row["candidate_seq"], False))
    terms = _query_terms(query)
    items.sort(key=lambda item: (item.bucket, -_query_score(item, terms), -item.sequence, item.identifier))
    return tuple(items), unsafe_count


def _content_is_safe(statement: str, value: Any) -> bool:
    try: reject_secrets(statement=statement, value=value)
    except SecretRejected: return False
    return True


def _permanent_bucket_and_label(record_type: str, *, certified: bool) -> tuple[int, str]:
    if record_type == "constitution": return (0, "CERTIFIED CONSTITUTION") if certified else (1, "UNVERIFIED CONSTITUTION")
    if record_type == "decision": return 3, "ACTIVE MEMORY"
    if record_type == "failure": return 4, "ACTIVE MEMORY"
    if record_type == "rejected_approach": return 5, "ACTIVE MEMORY"
    if record_type == "assumption": return 6, "UNVERIFIED ASSUMPTION"
    raise VerificationError("unsupported permanent memory record type")


def _permanent_evidence_uris(connection: sqlite3.Connection, record_id: str) -> tuple[str, ...]:
    return tuple(row["uri"] for row in connection.execute("SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal", (record_id,)))


def _candidate_evidence_uris(connection: sqlite3.Connection, candidate_id: str) -> tuple[str, ...]:
    evidence = [row["uri"] for row in connection.execute("SELECT uri FROM memory_candidate_evidence WHERE candidate_id = ? ORDER BY ordinal", (candidate_id,))]
    for row in connection.execute("SELECT action, payload_json FROM memory_candidate_events WHERE candidate_id = ? AND action <> 'capture' ORDER BY event_seq", (candidate_id,)):
        payload = parse_json_value(row["payload_json"])
        if not isinstance(payload, dict) or not isinstance(payload.get("evidence_uris"), list): raise VerificationError("candidate evidence payload is invalid")
        evidence.extend(payload["evidence_uris"])
    return tuple(evidence)


def _verified_evidence(connection: sqlite3.Connection, blobs: BlobStore, uris: Iterable[str], *, expected_session: str | None = None) -> tuple[str, ...]:
    references: list[str] = []
    for uri in uris:
        try:
            if urlsplit(uri).scheme.lower() == "codex-event":
                event = resolve_event_evidence(connection, blobs, uri)
                if expected_session is not None and event.session_id != expected_session: raise VerificationError("candidate evidence belongs to another session")
                reference = event.uri
            else: reference = safe_evidence_reference(uri)
        except VerificationError: raise
        except (BlobIntegrityError, OSError, RecordNotFound, ValueError) as exc: raise VerificationError("memory evidence verification failed") from exc
        references.append(reference)
    return _deduplicate(references)


def _verified_candidate_events(connection: sqlite3.Connection, blobs: BlobStore, candidate: sqlite3.Row) -> tuple[tuple[str, ...], tuple[EventEvidence, ...]]:
    references: list[str] = []; events: list[EventEvidence] = []
    for uri in _candidate_evidence_uris(connection, candidate["candidate_id"]):
        try:
            if urlsplit(uri).scheme.lower() == "codex-event":
                event = resolve_event_evidence(connection, blobs, uri)
                if event.session_id != candidate["session_id"]: raise VerificationError("candidate evidence belongs to another session")
                references.append(event.uri); events.append(event)
            else: references.append(safe_evidence_reference(uri))
        except VerificationError: raise
        except (BlobIntegrityError, OSError, RecordNotFound, ValueError) as exc: raise VerificationError("memory evidence verification failed") from exc
    return _deduplicate(references), tuple(events)


def constitution_certification(connection: sqlite3.Connection, blobs: BlobStore, *, record: sqlite3.Row) -> tuple[bool, tuple[str, ...]]:
    candidates = connection.execute("""SELECT c.* FROM memory_candidates AS c
        JOIN memory_candidate_current AS p ON p.candidate_id = c.candidate_id
        JOIN memory_candidate_events AS e ON e.candidate_id = c.candidate_id AND e.action = 'promote'
        WHERE p.state = 'promoted' AND p.permanent_record_id = ? ORDER BY c.candidate_seq DESC, c.candidate_id ASC""", (record["record_id"],)).fetchall()
    for candidate in candidates:
        if not _same_constitution_identity(candidate, record): continue
        references, events = _verified_candidate_events(connection, blobs, candidate)
        if any(event.event_name == "UserPromptSubmit" for event in events): return True, references
    return False, ()


_constitution_certification = constitution_certification

def _same_constitution_identity(candidate: sqlite3.Row, record: sqlite3.Row) -> bool:
    return candidate["record_type"] == "constitution" and record["record_type"] == "constitution" and candidate["content_sha256"] == record["content_sha256"] and candidate["scope_kind"] == record["scope_kind"] and scope_keys_equal(candidate["scope_key"], record["scope_key"]) and candidate["topic_key"] == record["topic_key"] and candidate["statement"] == record["statement"] and candidate["value_json"] == record["value_json"]

def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []; seen: set[str] = set()
    for value in values:
        if value not in seen: seen.add(value); result.append(value)
    return tuple(result)

def _query_terms(query: str | None) -> tuple[str, ...]:
    return () if not query else _deduplicate(_QUERY_TERM.findall(unicodedata.normalize("NFKC", query).casefold()))

def _query_score(item: MemoryItem, terms: tuple[str, ...]) -> int:
    haystack = unicodedata.normalize("NFKC", f"{item.topic_key} {item.statement}").casefold()
    return sum(term in haystack for term in terms)
