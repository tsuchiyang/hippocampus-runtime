from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3
from urllib.parse import urlsplit

from codex_memory.blob_store import BlobStore
from codex_memory.ledger_common import RecordNotFound, ValidationError, validate_evidence_uris


_EVENT_URI = re.compile(r"codex-event://([A-Za-z0-9._~-]+)/([1-9][0-9]*)\Z", re.ASCII)
_UNRESERVED_IDENTIFIER = re.compile(r"[A-Za-z0-9._~-]+\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class EventEvidence:
    event_id: int
    session_id: str
    turn_id: str | None
    event_name: str
    tool_name: str | None
    created_at: str
    uri: str


def event_uri(session_id: str, event_id: int) -> str:
    _validate_session_id(session_id)
    if type(event_id) is not int or event_id < 1:
        raise ValueError("invalid event ID")
    return f"codex-event://{session_id}/{event_id}"


def parse_event_uri(uri: str) -> tuple[str, int]:
    if not isinstance(uri, str):
        raise ValueError("invalid event evidence URI")
    match = _EVENT_URI.fullmatch(uri)
    if match is None:
        raise ValueError("invalid event evidence URI")
    session_id, raw_event_id = match.groups()
    _validate_session_id(session_id)
    return session_id, int(raw_event_id)


def list_event_evidence(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str | None = None,
    limit: int = 50,
) -> tuple[EventEvidence, ...]:
    _validate_session_id(session_id)
    if turn_id is not None:
        _validate_session_id(turn_id)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("evidence limit must be between 1 and 100")
    query = (
        "SELECT event_id, session_id, turn_id, event_name, tool_name, created_at "
        "FROM events WHERE session_id = ?"
    )
    parameters: list[object] = [session_id]
    if turn_id is not None:
        query += " AND turn_id = ?"
        parameters.append(turn_id)
    query += " ORDER BY event_id DESC LIMIT ?"
    parameters.append(limit)
    return tuple(_evidence_from_row(row) for row in connection.execute(query, parameters))


def resolve_event_evidence(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    uri: str,
) -> EventEvidence:
    session_id, event_id = parse_event_uri(uri)
    row = connection.execute(
        "SELECT event_id, session_id, turn_id, event_name, tool_name, created_at, "
        "payload_blob_sha256 FROM events WHERE session_id = ? AND event_id = ?",
        (session_id, event_id),
    ).fetchone()
    if row is None:
        raise RecordNotFound("event evidence not found")
    blobs.read_sha256(row["payload_blob_sha256"])
    return _evidence_from_row(row)


def safe_evidence_reference(uri: str) -> str:
    validate_evidence_uris((uri,))
    if urlsplit(uri).scheme.lower() == "codex-event":
        parse_event_uri(uri)
        return uri
    scheme = urlsplit(uri).scheme.lower()
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    return f"{scheme}:sha256:{digest}"


def _validate_session_id(value: str) -> None:
    if not isinstance(value, str) or _UNRESERVED_IDENTIFIER.fullmatch(value) is None:
        raise ValueError("invalid session ID")


def _evidence_from_row(row: sqlite3.Row) -> EventEvidence:
    return EventEvidence(
        event_id=row["event_id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        event_name=row["event_name"],
        tool_name=row["tool_name"],
        created_at=row["created_at"],
        uri=event_uri(row["session_id"], row["event_id"]),
    )
