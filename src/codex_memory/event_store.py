from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from codex_memory.blob_store import BlobStore
from codex_memory.hook_input import (
    HookEnvelope, HookEvent, HookInputError, MAX_PROMPT_BYTES, validate_hook_payload_limits,
)


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: int
    event_key: str
    session_id: str
    turn_id: str | None
    event: HookEvent
    tool_use_id: str | None
    tool_name: str | None
    created_at: str
    payload_blob_sha256: str


def _canonical_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _event_key(envelope: HookEnvelope, canonical_payload: bytes) -> str:
    parts = (
        canonical_payload,
        envelope.event.value.encode("utf-8"),
        envelope.session_id.encode("utf-8"),
        (envelope.turn_id or "").encode("utf-8"),
        (envelope.tool_use_id or "").encode("utf-8"),
    )
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big"))
        digest.update(part)
    return f"sha256:{digest.hexdigest()}"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class EventStore:
    def __init__(self, connection: sqlite3.Connection, blobs: BlobStore) -> None:
        self._connection = connection
        self._blobs = blobs

    def append(self, envelope: HookEnvelope) -> EventRecord:
        validate_hook_payload_limits(envelope.raw)
        if envelope.prompt is not None and len(envelope.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise HookInputError("prompt exceeds byte limit")
        canonical_payload = _canonical_payload(envelope.raw)
        payload_ref = self._blobs.put_bytes(canonical_payload)
        event_key = _event_key(envelope, canonical_payload)
        created_at = _utc_timestamp()

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_key, session_id, turn_id, event_name, tool_use_id, tool_name,
                    prompt_text, created_at, payload_blob_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_key,
                    envelope.session_id,
                    envelope.turn_id,
                    envelope.event.value,
                    envelope.tool_use_id,
                    envelope.tool_name,
                    envelope.prompt,
                    created_at,
                    payload_ref.sha256,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM events WHERE event_key = ?", (event_key,)
            ).fetchone()
            if row is None:
                raise RuntimeError("event row was not stored")
            self._connection.commit()
        except BaseException:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise
        return self._record_from_row(row)

    def highwater(self, session_id: str) -> int | None:
        row = self._connection.execute(
            "SELECT MAX(event_id) FROM events WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0]

    def latest_user_prompt(self, session_id: str) -> str | None:
        row = self._connection.execute(
            """
            SELECT prompt_text FROM events
            WHERE session_id = ? AND event_name = ? AND prompt_text IS NOT NULL
            ORDER BY event_id DESC
            LIMIT 1
            """,
            (session_id, HookEvent.USER_PROMPT_SUBMIT.value),
        ).fetchone()
        return None if row is None else row["prompt_text"]

    def recent_tool_names(self, session_id: str, limit: int) -> list[str]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        rows = self._connection.execute(
            """
            SELECT tool_name FROM events
            WHERE session_id = ? AND event_name = ? AND tool_name IS NOT NULL
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (session_id, HookEvent.POST_TOOL_USE.value, limit),
        ).fetchall()
        return [row["tool_name"] for row in rows]

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            event_key=row["event_key"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            event=HookEvent(row["event_name"]),
            tool_use_id=row["tool_use_id"],
            tool_name=row["tool_name"],
            created_at=row["created_at"],
            payload_blob_sha256=row["payload_blob_sha256"],
        )
