"""Canonical sequence based recent/state channel."""
from __future__ import annotations
import sqlite3
from .contracts import RetrievalRequest
from .exact import ChannelResult, _scope, _visible

def recent_candidates(connection: sqlite3.Connection, request: RetrievalRequest, *, generation_id: str, limit: int = 32, excluded_ids: frozenset[str] = frozenset()) -> ChannelResult:
    scope = _scope(request)
    rows = [row for row in connection.execute("SELECT * FROM retrieval_documents WHERE generation_id = ?", (generation_id,)) if _visible(row, scope) and row["record_id"] not in excluded_ids]
    rows.sort(key=lambda row: (-row["last_memory_event_seq"], -row["record_seq"], row["record_id"]))
    return ChannelResult(tuple(row["record_id"] for row in rows[:limit]), ("recent_state",))
