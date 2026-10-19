from __future__ import annotations

import os
import sqlite3

from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope


def _request() -> RetrievalRequest:
    return RetrievalRequest(None, "session-a", "turn-a", "context-a", None, RetrievalScope(r"C:\\Work\\Repo", (r"C:\\Work\\Shared",)))


def test_recent_orders_by_canonical_sequence_without_wall_clock_and_filters_scope_state() -> None:
    """Would fail if recent uses current time or lets inactive/cross-scope rows rank."""
    from codex_memory.retrieval.recent import recent_candidates

    matching_project = r"c:\\work\\repo" if os.name == "nt" else r"C:\\Work\\Repo"
    matching_workspace = r"C:\\WORK\\SHARED" if os.name == "nt" else r"C:\\Work\\Shared"
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE retrieval_documents (generation_id TEXT, record_id TEXT, record_seq INTEGER, scope_kind TEXT, scope_key TEXT, state TEXT, last_memory_event_seq INTEGER, created_at TEXT)")
    connection.executemany("INSERT INTO retrieval_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        ("gen", "newest", 8, "project", matching_project, "active", 5, "1900-01-01T00:00:00Z"),
        ("gen", "tie-b", 7, "workspace", matching_workspace, "active", 5, "3000-01-01T00:00:00Z"),
        ("gen", "tie-a", 6, "project", r"C:\\Work\\Repo", "active", 5, "2026-01-01T00:00:00Z"),
        ("gen", "closed", 99, "project", r"C:\\Work\\Repo", "closed", 99, "2099-01-01T00:00:00Z"),
        ("gen", "other", 98, "project", r"C:\\Other", "active", 98, "2099-01-01T00:00:00Z"),
    ])
    assert recent_candidates(connection, _request(), generation_id="gen").ids == ("newest", "tie-b", "tie-a")
