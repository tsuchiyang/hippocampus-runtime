from __future__ import annotations

import sqlite3

from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope


def _request(query: str = "") -> RetrievalRequest:
    return RetrievalRequest(query, "session-a", "turn-a", "context-a", None, RetrievalScope(r"C:\\Work\\Repo"))


def _cache() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE retrieval_documents (
          document_id TEXT, generation_id TEXT, record_id TEXT, record_seq INTEGER,
          scope_kind TEXT, scope_key TEXT, topic_key TEXT, statement TEXT,
          value_json TEXT, search_text TEXT, state TEXT, created_at TEXT,
          last_memory_event_seq INTEGER, content_sha256 TEXT
        );
        CREATE TABLE retrieval_symbols (
          document_id TEXT, generation_id TEXT, ordinal INTEGER, normalized_symbol TEXT
        );
    """)
    return connection


def _document(connection: sqlite3.Connection, *, identifier: str, seq: int, topic: str,
              statement: str, state: str = "active", scope: str = r"c:\\work\\repo",
              content: str | None = None) -> None:
    connection.execute(
        "INSERT INTO retrieval_documents VALUES (?, 'gen', ?, ?, 'project', ?, ?, ?, '{}', ?, ?, '2026-01-01T00:00:00Z', ?, ?)",
        ("doc-" + identifier, identifier, seq, scope, topic, statement, statement, state, seq, content or ("a" * 64)),
    )


def test_exact_channel_matches_all_literal_identity_forms_and_filters_before_ranking() -> None:
    """Would fail if an identity field, scope/state filter, or stable rank regressed."""
    from codex_memory.retrieval.exact import exact_candidates

    connection = _cache()
    _document(connection, identifier="rec-full", seq=1, topic="topic.one", statement="checkpoint cp-9 error 'disk full' --safe")
    _document(connection, identifier="rec-hash", seq=2, topic="topic.two", statement="event codex-event://session-a/7 path C:/Work/Repo/a.py --flag", content="b" * 64)
    _document(connection, identifier="rec-symbol", seq=3, topic="topic.Three", statement="run pkg.module.func --flag")
    _document(connection, identifier="closed", seq=99, topic="topic.one", statement="rec-full", state="closed")
    _document(connection, identifier="other", seq=100, topic="topic.one", statement="rec-full", scope=r"C:\\Other")
    connection.execute("INSERT INTO retrieval_symbols VALUES ('doc-rec-symbol', 'gen', 0, 'pkg.module.func')")
    assert exact_candidates(connection, _request("rec-full cp-9 codex-event://session-a/7 " + "b" * 64 + " topic.one C:/Work/Repo/a.py --flag pkg.module.func 'disk full'"), generation_id="gen").ids == (
        "rec-hash", "rec-full", "rec-symbol",
    )


def test_symbol_channel_normalizes_dotted_symbols_and_enforces_a_bounded_query() -> None:
    """Would fail if symbol lookup accepted unbounded/raw input or missed case folding."""
    from codex_memory.retrieval.exact import symbol_candidates

    connection = _cache()
    _document(connection, identifier="symbol", seq=3, topic="topic", statement="symbol")
    connection.execute("INSERT INTO retrieval_symbols VALUES ('doc-symbol', 'gen', 0, 'pkg.module.func')")
    assert symbol_candidates(connection, _request("PKG.MODULE.FUNC"), generation_id="gen").ids == ("symbol",)
    assert symbol_candidates(connection, _request("x" * 300), generation_id="gen").ids == ()


def test_exact_excludes_stale_records_before_applying_channel_limit() -> None:
    """Would fail if stale top hits consumed the limited cached channel result."""
    from codex_memory.retrieval.exact import exact_candidates

    connection = _cache()
    _document(connection, identifier="changed", seq=2, topic="needle", statement="needle")
    _document(connection, identifier="current", seq=1, topic="needle", statement="needle")

    assert exact_candidates(connection, _request("needle"), generation_id="gen", limit=1, excluded_ids=frozenset({"changed"})).ids == ("current",)
