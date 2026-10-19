from __future__ import annotations

import sqlite3

from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope


def _request(query: str) -> RetrievalRequest:
    return RetrievalRequest(query, "session-a", "turn-a", "context-a", None, RetrievalScope("repo"))


def test_bm25_quotes_bounded_normalized_terms_and_has_stable_order() -> None:
    """Would fail if user MATCH syntax is passed through or equal scores reorder."""
    from codex_memory.retrieval.bm25 import bm25_candidates, build_fts_match

    assert build_fts_match('alpha OR beta "gamma"') == '"alpha" "or" "beta" "gamma"'
    assert build_fts_match("x" * 300) == '"' + ("x" * 256) + '"'
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
      CREATE TABLE retrieval_documents(document_id TEXT, generation_id TEXT, record_id TEXT, record_seq INTEGER, scope_kind TEXT, scope_key TEXT, state TEXT);
      CREATE VIRTUAL TABLE retrieval_documents_fts USING fts5(document_id UNINDEXED, generation_id UNINDEXED, search_text);
      INSERT INTO retrieval_documents VALUES ('d1','gen','first',1,'project','repo','active');
      INSERT INTO retrieval_documents VALUES ('d2','gen','second',2,'project','repo','active');
      INSERT INTO retrieval_documents_fts VALUES ('d1','gen','alpha beta');
      INSERT INTO retrieval_documents_fts VALUES ('d2','gen','alpha beta');
    """)
    assert bm25_candidates(connection, _request('alpha beta'), generation_id="gen").ids == ("second", "first")


def test_bm25_reports_only_fts_unavailable_as_a_degraded_channel() -> None:
    """Would fail if absence of FTS leaked an SQL error or fabricated candidates."""
    from codex_memory.retrieval.bm25 import bm25_candidates

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    assert bm25_candidates(connection, _request("alpha"), generation_id="gen").reason_codes == ("fts5_unavailable",)


def test_bm25_rejects_over_budget_and_stops_before_scanning_many_tokens() -> None:
    """Would fail if normalization/findall receives an adversarial oversized query."""
    from codex_memory.retrieval.bm25 import bm25_candidates, build_fts_match

    assert build_fts_match(("word " * 10_000)) == ""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    assert bm25_candidates(connection, _request("word " * 10_000), generation_id="gen").ids == ()
