from __future__ import annotations

import hashlib
import sqlite3


_FINGERPRINT = "a" * 64
_PREPROCESSING = "b" * 64


def _posting(document_id: str, content: str, term: str, weight: float) -> tuple[str, str, str, str, str, str, str]:
    weight_hex = weight.hex()
    payload = {
        "content_sha256": content, "document_id": document_id,
        "model_fingerprint": _FINGERPRINT, "preprocessing_fingerprint": _PREPROCESSING,
        "term": term, "weight_hex": weight_hex,
    }
    from codex_memory.ledger_common import canonical_json
    return document_id, content, _FINGERPRINT, _PREPROCESSING, term, weight_hex, hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _semantic() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
      CREATE TABLE semantic_documents (ordinal INTEGER PRIMARY KEY, document_id TEXT, record_id TEXT, content_sha256 TEXT, text_sha256 TEXT);
      CREATE TABLE semantic_sparse_postings (posting_seq INTEGER PRIMARY KEY, document_id TEXT, content_sha256 TEXT, model_fingerprint TEXT, preprocessing_fingerprint TEXT, term TEXT, weight_hex TEXT, posting_sha256 TEXT);
    """)
    for ordinal, (record_id, weights) in enumerate((("alpha", (("x", 2.0), ("y", 1.0))), ("beta", (("x", 1.0),)), ("closed", (("x", 9.0),)))):
        document_id, content = f"doc-{record_id}", record_id[0] * 64
        connection.execute("INSERT INTO semantic_documents VALUES (?, ?, ?, ?, ?)", (ordinal, document_id, record_id, content, "c" * 64))
        for term, weight in weights:
            connection.execute("INSERT INTO semantic_sparse_postings(document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, term, weight_hex, posting_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)", _posting(document_id, content, term, weight))
    return connection


def _eligible() -> dict[str, dict[str, object]]:
    return {
        "alpha": {"candidate_id": "alpha", "document_id": "doc-alpha", "content_sha256": "a" * 64, "sequence": 4},
        "beta": {"candidate_id": "beta", "document_id": "doc-beta", "content_sha256": "b" * 64, "sequence": 3},
    }


def test_sparse_candidates_sum_sorted_terms_and_exclude_noneligible_rows() -> None:
    """Would fail if sparse candidates bypassed deterministic current eligibility."""
    from codex_memory.retrieval.sparse import sparse_candidates

    result = sparse_candidates(
        _semantic(), (("y", 4.0), ("x", 1.0)), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == ("alpha", "beta")
    assert result.raw_score_hex == (("alpha", 6.0.hex()), ("beta", 1.0.hex()))


def test_sparse_candidates_reject_negative_duplicate_and_overflow_query_weights() -> None:
    """Would fail if malformed sparse input could cause unbounded or ambiguous scoring."""
    from codex_memory.retrieval.sparse import MAX_QUERY_TERMS, sparse_candidates

    common = dict(semantic=_semantic(), eligible=_eligible(), generation_id="sem_0123456789abcdef", model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2)
    negative = sparse_candidates(query_weights=(("x", -1.0),), **common)
    duplicate = sparse_candidates(query_weights=(("x", 1.0), ("x", 2.0)), **common)
    overflow = sparse_candidates(query_weights=tuple((str(index), 1.0) for index in range(MAX_QUERY_TERMS + 1)), **common)

    assert negative.reason_codes == duplicate.reason_codes == overflow.reason_codes == ("sparse_invalid_query",)


def test_sparse_candidates_ignore_fingerprint_mismatched_postings() -> None:
    """Would fail if sparse rows from a different model became eligible."""
    from codex_memory.retrieval.sparse import sparse_candidates

    result = sparse_candidates(
        _semantic(), (("x", 1.0),), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint="d" * 64, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == ()


def test_sparse_candidates_reject_tampered_posting_identity() -> None:
    """Would fail if unchecked derived postings could alter semantic rank."""
    from codex_memory.retrieval.sparse import sparse_candidates

    semantic = _semantic()
    semantic.execute("UPDATE semantic_sparse_postings SET weight_hex = ? WHERE document_id = 'doc-alpha' AND term = 'x'", (9.0.hex(),))
    result = sparse_candidates(
        semantic, (("x", 1.0),), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == () and result.reason_codes == ("sparse_invalid_index",)


def test_sparse_candidates_reject_a_posting_row_with_different_content_identity() -> None:
    """Would fail if a posting hash could validate against the document rather than itself."""
    from codex_memory.retrieval.sparse import sparse_candidates

    semantic = _semantic()
    foreign = "d" * 64
    semantic.execute(
        "UPDATE semantic_sparse_postings SET content_sha256 = ?, posting_sha256 = ? WHERE document_id = 'doc-alpha' AND term = 'x'",
        (foreign, _posting("doc-alpha", foreign, "x", 2.0)[-1]),
    )
    result = sparse_candidates(
        semantic, (("x", 1.0),), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == () and result.reason_codes == ("sparse_invalid_index",)
