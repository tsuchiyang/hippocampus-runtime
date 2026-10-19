from __future__ import annotations

import hashlib
import sqlite3
import struct


_FINGERPRINT = "a" * 64
_PREPROCESSING = "b" * 64


def _semantic() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE semantic_documents (
            ordinal INTEGER PRIMARY KEY, document_id TEXT, record_id TEXT,
            content_sha256 TEXT, text_sha256 TEXT
        );
        CREATE TABLE semantic_dense_vectors (
            vector_seq INTEGER PRIMARY KEY, document_id TEXT, content_sha256 TEXT,
            model_fingerprint TEXT, preprocessing_fingerprint TEXT, dimension INTEGER,
            vector_blob BLOB, vector_sha256 TEXT
        );
        """
    )
    for ordinal, (record_id, vector) in enumerate((("alpha", (1.0, 0.0)), ("beta", (0.0, 1.0)), ("stale", (1.0, 1.0)))):
        document_id = f"doc-{record_id}"
        content = (record_id[0] * 64)
        blob = struct.pack("<2f", *vector)
        connection.execute(
            "INSERT INTO semantic_documents VALUES (?, ?, ?, ?, ?)",
            (ordinal, document_id, record_id, content, "c" * 64),
        )
        connection.execute(
            "INSERT INTO semantic_dense_vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ordinal + 1, document_id, content, _FINGERPRINT, _PREPROCESSING, 2, blob, hashlib.sha256(blob).hexdigest()),
        )
    return connection


def _eligible() -> dict[str, dict[str, object]]:
    return {
        "alpha": {"candidate_id": "alpha", "document_id": "doc-alpha", "content_sha256": "a" * 64, "sequence": 4},
        "beta": {"candidate_id": "beta", "document_id": "doc-beta", "content_sha256": "b" * 64, "sequence": 3},
        # A semantic row may exist but must not bypass current deterministic eligibility.
        "stale": {"candidate_id": "stale", "document_id": "doc-stale", "content_sha256": "changed" * 9 + "c", "sequence": 99},
    }


def test_dense_candidates_rank_normalized_eligible_rows_with_stable_ties() -> None:
    """Would fail if dense rows skipped identity checks or used unstable score ties."""
    from codex_memory.retrieval.dense import dense_candidates

    result = dense_candidates(
        _semantic(), (1.0, 1.0), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == ("alpha", "beta")
    assert [identifier for identifier, _ in result.raw_score_hex] == ["alpha", "beta"]
    assert float.fromhex(result.raw_score_hex[0][1]) == float.fromhex(result.raw_score_hex[1][1])


def test_dense_candidates_reject_invalid_query_or_index_without_candidates() -> None:
    """Would fail if NaN, zero-norm, or incompatible vectors influenced RRF."""
    from codex_memory.retrieval.dense import dense_candidates

    semantic = _semantic()
    invalid = dense_candidates(
        semantic, (float("nan"), 1.0), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )
    wrong_dimension = dense_candidates(
        semantic, (1.0, 0.0, 0.0), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert invalid.ids == () and invalid.reason_codes == ("dense_invalid_query",)
    assert wrong_dimension.ids == () and wrong_dimension.reason_codes == ("dense_invalid_index",)


def test_dense_candidates_reject_fingerprint_mismatch_without_using_other_rows() -> None:
    """Would fail if a changed model/preprocessing identity were silently reused."""
    from codex_memory.retrieval.dense import dense_candidates

    result = dense_candidates(
        _semantic(), (1.0, 0.0), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint="d" * 64, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == ()


def test_dense_candidates_reject_a_tampered_vector_identity() -> None:
    """Would fail if a semantic file swap could feed an unchecked vector into ranking."""
    from codex_memory.retrieval.dense import dense_candidates

    semantic = _semantic()
    semantic.execute("UPDATE semantic_dense_vectors SET vector_blob = ? WHERE document_id = 'doc-alpha'", (struct.pack("<2f", 0.0, 1.0),))
    result = dense_candidates(
        semantic, (1.0, 0.0), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == () and result.reason_codes == ("dense_invalid_index",)


def test_dense_candidates_reject_a_vector_row_with_different_content_identity() -> None:
    """Would fail if a derived vector could detach from its joined document content."""
    from codex_memory.retrieval.dense import dense_candidates

    semantic = _semantic()
    semantic.execute("UPDATE semantic_dense_vectors SET content_sha256 = ? WHERE document_id = 'doc-alpha'", ("d" * 64,))
    result = dense_candidates(
        semantic, (1.0, 0.0), _eligible(), generation_id="sem_0123456789abcdef",
        model_fingerprint=_FINGERPRINT, preprocessing_fingerprint=_PREPROCESSING, limit=2,
    )

    assert result.ids == () and result.reason_codes == ("dense_invalid_index",)
