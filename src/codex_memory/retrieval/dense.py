"""Bounded, read-only dense candidate scoring over immutable semantic rows."""
from __future__ import annotations

import math
import sqlite3
import struct
import hashlib
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """A ranked semantic channel plus bounded, diagnostic-only raw scores."""

    ids: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    raw_score_hex: tuple[tuple[str, str], ...] = ()


MAX_DENSE_DIMENSION = 4096
MAX_DENSE_LIMIT = 256


def dense_candidates(
    semantic: sqlite3.Connection,
    query_vector: tuple[float, ...],
    eligible: Mapping[str, Mapping[str, object]],
    *, generation_id: str,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    limit: int,
) -> ChannelResult:
    """Return only current deterministic documents ranked by normalized dot product.

    Semantic rows are treated as disposable hints: a missing, stale, malformed,
    or identity-mismatched row returns no semantic contribution and cannot make
    an ineligible record visible.
    """
    query = _normalized(query_vector)
    if query is None or not _valid_arguments(generation_id, model_fingerprint, preprocessing_fingerprint, limit):
        return ChannelResult((), ("dense_invalid_query",))
    try:
        rows = semantic.execute(
            """SELECT d.document_id, d.record_id, d.content_sha256,
                      v.content_sha256 AS vector_content_sha256,
                      v.model_fingerprint, v.preprocessing_fingerprint,
                      v.dimension, v.vector_blob, v.vector_sha256
                 FROM semantic_dense_vectors AS v
                 JOIN semantic_documents AS d ON d.document_id = v.document_id
                 WHERE v.model_fingerprint = ? AND v.preprocessing_fingerprint = ?
                 ORDER BY v.vector_seq""",
            (model_fingerprint, preprocessing_fingerprint),
        )
        scored: list[tuple[float, int, str]] = []
        for row in rows:
            if row["vector_content_sha256"] != row["content_sha256"]:
                return ChannelResult((), ("dense_invalid_index",))
            current = eligible.get(row["record_id"])
            if not _identity_matches(row, current):
                continue
            vector = _stored_vector(row, len(query))
            if vector is None:
                return ChannelResult((), ("dense_invalid_index",))
            score = math.fsum(left * right for left, right in zip(query, vector, strict=True))
            if not math.isfinite(score):
                return ChannelResult((), ("dense_invalid_index",))
            scored.append((score, _sequence(current), row["record_id"]))
    except (sqlite3.Error, TypeError, ValueError, struct.error):
        return ChannelResult((), ("dense_invalid_index",))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = scored[:limit]
    return ChannelResult(
        tuple(item[2] for item in selected),
        ("dense",) if selected else (),
        tuple((item[2], item[0].hex()) for item in selected),
    )


def _valid_arguments(generation_id: object, model_fingerprint: object, preprocessing_fingerprint: object, limit: object) -> bool:
    return (
        isinstance(generation_id, str) and generation_id.startswith("sem_")
        and isinstance(model_fingerprint, str) and len(model_fingerprint) == 64
        and isinstance(preprocessing_fingerprint, str) and len(preprocessing_fingerprint) == 64
        and type(limit) is int and 1 <= limit <= MAX_DENSE_LIMIT
    )


def _normalized(values: object) -> tuple[float, ...] | None:
    if not isinstance(values, tuple) or not values or len(values) > MAX_DENSE_DIMENSION:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        return None
    norm = math.sqrt(math.fsum(float(value) * float(value) for value in values))
    if not math.isfinite(norm) or norm == 0:
        return None
    return tuple(float(value) / norm for value in values)


def _identity_matches(row: sqlite3.Row, current: Mapping[str, object] | None) -> bool:
    return (
        isinstance(current, Mapping)
        and current.get("candidate_id") == row["record_id"]
        and current.get("document_id") == row["document_id"]
        and current.get("content_sha256") == row["content_sha256"]
    )


def _sequence(current: Mapping[str, object]) -> int:
    value = current.get("sequence")
    return value if type(value) is int and value >= 0 else -1


def _stored_vector(row: sqlite3.Row, expected_dimension: int) -> tuple[float, ...] | None:
    if type(row["dimension"]) is not int or row["dimension"] != expected_dimension:
        return None
    blob = row["vector_blob"]
    if (
        not isinstance(blob, bytes)
        or len(blob) != expected_dimension * 4
        or not isinstance(row["vector_sha256"], str)
        or hashlib.sha256(blob).hexdigest() != row["vector_sha256"]
    ):
        return None
    values = struct.unpack(f"<{expected_dimension}f", blob)
    norm = math.fsum(value * value for value in values)
    if not all(math.isfinite(value) for value in values) or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        return None
    return values
