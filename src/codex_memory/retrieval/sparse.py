"""Bounded, deterministic sparse candidate scoring over immutable postings."""
from __future__ import annotations

import math
import sqlite3
import hashlib
from typing import Mapping

from codex_memory.ledger_common import canonical_json
from .dense import ChannelResult


MAX_QUERY_TERMS = 256
MAX_TERM_BYTES = 256
MAX_SPARSE_LIMIT = 256


def sparse_candidates(
    semantic: sqlite3.Connection,
    query_weights: tuple[tuple[str, float], ...],
    eligible: Mapping[str, Mapping[str, object]],
    *, generation_id: str,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    limit: int,
) -> ChannelResult:
    """Return current eligible documents by exact, sorted-term sparse products."""
    weights = _valid_weights(query_weights)
    if weights is None or not _valid_arguments(generation_id, model_fingerprint, preprocessing_fingerprint, limit):
        return ChannelResult((), ("sparse_invalid_query",))
    if not weights:
        return ChannelResult(())
    try:
        terms = tuple(sorted(weights))
        placeholders = ",".join("?" for _ in terms)
        rows = semantic.execute(
            f"""SELECT d.document_id, d.record_id, d.content_sha256,
                       p.content_sha256 AS posting_content_sha256,
                       p.model_fingerprint, p.preprocessing_fingerprint,
                       p.term, p.weight_hex, p.posting_sha256
                  FROM semantic_sparse_postings AS p
                  JOIN semantic_documents AS d ON d.document_id = p.document_id
                 WHERE p.model_fingerprint = ? AND p.preprocessing_fingerprint = ?
                   AND p.term IN ({placeholders})
                 ORDER BY d.record_id, p.term, p.posting_seq""",
            (model_fingerprint, preprocessing_fingerprint, *terms),
        )
        by_record: dict[str, tuple[Mapping[str, object], dict[str, float]]] = {}
        for row in rows:
            if row["posting_content_sha256"] != row["content_sha256"]:
                return ChannelResult((), ("sparse_invalid_index",))
            current = eligible.get(row["record_id"])
            if not _identity_matches(row, current):
                continue
            try:
                stored = float.fromhex(row["weight_hex"])
            except (TypeError, ValueError):
                return ChannelResult((), ("sparse_invalid_index",))
            if not math.isfinite(stored) or stored < 0:
                return ChannelResult((), ("sparse_invalid_index",))
            expected_hash = hashlib.sha256(canonical_json({
                "content_sha256": row["posting_content_sha256"],
                "document_id": row["document_id"],
                "model_fingerprint": row["model_fingerprint"],
                "preprocessing_fingerprint": row["preprocessing_fingerprint"],
                "term": row["term"],
                "weight_hex": row["weight_hex"],
            }).encode("utf-8")).hexdigest()
            if row["posting_sha256"] != expected_hash:
                return ChannelResult((), ("sparse_invalid_index",))
            _, posting = by_record.setdefault(row["record_id"], (current, {}))
            if row["term"] in posting:
                return ChannelResult((), ("sparse_invalid_index",))
            posting[row["term"]] = stored
        scored = []
        for record_id, (current, postings) in by_record.items():
            score = math.fsum(weights[term] * postings[term] for term in sorted(postings))
            if not math.isfinite(score):
                return ChannelResult((), ("sparse_invalid_index",))
            if score > 0:
                scored.append((score, _sequence(current), record_id))
    except (sqlite3.Error, TypeError, ValueError):
        return ChannelResult((), ("sparse_invalid_index",))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = scored[:limit]
    return ChannelResult(
        tuple(item[2] for item in selected),
        ("sparse",) if selected else (),
        tuple((item[2], item[0].hex()) for item in selected),
    )


def _valid_arguments(generation_id: object, model_fingerprint: object, preprocessing_fingerprint: object, limit: object) -> bool:
    return (
        isinstance(generation_id, str) and generation_id.startswith("sem_")
        and isinstance(model_fingerprint, str) and len(model_fingerprint) == 64
        and isinstance(preprocessing_fingerprint, str) and len(preprocessing_fingerprint) == 64
        and type(limit) is int and 1 <= limit <= MAX_SPARSE_LIMIT
    )


def _valid_weights(value: object) -> dict[str, float] | None:
    if not isinstance(value, tuple) or len(value) > MAX_QUERY_TERMS:
        return None
    values: dict[str, float] = {}
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            return None
        term, weight = item
        if (
            not isinstance(term, str) or not term or len(term.encode("utf-8")) > MAX_TERM_BYTES
            or term in values or isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0
        ):
            return None
        values[term] = float(weight)
    return values


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
