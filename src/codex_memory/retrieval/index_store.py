from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Iterable

from codex_memory.ledger_common import canonical_json

from .contracts import CandidateChannel


_HASH = re.compile(r"\A[0-9a-f]{64}\Z")
_CHANNELS = {
    candidate.value for candidate in CandidateChannel
}
_CANDIDATE_KINDS = {"record", "temporary"}
_RUN_MODES = {"off", "shadow", "active"}


def _require_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a sha256 hex digest")
    return value


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be text")
    return value


def _require_canonical_json_text(value: str, name: str) -> str:
    _require_text(value, name)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if canonical_json(parsed) != value:
        raise ValueError(f"{name} must be canonical JSON")
    return value


def append_document(
    connection: sqlite3.Connection,
    *,
    document_id: str,
    generation_id: str,
    record_id: str,
    record_seq: int,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    value_json: str,
    search_text: str,
    state: str,
    created_at: str,
    last_memory_event_seq: int,
    evidence_count: int,
    content_sha256: str,
    document_sha256: str,
) -> str:
    _require_text(document_id, "document_id")
    _require_text(generation_id, "generation_id")
    _require_text(record_id, "record_id")
    _require_text(record_type, "record_type")
    _require_text(scope_kind, "scope_kind")
    _require_text(scope_key, "scope_key")
    _require_text(topic_key, "topic_key")
    _require_text(statement, "statement")
    _require_canonical_json_text(value_json, "value_json")
    _require_text(search_text, "search_text")
    _require_text(state, "state")
    _require_text(created_at, "created_at")
    for field_name, field_value in (
        ("record_seq", record_seq),
        ("last_memory_event_seq", last_memory_event_seq),
        ("evidence_count", evidence_count),
    ):
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    _require_hash(content_sha256, "content_sha256")
    _require_hash(document_sha256, "document_sha256")
    connection.execute(
        """INSERT INTO retrieval_documents(
            document_id, generation_id, record_id, record_seq, record_type,
            scope_kind, scope_key, topic_key, statement, value_json, search_text,
            state, created_at, last_memory_event_seq, evidence_count,
            content_sha256, document_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            document_id,
            generation_id,
            record_id,
            record_seq,
            record_type,
            scope_kind,
            scope_key,
            topic_key,
            statement,
            value_json,
            search_text,
            state,
            created_at,
            last_memory_event_seq,
            evidence_count,
            content_sha256,
            document_sha256,
        ),
    )
    return document_id


def append_document_source(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    generation_id: str,
    document_id: str,
    ordinal: int,
    safe_reference: str,
    source_sha256: str,
) -> str:
    _require_text(source_id, "source_id")
    _require_text(generation_id, "generation_id")
    _require_text(document_id, "document_id")
    _require_text(safe_reference, "safe_reference")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    _require_hash(source_sha256, "source_sha256")
    connection.execute(
        """INSERT INTO retrieval_document_sources(
            source_id, generation_id, document_id, ordinal, safe_reference, source_sha256
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        (source_id, generation_id, document_id, ordinal, safe_reference, source_sha256),
    )
    return source_id


def append_symbol(
    connection: sqlite3.Connection,
    *,
    symbol_id: str,
    generation_id: str,
    document_id: str,
    ordinal: int,
    symbol_kind: str,
    normalized_symbol: str,
    display_symbol: str,
    symbol_sha256: str,
) -> str:
    _require_text(symbol_id, "symbol_id")
    _require_text(generation_id, "generation_id")
    _require_text(document_id, "document_id")
    _require_text(symbol_kind, "symbol_kind")
    _require_text(normalized_symbol, "normalized_symbol")
    _require_text(display_symbol, "display_symbol")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    _require_hash(symbol_sha256, "symbol_sha256")
    connection.execute(
        """INSERT INTO retrieval_symbols(
            symbol_id, generation_id, document_id, ordinal, symbol_kind,
            normalized_symbol, display_symbol, symbol_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            symbol_id,
            generation_id,
            document_id,
            ordinal,
            symbol_kind,
            normalized_symbol,
            display_symbol,
            symbol_sha256,
        ),
    )
    return symbol_id


def append_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    mode: str,
    generation_id: str,
    query_sha256: str,
    canonical_highwater_json: str,
    indexed_highwater_json: str,
    status: str,
    degraded_reasons_json: str,
    trace_sha256: str,
    created_at: str,
    wall_clock_ms: int = 0,
    provider_trace_json: str = "[]",
    provider_trace_sha256: str | None = None,
) -> str:
    _require_text(run_id, "run_id")
    _require_text(mode, "mode")
    if mode not in _RUN_MODES:
        raise ValueError("mode must be a known value")
    _require_text(generation_id, "generation_id")
    _require_hash(query_sha256, "query_sha256")
    _require_canonical_json_text(canonical_highwater_json, "canonical_highwater_json")
    _require_canonical_json_text(indexed_highwater_json, "indexed_highwater_json")
    _require_text(status, "status")
    _require_canonical_json_text(degraded_reasons_json, "degraded_reasons_json")
    _require_hash(trace_sha256, "trace_sha256")
    _require_text(created_at, "created_at")
    if isinstance(wall_clock_ms, bool) or not isinstance(wall_clock_ms, int) or wall_clock_ms < 0:
        raise ValueError("wall_clock_ms must be a non-negative integer")
    _require_canonical_json_text(provider_trace_json, "provider_trace_json")
    provider_trace_sha256 = provider_trace_sha256 or __import__("hashlib").sha256(provider_trace_json.encode("utf-8")).hexdigest()
    _require_hash(provider_trace_sha256, "provider_trace_sha256")
    is_v3 = "wall_clock_ms" in {row[1] for row in connection.execute("PRAGMA table_info(retrieval_runs)")}
    if is_v3:
        connection.execute(
            """INSERT INTO retrieval_runs(
                run_id, mode, generation_id, query_sha256, canonical_highwater_json,
                indexed_highwater_json, status, wall_clock_ms, provider_trace_json,
                provider_trace_sha256, degraded_reasons_json, trace_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, mode, generation_id, query_sha256, canonical_highwater_json,
             indexed_highwater_json, status, wall_clock_ms, provider_trace_json,
             provider_trace_sha256, degraded_reasons_json, trace_sha256, created_at),
        )
    else:
        connection.execute(
        """INSERT INTO retrieval_runs(
            run_id, mode, generation_id, query_sha256, canonical_highwater_json,
            indexed_highwater_json, status, degraded_reasons_json, trace_sha256,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            mode,
            generation_id,
            query_sha256,
            canonical_highwater_json,
            indexed_highwater_json,
            status,
            degraded_reasons_json,
            trace_sha256,
            created_at,
        ),
        )
    return run_id


def append_candidate_trace(
    connection: sqlite3.Connection,
    *,
    trace_id: str,
    run_id: str,
    candidate_kind: str,
    candidate_id: str,
    channel: str,
    channel_rank: int,
    contribution_numerator: int,
    contribution_denominator: int,
    fused_rank: int | None,
    selected: bool,
    reason_codes_json: str,
    trace_sha256: str,
    raw_score_hex: str | None = None,
    rerank_score_hex: str | None = None,
) -> str:
    _require_text(trace_id, "trace_id")
    _require_text(run_id, "run_id")
    if candidate_kind not in _CANDIDATE_KINDS:
        raise ValueError("candidate_kind must be a known value")
    _require_text(candidate_id, "candidate_id")
    if channel not in _CHANNELS:
        raise ValueError("channel must be a known value")
    for field_name, field_value in (
        ("channel_rank", channel_rank),
        ("contribution_numerator", contribution_numerator),
        ("contribution_denominator", contribution_denominator),
    ):
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    if fused_rank is not None and (isinstance(fused_rank, bool) or not isinstance(fused_rank, int) or fused_rank < 0):
        raise ValueError("fused_rank must be a non-negative integer or null")
    if not isinstance(selected, bool):
        raise ValueError("selected must be boolean")
    _require_canonical_json_text(reason_codes_json, "reason_codes_json")
    _require_hash(trace_sha256, "trace_sha256")
    for value in (raw_score_hex, rerank_score_hex):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("score must be float.hex text or null")
            try:
                number = float.fromhex(value)
            except ValueError as exc:
                raise ValueError("score must be float.hex text or null") from exc
            if not __import__("math").isfinite(number) or number.hex() != value:
                raise ValueError("score must be finite")
    is_v3 = "raw_score_hex" in {row[1] for row in connection.execute("PRAGMA table_info(retrieval_candidate_traces)")}
    if is_v3:
        connection.execute(
            """INSERT INTO retrieval_candidate_traces(
                trace_id, run_id, candidate_kind, candidate_id, channel,
                channel_rank, contribution_numerator, contribution_denominator,
                fused_rank, selected, reason_codes_json, raw_score_hex,
                rerank_score_hex, trace_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trace_id, run_id, candidate_kind, candidate_id, channel, channel_rank,
             contribution_numerator, contribution_denominator, fused_rank, int(selected),
             reason_codes_json, raw_score_hex, rerank_score_hex, trace_sha256),
        )
    else:
        connection.execute(
        """INSERT INTO retrieval_candidate_traces(
            trace_id, run_id, candidate_kind, candidate_id, channel,
            channel_rank, contribution_numerator, contribution_denominator,
            fused_rank, selected, reason_codes_json, trace_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trace_id,
            run_id,
            candidate_kind,
            candidate_id,
            channel,
            channel_rank,
            contribution_numerator,
            contribution_denominator,
            fused_rank,
            int(selected),
            reason_codes_json,
            trace_sha256,
        ),
        )
    return trace_id


insert_document = append_document
insert_document_source = append_document_source
insert_symbol = append_symbol
insert_run = append_run
insert_candidate_trace = append_candidate_trace


def read_documents(connection: sqlite3.Connection, *, generation_id: str | None = None) -> tuple[sqlite3.Row, ...]:
    if generation_id is None:
        return tuple(connection.execute("SELECT * FROM retrieval_documents ORDER BY document_seq"))
    return tuple(
        connection.execute(
            "SELECT * FROM retrieval_documents WHERE generation_id = ? ORDER BY document_seq",
            (generation_id,),
        )
    )


def read_document_sources(
    connection: sqlite3.Connection, *, generation_id: str | None = None
) -> tuple[sqlite3.Row, ...]:
    if generation_id is None:
        return tuple(connection.execute("SELECT * FROM retrieval_document_sources ORDER BY source_seq"))
    return tuple(
        connection.execute(
            "SELECT * FROM retrieval_document_sources WHERE generation_id = ? ORDER BY source_seq",
            (generation_id,),
        )
    )


def read_symbols(connection: sqlite3.Connection, *, generation_id: str | None = None) -> tuple[sqlite3.Row, ...]:
    if generation_id is None:
        return tuple(connection.execute("SELECT * FROM retrieval_symbols ORDER BY symbol_seq"))
    return tuple(
        connection.execute(
            "SELECT * FROM retrieval_symbols WHERE generation_id = ? ORDER BY symbol_seq",
            (generation_id,),
        )
    )


def read_runs(connection: sqlite3.Connection, *, generation_id: str | None = None) -> tuple[sqlite3.Row, ...]:
    if generation_id is None:
        return tuple(connection.execute("SELECT * FROM retrieval_runs ORDER BY run_seq"))
    return tuple(
        connection.execute(
            "SELECT * FROM retrieval_runs WHERE generation_id = ? ORDER BY run_seq",
            (generation_id,),
        )
    )


def read_candidate_traces(
    connection: sqlite3.Connection, *, run_id: str | None = None
) -> tuple[sqlite3.Row, ...]:
    if run_id is None:
        return tuple(connection.execute("SELECT * FROM retrieval_candidate_traces ORDER BY trace_seq"))
    return tuple(
        connection.execute(
            "SELECT * FROM retrieval_candidate_traces WHERE run_id = ? ORDER BY trace_seq",
            (run_id,),
        )
    )
