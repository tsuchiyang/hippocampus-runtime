from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codex_memory.retrieval.cache_database import (
    CacheCorruptError,
    append_generation,
    connect_cache_database,
    deterministic_generation_id,
    initialize_schema_v2,
    verify_cache_database,
)
from codex_memory.retrieval.contracts import IndexHighwater
from codex_memory.retrieval import index_store


def _generation(connection: sqlite3.Connection, record_seq: int = 1) -> str:
    highwater = IndexHighwater(record_seq=record_seq)
    generation_id = deterministic_generation_id(highwater, "a" * 64)
    append_generation(
        connection,
        generation_id=generation_id,
        canonical_highwater=highwater,
        indexed_highwater=highwater,
        canonical_content_fingerprint="a" * 64,
        provider_fingerprint="unconfigured:p2.1",
        model_fingerprint="unconfigured:p2.1",
        created_at="2026-01-02T03:04:05.000000Z",
    )
    return generation_id


def _cache(path: Path, *, fts5_available: bool) -> sqlite3.Connection:
    connection = connect_cache_database(path)
    connection.execute("PRAGMA foreign_keys=ON")
    initialize_schema_v2(connection)
    return connection


@pytest.mark.parametrize("fts5_available", [False, True])
def test_v2_schema_supports_append_only_projection_rows_and_reads_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fts5_available: bool
) -> None:
    monkeypatch.setattr(
        "codex_memory.retrieval.cache_database._supports_fts5",
        lambda _connection: fts5_available,
        raising=False,
    )
    connection = _cache(tmp_path / f"retrieval-{fts5_available}.sqlite3", fts5_available=fts5_available)
    try:
        generation_id = _generation(connection)
        document_id = index_store.append_document(
            connection,
            document_id="doc-1",
            generation_id=generation_id,
            record_id="record-1",
            record_seq=1,
            record_type="decision",
            scope_kind="project",
            scope_key="C:/Repo",
            topic_key="topic.one",
            statement="Statement one.",
            value_json='{"value":1}',
            search_text="decision project topic.one statement one value 1",
            state="active",
            created_at="2026-01-02T03:04:05.000000Z",
            last_memory_event_seq=12,
            evidence_count=1,
            content_sha256="b" * 64,
            document_sha256="c" * 64,
        )
        index_store.append_document_source(
            connection,
            source_id="source-1",
            generation_id=generation_id,
            document_id=document_id,
            ordinal=0,
            safe_reference="https://example.test/evidence",
            source_sha256="d" * 64,
        )
        index_store.append_symbol(
            connection,
            symbol_id="symbol-1",
            generation_id=generation_id,
            document_id=document_id,
            ordinal=0,
            symbol_kind="topic_key",
            normalized_symbol="topic.one",
            display_symbol="topic.one",
            symbol_sha256="e" * 64,
        )
        index_store.append_run(
            connection,
            run_id="run-1",
            mode="shadow",
            generation_id=generation_id,
            query_sha256="f" * 64,
            canonical_highwater_json='{"candidate_event_seq":0,"candidate_evidence_seq":0,"candidate_seq":0,"evidence_seq":0,"memory_event_seq":0,"record_seq":1,"relation_seq":0}',
            indexed_highwater_json='{"candidate_event_seq":0,"candidate_evidence_seq":0,"candidate_seq":0,"evidence_seq":0,"memory_event_seq":0,"record_seq":1,"relation_seq":0}',
            status="current",
            degraded_reasons_json="[]",
            trace_sha256="1" * 64,
            created_at="2026-01-02T03:04:05.000000Z",
        )
        index_store.append_candidate_trace(
            connection,
            trace_id="trace-1",
            run_id="run-1",
            candidate_kind="record",
            candidate_id="record-1",
            channel="exact",
            channel_rank=1,
            contribution_numerator=8,
            contribution_denominator=61,
            fused_rank=1,
            selected=True,
            reason_codes_json='["exact_match"]',
            trace_sha256="2" * 64,
        )

        documents = index_store.read_documents(connection)
        sources = index_store.read_document_sources(connection)
        symbols = index_store.read_symbols(connection)
        runs = index_store.read_runs(connection)
        traces = index_store.read_candidate_traces(connection)

        assert documents[0]["document_id"] == document_id
        assert sources[0]["safe_reference"] == "https://example.test/evidence"
        assert symbols[0]["normalized_symbol"] == "topic.one"
        assert runs[0]["run_id"] == "run-1"
        assert traces[0]["candidate_id"] == "record-1"
        assert verify_cache_database(connection) is None
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE retrieval_documents SET statement = 'mutated' WHERE document_id = ?",
                (document_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM retrieval_runs WHERE run_id = 'run-1'")
    finally:
        connection.close()


@pytest.mark.parametrize("fts5_available", [False, True])
def test_v2_schema_detects_count_and_ddl_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fts5_available: bool
) -> None:
    monkeypatch.setattr(
        "codex_memory.retrieval.cache_database._supports_fts5",
        lambda _connection: fts5_available,
        raising=False,
    )
    path = tmp_path / f"mutations-{fts5_available}.sqlite3"
    connection = _cache(path, fts5_available=fts5_available)
    try:
        generation_id = _generation(connection)
        index_store.append_document(
            connection,
            document_id="doc-1",
            generation_id=generation_id,
            record_id="record-1",
            record_seq=1,
            record_type="decision",
            scope_kind="project",
            scope_key="C:/Repo",
            topic_key="topic.one",
            statement="Statement one.",
            value_json='{"value":1}',
            search_text="decision project topic.one statement one value 1",
            state="active",
            created_at="2026-01-02T03:04:05.000000Z",
            last_memory_event_seq=12,
            evidence_count=2,
            content_sha256="b" * 64,
            document_sha256="c" * 64,
        )
        index_store.append_document_source(
            connection,
            source_id="source-1",
            generation_id=generation_id,
            document_id="doc-1",
            ordinal=0,
            safe_reference="https://example.test/evidence",
            source_sha256="d" * 64,
        )
        with pytest.raises(CacheCorruptError, match="count"):
            verify_cache_database(connection)

        connection.execute("PRAGMA writable_schema=ON")
        connection.execute(
            """UPDATE sqlite_master
            SET sql = replace(sql, 'CREATE TABLE retrieval_documents', 'CREATE\u00A0TABLE retrieval_documents')
            WHERE type = 'table' AND name = 'retrieval_documents'"""
        )
        connection.execute("PRAGMA writable_schema=OFF")

        with pytest.raises(CacheCorruptError, match="schema"):
            verify_cache_database(connection)
    finally:
        connection.close()
