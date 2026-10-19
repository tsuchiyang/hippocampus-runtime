from __future__ import annotations

import importlib
from pathlib import Path
import sqlite3

import pytest

from codex_memory import database, structured_ledger
from codex_memory.paths import RuntimePaths
from codex_memory.retrieval.contracts import RetrievalStatus
from codex_memory.retrieval.paths import RetrievalPaths


def _canonical_database(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    repo = tmp_path / "index repository"
    repo.mkdir()
    runtime = RuntimePaths.for_repo(repo)
    runtime.ensure()
    connection = database.connect_database(runtime.db_path)
    database.migrate_database(connection)
    return repo, connection


@pytest.mark.parametrize("fts5_available", [False, True])
def test_rebuild_index_projects_generation_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fts5_available: bool
) -> None:
    indexer = importlib.import_module("codex_memory.retrieval.indexer")
    repo, connection = _canonical_database(tmp_path)
    record = structured_ledger.propose_record(
        connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(repo),
        topic_key="index.full",
        statement="Build the complete retrieval generation.",
        idempotency_key="index-full",
    )
    connection.close()
    monkeypatch.setattr(
        "codex_memory.retrieval.cache_database._supports_fts5",
        lambda _connection: fts5_available,
    )
    paths = RetrievalPaths.for_repo(repo)

    result = indexer.rebuild_index(paths)

    assert result.status is RetrievalStatus.CURRENT
    with sqlite3.connect(paths.database_path) as cache:
        documents = cache.execute(
            "SELECT record_id, state FROM retrieval_documents ORDER BY document_seq"
        ).fetchall()
        capability = cache.execute(
            "SELECT fts5_state, projected_document_count, omitted_unsafe_count "
            "FROM retrieval_generation_capabilities"
        ).fetchone()
        events = [
            row[0]
            for row in cache.execute(
                "SELECT action FROM retrieval_generation_events ORDER BY event_seq"
            )
        ]
        fts = cache.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
        ).fetchone()
        fts_rows = (
            cache.execute("SELECT document_id, generation_id, search_text FROM retrieval_documents_fts").fetchall()
            if fts5_available
            else []
        )
    assert documents == [(record.record_id, "proposed")]
    assert capability == ("available" if fts5_available else "unavailable", 1, 0)
    assert events[-2:] == ["validate", "activate"]
    assert bool(fts) is fts5_available
    assert len(fts_rows) == int(fts5_available)


def _logical_projection(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(
                f"SELECT {columns} FROM {table} ORDER BY {order}"
            ).fetchall()
            for table, columns, order in (
                ("retrieval_documents", "record_id, record_seq, record_type, scope_kind, scope_key, topic_key, statement, value_json, search_text, state, last_memory_event_seq, evidence_count, content_sha256, document_sha256", "record_id"),
                ("retrieval_document_sources", "document_id, ordinal, safe_reference, source_sha256", "document_id, ordinal"),
                ("retrieval_symbols", "document_id, ordinal, symbol_kind, normalized_symbol, display_symbol, symbol_sha256", "document_id, ordinal"),
                ("retrieval_documents_fts", "document_id, search_text", "document_id"),
            )
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
        }


def test_catch_up_index_creates_an_immutable_generation_equivalent_to_full_rebuild(tmp_path: Path) -> None:
    """Would fail if changed lifecycle/relation rows were updated in place or omitted."""
    indexer = importlib.import_module("codex_memory.retrieval.indexer")
    repo, connection = _canonical_database(tmp_path)
    first = structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="index.catch.first", statement="First indexed record.", idempotency_key="catch-first",
    )
    second = structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="index.catch.second", statement="Second indexed record.", idempotency_key="catch-second",
    )
    paths = RetrievalPaths.for_repo(repo)
    assert indexer.rebuild_index(paths).status is RetrievalStatus.CURRENT
    with sqlite3.connect(paths.database_path) as cache:
        initial_generation = cache.execute("SELECT active_generation_id FROM retrieval_state").fetchone()[0]
    cache.close()
    structured_ledger.apply_action(connection, record_id=first.record_id, action="activate", idempotency_key="catch-activate")
    structured_ledger.apply_action(
        connection, record_id=first.record_id, action="supersede", replacement_record_id=second.record_id,
        idempotency_key="catch-supersede",
    )
    connection.close()

    caught_up = indexer.catch_up_index(paths)
    caught_projection = _logical_projection(paths.database_path)
    with sqlite3.connect(paths.database_path) as cache:
        caught_generation = cache.execute("SELECT active_generation_id FROM retrieval_state").fetchone()[0]
        states = cache.execute("SELECT record_id, state FROM retrieval_documents ORDER BY record_id").fetchall()
    rebuilt = indexer.rebuild_index(paths)

    assert caught_up.status is rebuilt.status is RetrievalStatus.CURRENT
    assert caught_generation != initial_generation
    assert dict(states) == {first.record_id: "superseded", second.record_id: "proposed"}
    assert caught_projection == _logical_projection(paths.database_path)


def test_semantic_indexer_facade_defers_provider_selection_to_the_explicit_caller() -> None:
    import inspect

    indexer = importlib.import_module("codex_memory.retrieval.indexer")
    assert "provider" in inspect.signature(indexer.rebuild_semantic_index).parameters
    assert "provider" in inspect.signature(indexer.catch_up_semantic_index).parameters
