from __future__ import annotations

from contextlib import closing
import json
import sqlite3

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.memory_context import build_memory_packet, current_context_id
from codex_memory.memory_evidence import event_uri
from codex_memory.retrieval import cache_database, config
from codex_memory.retrieval.contracts import (
    DegradedReason, RetrievalRequest, RetrievalScope, RetrievalStatus,
)
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from test_p26c_selected_host import selected_host_lab  # fixture dependency
from test_p26d_compact import compact_lab, invoke


def test_selected_compact_deleted_active_generation_keeps_fresh_legacy(
    compact_lab, monkeypatch,
):
    """A deleted generation row must not be accepted as a current cache."""
    runtime, marker, _hooks = compact_lab
    assert marker["data_origin"] == "synthetic"
    assert marker["provider_policy"] == "disabled"

    def forbidden_provider(*args, **kwargs):
        pytest.fail("selected compact generation failure must not construct a provider")

    monkeypatch.setattr(config, "provider_from_config", forbidden_provider)
    prompt = "What is the selected host verification code?"
    rc, _out, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="before", prompt=prompt)
    assert rc == 0, err
    assert json.loads(err)["delivery"] == "emitted"

    immutable_tables = ("memory_records", "memory_events", "memory_evidence", "memory_relations")
    canonical_before = {
        table: [tuple(row) for row in runtime.connection.execute(f"SELECT * FROM {table}")]
        for table in immutable_tables
    }
    events_before = [tuple(row) for row in runtime.connection.execute(
        "SELECT * FROM events ORDER BY event_id")]
    manifests = runtime.paths.runtime_root / "injections"
    manifests_before = {path.relative_to(manifests): path.read_bytes()
        for path in manifests.glob("inj_*/manifest.json")}
    assert len(manifests_before) == 1

    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert paths.database_path.resolve().is_relative_to(runtime.paths.runtime_root.resolve())
    assert paths.database_path.resolve() != paths.canonical_database_path.resolve()
    assert paths.database_path.is_file()
    service = RetrievalService(paths)
    baseline = service.status()
    assert baseline.status is RetrievalStatus.CURRENT
    assert baseline.generation is not None
    active_id = baseline.generation.generation_id

    # Only this newly-created synthetic cache is damaged. Deterministic gen_*
    # identities are SQLite rows, not files in the optional semantic directory.
    # Restore the exact trigger so rejection cannot be due to schema damage.
    with closing(sqlite3.connect(
        f"{paths.database_path.as_uri()}?mode=rw", uri=True, isolation_level=None,
    )) as cache:
        cache.execute("PRAGMA foreign_keys=OFF")
        assert cache.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        schema_before = cache.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        generations_before = cache.execute(
            "SELECT * FROM retrieval_generations ORDER BY generation_seq").fetchall()
        assert sum(row[1] == active_id for row in generations_before) == 1
        preserved_tables = (
            "retrieval_state", "retrieval_generation_current", "retrieval_generation_events",
            "retrieval_generation_capabilities", "retrieval_documents",
            "retrieval_document_sources", "retrieval_symbols",
        )
        cache_before = {table: cache.execute(f"SELECT * FROM {table}").fetchall()
            for table in preserved_tables}
        with cache:
            cache.execute("BEGIN")
            cache.execute("DROP TRIGGER retrieval_generations_reject_delete")
            deleted = cache.execute(
                "DELETE FROM retrieval_generations WHERE generation_id = ?", (active_id,))
            assert deleted.rowcount == 1
            cache.execute(cache_database._append_only_trigger_statement("retrieval_generations", "delete"))
        assert cache.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall() == schema_before
        assert cache.execute(
            "SELECT * FROM retrieval_generations ORDER BY generation_seq").fetchall() == [
                row for row in generations_before if row[1] != active_id]
        assert {table: cache.execute(f"SELECT * FROM {table}").fetchall()
            for table in preserved_tables} == cache_before
        assert cache.execute(
            "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1").fetchone()[0] == active_id
        assert cache.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert cache.execute("PRAGMA foreign_key_check").fetchall()

    damaged_bytes = paths.database_path.read_bytes()
    damaged = service.status()
    assert damaged.status is RetrievalStatus.CORRUPT
    assert damaged.degraded_reason is DegradedReason.INDEX_CORRUPT
    assert damaged.generation is None
    rc, out, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="after", prompt=prompt)
    assert rc == 0, err
    metadata = json.loads(err)
    assert metadata["delivery"] == "legacy_fallback"
    assert metadata["reason"] == "retrieval_degraded"
    assert metadata["injection_id"] is None and metadata["manifest_sha256"] is None
    assert metadata["output_mode"] == "selected_compact"
    assert metadata["source_origin"] == "hook"
    assert metadata["host_acknowledgement"] == "unknown"
    assert metadata["cleanup_status"] == "complete"

    context = current_context_id(runtime.connection, session_id="selected-session")
    source_uri = event_uri("selected-session", runtime.events.highwater("selected-session"))
    legacy = build_memory_packet(runtime.connection, BlobStore(runtime.paths.blobs_dir),
        session_id="selected-session", turn_id="after", context_id=context,
        scope_key=str(runtime.paths.repo_root), prompt_event_uri=source_uri,
        query=prompt, max_chars=4000)
    output = json.loads(out)["hookSpecificOutput"]
    assert output["hookEventName"] == "UserPromptSubmit"
    assert output["additionalContext"] == legacy
    assert "HOST-P26C-EXAMPLE" in legacy and len(legacy) <= 4000
    result = service.query(RetrievalRequest(
        prompt, "selected-session", "after", context, source_uri,
        RetrievalScope(str(runtime.paths.repo_root)),
    ))
    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reasons == (DegradedReason.INDEX_CORRUPT,)
    assert result.generation is None and result.index_stale

    assert paths.database_path.read_bytes() == damaged_bytes
    assert {path.relative_to(manifests): path.read_bytes()
        for path in manifests.glob("inj_*/manifest.json")} == manifests_before
    assert {table: [tuple(row) for row in runtime.connection.execute(f"SELECT * FROM {table}")]
        for table in immutable_tables} == canonical_before
    events_after = [tuple(row) for row in runtime.connection.execute(
        "SELECT * FROM events ORDER BY event_id")]
    assert events_after[:-1] == events_before
    assert tuple(runtime.connection.execute(
        "SELECT event_name, session_id, turn_id FROM events ORDER BY event_id DESC LIMIT 1"
    ).fetchone()) == ("UserPromptSubmit", "selected-session", "after")
    assert runtime.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
