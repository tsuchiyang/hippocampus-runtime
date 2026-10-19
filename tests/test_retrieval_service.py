from __future__ import annotations

import shutil
import sqlite3
import multiprocessing
import os
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from codex_memory import database, memory_context, memory_review, paths as runtime_path_module, structured_ledger
from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import capture_candidate
from codex_memory.event_store import EventStore
from codex_memory.hook_input import parse_hook_payload
from codex_memory.memory_evidence import event_uri
from codex_memory.paths import RuntimePaths
from codex_memory.retrieval.canonical_snapshot import capture_canonical_snapshot
from codex_memory.retrieval import canonical_snapshot
from codex_memory.retrieval.cache_database import (
    append_generation,
    append_generation_event,
    connect_cache_database,
    inspect_cache_schema_version,
    deterministic_generation_id as cache_deterministic_generation_id,
    rebuild_projections,
    initialize_schema_v2,
    initialize_schema_v3,
)
from codex_memory.retrieval import index_store
from codex_memory.retrieval.contracts import IndexHighwater
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.provider import ProviderInput
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider_health import ProviderHealthStore
from codex_memory.ledger_common import VerificationError, canonical_json
from codex_memory.retrieval.contracts import RetrievalStatus
from codex_memory.retrieval.service import RetrievalService, deterministic_generation_id
from codex_memory.retrieval import service as retrieval_service
from codex_memory.retrieval.projector import project_stale_delta
from retrieval_provider_fakes import FakeProvider


def _hold_rebuild_lock(temp_dir: str, entered: object, release: object, active: object, overlap: object) -> None:
    with retrieval_service.process_rebuild_lock(Path(temp_dir)):
        with active.get_lock():
            active.value += 1
            if active.value > 1:
                overlap.set()
        entered.set()
        release.wait(5)
        with active.get_lock():
            active.value -= 1


def _run_retrieval_operation(repo: str, operation: str, release: object, results: object) -> None:
    release.wait(10)
    service = RetrievalService(RetrievalPaths.for_repo(Path(repo)))
    results.put(getattr(service, operation)().status.value)


def _run_retrieval_operation_at_barrier(
    repo: str,
    operation: str,
    label: str,
    entered: object,
    release: object,
    order: object,
    results: object,
) -> None:
    service = RetrievalService(RetrievalPaths.for_repo(Path(repo)))

    def inject(stage: str) -> None:
        if stage == "ddl":
            order.put(f"{label}:ddl")
            entered.set()
            release.wait(10)

    result = getattr(service, operation)(failure_injector=inject)
    results.put((label, result.status.value))


def _canonical_database(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    runtime = RuntimePaths.for_repo(repo)
    runtime.ensure()
    connection = database.connect_database(runtime.db_path)
    database.migrate_database(connection)
    return repo, connection


def test_semantic_service_methods_are_explicit_and_do_not_change_legacy_init_surface() -> None:
    assert hasattr(RetrievalService, "rebuild_semantic")
    assert hasattr(RetrievalService, "catch_up_semantic")


def _canonical_logical_rows(database_path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    with sqlite3.connect(database_path) as connection:
        tables = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        )
        return {
            table: tuple(
                tuple(row)
                for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
            )
            for table in tables
        }


def _installed_projection(database_path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    """Read only generation-independent projection semantics for equivalence checks."""
    with closing(sqlite3.connect(database_path)) as connection:
        active = connection.execute(
            "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
        ).fetchone()[0]
        result = {
            "documents": tuple(connection.execute(
                """SELECT record_id, record_seq, record_type, scope_kind, scope_key,
                topic_key, statement, value_json, search_text, state,
                last_memory_event_seq, evidence_count, content_sha256, document_sha256
                FROM retrieval_documents WHERE generation_id = ? ORDER BY record_seq""",
                (active,),
            )),
            "sources": tuple(connection.execute(
                """SELECT s.document_id, s.ordinal, s.safe_reference, s.source_sha256
                FROM retrieval_document_sources AS s
                JOIN retrieval_documents AS d ON d.document_id = s.document_id
                WHERE s.generation_id = ? ORDER BY d.record_seq, s.ordinal""",
                (active,),
            )),
            "symbols": tuple(connection.execute(
                """SELECT s.document_id, s.ordinal, s.symbol_kind,
                s.normalized_symbol, s.display_symbol, s.symbol_sha256
                FROM retrieval_symbols AS s
                JOIN retrieval_documents AS d ON d.document_id = s.document_id
                WHERE s.generation_id = ? ORDER BY d.record_seq, s.ordinal""",
                (active,),
            )),
            "capability": tuple(connection.execute(
                """SELECT fts5_state, index_policy_fingerprint,
                projection_manifest_sha256, projected_document_count,
                omitted_unsafe_count FROM retrieval_generation_capabilities
                WHERE generation_id = ?""",
                (active,),
            )),
        }
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
        ).fetchone():
            result["fts"] = tuple(connection.execute(
                """SELECT f.document_id, f.search_text
                FROM retrieval_documents_fts AS f
                JOIN retrieval_documents AS d ON d.document_id = f.document_id
                WHERE f.generation_id = ? ORDER BY d.record_seq""",
                (active,),
            ))
        return result


def _assert_catch_up_matches_clean_rebuild(paths: RetrievalPaths) -> None:
    canonical_before = paths.canonical_database_path.read_bytes()
    caught_up = RetrievalService(paths).catch_up()
    assert caught_up.status is RetrievalStatus.CURRENT
    assert caught_up.generation is not None
    assert caught_up.indexed_highwater == caught_up.canonical_highwater
    caught_projection = _installed_projection(paths.database_path)
    caught_highwater = caught_up.generation.indexed_highwater
    assert paths.canonical_database_path.read_bytes() == canonical_before

    clean_repo = paths.repo_root.parent / (
        f"clean-rebuild-{caught_highwater.memory_event_seq}-"
        f"{caught_highwater.candidate_event_seq}"
    )
    clean_repo.mkdir()
    clean_runtime = RuntimePaths.for_repo(clean_repo)
    clean_runtime.ensure()
    with closing(sqlite3.connect(paths.canonical_database_path)) as source:
        with closing(sqlite3.connect(clean_runtime.db_path)) as destination:
            source.backup(destination)
    clean_paths = RetrievalPaths.for_repo(clean_repo)
    rebuilt = RetrievalService(clean_paths).rebuild()

    assert rebuilt.status is RetrievalStatus.CURRENT
    assert rebuilt.generation is not None
    assert rebuilt.generation.indexed_highwater == caught_highwater
    assert rebuilt.indexed_highwater == caught_up.indexed_highwater
    assert _installed_projection(clean_paths.database_path) == caught_projection
    assert paths.canonical_database_path.read_bytes() == canonical_before


def _propose(connection: sqlite3.Connection, repo: Path, key: str, *, evidence: bool = False):
    return structured_ledger.propose_record(
        connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(repo),
        topic_key=f"snapshot.{key}",
        statement=f"Snapshot record {key}.",
        idempotency_key=f"record-{key}",
        evidence_uris=("https://example.test/evidence",) if evidence else (),
    )


def _capture(connection: sqlite3.Connection, repo: Path, key: str, *, evidence: bool = False):
    return capture_candidate(
        connection,
        BlobStore(RuntimePaths.for_repo(repo).blobs_dir),
        project_scope_key=str(repo),
        record_type="decision",
        scope_kind="project",
        scope_key=str(repo),
        topic_key=f"snapshot.candidate.{key}",
        statement=f"Snapshot candidate {key}.",
        session_id="snapshot-session",
        turn_id="snapshot-turn",
        context_id="startup",
        idempotency_key=f"candidate-{key}",
        evidence_uris=("https://example.test/candidate-evidence",) if evidence else (),
    )


def test_stale_delta_does_not_duplicate_a_direct_certified_constitution(runtime) -> None:
    """Candidate-review certification is direct-prefix authority even before catch-up."""
    from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope

    repo = runtime.paths.repo_root
    constitution = structured_ledger.propose_record(
        runtime.connection, record_type="constitution", scope_kind="project",
        scope_key=str(repo), topic_key="mandatory.stale", statement="Keep this policy.",
        idempotency_key="mandatory-stale-record",
    )
    structured_ledger.apply_action(
        runtime.connection, record_id=constitution.record_id, action="activate",
        idempotency_key="mandatory-stale-active",
    )
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(repo))
    assert service.rebuild().status is RetrievalStatus.CURRENT

    prompt = EventStore(runtime.connection, BlobStore(runtime.paths.blobs_dir)).append(
        parse_hook_payload({
            "hook_event_name": "UserPromptSubmit", "session_id": "mandatory-stale-session",
            "turn_id": "mandatory-stale-turn", "cwd": str(repo), "prompt": "certify policy",
        })
    )
    candidate = capture_candidate(
        runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key=str(repo),
        record_type="constitution", scope_kind="project", scope_key=str(repo),
        topic_key="mandatory.stale", statement="Keep this policy.",
        session_id="mandatory-stale-session", turn_id="mandatory-stale-turn",
        context_id="mandatory-stale-context", idempotency_key="mandatory-stale-candidate",
        evidence_uris=(event_uri("mandatory-stale-session", prompt.event_id),),
    )
    reviewed = memory_review.review_candidate(
        runtime.connection, BlobStore(runtime.paths.blobs_dir), candidate_id=candidate.candidate_id,
        outcome="promote", reason="Direct user certification.",
        idempotency_key="mandatory-stale-review",
    )
    assert reviewed.action == "promote" and reviewed.policy_code == "linked_duplicate"

    result = service.query(RetrievalRequest(
        "policy", "query-session", "query-turn", "query-context", None,
        RetrievalScope(str(repo)),
    ))
    identifiers = [candidate.candidate_id for candidate in result.candidates]
    assert identifiers.count(constitution.record_id) == 1


def _legacy_retrieval_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_cache_database(path)
    connection.executescript(
        """
        CREATE TABLE retrieval_schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE retrieval_generations (
            generation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id TEXT NOT NULL UNIQUE,
            canonical_highwater_json TEXT NOT NULL,
            indexed_highwater_json TEXT NOT NULL,
            canonical_content_fingerprint TEXT NOT NULL,
            provider_fingerprint TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_generation_events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            action TEXT NOT NULL CHECK (action IN ('create', 'validate', 'activate', 'retire', 'fail')),
            payload_json TEXT NOT NULL,
            event_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_generation_current (
            generation_id TEXT PRIMARY KEY REFERENCES retrieval_generations(generation_id),
            state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')),
            last_event_seq INTEGER NOT NULL,
            last_event_id TEXT NOT NULL REFERENCES retrieval_generation_events(event_id),
            validation_state TEXT NOT NULL,
            degraded_reason TEXT,
            failure_metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_generation_id TEXT REFERENCES retrieval_generations(generation_id),
            validation_state TEXT NOT NULL,
            degraded_reason TEXT,
            failure_metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX retrieval_generation_events_generation_seq ON retrieval_generation_events(generation_id, event_seq);
        CREATE INDEX retrieval_generation_current_state_generation ON retrieval_generation_current(state, generation_id);
        CREATE TRIGGER retrieval_generations_reject_update
            BEFORE UPDATE ON retrieval_generations
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generations_reject_delete
            BEFORE DELETE ON retrieval_generations
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generations_guard_insert_conflict
            BEFORE INSERT ON retrieval_generations
            WHEN EXISTS (
                SELECT 1 FROM retrieval_generations
                WHERE (generation_seq = NEW.generation_seq) OR (generation_id = NEW.generation_id)
            )
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generation_events_reject_update
            BEFORE UPDATE ON retrieval_generation_events
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generation_events_reject_delete
            BEFORE DELETE ON retrieval_generation_events
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generation_events_guard_insert_conflict
            BEFORE INSERT ON retrieval_generation_events
            WHEN EXISTS (
                SELECT 1 FROM retrieval_generation_events
                WHERE (event_seq = NEW.event_seq) OR (event_id = NEW.event_id)
            )
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        INSERT INTO retrieval_schema_meta(key, value) VALUES ('schema_version', '1');
        INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', NULL, '{}',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z');
        """
    )
    highwater = IndexHighwater(record_seq=1)
    generation_id = cache_deterministic_generation_id(highwater, "a" * 64, schema_version=1)
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
    for action in ("create", "validate", "activate"):
        append_generation_event(
            connection,
            generation_id=generation_id,
            action=action,
            created_at="2026-01-02T03:04:05.000000Z",
        )
    rebuild_projections(connection)
    connection.close()


def _legacy_retrieval_database_with_retired_history(path: Path) -> None:
    _legacy_retrieval_database(path)
    connection = connect_cache_database(path)
    try:
        first = cache_deterministic_generation_id(IndexHighwater(record_seq=1), "a" * 64, schema_version=1)
        second_highwater = IndexHighwater(record_seq=2)
        second = cache_deterministic_generation_id(second_highwater, "b" * 64, schema_version=1)
        append_generation(
            connection,
            generation_id=second,
            canonical_highwater=second_highwater,
            indexed_highwater=second_highwater,
            canonical_content_fingerprint="b" * 64,
            provider_fingerprint="unconfigured:p2.1",
            model_fingerprint="unconfigured:p2.1",
            created_at="2026-01-03T03:04:05.000000Z",
        )
        append_generation_event(connection, generation_id=first, action="retire", created_at="2026-01-03T03:04:05.000000Z")
        for action in ("create", "validate", "activate"):
            append_generation_event(
                connection,
                generation_id=second,
                action=action,
                created_at="2026-01-03T03:04:05.000000Z",
            )
        rebuild_projections(connection)
    finally:
        connection.close()


def test_snapshot_uses_verified_read_only_canonical_database_and_exact_highwater(
    tmp_path: Path,
) -> None:
    repo, connection = _canonical_database(tmp_path)
    record = _propose(connection, repo, "one", evidence=True)
    structured_ledger.apply_action(
        connection, record_id=record.record_id, action="activate", idempotency_key="activate-one"
    )
    _capture(connection, repo, "one", evidence=True)
    connection.close()

    snapshot = capture_canonical_snapshot(RetrievalPaths.for_repo(repo))

    assert snapshot.schema_version == 3
    assert snapshot.highwater == IndexHighwater(
        record_seq=1,
        memory_event_seq=2,
        evidence_seq=1,
        relation_seq=0,
        candidate_seq=1,
        candidate_event_seq=1,
        candidate_evidence_seq=1,
    )
    assert len(snapshot.content_fingerprint) == 64
    assert snapshot.source_timestamp.endswith("Z")


def test_snapshot_fingerprint_is_logical_not_sqlite_physical_layout(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "one", evidence=True)
    connection.close()
    original = RetrievalPaths.for_repo(repo).canonical_database_path
    copied_repo = tmp_path / "copied repo"
    shutil.copytree(repo, copied_repo)
    copied = RetrievalPaths.for_repo(copied_repo).canonical_database_path
    with copied.open("ab") as physical_copy:
        physical_copy.write(b"\x00")

    assert capture_canonical_snapshot(RetrievalPaths.for_repo(repo)).content_fingerprint == capture_canonical_snapshot(RetrievalPaths.for_repo(copied_repo)).content_fingerprint
    assert original.read_bytes() != copied.read_bytes()


@pytest.mark.parametrize("change", ["record", "record_event", "record_evidence", "relation", "candidate", "candidate_event", "candidate_evidence"])
def test_each_immutable_stream_change_changes_snapshot_fingerprint(tmp_path: Path, change: str) -> None:
    repo, connection = _canonical_database(tmp_path)
    before = capture_canonical_snapshot(RetrievalPaths.for_repo(repo))
    if change == "relation":
        old = _propose(connection, repo, "old")
        replacement = _propose(connection, repo, "replacement")
        structured_ledger.apply_action(connection, record_id=old.record_id, action="activate", idempotency_key="activate-old")
        structured_ledger.apply_action(connection, record_id=old.record_id, action="supersede", replacement_record_id=replacement.record_id, idempotency_key="supersede-old")
    elif change.startswith("record"):
        record = _propose(connection, repo, change, evidence=change == "record_evidence")
        if change == "record_event":
            structured_ledger.apply_action(connection, record_id=record.record_id, action="activate", idempotency_key="activate-record")
    else:
        candidate = _capture(connection, repo, change, evidence=change == "candidate_evidence")
        if change == "candidate_event":
            from codex_memory.candidate_memory import apply_candidate_action
            apply_candidate_action(connection, BlobStore(RuntimePaths.for_repo(repo).blobs_dir), candidate_id=candidate.candidate_id, action="defer", reason="Review later.", idempotency_key="defer-candidate")
    after = capture_canonical_snapshot(RetrievalPaths.for_repo(repo))
    connection.close()

    assert after.content_fingerprint != before.content_fingerprint


def test_projection_changes_do_not_change_snapshot_fingerprint(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "one")
    before = capture_canonical_snapshot(RetrievalPaths.for_repo(repo))
    structured_ledger.rebuild_projection(connection)
    after = capture_canonical_snapshot(RetrievalPaths.for_repo(repo))
    connection.close()

    assert after.content_fingerprint == before.content_fingerprint


def test_snapshot_rejects_missing_corrupt_or_noncanonical_database_without_creating_retrieval_state(tmp_path: Path) -> None:
    repo = tmp_path / "missing repo"
    repo.mkdir()
    paths = RetrievalPaths.for_repo(repo)

    with pytest.raises(VerificationError):
        capture_canonical_snapshot(paths)
    assert not paths.runtime_root.exists()

    db = repo / ".codex-memory" / "db"
    db.mkdir(parents=True)
    (db / "memory.sqlite3").write_bytes(b"not sqlite")
    with pytest.raises(VerificationError):
        capture_canonical_snapshot(paths)
    assert not paths.retrieval_root.exists()


def test_snapshot_rejects_schema_version_other_than_three(tmp_path: Path) -> None:
    repo = tmp_path / "unsupported repo"
    db = repo / ".codex-memory" / "db"
    db.mkdir(parents=True)
    with sqlite3.connect(db / "memory.sqlite3") as connection:
        connection.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version', '2')")

    with pytest.raises(VerificationError):
        capture_canonical_snapshot(RetrievalPaths.for_repo(repo))
    assert not (repo / ".codex-memory" / "retrieval").exists()


def test_service_builds_a_deterministic_current_empty_substrate_without_changing_canonical(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "one")
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    before_bytes = paths.canonical_database_path.read_bytes()
    before = capture_canonical_snapshot(paths)

    result = RetrievalService(paths).rebuild()

    assert result.status is RetrievalStatus.CURRENT
    assert result.generation is not None
    assert result.generation.generation_id == deterministic_generation_id(before)
    assert result.generation.indexed_highwater == before.highwater
    assert paths.canonical_database_path.read_bytes() == before_bytes
    assert capture_canonical_snapshot(paths) == before
    assert list(paths.generations_dir.iterdir()) == []


def test_rebuild_retries_when_projection_snapshot_changes(
    tmp_path: Path,
) -> None:
    repo, connection = _canonical_database(tmp_path)
    first = _propose(connection, repo, "before-retry")
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    before_rows = _canonical_logical_rows(paths.canonical_database_path)
    mutations = 0

    def mutate_after_projection(step: str) -> None:
        nonlocal mutations
        if step != "projection" or mutations:
            return
        mutations += 1
        writer = database.connect_database(paths.canonical_database_path)
        try:
            _propose(writer, repo, "after-retry")
        finally:
            writer.close()

    result = RetrievalService(paths).rebuild(failure_injector=mutate_after_projection)

    assert mutations == 1
    assert result.status is RetrievalStatus.CURRENT
    with sqlite3.connect(paths.database_path) as cache:
        document_ids = [
            row[0]
            for row in cache.execute(
                "SELECT record_id FROM retrieval_documents ORDER BY document_seq"
            )
        ]
    assert document_ids == [
        first.record_id,
        _canonical_logical_rows(paths.canonical_database_path)["memory_records"][1][1],
    ]
    assert _canonical_logical_rows(paths.canonical_database_path)["memory_records"][:1] == before_rows["memory_records"]


def test_service_distinguishes_absent_corrupt_unsupported_and_stale_fingerprint(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.status().status is RetrievalStatus.ABSENT
    connection.close()
    paths.prepare_for_write()
    paths.database_path.write_bytes(b"not sqlite")
    assert service.status().status is RetrievalStatus.CORRUPT
    assert service.rebuild().status is RetrievalStatus.CURRENT
    paths.database_path.unlink()
    with sqlite3.connect(paths.database_path) as future:
        future.execute("CREATE TABLE retrieval_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        future.execute("INSERT INTO retrieval_schema_meta VALUES ('schema_version', '99')")
    future.close()
    assert service.status().status is RetrievalStatus.UNSUPPORTED
    paths.database_path.unlink()
    current = service.rebuild()
    assert current.status is RetrievalStatus.CURRENT
    with sqlite3.connect(paths.database_path) as cache:
        cache.execute("DROP TRIGGER retrieval_generations_reject_update")
        cache.execute("UPDATE retrieval_generations SET canonical_content_fingerprint = ?", ("b" * 64,))
        cache.execute("""CREATE TRIGGER retrieval_generations_reject_update
            BEFORE UPDATE ON retrieval_generations
            BEGIN SELECT RAISE(ABORT, 'append-only table'); END""")
    assert service.status().status is RetrievalStatus.CORRUPT


def test_service_rebuilds_exact_legacy_v1_cache_without_promoting_normal_status_checks(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "legacy")
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    _legacy_retrieval_database(paths.database_path)
    service = RetrievalService(paths)

    assert service.status().status is RetrievalStatus.UNSUPPORTED

    rebuilt = service.rebuild()

    assert rebuilt.status is RetrievalStatus.CURRENT
    assert rebuilt.generation is not None
    assert service.status().status is RetrievalStatus.CURRENT


def test_service_rebuilds_exact_legacy_v1_cache_and_preserves_legacy_history_rows(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "legacy-history")
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    _legacy_retrieval_database_with_retired_history(paths.database_path)
    service = RetrievalService(paths)

    assert service.status().status is RetrievalStatus.UNSUPPORTED

    rebuilt = service.rebuild()

    assert rebuilt.status is RetrievalStatus.CURRENT
    assert rebuilt.generation is not None
    with connect_cache_database(paths.database_path, readonly=True) as connection:
        assert inspect_cache_schema_version(connection) == 4
        assert connection.execute("SELECT COUNT(*) FROM retrieval_generations").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM retrieval_generation_events").fetchone()[0] == 11
        assert connection.execute("SELECT COUNT(*) FROM retrieval_generation_current").fetchone()[0] == 3
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT state, COUNT(*) FROM retrieval_generation_current GROUP BY state ORDER BY state"
            )
        ] == [("active", 1), ("retired", 2)]
        assert connection.execute("SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1").fetchone()[0] is not None


def test_service_explicitly_upgrades_verified_v2_history_and_trace_rows_to_v4(tmp_path: Path) -> None:
    repo, canonical = _canonical_database(tmp_path)
    _propose(canonical, repo, "v2-upgrade")
    canonical.close()
    paths = RetrievalPaths.for_repo(repo)
    paths.prepare_for_write()
    cache = connect_cache_database(paths.database_path)
    try:
        initialize_schema_v2(cache)
        highwater = IndexHighwater(record_seq=1)
        legacy_id = cache_deterministic_generation_id(highwater, "a" * 64)
        append_generation(
            cache, generation_id=legacy_id, canonical_highwater=highwater,
            indexed_highwater=highwater, canonical_content_fingerprint="a" * 64,
            provider_fingerprint="unconfigured:p2.1", model_fingerprint="unconfigured:p2.1",
            created_at="2026-01-02T03:04:05.000000Z",
        )
        for action in ("create", "validate", "activate"):
            append_generation_event(cache, generation_id=legacy_id, action=action, created_at="2026-01-02T03:04:05.000000Z")
        index_store.append_run(
            cache, run_id="v2-run", mode="shadow", generation_id=legacy_id,
            query_sha256="b" * 64, canonical_highwater_json=canonical_json(highwater.to_dict()),
            indexed_highwater_json=canonical_json(highwater.to_dict()), status="current",
            degraded_reasons_json="[]", trace_sha256="c" * 64,
            created_at="2026-01-02T03:04:05.000000Z",
        )
        index_store.append_candidate_trace(
            cache, trace_id="v2-trace", run_id="v2-run", candidate_kind="record",
            candidate_id="record-1", channel="exact", channel_rank=1,
            contribution_numerator=1, contribution_denominator=1, fused_rank=1,
            selected=True, reason_codes_json="[]", trace_sha256="d" * 64,
        )
        rebuild_projections(cache)
    finally:
        cache.close()

    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT
    with connect_cache_database(paths.database_path, readonly=True) as upgraded:
        assert inspect_cache_schema_version(upgraded) == 4
        legacy = upgraded.execute(
            "SELECT generation_id, retrieval_schema_version FROM retrieval_generations WHERE generation_id = ?", (legacy_id,)
        ).fetchone()
        assert tuple(legacy) == (legacy_id, 2)
        assert tuple(upgraded.execute(
            "SELECT wall_clock_ms, provider_trace_json FROM retrieval_runs WHERE run_id = 'v2-run'"
        ).fetchone()) == (0, "[]")
        assert tuple(upgraded.execute(
            "SELECT raw_score_hex, rerank_score_hex FROM retrieval_candidate_traces WHERE trace_id = 'v2-trace'"
        ).fetchone()) == (None, None)


def test_service_explicitly_upgrades_verified_v3_to_single_file_v4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, canonical = _canonical_database(tmp_path)
    _propose(canonical, repo, "v3-upgrade")
    canonical.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    monkeypatch.setattr(retrieval_service, "initialize_schema", initialize_schema_v3)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    legacy = connect_cache_database(paths.database_path, readonly=True)
    try:
        assert inspect_cache_schema_version(legacy) == 3
    finally:
        legacy.close()
    monkeypatch.undo()

    rebuilt = service.rebuild()

    assert rebuilt.status is RetrievalStatus.CURRENT
    upgraded = connect_cache_database(paths.database_path, readonly=True)
    try:
        assert inspect_cache_schema_version(upgraded) == 4
        assert upgraded.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'retrieval_provider_attempt_admissions'"
        ).fetchone()
        assert upgraded.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'retrieval_provider_attempt_results'"
        ).fetchone()
    finally:
        upgraded.close()


@pytest.mark.parametrize("column", ("payload_sha256", "fencing_version"))
def test_rebuild_rejects_tampered_provider_control_history_before_activation(
    tmp_path: Path, column: str
) -> None:
    repo, canonical = _canonical_database(tmp_path)
    _propose(canonical, repo, f"tampered-{column}")
    canonical.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    provider = FakeProvider()
    capabilities = provider.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    response = ProviderCallController(
        provider,
        DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        ProviderHealthStore(paths.database_path),
        capabilities.fingerprint,
        now_ns=time.monotonic_ns,
    ).dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)
    assert response.call.status.value == "ok"

    connection = connect_cache_database(paths.database_path)
    try:
        active_before = connection.execute(
            "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
        ).fetchone()[0]
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'retrieval_provider_attempt_admissions_reject_update'"
        ).fetchone()[0]
        connection.execute(
            "DROP TRIGGER retrieval_provider_attempt_admissions_reject_update"
        )
        if column == "payload_sha256":
            connection.execute(
                "UPDATE retrieval_provider_attempt_admissions "
                "SET payload_sha256 = ?",
                ("0" * 64,),
            )
        else:
            connection.execute(
                "UPDATE retrieval_provider_attempt_admissions "
                "SET fencing_version = fencing_version + 1"
            )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    result = service.rebuild(failure_injector=lambda _stage: None)

    assert result.status is RetrievalStatus.CORRUPT
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        assert connection.execute(
            "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
        ).fetchone()[0] == active_before
    finally:
        connection.close()


def test_rebuild_permission_error_does_not_unlink_absent_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    actual_replace = retrieval_service.os.replace
    unlink_calls: list[Path] = []
    actual_unlink = Path.unlink

    def fail_install(source: Path | str, destination: Path | str) -> None:
        if Path(source).name.startswith("install-") and Path(destination) == paths.database_path:
            raise PermissionError("install denied")
        actual_replace(source, destination)

    def spy_unlink(self: Path, *args: object, **kwargs: object) -> object:
        if Path(self) == paths.database_path:
            unlink_calls.append(Path(self))
            raise AssertionError("destination unlink must not be called")
        return actual_unlink(self, *args, **kwargs)

    monkeypatch.setattr(retrieval_service.os, "replace", fail_install)
    monkeypatch.setattr(Path, "unlink", spy_unlink, raising=False)

    with pytest.raises(PermissionError, match="install denied"):
        service.rebuild()

    assert not paths.database_path.exists()
    assert unlink_calls == []


def test_rebuild_leaves_unknown_future_schema_untouched(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    paths.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(paths.database_path) as future:
        future.execute("CREATE TABLE retrieval_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        future.execute("CREATE TABLE retrieval_generations (generation_seq INTEGER PRIMARY KEY AUTOINCREMENT, generation_id TEXT NOT NULL UNIQUE, canonical_highwater_json TEXT NOT NULL, indexed_highwater_json TEXT NOT NULL, canonical_content_fingerprint TEXT NOT NULL, provider_fingerprint TEXT NOT NULL, model_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL)")
        future.execute("CREATE TABLE retrieval_generation_events (event_seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id), action TEXT NOT NULL CHECK (action IN ('create', 'validate', 'activate', 'retire', 'fail')), payload_json TEXT NOT NULL, event_sha256 TEXT NOT NULL, created_at TEXT NOT NULL)")
        future.execute("CREATE TABLE retrieval_generation_current (generation_id TEXT PRIMARY KEY REFERENCES retrieval_generations(generation_id), state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')), last_event_seq INTEGER NOT NULL, last_event_id TEXT NOT NULL REFERENCES retrieval_generation_events(event_id), validation_state TEXT NOT NULL, degraded_reason TEXT, failure_metadata_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        future.execute("CREATE TABLE retrieval_state (singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1), active_generation_id TEXT REFERENCES retrieval_generations(generation_id), validation_state TEXT NOT NULL, degraded_reason TEXT, failure_metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        future.execute("INSERT INTO retrieval_schema_meta VALUES ('schema_version', '99')")
    before = paths.database_path.read_bytes()
    service = RetrievalService(paths)
    assert service.status().status is RetrievalStatus.UNSUPPORTED
    rebuilt = service.rebuild()
    assert rebuilt.status is RetrievalStatus.UNSUPPORTED
    assert paths.database_path.read_bytes() == before


def test_init_is_idempotent_and_rebuild_recovers_after_entire_retrieval_directory_is_deleted(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    first = service.init()
    second = service.init()
    assert first.generation == second.generation
    shutil.rmtree(paths.retrieval_root)
    rebuilt = service.rebuild()
    assert rebuilt.status is RetrievalStatus.CURRENT


@pytest.mark.parametrize("step", ["ddl", "write", "projection", "verify", "replace"])
def test_rebuild_failure_cleans_owned_temp_and_preserves_previous_valid_database(tmp_path: Path, step: str) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    service.rebuild()
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, f"changed-{step}")
    connection.close()
    with pytest.raises(RuntimeError, match=step):
        service.rebuild(failure_injector=lambda candidate: (_ for _ in ()).throw(RuntimeError(candidate)) if candidate == step else None)
    assert paths.database_path.read_bytes() == previous
    assert [path.name for path in paths.temp_dir.iterdir()] == ["rebuild.lock"]


def test_rebuild_creates_schema_inside_the_same_immediate_transaction_as_history_and_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    observed: list[bool] = []
    original = retrieval_service.initialize_schema

    def inspect_transaction(connection: sqlite3.Connection, **kwargs: object) -> None:
        observed.append(connection.in_transaction)
        original(connection, **kwargs)

    monkeypatch.setattr(retrieval_service, "initialize_schema", inspect_transaction)
    assert RetrievalService(RetrievalPaths.for_repo(repo)).rebuild().status is RetrievalStatus.CURRENT
    assert observed == [True]


def test_status_reports_unsafe_retrieval_before_trying_canonical_snapshot(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    paths.runtime_root.mkdir(exist_ok=True)
    paths.retrieval_root.write_text("not a directory", encoding="utf-8")

    result = RetrievalService(paths).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "path_unsafe"


def test_status_classifies_missing_canonical_as_unavailable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    missing = RetrievalService(RetrievalPaths.for_repo(repo)).status()
    assert missing.degraded_reason.value == "canonical_unavailable"


@pytest.mark.parametrize("contents", ("empty_sqlite", "malformed_bytes"))
def test_status_classifies_opened_canonical_schema_damage_as_invalid(
    tmp_path: Path, contents: str
) -> None:
    repo = tmp_path / contents
    canonical_dir = repo / ".codex-memory" / "db"
    canonical_dir.mkdir(parents=True)
    canonical = canonical_dir / "memory.sqlite3"
    if contents == "empty_sqlite":
        with sqlite3.connect(canonical):
            pass
    else:
        canonical.write_bytes(b"not sqlite")

    result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_invalid"


def test_process_rebuild_lock_serializes_independent_processes_and_retains_lock_file(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    first_entered, second_entered, third_entered = context.Event(), context.Event(), context.Event()
    release_first, release_second, release_third, overlap = context.Event(), context.Event(), context.Event(), context.Event()
    active = context.Value("i", 0)
    first = context.Process(target=_hold_rebuild_lock, args=(str(temp_dir), first_entered, release_first, active, overlap))
    second = context.Process(target=_hold_rebuild_lock, args=(str(temp_dir), second_entered, release_second, active, overlap))
    third = context.Process(target=_hold_rebuild_lock, args=(str(temp_dir), third_entered, release_third, active, overlap))
    first.start()
    assert first_entered.wait(5)
    second.start()
    third.start()
    time.sleep(0.2)
    assert not second_entered.is_set()
    assert not third_entered.is_set()
    release_first.set()
    for _ in range(50):
        if second_entered.is_set() or third_entered.is_set():
            break
        time.sleep(0.05)
    assert second_entered.is_set() or third_entered.is_set()
    assert active.value == 1
    assert not overlap.is_set()
    release_second.set()
    release_third.set()
    assert second_entered.wait(5)
    assert third_entered.wait(5)
    first.join(5)
    second.join(5)
    third.join(5)
    assert first.exitcode == 0
    assert second.exitcode == 0
    assert third.exitcode == 0
    assert [path.name for path in temp_dir.iterdir()] == ["rebuild.lock"]


def test_process_rebuild_lock_one_millisecond_timeout_does_not_sleep_fifty_ms(
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    active = context.Value("i", 0)
    overlap = context.Event()
    holder = context.Process(
        target=_hold_rebuild_lock,
        args=(str(temp_dir), entered, release, active, overlap),
    )
    holder.start()
    assert entered.wait(5)

    started = time.monotonic()
    try:
        with pytest.raises(retrieval_service.CacheCorruptError):
            with retrieval_service.process_rebuild_lock(temp_dir, timeout_ms=1):
                pass
    finally:
        elapsed = time.monotonic() - started
        release.set()
        holder.join(5)

    assert holder.exitcode == 0
    assert elapsed < 0.04


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_process_rebuild_lock_rejects_preplanted_nonowned_lock_path_without_touching_external_file(
    tmp_path: Path, link_kind: str
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    external = tmp_path / "external.lock"
    external.write_bytes(b"external content")
    planted = temp_dir / "rebuild.lock"
    if link_kind == "symlink":
        try:
            planted.symlink_to(external)
        except OSError:
            pytest.skip("symlinks unavailable")
    else:
        os.link(external, planted)

    with pytest.raises(Exception):
        with retrieval_service.process_rebuild_lock(temp_dir):
            pass

    assert external.read_bytes() == b"external content"
    assert planted.exists() or planted.is_symlink()


def test_process_rebuild_lock_does_not_remove_same_token_replacement_after_rename(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    lock_path = temp_dir / "rebuild.lock"
    replacement = temp_dir / "saved-lock"
    with retrieval_service.process_rebuild_lock(temp_dir):
        pass
    token = lock_path.read_text(encoding="ascii")
    lock_path.rename(replacement)
    lock_path.write_text(token, encoding="ascii")
    with retrieval_service.process_rebuild_lock(temp_dir):
        pass

    assert replacement.read_text(encoding="ascii") == token
    assert lock_path.read_text(encoding="ascii") == token


def test_owned_temporary_cleanup_does_not_delete_a_foreign_replacement(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    service = RetrievalService(RetrievalPaths.for_repo(repo))
    owned = service._owned_temporary_path()
    original = owned.path
    moved = original.with_name("moved.sqlite3")
    original.rename(moved)
    original.write_bytes(b"foreign replacement")

    service._remove_owned_temporary(owned)

    assert moved.exists()
    assert original.read_bytes() == b"foreign replacement"


def test_rebuild_rechecks_owned_temporary_identity_before_replace(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    service.rebuild()
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "swap-at-replace")
    connection.close()
    moved: list[Path] = []

    def inject(step: str) -> None:
        if step == "replace":
            temporary = next(paths.temp_dir.glob("rebuild-*.sqlite3"))
            moved_path = temporary.with_name("verified-temp.sqlite3")
            temporary.rename(moved_path)
            temporary.write_bytes(b"foreign replacement")
            moved.append(moved_path)

    with pytest.raises(Exception, match="temporary"):
        service.rebuild(failure_injector=inject)

    assert paths.database_path.read_bytes() == previous
    assert paths.temp_dir.joinpath(next(path.name for path in paths.temp_dir.glob("rebuild-*.sqlite3"))).read_bytes() == b"foreign replacement"
    assert moved[0].exists()


def test_rebuild_installs_verified_object_when_temp_path_is_swapped_at_replace_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "swap-during-replace")
    connection.close()
    actual_replace = retrieval_service.os.replace

    def swap_temp_then_replace(source: Path | str, destination: Path | str) -> None:
        temporary = next(paths.temp_dir.glob("rebuild-*.sqlite3"))
        moved = temporary.with_name("verified-before-final-install.sqlite3")
        temporary.rename(moved)
        temporary.write_bytes(b"foreign replacement")
        actual_replace(source, destination)

    monkeypatch.setattr(retrieval_service.os, "replace", swap_temp_then_replace)

    result = service.rebuild()

    assert result.status is RetrievalStatus.CURRENT
    assert paths.database_path.read_bytes() != previous
    foreign = next(paths.temp_dir.glob("rebuild-*.sqlite3"))
    assert foreign.read_bytes() == b"foreign replacement"


def test_rebuild_restores_previous_cache_when_install_handoff_is_swapped_at_replace_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "swap-install-source")
    connection.close()
    actual_replace = retrieval_service.os.replace
    preserved_install: list[Path] = []

    def swap_install_then_replace(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        if source_path.name.startswith("install-"):
            moved = source_path.with_name("verified-install-source.sqlite3")
            source_path.rename(moved)
            source_path.write_bytes(b"foreign install source")
            preserved_install.append(moved)
        actual_replace(source, destination)

    monkeypatch.setattr(retrieval_service.os, "replace", swap_install_then_replace)

    with pytest.raises(retrieval_service.CacheCorruptError, match="install"):
        service.rebuild()

    assert paths.database_path.read_bytes() == previous
    assert preserved_install[0].exists()


def test_install_source_swap_with_unavailable_rollback_never_leaves_foreign_cache_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "install-source-race-with-blocked-rollback")
    connection.close()
    actual_replace = retrieval_service.os.replace
    foreign = b"foreign install source"
    preserved_install: list[Path] = []

    def race_install_and_block_rollback(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        if source_path.name.startswith("install-"):
            moved = source_path.with_name("verified-install-race-guard.sqlite3")
            source_path.rename(moved)
            source_path.write_bytes(foreign)
            preserved_install.append(moved)
        elif source_path.name.startswith("rollback-"):
            raise PermissionError("rollback source unavailable")
        actual_replace(source, destination)

    monkeypatch.setattr(retrieval_service.os, "replace", race_install_and_block_rollback)

    with pytest.raises(retrieval_service.CacheCorruptError, match="install"):
        service.rebuild()

    assert not paths.database_path.exists() or paths.database_path.read_bytes() != foreign
    assert preserved_install[0].exists()
    assert any(
        candidate.read_bytes() == previous
        for candidate in paths.temp_dir.glob("rollback-*.sqlite3")
    )
    assert any(
        candidate.read_bytes() == previous
        for candidate in paths.temp_dir.glob("preserve-*.sqlite3")
    )


def test_rollback_source_swap_recovers_from_an_independent_verified_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "rollback-source-race")
    connection.close()
    actual_replace = retrieval_service.os.replace
    foreign = b"foreign rollback source"
    preserved_rollback: list[Path] = []

    def swap_rollback_then_replace(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        if source_path.name.startswith("rollback-"):
            moved = source_path.with_name("verified-rollback-race-copy.sqlite3")
            source_path.rename(moved)
            source_path.write_bytes(foreign)
            preserved_rollback.append(moved)
        actual_replace(source, destination)

    monkeypatch.setattr(retrieval_service.os, "replace", swap_rollback_then_replace)

    def force_post_install_failure(step: str) -> None:
        if step == "replace":
            monkeypatch.setattr(
                service,
                "status",
                lambda: retrieval_service._empty_result(
                    RetrievalStatus.CORRUPT,
                    retrieval_service.DegradedReason.INDEX_CORRUPT,
                ),
            )

    with pytest.raises(
        retrieval_service.CacheCorruptError,
        match="installed retrieval cache did not verify",
    ):
        service.rebuild(failure_injector=force_post_install_failure)

    assert paths.database_path.read_bytes() == previous
    assert preserved_rollback[0].exists()
    assert preserved_rollback[0].read_bytes() == previous


def test_rebuild_restores_absent_cache_when_install_handoff_is_swapped_at_replace_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    actual_replace = retrieval_service.os.replace
    preserved_install: list[Path] = []
    foreign = paths.temp_dir / "foreign-install-source.sqlite3"

    def swap_install_then_replace(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        if source_path.name.startswith("install-"):
            moved = source_path.with_name("verified-absent-install-source.sqlite3")
            source_path.rename(moved)
            foreign.write_bytes(b"foreign install source")
            os.link(foreign, source_path)
            preserved_install.append(moved)
        actual_replace(source, destination)

    monkeypatch.setattr(retrieval_service.os, "replace", swap_install_then_replace)

    with pytest.raises(retrieval_service.CacheCorruptError, match="install"):
        service.rebuild()

    assert not paths.database_path.exists()
    assert foreign.read_bytes() == b"foreign install source"
    assert preserved_install[0].exists()


def test_successful_rebuilds_remove_rollback_handoffs_after_link_count_changes(tmp_path: Path) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    for key in ("rollback-one", "rollback-two"):
        connection = database.connect_database(paths.canonical_database_path)
        _propose(connection, repo, key)
        connection.close()

        assert service.rebuild().status is RetrievalStatus.CURRENT
        assert [path.name for path in paths.temp_dir.iterdir()] == ["rebuild.lock"]


def test_absent_cache_status_failure_preserves_original_install_error_and_restores_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)

    def inject(step: str) -> None:
        if step == "replace":
            monkeypatch.setattr(
                service,
                "status",
                lambda: retrieval_service._empty_result(
                    RetrievalStatus.CORRUPT,
                    retrieval_service.DegradedReason.INDEX_CORRUPT,
                ),
            )

    with pytest.raises(retrieval_service.CacheCorruptError, match="installed retrieval cache"):
        service.rebuild(failure_injector=inject)

    assert not paths.database_path.exists()


def test_restore_oserror_preserves_original_install_error_and_previous_cache_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "restore-permission-error")
    connection.close()
    actual_replace = retrieval_service.os.replace

    def fail_restore(source: Path | str, destination: Path | str) -> None:
        if Path(source).name.startswith("rollback-") and Path(destination) == paths.database_path:
            raise PermissionError("restore denied")
        actual_replace(source, destination)

    monkeypatch.setattr(retrieval_service.os, "replace", fail_restore)

    def inject(step: str) -> None:
        if step == "replace":
            monkeypatch.setattr(
                service,
                "status",
                lambda: retrieval_service._empty_result(
                    RetrievalStatus.CORRUPT,
                    retrieval_service.DegradedReason.INDEX_CORRUPT,
                ),
            )

    with pytest.raises(
        retrieval_service.CacheCorruptError,
        match="installed retrieval cache did not verify",
    ):
        service.rebuild(failure_injector=inject)

    backups = list(paths.temp_dir.glob("rollback-*.sqlite3"))
    assert paths.database_path.exists()
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous


def test_nonregular_install_target_preserves_old_cache_backup_without_masking_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    previous = paths.database_path.read_bytes()
    connection = database.connect_database(paths.canonical_database_path)
    _propose(connection, repo, "nonregular-install-target")
    connection.close()
    actual_replace = retrieval_service.os.replace

    def replace_install_with_directory(source: Path | str, destination: Path | str) -> None:
        actual_replace(source, destination)
        if Path(source).name.startswith("install-") and Path(destination) == paths.database_path:
            paths.database_path.unlink()
            paths.database_path.mkdir()

    monkeypatch.setattr(retrieval_service.os, "replace", replace_install_with_directory)

    with pytest.raises(retrieval_service.CacheCorruptError, match="unsafe"):
        service.rebuild()

    backups = list(paths.temp_dir.glob("rollback-*.sqlite3"))
    assert paths.database_path.is_dir()
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous


def test_process_rebuild_lock_rejects_path_swap_between_validation_and_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    lock_path = temp_dir / "rebuild.lock"
    lock_path.write_bytes(retrieval_service._LOCK_MAGIC)
    actual_open = retrieval_service.os.open
    swapped = False

    def swap_before_open(path: Path | str, flags: int, *args: object) -> int:
        nonlocal swapped
        if Path(path) == lock_path and not swapped and not flags & os.O_EXCL:
            swapped = True
            lock_path.rename(temp_dir / "original-lock")
            lock_path.write_bytes(retrieval_service._LOCK_MAGIC)
        return actual_open(path, flags, *args)

    monkeypatch.setattr(retrieval_service.os, "open", swap_before_open)

    with pytest.raises(Exception, match="unsafe"):
        with retrieval_service.process_rebuild_lock(temp_dir):
            pass


def test_process_rebuild_lock_rejects_hardlinked_existing_marker_without_touching_external_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    external = tmp_path / "external-owner"
    external.write_text("codex-memory-retrieval-lock-v1\n", encoding="ascii")
    os.link(external, temp_dir / "rebuild.lock")

    with pytest.raises(Exception, match="unsafe"):
        with retrieval_service.process_rebuild_lock(temp_dir):
            pass

    assert external.read_text(encoding="ascii") == "codex-memory-retrieval-lock-v1\n"
    assert (temp_dir / "rebuild.lock").exists()


def test_status_bounds_retrieval_path_oserror_as_path_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    service = RetrievalService(RetrievalPaths.for_repo(repo))
    monkeypatch.setattr(service, "_validate_retrieval_paths", lambda: (_ for _ in ()).throw(OSError("denied")))

    result = service.status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "path_unavailable"


def test_status_bounds_readonly_cache_open_failure_as_path_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    service = RetrievalService(RetrievalPaths.for_repo(repo))
    assert service.rebuild().status is RetrievalStatus.CURRENT
    monkeypatch.setattr(retrieval_service, "connect_cache_database", lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("denied")))

    result = service.status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "path_unavailable"


def test_status_bounds_active_generation_sqlite_failure_as_path_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    service = RetrievalService(RetrievalPaths.for_repo(repo))
    assert service.rebuild().status is RetrievalStatus.CURRENT
    monkeypatch.setattr(retrieval_service, "_active_generation", lambda connection: (_ for _ in ()).throw(sqlite3.OperationalError("read failed")))

    result = service.status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "path_unavailable"


def test_status_bounds_database_exists_oserror_as_path_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    snapshot = capture_canonical_snapshot(paths)
    monkeypatch.setattr(service, "_snapshot_or_none", lambda: (snapshot, None))
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: (_ for _ in ()).throw(OSError("denied")) if path == paths.database_path else original_exists(path))

    result = service.status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "path_unavailable"


def test_status_bounds_canonical_exists_oserror_as_canonical_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: (_ for _ in ()).throw(OSError("denied")) if path == paths.canonical_database_path else original_exists(path))

    result = RetrievalService(paths).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_unavailable"


def test_status_bounds_canonical_operational_error_as_canonical_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    monkeypatch.setattr(retrieval_service, "capture_canonical_snapshot", lambda paths: (_ for _ in ()).throw(sqlite3.OperationalError("denied")))

    result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_unavailable"


def test_status_classifies_wrapped_canonical_operational_error_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()

    def wrapped(_paths: RetrievalPaths):
        try:
            raise sqlite3.OperationalError("denied")
        except sqlite3.OperationalError as cause:
            raise VerificationError("canonical snapshot is unavailable") from cause

    monkeypatch.setattr(retrieval_service, "capture_canonical_snapshot", wrapped)
    result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert result.degraded_reason.value == "canonical_unavailable"


def test_status_classifies_direct_and_wrapped_canonical_oserror_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()

    monkeypatch.setattr(
        retrieval_service,
        "capture_canonical_snapshot",
        lambda paths: (_ for _ in ()).throw(OSError("denied")),
    )
    direct = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    def wrapped(_paths: RetrievalPaths):
        try:
            raise OSError("denied")
        except OSError as cause:
            raise VerificationError("canonical snapshot is unavailable") from cause

    monkeypatch.setattr(retrieval_service, "capture_canonical_snapshot", wrapped)
    wrapped_result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert direct.degraded_reason.value == "canonical_unavailable"
    assert wrapped_result.degraded_reason.value == "canonical_unavailable"


@pytest.mark.parametrize("error_type", [sqlite3.DatabaseError, sqlite3.OperationalError, OSError])
def test_status_classifies_direct_and_wrapped_canonical_io_errors_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[BaseException]
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()

    monkeypatch.setattr(
        retrieval_service,
        "capture_canonical_snapshot",
        lambda paths: (_ for _ in ()).throw(error_type("denied")),
    )
    direct = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    def wrapped(_paths: RetrievalPaths):
        try:
            raise error_type("denied")
        except error_type as cause:
            raise VerificationError("canonical snapshot is unavailable") from cause

    monkeypatch.setattr(retrieval_service, "capture_canonical_snapshot", wrapped)
    wrapped_result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert direct.status is RetrievalStatus.CORRUPT
    assert direct.degraded_reason.value == "canonical_unavailable"
    assert wrapped_result.status is RetrievalStatus.CORRUPT
    assert wrapped_result.degraded_reason.value == "canonical_unavailable"


def test_status_keeps_opened_canonical_ledger_validation_failure_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    monkeypatch.setattr(
        retrieval_service,
        "capture_canonical_snapshot",
        lambda paths: (_ for _ in ()).throw(VerificationError("ledger content is invalid")),
    )

    result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_invalid"


def test_status_validates_dangling_canonical_redirect_before_classifying_it_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    canonical_dir = repo / ".codex-memory" / "db"
    canonical_dir.mkdir(parents=True)
    canonical = canonical_dir / "memory.sqlite3"
    try:
        canonical.symlink_to(tmp_path / "missing-target.sqlite3")
    except OSError:
        pytest.skip("symlinks unavailable")

    result = RetrievalService(RetrievalPaths.for_repo(repo)).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_invalid"


def _structured_sqlite_error(code: int) -> sqlite3.Error:
    error = sqlite3.OperationalError("injected canonical read failure")
    error.sqlite_errorcode = code
    error.sqlite_errorname = f"SQLITE_{code}"
    return error


def _classify_canonical_failure(error: BaseException) -> str:
    classifier = getattr(canonical_snapshot, "_classify_canonical_failure", None)
    assert callable(classifier), "canonical failure classifier is required"
    result = classifier(error)
    value = getattr(result, "value", result)
    assert isinstance(value, str)
    return value


@pytest.mark.parametrize(
    "code",
    sorted(
        {
            sqlite3.SQLITE_IOERR,
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
            sqlite3.SQLITE_CANTOPEN,
            sqlite3.SQLITE_READONLY,
            sqlite3.SQLITE_FULL,
            sqlite3.SQLITE_PERM,
            sqlite3.SQLITE_AUTH,
            sqlite3.SQLITE_PROTOCOL,
            sqlite3.SQLITE_INTERRUPT,
            sqlite3.SQLITE_NOMEM,
        }
    ),
)
def test_canonical_failure_classifier_marks_environmental_sqlite_codes_unavailable(
    code: int,
) -> None:
    assert _classify_canonical_failure(_structured_sqlite_error(code)) == "unavailable"


@pytest.mark.parametrize("code", sorted({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}))
def test_canonical_failure_classifier_marks_proven_sqlite_damage_invalid(code: int) -> None:
    assert _classify_canonical_failure(_structured_sqlite_error(code)) == "invalid"


def test_canonical_failure_classifier_reduces_extended_environmental_code() -> None:
    extended = _structured_sqlite_error(sqlite3.SQLITE_IOERR | (7 << 8))

    assert _classify_canonical_failure(extended) == "unavailable"


def test_canonical_failure_classifier_keeps_generic_sqlite_error_unavailable() -> None:
    assert _classify_canonical_failure(_structured_sqlite_error(sqlite3.SQLITE_ERROR)) == "unavailable"
    semantic = VerificationError("schema metadata is missing")
    semantic.__cause__ = _structured_sqlite_error(sqlite3.SQLITE_ERROR)

    assert _classify_canonical_failure(semantic) == "invalid"


def test_canonical_failure_classifier_handles_wrapped_and_unstructured_errors() -> None:
    cause = _structured_sqlite_error(sqlite3.SQLITE_IOERR)
    try:
        raise cause
    except sqlite3.Error as original:
        wrapped = VerificationError("ledger verification failed")
        wrapped.__cause__ = original

    assert _classify_canonical_failure(wrapped) == "unavailable"
    raw_sqlite = sqlite3.OperationalError("read failed without structured code")
    wrapped_raw_sqlite = VerificationError("ledger verification failed")
    wrapped_raw_sqlite.__cause__ = raw_sqlite
    assert _classify_canonical_failure(wrapped_raw_sqlite) == "unavailable"
    assert _classify_canonical_failure(VerificationError("schema mismatch")) == "invalid"
    assert _classify_canonical_failure(sqlite3.OperationalError("database is locked")) == "unavailable"
    assert _classify_canonical_failure(OSError("read failed")) == "unavailable"


class _FailingCanonicalConnection:
    def __init__(self, connection: sqlite3.Connection, needle: str, error: sqlite3.Error):
        self._connection = connection
        self._needle = needle
        self.error = error

    def execute(self, sql: str, parameters: object = ()):
        if self._needle in sql:
            raise self.error
        return self._connection.execute(sql, parameters)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


@pytest.mark.parametrize(
    ("needle", "stage"),
    [
        ("SELECT value FROM schema_meta", "schema_query"),
        ("SELECT type, name, sql FROM sqlite_master", "ddl_read"),
    ],
)
def test_post_open_canonical_read_io_fails_unavailable_through_real_snapshot_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    needle: str,
    stage: str,
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    injected: list[sqlite3.Error] = []
    original_connect = canonical_snapshot.connect_readonly_database

    def open_proxy(path: Path) -> _FailingCanonicalConnection:
        error = _structured_sqlite_error(sqlite3.SQLITE_IOERR)
        injected.append(error)
        return _FailingCanonicalConnection(original_connect(path), needle, error)

    monkeypatch.setattr(canonical_snapshot, "connect_readonly_database", open_proxy)

    with pytest.raises(canonical_snapshot.CanonicalSnapshotUnavailableError) as raised:
        capture_canonical_snapshot(paths)
    diagnostic = raised.value
    stage_value = getattr(getattr(diagnostic, "stage", None), "value", None)
    assert stage_value == stage
    assert injected
    cause = diagnostic
    chain: list[BaseException] = []
    while cause is not None and cause not in chain:
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    assert injected[0] in chain

    result = RetrievalService(paths).status()
    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_unavailable"


@pytest.mark.parametrize("operation", ["status", "verify", "init", "rebuild"])
def test_canonical_unavailable_is_returned_without_mutation_by_all_service_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    before = paths.canonical_database_path.read_bytes()
    before_rows = _canonical_logical_rows(paths.canonical_database_path)
    original_connect = canonical_snapshot.connect_readonly_database

    def open_proxy(path: Path) -> _FailingCanonicalConnection:
        return _FailingCanonicalConnection(
            original_connect(path),
            "SELECT value FROM schema_meta",
            _structured_sqlite_error(sqlite3.SQLITE_IOERR),
        )

    monkeypatch.setattr(canonical_snapshot, "connect_readonly_database", open_proxy)
    result = getattr(RetrievalService(paths), operation)()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_unavailable"
    assert result.generation is None
    assert paths.canonical_database_path.read_bytes() == before
    assert _canonical_logical_rows(paths.canonical_database_path) == before_rows


def test_rebuild_preserves_typed_canonical_failure_if_snapshot_changes_after_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    absent = retrieval_service._empty_result(
        RetrievalStatus.ABSENT, retrieval_service.DegradedReason.INDEX_ABSENT
    )
    monkeypatch.setattr(service, "status", lambda: absent)
    unavailable = canonical_snapshot.CanonicalSnapshotUnavailableError(
        "not exposed", stage=canonical_snapshot.CanonicalValidationStage.SCHEMA_QUERY
    )

    def fail_snapshot() -> object:
        raise unavailable

    monkeypatch.setattr(service, "_capture_for_rebuild", fail_snapshot)

    result = service.rebuild()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason is retrieval_service.DegradedReason.CANONICAL_UNAVAILABLE


def test_process_rebuild_lock_rejects_deterministic_reparse_leaf_without_touching_external_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    lock_path = temp_dir / "rebuild.lock"
    external = tmp_path / "external-owner"
    external.write_text("codex-memory-retrieval-lock-v1\n", encoding="ascii")

    def pretend_redirect(path: Path) -> bool:
        return Path(path) == lock_path

    monkeypatch.setattr(runtime_path_module, "path_is_redirect", pretend_redirect)
    monkeypatch.setattr(retrieval_service, "path_is_redirect", pretend_redirect)

    with pytest.raises(Exception, match="unsafe|redirect"):
        with retrieval_service.process_rebuild_lock(temp_dir):
            pass

    assert external.read_text(encoding="ascii") == "codex-memory-retrieval-lock-v1\n"
    assert not lock_path.exists()


def test_status_rejects_deterministic_reparse_canonical_leaf_as_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, connection = _canonical_database(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    canonical = paths.canonical_database_path

    monkeypatch.setattr(
        runtime_path_module,
        "path_is_redirect",
        lambda path: Path(path) == canonical,
    )
    monkeypatch.setattr(retrieval_service, "path_is_redirect", lambda _path: False)

    result = RetrievalService(paths).status()

    assert result.status is RetrievalStatus.CORRUPT
    assert result.degraded_reason.value == "canonical_invalid"


def test_catch_up_failure_before_activation_preserves_the_prior_cache_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    """Would fail if catch-up advanced active state before verified installation."""
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "stable")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    cache_before = paths.database_path.read_bytes()
    canonical_before = paths.canonical_database_path.read_bytes()
    _propose(connection, repo, "new")

    with pytest.raises(RuntimeError, match="injected failure"):
        service.catch_up(failure_injector=lambda stage: (_ for _ in ()).throw(RuntimeError("injected failure")) if stage == "projection" else None)

    assert paths.database_path.read_bytes() == cache_before
    assert paths.canonical_database_path.read_bytes() == canonical_before


def test_catch_up_rejects_a_missing_unaffected_cache_row_before_activation(tmp_path: Path) -> None:
    """Would fail if catch-up advertised a full snapshot without copied rows."""
    repo, connection = _canonical_database(tmp_path)
    stable = _propose(connection, repo, "semantic-stable")
    changed = _propose(connection, repo, "semantic-changed")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    with sqlite3.connect(paths.database_path) as cache:
        from codex_memory.retrieval import cache_database

        document_id = cache.execute(
            "SELECT document_id FROM retrieval_documents WHERE record_id = ?", (stable.record_id,)
        ).fetchone()[0]
        for table in ("retrieval_symbols", "retrieval_document_sources", "retrieval_documents"):
            cache.execute(f"DROP TRIGGER {table}_reject_delete")
        cache.execute("DELETE FROM retrieval_documents_fts WHERE document_id = ?", (document_id,))
        cache.execute("DELETE FROM retrieval_symbols WHERE document_id = ?", (document_id,))
        cache.execute("DELETE FROM retrieval_document_sources WHERE document_id = ?", (document_id,))
        cache.execute("DELETE FROM retrieval_documents WHERE document_id = ?", (document_id,))
        for table in ("retrieval_symbols", "retrieval_document_sources", "retrieval_documents"):
            cache.execute(cache_database._append_only_trigger_statement(table, "delete"))
    cache.close()
    structured_ledger.apply_action(
        connection, record_id=changed.record_id, action="activate", idempotency_key="semantic-activate"
    )
    connection.close()

    with pytest.raises(retrieval_service.CacheCorruptError, match="projection"):
        service.catch_up()

    assert RetrievalService(paths).status().status is RetrievalStatus.STALE


def test_catch_up_over_budget_fails_closed_without_partial_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if production catch-up silently truncated an affected-ID scan."""
    repo, connection = _canonical_database(tmp_path)
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    baseline = paths.database_path.read_bytes()
    baseline_generation = service.status().generation
    assert baseline_generation is not None
    for number in range(3):
        _propose(connection, repo, f"over-budget-{number}")
    connection.close()
    monkeypatch.setattr(retrieval_service, "MAX_CATCH_UP_AFFECTED_RECORDS", 2)

    result = service.catch_up()

    assert result.status is RetrievalStatus.STALE
    assert result.degraded_reason.value == "index_stale"
    assert result.generation is not None
    assert result.generation.generation_id == baseline_generation.generation_id
    assert paths.database_path.read_bytes() == baseline


def test_catch_up_retries_when_the_canonical_highwater_changes_before_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if the highwater gate activated the first raced generation."""
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "retry-baseline")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    _propose(connection, repo, "retry-change")
    calls = 0
    original = service._capture_for_rebuild

    def raced_snapshot():
        nonlocal calls
        calls += 1
        snapshot = original()
        return replace(snapshot, content_fingerprint="f" * 64) if calls == 2 else snapshot

    monkeypatch.setattr(service, "_capture_for_rebuild", raced_snapshot)
    connection.close()

    result = service.catch_up()

    assert calls == 4
    assert result.status is RetrievalStatus.CURRENT
    assert result.generation is not None
    assert result.generation.indexed_highwater == result.canonical_highwater


def test_catch_up_second_highwater_race_keeps_the_old_generation_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if the second final-gate race installed a stale temporary cache."""
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "retry-failure-base")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    original_generation = service.status().generation
    assert original_generation is not None
    original_cache = paths.database_path.read_bytes()
    _propose(connection, repo, "retry-failure-change")
    calls = 0
    original = service._capture_for_rebuild

    def raced_snapshot():
        nonlocal calls
        calls += 1
        snapshot = original()
        return replace(snapshot, content_fingerprint="e" * 64) if calls in {2, 4} else snapshot

    monkeypatch.setattr(service, "_capture_for_rebuild", raced_snapshot)
    connection.close()

    result = service.catch_up()

    assert calls == 4
    assert result.status is RetrievalStatus.STALE
    assert result.degraded_reason.value == "index_stale"
    assert result.generation is None
    assert paths.database_path.read_bytes() == original_cache
    assert RetrievalService(paths).status().generation == original_generation


def test_catch_up_is_idempotent_when_the_index_is_current(tmp_path: Path) -> None:
    """Would fail if a no-change catch-up created another generation."""
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "idempotent")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    first = service.rebuild()
    before = paths.database_path.read_bytes()
    connection.close()

    second = service.catch_up()

    assert second == first
    assert paths.database_path.read_bytes() == before


def test_concurrent_catch_up_and_rebuild_serialize_through_production_operations(tmp_path: Path) -> None:
    """Would fail if either public operation bypassed the shared rebuild lock."""
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "concurrent-base")
    paths = RetrievalPaths.for_repo(repo)
    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT
    _propose(connection, repo, "concurrent-change")
    connection.close()
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    results = context.Queue()
    catch_up = context.Process(target=_run_retrieval_operation, args=(str(repo), "catch_up", release, results))
    rebuild = context.Process(target=_run_retrieval_operation, args=(str(repo), "rebuild", release, results))
    catch_up.start()
    rebuild.start()
    release.set()
    catch_up.join(20)
    rebuild.join(20)

    assert (catch_up.exitcode, rebuild.exitcode) == (0, 0)
    assert {results.get(timeout=2), results.get(timeout=2)} <= {"current", "stale"}
    final = RetrievalService(paths).status()
    assert final.status is RetrievalStatus.CURRENT
    assert final.generation is not None
    assert final.generation.indexed_highwater == final.canonical_highwater


def test_concurrent_rebuild_precedes_waiting_catch_up_at_public_operation_barrier(
    tmp_path: Path,
) -> None:
    """Would fail if rebuild entered its critical section while catch-up held it."""
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "ordered-concurrent-base")
    paths = RetrievalPaths.for_repo(repo)
    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT
    _propose(connection, repo, "ordered-concurrent-change")
    connection.close()
    context = multiprocessing.get_context("spawn")
    rebuild_entered = context.Event()
    rebuild_release = context.Event()
    catch_start = context.Event()
    order = context.Queue()
    rebuild_results = context.Queue()
    catch_results = context.Queue()
    rebuild = context.Process(
        target=_run_retrieval_operation_at_barrier,
        args=(str(repo), "rebuild", "rebuild", rebuild_entered, rebuild_release, order, rebuild_results),
    )
    catch_up = context.Process(
        target=_run_retrieval_operation,
        args=(str(repo), "catch_up", catch_start, catch_results),
    )

    rebuild.start()
    assert rebuild_entered.wait(10)
    catch_start.set()
    catch_up.start()
    time.sleep(0.5)
    assert catch_up.is_alive()
    rebuild_release.set()
    rebuild.join(20)
    catch_up.join(20)

    assert (rebuild.exitcode, catch_up.exitcode) == (0, 0)
    assert order.get(timeout=2) == "rebuild:ddl"
    assert rebuild_results.get(timeout=2) == ("rebuild", "current")
    assert catch_results.get(timeout=2) == "current"
    final = RetrievalService(paths).status()
    assert final.status is RetrievalStatus.CURRENT
    assert final.indexed_highwater == final.canonical_highwater


def test_catch_up_copies_unaffected_rows_in_configured_bounded_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if catch-up buffered every existing cache row before copying."""
    repo, connection = _canonical_database(tmp_path)
    for number in range(4):
        _propose(connection, repo, f"batch-base-{number}")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    _propose(connection, repo, "batch-change")
    connection.close()
    observed: list[int] = []
    original = retrieval_service._batches

    def observe(cursor):
        for rows in original(cursor):
            observed.append(len(rows))
            yield rows

    monkeypatch.setattr(retrieval_service, "CATCH_UP_COPY_BATCH_SIZE", 1)
    monkeypatch.setattr(retrieval_service, "_batches", observe)

    assert service.catch_up().status is RetrievalStatus.CURRENT
    assert observed and set(observed) == {1}


def test_catch_up_streams_canonical_projection_and_final_validation_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if production catch-up materialized a full projection or row sets."""
    repo, connection = _canonical_database(tmp_path)
    for number in range(4):
        _propose(connection, repo, f"stream-base-{number}")
    paths = RetrievalPaths.for_repo(repo)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    _propose(connection, repo, "stream-change")
    connection.close()
    observed: list[int] = []
    original_batches = retrieval_service._batches

    def observe(cursor):
        for rows in original_batches(cursor):
            observed.append(len(rows))
            yield rows

    def forbid_materialization(*_args, **_kwargs):
        raise AssertionError("catch-up called the full projection materializer")

    monkeypatch.setattr(retrieval_service, "CATCH_UP_COPY_BATCH_SIZE", 1)
    monkeypatch.setattr(retrieval_service, "_batches", observe)
    monkeypatch.setattr(retrieval_service, "project_full_snapshot", forbid_materialization)

    assert service.catch_up().status is RetrievalStatus.CURRENT
    # Copying alone visits 12 rows (four documents and their eight symbols).
    # The larger count proves final document/symbol/FTS equivalence also used
    # the observed one-row cursor batches.
    assert sum(observed) >= 30
    assert set(observed) == {1}


def test_public_catch_up_matches_clean_rebuild_across_lifecycle_dependencies(
    tmp_path: Path,
) -> None:
    """Would fail if any canonical dependency installed a partial/stale generation."""
    repo, connection = _canonical_database(tmp_path)
    activation = _propose(connection, repo, "matrix-activation")
    invalidation = _propose(connection, repo, "matrix-invalidation", evidence=True)
    closable = _propose(connection, repo, "matrix-close-reopen")
    relation_source = _propose(connection, repo, "matrix-relation-source")
    relation_target = _propose(connection, repo, "matrix-relation-target")
    for number, record in enumerate((invalidation, closable, relation_source, relation_target)):
        structured_ledger.apply_action(
            connection,
            record_id=record.record_id,
            action="activate",
            idempotency_key=f"matrix-initial-activate-{number}",
        )
    constitution = structured_ledger.propose_record(
        connection,
        record_type="constitution",
        scope_kind="project",
        scope_key=str(repo),
        topic_key="matrix.constitution",
        statement="Preserve verified public retrieval lifecycle behavior.",
        idempotency_key="matrix-constitution",
    )
    structured_ledger.apply_action(
        connection,
        record_id=constitution.record_id,
        action="activate",
        idempotency_key="matrix-constitution-activate",
    )
    blobs = BlobStore(RuntimePaths.for_repo(repo).blobs_dir)
    prompt = EventStore(connection, blobs).append(parse_hook_payload({
        "hook_event_name": "UserPromptSubmit",
        "session_id": "matrix-session",
        "turn_id": "matrix-turn",
        "cwd": str(repo),
        "prompt": "Preserve verified public retrieval lifecycle behavior.",
    }))
    prompt_uri = event_uri("matrix-session", prompt.event_id)
    paths = RetrievalPaths.for_repo(repo)
    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT

    evidence_since = capture_canonical_snapshot(paths).highwater
    candidate = capture_candidate(
        connection,
        blobs,
        project_scope_key=str(repo),
        record_type="constitution",
        scope_kind="project",
        scope_key=str(repo),
        topic_key="matrix.constitution",
        statement="Preserve verified public retrieval lifecycle behavior.",
        session_id="matrix-session",
        turn_id="matrix-turn",
        context_id="startup",
        idempotency_key="matrix-constitution-candidate",
        evidence_uris=(prompt_uri,),
    )
    evidence_after = capture_canonical_snapshot(paths).highwater
    assert evidence_after.record_seq == evidence_since.record_seq
    assert evidence_after.memory_event_seq == evidence_since.memory_event_seq
    assert evidence_after.candidate_seq > evidence_since.candidate_seq
    assert evidence_after.candidate_evidence_seq > evidence_since.candidate_evidence_seq
    assert project_stale_delta(connection, since=evidence_since, limit=8).record_ids == ()
    _assert_catch_up_matches_clean_rebuild(paths)

    structured_ledger.apply_action(
        connection, record_id=activation.record_id, action="activate",
        idempotency_key="matrix-activate",
    )
    _assert_catch_up_matches_clean_rebuild(paths)

    invalidation_since = capture_canonical_snapshot(paths).highwater
    structured_ledger.apply_action(
        connection, record_id=invalidation.record_id, action="invalidate",
        idempotency_key="matrix-invalidate",
    )
    assert invalidation.record_id not in project_stale_delta(
        connection, since=invalidation_since, limit=8
    ).record_ids
    _assert_catch_up_matches_clean_rebuild(paths)

    close_since = capture_canonical_snapshot(paths).highwater
    structured_ledger.apply_action(
        connection, record_id=closable.record_id, action="close",
        idempotency_key="matrix-close",
    )
    assert closable.record_id not in project_stale_delta(
        connection, since=close_since, limit=8
    ).record_ids
    _assert_catch_up_matches_clean_rebuild(paths)
    with closing(sqlite3.connect(paths.database_path)) as cache:
        # Every candidate surface applies the terminal-state gate before a row
        # can be handed to a later scorer; the immutable document may remain.
        assert cache.execute(
            "SELECT COUNT(*) FROM retrieval_documents WHERE record_id = ? AND state IN ('proposed', 'active')",
            (closable.record_id,),
        ).fetchone()[0] == 0
        assert cache.execute(
            """SELECT COUNT(*) FROM retrieval_symbols AS s
            JOIN retrieval_documents AS d ON d.document_id = s.document_id
            WHERE d.record_id = ? AND d.state IN ('proposed', 'active')""",
            (closable.record_id,),
        ).fetchone()[0] == 0
        if cache.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
        ).fetchone():
            assert cache.execute(
                """SELECT COUNT(*) FROM retrieval_documents_fts AS f
                JOIN retrieval_documents AS d ON d.document_id = f.document_id
                WHERE d.record_id = ? AND d.state IN ('proposed', 'active')""",
                (closable.record_id,),
            ).fetchone()[0] == 0

    reopen_since = capture_canonical_snapshot(paths).highwater
    structured_ledger.apply_action(
        connection, record_id=closable.record_id, action="reopen",
        idempotency_key="matrix-reopen",
    )
    assert closable.record_id in project_stale_delta(
        connection, since=reopen_since, limit=8
    ).record_ids
    _assert_catch_up_matches_clean_rebuild(paths)

    relation_since = capture_canonical_snapshot(paths).highwater
    structured_ledger.apply_action(
        connection,
        record_id=relation_source.record_id,
        action="supersede",
        replacement_record_id=relation_target.record_id,
        idempotency_key="matrix-relation",
    )
    relation_delta = project_stale_delta(connection, since=relation_since, limit=8)
    assert relation_source.record_id not in relation_delta.record_ids
    assert relation_target.record_id in relation_delta.record_ids
    _assert_catch_up_matches_clean_rebuild(paths)
    with closing(sqlite3.connect(paths.database_path)) as cache:
        search_text = dict(cache.execute(
            "SELECT record_id, search_text FROM retrieval_documents WHERE record_id IN (?, ?)",
            (relation_source.record_id, relation_target.record_id),
        ))
    assert relation_target.record_id in search_text[relation_source.record_id]
    assert relation_source.record_id in search_text[relation_target.record_id]

    promotion_since = capture_canonical_snapshot(paths).highwater
    before_packet = memory_context.build_memory_packet(
        connection,
        blobs,
        session_id="matrix-observer",
        turn_id="matrix-observer-turn",
        context_id="startup",
        scope_key=str(repo),
        prompt_event_uri=None,
    )
    assert "UNVERIFIED CONSTITUTION" in before_packet
    memory_review.review_candidate(
        connection,
        blobs,
        candidate_id=candidate.candidate_id,
        outcome="promote",
        reason="Direct user review certifies the matching Constitution record.",
        idempotency_key="matrix-candidate-promote",
        evidence_uris=(prompt_uri,),
        supersedes_record_id=None,
    )
    promotion_after = capture_canonical_snapshot(paths).highwater
    assert promotion_after.record_seq == promotion_since.record_seq
    assert promotion_after.memory_event_seq == promotion_since.memory_event_seq
    assert promotion_after.candidate_event_seq > promotion_since.candidate_event_seq
    assert promotion_after.candidate_evidence_seq == promotion_since.candidate_evidence_seq
    assert constitution.record_id in project_stale_delta(
        connection, since=promotion_since, limit=8
    ).record_ids
    after_packet = memory_context.build_memory_packet(
        connection,
        blobs,
        session_id="matrix-observer",
        turn_id="matrix-observer-turn",
        context_id="startup",
        scope_key=str(repo),
        prompt_event_uri=None,
    )
    assert "CERTIFIED CONSTITUTION" in after_packet
    _assert_catch_up_matches_clean_rebuild(paths)
    connection.close()
