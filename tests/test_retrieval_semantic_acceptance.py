"""Fail-closed activation and read snapshots for historical semantic generations.

Only fixture creation bypasses the unavailable installation capability to model
an existing generation written by the prior release. Normal installation tests
remain unchanged and intentionally expose the unresolved functional blocker.
"""
from contextlib import closing
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import tracemalloc

import pytest

from codex_memory import structured_ledger
from codex_memory.retrieval import semantic_index, semantic_store, service as service_module
from codex_memory.retrieval.cache_database import read_semantic_history_for_rebuild
from codex_memory.retrieval.contracts import (
    CandidateChannel, DegradedReason, RetrievalRequest, RetrievalScope, RetrievalStatus,
)
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
from codex_memory.retrieval.service import RetrievalService, SemanticQueryInputs
from retrieval_provider_fakes import FakeProvider
from test_retrieval_semantic_index import _config, _active_semantic_metadata
from test_retrieval_semantic_store import _prepared_install_pair


def _historical_generation(runtime, monkeypatch):
    """Produce an isolated generation through the production acceptance path."""
    scope = str(runtime.paths.repo_root)
    record = structured_ledger.propose_record(
        runtime.connection, record_type="decision", scope_kind="project", scope_key=scope,
        topic_key="semantic-existing", statement="semantic existing target",
        idempotency_key="semantic-existing",
    )
    structured_ledger.apply_action(runtime.connection, record_id=record.record_id,
                                  action="activate", idempotency_key="semantic-existing-active")
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    provider = FakeProvider()
    assert semantic_index.rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT
    return paths, provider, record


def _semantic_state(paths):
    with closing(sqlite3.connect(paths.database_path)) as cache:
        cache.row_factory = sqlite3.Row
        return (
            read_semantic_history_for_rebuild(cache),
            tuple(tuple(row) for row in cache.execute("SELECT * FROM retrieval_semantic_state")),
            tuple(tuple(row) for row in cache.execute("SELECT * FROM retrieval_semantic_generation_current")),
        )


def _semantic_registry_state(paths):
    with closing(sqlite3.connect(paths.database_path)) as cache:
        return tuple(
            tuple(tuple(row) for row in cache.execute(f"SELECT * FROM {table}"))
            for table in (
                "retrieval_semantic_generations",
                "retrieval_semantic_generation_events",
                "retrieval_semantic_components",
                "retrieval_semantic_generation_current",
                "retrieval_semantic_state",
            )
        )


def _query_in_new_process(repo_root: Path) -> dict[str, object]:
    script = r"""
import json
import sys
from pathlib import Path
from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService, SemanticQueryInputs

repo = Path(sys.argv[1])
result = RetrievalService(RetrievalPaths.for_repo(repo)).query(
    RetrievalRequest("semantic", "new-process", "turn", "conversation", None, RetrievalScope(str(repo))),
    semantic_query=SemanticQueryInputs(None, (("semantic", 1.0),)),
)
print(json.dumps({
    "candidate_ids": [item.candidate_id for item in result.candidates],
    "degraded_reasons": [item.value for item in result.degraded_reasons],
    "status": result.status.value,
    "trace_channels": {
        item.candidate_id: [channel.value for channel in item.source_channels]
        for item in result.trace
    },
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(repo_root)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_legal_fresh_install_activates_only_from_the_verified_snapshot(tmp_path):
    """A valid owned generation must reach its callback under the held guard."""
    staged, final, expected = _prepared_install_pair(tmp_path)
    accepted = []

    def accept(manifest, connection, guard):
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute(
            "SELECT value FROM semantic_schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "1"
        guard()
        accepted.append(manifest)

    semantic_store.install_semantic_generation(
        staged,
        final,
        acceptance_callback=accept,
    )

    assert accepted == [expected]
    assert final.database_path.is_file()
    assert final.manifest_path.is_file()


def test_verified_snapshot_rejects_database_larger_than_its_read_budget(tmp_path, monkeypatch):
    """The held reader must reject a leaf before allocating beyond its bound."""
    staged, _, _ = _prepared_install_pair(tmp_path)
    monkeypatch.setattr(
        semantic_store,
        "_MAX_SEMANTIC_DATABASE_BYTES",
        staged.database_path.stat().st_size - 1,
        raising=False,
    )

    with pytest.raises(semantic_store.SemanticStoreError, match="snapshot limit"):
        with semantic_store.open_verified_semantic_snapshot(staged):
            pytest.fail("oversized semantic database was consumed")


def test_verified_snapshot_rejects_when_read_deadline_is_already_exhausted(
    tmp_path, monkeypatch
):
    staged, _, _ = _prepared_install_pair(tmp_path)
    monkeypatch.setattr(semantic_store, "_SEMANTIC_SNAPSHOT_DEADLINE_NS", -1)

    with pytest.raises(semantic_store.SemanticStoreError, match="deadline"):
        with semantic_store.open_verified_semantic_snapshot(staged):
            pytest.fail("expired semantic snapshot was consumed")


@pytest.mark.parametrize("sidecar", ["semantic.sqlite3-wal", "semantic.sqlite3-journal"])
def test_verified_snapshot_rejects_sidecar_dependent_generation(tmp_path, sidecar):
    staged, _, _ = _prepared_install_pair(tmp_path)
    (staged.directory / sidecar).write_bytes(b"untrusted sidecar")

    with pytest.raises(semantic_store.SemanticStoreError, match="unexpected leaves"):
        with semantic_store.open_verified_semantic_snapshot(staged):
            pytest.fail("sidecar-dependent semantic generation was consumed")


def test_stale_activation_fence_cannot_retire_a_concurrently_newer_generation(
    runtime, monkeypatch
):
    """A callback built against A must not retire B after B wins the registry race."""
    paths, provider, _ = _historical_generation(runtime, monkeypatch)
    first_id = _active_semantic_metadata(paths)[0]
    first = SemanticGenerationPaths.for_generation(paths.generations_dir, first_id)
    with closing(sqlite3.connect(paths.database_path)) as cache:
        expected_projection = cache.execute(
            "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
        ).fetchone()[0]
        expected_event_seq = cache.execute(
            "SELECT COALESCE(MAX(event_seq), 0) FROM retrieval_semantic_generation_events"
        ).fetchone()[0]

    newer_config = replace(_config(provider), max_batch_items=1)
    assert semantic_index.rebuild_semantic_index(paths, provider, newer_config).status is RetrievalStatus.CURRENT
    newer_id = _active_semantic_metadata(paths)[0]
    assert newer_id != first_id

    def stale_handoff(manifest, snapshot, guard):
        outcomes = semantic_index._outcomes_from_connection(snapshot, manifest)
        try:
            semantic_index._handoff_verified_generation(
                paths,
                manifest,
                outcomes,
                verified_document_bindings=semantic_index._verified_document_bindings(snapshot),
                acceptance_guard=guard,
                expected_active_semantic_generation_id=first_id,
                expected_semantic_event_seq=expected_event_seq,
                expected_projection_generation_id=expected_projection,
            )
        except TypeError as exc:
            pytest.fail(f"semantic handoff has no activation fence: {exc}")

    with pytest.raises(semantic_store.SemanticStoreError, match="activation fence"):
        semantic_store.accept_semantic_generation(first, stale_handoff)
    assert _active_semantic_metadata(paths)[0] == newer_id


def test_activation_rejects_self_consistent_semantic_document_not_in_projection(
    runtime, monkeypatch
):
    """A semantic DB cannot authenticate its own forged record/catalog binding."""
    scope = str(runtime.paths.repo_root)
    structured_ledger.propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=scope,
        topic_key="catalog-binding",
        statement="catalog binding target",
        idempotency_key="catalog-binding",
    )
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    provider = FakeProvider()
    original_projection = semantic_index._read_permanent_projection

    def forged_projection(value, generation_id):
        rows = original_projection(value, generation_id)
        assert len(rows) == 1
        return ({**rows[0], "record_id": "mem_forged_catalog_record"},)

    monkeypatch.setattr(semantic_index, "_read_permanent_projection", forged_projection)
    for attempt in ("fresh install", "existing-final retry"):
        with pytest.raises(
            semantic_store.SemanticStoreError,
            match="catalog binding",
        ):
            semantic_index.rebuild_semantic_index(paths, provider, _config(provider))
        with closing(sqlite3.connect(paths.database_path)) as cache:
            assert cache.execute(
                "SELECT active_semantic_generation_id FROM retrieval_semantic_state"
            ).fetchone()[0] is None, attempt
            assert cache.execute(
                "SELECT COUNT(*) FROM retrieval_semantic_generations"
            ).fetchone()[0] == 0, attempt
            assert cache.execute(
                "SELECT COUNT(*) FROM retrieval_semantic_generation_events"
            ).fetchone()[0] == 0, attempt


def test_activation_rejects_self_consistent_provider_input_not_in_projection(
    runtime, monkeypatch
):
    """The verified text digest must derive from the deterministic search text."""
    structured_ledger.propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(runtime.paths.repo_root),
        topic_key="text-catalog-binding",
        statement="provider input binding target",
        idempotency_key="text-catalog-binding",
    )
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    provider = FakeProvider()
    original_projection = semantic_index._read_permanent_projection

    def forged_projection(value, generation_id):
        rows = original_projection(value, generation_id)
        assert len(rows) == 1
        return ({**rows[0], "search_text": "forged provider input"},)

    monkeypatch.setattr(semantic_index, "_read_permanent_projection", forged_projection)
    for attempt in ("fresh install", "existing-final retry"):
        with pytest.raises(
            semantic_store.SemanticStoreError,
            match="catalog binding",
        ):
            semantic_index.rebuild_semantic_index(paths, provider, _config(provider))
        with closing(sqlite3.connect(paths.database_path)) as cache:
            assert cache.execute(
                "SELECT active_semantic_generation_id FROM retrieval_semantic_state"
            ).fetchone()[0] is None, attempt
            assert cache.execute(
                "SELECT COUNT(*) FROM retrieval_semantic_generations"
            ).fetchone()[0] == 0, attempt
            assert cache.execute(
                "SELECT COUNT(*) FROM retrieval_semantic_generation_events"
            ).fetchone()[0] == 0, attempt


def test_semantic_activation_serializes_with_rebuild_replacement(runtime, monkeypatch):
    """A stale rebuild snapshot cannot replace a concurrently committed activation."""
    paths, provider, _ = _historical_generation(runtime, monkeypatch)
    old_active = _active_semantic_metadata(paths)[0]
    new_record = structured_ledger.propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(paths.repo_root),
        topic_key="rebuild-race",
        statement="rebuild race target",
        idempotency_key="rebuild-race",
    )
    structured_ledger.apply_action(
        runtime.connection,
        record_id=new_record.record_id,
        action="activate",
        idempotency_key="rebuild-race-active",
    )
    runtime.connection.commit()

    def fail_handoff(*args, **kwargs):
        raise semantic_store.SemanticStoreError("retain verified final for race")

    with monkeypatch.context() as first_attempt:
        first_attempt.setattr(semantic_index, "_handoff_verified_generation", fail_handoff)
        with pytest.raises(semantic_store.SemanticStoreError, match="retain verified final"):
            semantic_index.catch_up_semantic_index(paths, provider, _config(provider))

    new_final_path = next(
        item for item in paths.generations_dir.glob("sem_*") if item.name != old_active
    )
    new_final = SemanticGenerationPaths.for_generation(
        paths.generations_dir, new_final_path.name
    )
    with closing(sqlite3.connect(paths.database_path)) as cache:
        projection_id = cache.execute(
            "SELECT active_generation_id FROM retrieval_state"
        ).fetchone()[0]
        semantic_event_seq = cache.execute(
            "SELECT COALESCE(MAX(event_seq), 0) FROM retrieval_semantic_generation_events"
        ).fetchone()[0]

    later_record = structured_ledger.propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(paths.repo_root),
        topic_key="rebuild-race-later",
        statement="canonical change after the semantic snapshot",
        idempotency_key="rebuild-race-later",
    )
    structured_ledger.apply_action(
        runtime.connection,
        record_id=later_record.record_id,
        action="activate",
        idempotency_key="rebuild-race-later-active",
    )
    runtime.connection.commit()

    rebuild_ready = threading.Event()
    allow_replace = threading.Event()
    rebuild_done = threading.Event()
    activation_done = threading.Event()
    errors = []
    rebuild_results = []
    semantic_registry_before = _semantic_registry_state(paths)

    def pause_before_replace(stage):
        if stage == "replace":
            rebuild_ready.set()
            assert allow_replace.wait(5), "rebuild replacement release timed out"

    def run_rebuild():
        try:
            rebuild_results.append(
                RetrievalService(paths).catch_up(failure_injector=pause_before_replace)
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            rebuild_done.set()

    def run_activation():
        try:
            def handoff(manifest, snapshot, guard):
                semantic_index._handoff_verified_generation(
                    paths,
                    manifest,
                    semantic_index._outcomes_from_connection(snapshot, manifest),
                    verified_document_bindings=semantic_index._verified_document_bindings(snapshot),
                    acceptance_guard=guard,
                    expected_active_semantic_generation_id=old_active,
                    expected_semantic_event_seq=semantic_event_seq,
                    expected_projection_generation_id=projection_id,
                )

            semantic_store.accept_semantic_generation(new_final, handoff)
        except BaseException as exc:
            errors.append(exc)
        finally:
            activation_done.set()

    rebuild_thread = threading.Thread(target=run_rebuild, daemon=True)
    activation_thread = threading.Thread(target=run_activation, daemon=True)
    rebuild_thread.start()
    assert rebuild_ready.wait(5), f"rebuild did not reach replacement barrier: {errors!r}"
    activation_thread.start()
    try:
        assert not activation_done.wait(0.25), "activation bypassed the rebuild lock"
    finally:
        allow_replace.set()
    assert rebuild_done.wait(5)
    assert activation_done.wait(5)
    rebuild_thread.join(timeout=1)
    activation_thread.join(timeout=1)
    assert len(errors) == 1
    assert isinstance(errors[0], semantic_store.SemanticStoreError)
    assert "activation fence" in str(errors[0])
    assert [result.status for result in rebuild_results] == [RetrievalStatus.CURRENT]
    assert _semantic_registry_state(paths) == semantic_registry_before
    assert _active_semantic_metadata(paths)[0] == old_active


def test_new_process_reopens_and_reverifies_activated_semantic_content(runtime, monkeypatch):
    paths, _, record = _historical_generation(runtime, monkeypatch)

    first = _query_in_new_process(paths.repo_root)
    assert record.record_id in first["candidate_ids"]
    assert "sparse" in first["trace_channels"][record.record_id]
    assert "semantic_index_unavailable" not in first["degraded_reasons"]

    active = _active_semantic_metadata(paths)[0]
    generation = SemanticGenerationPaths.for_generation(paths.generations_dir, active)
    generation.database_path.write_bytes(b"post-activation corruption")

    reopened = _query_in_new_process(paths.repo_root)
    assert record.record_id in reopened["candidate_ids"]
    assert "sparse" not in reopened["trace_channels"][record.record_id]
    assert "semantic_index_unavailable" in reopened["degraded_reasons"]


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh", "retry"])
def test_multiple_link_precondition_refuses_callback_before_any_activation(
    tmp_path, retry_existing_final
):
    """A pre-existing alias is rejected for both fresh and retry acceptance."""
    staged, final, _ = _prepared_install_pair(tmp_path)
    if retry_existing_final:
        semantic_store.install_semantic_generation(staged, final)
        attacked = final
    else:
        attacked = staged
    peer = tmp_path / "external-peer"
    os.link(attacked.database_path, peer)
    peer_before = peer.read_bytes()
    calls = []
    with pytest.raises(semantic_store.SemanticStoreError):
        if retry_existing_final:
            semantic_store.accept_semantic_generation(final, lambda *args: calls.append(args))
        else:
            semantic_store.install_semantic_generation(staged, final, acceptance_callback=lambda *args: calls.append(args))
    assert calls == []
    assert final.directory.exists() is retry_existing_final
    assert peer.read_bytes() == peer_before


@pytest.mark.parametrize("leaf", ["semantic.sqlite3", "generation-manifest.json"])
def test_unsafe_existing_final_preserves_old_active_history_and_fallback(
    runtime, monkeypatch, tmp_path, leaf
):
    """A multi-linked retry final cannot retire old semantic state."""
    paths, provider, record = _historical_generation(runtime, monkeypatch)
    old_active = _active_semantic_metadata(paths)[0]
    structured_ledger.propose_record(
        runtime.connection, record_type="decision", scope_kind="project", scope_key=str(paths.repo_root),
        topic_key="semantic-new", statement="semantic new target", idempotency_key="semantic-new",
    )
    runtime.connection.commit()
    canonical_before = paths.canonical_database_path.read_bytes()
    def fail_handoff(*args, **kwargs):
        raise semantic_store.SemanticStoreError("fixture final retained without handoff")
    with monkeypatch.context() as first_attempt:
        first_attempt.setattr(semantic_index, "_handoff_verified_generation", fail_handoff)
        with pytest.raises(semantic_store.SemanticStoreError, match="fixture final retained"):
            semantic_index.catch_up_semantic_index(paths, provider, _config(provider))
    new_final = next(
        item for item in paths.generations_dir.glob("sem_*") if item.name != old_active
    )
    peer = tmp_path / f"unsafe-{leaf}"
    os.link(new_final / leaf, peer)
    before = _semantic_registry_state(paths)
    with pytest.raises(semantic_store.SemanticStoreError):
        semantic_index.catch_up_semantic_index(paths, provider, _config(provider))
    assert _semantic_registry_state(paths) == before
    assert _active_semantic_metadata(paths)[0] == old_active
    assert paths.canonical_database_path.read_bytes() == canonical_before
    service = RetrievalService(paths)
    request = RetrievalRequest("semantic", "s", "t", "c", None, RetrievalScope(str(paths.repo_root)))
    legacy = service.query(request)
    assert any(item.candidate_id == record.record_id for item in legacy.candidates)
    semantic = service.query(request, semantic_query=SemanticQueryInputs(None, (("decision", 1.0),)))
    trace = next(item for item in semantic.trace if item.candidate_id == record.record_id)
    assert CandidateChannel.SPARSE in trace.source_channels
    assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE not in semantic.degraded_reasons


@pytest.mark.parametrize("operation", ["replace", "in_place"])
def test_query_consumes_same_verified_bytes_after_verifier_returns(runtime, monkeypatch, tmp_path, operation):
    """Reopening a pathname after validation would consume changed sparse rows."""
    paths, _, record = _historical_generation(runtime, monkeypatch)
    generation = SemanticGenerationPaths.for_generation(paths.generations_dir, _active_semantic_metadata(paths)[0])
    replacement = tmp_path / "owned-replacement.sqlite3"
    replacement.write_bytes(generation.database_path.read_bytes())
    with closing(sqlite3.connect(replacement)) as altered:
        for (name,) in altered.execute("SELECT name FROM sqlite_schema WHERE type='trigger' AND tbl_name='semantic_sparse_postings'").fetchall():
            altered.execute(f'DROP TRIGGER "{name}"')
        altered.execute("UPDATE semantic_sparse_postings SET term='no-query-match-' || term")
        altered.commit()
    changed = replacement.read_bytes()
    reached = []
    errors = []

    def attack_after_verification(value):
        barrier = threading.Barrier(2)
        done = threading.Event()
        def attack():
            barrier.wait(timeout=2)
            try:
                if operation == "replace":
                    os.replace(replacement, generation.database_path)
                else:
                    generation.database_path.write_bytes(changed)
            except OSError as exc:
                errors.append(exc)
            finally:
                done.set()
        thread = threading.Thread(target=attack, daemon=True)
        thread.start()
        barrier.wait(timeout=2)
        assert done.wait(2), "bounded read-race operation did not finish"
        thread.join(timeout=1)
        reached.append(True)
        return value

    original_verify = semantic_store.verify_semantic_generation
    original_held_verify = semantic_store._verify_held_semantic_generation
    monkeypatch.setattr(service_module, "verify_semantic_generation",
                        lambda p: attack_after_verification(original_verify(p)), raising=False)
    monkeypatch.setattr(semantic_store, "_verify_held_semantic_generation",
                        lambda p, h: attack_after_verification(original_held_verify(p, h)))
    result = RetrievalService(paths).query(
        RetrievalRequest("semantic", "s", "t", "c", None, RetrievalScope(str(paths.repo_root))),
        semantic_query=SemanticQueryInputs(None, (("decision", 1.0),)),
    )
    assert reached == [True]
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)
    trace = next(item for item in result.trace if item.candidate_id == record.record_id)
    assert CandidateChannel.SPARSE in trace.source_channels
    assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE not in result.degraded_reasons


def test_verified_snapshot_is_memory_readonly_and_releases_handles(tmp_path):
    staged, _, expected = _prepared_install_pair(tmp_path)
    with semantic_store.open_verified_semantic_snapshot(staged) as (manifest, connection):
        assert manifest == expected
        cursor = connection.execute("PRAGMA database_list")
        assert not hasattr(cursor, "connection")
        assert cursor.fetchone()[2] == ""
        with pytest.raises(sqlite3.DatabaseError, match="authorized|readonly"):
            connection.execute("CREATE TABLE forbidden(value)")
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            connection.execute("ATTACH DATABASE ':memory:' AS forbidden_attachment")
        assert not hasattr(connection, "enable_load_extension")
        with pytest.raises(AttributeError):
            connection.enable_load_extension(True)
        with pytest.raises(sqlite3.DatabaseError, match="authorized|disabled"):
            connection.execute("SELECT load_extension('must-not-reach-the-os-loader')")
    # WRITE/DELETE handles must be gone after consumer exit.
    staged.database_path.write_bytes(staged.database_path.read_bytes())
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        (sqlite3.Connection.enable_load_extension, (True,)),
        (sqlite3.Connection.load_extension, ("must-not-reach-the-os-loader",)),
    ],
    ids=["enable", "load"],
)
def test_sqlite_baseclass_extension_dispatch_cannot_bypass_snapshot_boundary(
    tmp_path, method, arguments
):
    """Explicit base-class dispatch must reject the public snapshot facade."""
    staged, _, _ = _prepared_install_pair(tmp_path)
    with pytest.raises(semantic_store.SemanticStoreError) as raised:
        with semantic_store.open_verified_semantic_snapshot(staged) as (_, connection):
            method(connection, *arguments)
    assert isinstance(raised.value.__cause__, TypeError)


def test_snapshot_cursor_iteration_does_not_expose_sqlite_connection(tmp_path):
    """Iteration stays on the facade and yields rows without control capabilities."""
    staged, _, _ = _prepared_install_pair(tmp_path)
    with semantic_store.open_verified_semantic_snapshot(staged) as (_, connection):
        cursor = connection.execute("SELECT 1 AS value UNION ALL SELECT 2 ORDER BY value")
        iterator = iter(cursor)
        assert iterator is cursor
        assert not hasattr(iterator, "connection")
        assert not hasattr(iterator, "enable_load_extension")
        first = next(iterator)
        assert first[0] == 1
        assert not hasattr(first, "connection")
        assert not hasattr(first, "enable_load_extension")
        assert [row[0] for row in iterator] == [2]


@pytest.mark.parametrize(
    ("write_pragma", "state_pragma"),
    [
        ("PRAGMA query_only=OFF", "PRAGMA query_only"),
        ("PRAGMA user_version=123", "PRAGMA user_version"),
        ("PRAGMA journal_mode=OFF", "PRAGMA journal_mode"),
    ],
    ids=["query-only", "user-version", "journal-mode"],
)
def test_snapshot_facade_rejects_state_changing_pragmas(
    tmp_path, write_pragma, state_pragma
):
    """Public snapshot queries cannot mutate even disposable in-memory state."""
    staged, _, _ = _prepared_install_pair(tmp_path)
    with semantic_store.open_verified_semantic_snapshot(staged) as (_, connection):
        before = connection.execute(state_pragma).fetchone()[0]
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            connection.execute(write_pragma)
        assert connection.execute(state_pragma).fetchone()[0] == before
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("PRAGMA database_list").fetchone()[2] == ""
        assert connection.execute("SELECT 1").fetchone()[0] == 1


def test_held_leaf_read_checks_deadline_after_the_final_read(tmp_path, monkeypatch):
    """A final ReadFile that crosses the deadline must not return accepted bytes."""
    leaf = tmp_path / "bounded-leaf"
    leaf.write_bytes(b"verified bytes")
    handle, _ = semantic_store._open_windows_acceptance_handle(leaf, directory=False)
    ticks = iter((0, 2))
    monkeypatch.setattr(semantic_store.time, "monotonic_ns", lambda: next(ticks))
    try:
        with pytest.raises(semantic_store.SemanticStoreError, match="deadline"):
            semantic_store._read_windows_handle(handle, max_bytes=1024, deadline_ns=1)
    finally:
        assert semantic_store._windows_kernel32().CloseHandle(handle)


def test_held_leaf_read_has_bounded_aggregate_allocation(tmp_path):
    """The production handle reader must not retain blocks and join a second copy."""
    size = 8 * 1024 * 1024
    leaf = tmp_path / "large-bounded-leaf"
    leaf.write_bytes(b"x" * size)
    handle, _ = semantic_store._open_windows_acceptance_handle(leaf, directory=False)
    try:
        tracemalloc.start()
        data = semantic_store._read_windows_handle(
            handle,
            max_bytes=size,
            deadline_ns=semantic_store.time.monotonic_ns() + 10_000_000_000,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        assert semantic_store._windows_kernel32().CloseHandle(handle)
    assert len(data) == size
    assert data[:1] == b"x" and data[-1:] == b"x"
    assert peak < size + 2_500_000


@pytest.mark.parametrize("leaf", ["semantic.sqlite3", "generation-manifest.json"])
def test_verified_snapshot_rejects_existing_multiple_links(tmp_path, leaf):
    staged, _, _ = _prepared_install_pair(tmp_path)
    peer = tmp_path / "owned-peer"
    os.link(staged.directory / leaf, peer)
    before = peer.read_bytes()
    with pytest.raises(semantic_store.SemanticStoreError):
        with semantic_store.open_verified_semantic_snapshot(staged):
            pytest.fail("multi-linked snapshot was accepted")
    assert peer.read_bytes() == before


@pytest.mark.parametrize("leaf", ["semantic.sqlite3", "generation-manifest.json"])
def test_verified_snapshot_rejects_reparse_leaf(tmp_path, leaf):
    staged, _, _ = _prepared_install_pair(tmp_path)
    target = staged.directory / leaf
    displaced = tmp_path / "owned-displaced"
    target.rename(displaced)
    try:
        os.symlink(displaced, target)
    except OSError as exc:
        pytest.skip(f"new snapshot reparse coverage unavailable: {exc}")
    with pytest.raises(semantic_store.SemanticStoreError):
        with semantic_store.open_verified_semantic_snapshot(staged):
            pytest.fail("reparse snapshot was accepted")


@pytest.mark.parametrize("contract", [
    "test_semantic_input_failure_degrades_only_that_component",
    "test_semantic_query_never_bypasses_current_scope_eligibility",
    "test_stale_semantic_row_never_bypasses_current_lifecycle",
    "test_semantic_manifest_must_match_its_active_append_only_metadata",
    "test_historical_semantic_generation_only_contributes_the_current_unchanged_subset",
])
def test_historical_snapshot_preserves_existing_query_contracts(runtime, monkeypatch, contract):
    """Reuse the established query contracts through real semantic activation."""
    import test_retrieval_query as original

    getattr(original, contract)(runtime)


def test_verified_snapshot_releases_handles_on_consumer_failure(tmp_path):
    staged, _, _ = _prepared_install_pair(tmp_path)
    with pytest.raises(RuntimeError, match="consumer stopped"):
        with semantic_store.open_verified_semantic_snapshot(staged) as (_, connection):
            raise RuntimeError("consumer stopped")
    staged.database_path.write_bytes(staged.database_path.read_bytes())
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_first_public_install_activates_semantic_and_preserves_legacy_query(runtime):
    scope = str(runtime.paths.repo_root)
    record = structured_ledger.propose_record(
        runtime.connection, record_type="decision", scope_kind="project", scope_key=scope,
        topic_key="fallback", statement="semantic fallback target", idempotency_key="fallback",
    )
    structured_ledger.apply_action(runtime.connection, record_id=record.record_id,
                                  action="activate", idempotency_key="fallback-active")
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths)
    assert service.rebuild().status is RetrievalStatus.CURRENT
    canonical = paths.canonical_database_path.read_bytes()
    provider = FakeProvider()
    assert semantic_index.catch_up_semantic_index(
        paths, provider, _config(provider)
    ).status is RetrievalStatus.CURRENT
    assert _active_semantic_metadata(paths)[0].startswith("sem_")
    assert paths.canonical_database_path.read_bytes() == canonical
    request = RetrievalRequest("semantic", "s", "t", "c", None, RetrievalScope(scope))
    legacy = service.query(request)
    result = service.query(request, semantic_query=SemanticQueryInputs(None, (("decision", 1.0),)))
    assert any(item.candidate_id == record.record_id for item in result.candidates)
    assert any(item.candidate_id == record.record_id for item in legacy.candidates)
    assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE not in result.degraded_reasons
