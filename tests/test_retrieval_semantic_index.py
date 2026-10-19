from __future__ import annotations

from dataclasses import replace
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from codex_memory import database, structured_ledger
from codex_memory.candidate_memory import capture_candidate
from codex_memory.ledger_common import canonical_json
from codex_memory.paths import RuntimePaths
from codex_memory.retrieval.config import SemanticConfig
from codex_memory.retrieval.contracts import RetrievalStatus
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.provider import ModelComponent, ProviderCallStatus, ProviderOperation

from retrieval_provider_fakes import FakeProvider


_ABSENT_SEMANTIC_COVERAGE = {
    "record_seq": 0,
    "memory_event_seq": 0,
    "evidence_seq": 0,
    "relation_seq": 0,
    "candidate_seq": 0,
    "candidate_event_seq": 0,
    "candidate_evidence_seq": 0,
}


def _canonical_database(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    repo = tmp_path / "semantic repository"
    repo.mkdir()
    runtime = RuntimePaths.for_repo(repo)
    runtime.ensure()
    connection = database.connect_database(runtime.db_path)
    database.migrate_database(connection)
    return repo, connection


def _config(provider: FakeProvider, *, dense: bool = True, sparse: bool = True) -> SemanticConfig:
    capabilities = provider.capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    dense_fingerprint = (
        capabilities.dense.identity.fingerprint
        if capabilities.dense.identity is not None
        else "d" * 64
    )
    sparse_fingerprint = (
        capabilities.sparse.identity.fingerprint
        if capabilities.sparse.identity is not None
        else "e" * 64
    )
    return SemanticConfig(
        1, "local_http", "http://127.0.0.1:11434", "audited-local-v1",
        1_000, 10_000, 1_000, 1_000, dense, sparse, False,
        16, 16, 8, 2, 1, 0, 2, 30_000,
        capabilities.fingerprint,
        dense_fingerprint if dense else None,
        sparse_fingerprint if sparse else None,
        None,
    )


def _add_records(connection: sqlite3.Connection, repo: Path, prefix: str, count: int) -> None:
    for number in range(count):
        structured_ledger.propose_record(
            connection,
            record_type="decision",
            scope_kind="project",
            scope_key=str(repo),
            topic_key=f"{prefix}.{number}",
            statement=f"{prefix} record {number}",
            idempotency_key=f"{prefix}-{number}",
        )


def _active_semantic_metadata(paths: RetrievalPaths) -> tuple[str, dict[str, sqlite3.Row], dict[str, object]]:
    with closing(sqlite3.connect(paths.database_path)) as cache:
        cache.row_factory = sqlite3.Row
        generation_id = cache.execute(
            "SELECT active_semantic_generation_id FROM retrieval_semantic_state"
        ).fetchone()[0]
        assert isinstance(generation_id, str)
        components = {
            row["component"]: row
            for row in cache.execute(
                "SELECT * FROM retrieval_semantic_components WHERE semantic_generation_id = ?",
                (generation_id,),
            )
        }
        payload = json.loads(
            cache.execute(
                """SELECT payload_json FROM retrieval_semantic_generation_events
                WHERE semantic_generation_id = ? AND action = 'create'""",
                (generation_id,),
            ).fetchone()[0]
        )
    return generation_id, components, payload


def _exercise_final_acceptance_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retry_existing_final: bool,
    attack_kind: str,
    leaf_name: str | None = None,
    external_hardlink_destination: bool = False,
) -> tuple[BaseException | None, str, str]:
    """Run an attacker exactly after final verification and before cache commit."""
    from codex_memory.retrieval import semantic_index
    from codex_memory.retrieval.semantic_store import SemanticStoreError

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.acceptance.old", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    assert semantic_index.rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    old_active, _, _ = _active_semantic_metadata(paths)

    connection = database.connect_database(paths.canonical_database_path)
    _add_records(connection, repo, "semantic.acceptance.new", 1)
    connection.close()
    canonical_before = paths.canonical_database_path.read_bytes()
    real_handoff = semantic_index._handoff_verified_generation
    real_immediate_transaction = semantic_index.immediate_transaction

    if retry_existing_final:
        def fail_first_handoff(  # type: ignore[no-untyped-def]
            handoff_paths, manifest, outcomes, *, acceptance_guard=None, **activation_fence,
        ):
            if manifest.semantic_generation_id != old_active:
                raise SemanticStoreError("injected first handoff failure")
            return real_handoff(
                handoff_paths,
                manifest,
                outcomes,
                acceptance_guard=acceptance_guard,
                **activation_fence,
            )

        with monkeypatch.context() as first_attempt:
            first_attempt.setattr(semantic_index, "_handoff_verified_generation", fail_first_handoff)
            with pytest.raises(SemanticStoreError, match="injected first handoff failure"):
                semantic_index.catch_up_semantic_index(paths, provider, config)
        assert _active_semantic_metadata(paths)[0] == old_active

    attack_error: list[BaseException | None] = [None]
    reached_seam = False

    def handoff_under_attack(  # type: ignore[no-untyped-def]
        handoff_paths, manifest, outcomes, *, acceptance_guard=None, **activation_fence,
    ):
        nonlocal reached_seam
        if manifest.semantic_generation_id == old_active:
            return real_handoff(
                handoff_paths,
                manifest,
                outcomes,
                acceptance_guard=acceptance_guard,
                **activation_fence,
            )
        reached_seam = True
        final_directory = paths.generations_dir / manifest.semantic_generation_id
        displaced = tmp_path / f"displaced-{attack_kind}-{leaf_name or 'directory'}"
        peer = (
            final_directory / f"peer-{attack_kind}-{leaf_name or 'directory'}"
            if attack_kind == "hardlink_after_guard" and not external_hardlink_destination
            else tmp_path / f"peer-{attack_kind}-{leaf_name or 'directory'}"
        )
        barrier = threading.Barrier(2)

        def attack() -> None:
            barrier.wait()
            try:
                if attack_kind == "replace":
                    assert leaf_name is not None
                    target = final_directory / leaf_name
                    target.replace(displaced)
                    target.write_bytes(b"attacker replacement")
                elif attack_kind == "in_place_write":
                    assert leaf_name is not None
                    target = final_directory / leaf_name
                    with target.open("r+b") as stream:
                        stream.seek(0)
                        stream.write(b"ATTACK")
                        stream.flush()
                        os.fsync(stream.fileno())
                elif attack_kind in {"hardlink", "hardlink_after_guard"}:
                    assert leaf_name is not None
                    os.link(final_directory / leaf_name, peer)
                elif attack_kind == "reparse":
                    assert leaf_name is not None
                    target = final_directory / leaf_name
                    target.replace(displaced)
                    os.symlink(peer, target)
                elif attack_kind == "directory_exchange":
                    final_directory.replace(displaced)
                    final_directory.mkdir()
                else:
                    raise AssertionError(f"unknown attack: {attack_kind}")
            except BaseException as exc:  # captured for assertion on the caller thread
                attack_error[0] = exc

        def run_attacker() -> None:
            attacker = threading.Thread(target=attack, name=f"semantic-{attack_kind}-attacker")
            attacker.start()
            barrier.wait()
            attacker.join(timeout=5)
            assert not attacker.is_alive(), "acceptance attack did not finish deterministically"

        if attack_kind == "hardlink_after_guard":
            @contextmanager
            def transaction_with_attack(connection, *, before_commit=None):  # type: ignore[no-untyped-def]
                def guard_then_attack() -> None:
                    if before_commit is not None:
                        before_commit()
                    run_attacker()

                with real_immediate_transaction(connection, before_commit=guard_then_attack):
                    yield

            monkeypatch.setattr(semantic_index, "immediate_transaction", transaction_with_attack)
            return real_handoff(
                handoff_paths,
                manifest,
                outcomes,
                acceptance_guard=acceptance_guard,
                **activation_fence,
            )

        run_attacker()
        if attack_error[0] is not None and not isinstance(attack_error[0], OSError):
            raise attack_error[0]
        return real_handoff(
            handoff_paths,
            manifest,
            outcomes,
            acceptance_guard=acceptance_guard,
            **activation_fence,
        )

    monkeypatch.setattr(semantic_index, "_handoff_verified_generation", handoff_under_attack)
    try:
        result = semantic_index.catch_up_semantic_index(paths, provider, config)
    except SemanticStoreError:
        if attack_error[0] is not None:
            raise
        if _active_semantic_metadata(paths)[0] != old_active:
            raise
    else:
        assert result.status is RetrievalStatus.CURRENT
    assert reached_seam
    assert paths.canonical_database_path.read_bytes() == canonical_before
    return attack_error[0], old_active, _active_semantic_metadata(paths)[0]


def test_rebuild_semantic_index_uses_verified_permanent_projection_without_writing_canonical(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    record = structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.permanent", statement="Index permanent memory only.",
        idempotency_key="semantic-permanent",
    )
    capture_candidate(
        connection, RuntimePaths.for_repo(repo).blobs_dir, project_scope_key=str(repo),
        record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.temporary", statement="This candidate must not be embedded.",
        session_id="session", turn_id="turn", context_id="window", idempotency_key="semantic-temp",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    canonical_before = paths.canonical_database_path.read_bytes()
    provider = FakeProvider()

    result = rebuild_semantic_index(paths, provider, _config(provider))

    assert result.status is RetrievalStatus.CURRENT
    assert paths.canonical_database_path.read_bytes() == canonical_before
    semantic_databases = list(paths.generations_dir.glob("sem_*/semantic.sqlite3"))
    assert len(semantic_databases) == 1
    with sqlite3.connect(semantic_databases[0]) as semantic:
        assert semantic.execute("SELECT record_id FROM semantic_documents").fetchall() == [(record.record_id,)]
        assert semantic.execute("SELECT COUNT(*) FROM semantic_dense_vectors").fetchone() == (1,)
        assert semantic.execute("SELECT COUNT(*) FROM semantic_sparse_postings").fetchone()[0] > 0
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 1
    assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 1


def test_semantic_catch_up_keeps_old_generation_when_component_is_unavailable(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_index import catch_up_semantic_index, rebuild_semantic_index
    from codex_memory.retrieval.semantic_store import verify_semantic_generation
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.first", statement="First semantic record.", idempotency_key="semantic-first",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    first_provider = FakeProvider()
    assert rebuild_semantic_index(paths, first_provider, _config(first_provider)).status is RetrievalStatus.CURRENT
    previous = next(paths.generations_dir.glob("sem_*"))
    previous_manifest = verify_semantic_generation(
        SemanticGenerationPaths.for_generation(paths.generations_dir, previous.name)
    )

    connection = database.connect_database(paths.canonical_database_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.second", statement="Second semantic record.", idempotency_key="semantic-second",
    )
    connection.close()
    unavailable = FakeProvider({ModelComponent.DENSE: ProviderCallStatus.UNAVAILABLE})

    result = catch_up_semantic_index(paths, unavailable, _config(unavailable, dense=False, sparse=True))

    assert result.status is RetrievalStatus.CURRENT
    assert verify_semantic_generation(SemanticGenerationPaths.for_generation(paths.generations_dir, previous.name)) == previous_manifest
    assert len(list(paths.generations_dir.glob("sem_*"))) >= 2


def test_semantic_catch_up_reuses_exact_unchanged_vectors_in_fixed_batches(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_index import catch_up_semantic_index, rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    for number in range(3):
        structured_ledger.propose_record(
            connection, record_type="decision", scope_kind="project", scope_key=str(repo),
            topic_key=f"semantic.batch.{number}", statement=f"Batch record {number}.",
            idempotency_key=f"semantic-batch-{number}",
        )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)

    assert rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 2
    assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 2
    connection = database.connect_database(paths.canonical_database_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.batch.new", statement="Only this record needs new vectors.",
        idempotency_key="semantic-batch-new",
    )
    connection.close()

    assert catch_up_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 3
    assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 3


def test_semantic_stage_failure_never_replaces_a_verified_old_generation(tmp_path: Path, monkeypatch) -> None:
    from codex_memory.retrieval import semantic_index
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import verify_semantic_generation

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.failure.first", statement="Existing semantic generation.",
        idempotency_key="semantic-failure-first",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    assert semantic_index.rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    previous = next(paths.generations_dir.glob("sem_*"))
    previous_manifest = verify_semantic_generation(SemanticGenerationPaths.for_generation(paths.generations_dir, previous.name))
    connection = database.connect_database(paths.canonical_database_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.failure.second", statement="Failure must retain old generation.",
        idempotency_key="semantic-failure-second",
    )
    connection.close()
    monkeypatch.setattr(semantic_index, "initialize_semantic_database", lambda _paths: (_ for _ in ()).throw(OSError("injected schema failure")))

    with pytest.raises(OSError, match="injected schema failure"):
        semantic_index.catch_up_semantic_index(paths, provider, config)

    assert verify_semantic_generation(SemanticGenerationPaths.for_generation(paths.generations_dir, previous.name)) == previous_manifest


def test_verified_semantic_build_replays_active_lifecycle_and_component_coverage(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.handoff", statement="Verified generation must have metadata handoff.",
        idempotency_key="semantic-handoff",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()

    result = rebuild_semantic_index(paths, provider, _config(provider))

    assert result.generation is not None
    with closing(sqlite3.connect(paths.database_path)) as cache:
        generation_id = cache.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state").fetchone()[0]
        assert generation_id is not None
        assert [row[0] for row in cache.execute(
            "SELECT action FROM retrieval_semantic_generation_events WHERE semantic_generation_id = ? ORDER BY event_seq",
            (generation_id,),
        )] == ["create", "validate", "activate"]
        components = cache.execute(
            "SELECT component, status, indexed_document_count, coverage_highwater_json FROM retrieval_semantic_components WHERE semantic_generation_id = ? ORDER BY component",
            (generation_id,),
        ).fetchall()
    assert [(component, status, count) for component, status, count, _ in components] == [
        ("dense", "ready", 1), ("reranker", "disabled", 0), ("sparse", "ready", 1),
    ]
    coverage_by_component = {
        component: json.loads(coverage)
        for component, _, _, coverage in components
    }
    assert coverage_by_component["dense"] == result.indexed_highwater.to_dict()
    assert coverage_by_component["sparse"] == result.indexed_highwater.to_dict()
    assert coverage_by_component["reranker"] == _ABSENT_SEMANTIC_COVERAGE


def test_cache_verification_rejects_semantic_current_projection_tamper(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import CacheCorruptError, connect_cache_database, verify_cache_database
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.projection-tamper", statement="Semantic projection must replay from events.",
        idempotency_key="semantic-projection-tamper",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    assert rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT

    with closing(sqlite3.connect(paths.database_path)) as cache:
        cache.execute(
            "UPDATE retrieval_semantic_generation_current SET state = 'retired'"
        )
        cache.commit()
    cache = connect_cache_database(paths.database_path, readonly=True)
    try:
        with pytest.raises(CacheCorruptError, match="semantic lifecycle projection"):
            verify_cache_database(cache)
    finally:
        cache.close()


def test_timeout_component_is_activated_as_explicit_partial_degradation(tmp_path: Path, monkeypatch) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index
    from codex_memory.retrieval.provider import DenseResponse, ProviderCall

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.timeout", statement="Timeout must be visible in component metadata.",
        idempotency_key="semantic-timeout",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    original = provider.embed_documents

    def timeout(items, *, deadline_ms):
        call = original(items, deadline_ms=deadline_ms).call
        return DenseResponse(ProviderCall(call.operation, call.component, ProviderCallStatus.TIMEOUT, call.provider_fingerprint, call.model_fingerprint, call.elapsed_ms, "timeout"), ())

    monkeypatch.setattr(provider, "embed_documents", timeout)
    assert rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT
    with closing(sqlite3.connect(paths.database_path)) as cache:
        statuses = dict(cache.execute("SELECT component, status FROM retrieval_semantic_components"))
    assert statuses == {"dense": "failed", "sparse": "ready", "reranker": "disabled"}


def test_provider_fingerprint_mismatch_is_audited_without_embedding(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.provider-mismatch", statement="Provider identity must match before embedding.",
        idempotency_key="semantic-provider-mismatch",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = replace(_config(provider), expected_provider_fingerprint="0" * 64)

    assert rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 0
    assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 0
    with closing(sqlite3.connect(paths.database_path)) as cache:
        statuses = dict(cache.execute("SELECT component, status FROM retrieval_semantic_components"))
    assert statuses == {"dense": "failed", "sparse": "failed", "reranker": "disabled"}


def test_main_cache_handoff_failure_keeps_old_active_semantic_generation(tmp_path: Path, monkeypatch) -> None:
    from codex_memory.retrieval import semantic_index

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.handoff.first", statement="Old metadata must stay active.",
        idempotency_key="semantic-handoff-first",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    assert semantic_index.rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    with closing(sqlite3.connect(paths.database_path)) as cache:
        old_active = cache.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state").fetchone()[0]
    connection = database.connect_database(paths.canonical_database_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.handoff.second", statement="Failed handoff cannot move active metadata.",
        idempotency_key="semantic-handoff-second",
    )
    connection.close()
    monkeypatch.setattr(semantic_index, "append_semantic_generation", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected main handoff failure")))

    with pytest.raises(RuntimeError, match="injected main handoff failure"):
        semantic_index.rebuild_semantic_index(paths, provider, config)

    with closing(sqlite3.connect(paths.database_path)) as cache:
        assert cache.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state").fetchone()[0] == old_active

    monkeypatch.undo()
    assert semantic_index.rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    with closing(sqlite3.connect(paths.database_path)) as cache:
        new_active = cache.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state").fetchone()[0]
        assert new_active != old_active
        assert cache.execute("SELECT COUNT(*) FROM retrieval_semantic_generation_events WHERE semantic_generation_id = ?", (new_active,)).fetchone() == (3,)


def test_capabilities_timeout_keeps_deterministic_floor_and_records_unavailable_components(tmp_path: Path, monkeypatch) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index
    from codex_memory.retrieval.provider import CapabilitiesResponse, ProviderCall

    repo, connection = _canonical_database(tmp_path)
    structured_ledger.propose_record(
        connection, record_type="decision", scope_kind="project", scope_key=str(repo),
        topic_key="semantic.capability-timeout", statement="Capability timeout retains deterministic retrieval.",
        idempotency_key="semantic-capability-timeout",
    )
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    original = provider.capabilities

    def timeout(*, deadline_ms):
        call = original(deadline_ms=deadline_ms).call
        return CapabilitiesResponse(ProviderCall(call.operation, None, ProviderCallStatus.TIMEOUT, None, None, call.elapsed_ms, "timeout"), None)

    monkeypatch.setattr(provider, "capabilities", timeout)
    result = rebuild_semantic_index(paths, provider, config)

    assert result.status is RetrievalStatus.CURRENT
    with closing(sqlite3.connect(paths.database_path)) as cache:
        assert dict(cache.execute("SELECT component, status FROM retrieval_semantic_components")) == {
            "dense": "unavailable", "sparse": "unavailable", "reranker": "disabled",
        }
        payload = cache.execute("SELECT payload_json FROM retrieval_semantic_generation_events WHERE action = 'create'").fetchone()[0]
    assert "timeout" in payload


def test_orphan_retry_preserves_partial_sparse_dense_dimension_and_component_reasons(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed cache handoff must not manufacture replacement outcome metadata."""
    from codex_memory.retrieval import semantic_index
    from codex_memory.retrieval.provider import ProviderCall, SparseResponse

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.retry.partial", 3)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    original_sparse = provider.sparse_documents
    sparse_call = 0

    def timeout_last_batch(items, *, deadline_ms):
        nonlocal sparse_call
        sparse_call += 1
        response = original_sparse(items, deadline_ms=deadline_ms)
        if sparse_call != 2:
            return response
        call = response.call
        return SparseResponse(
            ProviderCall(
                call.operation,
                call.component,
                ProviderCallStatus.TIMEOUT,
                call.provider_fingerprint,
                call.model_fingerprint,
                call.elapsed_ms,
                "timeout",
            ),
            (),
        )

    monkeypatch.setattr(provider, "sparse_documents", timeout_last_batch)
    original_handoff = semantic_index.append_semantic_generation
    monkeypatch.setattr(
        semantic_index,
        "append_semantic_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("handoff stopped")),
    )
    with pytest.raises(RuntimeError, match="handoff stopped"):
        semantic_index.rebuild_semantic_index(paths, provider, config)

    monkeypatch.setattr(semantic_index, "append_semantic_generation", original_handoff)
    assert semantic_index.rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT

    _, components, payload = _active_semantic_metadata(paths)
    assert (components["dense"]["status"], components["dense"]["vector_dimension"]) == (
        "ready",
        4,
    )
    assert (
        components["sparse"]["status"],
        components["sparse"]["eligible_document_count"],
        components["sparse"]["indexed_document_count"],
    ) == ("partial", 3, 2)
    assert payload["component_degraded_reasons"] == {
        "dense": None,
        "reranker": "disabled",
        "sparse": "timeout",
    }

    with closing(sqlite3.connect(paths.database_path)) as cache:
        deterministic_highwater = json.loads(
            cache.execute(
                """SELECT g.indexed_highwater_json
                FROM retrieval_state AS s JOIN retrieval_generations AS g
                  ON g.generation_id = s.active_generation_id"""
            ).fetchone()[0]
        )
    assert json.loads(components["dense"]["coverage_highwater_json"]) == deterministic_highwater
    assert (
        json.loads(components["sparse"]["coverage_highwater_json"])
        == _ABSENT_SEMANTIC_COVERAGE
    )
    assert json.loads(components["sparse"]["coverage_highwater_json"]) != deterministic_highwater

    from codex_memory.retrieval.service import RetrievalService

    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT
    _, replayed_components, _ = _active_semantic_metadata(paths)
    assert (
        replayed_components["sparse"]["coverage_highwater_json"]
        == components["sparse"]["coverage_highwater_json"]
    )


@pytest.mark.parametrize(
    ("provider_status", "dense_enabled", "expected_status", "expected_reason"),
    [
        (ProviderCallStatus.OK, False, "disabled", "disabled"),
        (ProviderCallStatus.UNSUPPORTED, True, "unsupported", "unsupported"),
        (ProviderCallStatus.UNAVAILABLE, True, "unavailable", "unavailable"),
    ],
)
def test_orphan_retry_preserves_zero_row_component_state_and_reason(
    tmp_path: Path,
    monkeypatch,
    provider_status: ProviderCallStatus,
    dense_enabled: bool,
    expected_status: str,
    expected_reason: str,
) -> None:
    from codex_memory.retrieval import semantic_index

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, f"semantic.retry.{expected_status}", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    statuses = {} if provider_status is ProviderCallStatus.OK else {ModelComponent.DENSE: provider_status}
    provider = FakeProvider(statuses)
    config = _config(provider, dense=dense_enabled, sparse=True)
    original_handoff = semantic_index.append_semantic_generation
    monkeypatch.setattr(
        semantic_index,
        "append_semantic_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("handoff stopped")),
    )
    with pytest.raises(RuntimeError, match="handoff stopped"):
        semantic_index.rebuild_semantic_index(paths, provider, config)

    monkeypatch.setattr(semantic_index, "append_semantic_generation", original_handoff)
    assert semantic_index.rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    _, components, payload = _active_semantic_metadata(paths)
    assert components["dense"]["status"] == expected_status
    assert components["dense"]["indexed_document_count"] == 0
    assert payload["component_degraded_reasons"]["dense"] == expected_reason


def test_input_too_long_is_persisted_as_a_component_specific_reason(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index
    from codex_memory.retrieval.provider import ComponentCapability, ProviderCapabilities

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.too.long", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    capabilities = provider._capabilities
    assert capabilities.dense.identity is not None
    short_dense = replace(capabilities.dense.identity, max_input_chars=1)
    provider._capabilities = ProviderCapabilities(
        capabilities.protocol_version,
        capabilities.provider_type,
        capabilities.provider_version,
        capabilities.provider_digest,
        ComponentCapability(ModelComponent.DENSE, capabilities.dense.state, short_dense, None),
        capabilities.sparse,
        capabilities.reranker,
    )

    assert rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT
    _, components, payload = _active_semantic_metadata(paths)
    assert components["dense"]["status"] == "failed"
    assert components["dense"]["indexed_document_count"] == 0
    assert payload["component_degraded_reasons"]["dense"] == "input_too_long"
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 0


def test_partial_provider_response_is_component_scoped_and_auditable(
    tmp_path: Path, monkeypatch
) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index
    from codex_memory.retrieval.provider import DenseResponse

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.partial.response", 3)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    original_dense = provider.embed_documents
    dense_call = 0

    def omit_last_batch(items, *, deadline_ms):
        nonlocal dense_call
        dense_call += 1
        response = original_dense(items, deadline_ms=deadline_ms)
        return response if dense_call == 1 else DenseResponse(response.call, ())

    monkeypatch.setattr(provider, "embed_documents", omit_last_batch)
    assert rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT
    _, components, payload = _active_semantic_metadata(paths)
    assert (
        components["dense"]["status"],
        components["dense"]["eligible_document_count"],
        components["dense"]["indexed_document_count"],
    ) == ("partial", 3, 2)
    assert payload["component_degraded_reasons"]["dense"] == "invalid_response"
    assert components["sparse"]["status"] == "ready"


def test_component_capability_timeout_keeps_its_specific_reason(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.component.timeout", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider({ModelComponent.DENSE: ProviderCallStatus.TIMEOUT})

    assert rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT
    _, components, payload = _active_semantic_metadata(paths)
    assert components["dense"]["status"] == "unavailable"
    assert payload["component_degraded_reasons"]["dense"] == "timeout"


def test_capabilities_timeout_does_not_relabel_an_intentionally_disabled_component(
    tmp_path: Path, monkeypatch
) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index
    from codex_memory.retrieval.provider import CapabilitiesResponse, ProviderCall

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.disabled.timeout", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider, dense=False, sparse=True)
    original = provider.capabilities

    def timeout(*, deadline_ms):
        call = original(deadline_ms=deadline_ms).call
        return CapabilitiesResponse(
            ProviderCall(
                call.operation,
                None,
                ProviderCallStatus.TIMEOUT,
                None,
                None,
                call.elapsed_ms,
                "timeout",
            ),
            None,
        )

    monkeypatch.setattr(provider, "capabilities", timeout)
    assert rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    _, components, payload = _active_semantic_metadata(paths)
    assert components["dense"]["status"] == "disabled"
    assert payload["component_degraded_reasons"]["dense"] == "disabled"
    assert components["sparse"]["status"] == "unavailable"
    assert payload["component_degraded_reasons"]["sparse"] == "timeout"


def test_outcome_metadata_is_validated_before_final_install(
    tmp_path: Path, monkeypatch
) -> None:
    from codex_memory.retrieval import semantic_index
    from codex_memory.retrieval.semantic_store import SemanticStoreError

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.outcome.validation", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()

    def append_conflicting_outcomes(connection, outcomes):
        value = semantic_index._component_outcomes_value(outcomes)
        value["dense"]["reason"] = "timeout"
        connection.execute(
            "INSERT INTO semantic_schema_meta(key, value) VALUES (?, ?)",
            (semantic_index._COMPONENT_OUTCOMES_META_KEY, canonical_json(value)),
        )

    monkeypatch.setattr(semantic_index, "_append_component_outcomes", append_conflicting_outcomes)
    with pytest.raises(SemanticStoreError, match="outcome|ready"):
        semantic_index.rebuild_semantic_index(paths, provider, _config(provider))
    assert list(paths.generations_dir.glob("sem_*")) == []


def test_semantic_reuse_requires_content_text_dimension_and_preprocessing_identity(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval import semantic_index

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.reuse.identity", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    result = semantic_index.rebuild_semantic_index(paths, provider, config)
    assert result.generation is not None
    documents = semantic_index._read_permanent_projection(paths, result.generation.generation_id)
    capabilities = provider.capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    assert capabilities.dense.identity is not None
    assert capabilities.sparse.identity is not None
    preprocessing = semantic_index._preprocessing_fingerprint()
    dense_arguments = (
        paths,
        documents,
        capabilities.dense.identity.fingerprint,
    )
    sparse_arguments = (
        paths,
        documents,
        capabilities.sparse.identity.fingerprint,
    )

    changed_content = (dict(documents[0], content_sha256="0" * 64),)
    changed_text = (dict(documents[0], search_text=documents[0]["search_text"] + " changed"),)
    assert semantic_index._reusable_dense_rows(
        *dense_arguments,
        dimension=capabilities.dense.identity.dimension,
        preprocessing_fingerprint=preprocessing,
    )
    assert semantic_index._reusable_dense_rows(
        paths,
        changed_content,
        capabilities.dense.identity.fingerprint,
        dimension=capabilities.dense.identity.dimension,
        preprocessing_fingerprint=preprocessing,
    ) == ()
    assert semantic_index._reusable_dense_rows(
        paths,
        changed_text,
        capabilities.dense.identity.fingerprint,
        dimension=capabilities.dense.identity.dimension,
        preprocessing_fingerprint=preprocessing,
    ) == ()
    assert semantic_index._reusable_dense_rows(
        *dense_arguments,
        dimension=capabilities.dense.identity.dimension + 1,
        preprocessing_fingerprint=preprocessing,
    ) == ()
    assert semantic_index._reusable_dense_rows(
        *dense_arguments,
        dimension=capabilities.dense.identity.dimension,
        preprocessing_fingerprint="0" * 64,
    ) == ()
    assert semantic_index._reusable_sparse_rows(
        *sparse_arguments,
        preprocessing_fingerprint=preprocessing,
    )
    assert semantic_index._reusable_sparse_rows(
        paths,
        changed_text,
        capabilities.sparse.identity.fingerprint,
        preprocessing_fingerprint=preprocessing,
    ) == ()
    assert semantic_index._reusable_sparse_rows(
        *sparse_arguments,
        preprocessing_fingerprint="0" * 64,
    ) == ()


def test_semantic_input_v1_applies_only_the_declared_deterministic_transformations() -> None:
    from codex_memory.retrieval import semantic_index

    assert semantic_index._semantic_input_text("  Ａ\r\nB\r\n\n") == "  A\nB\n"
    assert semantic_index._semantic_input_text(" keep outer spaces ") == " keep outer spaces "


def test_same_projection_capabilities_and_config_reuse_identical_generation(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.semantic_index import rebuild_semantic_index

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.deterministic.generation", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)

    assert rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    generation = next(paths.generations_dir.glob("sem_*"))
    manifest_before = (generation / "generation-manifest.json").read_bytes()
    dense_calls = provider.call_counts[ProviderOperation.DENSE_DOCUMENTS]
    sparse_calls = provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS]

    assert rebuild_semantic_index(paths, provider, config).status is RetrievalStatus.CURRENT
    assert list(paths.generations_dir.glob("sem_*")) == [generation]
    assert (generation / "generation-manifest.json").read_bytes() == manifest_before
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == dense_calls
    assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == sparse_calls


def test_provider_calls_finish_before_semantic_or_cache_write_connections_open(
    tmp_path: Path, monkeypatch
) -> None:
    from codex_memory.retrieval import semantic_index

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.transaction.isolation", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    original_initialize = semantic_index.initialize_semantic_database
    original_connect_cache = semantic_index.connect_cache_database

    def initialize_after_provider(generation_paths):
        assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 1
        assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 1
        return original_initialize(generation_paths)

    def cache_after_provider(path, *, readonly=False):
        if not readonly:
            assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 1
            assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 1
        return original_connect_cache(path, readonly=readonly)

    monkeypatch.setattr(semantic_index, "initialize_semantic_database", initialize_after_provider)
    monkeypatch.setattr(semantic_index, "connect_cache_database", cache_after_provider)
    assert semantic_index.rebuild_semantic_index(paths, provider, _config(provider)).status is RetrievalStatus.CURRENT


def test_partial_semantic_highwater_is_separate_from_deterministic_highwater(
    tmp_path: Path, monkeypatch
) -> None:
    from codex_memory.retrieval import semantic_index
    from codex_memory.retrieval.provider import DenseResponse, ProviderCall

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.highwater.first", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    first = semantic_index.rebuild_semantic_index(paths, provider, config)
    assert first.generation is not None
    connection = database.connect_database(paths.canonical_database_path)
    _add_records(connection, repo, "semantic.highwater.second", 1)
    connection.close()
    original_dense = provider.embed_documents

    def timeout_new_document(items, *, deadline_ms):
        call = original_dense(items, deadline_ms=deadline_ms).call
        return DenseResponse(
            ProviderCall(
                call.operation,
                call.component,
                ProviderCallStatus.TIMEOUT,
                call.provider_fingerprint,
                call.model_fingerprint,
                call.elapsed_ms,
                "timeout",
            ),
            (),
        )

    monkeypatch.setattr(provider, "embed_documents", timeout_new_document)
    result = semantic_index.catch_up_semantic_index(paths, provider, config)
    assert result.generation is not None

    semantic_id, components, _ = _active_semantic_metadata(paths)
    with closing(sqlite3.connect(paths.database_path)) as cache:
        deterministic_id, indexed_highwater_json = cache.execute(
            """SELECT g.generation_id, g.indexed_highwater_json
            FROM retrieval_state AS s JOIN retrieval_generations AS g
              ON g.generation_id = s.active_generation_id"""
        ).fetchone()
        projection_id = cache.execute(
            "SELECT projection_generation_id FROM retrieval_semantic_generations WHERE semantic_generation_id = ?",
            (semantic_id,),
        ).fetchone()[0]
        semantic_reasons = cache.execute(
            "SELECT degraded_reasons_json FROM retrieval_semantic_state"
        ).fetchone()[0]
    assert deterministic_id == result.generation.generation_id == projection_id
    assert json.loads(indexed_highwater_json) == result.indexed_highwater.to_dict()
    assert (
        components["dense"]["status"],
        components["dense"]["eligible_document_count"],
        components["dense"]["indexed_document_count"],
    ) == ("partial", 2, 1)
    partial_coverage = json.loads(components["dense"]["coverage_highwater_json"])
    assert partial_coverage == _ABSENT_SEMANTIC_COVERAGE
    assert partial_coverage != result.indexed_highwater.to_dict()
    assert (
        json.loads(components["sparse"]["coverage_highwater_json"])
        == result.indexed_highwater.to_dict()
    )
    assert json.loads(semantic_reasons) == ["timeout"]


def test_cache_verification_rejects_partial_component_claiming_current_highwater(
    tmp_path: Path, monkeypatch
) -> None:
    from codex_memory.retrieval import semantic_index
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError,
        connect_cache_database,
        verify_cache_database,
    )
    from codex_memory.retrieval.provider import DenseResponse, ProviderCall

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, "semantic.coverage.validation", 3)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    original_dense = provider.embed_documents
    dense_call = 0

    def timeout_last_batch(items, *, deadline_ms):
        nonlocal dense_call
        dense_call += 1
        response = original_dense(items, deadline_ms=deadline_ms)
        if dense_call == 1:
            return response
        call = response.call
        return DenseResponse(
            ProviderCall(
                call.operation,
                call.component,
                ProviderCallStatus.TIMEOUT,
                call.provider_fingerprint,
                call.model_fingerprint,
                call.elapsed_ms,
                "timeout",
            ),
            (),
        )

    monkeypatch.setattr(provider, "embed_documents", timeout_last_batch)
    assert semantic_index.rebuild_semantic_index(
        paths, provider, _config(provider)
    ).status is RetrievalStatus.CURRENT

    with closing(connect_cache_database(paths.database_path)) as cache:
        current_highwater = cache.execute(
            """SELECT g.indexed_highwater_json
            FROM retrieval_state AS s JOIN retrieval_generations AS g
              ON g.generation_id = s.active_generation_id"""
        ).fetchone()[0]
        trigger_sql = cache.execute(
            """SELECT sql FROM sqlite_master
            WHERE type = 'trigger'
              AND name = 'retrieval_semantic_components_reject_update'"""
        ).fetchone()[0]
        cache.execute("DROP TRIGGER retrieval_semantic_components_reject_update")
        cache.execute(
            """UPDATE retrieval_semantic_components
            SET coverage_highwater_json = ?
            WHERE component = 'dense'""",
            (current_highwater,),
        )
        cache.execute(trigger_sql)

        with pytest.raises(CacheCorruptError, match="coverage"):
            verify_cache_database(cache)


@pytest.mark.parametrize(
    "failure_stage",
    ["schema", "row_write", "manifest", "rename", "final_verification"],
)
def test_semantic_failure_boundaries_preserve_new_deterministic_and_old_semantic_generation(
    tmp_path: Path, monkeypatch, failure_stage: str
) -> None:
    from codex_memory.retrieval import semantic_index, semantic_store
    from codex_memory.retrieval.semantic_store import SemanticStoreError

    repo, connection = _canonical_database(tmp_path)
    _add_records(connection, repo, f"semantic.failure.boundary.{failure_stage}.first", 1)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)
    first = semantic_index.rebuild_semantic_index(paths, provider, config)
    assert first.generation is not None
    old_semantic_id, _, _ = _active_semantic_metadata(paths)
    connection = database.connect_database(paths.canonical_database_path)
    _add_records(connection, repo, f"semantic.failure.boundary.{failure_stage}.second", 1)
    connection.close()
    canonical_before = paths.canonical_database_path.read_bytes()

    if failure_stage == "schema":
        monkeypatch.setattr(
            semantic_index,
            "initialize_semantic_database",
            lambda _paths: (_ for _ in ()).throw(SemanticStoreError("injected schema failure")),
        )
    elif failure_stage == "row_write":
        monkeypatch.setattr(
            semantic_index,
            "append_dense_vector",
            lambda *args, **kwargs: (_ for _ in ()).throw(SemanticStoreError("injected row failure")),
        )
    elif failure_stage == "manifest":
        monkeypatch.setattr(
            semantic_index,
            "build_semantic_manifest",
            lambda *args, **kwargs: (_ for _ in ()).throw(SemanticStoreError("injected manifest failure")),
        )
    elif failure_stage == "rename":
        monkeypatch.setattr(
            semantic_index,
            "install_semantic_generation",
            lambda *args, **kwargs: (_ for _ in ()).throw(SemanticStoreError("injected rename failure")),
        )
    else:
        original_verify = semantic_store._verify_held_semantic_generation

        def fail_new_final(generation_paths, held):
            result = original_verify(generation_paths, held)
            if (
                generation_paths.directory.parent == paths.generations_dir
                and generation_paths.semantic_generation_id != old_semantic_id
            ):
                raise SemanticStoreError("injected final verification failure")
            return result

        monkeypatch.setattr(semantic_store, "_verify_held_semantic_generation", fail_new_final)

    with pytest.raises(SemanticStoreError, match="injected"):
        semantic_index.catch_up_semantic_index(paths, provider, config)

    assert paths.canonical_database_path.read_bytes() == canonical_before
    with closing(sqlite3.connect(paths.database_path)) as cache:
        new_deterministic_id = cache.execute(
            "SELECT active_generation_id FROM retrieval_state"
        ).fetchone()[0]
        retained_semantic_id = cache.execute(
            "SELECT active_semantic_generation_id FROM retrieval_semantic_state"
        ).fetchone()[0]
    assert new_deterministic_id != first.generation.generation_id
    assert retained_semantic_id == old_semantic_id


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh-install", "existing-final-retry"])
@pytest.mark.parametrize("leaf_name", ["semantic.sqlite3", "generation-manifest.json"])
@pytest.mark.parametrize("attack_kind", ["replace", "in_place_write"])
def test_final_leaf_is_held_from_verified_identity_through_active_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_existing_final: bool,
    leaf_name: str,
    attack_kind: str,
) -> None:
    attack_error, old_active, active = _exercise_final_acceptance_attack(
        tmp_path,
        monkeypatch,
        retry_existing_final=retry_existing_final,
        attack_kind=attack_kind,
        leaf_name=leaf_name,
    )

    if attack_error is None:
        assert active == old_active, "a successful leaf attack must not be activated"
    else:
        assert isinstance(attack_error, OSError)
        assert active != old_active


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh-install", "existing-final-retry"])
@pytest.mark.parametrize("leaf_name", ["semantic.sqlite3", "generation-manifest.json"])
def test_final_leaf_hardlink_creation_is_rejected_during_active_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_existing_final: bool,
    leaf_name: str,
) -> None:
    probe_source = tmp_path / "hardlink-probe-source"
    probe_peer = tmp_path / "hardlink-probe-peer"
    probe_source.write_bytes(b"probe")
    try:
        os.link(probe_source, probe_peer)
    except OSError as exc:
        pytest.skip(f"Windows hardlink creation unavailable for this filesystem: {exc}")
    finally:
        probe_peer.unlink(missing_ok=True)
        probe_source.unlink(missing_ok=True)

    attack_error, old_active, active = _exercise_final_acceptance_attack(
        tmp_path,
        monkeypatch,
        retry_existing_final=retry_existing_final,
        attack_kind="hardlink",
        leaf_name=leaf_name,
    )
    if attack_error is None:
        assert active == old_active, "a successful hardlink attack must not be activated"
    else:
        assert isinstance(attack_error, OSError)
        assert active != old_active


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh-install", "existing-final-retry"])
@pytest.mark.parametrize("leaf_name", ["semantic.sqlite3", "generation-manifest.json"])
def test_hardlink_after_acceptance_guard_is_refused_before_active_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_existing_final: bool,
    leaf_name: str,
) -> None:
    probe_source = tmp_path / "post-guard-hardlink-probe-source"
    probe_peer = tmp_path / "post-guard-hardlink-probe-peer"
    probe_source.write_bytes(b"probe")
    try:
        os.link(probe_source, probe_peer)
    except OSError as exc:
        pytest.skip(f"Windows hardlink creation unavailable for this filesystem: {exc}")
    finally:
        probe_peer.unlink(missing_ok=True)
        probe_source.unlink(missing_ok=True)

    attack_error, old_active, active = _exercise_final_acceptance_attack(
        tmp_path,
        monkeypatch,
        retry_existing_final=retry_existing_final,
        attack_kind="hardlink_after_guard",
        leaf_name=leaf_name,
    )

    assert isinstance(attack_error, OSError)
    assert active != old_active


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh-install", "existing-final-retry"])
@pytest.mark.parametrize("leaf_name", ["semantic.sqlite3", "generation-manifest.json"])
def test_external_hardlink_after_guard_distinguishes_alias_from_unverified_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_existing_final: bool,
    leaf_name: str,
) -> None:
    """CR-02 permits alias-only activation but future reads must still reverify."""
    probe_source = tmp_path / "external-link-probe-source"
    probe_peer = tmp_path / "external-link-probe-peer"
    probe_source.write_bytes(b"probe")
    try:
        os.link(probe_source, probe_peer)
    except OSError as exc:
        pytest.skip(f"external hardlink fixture unavailable: {exc}")
    finally:
        probe_peer.unlink(missing_ok=True)
        probe_source.unlink(missing_ok=True)

    attack_error, old_active, active = _exercise_final_acceptance_attack(
        tmp_path,
        monkeypatch,
        retry_existing_final=retry_existing_final,
        attack_kind="hardlink_after_guard",
        leaf_name=leaf_name,
        external_hardlink_destination=True,
    )
    peer = tmp_path / f"peer-hardlink_after_guard-{leaf_name}"
    if attack_error is None:
        from codex_memory.retrieval.contracts import DegradedReason, RetrievalRequest, RetrievalScope
        from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
        from codex_memory.retrieval.service import RetrievalService, SemanticQueryInputs

        assert peer.is_file(), "the external hardlink was not actually created"
        assert peer.stat().st_nlink >= 2
        assert active != old_active, "verified alias-only content was not activated"
        paths = RetrievalPaths.for_repo(tmp_path / "semantic repository")
        active_paths = SemanticGenerationPaths.for_generation(paths.generations_dir, active)
        assert peer.read_bytes() == (active_paths.directory / leaf_name).read_bytes()
        fallback = RetrievalService(paths).query(
            RetrievalRequest(
                "semantic",
                "session",
                "turn",
                "conversation",
                None,
                RetrievalScope(str(paths.repo_root)),
            ),
            semantic_query=SemanticQueryInputs(None, (("decision", 1.0),)),
        )
        assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE in fallback.degraded_reasons
    else:
        assert isinstance(attack_error, OSError)
        assert getattr(attack_error, "winerror", None) in {5, 32}
        assert not peer.exists()


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh-install", "existing-final-retry"])
@pytest.mark.parametrize("leaf_name", ["semantic.sqlite3", "generation-manifest.json"])
def test_final_leaf_reparse_exchange_is_rejected_during_active_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_existing_final: bool,
    leaf_name: str,
) -> None:
    probe_target = tmp_path / "symlink-probe-target"
    probe_link = tmp_path / "symlink-probe-link"
    probe_target.write_bytes(b"probe")
    try:
        os.symlink(probe_target, probe_link)
    except OSError as exc:
        pytest.skip(f"Windows file symlink/reparse creation unavailable: {exc}")
    finally:
        probe_link.unlink(missing_ok=True)
        probe_target.unlink(missing_ok=True)

    attack_error, old_active, active = _exercise_final_acceptance_attack(
        tmp_path,
        monkeypatch,
        retry_existing_final=retry_existing_final,
        attack_kind="reparse",
        leaf_name=leaf_name,
    )
    if attack_error is None:
        assert active == old_active, "a successful reparse exchange must not be activated"
    else:
        assert isinstance(attack_error, OSError)
        assert active != old_active


@pytest.mark.parametrize("retry_existing_final", [False, True], ids=["fresh-install", "existing-final-retry"])
def test_final_directory_is_held_through_active_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_existing_final: bool,
) -> None:
    attack_error, old_active, active = _exercise_final_acceptance_attack(
        tmp_path,
        monkeypatch,
        retry_existing_final=retry_existing_final,
        attack_kind="directory_exchange",
    )
    if attack_error is None:
        assert active == old_active, "a successful install-directory exchange must not be activated"
    else:
        assert isinstance(attack_error, OSError)
        assert active != old_active
