from __future__ import annotations

import pytest
from pathlib import Path
import sqlite3

from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
from codex_memory.retrieval.errors import RetrievalValidationError
from codex_memory.retrieval.query import serialize_query_result
from codex_memory.retrieval.service import RetrievalService
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.contracts import RetrievedCandidate


def _semantic_config(provider):
    from codex_memory.retrieval.config import SemanticConfig

    capabilities = provider.capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    assert capabilities.dense.identity is not None
    assert capabilities.sparse.identity is not None
    return SemanticConfig(
        1, "local_http", "http://127.0.0.1:11434", "audited-local-v1",
        1_000, 10_000, 1_000, 1_000, True, True, False,
        16, 16, 8, 2, 1, 0, 2, 30_000,
        capabilities.fingerprint, capabilities.dense.identity.fingerprint,
        capabilities.sparse.identity.fingerprint, None,
    )


def _request(*, mode: RetrievalMode = RetrievalMode.OFF) -> RetrievalRequest:
    return RetrievalRequest("needle", "session", "turn", "context", None, RetrievalScope("project"), mode=mode)


def test_query_rejects_active_before_reading_retrieval_paths(tmp_path) -> None:
    """Would fail if an active request could use the offline query surface."""
    with pytest.raises(RetrievalValidationError, match="active retrieval mode is unavailable"):
        RetrievalService(RetrievalPaths.for_repo(tmp_path)).query(_request(mode=RetrievalMode.ACTIVE))


def test_optional_rerank_orchestration_is_injectable_and_fallback_is_deterministic(runtime) -> None:
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.provider import ModelComponent, ProviderInput, ProviderCallStatus
    from codex_memory.retrieval.provider_health import ProviderHealthStore
    from retrieval_provider_fakes import FakeProvider

    first = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="rerank-one", statement="one", value=None, evidence_uris=(), idempotency_key="rerank-one")
    second = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="rerank-two", statement="two", value=None, evidence_uris=(), idempotency_key="rerank-two")
    for record in (first, second):
        apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key=f"activate-{record.record_id}")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); assert service.rebuild().status.value == "current"
    request = RetrievalRequest("one", "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root)))
    provider = FakeProvider({})
    documents = {record.record_id: ProviderInput.from_text(record.record_id, index, record.statement) for index, record in enumerate((first, second))}
    result = service.query_with_rerank(request, provider=provider, documents=documents, deadline_ms=500, health_store=ProviderHealthStore(service.paths.database_path))

    assert result.rerank is not None and result.rerank.status is ProviderCallStatus.OK
    assert result.result.to_dict() == service.query(request).to_dict()


def _stalled_rerank_case(runtime):
    import threading
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    class Stalled(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            self.started.set()
            self.release.wait(1)
            return super().rerank(query, documents, deadline_ms=deadline_ms)

    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="rerank-stall", statement="stall", value=None, evidence_uris=(), idempotency_key="rerank-stall")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="rerank-stall-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); assert service.rebuild().status.value == "current"
    request = RetrievalRequest("stall", "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root)))
    legacy = service.query(request)
    provider = Stalled()
    return service, request, record, legacy, provider


def _synchronize_logical_rerank_workers(provider, clock_ns, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture readiness for state assertions; never used by wall-clock tests."""
    import threading
    from codex_memory.retrieval import provider_control

    real_thread = threading.Thread

    class StageReadyThread(real_thread):
        def start(self):
            super().start()
            if self.name == "codex-memory-provider-rerank":
                # Preserve the real worker/start gate. Timeout is known only
                # after actual provider entry, not merely after admission.
                assert provider.started.wait(1), "provider must actually enter"
                clock_ns[0] = 120_000_000
            elif self.name == "codex-memory-health-verify":
                # Retain real schema/history reads and connection close. This
                # independent fixture watchdog separates readiness from the
                # production 50ms wait; it is not a production time allowance.
                self.join(1)
                assert not self.is_alive(), "real health verifier must finish"

    monkeypatch.setattr(provider_control.threading, "Thread", StageReadyThread)


def test_optional_rerank_orchestration_returns_legacy_result_inside_deadline(runtime) -> None:
    """Real 180ms budget; safe refusal is not proof of provider dispatch."""
    import time
    from codex_memory.retrieval.provider import ProviderInput, ProviderCallStatus, ProviderOperation
    from codex_memory.retrieval.provider_health import ProviderHealthStore

    service, request, record, legacy, provider = _stalled_rerank_case(runtime)
    started = time.monotonic()
    try:
        observation = service.query_with_rerank(request, provider=provider, documents={record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)}, deadline_ms=180, health_store=ProviderHealthStore(service.paths.database_path), base_result=legacy)
        elapsed = time.monotonic() - started
    finally:
        provider.release.set()

    assert elapsed < 0.25
    assert observation.rerank is not None and observation.rerank.status in {ProviderCallStatus.TIMEOUT, ProviderCallStatus.DEADLINE_EXHAUSTED}
    assert observation.rerank.completion is not None
    assert observation.rerank.completion.wait(1)
    assert observation.result.to_dict() == legacy.to_dict()
    assert observation.rerank.call is not None
    if provider.started.is_set():
        assert observation.rerank.call.status is ProviderCallStatus.TIMEOUT
        assert provider.call_counts[ProviderOperation.RERANK] == 1
    else:
        assert observation.rerank.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
        assert provider.call_counts[ProviderOperation.RERANK] == 0


def test_optional_rerank_started_provider_timeout_preserves_legacy_and_settles(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mandatory dispatch/timeout with a logical budget, not a wall-clock claim."""
    from codex_memory.retrieval.provider import ProviderInput, ProviderCallStatus, ProviderOperation
    from codex_memory.retrieval.provider_health import ProviderHealthStore

    service, request, record, legacy, provider = _stalled_rerank_case(runtime)
    clock_ns = [0]
    _synchronize_logical_rerank_workers(provider, clock_ns, monkeypatch)
    try:
        observation = service.query_with_rerank(request, provider=provider,
            documents={record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)},
            deadline_ms=180, now_ns=lambda: clock_ns[0],
            health_store=ProviderHealthStore(service.paths.database_path), base_result=legacy)
    finally:
        provider.release.set()

    assert provider.started.is_set()
    assert observation.rerank is not None
    assert observation.rerank.status is ProviderCallStatus.TIMEOUT
    assert observation.rerank.call is not None and observation.rerank.call.status is ProviderCallStatus.TIMEOUT
    assert observation.rerank.completion is not None and observation.rerank.completion.wait(1)
    assert provider.call_counts[ProviderOperation.RERANK] == 1
    assert observation.result.to_dict() == legacy.to_dict()
    with sqlite3.connect(service.paths.database_path) as connection:
        assert [tuple(row) for row in connection.execute(
            "SELECT outcome,provider_status FROM retrieval_provider_attempt_results"
        )] == [("failure", "timeout")]
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events WHERE action='failure' AND failure_kind='timeout'"
        ).fetchone()[0] == 1


@pytest.mark.parametrize("stage", ("admission", "settlement"))
def test_rerank_lock_preparation_cannot_restart_the_shared_deadline(
    runtime, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """Real lock contention; a logical clock isolates the rebased wait defect.

    Preparation cost is inside the existing 180ms request, not a new
    allowance. This does not reproduce or explain a historical wall-clock run.
    """
    import threading
    from types import SimpleNamespace

    from codex_memory.retrieval import service as retrieval_service
    from codex_memory.retrieval.deadline import DeadlineBudget
    from codex_memory.retrieval.provider import ModelComponent, ProviderCallStatus, ProviderInput, ProviderOperation
    from codex_memory.retrieval.provider_health import HealthWriteOutcome, ProviderHealthStore

    service, request, record, legacy, provider = _stalled_rerank_case(runtime)
    health = ProviderHealthStore(service.paths.database_path)
    decision = None
    if stage == "settlement":
        decision = health.acquire("a" * 64, ModelComponent.RERANKER, now_ms=0)
        assert decision.allowed
    temp_dir = service.paths.database_path.parent / "temp"
    acquired, release = threading.Event(), threading.Event()
    holder_errors: list[BaseException] = []

    def hold_real_lock() -> None:
        try:
            with retrieval_service.process_rebuild_lock(temp_dir):
                acquired.set()
                assert release.wait(2), "fixture lock was not released"
        except BaseException as exc:
            holder_errors.append(exc)
            acquired.set()

    holder = threading.Thread(target=hold_real_lock)
    holder.start()
    current = [0 if stage == "admission" else 120_000_000]
    waits: list[float] = []
    original_directory_check = retrieval_service.require_safe_directory

    def checked_directory(path) -> None:
        original_directory_check(path)
        if path == temp_dir:
            current[0] += 80_000_000 if stage == "admission" else 20_000_000

    def consume_lock_wait(seconds: float) -> None:
        waits.append(seconds)
        current[0] += round(seconds * 1_000_000_000)

    try:
        assert acquired.wait(1) and not holder_errors
        before = tuple(runtime.connection.iterdump())
        monkeypatch.setattr(retrieval_service, "require_safe_directory", checked_directory)
        monkeypatch.setattr(retrieval_service, "time", SimpleNamespace(
            monotonic=lambda: current[0] / 1_000_000_000,
            monotonic_ns=lambda: current[0], sleep=consume_lock_wait,
        ))
        if stage == "admission":
            observation = service.query_with_rerank(
                request, provider=provider,
                documents={record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)},
                deadline_ms=180, now_ns=lambda: current[0],
                health_store=health, base_result=legacy,
            )
            assert observation.result.to_dict() == legacy.to_dict()
        else:
            outcome = health.record_failure(
                decision, ProviderCallStatus.TIMEOUT, now_ms=120,
                deadline=DeadlineBudget(0, 170_000_000), now_ns=lambda: current[0],
            )
            assert outcome is HealthWriteOutcome.NOT_PERSISTED
        cutoff_ns = 120_000_000 if stage == "admission" else 170_000_000
        assert current[0] <= cutoff_ns, "lock preparation restarted the shared cutoff"
        assert waits, "the real OS lock must actually reject the contender"
        assert not provider.started.is_set()
        assert provider.call_counts[ProviderOperation.RERANK] == 0
        assert tuple(runtime.connection.iterdump()) == before
        with sqlite3.connect(service.paths.database_path) as connection:
            assert connection.execute("SELECT count(*) FROM retrieval_provider_attempt_results").fetchone()[0] == 0
            assert connection.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] == 0
    finally:
        release.set()
        provider.release.set()
        holder.join(2)
    assert not holder.is_alive() and not holder_errors


def test_optional_rerank_admission_cutoff_never_dispatches_or_records_provider_timeout(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real admission at the provider cutoff is released, never fake timeout."""
    from codex_memory.retrieval.provider import ProviderInput, ProviderCallStatus, ProviderOperation
    from codex_memory.retrieval.provider_health import ProviderHealthStore

    service, request, record, legacy, provider = _stalled_rerank_case(runtime)
    clock_ns = [0]
    _synchronize_logical_rerank_workers(provider, clock_ns, monkeypatch)
    health = ProviderHealthStore(service.paths.database_path)
    acquire = health.acquire
    seen = []
    before = tuple(runtime.connection.iterdump())

    def admitted_at_cutoff(*args, **kwargs):
        decision = acquire(*args, **kwargs)
        assert decision.allowed
        seen.append((kwargs["deadline"].started_ns, kwargs["deadline"].deadline_ns))
        clock_ns[0] = kwargs["deadline"].deadline_ns
        return decision

    monkeypatch.setattr(health, "acquire", admitted_at_cutoff)
    observation = service.query_with_rerank(request, provider=provider,
        documents={record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)},
        deadline_ms=180, now_ns=lambda: clock_ns[0], health_store=health, base_result=legacy)

    assert seen == [(0, 120_000_000)]
    assert not provider.started.is_set()
    assert provider.call_counts[ProviderOperation.RERANK] == 0
    assert observation.rerank is not None and observation.rerank.call is not None
    # Preserve the established aggregate/underlying distinction. Only the
    # underlying call and durable result identify confirmed provider failure.
    assert observation.rerank.status is ProviderCallStatus.TIMEOUT
    assert observation.rerank.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert observation.rerank.completion is not None and observation.rerank.completion.is_set()
    assert observation.result.to_dict() == legacy.to_dict()
    assert tuple(runtime.connection.iterdump()) == before
    with sqlite3.connect(service.paths.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM retrieval_provider_attempt_admissions").fetchone()[0] == 1
        assert [tuple(row) for row in connection.execute(
            "SELECT outcome,provider_status FROM retrieval_provider_attempt_results"
        )] == [("released", None)]
        assert connection.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] == 0


def test_timeout_keeps_provider_admission_busy_until_worker_finishes(runtime) -> None:
    import threading
    import time
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.provider import ModelComponent, ProviderInput, ProviderCallStatus
    from codex_memory.retrieval.provider_health import ProviderHealthStore
    from retrieval_provider_fakes import FakeProvider

    class Stalled(FakeProvider):
        def __init__(self) -> None:
            super().__init__(); self.started = threading.Event(); self.release = threading.Event()
        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            self.started.set(); self.release.wait(1)
            return super().rerank(query, documents, deadline_ms=deadline_ms)

    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="rerank-lease", statement="lease", value=None, evidence_uris=(), idempotency_key="rerank-lease")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="rerank-lease-active"); runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); assert service.rebuild().status.value == "current"
    request = RetrievalRequest("lease", "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root)))
    provider = Stalled(); documents = {record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)}
    health = ProviderHealthStore(service.paths.database_path, failure_threshold=10, cooldown_ms=10)
    first = service.query_with_rerank(request, provider=provider, documents=documents, deadline_ms=300, health_store=health)
    assert provider.started.is_set() and first.rerank is not None and first.rerank.status is ProviderCallStatus.TIMEOUT
    with sqlite3.connect(service.paths.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM retrieval_provider_health_events WHERE action = 'failure'").fetchone()[0] == 1
    second = service.query_with_rerank(request, provider=provider, documents=documents, deadline_ms=300, health_store=health)
    assert second.rerank is not None and second.rerank.status is ProviderCallStatus.BUSY
    provider.release.set(); time.sleep(0.03)
    third = health.acquire("0" * 64, ModelComponent.RERANKER, now_ms=time.monotonic_ns() // 1_000_000)
    assert third.allowed is True


def test_total_budget_starts_before_deterministic_query(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.provider import ProviderInput, ProviderCallStatus
    from codex_memory.retrieval.provider_health import ProviderHealthStore
    from retrieval_provider_fakes import FakeProvider

    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="rerank-budget", statement="budget", value=None, evidence_uris=(), idempotency_key="rerank-budget")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="rerank-budget-active"); runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); assert service.rebuild().status.value == "current"
    request = RetrievalRequest("budget", "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root)))
    original = service.query
    def delayed_query(*args, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.03); return original(*args, **kwargs)
    monkeypatch.setattr(service, "query", delayed_query)
    provider = FakeProvider()

    observation = service.query_with_rerank(request, provider=provider, documents={record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)}, deadline_ms=20, health_store=ProviderHealthStore(service.paths.database_path))

    assert observation.rerank is not None and observation.rerank.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert sum(provider.call_counts.values()) == 0


def test_query_json_is_one_line_canonical_and_excludes_query_text() -> None:
    """Would fail if offline query output leaked prompt text or formatting drifted."""
    from codex_memory.retrieval.contracts import RetrievalQueryResult, RetrievalStatus

    value = serialize_query_result(RetrievalQueryResult(RetrievalStatus.CORRUPT, RetrievalMode.OFF, None, None, None, False, (), (), ()))
    assert value.endswith("\n") and value.count("\n") == 1
    assert "needle" not in value


def test_canonical_validation_ignores_unsafe_disposable_retrieval_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Would fail if a cache-only unsafe path was misclassified as canonical failure."""
    repo = tmp_path / "repo"; repo.mkdir()
    runtime = repo / ".codex-memory"; (runtime / "db").mkdir(parents=True)
    (runtime / "db" / "memory.sqlite3").write_bytes(b"sqlite")
    paths = RetrievalPaths.for_repo(repo)
    monkeypatch.setattr(RetrievalPaths, "_validate_existing_retrieval_state", lambda _self: (_ for _ in ()).throw(AssertionError("unsafe cache")))

    paths.validate_canonical()


def test_temporary_candidates_expose_the_exact_trust_label() -> None:
    """Would fail if consumers could not distinguish unverified temporary data."""
    candidate = RetrievedCandidate("temporary", "candidate", "decision", "project", "scope", "temporary", 1, "a" * 64, False, 0, 1, (), "UNVERIFIED TEMPORARY")
    assert candidate.trust_label == "UNVERIFIED TEMPORARY"


def test_stale_delta_uses_nfkc_exact_match_rank_without_full_snapshot_materialization() -> None:
    """Would fail if stale delta split raw text or projected unrelated canonical records."""
    from codex_memory.retrieval.service import _delta_rows

    connection = sqlite3.connect(":memory:"); connection.row_factory = sqlite3.Row
    connection.executescript("""
      CREATE TABLE memory_records(record_id TEXT, record_seq INTEGER, record_type TEXT, scope_kind TEXT, scope_key TEXT, topic_key TEXT, statement TEXT, value_json TEXT, created_at TEXT, content_sha256 TEXT);
      CREATE TABLE memory_current(record_id TEXT, state TEXT, last_event_seq INTEGER);
      CREATE TABLE memory_relations(relation_seq INTEGER, source_record_id TEXT, target_record_id TEXT);
      INSERT INTO memory_records VALUES ('plain',1,'decision','project','project','topic','Cafe','{}','2026-01-01T00:00:00Z','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
      INSERT INTO memory_records VALUES ('nfkc',2,'decision','project','project','topic','Ｃａｆｅ','{}','2026-01-01T00:00:00Z','bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb');
      INSERT INTO memory_current VALUES ('plain','active',5);
      INSERT INTO memory_current VALUES ('nfkc','active',4);
    """)

    rows = _delta_rows(connection, None, RetrievalRequest("Cafe", "session", "turn", "context", None, RetrievalScope("project")), ("plain", "nfkc"))
    assert [row["candidate_id"] for row in rows] == ["plain", "nfkc"]


def test_stale_delta_uses_shared_windows_scope_matching_for_spaced_paths() -> None:
    """Would fail if stale records used literal instead of shared scope comparison."""
    from codex_memory.retrieval.service import _delta_rows

    connection = sqlite3.connect(":memory:"); connection.row_factory = sqlite3.Row
    connection.executescript("""
      CREATE TABLE memory_records(record_id TEXT, record_seq INTEGER, record_type TEXT, scope_kind TEXT, scope_key TEXT, topic_key TEXT, statement TEXT, value_json TEXT, created_at TEXT, content_sha256 TEXT);
      CREATE TABLE memory_current(record_id TEXT, state TEXT, last_event_seq INTEGER);
      CREATE TABLE memory_relations(relation_seq INTEGER, source_record_id TEXT, target_record_id TEXT);
      INSERT INTO memory_records VALUES ('changed',1,'decision','project','c:\\repo with spaces','topic','needle','{}','2026-01-01T00:00:00Z','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
      INSERT INTO memory_current VALUES ('changed','active',1);
    """)

    request = RetrievalRequest("needle", "session", "turn", "context", None, RetrievalScope(r"C:\Repo With Spaces"))
    assert [row["candidate_id"] for row in _delta_rows(connection, None, request, ("changed",))] == ["changed"]


def test_current_query_is_repeatable_read_only_and_explain_matches_selected(runtime) -> None:
    """Would fail if query mutated either database or exposed a trace inconsistent with candidates."""
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.service import RetrievalService

    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="needle", statement="needle result", value=None, evidence_uris=(), idempotency_key="query-decision")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="query-decision-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root))
    assert service.rebuild().status.value == "current"
    request = RetrievalRequest("needle", "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root)))
    before = runtime.paths.db_path.read_bytes()
    first = service.query(request); second = service.query(request)

    assert first.to_dict() == second.to_dict()
    assert runtime.paths.db_path.read_bytes() == before
    assert [candidate.candidate_id for candidate in first.candidates] == [record.record_id]
    assert {trace.candidate_id for trace in first.trace if trace.selected} == {candidate.candidate_id for candidate in first.candidates}


def test_cache_path_failure_does_not_hide_verified_canonical_highwater(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    """Would fail if disposable-cache validation was allowed to erase canonical truth."""
    from codex_memory.paths import RuntimePathError
    from codex_memory.retrieval.service import RetrievalService

    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    monkeypatch.setattr(RetrievalService, "_validate_retrieval_paths", lambda _self: (_ for _ in ()).throw(RuntimePathError("unsafe cache")))
    result = RetrievalService(paths).query(RetrievalRequest(None, "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root))))

    assert result.canonical_highwater is not None
    assert result.degraded_reasons == ("path_unsafe",)


def test_public_query_uses_changed_active_rows_when_generation_is_stale(runtime) -> None:
    """Would fail if a stale query suppressed changed IDs without directly retrieving them."""
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.service import RetrievalService

    scope = str(runtime.paths.repo_root)
    first = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="old", statement="old", value=None, evidence_uris=(), idempotency_key="stale-old")
    apply_action(runtime.connection, record_id=first.record_id, action="activate", idempotency_key="stale-old-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); assert service.rebuild().status.value == "current"
    changed = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="fresh", statement="fresh needle", value=None, evidence_uris=(), idempotency_key="stale-fresh")
    apply_action(runtime.connection, record_id=changed.record_id, action="activate", idempotency_key="stale-fresh-active")
    runtime.connection.commit()

    result = service.query(RetrievalRequest("needle", "session", "turn", "context", None, RetrievalScope(scope)))
    assert result.status.value == "stale"
    assert result.degraded_reasons[-1].value == "stale_delta_used"
    assert result.candidates[0].candidate_id == changed.record_id


def test_stale_delta_overflow_returns_no_cache_candidates(runtime) -> None:
    """Would fail if an overflowed stale cache still claimed ordinary cache results."""
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.contracts import RetrievalPolicy
    from codex_memory.retrieval.service import RetrievalService

    scope = str(runtime.paths.repo_root)
    base = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="base", statement="base", value=None, evidence_uris=(), idempotency_key="overflow-base")
    apply_action(runtime.connection, record_id=base.record_id, action="activate", idempotency_key="overflow-base-active"); runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); assert service.rebuild().status.value == "current"
    for index in range(2):
        record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key=f"changed-{index}", statement="changed", value=None, evidence_uris=(), idempotency_key=f"overflow-{index}")
        apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key=f"overflow-active-{index}")
    runtime.connection.commit()

    result = service.query(RetrievalRequest("changed", "session", "turn", "context", None, RetrievalScope(scope)), policy=RetrievalPolicy(stale_delta_limit=1))
    assert result.candidates == ()
    assert result.degraded_reasons[-1].value == "stale_delta_exceeded"


def test_explicit_semantic_inputs_join_real_query_fusion_without_provider_calls(runtime) -> None:
    """Would fail if verified semantic rows were not reachable from the real query path."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.retrieval.contracts import CandidateChannel
    from codex_memory.retrieval.provider import ProviderOperation
    from codex_memory.retrieval.service import SemanticQueryInputs
    from codex_memory.structured_ledger import apply_action, propose_record

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-query", statement="semantic query target", value=None, evidence_uris=(), idempotency_key="semantic-query")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="semantic-query-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root))
    provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"
    request = RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(scope))

    legacy = service.query(request)
    semantic = service.query(
        request,
        semantic_query=SemanticQueryInputs((1.0, 1.0, 1.0, 1.0), (("decision", 1.0),)),
    )

    assert legacy.to_dict() == service.query(request, semantic_query=None).to_dict()
    trace = next(item for item in semantic.trace if item.candidate_id == record.record_id)
    assert CandidateChannel.DENSE in trace.source_channels
    assert CandidateChannel.SPARSE in trace.source_channels
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    assert provider.call_counts[ProviderOperation.SPARSE_QUERY] == 0


def test_semantic_input_failure_degrades_only_that_component(runtime) -> None:
    """Would fail if an invalid dense input hid a still-valid sparse semantic path."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.retrieval.contracts import CandidateChannel, DegradedReason
    from codex_memory.retrieval.service import SemanticQueryInputs
    from codex_memory.structured_ledger import apply_action, propose_record

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-degrade", statement="semantic degrade target", value=None, evidence_uris=(), idempotency_key="semantic-degrade")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="semantic-degrade-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"
    result = service.query(
        RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(scope)),
        semantic_query=SemanticQueryInputs((0.0, 0.0, 0.0, 0.0), (("decision", 1.0),)),
    )

    trace = next(item for item in result.trace if item.candidate_id == record.record_id)
    assert CandidateChannel.SPARSE in trace.source_channels
    assert CandidateChannel.DENSE not in trace.source_channels
    assert DegradedReason.SEMANTIC_DENSE_UNAVAILABLE in result.degraded_reasons


def test_semantic_query_never_bypasses_current_scope_eligibility(runtime) -> None:
    """Would fail if an indexed foreign-scope row became visible through semantic scoring."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.retrieval.service import SemanticQueryInputs
    from codex_memory.structured_ledger import apply_action, propose_record

    foreign_scope = str(runtime.paths.repo_root / "outside")
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=foreign_scope, topic_key="foreign-semantic", statement="foreign semantic target", value=None, evidence_uris=(), idempotency_key="foreign-semantic")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="foreign-semantic-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"

    result = service.query(
        RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(str(runtime.paths.repo_root))),
        semantic_query=SemanticQueryInputs((1.0, 1.0, 1.0, 1.0), (("decision", 1.0),)),
    )

    assert record.record_id not in [candidate.candidate_id for candidate in result.candidates]


def test_stale_semantic_row_never_bypasses_current_lifecycle(runtime) -> None:
    """Would fail if an old semantic generation resurrected an invalidated record."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.retrieval.service import SemanticQueryInputs
    from codex_memory.structured_ledger import apply_action, propose_record

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-lifecycle", statement="semantic lifecycle target", value=None, evidence_uris=(), idempotency_key="semantic-lifecycle")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="semantic-lifecycle-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"
    apply_action(runtime.connection, record_id=record.record_id, action="invalidate", idempotency_key="semantic-lifecycle-invalidate")
    runtime.connection.commit()

    result = service.query(
        RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(scope)),
        semantic_query=SemanticQueryInputs((1.0, 1.0, 1.0, 1.0), (("decision", 1.0),)),
    )

    assert record.record_id not in [candidate.candidate_id for candidate in result.candidates]


def test_semantic_manifest_must_match_its_active_append_only_metadata(runtime) -> None:
    """Would fail if a self-consistent replacement manifest could change semantic ranks."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import build_semantic_manifest, verify_semantic_generation
    from codex_memory.retrieval.service import SemanticQueryInputs
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.contracts import CandidateChannel, DegradedReason

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-binding", statement="semantic binding target", value=None, evidence_uris=(), idempotency_key="semantic-binding")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="semantic-binding-active")
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths); provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"
    with sqlite3.connect(paths.database_path) as cache:
        semantic_id = cache.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state").fetchone()[0]
    semantic_paths = SemanticGenerationPaths.for_generation(paths.generations_dir, semantic_id)
    manifest = verify_semantic_generation(semantic_paths)
    replacement = build_semantic_manifest(
        paths=semantic_paths, projection_generation_id=manifest.projection_generation_id,
        canonical_highwater=manifest.canonical_highwater, indexed_highwater=manifest.indexed_highwater,
        provider_fingerprint=manifest.provider_fingerprint,
        dense_model_fingerprint=manifest.dense_model_fingerprint,
        sparse_model_fingerprint=manifest.sparse_model_fingerprint,
        reranker_model_fingerprint=manifest.reranker_model_fingerprint,
        preprocessing_fingerprint=manifest.preprocessing_fingerprint,
        build_config_sha256=manifest.build_config_sha256,
        created_at="2026-01-01T00:00:00.000000Z",
    )
    semantic_paths.manifest_path.write_bytes(replacement.to_json_bytes())

    result = service.query(
        RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(scope)),
        semantic_query=SemanticQueryInputs((1.0, 1.0, 1.0, 1.0), (("decision", 1.0),)),
    )

    assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE in result.degraded_reasons
    assert all(CandidateChannel.DENSE not in trace.source_channels and CandidateChannel.SPARSE not in trace.source_channels for trace in result.trace)


def test_historical_semantic_generation_only_contributes_the_current_unchanged_subset(runtime) -> None:
    """Would fail if a verified older semantic generation was disabled instead of safely subset-filtered."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.retrieval.contracts import CandidateChannel, DegradedReason
    from codex_memory.retrieval.service import SemanticQueryInputs
    from codex_memory.structured_ledger import apply_action, propose_record

    scope = str(runtime.paths.repo_root)
    stable = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-stable", statement="semantic stable target", value=None, evidence_uris=(), idempotency_key="semantic-stable")
    apply_action(runtime.connection, record_id=stable.record_id, action="activate", idempotency_key="semantic-stable-active")
    changed = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-changed", statement="semantic changed old target", value=None, evidence_uris=(), idempotency_key="semantic-changed")
    apply_action(runtime.connection, record_id=changed.record_id, action="activate", idempotency_key="semantic-changed-active")
    runtime.connection.commit()
    service = RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)); provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"
    replacement = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-changed", statement="semantic changed replacement", value=None, evidence_uris=(), idempotency_key="semantic-replacement")
    apply_action(runtime.connection, record_id=replacement.record_id, action="activate", idempotency_key="semantic-replacement-active")
    apply_action(runtime.connection, record_id=changed.record_id, action="supersede", replacement_record_id=replacement.record_id, idempotency_key="semantic-changed-supersede")
    newer = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="semantic-new", statement="semantic new target", value=None, evidence_uris=(), idempotency_key="semantic-new")
    apply_action(runtime.connection, record_id=newer.record_id, action="activate", idempotency_key="semantic-new-active")
    runtime.connection.commit()
    assert service.catch_up().status.value == "current"

    result = service.query(
        RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(scope)),
        semantic_query=SemanticQueryInputs((1.0, 1.0, 1.0, 1.0), (("decision", 1.0),)),
    )

    traces = {trace.candidate_id: trace for trace in result.trace}
    assert CandidateChannel.DENSE in traces[stable.record_id].source_channels
    assert CandidateChannel.SPARSE in traces[stable.record_id].source_channels
    assert changed.record_id not in traces
    assert CandidateChannel.DENSE not in traces[replacement.record_id].source_channels
    assert CandidateChannel.SPARSE not in traces[replacement.record_id].source_channels
    assert CandidateChannel.DENSE not in traces[newer.record_id].source_channels
    assert CandidateChannel.SPARSE not in traces[newer.record_id].source_channels
    assert DegradedReason.SEMANTIC_INDEX_STALE in result.degraded_reasons


@pytest.mark.parametrize("component", ("dense", "sparse"))
def test_partial_semantic_component_counts_must_match_the_derived_store(runtime, component: str) -> None:
    """Would fail if partial component metadata could claim fewer indexed documents."""
    from retrieval_provider_fakes import FakeProvider
    from codex_memory.ledger_common import canonical_json
    from codex_memory.retrieval.contracts import IndexHighwater
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError,
        connect_semantic_database,
        verify_semantic_generation,
    )
    from codex_memory.retrieval.service import _bound_semantic_components
    from codex_memory.structured_ledger import apply_action, propose_record

    scope = str(runtime.paths.repo_root)
    for index in range(2):
        record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key=f"partial-semantic-{index}", statement=f"partial semantic {index}", value=None, evidence_uris=(), idempotency_key=f"partial-semantic-{index}")
        apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key=f"partial-semantic-{index}-active")
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths); provider = FakeProvider()
    assert service.rebuild_semantic(provider, _semantic_config(provider)).status.value == "current"

    with sqlite3.connect(paths.database_path) as cache:
        cache.row_factory = sqlite3.Row
        semantic_id = cache.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state").fetchone()[0]
        trigger = cache.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name = 'retrieval_semantic_components_reject_update'").fetchone()
        assert trigger is not None
        cache.execute(f"DROP TRIGGER {trigger['name']}")
        cache.execute(
            """UPDATE retrieval_semantic_components
               SET status = 'partial', indexed_document_count = 1, coverage_highwater_json = ?
               WHERE semantic_generation_id = ? AND component = ?""",
            (canonical_json(IndexHighwater().to_dict()), semantic_id, component),
        )
        cache.execute(trigger["sql"])
        cache.commit()
        semantic_paths = SemanticGenerationPaths.for_generation(paths.generations_dir, semantic_id)
        manifest = verify_semantic_generation(semantic_paths)
        semantic = connect_semantic_database(semantic_paths, readonly=True)
        try:
            with pytest.raises(SemanticStoreError, match="semantic component coverage is stale"):
                _bound_semantic_components(cache, semantic, semantic_id, manifest, manifest.projection_generation_id)
        finally:
            semantic.close()
