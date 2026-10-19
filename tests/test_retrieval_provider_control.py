from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from codex_memory.retrieval.cache_database import connect_cache_database, initialize_schema
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.provider import (
    DenseResponse,
    ModelComponent,
    ProviderCall,
    ProviderCallStatus,
    ProviderInput,
    ProviderOperation,
)
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider_health import HealthWriteOutcome, ProviderHealthStore
from retrieval_provider_fakes import FakeProvider


def _health_store(tmp_path: Path, *, cooldown_ms: int = 100) -> ProviderHealthStore:
    database = tmp_path / "retrieval.sqlite3"
    with connect_cache_database(database) as connection:
        initialize_schema(connection)
    return ProviderHealthStore(database, failure_threshold=2, cooldown_ms=cooldown_ms)


def _fingerprint(provider: FakeProvider) -> str:
    capabilities = provider.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    return capabilities.fingerprint


def test_dense_sparse_and_rerank_share_one_absolute_deadline(tmp_path: Path) -> None:
    class Clock:
        value = 0

        def __call__(self) -> int:
            return self.value

    class TimedProvider(FakeProvider):
        def __init__(self, clock: Clock) -> None:
            super().__init__()
            self.clock = clock
            self.deadlines: list[int] = []

        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            self.deadlines.append(deadline_ms)
            self.clock.value += 20_000_000
            return super().embed_query(item, deadline_ms=deadline_ms)

        def sparse_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            self.deadlines.append(deadline_ms)
            self.clock.value += 20_000_000
            return super().sparse_query(item, deadline_ms=deadline_ms)

        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            self.deadlines.append(deadline_ms)
            self.clock.value += 20_000_000
            return super().rerank(query, documents, deadline_ms=deadline_ms)

    clock = Clock()
    provider = TimedProvider(clock)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(150, now_ns=clock),
        _health_store(tmp_path),
        _fingerprint(provider),
        now_ns=clock,
    )
    item = ProviderInput.from_text("query", 0, "query")

    dense = controller.dense_query(item, dimension=4)
    sparse = controller.sparse_query(item)
    rerank = controller.rerank(item, (item,))

    assert [dense.call.status, sparse.call.status, rerank.call.status] == [
        ProviderCallStatus.OK,
        ProviderCallStatus.OK,
        ProviderCallStatus.OK,
    ]
    assert provider.deadlines == [90, 70, 50]


def test_expired_and_circuit_open_calls_never_invoke_provider(tmp_path: Path) -> None:
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path, cooldown_ms=3_600_000_000)
    item = ProviderInput.from_text("query", 0, "query")
    expired = ProviderCallController(
        provider, DeadlineBudget(0, 0), store, fingerprint, now_ns=lambda: 0
    ).dense_query(item, dimension=4)

    wall_now = time.time_ns() // 1_000_000
    for now_ms in (wall_now, wall_now + 2):
        decision = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=now_ms)
        store.record_failure(decision, ProviderCallStatus.TIMEOUT, now_ms=now_ms + 1)
    opened = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    ).dense_query(item, dimension=4)

    assert expired.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert opened.call.status is ProviderCallStatus.CIRCUIT_OPEN
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0


def test_provider_without_absolute_deadline_contract_is_never_invoked(
    tmp_path: Path,
) -> None:
    class RelativeOnlyProvider(FakeProvider):
        absolute_deadline_aware = False

    provider = RelativeOnlyProvider()
    response = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=time.monotonic_ns),
        _health_store(tmp_path),
        _fingerprint(provider),
        now_ns=time.monotonic_ns,
    ).dense_query(ProviderInput.from_text("query", 0, "query"), dimension=4)

    assert response.call.status is ProviderCallStatus.UNAVAILABLE
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0


@pytest.mark.parametrize("remaining_ms", [1, 10, 20, 40, 50, 60])
@pytest.mark.parametrize("phase", ["before_admission", "after_admission"])
def test_short_remaining_budget_skips_provider_without_timeout_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remaining_ms: int, phase: str
) -> None:
    from contextlib import closing

    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    current = [0]
    clock = lambda: current[0]
    deadline = DeadlineBudget.start(100, now_ns=clock)
    original_acquire = store.acquire
    admission_calls: list[bool] = []

    def observed_acquire(*args, **kwargs):  # type: ignore[no-untyped-def]
        admission_calls.append(True)
        decision = original_acquire(*args, **kwargs)
        if phase == "after_admission":
            current[0] = (100 - remaining_ms) * 1_000_000
        return decision

    monkeypatch.setattr(store, "acquire", observed_acquire)
    if phase == "before_admission":
        current[0] = (100 - remaining_ms) * 1_000_000
    controller = ProviderCallController(
        provider, deadline, store, fingerprint, now_ns=clock
    )
    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )

    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    if phase == "before_admission":
        assert admission_calls == []
    with closing(connect_cache_database(store.database_path, readonly=True)) as db:
        assert db.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0
        unresolved = db.execute(
            """SELECT count(*)
               FROM retrieval_provider_attempt_admissions AS admission
               LEFT JOIN retrieval_provider_attempt_results AS result
                 ON result.attempt_id = admission.attempt_id
               WHERE result.attempt_id IS NULL"""
        ).fetchone()[0]
    decision = original_acquire(
        fingerprint, ModelComponent.DENSE,
        now_ms=time.time_ns() // 1_000_000,
    )
    if phase == "after_admission" and remaining_ms <= 10:
        assert unresolved == 1
        assert controller.health_available is False
        assert decision.allowed is False
        assert decision.status is ProviderCallStatus.BUSY
        assert decision.state == "unresolved"
    else:
        assert unresolved == 0
        assert decision.allowed
        store.release(decision)


def test_admission_deadline_preserves_independent_release_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-call admission must settle before the total Hook budget expires."""
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    current = [0]
    observed_deadlines: list[DeadlineBudget] = []
    release_deadlines: list[DeadlineBudget] = []
    original_acquire = store.acquire
    original_release = store.release

    def acquire_at_its_deadline(*args, **kwargs):  # type: ignore[no-untyped-def]
        observed_deadlines.append(kwargs["deadline"])
        decision = original_acquire(*args, **kwargs)
        current[0] = kwargs["deadline"].deadline_ns
        return decision

    def observed_release(*args, **kwargs):  # type: ignore[no-untyped-def]
        release_deadlines.append(kwargs["deadline"])
        return original_release(*args, **kwargs)

    monkeypatch.setattr(store, "acquire", acquire_at_its_deadline)
    monkeypatch.setattr(store, "release", observed_release)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        store,
        fingerprint,
        now_ns=lambda: current[0],
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )

    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    with connect_cache_database(store.database_path, readonly=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT outcome, provider_status "
                "FROM retrieval_provider_attempt_results"
            )
        ] == [("released", None)]
        assert connection.execute(
            """SELECT count(*)
               FROM retrieval_provider_attempt_admissions AS admission
               LEFT JOIN retrieval_provider_attempt_results AS result
                 ON result.attempt_id = admission.attempt_id
               WHERE result.attempt_id IS NULL"""
        ).fetchone()[0] == 0
    assert observed_deadlines == [DeadlineBudget(0, 40_000_000)]
    assert release_deadlines == [DeadlineBudget(0, 90_000_000)]


def test_worker_delayed_past_provider_deadline_never_invokes_or_records_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that never claimed the call is released, not timed out."""
    from codex_memory.retrieval import provider_control

    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    current = [0]
    deferred_targets = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):  # type: ignore[no-untyped-def]
            assert name == "codex-memory-provider-dense_query"
            assert daemon is True
            self.target = target

        def start(self) -> None:
            deferred_targets.append(self.target)
            current[0] = 40_000_000

    monkeypatch.setattr(provider_control.threading, "Thread", DeferredThread)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        store,
        fingerprint,
        now_ns=lambda: current[0],
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )
    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    assert len(deferred_targets) == 1

    deferred_targets[0]()

    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    with connect_cache_database(store.database_path, readonly=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT outcome, provider_status "
                "FROM retrieval_provider_attempt_results"
            )
        ] == [("released", None)]
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0


def test_worker_self_cancels_at_provider_deadline_without_timeout_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late worker wake-up cannot turn an unstarted call into a timeout."""
    from codex_memory.retrieval import provider_control

    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    current = [0]
    deferred_targets = []
    real_event_type = threading.Event

    class DeferredThread:
        def __init__(self, *, target, name, daemon):  # type: ignore[no-untyped-def]
            assert name == "codex-memory-provider-dense_query"
            assert daemon is True
            self.target = target

        def start(self) -> None:
            deferred_targets.append(self.target)

    class WakeWorkerAtDeadlineEvent:
        def __init__(self) -> None:
            self.inner = real_event_type()

        def is_set(self) -> bool:
            return self.inner.is_set()

        def set(self) -> None:
            self.inner.set()

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            assert timeout == 0.04
            current[0] = 40_000_000
            deferred_targets[0]()
            return self.inner.wait(0)

    monkeypatch.setattr(
        provider_control.threading, "Thread", DeferredThread
    )
    monkeypatch.setattr(
        provider_control.threading, "Event", WakeWorkerAtDeadlineEvent
    )
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        store,
        fingerprint,
        now_ns=lambda: current[0],
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )

    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    with connect_cache_database(store.database_path, readonly=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT outcome, provider_status "
                "FROM retrieval_provider_attempt_results"
            )
        ] == [("released", None)]
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0


def test_parent_timeout_cancels_claimed_worker_before_provider_method_call(
    tmp_path: Path,
) -> None:
    """A delayed method lookup cannot launch provider I/O after parent return."""
    lookup_started = threading.Event()
    release_lookup = threading.Event()

    class DelayedLookupProvider(FakeProvider):
        armed = False

        def __getattribute__(self, name):  # type: ignore[no-untyped-def]
            if name == "embed_query" and object.__getattribute__(self, "armed"):
                lookup_started.set()
                release_lookup.wait(2)
            return super().__getattribute__(name)

    provider = DelayedLookupProvider()
    fingerprint = _fingerprint(provider)
    provider.armed = True
    store = _health_store(tmp_path)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(150, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )
    try:
        assert lookup_started.is_set()
        assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
        assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    finally:
        release_lookup.set()

    assert controller.last_completion is not None
    assert controller.last_completion.wait(1)
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    with connect_cache_database(store.database_path, readonly=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT outcome, provider_status "
                "FROM retrieval_provider_attempt_results"
            )
        ] == [("released", None)]
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("response_kind", "expected_status", "expected_result", "expected_events"),
    (
        ("valid", ProviderCallStatus.DEADLINE_EXHAUSTED,
         [("released", None)], []),
        ("wrong-operation", ProviderCallStatus.TIMEOUT,
         [("failure", "timeout")], [("failure", "timeout")]),
        ("wrong-component", ProviderCallStatus.TIMEOUT,
         [("failure", "timeout")], [("failure", "timeout")]),
        ("nonempty", ProviderCallStatus.TIMEOUT,
         [("failure", "timeout")], [("failure", "timeout")]),
        ("incomplete", ProviderCallStatus.TIMEOUT,
         [("failure", "timeout")], [("failure", "timeout")]),
    ),
)
def test_completed_deadline_result_wins_only_after_exact_response_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response_kind: str,
    expected_status: ProviderCallStatus,
    expected_result: list[tuple[str, str | None]],
    expected_events: list[tuple[str, str | None]],
) -> None:
    """Only an exact completed deadline result may avoid timeout accounting."""
    from types import SimpleNamespace
    import codex_memory.retrieval.provider_control as provider_control

    provider_entered = threading.Event()
    release_provider = threading.Event()
    worker_finalized = threading.Event()
    current_ns = [0]

    class DeadlineProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            provider_entered.set()
            assert release_provider.wait(1)
            response = super().embed_query(item, deadline_ms=deadline_ms)
            if response_kind == "wrong-operation":
                return DenseResponse(
                    ProviderCall(
                        ProviderOperation.DENSE_DOCUMENTS,
                        ModelComponent.DENSE,
                        ProviderCallStatus.DEADLINE_EXHAUSTED,
                        response.call.provider_fingerprint,
                        response.call.model_fingerprint,
                        response.call.elapsed_ms,
                        response.call.reason,
                    ),
                    (),
                )
            if response_kind == "wrong-component":
                call = object.__new__(ProviderCall)
                for name, value in (
                    ("operation", ProviderOperation.DENSE_QUERY),
                    ("component", ModelComponent.SPARSE),
                    ("status", ProviderCallStatus.DEADLINE_EXHAUSTED),
                    ("provider_fingerprint", response.call.provider_fingerprint),
                    ("model_fingerprint", response.call.model_fingerprint),
                    ("elapsed_ms", response.call.elapsed_ms),
                    ("reason", response.call.reason),
                ):
                    object.__setattr__(call, name, value)
                forged = object.__new__(DenseResponse)
                object.__setattr__(forged, "call", call)
                object.__setattr__(forged, "items", ())
                return forged
            if response_kind == "nonempty":
                forged = object.__new__(DenseResponse)
                object.__setattr__(forged, "call", response.call)
                object.__setattr__(forged, "items", (object(),))
                return forged
            if response_kind == "incomplete":
                return object.__new__(DenseResponse)
            return response

    class InterleavingLock:
        """Finish the worker after wait=False but before the parent reads state."""

        def __init__(self) -> None:
            self.inner = threading.Lock()
            self.worker_acquires = 0

        def __enter__(self):  # type: ignore[no-untyped-def]
            assert self.acquire()
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            self.release()

        def acquire(self, blocking=True):  # type: ignore[no-untyped-def]
            is_worker = threading.current_thread().name.startswith(
                "codex-memory-provider-"
            )
            if not is_worker and blocking is False:
                assert provider_entered.is_set()
                release_provider.set()
                assert worker_finalized.wait(1)
                current_ns[0] = 40_000_000
            acquired = self.inner.acquire(blocking)
            if acquired and is_worker:
                self.worker_acquires += 1
            return acquired

        def release(self) -> None:
            is_worker = threading.current_thread().name.startswith(
                "codex-memory-provider-"
            )
            final_release = is_worker and self.worker_acquires == 3
            self.inner.release()
            if final_release:
                worker_finalized.set()

    monkeypatch.setattr(
        provider_control,
        "threading",
        SimpleNamespace(
            Event=threading.Event,
            Lock=InterleavingLock,
            Thread=threading.Thread,
        ),
    )
    provider = DeadlineProvider(
        {ModelComponent.DENSE: ProviderCallStatus.DEADLINE_EXHAUSTED}
    )
    store = _health_store(tmp_path)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: current_ns[0]),
        store,
        _fingerprint(provider),
        now_ns=lambda: current_ns[0],
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )

    assert response.call.status is expected_status
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 1
    with connect_cache_database(store.database_path, readonly=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT outcome, provider_status "
                "FROM retrieval_provider_attempt_results"
            )
        ] == expected_result
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT action, failure_kind "
                "FROM retrieval_provider_health_events"
            )
        ] == expected_events


def test_parent_never_waits_for_contended_start_gate_and_unknown_expires_once(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    gate_entered = threading.Event()
    release_gate = threading.Event()
    worker_clock_calls = [0]
    wall_now = 1_800_000_000_000

    def clock() -> int:
        if threading.current_thread().name == "codex-memory-provider-dense_query":
            worker_clock_calls[0] += 1
            if worker_clock_calls[0] == 2:
                gate_entered.set()
                release_gate.wait(2)
        return time.monotonic_ns()

    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(150, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=clock,
        wall_clock_ms=lambda: wall_now,
    )

    started = time.monotonic()
    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )
    elapsed = time.monotonic() - started
    try:
        assert gate_entered.is_set()
        assert elapsed < 0.5
        assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
        assert controller.health_available is False
        assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
        with connect_cache_database(store.database_path, readonly=True) as connection:
            first_attempt = connection.execute(
                "SELECT attempt_id FROM retrieval_provider_attempt_admissions"
            ).fetchone()[0]
            assert connection.execute(
                "SELECT count(*) FROM retrieval_provider_attempt_results"
            ).fetchone()[0] == 0
    finally:
        release_gate.set()
    assert controller.last_completion is not None
    assert controller.last_completion.wait(1)

    next_decision = store.acquire(
        fingerprint,
        ModelComponent.DENSE,
        now_ms=wall_now + 30_001,
        operation=ProviderOperation.DENSE_QUERY,
    )
    store.release(next_decision)
    with connect_cache_database(store.database_path, readonly=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                """SELECT outcome, provider_status
                   FROM retrieval_provider_attempt_results
                   WHERE attempt_id = ?""",
                (first_attempt,),
            )
        ] == [("expired_unknown", None)]


@pytest.mark.parametrize(
    ("method", "operation"),
    (
        ("capabilities", ProviderOperation.CAPABILITIES),
        ("embed_query", ProviderOperation.DENSE_QUERY),
        ("sparse_query", ProviderOperation.SPARSE_QUERY),
        ("rerank", ProviderOperation.RERANK),
    ),
)
def test_sixty_one_ms_remaining_grants_exact_one_ms_provider_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, operation: ProviderOperation
) -> None:
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    current = [39_000_000]
    received: list[int] = []
    original = getattr(provider, method)

    def observed(*args, deadline_ms):  # type: ignore[no-untyped-def]
        received.append(deadline_ms)
        return original(*args, deadline_ms=deadline_ms)

    monkeypatch.setattr(provider, method, observed)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        store,
        fingerprint,
        now_ns=lambda: current[0],
    )
    item = ProviderInput.from_text("query", 0, "query")
    calls_before = provider.call_counts[operation]
    if operation is ProviderOperation.CAPABILITIES:
        response = controller.capabilities()
    elif operation is ProviderOperation.DENSE_QUERY:
        response = controller.dense_query(item, dimension=4)
    elif operation is ProviderOperation.SPARSE_QUERY:
        response = controller.sparse_query(item)
    else:
        response = controller.rerank(item, (item,))

    assert response.call.status is ProviderCallStatus.OK
    assert received == [1]
    assert provider.call_counts[operation] == calls_before + 1


def test_response_validation_completion_after_provider_deadline_is_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from codex_memory.retrieval import provider_control

    current = [0]
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    original = provider_control.checked_dense_response

    def late_validation(*args, **kwargs):  # type: ignore[no-untyped-def]
        response = original(*args, **kwargs)
        current[0] = 40_000_000
        return response

    monkeypatch.setattr(provider_control, "checked_dense_response", late_validation)
    response = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        _health_store(tmp_path),
        fingerprint,
        now_ns=lambda: current[0],
    ).dense_query(ProviderInput.from_text("query", 0, "query"), dimension=4)

    assert response.call.status is ProviderCallStatus.TIMEOUT


def test_post_admission_release_failure_is_internal_health_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    current = [0]
    original_acquire = store.acquire

    def acquire_then_exhaust(*args, **kwargs):  # type: ignore[no-untyped-def]
        decision = original_acquire(*args, **kwargs)
        current[0] = 40_000_000
        return decision

    def unavailable_release(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("private release cause")

    monkeypatch.setattr(store, "acquire", acquire_then_exhaust)
    monkeypatch.setattr(store, "release", unavailable_release)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        store,
        fingerprint,
        now_ns=lambda: current[0],
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )

    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert controller.last_health_outcome is HealthWriteOutcome.NOT_PERSISTED
    assert controller.health_available is False
    assert controller.health_diagnostics[-1].phase == "release_after_admission"
    assert controller.health_diagnostics[-1].code == "health_unavailable"
    assert controller.health_diagnostics[-1].cause_types == ("RuntimeError",)
    assert "private release cause" not in repr(controller.health_diagnostics[-1])


def test_record_failure_exception_is_structured_as_internal_health_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = FakeProvider({ModelComponent.DENSE: ProviderCallStatus.TIMEOUT})
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    original_release = store.release

    def unavailable_record(decision, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_release(decision, completion=kwargs.get("completion"))
        raise ValueError("private record cause")

    monkeypatch.setattr(store, "record_failure", unavailable_record)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    )

    response = controller.dense_query(
        ProviderInput.from_text("query", 0, "query"), dimension=4
    )

    assert response.call.status is ProviderCallStatus.TIMEOUT
    assert controller.last_health_outcome is HealthWriteOutcome.NOT_PERSISTED
    assert controller.health_available is False
    assert controller.health_diagnostics[-1].phase == "record_failure"
    assert controller.health_diagnostics[-1].code == "health_unavailable"
    assert controller.health_diagnostics[-1].cause_types == ("ValueError",)
    assert "private record cause" not in repr(controller.health_diagnostics[-1])


@pytest.mark.parametrize("operation", ("dense", "sparse"))
def test_stalled_call_is_discarded_and_keeps_health_ownership(
    tmp_path: Path, operation: str
) -> None:
    """Real-clock pair for the controlled logical-budget durability test below."""
    from contextlib import closing

    release = threading.Event()
    entered = 0
    entered_lock = threading.Lock()

    class StalledProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            nonlocal entered
            with entered_lock:
                entered += 1
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

        def sparse_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            nonlocal entered
            with entered_lock:
                entered += 1
            release.wait(1)
            return super().sparse_query(item, deadline_ms=deadline_ms)

    provider = StalledProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    item = ProviderInput.from_text("query", 0, "query")
    first = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    )
    try:
        started = time.monotonic()
        response = (
            first.dense_query(item, dimension=4)
            if operation == "dense"
            else first.sparse_query(item)
        )
        elapsed = time.monotonic() - started
        assert elapsed < 0.15
        assert response.call.status is ProviderCallStatus.TIMEOUT
        assert response.items == ()
        assert entered == 1
        assert first.last_health_outcome is HealthWriteOutcome.PERSISTED
        assert first.health_available is True
        # Inspect durable state before a second admission or releasing the
        # provider. This cannot be satisfied by a later bookkeeping write.
        with closing(connect_cache_database(store.database_path, readonly=True)) as db:
            assert [tuple(row) for row in db.execute(
                "SELECT outcome, provider_status FROM retrieval_provider_attempt_results"
            )] == [("failure", "timeout")]
            assert [tuple(row) for row in db.execute(
                "SELECT action, failure_kind FROM retrieval_provider_health_events"
            )] == [("failure", "timeout")]
        second = ProviderCallController(
            provider,
            DeadlineBudget.start(100, now_ns=time.monotonic_ns),
            store,
            fingerprint,
            now_ns=time.monotonic_ns,
        )
        blocked = (
            second.dense_query(item, dimension=4)
            if operation == "dense"
            else second.sparse_query(item)
        )
        assert blocked.call.status is ProviderCallStatus.BUSY
        assert entered == 1
    finally:
        release.set()
        if first.last_completion is not None:
            assert first.last_completion.wait(1)
    with closing(connect_cache_database(store.database_path, readonly=True)) as db:
        assert db.execute("SELECT count(*) FROM retrieval_provider_attempt_results").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] == 1


def test_one_provider_admission_is_shared_across_components_and_capabilities(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class StalledProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            entered.set()
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    provider = StalledProvider()
    fingerprint = _fingerprint(provider)
    capability_calls_before = provider.call_counts[ProviderOperation.CAPABILITIES]
    store = _health_store(tmp_path)
    item = ProviderInput.from_text("query", 0, "query")
    first = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns), store,
        fingerprint, now_ns=time.monotonic_ns,
    )
    thread = threading.Thread(target=lambda: first.dense_query(item, dimension=4))
    thread.start()
    assert entered.wait(1)

    second = ProviderCallController(
        provider, DeadlineBudget.start(100, now_ns=time.monotonic_ns), store,
        fingerprint, now_ns=time.monotonic_ns,
    )
    sparse = second.sparse_query(item)
    capabilities = second.capabilities()
    release.set()
    thread.join(1)

    assert sparse.call.status is ProviderCallStatus.BUSY
    assert capabilities.call.status is ProviderCallStatus.BUSY
    assert provider.call_counts[ProviderOperation.SPARSE_QUERY] == 0
    assert provider.call_counts[ProviderOperation.CAPABILITIES] == capability_calls_before


@pytest.mark.parametrize("operation", ("dense", "sparse"))
def test_unrelated_admission_io_cannot_hold_a_timeout_settlement_past_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """Two real stores and workers; only foreign validation has a finite barrier.

    Its 1000ms admission remains within its budget. The target's original
    100ms/150ms contract must not inherit that unrelated Python lock wait.
    This is a reachable contention regression, not the historical343ms trace.
    """
    from contextlib import closing

    target_dir, other_dir = tmp_path / "target", tmp_path / "other"
    target_dir.mkdir()
    other_dir.mkdir()
    store, other = _health_store(target_dir), _health_store(other_dir)
    other_entered, release_other = threading.Event(), threading.Event()
    provider_entered, release_provider = threading.Event(), threading.Event()
    target_done = threading.Event()
    results, errors, foreign_decisions = [], [], []
    actual_verify = ProviderHealthStore._verify_admission_database

    def verify_with_foreign_barrier(connection):  # type: ignore[no-untyped-def]
        actual_verify(connection)
        database_path = Path(connection.execute("PRAGMA database_list").fetchone()[2])
        if database_path == other.database_path:
            other_entered.set()
            assert release_other.wait(1), "foreign fixture was not released"

    def foreign_admission() -> None:
        try:
            decision = other.acquire("d" * 64, ModelComponent.DENSE,
                now_ms=time.time_ns() // 1_000_000,
                deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns),
                now_ns=time.monotonic_ns)
            foreign_decisions.append(decision)
            if decision.allowed:
                other.release(decision)
        except BaseException as exc:
            errors.append(exc)

    foreign = threading.Thread(target=foreign_admission)

    class StalledProvider(FakeProvider):
        def _stall(self) -> None:
            provider_entered.set()
            foreign.start()
            assert other_entered.wait(1), "fixture did not reach real foreign admission"
            assert release_provider.wait(1), "provider fixture was not released"

        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            self._stall()
            return super().embed_query(item, deadline_ms=deadline_ms)

        def sparse_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            self._stall()
            return super().sparse_query(item, deadline_ms=deadline_ms)

    provider = StalledProvider()
    fingerprint = _fingerprint(provider)
    item = ProviderInput.from_text("query", 0, "query")
    controller = None

    def target_call() -> None:
        nonlocal controller
        try:
            controller = ProviderCallController(provider,
                DeadlineBudget.start(100, now_ns=time.monotonic_ns), store,
                fingerprint, now_ns=time.monotonic_ns)
            started = time.monotonic()
            response = (controller.dense_query(item, dimension=4)
                if operation == "dense" else controller.sparse_query(item))
            results.append((response, time.monotonic() - started))
        except BaseException as exc:
            errors.append(exc)
        finally:
            target_done.set()

    target = threading.Thread(target=target_call)
    monkeypatch.setattr(ProviderHealthStore, "_verify_admission_database",
        staticmethod(verify_with_foreign_barrier))
    try:
        target.start()
        assert other_entered.wait(1), "provider/foreign admission never started"
        assert target_done.wait(0.15), "unrelated admission blocked timeout settlement"
        assert not errors
        response, elapsed = results[0]
        assert elapsed < 0.15
        assert response.call.status is ProviderCallStatus.TIMEOUT and response.items == ()
        assert controller.last_health_outcome is HealthWriteOutcome.PERSISTED
        assert controller.health_available and provider_entered.is_set()
        assert not release_other.is_set(), "test must prove completion before foreign I/O returns"
        with closing(connect_cache_database(store.database_path, readonly=True)) as db:
            assert [tuple(row) for row in db.execute(
                "SELECT outcome,provider_status FROM retrieval_provider_attempt_results"
            )] == [("failure", "timeout")]
            assert [tuple(row) for row in db.execute(
                "SELECT action,failure_kind FROM retrieval_provider_health_events"
            )] == [("failure", "timeout")]
    finally:
        release_other.set()
        release_provider.set()
        target.join(1)
        if foreign.ident is not None:
            foreign.join(1)
        if controller is not None and controller.last_completion is not None:
            assert controller.last_completion.wait(1)
        assert not target.is_alive() and not foreign.is_alive()
    assert not errors
    assert len(foreign_decisions) == 1 and foreign_decisions[0].allowed


def test_capabilities_respects_a_persisted_component_circuit(tmp_path: Path) -> None:
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path, cooldown_ms=3_600_000_000)
    wall_now = time.time_ns() // 1_000_000
    for now_ms in (wall_now, wall_now + 2):
        decision = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=now_ms)
        store.record_failure(decision, ProviderCallStatus.TIMEOUT, now_ms=now_ms + 1)
    calls_before = provider.call_counts[ProviderOperation.CAPABILITIES]
    response = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    ).capabilities()
    assert response.call.status is ProviderCallStatus.CIRCUIT_OPEN
    assert provider.call_counts[ProviderOperation.CAPABILITIES] == calls_before


def test_timeout_health_failure_recording_is_durable_within_total_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logical cutoff/reserve contract, not a physical100ms performance claim.

    Real-clock coverage remains in the dense/sparse stalled-call pair above.
    Here real workers are ordered so the intended confirmed-timeout branch,
    rather than a valid pre-dispatch expiry, is exercised deterministically.
    """
    from contextlib import closing
    from types import SimpleNamespace

    from codex_memory.retrieval import provider_control, provider_health

    release = threading.Event()
    provider_entered = threading.Event()
    current = [0]
    received_budgets: list[int] = []
    acquisition_deadlines: list[DeadlineBudget] = []
    recording_deadlines: list[DeadlineBudget] = []
    workers: list[threading.Thread] = []

    class StalledProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            received_budgets.append(int(deadline_ms))
            provider_entered.set()
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    class OrderedThread(threading.Thread):
        def start(self) -> None:
            workers.append(self)
            super().start()
            if self.name == "codex-memory-provider-dense_query":
                assert provider_entered.wait(1), "fixture never reached provider"
                current[0] = 40_000_000
            else:
                assert self.name == "codex-memory-health-verify"
                # Test-only logical scheduling: do not claim this one-second
                # coordination limit is the production healthI/O allowance.
                self.join(1)
                assert not self.is_alive(), "fixture health verifier did not finish"

    provider = StalledProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    item = ProviderInput.from_text("query", 0, "query")
    original_acquire, original_record = store.acquire, store.record_failure

    def observed_acquire(*args, **kwargs):  # type: ignore[no-untyped-def]
        acquisition_deadlines.append(kwargs["deadline"])
        return original_acquire(*args, **kwargs)

    def observed_record(*args, **kwargs):  # type: ignore[no-untyped-def]
        recording_deadlines.append(kwargs["deadline"])
        return original_record(*args, **kwargs)

    # Patch only the two modules' scheduling references, not global threading,
    # not their validators, and not any store outcome or SQLite operation.
    scheduling = SimpleNamespace(Thread=OrderedThread, Event=threading.Event,
        Lock=threading.Lock, get_ident=threading.get_ident, local=threading.local)
    monkeypatch.setattr(provider_control, "threading", scheduling)
    monkeypatch.setattr(provider_health, "threading", scheduling)
    monkeypatch.setattr(store, "acquire", observed_acquire)
    monkeypatch.setattr(store, "record_failure", observed_record)
    controller = ProviderCallController(
        provider, DeadlineBudget.start(100, now_ns=lambda: current[0]), store,
        fingerprint, now_ns=lambda: current[0],
    )
    try:
        response = controller.dense_query(item, dimension=4)
        assert response.call.status is ProviderCallStatus.TIMEOUT
        assert provider_entered.is_set()
        assert response.items == ()
        assert received_budgets == [40]
        assert acquisition_deadlines == [DeadlineBudget(0, 40_000_000)]
        assert recording_deadlines == [DeadlineBudget(0, 90_000_000)]
        assert controller.last_health_outcome is HealthWriteOutcome.PERSISTED
        assert controller.health_available is True
        with closing(connect_cache_database(store.database_path, readonly=True)) as connection:
            assert [
                tuple(row)
                for row in connection.execute(
                    """SELECT result.outcome, result.provider_status
                       FROM retrieval_provider_attempt_admissions AS admission
                       JOIN retrieval_provider_attempt_results AS result
                         ON result.attempt_id = admission.attempt_id"""
                )
            ] == [("failure", "timeout")]
            assert [
                tuple(row)
                for row in connection.execute(
                    """SELECT action, failure_kind
                       FROM retrieval_provider_health_events
                       ORDER BY event_seq"""
                )
            ] == [("failure", "timeout")]
    finally:
        release.set()
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
    assert controller.last_completion is not None
    assert controller.last_completion.wait(1)

    with closing(connect_cache_database(store.database_path, readonly=True)) as db:
        assert db.execute("SELECT count(*) FROM retrieval_provider_attempt_results").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] == 1


@pytest.mark.parametrize("startup_ms", (30, 35))
def test_health_verification_wait_recomputes_remaining_after_thread_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, startup_ms: int,
) -> None:
    """Scheduling/API seam, not a claim about the historical391ms sample.

    Real health workers and SQL reach validation. The controlled wait advances
    the same logical clock by its requested timeout while the verifier is held
    at a finite barrier. Startup consumes part of, never a fresh,90ms budget.
    """
    from contextlib import closing
    from types import SimpleNamespace

    from codex_memory.retrieval import provider_health

    current = [55_000_000]
    entered = threading.Event()
    release = threading.Event()
    provider_completed = threading.Event()
    workers: list[threading.Thread] = []
    requested_waits: list[float] = []
    original_verify = ProviderHealthStore._verify_health_database

    def held_verification(connection):  # type: ignore[no-untyped-def]
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_attempt_admissions"
        ).fetchone()[0] == 1
        entered.set()
        assert release.wait(1), "fixture did not release the real verifier"
        return original_verify(connection)

    class CompletedEvent:
        def __init__(self) -> None:
            self.event = threading.Event()

        def set(self) -> None:
            self.event.set()

        def wait(self, timeout: float) -> bool:
            assert not self.event.is_set(), "fixture verifier escaped its barrier"
            requested_waits.append(timeout)
            current[0] += round(timeout * 1_000_000_000)
            return False

    class OrderedThread(threading.Thread):
        def start(self) -> None:
            assert self.name == "codex-memory-health-verify"
            workers.append(self)
            super().start()
            assert entered.wait(1), "fixture never reached real validation read"
            current[0] += startup_ms * 1_000_000

    store = _health_store(tmp_path)
    decision = store.acquire(
        "a" * 64, ModelComponent.SPARSE, now_ms=55,
        operation=ProviderOperation.SPARSE_QUERY, completion=provider_completed,
    )
    assert decision.allowed
    scheduling = SimpleNamespace(**vars(threading))
    scheduling.Thread, scheduling.Event = OrderedThread, CompletedEvent
    monkeypatch.setattr(provider_health, "threading", scheduling)
    monkeypatch.setattr(
        ProviderHealthStore, "_verify_health_database", staticmethod(held_verification)
    )
    try:
        outcome = store.record_failure(
            decision, ProviderCallStatus.TIMEOUT, now_ms=55,
            completion=provider_completed, deadline=DeadlineBudget(0, 90_000_000),
            now_ns=lambda: current[0],
        )
        assert entered.is_set()
        assert outcome is HealthWriteOutcome.NOT_PERSISTED
        with closing(connect_cache_database(store.database_path, readonly=True)) as db:
            assert db.execute("SELECT count(*) FROM retrieval_provider_attempt_results").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] == 0
        assert requested_waits == ([0.005] if startup_ms == 30 else [])
        assert current[0] == 90_000_000
    finally:
        release.set()
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive(), "fixture left a validation worker running"
        provider_completed.set()
        store.release(decision)


def test_timeout_bookkeeping_does_not_wait_for_post_admission_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A slow health verifier after admission cannot extend the provider deadline."""
    from codex_memory.retrieval import provider_health

    release = threading.Event()

    class StalledProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    original_verify = provider_health.ProviderHealthStore._verify_health_database

    def slow_retained_verify(connection):  # type: ignore[no-untyped-def]
        time.sleep(1)
        return original_verify(connection)

    monkeypatch.setattr(
        provider_health.ProviderHealthStore,
        "_verify_health_database",
        staticmethod(slow_retained_verify),
    )
    provider = StalledProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    item = ProviderInput.from_text("query", 0, "query")

    started = time.monotonic()
    response = ProviderCallController(
        provider, DeadlineBudget.start(100, now_ns=time.monotonic_ns), store,
        fingerprint, now_ns=time.monotonic_ns,
    ).dense_query(item, dimension=4)
    elapsed = time.monotonic() - started

    assert response.call.status is ProviderCallStatus.TIMEOUT
    assert elapsed < 0.35
    with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0
    release.set()


def test_completed_provider_lease_is_not_closed_by_a_contending_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Only the controller thread that owns a retained SQLite lease may close it."""
    from codex_memory.retrieval import provider_control

    validation_entered = threading.Event()
    release_validation = threading.Event()
    original_validate = provider_control.checked_dense_response

    def blocked_validate(*args, **kwargs):  # type: ignore[no-untyped-def]
        validation_entered.set()
        assert release_validation.wait(2)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(provider_control, "checked_dense_response", blocked_validate)
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    )
    item = ProviderInput.from_text("query", 0, "query")
    first_result: dict[str, object] = {}

    def invoke_first() -> None:
        try:
            first_result["response"] = controller.dense_query(item, dimension=4)
        except BaseException as exc:
            first_result["error"] = exc

    owner = threading.Thread(target=invoke_first)
    owner.start()
    assert validation_entered.wait(2)
    try:
        contender = store.acquire(
            fingerprint, ModelComponent.DENSE, now_ms=time.time_ns() // 1_000_000
        )
    finally:
        release_validation.set()
        owner.join(2)

    assert not owner.is_alive()
    assert "error" not in first_result
    assert contender.allowed is False
    assert contender.status in {ProviderCallStatus.BUSY, ProviderCallStatus.UNAVAILABLE}
    after = store.acquire(
        fingerprint, ModelComponent.DENSE, now_ms=time.time_ns() // 1_000_000
    )
    assert after.allowed is True
    store.release(after)


def test_timeout_bookkeeping_is_synchronous_and_never_spawns_a_late_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A timeout is recorded before return and never by deferred work."""
    import codex_memory.retrieval.provider_health as provider_health

    release = threading.Event()
    provider_entered = threading.Event()
    health_recorded = threading.Event()
    health_outcomes: list[HealthWriteOutcome] = []

    class StalledProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            provider_entered.set()
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    original = provider_health.ProviderHealthStore.record_failure

    def observed(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        outcome = original(self, *args, **kwargs)
        health_outcomes.append(outcome)
        health_recorded.set()
        return outcome

    monkeypatch.setattr(provider_health.ProviderHealthStore, "record_failure", observed)
    provider = StalledProvider()
    store = _health_store(tmp_path)
    fingerprint = _fingerprint(provider)
    now_ns = lambda: 0
    # Fixture initialization is outside this state-machine budget. A fixed
    # monotonic clock makes only the provider wait consume the timeout path.
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=now_ns),
        store,
        fingerprint,
        now_ns=now_ns,
    )
    response = controller.dense_query(ProviderInput.from_text("query", 0, "query"), dimension=4)

    def persisted_events() -> list[tuple[str, str]]:
        with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
            return [
                tuple(row)
                for row in connection.execute(
                    """SELECT action, failure_kind
                       FROM retrieval_provider_health_events
                       ORDER BY event_seq"""
                )
            ]

    assert response.call.status is ProviderCallStatus.TIMEOUT
    assert health_recorded.is_set()
    assert health_outcomes == [HealthWriteOutcome.PERSISTED]
    assert persisted_events() == [("failure", "timeout")]
    time.sleep(0.08)
    assert persisted_events() == [("failure", "timeout")]
    release.set()
    assert controller.last_completion is not None
    assert controller.last_completion.wait(1)
    assert provider_entered.is_set()
    assert persisted_events() == [("failure", "timeout")]


def test_timeout_does_not_wait_for_a_blocked_read_only_health_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A blocked verifier cannot extend the Hook and closes its own connection."""
    from codex_memory.retrieval import provider_health

    release_provider = threading.Event()
    verification_entered = threading.Event()
    release_verification = threading.Event()
    verification_closed = threading.Event()

    class StalledProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            release_provider.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    original_connect = provider_health.connect_cache_database
    original_verify = provider_health.ProviderHealthStore._verify_health_database
    connections = 0

    class TrackingConnection:
        def __init__(self, connection):  # type: ignore[no-untyped-def]
            self.connection = connection

        def __getattr__(self, name):  # type: ignore[no-untyped-def]
            return getattr(self.connection, name)

        def close(self) -> None:
            try:
                self.connection.close()
            finally:
                verification_closed.set()

    def track_readonly_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal connections
        connections += 1
        connection = original_connect(*args, **kwargs)
        if kwargs.get("readonly") is True:
            return TrackingConnection(connection)
        return connection

    def block_verification(connection):  # type: ignore[no-untyped-def]
        verification_entered.set()
        assert release_verification.wait(2)
        return original_verify(connection)

    monkeypatch.setattr(provider_health, "connect_cache_database", track_readonly_connect)
    monkeypatch.setattr(
        provider_health.ProviderHealthStore,
        "_verify_health_database",
        staticmethod(block_verification),
    )
    provider = StalledProvider()
    # Test-only schema construction and capability-fixture lookup are not the
    # call being timed. Keep admission, the real blocked verifier, rollback and
    # connection cleanup inside the original200ms assertion below.
    store = _health_store(tmp_path)
    fingerprint = _fingerprint(provider)
    started = time.monotonic()
    # Freeze the injected budget clock so scheduler load cannot consume the
    # 100 ms branch-under-test before the deliberately blocked provider runs.
    response = ProviderCallController(
        provider, DeadlineBudget.start(100, now_ns=lambda: 0), store,
        fingerprint, now_ns=lambda: 0,
    ).dense_query(ProviderInput.from_text("query", 0, "query"), dimension=4)
    elapsed = time.monotonic() - started

    try:
        assert response.call.status is ProviderCallStatus.TIMEOUT
        assert elapsed < 0.2
        assert verification_entered.is_set()
        # Phase B owns a fresh writable connection; its bounded integrity
        # verifier is the third, read-only connection.
        assert connections == 3
        with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
            assert connection.execute(
                "SELECT count(*) FROM retrieval_provider_health_events"
            ).fetchone()[0] == 0
    finally:
        release_verification.set()
        release_provider.set()
    assert verification_closed.wait(2)


def test_timeout_never_appends_to_a_database_tampered_after_admission(
    tmp_path: Path,
) -> None:
    """A post-admission integrity change fails closed instead of receiving a timeout event."""
    import sqlite3

    database = tmp_path / "retrieval.sqlite3"
    release = threading.Event()

    class TamperingProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE retrieval_schema_meta SET value = '999' WHERE key = 'schema_version'"
                )
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    provider = TamperingProvider()
    response = ProviderCallController(
        provider, DeadlineBudget.start(100, now_ns=lambda: 0), _health_store(tmp_path),
        _fingerprint(provider), now_ns=lambda: 0,
    ).dense_query(ProviderInput.from_text("query", 0, "query"), dimension=4)

    assert response.call.status is ProviderCallStatus.TIMEOUT
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0
    release.set()


def test_timeout_never_appends_after_health_guard_trigger_is_removed(
    tmp_path: Path,
) -> None:
    """Post-admission health DDL corruption prevents timeout evidence writes."""
    import sqlite3

    database = tmp_path / "retrieval.sqlite3"
    release = threading.Event()

    class TriggerDroppingProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "DROP TRIGGER retrieval_provider_health_events_reject_update"
                )
            release.wait(1)
            return super().embed_query(item, deadline_ms=deadline_ms)

    provider = TriggerDroppingProvider()
    response = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        _health_store(tmp_path),
        _fingerprint(provider),
        now_ns=lambda: 0,
    ).dense_query(ProviderInput.from_text("query", 0, "query"), dimension=4)

    assert response.call.status is ProviderCallStatus.TIMEOUT
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0
    release.set()


def test_controller_uses_wall_clock_for_durable_cooldown_after_restart(tmp_path: Path) -> None:
    """A low post-restart monotonic uptime cannot prolong a stored cooldown."""
    provider = FakeProvider({ModelComponent.DENSE: ProviderCallStatus.TIMEOUT})
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path, cooldown_ms=100)
    wall_start = 1_700_000_000_000
    item = ProviderInput.from_text("query", 0, "query")
    for wall_now in (wall_start, wall_start + 1):
        response = ProviderCallController(
            provider, DeadlineBudget.start(100, now_ns=lambda: 0), store,
            fingerprint, now_ns=lambda: 0, wall_clock_ms=lambda: wall_now,
        ).dense_query(item, dimension=4)
        assert response.call.status is ProviderCallStatus.TIMEOUT
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
                if connection.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] >= 2:
                    break
            time.sleep(0.01)

    response = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=lambda: 0),
        store,
        fingerprint,
        now_ns=lambda: 0,
        wall_clock_ms=lambda: wall_start + 101,
    ).dense_query(item, dimension=4)

    assert response.call.status is ProviderCallStatus.TIMEOUT


def test_repeated_controller_timeouts_open_the_durable_circuit(tmp_path: Path) -> None:
    """Two client-wait timeouts deny later calls across controller instances."""
    import time

    class Slow(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            time.sleep(0.6)
            return super().embed_query(item, deadline_ms=deadline_ms)

    provider = Slow()
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path, cooldown_ms=3_600_000_000)
    item = ProviderInput.from_text("query", 0, "query")
    for _ in range(2):
        response = ProviderCallController(
            provider,
            DeadlineBudget.start(500, now_ns=time.monotonic_ns),
            store,
            fingerprint,
            now_ns=time.monotonic_ns,
        ).dense_query(item, dimension=4)
        assert response.call.status is ProviderCallStatus.TIMEOUT
        time.sleep(0.7)

    response = ProviderCallController(
        provider,
        DeadlineBudget.start(100, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
    ).dense_query(item, dimension=4)
    assert response.call.status is ProviderCallStatus.CIRCUIT_OPEN
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 2


@pytest.mark.parametrize(
    "status",
    (ProviderCallStatus.UNSUPPORTED, ProviderCallStatus.DISABLED, ProviderCallStatus.INPUT_TOO_LONG),
)
def test_non_fault_provider_outcomes_do_not_poison_circuit(
    tmp_path: Path, status: ProviderCallStatus
) -> None:
    provider = FakeProvider({ModelComponent.DENSE: status})
    fingerprint = _fingerprint(provider)
    store = _health_store(tmp_path)
    item = ProviderInput.from_text("query", 0, "query")

    responses = [
        ProviderCallController(
            provider,
            DeadlineBudget.start(100, now_ns=lambda: 0),
            store,
            fingerprint,
            now_ns=lambda: 0,
        ).dense_query(item, dimension=4)
        for _ in range(3)
    ]

    assert [response.call.status for response in responses] == [status] * 3
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 3
    with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0
