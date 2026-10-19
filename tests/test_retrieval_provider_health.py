from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from codex_memory.retrieval.cache_database import connect_cache_database, initialize_schema
from codex_memory.retrieval.provider import ModelComponent, ProviderCallStatus
from codex_memory.retrieval.provider_health import HealthWriteOutcome, ProviderHealthStore


def _store(tmp_path: Path) -> ProviderHealthStore:
    database = tmp_path / "retrieval.sqlite3"
    connection = connect_cache_database(database)
    try:
        initialize_schema(connection)
    finally:
        connection.close()
    return ProviderHealthStore(database, failure_threshold=2, cooldown_ms=100)


def test_health_store_opens_then_grants_exactly_one_half_open_probe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fingerprint = "a" * 64
    first = store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=1)
    store.record_failure(first, ProviderCallStatus.TIMEOUT, now_ms=2)
    second = store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=3)
    store.record_failure(second, ProviderCallStatus.TIMEOUT, now_ms=4)

    assert store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=50).status is ProviderCallStatus.CIRCUIT_OPEN
    probe = store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=105)
    assert probe.allowed is True
    assert store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=106).status is ProviderCallStatus.BUSY
    store.record_success(probe, now_ms=107)
    assert store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=108).allowed is True


def test_health_events_replay_after_store_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fingerprint = "b" * 64
    decision = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=1)
    store.record_failure(decision, ProviderCallStatus.UNAVAILABLE, now_ms=2)
    decision = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=3)
    store.record_failure(decision, ProviderCallStatus.UNAVAILABLE, now_ms=4)

    reopened = ProviderHealthStore(tmp_path / "retrieval.sqlite3", failure_threshold=2, cooldown_ms=100)
    assert reopened.acquire(fingerprint, ModelComponent.DENSE, now_ms=5).status is ProviderCallStatus.CIRCUIT_OPEN


def test_component_failure_isolated_and_success_closes_only_its_circuit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fingerprint = "c" * 64
    for now_ms in (1, 3):
        decision = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=now_ms)
        store.record_failure(decision, ProviderCallStatus.TIMEOUT, now_ms=now_ms + 1)

    assert store.acquire(fingerprint, ModelComponent.DENSE, now_ms=10).status is ProviderCallStatus.CIRCUIT_OPEN
    sparse = store.acquire(fingerprint, ModelComponent.SPARSE, now_ms=10)
    assert sparse.allowed is True
    store.record_success(sparse, now_ms=11)
    assert store.acquire(fingerprint, ModelComponent.SPARSE, now_ms=12).allowed is True


def test_cross_instance_admission_and_stale_half_open_completion_are_rejected(tmp_path: Path) -> None:
    first_store = _store(tmp_path)
    second_store = ProviderHealthStore(tmp_path / "retrieval.sqlite3", failure_threshold=2, cooldown_ms=100)
    fingerprint = "d" * 64
    first = first_store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=1)
    assert second_store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=1).status is ProviderCallStatus.BUSY
    first_store.record_failure(first, ProviderCallStatus.TIMEOUT, now_ms=2)
    second = second_store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=3)
    second_store.record_failure(second, ProviderCallStatus.TIMEOUT, now_ms=4)
    old_probe = first_store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=105)
    replacement_probe = second_store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=206)

    assert first_store.record_success(
        old_probe, now_ms=207
    ) is HealthWriteOutcome.NOT_PERSISTED
    assert first_store.acquire(fingerprint, ModelComponent.RERANKER, now_ms=207).status is ProviderCallStatus.BUSY
    second_store.record_success(replacement_probe, now_ms=208)


def test_independent_processes_grant_exactly_one_post_cooldown_probe(
    tmp_path: Path,
) -> None:
    """The durable half-open lease admits one process and denies all peers."""
    database = tmp_path / "retrieval.sqlite3"
    store = _store(tmp_path)
    fingerprint = "8" * 64
    first = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=1)
    store.record_failure(first, ProviderCallStatus.TIMEOUT, now_ms=2)
    second = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=3)
    store.record_failure(second, ProviderCallStatus.TIMEOUT, now_ms=4)
    start = tmp_path / "start"
    script = """
import json
from pathlib import Path
import sys
import time
from codex_memory.retrieval.provider import ModelComponent
from codex_memory.retrieval.provider_health import ProviderHealthStore

database, start = map(Path, sys.argv[1:])
deadline = time.monotonic() + 5
while not start.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(3)
    time.sleep(0.001)
decision = ProviderHealthStore(
    database, failure_threshold=2, cooldown_ms=100
).acquire("8" * 64, ModelComponent.DENSE, now_ms=105)
print(json.dumps({"allowed": decision.allowed, "status": decision.status.value}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(database), str(start)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(12)
    ]
    start.write_text("go", encoding="utf-8")
    completed = [process.communicate(timeout=10) for process in processes]

    assert all(process.returncode == 0 and stderr == "" for process, (_, stderr) in zip(processes, completed))
    outcomes = [json.loads(stdout) for stdout, _ in completed]
    assert sum(outcome["allowed"] for outcome in outcomes) == 1
    assert sum(outcome["status"] == "ok" for outcome in outcomes) == 1
    assert all(
        outcome["allowed"] or outcome["status"] in {"busy", "unavailable"}
        for outcome in outcomes
    )

    with connect_cache_database(database, readonly=True) as connection:
        actions = [
            row[0]
            for row in connection.execute(
                """SELECT action FROM retrieval_provider_health_events
                   WHERE action IN ('half_open', 'probe_acquired')
                   ORDER BY event_seq"""
            )
        ]
    assert actions == ["half_open", "probe_acquired"]


def test_health_store_failure_denies_without_admitting(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from codex_memory.retrieval import provider_health

    store = _store(tmp_path)
    monkeypatch.setattr(provider_health, "connect_cache_database", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")))

    result = store.acquire("e" * 64, ModelComponent.DENSE, now_ms=1)

    assert result.allowed is False
    assert result.status is ProviderCallStatus.UNAVAILABLE


def test_locked_health_database_denies_without_admitting(tmp_path: Path) -> None:
    """A contended durable store fails closed instead of calling the provider."""
    store = _store(tmp_path)
    lock = sqlite3.connect(store.database_path, isolation_level=None)
    lock.execute("PRAGMA busy_timeout=0")
    lock.execute("BEGIN EXCLUSIVE")
    try:
        result = store.acquire("7" * 64, ModelComponent.DENSE, now_ms=1)
    finally:
        lock.rollback()
        lock.close()

    assert result.allowed is False
    assert result.status is ProviderCallStatus.UNAVAILABLE


def test_health_acquire_budget_exhaustion_skips_late_mutation(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import time
    from codex_memory.retrieval.deadline import DeadlineBudget
    from codex_memory.retrieval import provider_health

    store = _store(tmp_path)
    original = ProviderHealthStore._verify_admission_database
    def blocked(connection):  # type: ignore[no-untyped-def]
        time.sleep(0.03); return original(connection)
    monkeypatch.setattr(
        ProviderHealthStore, "_verify_admission_database", staticmethod(blocked)
    )
    budget = DeadlineBudget.start(10, now_ns=time.monotonic_ns)

    result = store.acquire("f" * 64, ModelComponent.DENSE, now_ms=time.monotonic_ns() // 1_000_000, deadline=budget, now_ns=time.monotonic_ns)

    assert result.allowed is False
    assert result.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
        assert connection.execute("SELECT count(*) FROM retrieval_provider_health_events").fetchone()[0] == 0


def test_expired_health_write_reports_not_persisted_and_rolls_back(tmp_path: Path) -> None:
    """A failed health transaction is observable and leaves no durable history."""
    from codex_memory.retrieval.deadline import DeadlineBudget

    store = _store(tmp_path)
    fingerprint = "1" * 64
    decision = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=1)

    outcome = store.record_failure(
        decision,
        ProviderCallStatus.TIMEOUT,
        now_ms=2,
        deadline=DeadlineBudget(0, 0),
        now_ns=lambda: 0,
    )

    assert outcome is HealthWriteOutcome.NOT_PERSISTED
    with connect_cache_database(tmp_path / "retrieval.sqlite3", readonly=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events"
        ).fetchone()[0] == 0
    reopened = ProviderHealthStore(
        tmp_path / "retrieval.sqlite3", failure_threshold=2, cooldown_ms=100
    )
    unresolved = reopened.acquire(fingerprint, ModelComponent.DENSE, now_ms=3)
    assert unresolved.status is ProviderCallStatus.BUSY
    replacement = reopened.acquire(fingerprint, ModelComponent.DENSE, now_ms=102)
    assert replacement.allowed
    reopened.release(replacement)


def test_health_write_verification_failure_has_internal_phase_and_cause(
    monkeypatch, tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    from codex_memory.retrieval import provider_health

    store = _store(tmp_path)
    decision = store.acquire("2" * 64, ModelComponent.DENSE, now_ms=1)

    def corrupt_verification(_connection):  # type: ignore[no-untyped-def]
        raise sqlite3.DatabaseError("private verification cause")

    monkeypatch.setattr(provider_health, "verify_cache_database", corrupt_verification)
    outcome = store.record_failure(
        decision, ProviderCallStatus.TIMEOUT, now_ms=2
    )

    assert outcome is HealthWriteOutcome.NOT_PERSISTED
    assert store.last_write_failure_phase == "verification"
    assert store.last_write_failure_cause_types == ("DatabaseError",)
    assert "private verification cause" not in repr(
        store.last_write_failure_cause_types
    )


def test_half_open_probe_failure_is_reopened_as_one_durable_transaction(tmp_path: Path) -> None:
    """Replay never observes a committed half-open failure without its reopen."""
    database = tmp_path / "retrieval.sqlite3"
    with connect_cache_database(database) as connection:
        initialize_schema(connection)
    store = ProviderHealthStore(database, failure_threshold=1, cooldown_ms=100)
    fingerprint = "f" * 64
    first = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=1)
    store.record_failure(first, ProviderCallStatus.TIMEOUT, now_ms=2)
    probe = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=102)
    store.record_failure(probe, ProviderCallStatus.TIMEOUT, now_ms=103)

    assert store.acquire(fingerprint, ModelComponent.DENSE, now_ms=104).status is ProviderCallStatus.CIRCUIT_OPEN


def test_slow_health_recording_does_not_hold_the_admission_lock(
    monkeypatch, tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    """A later admission must not queue behind verification I/O under the lease lock."""
    import threading
    import time
    from codex_memory.retrieval import provider_health
    from codex_memory.retrieval.deadline import DeadlineBudget

    store = _store(tmp_path)
    fingerprint = "9" * 64
    first = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=1)
    original_verify = provider_health.verify_cache_database
    entered = threading.Event()

    def slow_verify(connection):  # type: ignore[no-untyped-def]
        entered.set()
        time.sleep(0.2)
        return original_verify(connection)

    monkeypatch.setattr(provider_health, "verify_cache_database", slow_verify)
    worker = threading.Thread(
        target=lambda: store.record_failure(first, ProviderCallStatus.INVALID_RESPONSE, now_ms=2),
    )
    worker.start()
    assert entered.wait(1)
    started = time.monotonic()
    result = store.acquire(
        fingerprint, ModelComponent.DENSE, now_ms=3,
        deadline=DeadlineBudget.start(30, now_ns=time.monotonic_ns), now_ns=time.monotonic_ns,
    )
    elapsed = time.monotonic() - started
    worker.join(1)

    assert elapsed < 0.1
    assert result.status is ProviderCallStatus.BUSY
