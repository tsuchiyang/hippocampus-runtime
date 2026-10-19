from __future__ import annotations

import json
from dataclasses import replace
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from codex_memory.retrieval.cache_database import (
    connect_cache_database,
    initialize_schema,
    initialize_schema_v3,
)
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.provider import (
    CapabilitiesResponse,
    ModelComponent,
    ProviderCallStatus,
    ProviderInput,
    ProviderOperation,
)
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider_health import HealthWriteOutcome, ProviderHealthStore
from retrieval_provider_fakes import FakeProvider


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "retrieval.sqlite3"
    connection = connect_cache_database(database)
    try:
        initialize_schema(connection)
    finally:
        connection.close()
    return database


def _attempt_database(database: Path) -> Path:
    return database.with_name(database.name + ".provider-attempts.sqlite3")


def _fingerprint(provider: FakeProvider) -> str:
    capabilities = provider.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    return capabilities.fingerprint


def test_process_exit_after_admission_is_durably_unresolved_and_denies_next_provider(
    tmp_path: Path,
) -> None:
    """Deleting process-local lease state must not reopen provider capacity."""
    database = _database(tmp_path)
    first = r'''
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path.cwd() / "tests"))
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.provider import ProviderInput
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider_health import ProviderHealthStore
from retrieval_provider_fakes import FakeProvider
class ExitProvider(FakeProvider):
    def embed_query(self, item, *, deadline_ms):
        os._exit(0)
provider = ExitProvider()
ProviderCallController(
    provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
    ProviderHealthStore(Path(sys.argv[1]), cooldown_ms=10_000), "f" * 64,
    now_ns=time.monotonic_ns, wall_clock_ms=lambda: 1_000,
).dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)
'''
    exited = subprocess.run(
        [sys.executable, "-c", first, str(database)],
        cwd=Path.cwd(), capture_output=True, text=True, timeout=10, check=False,
    )
    assert exited.returncode == 0, exited.stderr

    second = r'''
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path.cwd() / "tests"))
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.provider import ProviderInput, ProviderOperation
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider_health import ProviderHealthStore
from retrieval_provider_fakes import FakeProvider
provider = FakeProvider()
response = ProviderCallController(
    provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
    ProviderHealthStore(Path(sys.argv[1]), cooldown_ms=10_000), "f" * 64,
    now_ns=time.monotonic_ns, wall_clock_ms=lambda: 1_001,
).dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)
print(json.dumps({"status": response.call.status.value,
                  "calls": provider.call_counts[ProviderOperation.DENSE_QUERY]}))
'''
    denied = subprocess.run(
        [sys.executable, "-c", second, str(database)],
        cwd=Path.cwd(), capture_output=True, text=True, timeout=10, check=False,
    )
    assert denied.returncode == 0, denied.stderr
    assert json.loads(denied.stdout) == {"status": "busy", "calls": 0}
    with connect_cache_database(database, readonly=True) as connection:
        unresolved = connection.execute(
            """SELECT count(*) FROM retrieval_provider_attempt_admissions AS admission
               LEFT JOIN retrieval_provider_attempt_results AS result
                 ON result.attempt_id = admission.attempt_id
               WHERE result.attempt_id IS NULL"""
        ).fetchone()[0]
    assert unresolved == 1


def test_successful_provider_result_is_discarded_when_result_commit_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A provider success is not semantic evidence until phase B commits."""
    from codex_memory.retrieval import provider_health

    database = _database(tmp_path)
    provider = FakeProvider()
    original_connect = provider_health.connect_cache_database

    class FailResultCommit:
        def __init__(self, connection, *, fail):  # type: ignore[no-untyped-def]
            self.connection = connection
            self.fail = fail

        def __getattr__(self, name):  # type: ignore[no-untyped-def]
            return getattr(self.connection, name)

        def commit(self) -> None:
            if self.fail:
                raise sqlite3.OperationalError("injected result commit failure")
            self.connection.commit()

    writes = 0
    def faulty_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal writes
        connection = original_connect(*args, **kwargs)
        if kwargs.get("readonly"):
            return connection
        writes += 1
        return FailResultCommit(connection, fail=writes == 2)

    monkeypatch.setattr(provider_health, "connect_cache_database", faulty_connect)
    controller = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        ProviderHealthStore(database, cooldown_ms=10_000), _fingerprint(provider),
        now_ns=time.monotonic_ns, wall_clock_ms=lambda: 2_000,
    )
    response = controller.dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)

    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 1
    assert response.call.status is ProviderCallStatus.UNAVAILABLE
    assert response.items == ()
    assert controller.last_health_outcome is HealthWriteOutcome.NOT_PERSISTED
    assert controller.health_diagnostics[-1].phase == "result"


def test_expired_unknown_attempt_is_fenced_from_late_result(tmp_path: Path) -> None:
    """An expired old owner cannot append a timeout against a newer fence."""
    database = _database(tmp_path)
    store = ProviderHealthStore(database, failure_threshold=2, cooldown_ms=100)
    fingerprint = "a" * 64
    old = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=1)
    replacement = store.acquire(fingerprint, ModelComponent.DENSE, now_ms=102)

    assert replacement.allowed
    assert store.record_failure(old, ProviderCallStatus.TIMEOUT, now_ms=103) is HealthWriteOutcome.NOT_PERSISTED
    assert store.record_failure(replacement, ProviderCallStatus.TIMEOUT, now_ms=104) is HealthWriteOutcome.PERSISTED
    with connect_cache_database(database, readonly=True) as connection:
        outcomes = [tuple(row) for row in connection.execute(
            "SELECT outcome, provider_status FROM retrieval_provider_attempt_results ORDER BY result_seq"
        )]
        failures = connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events WHERE action = 'failure'"
        ).fetchone()[0]
    assert outcomes == [("expired_unknown", None), ("failure", "timeout")]
    assert failures == 1


def test_capabilities_is_admitted_and_settled_like_every_other_operation(tmp_path: Path) -> None:
    database = _database(tmp_path)
    provider = FakeProvider()
    controller = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        ProviderHealthStore(database), _fingerprint(provider), now_ns=time.monotonic_ns,
    )

    response = controller.capabilities()

    assert response.call.status is ProviderCallStatus.OK
    assert controller.last_health_outcome is HealthWriteOutcome.PERSISTED
    with connect_cache_database(database, readonly=True) as connection:
        row = connection.execute(
            """SELECT admission.capability, admission.operation, result.outcome
               FROM retrieval_provider_attempt_admissions AS admission
               JOIN retrieval_provider_attempt_results AS result USING (attempt_id)"""
        ).fetchone()
    assert tuple(row) == ("capabilities", ProviderOperation.CAPABILITIES.value, "success")


def test_twelve_short_lived_processes_stop_after_two_confirmed_timeouts(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    script = r'''
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path.cwd() / "tests"))
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.provider import ModelComponent, ProviderCallStatus, ProviderInput, ProviderOperation
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider_health import ProviderHealthStore
from retrieval_provider_fakes import FakeProvider
provider = FakeProvider({ModelComponent.DENSE: ProviderCallStatus.TIMEOUT})
response = ProviderCallController(
    provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
    ProviderHealthStore(Path(sys.argv[1]), failure_threshold=2, cooldown_ms=100_000),
    "f" * 64, now_ns=time.monotonic_ns, wall_clock_ms=lambda: int(sys.argv[2]),
).dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)
print(json.dumps({"status": response.call.status.value,
                  "calls": provider.call_counts[ProviderOperation.DENSE_QUERY]}))
'''
    outcomes = []
    for ordinal in range(12):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(database), str(1_000 + ordinal)],
            cwd=Path.cwd(), capture_output=True, text=True, timeout=10, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outcomes.append(json.loads(completed.stdout))

    assert outcomes[:2] == [
        {"status": "timeout", "calls": 1},
        {"status": "timeout", "calls": 1},
    ]
    assert outcomes[2:] == [{"status": "circuit_open", "calls": 0}] * 10


def test_attempt_journal_is_in_the_single_retrieval_database(tmp_path: Path) -> None:
    database = _database(tmp_path)
    provider = FakeProvider()
    response = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        ProviderHealthStore(database), _fingerprint(provider), now_ns=time.monotonic_ns,
    ).capabilities()

    assert response.call.status is ProviderCallStatus.OK
    with connect_cache_database(database, readonly=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "retrieval_provider_attempt_admissions" in tables
    assert "retrieval_provider_attempt_results" in tables
    assert not _attempt_database(database).exists()


def test_v3_without_controlled_upgrade_denies_provider_and_creates_nothing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "retrieval.sqlite3"
    with connect_cache_database(database) as connection:
        initialize_schema_v3(connection)
    provider = FakeProvider()
    response = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        ProviderHealthStore(database), _fingerprint(provider), now_ns=time.monotonic_ns,
    ).dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)

    assert response.call.status is ProviderCallStatus.UNAVAILABLE
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    assert not _attempt_database(database).exists()


def test_provider_exception_keeps_provider_phase_and_cause_chain(tmp_path: Path) -> None:
    class RaisingProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            try:
                raise ValueError("private inner")
            except ValueError as exc:
                raise RuntimeError("private outer") from exc

    database = _database(tmp_path)
    provider = RaisingProvider()
    controller = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        ProviderHealthStore(database), _fingerprint(provider), now_ns=time.monotonic_ns,
    )
    response = controller.dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)

    assert response.call.status is ProviderCallStatus.UNAVAILABLE
    diagnostic = controller.health_diagnostics[-1]
    assert diagnostic.phase == "provider"
    assert diagnostic.cause_types == ("RuntimeError", "ValueError")
    assert "private" not in repr(diagnostic)


def test_capabilities_timeouts_open_provider_wide_circuit(tmp_path: Path) -> None:
    class TimeoutCapabilities(FakeProvider):
        def capabilities(self, *, deadline_ms):  # type: ignore[no-untyped-def]
            response = super().capabilities(deadline_ms=deadline_ms)
            return CapabilitiesResponse(
                replace(
                    response.call,
                    status=ProviderCallStatus.TIMEOUT,
                    reason=ProviderCallStatus.TIMEOUT.value,
                ),
                None,
            )

    database = _database(tmp_path)
    provider = TimeoutCapabilities()
    fingerprint = _fingerprint(FakeProvider())
    statuses = []
    for now_ms in (1_000, 1_001, 1_002):
        statuses.append(
            ProviderCallController(
                provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
                ProviderHealthStore(database, failure_threshold=2, cooldown_ms=100_000),
                fingerprint, now_ns=time.monotonic_ns, wall_clock_ms=lambda value=now_ms: value,
            ).capabilities().call.status
        )

    assert statuses == [
        ProviderCallStatus.TIMEOUT,
        ProviderCallStatus.TIMEOUT,
        ProviderCallStatus.CIRCUIT_OPEN,
    ]
    assert provider.call_counts[ProviderOperation.CAPABILITIES] == 2


def test_attempt_history_capacity_denies_instead_of_growing_without_bound(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    provider = FakeProvider()
    store = ProviderHealthStore(database, max_attempt_history=1)
    first = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        store, _fingerprint(provider), now_ns=time.monotonic_ns,
    ).dense_query(ProviderInput.from_text("q1", 0, "q1"), dimension=4)
    second = ProviderCallController(
        provider, DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        store, _fingerprint(provider), now_ns=time.monotonic_ns,
    ).dense_query(ProviderInput.from_text("q2", 0, "q2"), dimension=4)

    assert first.call.status is ProviderCallStatus.OK
    assert second.call.status is ProviderCallStatus.UNAVAILABLE
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 1


def test_history_capacity_expires_unknown_before_denying_new_admission(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    provider = FakeProvider()
    fingerprint = _fingerprint(provider)
    store = ProviderHealthStore(
        database, cooldown_ms=100, max_attempt_history=1
    )
    unresolved = store.acquire(
        fingerprint, ModelComponent.DENSE, now_ms=1
    )
    assert unresolved.allowed

    response = ProviderCallController(
        provider,
        DeadlineBudget.start(500, now_ns=time.monotonic_ns),
        store,
        fingerprint,
        now_ns=time.monotonic_ns,
        wall_clock_ms=lambda: 102,
    ).dense_query(ProviderInput.from_text("q", 0, "q"), dimension=4)

    assert response.call.status is ProviderCallStatus.UNAVAILABLE
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 0
    connection = connect_cache_database(database, readonly=True)
    try:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT outcome, provider_status "
                "FROM retrieval_provider_attempt_results"
            )
        ] == [("expired_unknown", None)]
    finally:
        connection.close()


def test_nonfault_release_does_not_wait_past_hook_budget_for_rebuild_lock(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.service import process_rebuild_lock

    database = _database(tmp_path)
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    provider_entered = threading.Event()
    release_provider = threading.Event()
    finished = threading.Event()
    responses = []
    elapsed = []

    class BlockingUnsupportedProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            provider_entered.set()
            assert release_provider.wait(2)
            return super().embed_query(item, deadline_ms=deadline_ms)

    provider = BlockingUnsupportedProvider(
        {ModelComponent.DENSE: ProviderCallStatus.UNSUPPORTED}
    )
    fingerprint = _fingerprint(provider)

    def invoke() -> None:
        started = time.monotonic()
        try:
            controller = ProviderCallController(
                provider,
                DeadlineBudget.start(120, now_ns=lambda: 0),
                ProviderHealthStore(database),
                fingerprint,
                now_ns=lambda: 0,
            )
            responses.append(
                controller.dense_query(
                    ProviderInput.from_text("q", 0, "q"), dimension=4
                )
            )
            responses.append(controller)
        finally:
            elapsed.append(time.monotonic() - started)
            finished.set()

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    assert provider_entered.wait(1)
    with process_rebuild_lock(temp_dir):
        release_provider.set()
        finished_within_budget = finished.wait(0.2)
    worker.join(2)

    assert finished_within_budget
    assert not worker.is_alive()
    assert elapsed[0] < 0.2
    assert responses[0].call.status is ProviderCallStatus.UNSUPPORTED
    assert responses[1].health_diagnostics[-1].phase in {
        "release_not_applicable", "result"
    }


def test_result_commit_survives_rebuild_snapshot_replace_race(tmp_path: Path) -> None:
    """Phase B must target the installed DB when rebuild snapshots after Phase A."""
    from codex_memory.retrieval.service import process_rebuild_lock

    database = _database(tmp_path)
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    provider_entered = threading.Event()
    release_provider = threading.Event()
    controller_finished = threading.Event()

    class BlockingProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            provider_entered.set()
            assert release_provider.wait(5)
            return super().embed_query(item, deadline_ms=deadline_ms)

    provider = BlockingProvider()
    responses = []

    def call_provider() -> None:
        try:
            responses.append(
                ProviderCallController(
                    provider,
                    DeadlineBudget.start(3_000, now_ns=time.monotonic_ns),
                    ProviderHealthStore(database),
                    _fingerprint(provider),
                    now_ns=time.monotonic_ns,
                ).dense_query(
                    ProviderInput.from_text("q", 0, "q"), dimension=4
                )
            )
        finally:
            controller_finished.set()

    worker = threading.Thread(target=call_provider, daemon=True)
    worker.start()
    assert provider_entered.wait(2)

    replacement = tmp_path / "replacement.sqlite3"
    with process_rebuild_lock(temp_dir):
        shutil.copy2(database, replacement)
        release_provider.set()
        # Without shared rebuild fencing Phase B commits to the old file here,
        # after which replacement deterministically discards that result.
        controller_finished.wait(1)
        os.replace(replacement, database)

    worker.join(3)
    assert not worker.is_alive()
    assert responses[0].call.status is ProviderCallStatus.OK
    with connect_cache_database(database, readonly=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_attempt_results"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT count(*) FROM retrieval_provider_health_events "
            "WHERE action = 'success'"
        ).fetchone()[0] == 1
