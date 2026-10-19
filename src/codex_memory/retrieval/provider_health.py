"""Short append-only circuit-breaker transactions for optional providers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from pathlib import Path
import sqlite3
import threading
from threading import RLock
import uuid
from weakref import WeakValueDictionary

from codex_memory.ledger_common import canonical_json

from .cache_database import (
    CacheCorruptError,
    SCHEMA_VERSION_V4,
    _V4_TABLE_COLUMNS,
    _normal_sql,
    _provider_health_projection_rows,
    _schema_sql_from_statements,
    _v4_schema_statements,
    append_provider_health_event,
    connect_cache_database,
    immediate_transaction,
    inspect_cache_schema_version,
    rebuild_provider_health_projections,
    verify_cache_database,
    verify_provider_attempt_history,
)
from .errors import RetrievalValidationError
from .deadline import DeadlineBudget
from .provider import ModelComponent, ProviderCallStatus, ProviderOperation


CIRCUIT_FAILURE_STATUSES = frozenset({
    ProviderCallStatus.UNAVAILABLE,
    ProviderCallStatus.BUSY,
    ProviderCallStatus.TIMEOUT,
    ProviderCallStatus.INVALID_RESPONSE,
})


class HealthWriteOutcome(StrEnum):
    """Whether a health completion was durably committed."""

    PERSISTED = "persisted"
    NOT_PERSISTED = "not_persisted"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class HealthDecision:
    provider_fingerprint: str
    component: ModelComponent | None
    allowed: bool
    status: ProviderCallStatus
    state: str
    lease_until_ms: int | None = None
    lease_id: str | None = None
    operation: ProviderOperation | None = None
    circuit_key: str | None = None
    history_epoch: int | None = None
    fencing_version: int | None = None


@dataclass(frozen=True, slots=True)
class _Lease:
    lease_id: str
    expires_ms: int
    completion: threading.Event | None = None
    recording: bool = False
    connection: sqlite3.Connection | None = None
    owner_thread_id: int | None = None


_LEASE_GUARD = threading.RLock()
_LEASE_KEY_GUARDS: WeakValueDictionary = WeakValueDictionary()
# Admission is provider-wide: a shared local service must never receive two
# in-process calls merely because they target different components.
_LEASES: dict[tuple[str, str], _Lease] = {}


def _lease_guard(key: tuple[str, str]):
    """Serialize one database/provider, never unrelated provider health I/O.

    The global guard only publishes the per-key lock. An active ``with`` keeps
    that lock alive; inactive keys disappear without a process-lifetime cache.
    All components of the same provider still use the same admission key.
    """
    with _LEASE_GUARD:
        guard = _LEASE_KEY_GUARDS.get(key)
        if guard is None:
            guard = RLock()
            _LEASE_KEY_GUARDS[key] = guard
        return guard


class _DeadlineExpired(RuntimeError):
    pass


class ProviderHealthStore:
    """A small durable circuit state; provider I/O is deliberately absent here."""

    def __init__(
        self,
        database_path: Path,
        *,
        failure_threshold: int = 2,
        cooldown_ms: int = 30_000,
        max_inflight: int = 1,
        queue_limit: int = 0,
        max_attempt_history: int = 10_000,
    ) -> None:
        if isinstance(failure_threshold, bool) or not isinstance(failure_threshold, int) or failure_threshold < 1:
            raise RetrievalValidationError("failure_threshold must be positive")
        if isinstance(cooldown_ms, bool) or not isinstance(cooldown_ms, int) or cooldown_ms < 1:
            raise RetrievalValidationError("cooldown_ms must be positive")
        if max_inflight != 1 or queue_limit != 0:
            raise RetrievalValidationError("provider health only supports one in-flight and zero queue")
        if (
            isinstance(max_attempt_history, bool)
            or not isinstance(max_attempt_history, int)
            or max_attempt_history < 1
        ):
            raise RetrievalValidationError("max_attempt_history must be positive")
        self.database_path = Path(database_path)
        self.failure_threshold = failure_threshold
        self.cooldown_ms = cooldown_ms
        self.max_attempt_history = max_attempt_history
        self._write_failure = threading.local()

    @property
    def last_write_failure_phase(self) -> str | None:
        return getattr(self._write_failure, "phase", None)

    @property
    def last_write_failure_cause_types(self) -> tuple[str, ...]:
        return getattr(self._write_failure, "cause_types", ())

    def _clear_write_failure(self) -> None:
        self._write_failure.phase = None
        self._write_failure.cause_types = ()

    def _set_write_failure(self, phase: str, exc: BaseException | None) -> None:
        causes: list[str] = []
        seen: set[int] = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            causes.append(type(exc).__name__)
            exc = exc.__cause__ or exc.__context__
        self._write_failure.phase = phase
        self._write_failure.cause_types = tuple(causes)

    def acquire(
        self,
        provider_fingerprint: str,
        component: ModelComponent | None,
        *,
        now_ms: int,
        operation: ProviderOperation | None = None,
        deadline: DeadlineBudget | None = None,
        now_ns=None,
        completion: threading.Event | None = None,
        retain_connection: bool = False,
    ) -> HealthDecision:
        self._validate(provider_fingerprint, component, now_ms)
        budget_clock = self._budget_clock(deadline, now_ns)
        try:
            with self._rebuild_guard(deadline, budget_clock):
                return self._acquire_under_rebuild_lock(
                    provider_fingerprint,
                    component,
                    now_ms=now_ms,
                    operation=operation,
                    deadline=deadline,
                    now_ns=budget_clock,
                    completion=completion,
                    retain_connection=retain_connection,
                )
        except (CacheCorruptError, OSError, sqlite3.Error, ValueError) as exc:
            self._set_write_failure("admission", exc)
            status = (
                ProviderCallStatus.DEADLINE_EXHAUSTED
                if self._expired(deadline, budget_clock)
                else ProviderCallStatus.UNAVAILABLE
            )
            state = (
                "deadline_exhausted"
                if status is ProviderCallStatus.DEADLINE_EXHAUSTED
                else "health_store_failure"
            )
            return HealthDecision(provider_fingerprint, component, False, status, state)

    def _acquire_under_rebuild_lock(self, provider_fingerprint: str, component: ModelComponent | None, *, now_ms: int, operation: ProviderOperation | None = None, deadline: DeadlineBudget | None = None, now_ns=None, completion: threading.Event | None = None, retain_connection: bool = False) -> HealthDecision:
        self._validate(provider_fingerprint, component, now_ms)
        operation = self._operation(component, operation)
        if not isinstance(retain_connection, bool):
            raise RetrievalValidationError("retain_connection must be boolean")
        now_ns = self._budget_clock(deadline, now_ns)
        if self._expired(deadline, now_ns):
            return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.DEADLINE_EXHAUSTED, "deadline_exhausted")
        key = self._lease_key(provider_fingerprint, component)
        with _lease_guard(key):
            existing = _LEASES.get(key)
            if existing is not None and existing.recording:
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.BUSY, "recording", existing.expires_ms)
            if (
                existing is not None
                and existing.connection is not None
                and existing.owner_thread_id != threading.get_ident()
            ):
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.BUSY, "busy", existing.expires_ms)
            if existing is not None and existing.completion is not None:
                if not existing.completion.is_set():
                    return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.BUSY, "busy", existing.expires_ms)
                self._close_connection(existing)
                del _LEASES[key]
                existing = None
            if existing is not None and now_ms < existing.expires_ms:
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.BUSY, "busy", existing.expires_ms)
            if existing is not None:
                self._close_connection(existing)
                del _LEASES[key]
            try:
                connection = connect_cache_database(self.database_path, busy_timeout_ms=0)
            except (OSError, sqlite3.Error, ValueError) as exc:
                self._set_write_failure("admission", exc)
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.UNAVAILABLE, "health_store_failure")
            try:
                if deadline is not None:
                    connection.set_progress_handler(
                        lambda: 1 if self._expired(deadline, now_ns) else 0,
                        100,
                    )
                with immediate_transaction(connection):
                    self._require_budget(deadline, now_ns)
                    self._verify_admission_database(connection)
                    self._require_budget(deadline, now_ns)
                    self._verify_attempt_history(connection)
                    rebuild_provider_health_projections(connection)
                    self._require_budget(deadline, now_ns)
                    self._expire_unknown_attempts(connection, provider_fingerprint, now_ms)
                    active = connection.execute(
                        """SELECT admission.expires_at
                           FROM retrieval_provider_attempt_admissions AS admission
                           LEFT JOIN retrieval_provider_attempt_results AS result
                             ON result.attempt_id = admission.attempt_id
                           WHERE admission.provider_fingerprint = ?
                             AND result.attempt_id IS NULL
                           ORDER BY admission.attempt_seq DESC LIMIT 1""",
                        (provider_fingerprint,),
                    ).fetchone()
                    if active is not None:
                        active_until = _to_ms(active["expires_at"])
                        return HealthDecision(
                            provider_fingerprint, component, False,
                            ProviderCallStatus.BUSY, "unresolved",
                            active_until,
                        )
                    if connection.execute(
                        "SELECT count(*) FROM retrieval_provider_attempt_admissions"
                    ).fetchone()[0] >= self.max_attempt_history:
                        return HealthDecision(
                            provider_fingerprint, component, False,
                            ProviderCallStatus.UNAVAILABLE, "history_capacity",
                        )
                    component_value = (
                        "capabilities" if component is None else component.value
                    )
                    row = connection.execute(
                        "SELECT * FROM retrieval_provider_health_current WHERE provider_fingerprint = ? AND component = ?",
                        (provider_fingerprint, component_value),
                    ).fetchone()
                    if component is None:
                        # Capabilities is a provider-level operation. It uses
                        # the same provider-wide admission lease and is
                        # denied while any persisted component circuit is
                        # open or half-open. v4 also records its own durable
                        # provider-wide circuit row under ``capabilities``.
                        rows = connection.execute(
                            "SELECT state, open_until, probe_lease_until FROM retrieval_provider_health_current WHERE provider_fingerprint = ?",
                            (provider_fingerprint,),
                        ).fetchall()
                        if any(
                            row["state"] == "open"
                            and row["open_until"] is not None
                            and now_ms < _to_ms(row["open_until"])
                            for row in rows
                        ):
                            return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.CIRCUIT_OPEN, "open")
                        if any(
                            row["state"] == "half_open"
                            and row["probe_lease_until"] is not None
                            and now_ms < _to_ms(row["probe_lease_until"])
                            for row in rows
                        ):
                            return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.BUSY, "half_open")
                    state = "closed" if row is None else row["state"]
                    open_until = None if row is None else _to_ms(row["open_until"])
                    lease_until = None if row is None else _to_ms(row["probe_lease_until"])
                    if state == "open" and open_until is not None and now_ms < open_until:
                        return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.CIRCUIT_OPEN, "open", open_until)
                    if state == "open":
                        self._event(connection, provider_fingerprint, component, "half_open", now_ms, deadline=deadline, now_ns=now_ns)
                        state = "half_open"
                    if state == "half_open":
                        if lease_until is not None and now_ms < lease_until:
                            return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.BUSY, "half_open", lease_until)
                        if lease_until is not None:
                            self._event(connection, provider_fingerprint, component, "probe_released", now_ms, deadline=deadline, now_ns=now_ns)
                        lease_until = now_ms + self.cooldown_ms
                        self._event(connection, provider_fingerprint, component, "probe_acquired", now_ms, lease_until=lease_until, deadline=deadline, now_ns=now_ns)
                    expiry = lease_until if lease_until is not None else now_ms + self.cooldown_ms
                    lease_id = uuid.uuid4().hex
                    circuit_key = hashlib.sha256(
                        f"provider:{provider_fingerprint}".encode("ascii")
                    ).hexdigest()
                    history_epoch = 1
                    fencing_version = int(connection.execute(
                        """SELECT COALESCE(MAX(fencing_version), 0) + 1
                           FROM retrieval_provider_attempt_admissions
                           WHERE circuit_key = ? AND history_epoch = ?""",
                        (circuit_key, history_epoch),
                    ).fetchone()[0])
                    admitted_at = _timestamp(now_ms)
                    expires_at = _timestamp(expiry)
                    payload_sha256 = hashlib.sha256(canonical_json({
                        "attempt_id": lease_id,
                        "capability": "capabilities" if component is None else component.value,
                        "circuit_key": circuit_key,
                        "expires_at": expires_at,
                        "fencing_version": fencing_version,
                        "history_epoch": history_epoch,
                        "operation": operation.value,
                        "provider_fingerprint": provider_fingerprint,
                        "admitted_at": admitted_at,
                    }).encode("utf-8")).hexdigest()
                    connection.execute(
                        """INSERT INTO retrieval_provider_attempt_admissions(
                               attempt_id, provider_fingerprint, capability, operation,
                               circuit_key, history_epoch, fencing_version, admitted_at,
                               expires_at, payload_sha256
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            lease_id, provider_fingerprint,
                            "capabilities" if component is None else component.value,
                            operation.value, circuit_key, history_epoch, fencing_version,
                            admitted_at, expires_at, payload_sha256,
                        ),
                    )
                    self._require_budget(deadline, now_ns)
                _LEASES[key] = _Lease(
                    lease_id,
                    expiry,
                    completion,
                    connection=connection if retain_connection else None,
                    owner_thread_id=threading.get_ident() if retain_connection else None,
                )
                return HealthDecision(
                    provider_fingerprint, component, True, ProviderCallStatus.OK,
                    state, expiry, lease_id, operation, circuit_key,
                    history_epoch, fencing_version,
                )
            except _DeadlineExpired as exc:
                self._set_write_failure("admission", exc)
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.DEADLINE_EXHAUSTED, "deadline_exhausted")
            except CacheCorruptError as exc:
                self._set_write_failure("admission", exc)
                if self._expired(deadline, now_ns):
                    return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.DEADLINE_EXHAUSTED, "deadline_exhausted")
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.UNAVAILABLE, "health_store_failure")
            except (OSError, sqlite3.Error, ValueError) as exc:
                self._set_write_failure("admission", exc)
                return HealthDecision(provider_fingerprint, component, False, ProviderCallStatus.UNAVAILABLE, "health_store_failure")
            finally:
                connection.set_progress_handler(None, 0)
                if not (
                    retain_connection
                    and key in _LEASES
                    and _LEASES[key].connection is connection
                ):
                    connection.close()

    def record_success(self, decision: HealthDecision, *, now_ms: int, deadline: DeadlineBudget | None = None, now_ns=None) -> HealthWriteOutcome:
        return self._record(decision, ProviderCallStatus.OK, now_ms, deadline=deadline, now_ns=now_ns)

    def record_failure(self, decision: HealthDecision, status: ProviderCallStatus, *, now_ms: int, completion: threading.Event | None = None, deadline: DeadlineBudget | None = None, now_ns=None) -> HealthWriteOutcome:
        if not isinstance(status, ProviderCallStatus):
            raise RetrievalValidationError("status must be a ProviderCallStatus")
        if status is ProviderCallStatus.OK:
            raise RetrievalValidationError("failure status cannot be ok")
        if status not in CIRCUIT_FAILURE_STATUSES:
            self.release(
                decision, completion=completion, deadline=deadline, now_ns=now_ns
            )
            return HealthWriteOutcome.NOT_APPLICABLE
        return self._record(decision, status, now_ms, completion=completion, deadline=deadline, now_ns=now_ns)

    def release(
        self,
        decision: HealthDecision,
        *,
        completion: threading.Event | None = None,
        deadline: DeadlineBudget | None = None,
        now_ns=None,
    ) -> None:
        """Release an admitted call without appending health-failure evidence."""
        if not isinstance(decision, HealthDecision) or not decision.allowed:
            raise RetrievalValidationError("decision must be an allowed health decision")
        outcome = self._record(
            decision,
            ProviderCallStatus.DEADLINE_EXHAUSTED,
            now_ms=__import__("time").time_ns() // 1_000_000,
            completion=completion,
            deadline=deadline,
            now_ns=now_ns,
            result_outcome="released",
        )
        if outcome is not HealthWriteOutcome.PERSISTED:
            raise RetrievalValidationError("provider health lease is stale")

    def _record(self, decision: HealthDecision, status: ProviderCallStatus, now_ms: int, *, completion: threading.Event | None = None, deadline: DeadlineBudget | None = None, now_ns=None, result_outcome: str | None = None) -> HealthWriteOutcome:
        try:
            with self._rebuild_guard(deadline, now_ns):
                return self._record_under_rebuild_lock(
                    decision,
                    status,
                    now_ms,
                    completion=completion,
                    deadline=deadline,
                    now_ns=now_ns,
                    result_outcome=result_outcome,
                )
        except (CacheCorruptError, OSError, sqlite3.Error, ValueError) as exc:
            self._set_write_failure("result", exc)
            key = self._lease_key(decision.provider_fingerprint, decision.component)
            with _lease_guard(key):
                current = _LEASES.get(key)
                if current is not None and current.lease_id == decision.lease_id:
                    self._close_connection(current)
                    del _LEASES[key]
            return HealthWriteOutcome.NOT_PERSISTED

    def _record_under_rebuild_lock(self, decision: HealthDecision, status: ProviderCallStatus, now_ms: int, *, completion: threading.Event | None = None, deadline: DeadlineBudget | None = None, now_ns=None, result_outcome: str | None = None) -> HealthWriteOutcome:
        if not isinstance(decision, HealthDecision) or not decision.allowed:
            raise RetrievalValidationError("decision must be an allowed health decision")
        self._validate(decision.provider_fingerprint, decision.component, now_ms)
        self._clear_write_failure()
        now_ns = self._budget_clock(deadline, now_ns)
        key = self._lease_key(decision.provider_fingerprint, decision.component)
        owns_local_lease = False
        connection = None
        with _lease_guard(key):
            lease = _LEASES.get(key)
            if lease is not None and decision.lease_id == lease.lease_id:
                self._require_connection_owner(lease)
                owns_local_lease = True
                connection = lease.connection
                _LEASES[key] = _Lease(
                    lease.lease_id, lease.expires_ms, lease.completion, True,
                    connection, lease.owner_thread_id,
                )
        # Health I/O deliberately occurs outside this provider's lease
        # guard.  A slow verifier can consume this call's own deadline, but it
        # must never make another provider admission wait on that Python lock.
        cleanup_failed = False
        try:
            phase = "verification"
            if connection is None:
                connection = connect_cache_database(
                    self.database_path, busy_timeout_ms=0
                )
            try:
                if deadline is not None:
                    connection.set_progress_handler(
                        lambda: 1 if self._expired(deadline, now_ns) else 0,
                        100,
                    )
                with immediate_transaction(connection):
                    self._require_budget(deadline, now_ns)
                    if result_outcome == "released":
                        self._verify_admission_database(connection)
                    elif deadline is not None:
                        self._bounded_health_verification(deadline, now_ns)
                    else:
                        verify_cache_database(connection)
                    self._require_budget(deadline, now_ns)
                    self._verify_attempt_history(connection)
                    admission = connection.execute(
                        """SELECT * FROM retrieval_provider_attempt_admissions
                           WHERE attempt_id = ?""",
                        (decision.lease_id,),
                    ).fetchone()
                    latest_fence = connection.execute(
                        """SELECT MAX(fencing_version)
                           FROM retrieval_provider_attempt_admissions
                           WHERE circuit_key = ? AND history_epoch = ?""",
                        (decision.circuit_key, decision.history_epoch),
                    ).fetchone()[0]
                    already_settled = connection.execute(
                        "SELECT 1 FROM retrieval_provider_attempt_results WHERE attempt_id = ?",
                        (decision.lease_id,),
                    ).fetchone()
                    if (
                        admission is None
                        or already_settled is not None
                        or latest_fence != decision.fencing_version
                    ):
                        self._set_write_failure("result_fencing", None)
                        return HealthWriteOutcome.NOT_PERSISTED
                    phase = "result"
                    outcome = result_outcome or (
                        "success" if status is ProviderCallStatus.OK else "failure"
                    )
                    self._append_attempt_result(
                        connection, decision, outcome,
                        None if outcome in {"released", "expired_unknown"} else status.value,
                        now_ms,
                    )
                    if outcome == "released":
                        phase = "commit"
                        self._require_budget(deadline, now_ns)
                        return HealthWriteOutcome.PERSISTED
                    if status is ProviderCallStatus.OK:
                        phase = "record_success"
                        self._event(connection, decision.provider_fingerprint, decision.component, "success", now_ms, deadline=deadline, now_ns=now_ns)
                    else:
                        phase = "record_failure"
                        row = connection.execute(
                            "SELECT consecutive_failures FROM retrieval_provider_health_current WHERE provider_fingerprint = ? AND component = ?",
                            (
                                decision.provider_fingerprint,
                                "capabilities" if decision.component is None else decision.component.value,
                            ),
                        ).fetchone()
                        failures = 0 if row is None else int(row["consecutive_failures"])
                        self._event(
                            connection, decision.provider_fingerprint, decision.component,
                            "failure", now_ms, failure_kind=status.value,
                            deadline=deadline, now_ns=now_ns, rebuild=False,
                        )
                        if decision.state == "half_open" or failures + 1 >= self.failure_threshold:
                            self._event(
                                connection, decision.provider_fingerprint, decision.component,
                                "open", now_ms, open_until=now_ms + self.cooldown_ms,
                                deadline=deadline, now_ns=now_ns, rebuild=False,
                            )
                        self._require_budget(deadline, now_ns)
                        rebuild_provider_health_projections(connection)
                    phase = "commit"
                    self._require_budget(deadline, now_ns)
            except _DeadlineExpired as exc:
                self._set_write_failure(phase, exc)
                return HealthWriteOutcome.NOT_PERSISTED
            except (CacheCorruptError, OSError, sqlite3.Error, ValueError) as exc:
                self._set_write_failure(phase, exc)
                return HealthWriteOutcome.NOT_PERSISTED
            finally:
                try:
                    connection.set_progress_handler(None, 0)
                    connection.close()
                except (OSError, sqlite3.Error, ValueError) as exc:
                    self._set_write_failure("cleanup", exc)
                    cleanup_failed = True
            return (
                HealthWriteOutcome.NOT_PERSISTED
                if cleanup_failed
                else HealthWriteOutcome.PERSISTED
            )
        finally:
            with _lease_guard(key):
                current = _LEASES.get(key)
                if owns_local_lease and current is not None and current.lease_id == decision.lease_id:
                    if completion is not None and not completion.is_set():
                        _LEASES[key] = _Lease(current.lease_id, current.expires_ms, completion)
                    else:
                        del _LEASES[key]

    @staticmethod
    def _close_connection(lease: _Lease) -> None:
        if lease.connection is not None:
            lease.connection.close()

    @staticmethod
    def _require_connection_owner(lease: _Lease) -> None:
        if (
            lease.connection is not None
            and lease.owner_thread_id != threading.get_ident()
        ):
            raise RetrievalValidationError("provider health connection is owned by another thread")

    def _rebuild_guard(self, deadline: DeadlineBudget | None, now_ns):
        temp_dir = self.database_path.parent / "temp"
        if not temp_dir.is_dir():
            return nullcontext()
        from .service import process_rebuild_lock

        return process_rebuild_lock(
            temp_dir, deadline=deadline,
            now_ns=self._budget_clock(deadline, now_ns),
        )

    def _bounded_health_verification(self, deadline: DeadlineBudget | None, now_ns) -> None:
        if deadline is None:
            raise RetrievalValidationError("retained health verification requires a deadline")
        remaining_ms = deadline.remaining_ms(now_ns=now_ns)
        if remaining_ms == 0:
            raise _DeadlineExpired
        completed = threading.Event()
        errors: list[BaseException] = []

        def verify() -> None:
            try:
                verification = connect_cache_database(
                    self.database_path, readonly=True, busy_timeout_ms=0
                )
                try:
                    verification.set_progress_handler(
                        lambda: 1 if self._expired(deadline, now_ns) else 0,
                        100,
                    )
                    self._verify_health_database(verification)
                finally:
                    try:
                        verification.set_progress_handler(None, 0)
                    finally:
                        verification.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(
            target=verify,
            name="codex-memory-health-verify",
            daemon=True,
        )
        worker.start()
        # Thread construction/start consumes the same absolute budget. Never
        # wait with the allowance sampled before starting the verifier.
        remaining_ms = deadline.remaining_ms(now_ns=now_ns)
        if remaining_ms == 0:
            raise _DeadlineExpired
        if not completed.wait(remaining_ms / 1000):
            raise _DeadlineExpired
        if errors:
            raise errors[0]
        self._require_budget(deadline, now_ns)

    @staticmethod
    def _verify_health_database(connection: sqlite3.Connection) -> None:
        ProviderHealthStore._verify_provider_control_database(connection)

    @staticmethod
    def _verify_admission_database(connection: sqlite3.Connection) -> None:
        ProviderHealthStore._verify_provider_control_database(connection)

    @staticmethod
    def _verify_provider_control_database(connection: sqlite3.Connection) -> None:
        if inspect_cache_schema_version(connection) != SCHEMA_VERSION_V4:
            raise CacheCorruptError("provider health schema is unavailable")
        health_tables = {
            "retrieval_provider_health_events",
            "retrieval_provider_health_current",
            "retrieval_provider_attempt_admissions",
            "retrieval_provider_attempt_results",
        }
        expected_schema = {
            key: value
            for key, value in _schema_sql_from_statements(
                _v4_schema_statements(False)
            ).items()
            if key[1].startswith("retrieval_provider_health_")
            or key[1].startswith("retrieval_provider_attempt_")
        }
        actual_schema = {
            (row["type"], row["name"]): _normal_sql(row["sql"])
            for row in connection.execute(
                """SELECT type, name, tbl_name, sql FROM sqlite_master
                   WHERE sql IS NOT NULL AND tbl_name IN (?, ?, ?, ?)""",
                tuple(sorted(health_tables)),
            )
        }
        if actual_schema != expected_schema:
            raise CacheCorruptError("provider health schema definition is invalid")
        for table in sorted(health_tables):
            if tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            ) != _V4_TABLE_COLUMNS[table]:
                raise CacheCorruptError("provider health schema columns are invalid")
        if list(
            connection.execute(
                "PRAGMA foreign_key_check(retrieval_provider_health_current)"
            )
        ):
            raise CacheCorruptError("provider health foreign key check failed")
        expected = _provider_health_projection_rows(connection)
        actual = {
            (row["provider_fingerprint"], row["component"]): tuple(row)[2:]
            for row in connection.execute(
                """SELECT * FROM retrieval_provider_health_current
                   ORDER BY provider_fingerprint, component"""
            )
        }
        if actual != expected:
            raise CacheCorruptError("provider health projection does not replay")

    def _event(
        self,
        connection,
        provider_fingerprint: str,
        component: ModelComponent | None,
        action: str,
        now_ms: int,
        *,
        failure_kind: str | None = None,
        open_until: int | None = None,
        lease_until: int | None = None,
        deadline: DeadlineBudget | None = None,
        now_ns=None,
        rebuild: bool = True,
    ) -> None:
        self._require_budget(deadline, now_ns)
        ordinal = connection.execute("SELECT COUNT(*) FROM retrieval_provider_health_events").fetchone()[0]
        payload = hashlib.sha256(canonical_json({"action": action, "now_ms": now_ms, "ordinal": ordinal + 1}).encode("utf-8")).hexdigest()
        append_provider_health_event(
            connection,
            provider_fingerprint=provider_fingerprint,
            component="capabilities" if component is None else component.value,
            action=action,
            failure_kind=failure_kind,
            open_until=None if open_until is None else _timestamp(open_until),
            lease_until=None if lease_until is None else _timestamp(lease_until),
            payload_sha256=payload,
            created_at=_timestamp(now_ms),
        )
        self._require_budget(deadline, now_ns)
        if rebuild:
            rebuild_provider_health_projections(connection)

    @staticmethod
    def _operation(
        component: ModelComponent | None, operation: ProviderOperation | None
    ) -> ProviderOperation:
        if operation is not None:
            if not isinstance(operation, ProviderOperation):
                raise RetrievalValidationError("operation must be ProviderOperation")
            if (component is None) != (operation is ProviderOperation.CAPABILITIES):
                raise RetrievalValidationError("operation does not match component")
            return operation
        if component is None:
            return ProviderOperation.CAPABILITIES
        return {
            ModelComponent.DENSE: ProviderOperation.DENSE_QUERY,
            ModelComponent.SPARSE: ProviderOperation.SPARSE_QUERY,
            ModelComponent.RERANKER: ProviderOperation.RERANK,
        }[component]

    @staticmethod
    def _verify_attempt_history(connection: sqlite3.Connection) -> None:
        verify_provider_attempt_history(connection)

    def _expire_unknown_attempts(
        self, connection: sqlite3.Connection, provider_fingerprint: str, now_ms: int
    ) -> None:
        rows = connection.execute(
            """SELECT admission.*
               FROM retrieval_provider_attempt_admissions AS admission
               LEFT JOIN retrieval_provider_attempt_results AS result
                 ON result.attempt_id = admission.attempt_id
               WHERE admission.provider_fingerprint = ?
                 AND result.attempt_id IS NULL
                 AND admission.expires_at <= ?
               ORDER BY admission.attempt_seq""",
            (provider_fingerprint, _timestamp(now_ms)),
        ).fetchall()
        for row in rows:
            decision = HealthDecision(
                row["provider_fingerprint"],
                None if row["capability"] == "capabilities" else ModelComponent(row["capability"]),
                True, ProviderCallStatus.OK, "expired", _to_ms(row["expires_at"]),
                row["attempt_id"], ProviderOperation(row["operation"]), row["circuit_key"],
                row["history_epoch"], row["fencing_version"],
            )
            self._append_attempt_result(
                connection, decision, "expired_unknown", None, now_ms
            )

    @staticmethod
    def _append_attempt_result(
        connection: sqlite3.Connection,
        decision: HealthDecision,
        outcome: str,
        provider_status: str | None,
        now_ms: int,
    ) -> None:
        recorded_at = _timestamp(now_ms)
        preimage = {
            "attempt_id": decision.lease_id,
            "outcome": outcome,
            "provider_status": provider_status,
            "recorded_at": recorded_at,
        }
        digest = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
        connection.execute(
            """INSERT INTO retrieval_provider_attempt_results(
                   result_id, attempt_id, outcome, provider_status, recorded_at,
                   payload_sha256
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                f"attempt_result_{digest[:32]}", decision.lease_id, outcome,
                provider_status, recorded_at, digest,
            ),
        )

    @staticmethod
    def _budget_clock(deadline: DeadlineBudget | None, now_ns):
        if deadline is not None and not isinstance(deadline, DeadlineBudget):
            raise RetrievalValidationError("deadline must be DeadlineBudget or null")
        if now_ns is None:
            import time
            now_ns = time.monotonic_ns
        if not callable(now_ns):
            raise RetrievalValidationError("now_ns must be callable")
        return now_ns

    @staticmethod
    def _expired(deadline: DeadlineBudget | None, now_ns) -> bool:
        return deadline is not None and deadline.remaining_ms(now_ns=now_ns) == 0

    def _require_budget(self, deadline: DeadlineBudget | None, now_ns) -> None:
        if self._expired(deadline, now_ns):
            raise _DeadlineExpired

    @staticmethod
    def _validate(provider_fingerprint: str, component: ModelComponent | None, now_ms: int) -> None:
        if not isinstance(provider_fingerprint, str) or len(provider_fingerprint) != 64 or any(char not in "0123456789abcdef" for char in provider_fingerprint):
            raise RetrievalValidationError("provider_fingerprint must be sha256")
        if component is not None and not isinstance(component, ModelComponent):
            raise RetrievalValidationError("component must be ModelComponent or null")
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise RetrievalValidationError("now_ms must be non-negative")

    def _lease_key(self, provider_fingerprint: str, component: ModelComponent | None) -> tuple[str, str]:
        del component
        return (str(self.database_path.resolve()).casefold(), provider_fingerprint)


def _timestamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _to_ms(timestamp: str | None) -> int | None:
    if timestamp is None:
        return None
    return int(datetime.fromisoformat(timestamp[:-1] + "+00:00").timestamp() * 1000)
