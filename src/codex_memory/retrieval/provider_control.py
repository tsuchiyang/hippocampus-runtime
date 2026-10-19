"""One bounded admission and validation boundary for provider calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import threading
import time
from typing import Any

from .deadline import AbsoluteDeadlineMilliseconds, DeadlineBudget
from .errors import RetrievalValidationError
from .provider import (
    CapabilitiesResponse,
    DenseResponse,
    ModelComponent,
    ProviderCall,
    ProviderCallStatus,
    ProviderInput,
    ProviderOperation,
    RerankResponse,
    RetrievalModelProvider,
    SparseResponse,
    checked_dense_response,
    checked_rerank_response,
    checked_sparse_response,
)
from .provider_health import (
    CIRCUIT_FAILURE_STATUSES,
    HealthWriteOutcome,
    ProviderHealthStore,
)


_TIMEOUT_HEALTH_RESERVE_MS = 50
_RETURN_RESERVE_MS = 10


class _ProviderStartCancelled(Exception):
    """Internal signal that the parent won the final provider-call gate."""


def _is_exact_deadline_exhausted_response(
    response: object,
    response_type: type[Any],
    operation: ProviderOperation,
    component: ModelComponent | None,
) -> bool:
    """Recognize only a fully formed empty response for this exact call."""
    if type(response) is not response_type:
        return False
    try:
        call = response.call
        if (
            type(call) is not ProviderCall
            or ProviderCall.from_dict(call.to_dict()) != call
            or call.operation is not operation
            or call.component is not component
            or call.status is not ProviderCallStatus.DEADLINE_EXHAUSTED
        ):
            return False
        if operation is ProviderOperation.CAPABILITIES:
            return response.capabilities is None
        return type(response.items) is tuple and response.items == ()
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class _HealthDiagnostic:
    """Internal-only health failure evidence with no exception text."""

    phase: str
    code: str
    cause_types: tuple[str, ...]


class ProviderCallController:
    """Apply one absolute budget, worker boundary, health lease, and validator."""

    def __init__(
        self,
        provider: RetrievalModelProvider,
        deadline: DeadlineBudget,
        health_store: ProviderHealthStore | None,
        expected_provider_fingerprint: str | None,
        *,
        now_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        expected_dense_fingerprint: str | None = None,
        expected_sparse_fingerprint: str | None = None,
        expected_reranker_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(deadline, DeadlineBudget):
            raise RetrievalValidationError("deadline must be DeadlineBudget")
        if health_store is not None and not isinstance(health_store, ProviderHealthStore):
            raise RetrievalValidationError("health_store must be ProviderHealthStore or null")
        for name, value in (
            ("expected_provider_fingerprint", expected_provider_fingerprint),
            ("expected_dense_fingerprint", expected_dense_fingerprint),
            ("expected_sparse_fingerprint", expected_sparse_fingerprint),
            ("expected_reranker_fingerprint", expected_reranker_fingerprint),
        ):
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise RetrievalValidationError(f"{name} must be sha256 or null")
        if not callable(now_ns):
            raise RetrievalValidationError("now_ns must be callable")
        if not callable(wall_clock_ms):
            raise RetrievalValidationError("wall_clock_ms must be callable")
        self.provider = provider
        self.deadline = deadline
        self.health_store = health_store
        self.expected_provider_fingerprint = expected_provider_fingerprint
        self.expected_model_fingerprints = {
            ModelComponent.DENSE: expected_dense_fingerprint,
            ModelComponent.SPARSE: expected_sparse_fingerprint,
            ModelComponent.RERANKER: expected_reranker_fingerprint,
        }
        self.now_ns = now_ns
        self.wall_clock_ms = wall_clock_ms
        self.last_completion: threading.Event | None = None
        self.last_health_outcome = HealthWriteOutcome.NOT_APPLICABLE
        self.health_available = True
        self.health_diagnostics: list[_HealthDiagnostic] = []

    def capabilities(self) -> CapabilitiesResponse:
        return self._invoke(
            ProviderOperation.CAPABILITIES,
            None,
            CapabilitiesResponse,
            lambda remaining_ms: self.provider.capabilities(
                deadline_ms=remaining_ms()
            ),
            lambda response, _provider_deadline: self._validate_capabilities(response),
        )

    def _validate_capabilities(self, response: CapabilitiesResponse) -> CapabilitiesResponse:
        if (
            response.call.status is ProviderCallStatus.OK
            and response.capabilities is not None
            and self.expected_provider_fingerprint is not None
            and (
                response.capabilities.fingerprint != self.expected_provider_fingerprint
                or response.call.provider_fingerprint != self.expected_provider_fingerprint
            )
        ):
            return CapabilitiesResponse(
                replace(response.call, status=ProviderCallStatus.FINGERPRINT_MISMATCH, reason=ProviderCallStatus.FINGERPRINT_MISMATCH.value),
                None,
            )
        return response

    def dense_documents(
        self, items: Sequence[ProviderInput], *, dimension: int
    ) -> DenseResponse:
        requested = tuple(items)
        return self._invoke(
            ProviderOperation.DENSE_DOCUMENTS,
            ModelComponent.DENSE,
            DenseResponse,
            lambda remaining_ms: self.provider.embed_documents(
                requested, deadline_ms=remaining_ms()
            ),
            lambda response, provider_deadline: checked_dense_response(
                response,
                requested,
                dimension=dimension,
                deadline=provider_deadline,
                now_ns=self.now_ns,
                expected_provider_fingerprint=self.expected_provider_fingerprint,
                expected_model_fingerprint=self.expected_model_fingerprints[ModelComponent.DENSE],
            ),
        )

    def dense_query(self, item: ProviderInput, *, dimension: int | None = None) -> DenseResponse:
        return self._invoke(
            ProviderOperation.DENSE_QUERY,
            ModelComponent.DENSE,
            DenseResponse,
            lambda remaining_ms: self.provider.embed_query(
                item, deadline_ms=remaining_ms()
            ),
            lambda response, provider_deadline: checked_dense_response(
                response,
                (item,),
                dimension=(dimension if dimension is not None else (len(response.items[0].values) if response.items else 1)),
                deadline=provider_deadline,
                now_ns=self.now_ns,
                expected_provider_fingerprint=self.expected_provider_fingerprint,
                expected_model_fingerprint=self.expected_model_fingerprints[ModelComponent.DENSE],
            ),
        )

    def sparse_documents(self, items: Sequence[ProviderInput]) -> SparseResponse:
        requested = tuple(items)
        return self._invoke(
            ProviderOperation.SPARSE_DOCUMENTS,
            ModelComponent.SPARSE,
            SparseResponse,
            lambda remaining_ms: self.provider.sparse_documents(
                requested, deadline_ms=remaining_ms()
            ),
            lambda response, provider_deadline: checked_sparse_response(
                response, requested, deadline=provider_deadline, now_ns=self.now_ns,
                expected_provider_fingerprint=self.expected_provider_fingerprint,
                expected_model_fingerprint=self.expected_model_fingerprints[ModelComponent.SPARSE],
            ),
        )

    def sparse_query(self, item: ProviderInput) -> SparseResponse:
        return self._invoke(
            ProviderOperation.SPARSE_QUERY,
            ModelComponent.SPARSE,
            SparseResponse,
            lambda remaining_ms: self.provider.sparse_query(
                item, deadline_ms=remaining_ms()
            ),
            lambda response, provider_deadline: checked_sparse_response(
                response, (item,), deadline=provider_deadline, now_ns=self.now_ns,
                expected_provider_fingerprint=self.expected_provider_fingerprint,
                expected_model_fingerprint=self.expected_model_fingerprints[ModelComponent.SPARSE],
            ),
        )

    def rerank(
        self,
        query: ProviderInput,
        documents: Sequence[ProviderInput],
        *,
        expected_model_fingerprint: str | None = None,
    ) -> RerankResponse:
        requested = tuple(documents)
        return self._invoke(
            ProviderOperation.RERANK,
            ModelComponent.RERANKER,
            RerankResponse,
            lambda remaining_ms: self.provider.rerank(
                query, requested, deadline_ms=remaining_ms()
            ),
            lambda response, provider_deadline: checked_rerank_response(
                response,
                requested,
                deadline=provider_deadline,
                now_ns=self.now_ns,
                expected_provider_fingerprint=self.expected_provider_fingerprint,
                expected_model_fingerprint=(
                    expected_model_fingerprint
                    if expected_model_fingerprint is not None
                    else self.expected_model_fingerprints[ModelComponent.RERANKER]
                ),
            ),
        )

    def _invoke(
        self,
        operation: ProviderOperation,
        component: ModelComponent | None,
        response_type: type[Any],
        invoke: Callable[[Callable[[], int]], object],
        validate: Callable[[Any, DeadlineBudget], Any],
    ):
        if getattr(self.provider, "absolute_deadline_aware", False) is not True:
            return self._failure(
                operation, component, ProviderCallStatus.UNAVAILABLE
            )
        if self._available_provider_ms() == 0:
            return self._failure(operation, component, ProviderCallStatus.DEADLINE_EXHAUSTED)

        completion = threading.Event()
        self.last_completion = completion
        decision = None
        provider_deadline = self._provider_deadline()
        if self.health_store is not None:
            decision = self.health_store.acquire(
                self.expected_provider_fingerprint or "0" * 64,
                component,
                now_ms=self.wall_clock_ms(),
                operation=operation,
                deadline=provider_deadline,
                now_ns=self.now_ns,
                completion=completion,
                retain_connection=False,
            )
            if not decision.allowed:
                if decision.state == "health_store_failure":
                    phase, causes = self._health_write_diagnostic("admission")
                    self._mark_health_unavailable(phase, cause_types=causes)
                return self._failure(operation, component, decision.status)

        provider_wait_ms = self._available_provider_ms()
        if provider_wait_ms == 0:
            completion.set()
            self._release_after_skipped_admission(decision, completion)
            return self._failure(operation, component, ProviderCallStatus.DEADLINE_EXHAUSTED)

        result: list[object] = []
        provider_errors: list[BaseException] = []
        start_guard = threading.Lock()
        start_state = ["pending"]
        cancel_before_invoke = threading.Event()

        def provider_milliseconds() -> int:
            with start_guard:
                remaining_ms = provider_deadline.remaining_ms(
                    now_ns=self.now_ns
                )
                if (
                    cancel_before_invoke.is_set()
                    or start_state[0] != "claimed"
                    or remaining_ms == 0
                ):
                    start_state[0] = "cancelled_before_start"
                    raise _ProviderStartCancelled()
                # This is the final provider-call gate.  The enriched integer
                # keeps the same absolute cutoff inside deadline-aware I/O.
                start_state[0] = "invoking"
                return AbsoluteDeadlineMilliseconds(
                    remaining_ms, provider_deadline
                )

        def worker_call() -> None:
            cancelled = False
            with start_guard:
                if start_state[0] != "pending":
                    completion.set()
                    return
                worker_wait_ms = provider_deadline.remaining_ms(
                    now_ns=self.now_ns
                )
                if worker_wait_ms == 0:
                    start_state[0] = "cancelled_before_start"
                    cancelled = True
                else:
                    start_state[0] = "claimed"
            if cancelled:
                completion.set()
                return
            try:
                result.append(invoke(provider_milliseconds))
            except _ProviderStartCancelled:
                cancelled = True
            except BaseException as exc:
                provider_errors.append(exc)
            finally:
                with start_guard:
                    if not cancelled:
                        start_state[0] = "finished"
                completion.set()

        worker = threading.Thread(
            target=worker_call,
            name=f"codex-memory-provider-{operation.value}",
            daemon=True,
        )
        worker.start()
        parent_wait_ms = provider_deadline.remaining_ms(now_ns=self.now_ns)
        completed = (
            parent_wait_ms > 0
            and completion.wait(parent_wait_ms / 1000)
        )
        # Event.wait() may report its timeout edge even though the worker has
        # already published its result and set the event.  Linearize the
        # classification on the event state before cancelling or recording a
        # provider timeout; a completed DEADLINE_EXHAUSTED result must release
        # its admission rather than poison the durable circuit.
        if not completed and completion.is_set():
            completed = True
        lock_contended = False
        if completed:
            with start_guard:
                cancelled_before_start = (
                    start_state[0] == "cancelled_before_start"
                )
        else:
            cancel_before_invoke.set()
            if start_guard.acquire(blocking=False):
                try:
                    if start_state[0] in {"pending", "claimed"}:
                        start_state[0] = "cancelled_before_start"
                    elif start_state[0] == "finished":
                        # The worker linearized completion after wait() timed
                        # out but before the parent inspected the start gate.
                        completed = True
                        completion.set()
                    cancelled_before_start = (
                        start_state[0] == "cancelled_before_start"
                    )
                finally:
                    start_guard.release()
            else:
                # Never wait past the absolute deadline for a worker-owned
                # start gate.  The unresolved admission fails closed.
                lock_contended = True
                cancelled_before_start = False
        if cancelled_before_start:
            completion.set()
            self._release_after_skipped_admission(decision, completion)
            return self._failure(
                operation, component,
                ProviderCallStatus.DEADLINE_EXHAUSTED,
            )
        if lock_contended:
            self._mark_health_unavailable("provider_start_gate")
            return self._failure(
                operation, component,
                ProviderCallStatus.DEADLINE_EXHAUSTED,
            )
        if not completed:
            self._record(decision, ProviderCallStatus.TIMEOUT, completion)
            return self._failure(operation, component, ProviderCallStatus.TIMEOUT)
        if (
            not provider_errors
            and len(result) == 1
            and _is_exact_deadline_exhausted_response(
                result[0], response_type, operation, component
            )
        ):
            # The deadline-aware adapter reached the shared cutoff without a
            # confirmed provider timeout. Preserve that completed outcome even
            # if the parent observes it at the same cutoff edge.
            self._record(
                decision, ProviderCallStatus.DEADLINE_EXHAUSTED, completion
            )
            return self._failure(
                operation, component, ProviderCallStatus.DEADLINE_EXHAUSTED
            )
        if provider_deadline.remaining_ms(now_ns=self.now_ns) == 0:
            self._record(decision, ProviderCallStatus.TIMEOUT, completion)
            return self._failure(operation, component, ProviderCallStatus.TIMEOUT)
        if provider_errors:
            self._mark_provider_failure(provider_errors[0])
            self._record(decision, ProviderCallStatus.UNAVAILABLE, completion)
            return self._failure(operation, component, ProviderCallStatus.UNAVAILABLE)
        if not result or not isinstance(result[0], response_type):
            self._record(decision, ProviderCallStatus.INVALID_RESPONSE, completion)
            return self._failure(operation, component, ProviderCallStatus.INVALID_RESPONSE)
        try:
            response = validate(result[0], provider_deadline)
        except Exception:
            self._record(decision, ProviderCallStatus.INVALID_RESPONSE, completion)
            return self._failure(operation, component, ProviderCallStatus.INVALID_RESPONSE)
        if provider_deadline.remaining_ms(now_ns=self.now_ns) == 0:
            self._record(decision, ProviderCallStatus.TIMEOUT, completion)
            return self._failure(operation, component, ProviderCallStatus.TIMEOUT)
        outcome = self._record(decision, response.call.status, completion)
        if (
            response.call.status is ProviderCallStatus.OK
            and outcome is HealthWriteOutcome.NOT_PERSISTED
        ):
            return self._failure(operation, component, ProviderCallStatus.UNAVAILABLE)
        return response

    def _available_provider_ms(self) -> int:
        return self.deadline.remaining_ms(
            now_ns=self.now_ns,
            reserve_ms=_TIMEOUT_HEALTH_RESERVE_MS + _RETURN_RESERVE_MS,
        )

    def _provider_deadline(self) -> DeadlineBudget:
        return DeadlineBudget(
            self.deadline.started_ns,
            max(
                self.deadline.started_ns,
                self.deadline.deadline_ns
                - (_TIMEOUT_HEALTH_RESERVE_MS + _RETURN_RESERVE_MS) * 1_000_000,
            ),
        )

    def _health_deadline(self) -> DeadlineBudget:
        return DeadlineBudget(
            self.deadline.started_ns,
            max(
                self.deadline.started_ns,
                self.deadline.deadline_ns - _RETURN_RESERVE_MS * 1_000_000,
            ),
        )

    def _release_after_skipped_admission(self, decision, completion: threading.Event) -> None:
        if decision is None or self.health_store is None:
            return
        try:
            self.health_store.release(
                decision,
                completion=completion,
                deadline=self._health_deadline(),
                now_ns=self.now_ns,
            )
        except Exception as exc:
            self._mark_health_unavailable("release_after_admission", exc)

    def _record(self, decision, status: ProviderCallStatus, completion: threading.Event) -> HealthWriteOutcome:
        if decision is None or self.health_store is None:
            return HealthWriteOutcome.NOT_APPLICABLE
        phase = "release_not_applicable"
        try:
            if status is ProviderCallStatus.DEADLINE_EXHAUSTED:
                phase = "release_deadline_exhausted"
                self.health_store.release(
                    decision,
                    completion=completion,
                    deadline=self._health_deadline(),
                    now_ns=self.now_ns,
                )
                self.last_health_outcome = HealthWriteOutcome.NOT_APPLICABLE
                return self.last_health_outcome
            now_ms = self.wall_clock_ms()
            health_deadline = self._health_deadline()
            if status is ProviderCallStatus.OK:
                phase = "record_success"
                outcome = self.health_store.record_success(
                    decision, now_ms=now_ms, deadline=health_deadline, now_ns=self.now_ns
                )
            elif status in CIRCUIT_FAILURE_STATUSES:
                phase = "record_failure"
                outcome = self.health_store.record_failure(
                    decision,
                    status,
                    now_ms=now_ms,
                    completion=completion,
                    deadline=health_deadline,
                    now_ns=self.now_ns,
                )
            else:
                # Stable capability/request outcomes are not circuit faults.
                # In particular, UNSUPPORTED, DISABLED, INPUT_TOO_LONG,
                # identity mismatch, and an already exhausted budget must
                # never poison the provider health projection.
                self.health_store.release(
                    decision,
                    completion=completion,
                    deadline=self._health_deadline(),
                    now_ns=self.now_ns,
                )
                outcome = HealthWriteOutcome.NOT_APPLICABLE
            self.last_health_outcome = outcome
            if outcome is HealthWriteOutcome.NOT_PERSISTED:
                self._mark_health_unavailable(*self._health_write_diagnostic(phase))
            return outcome
        except Exception as exc:
            diagnostic_phase, diagnostic_causes = self._health_write_diagnostic(phase)
            self._mark_health_unavailable(
                diagnostic_phase, exc, cause_types=diagnostic_causes
            )
            return HealthWriteOutcome.NOT_PERSISTED

    def _health_write_diagnostic(self, fallback_phase: str) -> tuple[str, tuple[str, ...]]:
        if self.health_store is None:
            return fallback_phase, ()
        stored_phase = self.health_store.last_write_failure_phase
        phase = stored_phase or fallback_phase
        if stored_phase in {"verification", "record_success", "record_failure", "commit", "result_fencing"}:
            phase = "result"
        return phase, self.health_store.last_write_failure_cause_types

    def _mark_health_unavailable(
        self,
        phase: str,
        exc: BaseException | None = None,
        *,
        cause_types: tuple[str, ...] = (),
    ) -> None:
        captured_types: list[str] = list(cause_types)
        seen: set[int] = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            captured_types.append(type(exc).__name__)
            exc = exc.__cause__ or exc.__context__
        self.last_health_outcome = HealthWriteOutcome.NOT_PERSISTED
        self.health_available = False
        self.health_diagnostics.append(
            _HealthDiagnostic(phase, "health_unavailable", tuple(captured_types))
        )

    def _mark_provider_failure(self, exc: BaseException) -> None:
        captured_types: list[str] = []
        seen: set[int] = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            captured_types.append(type(exc).__name__)
            exc = exc.__cause__ or exc.__context__
        self.health_diagnostics.append(
            _HealthDiagnostic("provider", "provider_failure", tuple(captured_types))
        )

    def _failure(
        self,
        operation: ProviderOperation,
        component: ModelComponent | None,
        status: ProviderCallStatus,
    ):
        call = ProviderCall(
            operation,
            component,
            status,
            self.expected_provider_fingerprint,
            None,
            # Do not sample the clock again here: callers may use a strict
            # deterministic clock seam and the status path already performed
            # its final budget check.
            max(0, (self.deadline.deadline_ns - self.deadline.started_ns) // 1_000_000),
            status.value,
        )
        if operation is ProviderOperation.CAPABILITIES:
            return CapabilitiesResponse(call, None)
        if component is ModelComponent.DENSE:
            return DenseResponse(call, ())
        if component is ModelComponent.SPARSE:
            return SparseResponse(call, ())
        return RerankResponse(call, ())
