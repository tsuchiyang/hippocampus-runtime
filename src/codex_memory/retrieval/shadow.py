"""Bounded, output-inert audit of an explicitly requested shadow query."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import threading
import time
from typing import Callable, Final

from codex_memory.ledger_common import canonical_json

from .config import SemanticConfig
from .contracts import (
    RetrievedCandidate,
    RetrievalMode,
    RetrievalQueryResult,
    RetrievalRequest,
    RetrievalStatus,
)
from .paths import RetrievalPaths
from .deadline import DeadlineBudget
from .provider import (
    ProviderInput,
    ProviderOperation,
    RetrievalModelProvider,
    checked_dense_response,
    checked_sparse_response,
)
from .errors import RetrievalValidationError


_AUDIT_MODE: Final[str] = "shadow"
_OPTIONAL_CALLER_ALLOWANCE_NS: Final[int] = 250_000_000


class _DeadlineExpired(RuntimeError):
    pass


def _start_optional_budget(
    total_ms: int | None,
    optional_deadline_ns: int | None,
) -> DeadlineBudget | None:
    """Intersect a local duration with an already-running absolute deadline."""
    if total_ms is None:
        if optional_deadline_ns is None:
            return None
        started_ns = time.monotonic_ns()
        if optional_deadline_ns <= started_ns:
            raise _DeadlineExpired
        return DeadlineBudget(started_ns, optional_deadline_ns)

    budget = DeadlineBudget.start(total_ms, now_ns=time.monotonic_ns)
    if optional_deadline_ns is None or budget.deadline_ns <= optional_deadline_ns:
        return budget
    if optional_deadline_ns <= budget.started_ns:
        raise _DeadlineExpired
    return DeadlineBudget(budget.started_ns, optional_deadline_ns)


def _ordinary_shadow_deadline_ns(optional_deadline_ns: int | None) -> int | None:
    """Reserve the approved caller-to-installed-entry allowance once."""
    if optional_deadline_ns is None:
        return None
    return max(0, optional_deadline_ns - _OPTIONAL_CALLER_ALLOWANCE_NS)


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """A private comparison outcome; it never contains Hook response content."""

    response_content: None = None
    reason: str | None = None
    recorded: bool = False
    wall_clock_ms: int = 0
    injection_id: str | None = None

    @classmethod
    def skipped(cls, reason: str, *, wall_clock_ms: int = 0) -> "ShadowComparison":
        return cls(reason=reason, wall_clock_ms=max(0, wall_clock_ms))


@dataclass(frozen=True, slots=True)
class _ShadowQueryOutcome:
    result: RetrievalQueryResult
    provider_trace: tuple[dict[str, str], ...]
    reranked_result: RetrievalQueryResult | None = None


@dataclass(frozen=True, slots=True)
class ShadowQueryObservation:
    """Output-inert query evidence, including an optional reranked view."""

    result: RetrievalQueryResult
    provider_trace: tuple[dict[str, str], ...]
    reranked_result: RetrievalQueryResult | None = None


def shadow_run_id(
    *, session_id: str, turn_id: str, hook_event: str, query: str | None
) -> str:
    """Hash all request identity material so no raw Hook input enters the audit."""
    return _sha256(
        {
            "hook_event": hook_event,
            "query_sha256": _sha256(query),
            "session_id": session_id,
            "turn_id": turn_id,
        }
    )


def compare(
    *, paths: RetrievalPaths, request: RetrievalRequest, hook_event: str,
    rerank_provider: RetrievalModelProvider | None = None,
    rerank_documents: dict[str, ProviderInput] | None = None,
    rerank_deadline_ms: int = 0, health_store: ProviderHealthStore | None = None,
    deadline_ms: int | None = None,
    semantic_provider: RetrievalModelProvider | None = None,
    semantic_provider_factory: Callable[[], RetrievalModelProvider] | None = None,
    semantic_config: SemanticConfig | None = None,
    disable_dense: bool = False, disable_sparse: bool = False,
    disable_reranker: bool = False, audit_context: dict[str, object] | None = None,
    observer=None, deadline: DeadlineBudget | None = None,
    optional_deadline_ns: int | None = None,
) -> ShadowComparison:
    kwargs = dict(paths=paths, request=request, hook_event=hook_event,
        rerank_provider=rerank_provider, rerank_documents=rerank_documents,
        rerank_deadline_ms=rerank_deadline_ms, health_store=health_store,
        deadline_ms=deadline_ms, semantic_provider=semantic_provider,
        semantic_provider_factory=semantic_provider_factory, semantic_config=semantic_config,
        disable_dense=disable_dense, disable_sparse=disable_sparse,
        disable_reranker=disable_reranker, audit_context=audit_context,
        optional_deadline_ns=optional_deadline_ns)
    if observer is None:
        if deadline is not None:
            return ShadowComparison.skipped("audit_boundary_invalid")
        return _compare(**kwargs)
    from codex_memory.acceptance_observer import AcceptanceObserverError
    try:
        if (optional_deadline_ns is not None or deadline is None
                or deadline is not observer.deadline or type(deadline_ms) is not int
                or deadline_ms != 1000 or deadline.deadline_ns - deadline.started_ns != deadline_ms * 1_000_000):
            raise AcceptanceObserverError("budget_binding")
        result = _compare(observer=observer, deadline=deadline, **kwargs)
        observer.terminal(result)
        return result
    except AcceptanceObserverError:
        return ShadowComparison.skipped("acceptance_observer_failure")


def _compare(
    *,
    paths: RetrievalPaths,
    request: RetrievalRequest,
    hook_event: str,
    rerank_provider: RetrievalModelProvider | None = None,
    rerank_documents: dict[str, ProviderInput] | None = None,
    rerank_deadline_ms: int = 0,
    health_store: ProviderHealthStore | None = None,
    deadline_ms: int | None = None,
    semantic_provider: RetrievalModelProvider | None = None,
    semantic_provider_factory: Callable[[], RetrievalModelProvider] | None = None,
    semantic_config: SemanticConfig | None = None,
    disable_dense: bool = False,
    disable_sparse: bool = False,
    disable_reranker: bool = False,
    audit_context: dict[str, object] | None = None,
    deadline: DeadlineBudget | None = None,
    observer=None,
    optional_deadline_ns: int | None = None,
) -> ShadowComparison:
    """Query a healthy cache and append only safe hashes; every failure is inert."""
    started_ns = time.monotonic_ns()
    started_at = None
    if audit_context is not None:
        from datetime import UTC, datetime

        started_at = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        if hook_event != "UserPromptSubmit" or deadline_ms is None:
            return ShadowComparison.skipped("audit_boundary_invalid")
    if not isinstance(request, RetrievalRequest) or request.mode is RetrievalMode.ACTIVE:
        return ShadowComparison.skipped("active_rejected", wall_clock_ms=_elapsed_ms(started_ns))
    if semantic_provider is not None and semantic_provider_factory is not None:
        return ShadowComparison.skipped("provider_conflict", wall_clock_ms=_elapsed_ms(started_ns))
    if semantic_provider is not None or semantic_provider_factory is not None or semantic_config is not None:
        if request.mode is not RetrievalMode.SHADOW:
            return ShadowComparison.skipped("shadow_mode_required", wall_clock_ms=_elapsed_ms(started_ns))
    try:
        if deadline_ms is not None and (isinstance(deadline_ms, bool) or not isinstance(deadline_ms, int) or deadline_ms < 1):
            return ShadowComparison.skipped("deadline_exhausted", wall_clock_ms=_elapsed_ms(started_ns))
        if optional_deadline_ns is not None and (
            isinstance(optional_deadline_ns, bool)
            or not isinstance(optional_deadline_ns, int)
            or optional_deadline_ns < 0
        ):
            return ShadowComparison.skipped("deadline_exhausted", wall_clock_ms=_elapsed_ms(started_ns))
        ordinary_deadline_ns = _ordinary_shadow_deadline_ns(
            optional_deadline_ns
        )
        if deadline is not None:
            budget = deadline
        elif (
            semantic_provider is not None or semantic_provider_factory is not None
        ) and semantic_config is not None:
            effective_deadline_ms = semantic_config.query_deadline_ms if deadline_ms is None else deadline_ms
            budget = _start_optional_budget(
                effective_deadline_ms, ordinary_deadline_ns
            )
        else:
            budget = _start_optional_budget(deadline_ms, ordinary_deadline_ns)
        if semantic_provider_factory is not None and semantic_config is not None:
            semantic_provider = _bounded_call(semantic_provider_factory, budget)
        if semantic_provider is not None and semantic_config is not None:
            outcome = _shadow_query(
                paths=paths,
                request=request,
                provider=semantic_provider,
                semantic_config=semantic_config,
                disable_dense=disable_dense,
                disable_sparse=disable_sparse,
                disable_reranker=disable_reranker,
                deadline=budget,
                observer=observer,
            )
            result = outcome.result
            provider_trace = outcome.provider_trace
        else:
            # Keep the query engine's cold initialization inside the same
            # absolute query budget.  The Hook imports this boundary before it
            # starts that budget, so an eager module import would otherwise be
            # unbounded process-start work.
            if rerank_provider is not None:
                if rerank_deadline_ms < 1:
                    return ShadowComparison.skipped("deadline_exhausted", wall_clock_ms=_elapsed_ms(started_ns))
                if budget is None:
                    budget = DeadlineBudget.start(rerank_deadline_ms, now_ns=time.monotonic_ns)
            RetrievalService = _bounded_call(_load_retrieval_service, budget)

            service = RetrievalService(paths)
            if rerank_provider is None:
                result = _bounded_call(lambda: service.query(request, **({"deadline": budget, "observer": observer} if observer is not None else {})), budget)
                provider_trace = ()
            else:
                ProviderHealthStore = _bounded_call(_load_provider_health_store, budget)
                result = _bounded_call(lambda: service.query(request, **({"deadline": budget, "observer": observer} if observer is not None else {})), budget)
                rerank_budget = _start_optional_budget(
                    rerank_deadline_ms,
                    budget.deadline_ns if budget is not None else None,
                )
                observation = service.query_with_rerank(
                    request,
                    provider=rerank_provider,
                    documents=rerank_documents or {},
                    deadline_ms=rerank_deadline_ms,
                    health_store=health_store or ProviderHealthStore(paths.database_path),
                    deadline=rerank_budget,
                    base_result=result,
                )
                # Shadow intentionally audits and returns only the unchanged
                # deterministic result; the optional observation is never Hook IO.
                result = observation.result
                provider_trace = () if observation.rerank is None else ({
                    "component": "reranker",
                    "operation": ProviderOperation.RERANK.value,
                    "status": observation.rerank.status.value,
                },)
    except _DeadlineExpired:
        return ShadowComparison.skipped("deadline_exhausted", wall_clock_ms=_elapsed_ms(started_ns))
    except Exception as exc:
        if observer is not None:
            from codex_memory.acceptance_observer import AcceptanceObserverError
            if isinstance(exc, AcceptanceObserverError):
                raise
        return ShadowComparison.skipped("query_failure", wall_clock_ms=_elapsed_ms(started_ns))
    if observer is not None:
        observer.query_result(result)
    if (
        result.generation is None
        or result.canonical_highwater is None
        or result.indexed_highwater is None
    ):
        return ShadowComparison.skipped(_safe_reason(result), wall_clock_ms=_elapsed_ms(started_ns))
    if budget is not None and budget.remaining_ms(now_ns=time.monotonic_ns) == 0:
        return ShadowComparison.skipped("deadline_exhausted", wall_clock_ms=_elapsed_ms(started_ns))
    query_duration_ms = _elapsed_ms(started_ns)
    if observer is not None:
        observer.emit("audit_attempt", {})
    try:
        append = lambda: _append_audit(
            paths, request, hook_event, result, deadline=budget,
            now_ns=time.monotonic_ns, started_ns=started_ns,
            provider_trace=provider_trace,
        )
        if audit_context is not None:
            _bounded_call(append, budget)
        else:
            append()
    except _DeadlineExpired:
        if observer is not None:
            observer.emit("audit_result", {"recorded": False})
        return ShadowComparison.skipped("deadline_exhausted", wall_clock_ms=_elapsed_ms(started_ns))
    except Exception:
        if observer is not None:
            observer.emit("audit_result", {"recorded": False})
        return ShadowComparison.skipped("audit_failure", wall_clock_ms=_elapsed_ms(started_ns))
    if observer is not None:
        observer.emit("audit_result", {"recorded": True})
    injection_id = None
    if audit_context is not None:
        try:
            injection_id = _bounded_call(lambda: _capture_injection(
                paths=paths, request=request, result=result,
                audit_context=audit_context, deadline=budget,
                started_ns=started_ns, started_at=started_at,
                query_duration_ms=query_duration_ms, semantic_config=semantic_config,
            ), budget)
        except _DeadlineExpired:
            return ShadowComparison(reason="deadline_exhausted", recorded=True, wall_clock_ms=_elapsed_ms(started_ns))
        except Exception as exc:
            from codex_memory.injection_manifest import InjectionAuditError

            reason = exc.reason.value if isinstance(exc, InjectionAuditError) else "injection_failure"
            return ShadowComparison(reason=reason, recorded=True, wall_clock_ms=_elapsed_ms(started_ns))
    return ShadowComparison(recorded=True, wall_clock_ms=_elapsed_ms(started_ns), injection_id=injection_id)


def _capture_injection(
    *, paths, request, result, audit_context, deadline, started_ns,
    started_at, query_duration_ms, semantic_config,
) -> str:
    """Optional write stage, bounded by the query's original absolute deadline.

    The parent stops waiting at that deadline. Uninterruptible OS I/O may still
    finish with a prepared artifact, but never upgrades the parent's outcome.
    No worker receives the main Hook SQLite connection or output streams.
    """
    from datetime import UTC, datetime
    from codex_memory.blob_store import BlobStore
    from codex_memory.injection_manifest import InjectionAuditError, InjectionAuditReason
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    from .assembler import build_phase2_packet
    from .injection_audit import implementation_fingerprint, prepare_shadow_manifest

    _require_budget(deadline, time.monotonic_ns)
    if _sha256(audit_context["source_payload"]) != audit_context["source"]["event_blob_sha256"]:
        raise InjectionAuditError(InjectionAuditReason.SOURCE_MISMATCH)
    assembly_started = time.monotonic_ns()
    assembled = build_phase2_packet(paths, request=request, result=result)
    assembly_ms = _elapsed_ms(assembly_started)
    _require_budget(deadline, time.monotonic_ns)
    disabled = {"configured": "disabled", "observed": "disabled", "validated": "disabled"}
    environment = {
        "provider": dict(disabled), "model": dict(disabled), "preprocessing": dict(disabled),
        "config_sha256": audit_context["config_sha256"],
        "implementation_sha256": implementation_fingerprint(deadline=deadline),
        "legacy_delivery": audit_context["legacy_delivery"],
        "host_acknowledgement": "unknown",
    }
    if semantic_config is not None:
        # Config is not proof of any service response or model identity. This
        # boundary has no verified identity observation to attest, so says so.
        environment["provider"] = {"configured": semantic_config.expected_provider_fingerprint, "observed": "unknown", "validated": "unknown"}
        environment["model"] = {"configured": semantic_config.expected_dense_fingerprint or "disabled", "observed": "unknown", "validated": "unknown"}
        environment["preprocessing"] = {"configured": "unknown", "observed": "unknown", "validated": "unknown"}
    blobs = BlobStore(paths.runtime_root / "blobs")
    manifest = prepare_shadow_manifest(
        request=request, result=result, assembled=assembled, source=audit_context["source"],
        legacy_packet=audit_context["legacy_packet"], legacy_response_bytes=audit_context["legacy_response_bytes"],
        environment=environment,
        timing={
            "started_at": started_at,
            "prepared_at": datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "query_duration_ms": query_duration_ms, "assembly_duration_ms": assembly_ms,
            "prepare_duration_ms": _elapsed_ms(started_ns),
            "budget_ms": (deadline.deadline_ns - deadline.started_ns) // 1_000_000,
        }, blobs=blobs, deadline=deadline,
    )
    store = InjectionStore(InjectionPaths.for_repo(paths.repo_root), blobs)
    committed = store.commit(manifest, deadline=deadline)
    _require_budget(deadline, time.monotonic_ns)
    return committed.injection_id


def query_shadow(
    *,
    paths: RetrievalPaths,
    request: RetrievalRequest,
    provider: RetrievalModelProvider,
    semantic_config: SemanticConfig,
    disable_dense: bool = False,
    disable_sparse: bool = False,
    disable_reranker: bool = False,
) -> RetrievalQueryResult:
    """Run the explicit semantic read path without building or catching up.

    Callers must have already established a valid shadow configuration.  This
    helper owns no configuration file and performs no canonical write.  It is
    intentionally safe to call from a CLI or the post-output Hook shadow seam.
    """
    return query_shadow_observation(
        paths=paths, request=request, provider=provider,
        semantic_config=semantic_config, disable_dense=disable_dense,
        disable_sparse=disable_sparse, disable_reranker=disable_reranker,
    ).result


def query_shadow_observation(
    *,
    paths: RetrievalPaths,
    request: RetrievalRequest,
    provider: RetrievalModelProvider,
    semantic_config: SemanticConfig,
    disable_dense: bool = False,
    disable_sparse: bool = False,
    disable_reranker: bool = False,
) -> ShadowQueryObservation:
    """Run the shadow query and expose sanitized provider/rerank evidence.

    ``result`` remains the deterministic legacy view.  The optional
    ``reranked_result`` is for offline evaluation only and is never used by
    Hook response serialization.
    """
    budget = DeadlineBudget.start(semantic_config.query_deadline_ms, now_ns=time.monotonic_ns)
    outcome = _shadow_query(
        paths=paths, request=request, provider=provider,
        semantic_config=semantic_config, disable_dense=disable_dense,
        disable_sparse=disable_sparse, disable_reranker=disable_reranker,
        deadline=budget,
    )
    return ShadowQueryObservation(
        outcome.result, outcome.provider_trace, outcome.reranked_result
    )


def _shadow_query(
    *,
    paths: RetrievalPaths,
    request: RetrievalRequest,
    provider: RetrievalModelProvider,
    semantic_config: SemanticConfig,
    disable_dense: bool,
    disable_sparse: bool,
    disable_reranker: bool,
    deadline: DeadlineBudget | None = None,
    observer=None,
) -> _ShadowQueryOutcome:
    if not isinstance(request, RetrievalRequest):
        raise RetrievalValidationError("invalid retrieval query")
    if request.mode is RetrievalMode.ACTIVE:
        raise RetrievalValidationError("active retrieval mode is unavailable")
    if request.mode is not RetrievalMode.SHADOW:
        raise RetrievalValidationError("shadow retrieval mode is required")
    if not isinstance(semantic_config, SemanticConfig):
        raise TypeError("semantic shadow configuration is required")
    if any(type(value) is not bool for value in (disable_dense, disable_sparse, disable_reranker)):
        raise TypeError("semantic component switches must be booleans")
    budget = deadline or DeadlineBudget.start(semantic_config.query_deadline_ms, now_ns=time.monotonic_ns)
    # Import after the deadline exists: module initialization is part of the
    # bounded semantic query, just like cache opening and provider admission.
    (
        RetrievalService,
        SemanticQueryInputs,
        ProviderCallController,
        ProviderHealthStore,
    ) = _bounded_call(_load_semantic_query_dependencies, budget)

    service = RetrievalService(paths)
    controller = ProviderCallController(
        provider,
        budget,
        ProviderHealthStore(paths.database_path),
        semantic_config.expected_provider_fingerprint,
        now_ns=time.monotonic_ns,
        expected_dense_fingerprint=semantic_config.expected_dense_fingerprint,
        expected_sparse_fingerprint=semantic_config.expected_sparse_fingerprint,
        expected_reranker_fingerprint=semantic_config.expected_reranker_fingerprint,
    )
    # Establish the deterministic floor first.  A missing, corrupt, stale
    # beyond-recovery, or unavailable canonical/cache state must not trigger
    # model I/O merely because shadow mode was requested.
    result = _bounded_call(lambda: service.query(request, **({"deadline": budget, "observer": observer} if observer is not None else {})), budget)
    if (
        result.status not in {RetrievalStatus.CURRENT, RetrievalStatus.STALE}
        or result.generation is None
        or not result.candidates
    ):
        return _ShadowQueryOutcome(result, ())
    query_input = ProviderInput.from_text("query", 0, request.query or "")
    dense_vector = None
    sparse_weights = None
    semantic_fingerprint_mismatch = False
    trace: list[dict[str, str]] = []
    reranked_result: RetrievalQueryResult | None = None
    active_identities = _bounded_call(lambda: _active_semantic_identities(paths), budget)
    if active_identities is not None:
        expected_components = {
            "dense": semantic_config.expected_dense_fingerprint,
            "sparse": semantic_config.expected_sparse_fingerprint,
            "reranker": semantic_config.expected_reranker_fingerprint,
        }
        active_provider = active_identities.get("provider")
        if (
            isinstance(active_provider, str)
            and active_provider != semantic_config.expected_provider_fingerprint
        ) or any(
            expected_components[name] is not None
            and active_identities.get(name) is not None
            and active_identities.get(name) != expected_components[name]
            for name in ("dense", "sparse", "reranker")
        ):
            return _ShadowQueryOutcome(
                result,
                (_trace("provider", ProviderOperation.CAPABILITIES, "fingerprint_mismatch"),),
            )
    if semantic_config.dense and not disable_dense:
        response = controller.dense_query(query_input)
        dense_valid = _matches_expected_fingerprints(
            response.call, semantic_config.expected_provider_fingerprint,
            semantic_config.expected_dense_fingerprint,
        )
        if (
            active_identities is not None
            and active_identities["dense"] is not None
            and active_identities["dense"] != semantic_config.expected_dense_fingerprint
        ):
            dense_valid = False
        trace.append(_trace("dense", ProviderOperation.DENSE_QUERY, _provider_trace_status(response.call, dense_valid)))
        if dense_valid and response.items:
            dense_vector = response.items[0].values
        elif response.call.status.value in {"ok", "fingerprint_mismatch"}:
            semantic_fingerprint_mismatch = True
    if semantic_config.sparse and not disable_sparse:
        response = controller.sparse_query(query_input)
        sparse_valid = _matches_expected_fingerprints(
            response.call, semantic_config.expected_provider_fingerprint,
            semantic_config.expected_sparse_fingerprint,
        )
        if (
            active_identities is not None
            and active_identities["sparse"] is not None
            and active_identities["sparse"] != semantic_config.expected_sparse_fingerprint
        ):
            sparse_valid = False
        trace.append(_trace("sparse", ProviderOperation.SPARSE_QUERY, _provider_trace_status(response.call, sparse_valid)))
        if sparse_valid and response.items:
            sparse_weights = response.items[0].weights
        elif response.call.status.value in {"ok", "fingerprint_mismatch"}:
            semantic_fingerprint_mismatch = True
    result = _bounded_call(
        lambda: service.query(
            request,
            semantic_query=SemanticQueryInputs(dense_vector, sparse_weights),
        ),
        budget,
    )
    if semantic_config.reranker and not disable_reranker and not semantic_fingerprint_mismatch:
        documents = _bounded_call(lambda: _rerank_documents(paths, result), budget)
        if documents:
            observation = service.query_with_rerank(
                request,
                provider=provider,
                documents=documents,
                deadline_ms=semantic_config.query_deadline_ms,
                health_store=ProviderHealthStore(paths.database_path),
                expected_provider_fingerprint=semantic_config.expected_provider_fingerprint,
                expected_model_fingerprint=semantic_config.expected_reranker_fingerprint,
                provider_controller=controller,
                base_result=result,
            )
            if observation.rerank is not None:
                trace.append(_trace("reranker", ProviderOperation.RERANK, observation.rerank.status.value))
                if observation.rerank.status.value == "ok":
                    reranked_result = _reranked_result(result, observation.rerank.candidates)
    return _ShadowQueryOutcome(result, tuple(trace), reranked_result)


def _reranked_result(
    result: RetrievalQueryResult,
    candidates: tuple[RetrievedCandidate, ...],
) -> RetrievalQueryResult | None:
    """Build a private reranked view while retaining the immutable result fields."""
    if not isinstance(candidates, tuple):
        return None
    baseline_ids = tuple(candidate.candidate_id for candidate in result.candidates)
    reranked_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(reranked_ids) != len(baseline_ids) or set(reranked_ids) != set(baseline_ids):
        return None
    trace_by_id = {item.candidate_id: item for item in result.trace}
    if len(trace_by_id) != len(result.trace) or any(identifier not in trace_by_id for identifier in reranked_ids):
        return None
    ordered_trace = [
        replace(
            trace_by_id[identifier],
            selected=True,
            reason_codes=tuple((*trace_by_id[identifier].reason_codes, "reranked")),
        )
        for identifier in reranked_ids
    ]
    selected_ids = set(reranked_ids)
    ordered_trace.extend(
        replace(item, selected=False)
        for item in result.trace
        if item.candidate_id not in selected_ids
    )
    return replace(result, candidates=candidates, trace=tuple(ordered_trace))


def _rerank_documents(
    paths: RetrievalPaths, result: RetrievalQueryResult
) -> dict[str, ProviderInput]:
    """Read only bounded derived projection text for an optional reranker."""
    if result.generation is None:
        return {}
    identifiers = tuple(candidate.candidate_id for candidate in result.candidates if candidate.candidate_kind == "record")
    if not identifiers:
        return {}
    try:
        from .cache_database import connect_cache_database

        connection = connect_cache_database(paths.database_path, readonly=True)
        try:
            placeholders = ",".join("?" for _ in identifiers)
            rows = connection.execute(
                f"SELECT record_id, search_text FROM retrieval_documents WHERE generation_id = ? AND record_id IN ({placeholders}) ORDER BY record_seq",
                (result.generation.generation_id, *identifiers),
            ).fetchall()
            return {
                row["record_id"]: ProviderInput.from_text(row["record_id"], ordinal, row["search_text"])
                for ordinal, row in enumerate(rows)
            }
        finally:
            connection.close()
    except Exception:
        return {}


def _load_retrieval_service():
    from .service import RetrievalService

    return RetrievalService


def _load_provider_health_store():
    from .provider_health import ProviderHealthStore

    return ProviderHealthStore


def _load_semantic_query_dependencies():
    from .provider_control import ProviderCallController
    from .provider_health import ProviderHealthStore
    from .service import RetrievalService, SemanticQueryInputs

    return (
        RetrievalService,
        SemanticQueryInputs,
        ProviderCallController,
        ProviderHealthStore,
    )


def _bounded_call(callback, budget: DeadlineBudget | None):
    """Run a read-only shadow stage without allowing SQLite to outlive its budget."""
    if budget is None:
        return callback()
    remaining_ns = budget.deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise _DeadlineExpired
    completed = threading.Event()
    result: list[object] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(callback())
        except BaseException as exc:
            error.append(exc)
        finally:
            completed.set()

    threading.Thread(
        target=worker,
        name="codex-memory-shadow-read",
        daemon=True,
    ).start()
    # Use the existing absolute deadline, without truncating its final sub-ms.
    remaining_ns = max(0, budget.deadline_ns - time.monotonic_ns())
    if not completed.wait(remaining_ns / 1_000_000_000):
        raise _DeadlineExpired
    if time.monotonic_ns() >= budget.deadline_ns:
        raise _DeadlineExpired
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("bounded shadow stage returned no result")
    return result[0]


def _trace(component: str, operation: ProviderOperation, status: str) -> dict[str, str]:
    return {"component": component, "operation": operation.value, "status": status}


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.monotonic_ns() - started_ns) // 1_000_000)


def _matches_expected_fingerprints(
    call: object,
    expected_provider_fingerprint: str,
    expected_model_fingerprint: str | None,
) -> bool:
    """Authorise a semantic response only when both configured identities match."""
    return (
        getattr(call, "status", None).value == "ok"
        and getattr(call, "provider_fingerprint", None) == expected_provider_fingerprint
        and expected_model_fingerprint is not None
        and getattr(call, "model_fingerprint", None) == expected_model_fingerprint
    )


def _provider_trace_status(call: object, identity_matches: bool) -> str:
    """Keep transport/result failures distinct from an OK identity mismatch."""
    status = getattr(call, "status", None)
    if status is not None and status.value != "ok":
        return status.value
    return "ok" if identity_matches else "fingerprint_mismatch"


def _active_semantic_identities(paths: RetrievalPaths) -> dict[str, str | None] | None:
    """Read the active semantic manifest identity without trusting config alone.

    ``None`` means no usable active semantic generation was established.  A
    returned mapping may contain a component ``None`` value, which explicitly
    means that the active generation has no identity for that component.
    """
    connection = None
    try:
        from .cache_database import connect_cache_database, verify_cache_database
        from .semantic_paths import SemanticGenerationPaths
        from .semantic_store import verify_semantic_generation

        connection = connect_cache_database(paths.database_path, readonly=True)
        verify_cache_database(connection)
        state = connection.execute(
            "SELECT active_semantic_generation_id, validation_state FROM retrieval_semantic_state WHERE singleton_id = 1"
        ).fetchone()
        if state is None or state["validation_state"] != "validated" or not isinstance(state["active_semantic_generation_id"], str):
            return None
        semantic_paths = SemanticGenerationPaths.for_generation(
            paths.generations_dir, state["active_semantic_generation_id"]
        )
        manifest = verify_semantic_generation(semantic_paths)
        return {
            "provider": manifest.provider_fingerprint,
            "dense": manifest.dense_model_fingerprint,
            "sparse": manifest.sparse_model_fingerprint,
            "reranker": manifest.reranker_model_fingerprint,
        }
    except Exception:
        return None
    finally:
        if connection is not None:
            connection.close()


def _append_audit(
    paths: RetrievalPaths,
    request: RetrievalRequest,
    hook_event: str,
    result: RetrievalQueryResult,
    *,
    deadline: DeadlineBudget | None = None,
    now_ns=time.monotonic_ns,
    started_ns: int | None = None,
    provider_trace: tuple[dict[str, str], ...] = (),
) -> None:
    """Append a deterministic run and trace set in exactly one transaction."""
    from . import index_store
    from .cache_database import (
        connect_cache_database,
        immediate_transaction,
        verify_cache_database,
    )

    generation = result.generation
    assert generation is not None
    run_id = _sha256(
        {
            "generation_id": generation.generation_id,
            "hook_event": hook_event,
            "request_id": shadow_run_id(
                session_id=request.session_id,
                turn_id=request.turn_id,
                hook_event=hook_event,
                query=request.query,
            ),
            "result": _safe_result_document(result),
        }
    )
    paths.validate()
    _require_budget(deadline, now_ns)
    connection = connect_cache_database(paths.database_path)
    try:
        connection.execute("PRAGMA busy_timeout=0")
        with immediate_transaction(connection):
            _require_budget(deadline, now_ns)
            verify_cache_database(connection)
            _require_budget(deadline, now_ns)
            current = connection.execute(
                "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
            ).fetchone()
            if current is None or current["active_generation_id"] != generation.generation_id:
                raise RuntimeError("retrieval generation changed")
            if connection.execute(
                "SELECT 1 FROM retrieval_runs WHERE run_id = ?", (run_id,)
            ).fetchone() is not None:
                return
            _require_budget(deadline, now_ns)
            trace_document = _safe_trace_document(result)
            index_store.append_run(
                connection,
                run_id=run_id,
                mode=_AUDIT_MODE,
                generation_id=generation.generation_id,
                query_sha256=_sha256(request.query),
                canonical_highwater_json=canonical_json(
                    result.canonical_highwater.to_dict()
                ),
                indexed_highwater_json=canonical_json(
                    result.indexed_highwater.to_dict()
                ),
                status=result.status.value,
                degraded_reasons_json=canonical_json(
                    [reason.value for reason in result.degraded_reasons]
                ),
                trace_sha256=_sha256(trace_document),
                created_at=generation.updated_at,
                wall_clock_ms=0 if started_ns is None else _elapsed_ms(started_ns),
                provider_trace_json=canonical_json(list(provider_trace)),
            )
            kinds = {candidate.candidate_id: candidate.candidate_kind for candidate in result.candidates}
            for ordinal, trace in enumerate(result.trace):
                for channel in trace.source_channels:
                    _require_budget(deadline, now_ns)
                    contribution = next(
                        (
                            item
                            for item in trace.contributions
                            if item.channel is channel
                        ),
                        None,
                    )
                    channel_rank = 1 if contribution is None else contribution.channel_rank
                    numerator = 0 if contribution is None else contribution.numerator
                    denominator = 1 if contribution is None else contribution.denominator
                    trace_value = {
                        "candidate_id": trace.candidate_id,
                        "channel": channel.value,
                        "ordinal": ordinal,
                        "run_id": run_id,
                    }
                    index_store.append_candidate_trace(
                        connection,
                        trace_id=_sha256(trace_value),
                        run_id=run_id,
                        candidate_kind=kinds.get(trace.candidate_id, "record"),
                        candidate_id=trace.candidate_id,
                        channel=channel.value,
                        channel_rank=channel_rank,
                        contribution_numerator=numerator,
                        contribution_denominator=denominator,
                        fused_rank=trace.fused_rank,
                        selected=trace.selected,
                        reason_codes_json=canonical_json(list(trace.reason_codes)),
                        trace_sha256=_sha256(
                            {
                                "candidate_id": trace.candidate_id,
                                "channel": channel.value,
                                "fused_rank": trace.fused_rank,
                                "reason_codes": list(trace.reason_codes),
                                "selected": trace.selected,
                            }
                        ),
                    )
    finally:
        connection.close()


def _require_budget(deadline: DeadlineBudget | None, now_ns) -> None:
    if deadline is not None and deadline.remaining_ms(now_ns=now_ns) == 0:
        raise _DeadlineExpired


def _safe_result_document(result: RetrievalQueryResult) -> dict[str, object]:
    return {
        "canonical_highwater": result.canonical_highwater.to_dict()
        if result.canonical_highwater
        else None,
        "degraded_reasons": [reason.value for reason in result.degraded_reasons],
        "generation_id": result.generation.generation_id if result.generation else None,
        "indexed_highwater": result.indexed_highwater.to_dict()
        if result.indexed_highwater
        else None,
        "status": result.status.value,
        "trace": _safe_trace_document(result),
    }


def _safe_trace_document(result: RetrievalQueryResult) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": trace.candidate_id,
            "contributions": [item.to_dict() for item in trace.contributions],
            "fused_rank": trace.fused_rank,
            "reason_codes": list(trace.reason_codes),
            "selected": trace.selected,
            "source_channels": [channel.value for channel in trace.source_channels],
        }
        for trace in result.trace
    ]


def _safe_reason(result: RetrievalQueryResult) -> str:
    if result.degraded_reasons:
        return result.degraded_reasons[0].value
    return result.status.value


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
