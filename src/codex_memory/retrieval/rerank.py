"""Bounded, fail-closed reranking for already eligible retrieval candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import threading
import time

from .contracts import RetrievedCandidate
from .deadline import DeadlineBudget
from .errors import RetrievalValidationError
from .provider import (
    ProviderCall,
    ProviderCallStatus,
    ProviderInput,
    RetrievalModelProvider,
    checked_rerank_response,
)


@dataclass(frozen=True, slots=True)
class RerankResult:
    """A safe replacement for ordinary candidate slots, never a new candidate set."""

    candidates: tuple[RetrievedCandidate, ...]
    status: ProviderCallStatus
    call: ProviderCall | None
    scores: tuple[tuple[str, float], ...] = ()
    completion: threading.Event | None = None


def rerank_candidates(
    provider: RetrievalModelProvider,
    query: ProviderInput,
    candidates: Sequence[RetrievedCandidate],
    documents: Mapping[str, ProviderInput],
    *,
    top_n: int,
    deadline_ms: int,
    deadline: DeadlineBudget | None = None,
    now_ns=None,
    expected_provider_fingerprint: str | None = None,
    expected_model_fingerprint: str | None = None,
    completion: threading.Event | None = None,
    reserve_ms: int = 0,
    controller: object | None = None,
) -> RerankResult:
    """Rerank at most ``top_n`` ordinary candidates or preserve fused ordering.

    Constitution-mandatory and same-session temporary candidates occupy their
    original slots.  Consequently an optional model can never drop, move, or
    manufacture protected candidates.
    """
    if not isinstance(query, ProviderInput):
        raise RetrievalValidationError("query must be ProviderInput")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 64:
        raise RetrievalValidationError("top_n must be an integer between 1 and 64")
    if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, int) or deadline_ms < 0:
        raise RetrievalValidationError("deadline_ms must be a non-negative integer")
    if deadline is not None and not isinstance(deadline, DeadlineBudget):
        raise RetrievalValidationError("deadline must be DeadlineBudget or null")
    if isinstance(reserve_ms, bool) or not isinstance(reserve_ms, int) or reserve_ms < 0:
        raise RetrievalValidationError("reserve_ms must be a non-negative integer")
    if now_ns is None:
        now_ns = time.monotonic_ns
    if not callable(now_ns):
        raise RetrievalValidationError("now_ns must be callable")
    baseline = tuple(candidates)
    if any(not isinstance(candidate, RetrievedCandidate) for candidate in baseline):
        raise RetrievalValidationError("candidates must be RetrievedCandidate values")
    if deadline_ms == 0:
        return RerankResult(baseline, ProviderCallStatus.DEADLINE_EXHAUSTED, None)
    budget = deadline or DeadlineBudget.start(deadline_ms, now_ns=now_ns)
    remaining_ms = budget.remaining_ms(now_ns=now_ns, cap_ms=deadline_ms, reserve_ms=reserve_ms)
    if remaining_ms == 0:
        return RerankResult(baseline, ProviderCallStatus.DEADLINE_EXHAUSTED, None)

    ordinary_slots = [
        index for index, candidate in enumerate(baseline)
        if not candidate.mandatory and candidate.candidate_kind != "temporary"
    ][:top_n]
    if not ordinary_slots:
        return RerankResult(baseline, ProviderCallStatus.OK, None)
    requested: list[ProviderInput] = []
    for position, slot in enumerate(ordinary_slots):
        source = documents.get(baseline[slot].candidate_id)
        if not isinstance(source, ProviderInput):
            return RerankResult(baseline, ProviderCallStatus.INVALID_RESPONSE, None)
        requested.append(ProviderInput.from_text(source.item_id, position, source.text))

    if controller is None:
        from .provider_control import ProviderCallController
        controller = ProviderCallController(
            provider,
            budget,
            None,
            expected_provider_fingerprint,
            now_ns=now_ns,
            expected_reranker_fingerprint=expected_model_fingerprint,
        )
    if not hasattr(controller, "rerank"):
        return RerankResult(baseline, ProviderCallStatus.INVALID_RESPONSE, None, completion=completion)
    response = controller.rerank(
        query,
        tuple(requested),
        expected_model_fingerprint=expected_model_fingerprint,
    )
    completion = getattr(controller, "last_completion", None) or completion
    if completion is None:
        completion = threading.Event()
        completion.set()
    if response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED:
        return RerankResult(baseline, ProviderCallStatus.TIMEOUT, response.call, completion=completion)
    # ``ProviderCallController`` has already discarded any late response at
    # its worker boundary.  Do not sample the clock a second time here: this
    # keeps one request on one absolute budget and preserves deterministic
    # clock seams used by callers.
    response = checked_rerank_response(
        response, requested,
        expected_provider_fingerprint=expected_provider_fingerprint,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    if response.call.status is not ProviderCallStatus.OK:
        return RerankResult(baseline, response.call.status, response.call, completion=completion)

    by_id = {item.item_id: item.score for item in response.items}
    if len(by_id) != len(requested):
        return RerankResult(baseline, ProviderCallStatus.INVALID_RESPONSE, response.call, completion=completion)
    ordered_slots = sorted(
        ordinary_slots,
        key=lambda slot: (-by_id[baseline[slot].candidate_id], ordinary_slots.index(slot)),
    )
    replacement = list(baseline)
    for destination, source in zip(ordinary_slots, ordered_slots, strict=True):
        replacement[destination] = baseline[source]
    return RerankResult(
        tuple(replacement),
        ProviderCallStatus.OK,
        response.call,
        tuple((item.item_id, item.score) for item in response.items),
        completion,
    )


def _bounded_rerank_call(
    controller: object,
    query: ProviderInput,
    documents: tuple[ProviderInput, ...],
    remaining_ms: int,
    *,
    completion: threading.Event | None,
):
    """Compatibility shim for callers of the old private helper.

    Runtime callers use ``ProviderCallController`` directly; keeping this
    helper free of provider I/O prevents a second, unbudgeted call path.
    """
    if not hasattr(controller, "rerank"):
        return None, completion or threading.Event()
    response = controller.rerank(query, documents)
    complete = completion or threading.Event()
    complete.set()
    return response, complete
