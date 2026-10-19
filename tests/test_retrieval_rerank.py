from __future__ import annotations

from codex_memory.retrieval.contracts import CandidateChannel, CandidateContribution, RetrievedCandidate
from codex_memory.retrieval.provider import ProviderInput
from codex_memory.retrieval.rerank import rerank_candidates
from retrieval_provider_fakes import FakeProvider


def _candidate(identifier: str, *, mandatory: bool = False, temporary: bool = False) -> RetrievedCandidate:
    return RetrievedCandidate(
        "temporary" if temporary else "record", identifier, "constitution" if mandatory else "decision",
        "project", "project", "active", 1, "a" * 64, mandatory, 1, 1,
        (CandidateContribution(CandidateChannel.EXACT, 1, 1, 1),),
    )


def _input(identifier: str, position: int) -> ProviderInput:
    return ProviderInput.from_text(identifier, position, f"text for {identifier}")


def test_rerank_only_reorders_bounded_ordinary_slots_and_preserves_protected_slots() -> None:
    provider = FakeProvider()
    candidates = (_candidate("mandatory", mandatory=True), _candidate("first"), _candidate("temporary", temporary=True), _candidate("second"), _candidate("third"))
    documents = {candidate.candidate_id: _input(candidate.candidate_id, index) for index, candidate in enumerate(candidates)}

    result = rerank_candidates(provider, _input("query", 0), candidates, documents, top_n=2, deadline_ms=500)

    assert result.status.value == "ok"
    assert result.candidates[0].candidate_id == "mandatory"
    assert result.candidates[2].candidate_id == "temporary"
    assert {candidate.candidate_id for candidate in (result.candidates[1], result.candidates[3])} == {"first", "second"}
    assert result.candidates[4].candidate_id == "third"
    assert provider.call_counts[next(operation for operation in provider.call_counts if operation.value == "rerank")] == 1


def test_rerank_invalid_response_falls_back_to_exact_fused_order() -> None:
    from codex_memory.retrieval.provider import ProviderCall, ProviderCallStatus, ProviderOperation, RerankResponse, RerankItem

    class WrongOrder(FakeProvider):
        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            call = ProviderCall(ProviderOperation.RERANK, self._capabilities.reranker.component, ProviderCallStatus.OK, self._capabilities.fingerprint, self._capabilities.reranker.identity.fingerprint, 0, None)
            return RerankResponse(call, tuple(RerankItem(item.item_id, item.position, 0.1, False) for item in reversed(documents)))

    candidates = (_candidate("one"), _candidate("two"))
    documents = {candidate.candidate_id: _input(candidate.candidate_id, index) for index, candidate in enumerate(candidates)}
    result = rerank_candidates(WrongOrder(), _input("query", 0), candidates, documents, top_n=2, deadline_ms=500)

    assert result.status.value == "invalid_response"
    assert result.candidates == candidates


def test_rerank_truncation_and_fingerprint_failures_keep_fused_order() -> None:
    from codex_memory.retrieval.provider import ProviderCall, ProviderCallStatus, ProviderOperation, RerankResponse, RerankItem

    class Truncated(FakeProvider):
        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            call = ProviderCall(ProviderOperation.RERANK, self._capabilities.reranker.component, ProviderCallStatus.OK, self._capabilities.fingerprint, self._capabilities.reranker.identity.fingerprint, 0, None)
            return RerankResponse(call, tuple(RerankItem(item.item_id, item.position, 0.1, True) for item in documents))

    candidates = (_candidate("one"), _candidate("two"))
    documents = {candidate.candidate_id: _input(candidate.candidate_id, index) for index, candidate in enumerate(candidates)}
    result = rerank_candidates(Truncated(), _input("query", 0), candidates, documents, top_n=2, deadline_ms=500)

    assert result.status.value == "input_too_long"
    assert result.candidates == candidates


def test_rerank_rejects_missing_document_without_provider_call() -> None:
    provider = FakeProvider()
    candidates = (_candidate("one"),)

    result = rerank_candidates(provider, _input("query", 0), candidates, {}, top_n=1, deadline_ms=500)

    assert result.status.value == "invalid_response"
    assert sum(provider.call_counts.values()) == 0


def test_rerank_fingerprint_mismatch_falls_back_to_fused_order() -> None:
    provider = FakeProvider()
    candidates = (_candidate("one"), _candidate("two"))
    documents = {candidate.candidate_id: _input(candidate.candidate_id, index) for index, candidate in enumerate(candidates)}

    result = rerank_candidates(
        provider, _input("query", 0), candidates, documents, top_n=2, deadline_ms=500,
        expected_provider_fingerprint="0" * 64,
        expected_model_fingerprint="1" * 64,
    )

    assert result.status.value == "fingerprint_mismatch"
    assert result.candidates == candidates
