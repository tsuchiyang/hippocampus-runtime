from __future__ import annotations

from codex_memory.retrieval.contracts import CandidateChannel, CandidateContribution, RetrievedCandidate
from codex_memory.retrieval.provider import ProviderInput
from codex_memory.retrieval.rerank import rerank_candidates
from retrieval_provider_fakes import FakeProvider


def _candidate(identifier: str) -> RetrievedCandidate:
    return RetrievedCandidate("record", identifier, "decision", "project", "project", "active", 1, "c" * 64, False, 1, 1, (CandidateContribution(CandidateChannel.EXACT, 1, 1, 1),))


def test_exhausted_deadline_never_invokes_provider() -> None:
    provider = FakeProvider()
    candidates = (_candidate("one"),)
    documents = {"one": ProviderInput.from_text("one", 0, "one")}

    result = rerank_candidates(provider, ProviderInput.from_text("query", 0, "query"), candidates, documents, top_n=1, deadline_ms=0)

    assert result.status.value == "deadline_exhausted"
    assert result.candidates == candidates
    assert sum(provider.call_counts.values()) == 0


def test_late_provider_response_is_not_used(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from codex_memory.retrieval import rerank as rerank_module

    provider = FakeProvider()
    candidates = (_candidate("one"), _candidate("two"))
    documents = {candidate.candidate_id: ProviderInput.from_text(candidate.candidate_id, index, candidate.candidate_id) for index, candidate in enumerate(candidates)}
    ticks = iter((0, 0, 5_000_000))
    monkeypatch.setattr(rerank_module.time, "monotonic_ns", lambda: next(ticks))

    result = rerank_candidates(provider, ProviderInput.from_text("query", 0, "query"), candidates, documents, top_n=2, deadline_ms=5)

    assert result.status.value == "timeout"
    assert result.candidates == candidates


def test_stalled_provider_returns_at_deadline_without_using_its_late_result() -> None:
    import time

    class Stalled(FakeProvider):
        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            time.sleep(0.2)
            return super().rerank(query, documents, deadline_ms=deadline_ms)

    candidates = (_candidate("one"),)
    documents = {"one": ProviderInput.from_text("one", 0, "one")}
    started = time.monotonic()
    result = rerank_candidates(Stalled(), ProviderInput.from_text("query", 0, "query"), candidates, documents, top_n=1, deadline_ms=20)

    assert time.monotonic() - started < 0.15
    assert result.status.value == "timeout"
    assert result.candidates == candidates
