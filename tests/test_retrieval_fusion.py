from __future__ import annotations

from fractions import Fraction

from codex_memory.retrieval.contracts import CandidateChannel, RetrievalPolicy
from codex_memory.retrieval.fusion import FusionInput, fuse_candidates


def _row(identifier: str, *, record_type: str = "decision", sequence: int = 1) -> dict[str, object]:
    return {
        "candidate_kind": "record", "candidate_id": identifier,
        "record_type": record_type, "scope_kind": "project", "scope_key": "p",
        "state": "active", "sequence": sequence, "last_lifecycle_event_sequence": sequence,
        "content_sha256": "a" * 64,
    }


def test_weighted_rrf_is_exact_deduplicated_and_stably_tied() -> None:
    """Would fail if scores used floats, duplicate IDs were not combined, or ties drifted."""
    result = fuse_candidates(
        (
            FusionInput(CandidateChannel.EXACT, (_row("b", sequence=3), _row("a", sequence=3))),
            FusionInput(CandidateChannel.BM25, (_row("a", sequence=3),)),
        ),
        RetrievalPolicy(rrf_k=2, exact_weight=8, bm25_weight=4),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["a", "b"]
    a, b = result.candidates
    assert Fraction(a.fused_score_numerator, a.fused_score_denominator) == Fraction(8, 4) + Fraction(4, 3)
    assert Fraction(b.fused_score_numerator, b.fused_score_denominator) == Fraction(8, 3)
    assert [(part.channel, part.channel_rank, part.numerator, part.denominator) for part in a.contributions] == [
        (CandidateChannel.EXACT, 2, 8, 4), (CandidateChannel.BM25, 1, 4, 3),
    ]


def test_fusion_limit_marks_unselected_trace_without_dropping_explain() -> None:
    """Would fail if the candidate cap silently made explain disagree with selection."""
    result = fuse_candidates(
        (FusionInput(CandidateChannel.RECENT, (_row("a"), _row("b"))),),
        RetrievalPolicy(fused_candidate_limit=1),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["a"]
    assert [(trace.candidate_id, trace.fused_rank, trace.selected) for trace in result.trace] == [
        ("a", 1, True), ("b", 2, False),
    ]


def test_dense_and_sparse_contributions_are_ranked_by_rrf_not_raw_model_score() -> None:
    """Would fail if semantic raw scores replaced rank-based deterministic fusion."""
    result = fuse_candidates(
        (
            FusionInput(
                CandidateChannel.DENSE,
                (_row("dense-first"), _row("sparse-first")),
                {"dense-first": "0x1.0000000000000p+0", "sparse-first": "0x1.0000000000000p-1"},
            ),
            FusionInput(
                CandidateChannel.SPARSE,
                (_row("sparse-first"),),
                {"sparse-first": "0x1.0000000000000p+9"},
            ),
        ),
        RetrievalPolicy(rrf_k=2, dense_weight=4, sparse_weight=4),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["sparse-first", "dense-first"]
    assert result.trace[0].raw_score_hex == "0x1.0000000000000p-1"
    assert result.trace[0].source_channels == (CandidateChannel.DENSE, CandidateChannel.SPARSE)


def test_semantic_channels_use_their_own_bounded_top_k() -> None:
    """Would fail if a P2.2 per-channel cap silently truncated semantic candidates."""
    result = fuse_candidates(
        (FusionInput(CandidateChannel.DENSE, (_row("a"), _row("b"))),),
        RetrievalPolicy(per_channel_limit=1, dense_top_k=2, fused_candidate_limit=2),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["a", "b"]
