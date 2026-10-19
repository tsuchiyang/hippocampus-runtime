"""Exact, deterministic reciprocal-rank fusion for offline retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .contracts import CandidateChannel, CandidateContribution, CandidateTrace, RetrievedCandidate, RetrievalPolicy


@dataclass(frozen=True, slots=True)
class FusionInput:
    channel: CandidateChannel
    candidates: Sequence[Mapping[str, object]]
    raw_scores: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class FusionResult:
    candidates: tuple[RetrievedCandidate, ...]
    trace: tuple[CandidateTrace, ...]


_WEIGHTS = {
    CandidateChannel.EXACT: "exact_weight", CandidateChannel.SYMBOL: "symbol_weight",
    CandidateChannel.BM25: "bm25_weight", CandidateChannel.RECENT: "recent_weight",
    CandidateChannel.STALE_DELTA: "stale_delta_weight",
    CandidateChannel.DENSE: "dense_weight", CandidateChannel.SPARSE: "sparse_weight",
}
_LIMITS = {
    CandidateChannel.DENSE: "dense_top_k",
    CandidateChannel.SPARSE: "sparse_top_k",
}
_BUCKETS = {"constitution": 0, "decision": 3, "failure": 4, "rejected_approach": 5, "assumption": 6}


def fuse_candidates(inputs: Sequence[FusionInput], policy: RetrievalPolicy) -> FusionResult:
    """Fuse pre-ranked ordinary candidates without floats or content-hash collapsing."""
    rows: dict[tuple[str, str], Mapping[str, object]] = {}
    contributions: dict[tuple[str, str], list[CandidateContribution]] = {}
    channels: dict[tuple[str, str], list[CandidateChannel]] = {}
    raw_scores: dict[tuple[str, str], str] = {}
    for source in inputs:
        weight_name = _WEIGHTS.get(source.channel)
        if weight_name is None:
            continue
        weight = getattr(policy, weight_name)
        limit = getattr(policy, _LIMITS.get(source.channel, "per_channel_limit"))
        for rank, row in enumerate(source.candidates[:limit], 1):
            kind, identifier = str(row["candidate_kind"]), str(row["candidate_id"])
            key = kind, identifier
            rows.setdefault(key, row)
            contribution = CandidateContribution(source.channel, rank, weight, policy.rrf_k + rank)
            contributions.setdefault(key, []).append(contribution)
            channels.setdefault(key, []).append(source.channel)
            if source.raw_scores is not None and source.channel in {CandidateChannel.DENSE, CandidateChannel.SPARSE}:
                raw = source.raw_scores.get(identifier)
                if raw is not None:
                    # A candidate that appears on both semantic channels keeps
                    # the first channel's diagnostic score deterministically.
                    raw_scores.setdefault(key, raw)
    ranked: list[tuple[Fraction, Mapping[str, object], tuple[CandidateContribution, ...], tuple[CandidateChannel, ...]]] = []
    for key, row in rows.items():
        parts = tuple(contributions[key])
        score = sum((Fraction(part.numerator, part.denominator) for part in parts), Fraction())
        ranked.append((score, row, parts, tuple(channels[key])))
    ranked.sort(key=lambda item: (
        -item[0], _BUCKETS.get(str(item[1]["record_type"]), 99),
        -int(item[1].get("last_lifecycle_event_sequence", item[1]["sequence"])),
        -int(item[1]["sequence"]), str(item[1]["candidate_id"]),
    ))
    selected: list[RetrievedCandidate] = []
    trace: list[CandidateTrace] = []
    for rank, (score, row, parts, source_channels) in enumerate(ranked, 1):
        is_selected = rank <= policy.fused_candidate_limit
        identifier = str(row["candidate_id"])
        if is_selected:
            selected.append(RetrievedCandidate(
                str(row["candidate_kind"]), identifier, str(row["record_type"]), str(row["scope_kind"]),
                str(row["scope_key"]), str(row["state"]), int(row["sequence"]), str(row["content_sha256"]),
                False, score.numerator, score.denominator, parts,
            ))
        trace.append(CandidateTrace(identifier, source_channels, parts, rank, is_selected,
                                    ("fused",) if is_selected else ("candidate_limit",), raw_scores.get((str(row["candidate_kind"]), identifier))))
    return FusionResult(tuple(selected), tuple(trace))
