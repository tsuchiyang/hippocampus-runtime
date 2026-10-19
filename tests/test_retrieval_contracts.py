from dataclasses import FrozenInstanceError
import json
import sqlite3
from pathlib import Path

import pytest

import codex_memory.ledger_common as ledger_common
from codex_memory.ledger_common import canonical_json
from codex_memory.retrieval.contracts import (
    CandidateChannel,
    CandidateContribution,
    CandidateTrace,
    DegradedReason,
    EpistemicContext,
    IndexGeneration,
    IndexHighwater,
    RetrievalMode,
    RetrievalPolicy,
    RetrievalRequest,
    RetrievalResult,
    RetrievalQueryResult,
    RetrievalScope,
    RetrievalStatus,
    RetrievedCandidate,
    canonical_contract_sha256,
)
from codex_memory.retrieval.errors import RetrievalValidationError


def test_enum_values_are_frozen() -> None:
    assert [x.value for x in RetrievalStatus] == ["current", "stale", "absent", "corrupt", "unsupported"]
    assert [x.value for x in RetrievalMode] == ["off", "shadow", "active"]
    assert [x.value for x in DegradedReason] == [
        "index_absent", "index_stale", "index_corrupt", "path_unavailable",
        "path_unsafe", "schema_unsupported", "canonical_unavailable", "canonical_invalid",
        "fts5_unavailable", "stale_delta_used", "stale_delta_exceeded",
        "semantic_index_unavailable", "semantic_index_stale", "semantic_dense_unavailable", "semantic_sparse_unavailable",
    ]


def test_defaults_and_immutability() -> None:
    context = EpistemicContext()
    assert context.memory_view_id == "ground_truth.default"
    assert context.truth_domain == "ground_truth"
    assert RetrievalRequest(query=None, session_id="s", turn_id="t", context_id="c",
                            prompt_event_uri=None, scope=RetrievalScope("p")).max_chars == 4000
    with pytest.raises(FrozenInstanceError):
        context.actor_id = "other"  # type: ignore[misc]


def test_every_contract_round_trips_strictly_and_is_json_safe() -> None:
    highwater = IndexHighwater(record_seq=1, memory_event_seq=2, evidence_seq=3,
                               relation_seq=4, candidate_seq=5, candidate_event_seq=6,
                               candidate_evidence_seq=7)
    scope = RetrievalScope("C:/Repo", ("workspace-a", "workspace-b"))
    generation = IndexGeneration("g1", "ready", highwater, highwater, "a" * 64,
                                 "unconfigured:p2.1", "unconfigured:p2.1", "valid",
                                 "2026-09-04T00:00:00Z", "2026-09-04T00:00:01Z")
    request = RetrievalRequest("hello", "s", "t", "c", "codex-event://s/1", scope)
    result = RetrievalResult(RetrievalStatus.CURRENT, RetrievalMode.OFF, generation,
                             highwater, highwater, None, ("r1", "r2"))
    for value in (scope, highwater, generation, request, result):
        payload = value.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert type(value).from_dict(payload) == value


def test_query_contracts_have_frozen_values_and_field_order() -> None:
    assert [channel.value for channel in CandidateChannel] == [
        "mandatory_constitution",
        "temporary",
        "exact",
        "bm25",
        "symbol",
        "recent",
        "stale_delta",
        "dense",
        "sparse",
    ]
    assert RetrievalPolicy() == RetrievalPolicy(
        rrf_k=60,
        exact_weight=8,
        symbol_weight=6,
        bm25_weight=4,
        recent_weight=1,
        stale_delta_weight=8,
        dense_weight=4,
        sparse_weight=4,
        dense_top_k=64,
        sparse_top_k=64,
        rerank_top_n=40,
        per_channel_limit=32,
        fused_candidate_limit=64,
        stale_delta_limit=128,
    )
    assert tuple(name for name in RetrievalPolicy.__dataclass_fields__ if not name.startswith("_")) == (
        "rrf_k",
        "exact_weight",
        "symbol_weight",
        "bm25_weight",
        "recent_weight",
        "stale_delta_weight",
        "dense_weight",
        "sparse_weight",
        "dense_top_k",
        "sparse_top_k",
        "rerank_top_n",
        "per_channel_limit",
        "fused_candidate_limit",
        "stale_delta_limit",
    )


def test_query_contract_round_trip_is_strict_and_tuple_normalized() -> None:
    contribution = CandidateContribution(CandidateChannel.EXACT, 1, 8, 61)
    candidate = RetrievedCandidate(
        candidate_kind="record",
        candidate_id="record-1",
        record_type="decision",
        scope_kind="project",
        scope_key="C:/Repo",
        state="active",
        sequence=9,
        content_sha256="a" * 64,
        mandatory=True,
        fused_score_numerator=8,
        fused_score_denominator=61,
        contributions=(contribution,),
    )
    trace = CandidateTrace(
        candidate_id="record-1",
        source_channels=(CandidateChannel.EXACT, CandidateChannel.RECENT),
        contributions=(contribution,),
        fused_rank=1,
        selected=True,
        reason_codes=("exact_match",),
    )
    result = RetrievalQueryResult(
        status=RetrievalStatus.CURRENT,
        mode=RetrievalMode.SHADOW,
        generation=None,
        canonical_highwater=IndexHighwater(),
        indexed_highwater=IndexHighwater(),
        index_stale=False,
        degraded_reasons=(DegradedReason.INDEX_ABSENT,),
        candidates=(candidate,),
        trace=(trace,),
    )

    payload = result.to_dict()

    assert list(payload) == [
        "status",
        "mode",
        "generation",
        "canonical_highwater",
        "indexed_highwater",
        "index_stale",
        "degraded_reasons",
        "candidates",
        "trace",
    ]
    assert json.loads(json.dumps(payload)) == payload
    assert RetrievalQueryResult.from_dict(payload) == result


def test_semantic_raw_score_is_optional_trace_metadata_and_is_strictly_hex() -> None:
    """Would fail if model scores leaked into P2.2 output or accepted NaN/Inf."""
    base = CandidateTrace("record-1", (CandidateChannel.DENSE,), (), 1, True, ("fused",))
    assert "raw_score_hex" not in base.to_dict()
    semantic = CandidateTrace("record-1", (CandidateChannel.DENSE,), (), 1, True, ("fused",), "0x1.0000000000000p+0")
    assert CandidateTrace.from_dict(semantic.to_dict()) == semantic
    with pytest.raises(RetrievalValidationError):
        CandidateTrace("record-1", (CandidateChannel.DENSE,), (), 1, True, ("fused",), "nan")
    with pytest.raises(RetrievalValidationError):
        CandidateTrace("record-1", (CandidateChannel.DENSE,), (), 1, True, ("fused",), "1.0")
    from codex_memory.retrieval import index_store
    with pytest.raises(ValueError, match="score"):
        index_store.append_candidate_trace(
            sqlite3.connect(":memory:"), trace_id="trace", run_id="run",
            candidate_kind="record", candidate_id="record", channel="dense",
            channel_rank=1, contribution_numerator=1, contribution_denominator=1,
            fused_rank=1, selected=True, reason_codes_json="[]", trace_sha256="a" * 64,
            raw_score_hex="1.0",
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "current", "mode": "shadow", "generation": None, "canonical_highwater": None, "indexed_highwater": None, "index_stale": False, "degraded_reasons": [], "candidates": [], "trace": [], "extra": 1},
        {"status": "current", "mode": "shadow", "generation": None, "canonical_highwater": None, "indexed_highwater": None, "index_stale": False, "degraded_reasons": ["index_absent"], "candidates": [], "trace": []},
    ],
)
def test_query_contract_unknown_fields_and_bounds_are_rejected(bad) -> None:
    if "extra" in bad:
        with pytest.raises(RetrievalValidationError):
            RetrievalQueryResult.from_dict(bad)
        return

    with pytest.raises(RetrievalValidationError):
        RetrievalPolicy(rrf_k=True)
    with pytest.raises(RetrievalValidationError):
        CandidateContribution(CandidateChannel.EXACT, 0, 1, 1)
    with pytest.raises(RetrievalValidationError):
        RetrievalQueryResult.from_dict({**bad, "degraded_reasons": ["bogus"]})


def test_tuple_list_normalization_at_json_boundary() -> None:
    scope = RetrievalScope.from_dict({"project_scope_key": "p", "workspace_scope_keys": ["a", "a", "b"]})
    assert scope.workspace_scope_keys == ("a", "b")
    result = RetrievalResult.from_dict({"status": "absent", "mode": "off", "generation": None,
                                        "canonical_highwater": None, "indexed_highwater": None,
                                        "degraded_reason": "index_absent", "record_ids": ["r"]})
    assert result.record_ids == ("r",)


def test_workspace_keys_use_host_aware_casefold_deduplication(monkeypatch) -> None:
    monkeypatch.setattr(ledger_common.os, "name", "nt")
    scope = RetrievalScope(
        r"C:\Project With Spaces",
        (r"C:\Workspace One", r"c:\workspace one", r"C:\Workspace Two"),
    )
    assert scope.workspace_scope_keys == (r"C:\Workspace One", r"C:\Workspace Two")
    monkeypatch.setattr(ledger_common.os, "name", "posix")
    scope = RetrievalScope(
        "/srv/Project With Spaces",
        ("/srv/Workspace One", "/srv/workspace one", "/srv/Workspace Two"),
    )
    assert scope.workspace_scope_keys == (
        "/srv/Workspace One",
        "/srv/workspace one",
        "/srv/Workspace Two",
    )


def test_scope_keys_reuse_phase1b_bounded_nonblank_semantics() -> None:
    boundary = "é" * 128
    assert RetrievalScope(boundary, (boundary,)).to_dict() == {
        "project_scope_key": boundary,
        "workspace_scope_keys": [boundary],
    }

    for invalid in ("", " \t\r\n", "é" * 129):
        with pytest.raises(RetrievalValidationError):
            RetrievalScope(invalid)
        with pytest.raises(RetrievalValidationError):
            RetrievalScope("project", (invalid,))

    with pytest.raises(RetrievalValidationError):
        RetrievalScope("project", tuple(f"workspace-{index}" for index in range(33)))


def test_highwater_canonical_json_serialization_is_repeatably_ordered() -> None:
    value = IndexHighwater(1, 2, 3, 4, 5, 6, 7)
    assert list(value.to_dict()) == [
        "record_seq", "memory_event_seq", "evidence_seq", "relation_seq",
        "candidate_seq", "candidate_event_seq", "candidate_evidence_seq",
    ]
    serialized = {canonical_json(value.to_dict()) for _ in range(100)}
    assert len(serialized) == 1
    assert next(iter(serialized)) == (
        '{"candidate_event_seq":6,"candidate_evidence_seq":7,"candidate_seq":5,'
        '"evidence_seq":3,"memory_event_seq":2,"record_seq":1,"relation_seq":4}'
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("record_seq"),
        lambda value: value.__setitem__("extra", 0),
        lambda value: value.__setitem__("record_seq", -1),
        lambda value: value.__setitem__("record_seq", True),
    ],
)
def test_highwater_rejects_missing_extra_negative_and_boolean_fields(mutate) -> None:
    value = IndexHighwater().to_dict()
    mutate(value)
    with pytest.raises(RetrievalValidationError):
        IndexHighwater.from_dict(value)


def test_contract_metadata_hash_is_canonical_and_rejects_non_json_values() -> None:
    assert canonical_contract_sha256({"b": [2], "a": 1}) == canonical_contract_sha256({"a": 1, "b": [2]})
    with pytest.raises(RetrievalValidationError):
        canonical_contract_sha256({"not_json": object()})


@pytest.mark.parametrize("bad", [
    {}, {"project_scope_key": "p", "extra": 1},
    {"project_scope_key": "p", "workspace_scope_keys": [1]},
])
def test_unknown_missing_and_wrong_types_rejected(bad) -> None:
    with pytest.raises(RetrievalValidationError):
        RetrievalScope.from_dict(bad)


def test_boolean_is_not_an_integer_and_sizes_are_bounded() -> None:
    with pytest.raises(RetrievalValidationError):
        IndexHighwater.from_dict({k: True for k in IndexHighwater.__dataclass_fields__})
    base = RetrievalRequest(query=None, session_id="s", turn_id="t", context_id="c",
                            prompt_event_uri=None, scope=RetrievalScope("p")).to_dict()
    base["max_chars"] = 0
    with pytest.raises(RetrievalValidationError): RetrievalRequest.from_dict(base)
    with pytest.raises(RetrievalValidationError): RetrievalPolicy(dense_top_k=257)
    with pytest.raises(RetrievalValidationError): RetrievalPolicy(sparse_top_k=257)
    with pytest.raises(RetrievalValidationError): RetrievalPolicy(rerank_top_n=65)
    base["max_chars"] = 4001
    with pytest.raises(RetrievalValidationError): RetrievalRequest.from_dict(base)


def test_malformed_hash_ids_and_timestamps_rejected() -> None:
    with pytest.raises(RetrievalValidationError):
        IndexGeneration("", "ready", IndexHighwater(), IndexHighwater(), "x", "p", "m", "valid",
                        "not-a-timestamp", "not-a-timestamp")
    valid = IndexGeneration("g1", "ready", IndexHighwater(), IndexHighwater(), "a" * 64,
                            "p", "m", "valid", "2026-09-04T00:00:00Z", "2026-09-04T00:00:01Z").to_dict()
    valid["canonical_content_fingerprint"] = "z" * 64
    with pytest.raises(RetrievalValidationError): IndexGeneration.from_dict(valid)


@pytest.mark.parametrize("field,value", [
    ("memory_view_id", "synthetic.default"), ("memory_view_id", "counterfactual.default"),
    ("memory_view_id", "unknown"), ("memory_view_id", "GROUND_TRUTH.DEFAULT"),
    ("truth_domain", "synthetic"), ("truth_domain", "GROUND_TRUTH"),
    ("epistemic_policy_id", "other"),
])
def test_non_ground_truth_context_rejected_without_filesystem_access(monkeypatch, field, value) -> None:
    called = False
    def fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("filesystem accessed")
    monkeypatch.setattr(Path, "exists", fail)
    context = EpistemicContext().__dict__ if hasattr(EpistemicContext(), "__dict__") else {
        "actor_id": "codex", "role": "operator", "memory_view_id": "ground_truth.default",
        "epistemic_policy_id": "default", "truth_domain": "ground_truth",
    }
    context[field] = value
    with pytest.raises(RetrievalValidationError): EpistemicContext.from_dict(context)
    assert called is False
