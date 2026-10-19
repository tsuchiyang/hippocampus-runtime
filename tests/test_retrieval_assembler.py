from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import import_module

import pytest

from codex_memory.retrieval.canonical_snapshot import CanonicalSnapshot
from codex_memory.retrieval.contracts import (
    CandidateChannel,
    CandidateTrace,
    DegradedReason,
    EpistemicContext,
    IndexHighwater,
    RetrievalMode,
    RetrievalQueryResult,
    RetrievalRequest,
    RetrievalScope,
    RetrievalStatus,
)
from codex_memory.retrieval.evidence_pack import (
    EvidencePack,
    EvidencePackBuildResult,
    LedgerEvent,
    MemoryCard,
    PackOmission,
    SafeEvidence,
)


SCOPE = r"C:\Work\Memory Runtime"


def _assembler():
    try:
        return import_module("codex_memory.retrieval.assembler")
    except ModuleNotFoundError:
        pytest.fail("P2.4 bounded assembler module is missing", pytrace=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _highwater(**changes: int) -> IndexHighwater:
    values = {
        "record_seq": 3,
        "memory_event_seq": 6,
        "evidence_seq": 2,
        "relation_seq": 0,
        "candidate_seq": 1,
        "candidate_event_seq": 2,
        "candidate_evidence_seq": 1,
    }
    values.update(changes)
    return IndexHighwater(**values)


def _snapshot(highwater: IndexHighwater | None = None) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        schema_version=3,
        highwater=highwater or _highwater(),
        content_fingerprint=_hash("canonical"),
        source_timestamp="2026-09-13T00:00:00.000000Z",
    )


def _request(
    *,
    session_id: str = "session_a",
    context_id: str = "startup",
    max_chars: int = 4000,
    prompt_event_uri: str | None = None,
    scope: RetrievalScope | None = None,
    epistemic_context: EpistemicContext | None = None,
) -> RetrievalRequest:
    return RetrievalRequest(
        query="memory query",
        session_id=session_id,
        turn_id="turn_2",
        context_id=context_id,
        prompt_event_uri=prompt_event_uri,
        scope=scope or RetrievalScope(SCOPE),
        epistemic_context=epistemic_context or EpistemicContext(),
        mode=RetrievalMode.OFF,
        max_chars=max_chars,
    )


def _pack(
    identifier: str,
    *,
    candidate_kind: str = "record",
    record_type: str = "decision",
    topic: str = "runtime.storage",
    content: str | None = None,
    certified: bool = False,
    sequence: int = 1,
    last_event_sequence: int = 2,
) -> EvidencePack:
    content_sha256 = _hash(content or identifier)
    action = "capture" if candidate_kind == "temporary" else "activate"
    return EvidencePack(
        card=MemoryCard(
            identifier=identifier,
            candidate_kind=candidate_kind,
            record_type=record_type,
            scope_kind="project",
            scope_key=SCOPE,
            topic_key=topic,
            statement=f"Statement for {identifier}.",
            value=None,
            state="temporary" if candidate_kind == "temporary" else "active",
            sequence=sequence,
            content_sha256=content_sha256,
            trust_label=(
                "CERTIFIED CONSTITUTION"
                if certified
                else "UNVERIFIED TEMPORARY"
                if candidate_kind == "temporary"
                else "RETRIEVED MEMORY"
            ),
            certified=certified,
            last_event=LedgerEvent(
                event_id=f"event_{identifier}",
                event_sequence=last_event_sequence,
                action=action,
                event_sha256=_hash(f"event:{identifier}"),
                created_at="2026-09-13T00:00:00.000000Z",
            ),
        ),
        evidence=(),
    )


def _result(
    snapshot: CanonicalSnapshot,
    *,
    degraded=(),
    trace: tuple[CandidateTrace, ...] = (),
) -> RetrievalQueryResult:
    return RetrievalQueryResult(
        status=RetrievalStatus.CURRENT,
        mode=RetrievalMode.OFF,
        generation=None,
        canonical_highwater=snapshot.highwater,
        indexed_highwater=snapshot.highwater,
        index_stale=False,
        degraded_reasons=tuple(degraded),
        candidates=(),
        trace=trace,
    )


def _build(*packs: EvidencePack) -> EvidencePackBuildResult:
    return EvidencePackBuildResult(packs, (), 0, ())


def test_selection_orders_trust_partitions_and_deduplicates_content_hashes() -> None:
    module = _assembler()
    ordinary = _pack("mem_ordinary", sequence=30)
    temporary = _pack("cand_temp", candidate_kind="temporary", sequence=20)
    certified = _pack(
        "mem_constitution",
        record_type="constitution",
        topic="safety.append-only",
        certified=True,
        sequence=10,
    )
    duplicate = _pack(
        "mem_duplicate",
        content="mem_ordinary",
        topic="runtime.duplicate",
        sequence=40,
    )
    snapshot = _snapshot()

    selection = module.select_evidence_packs(
        request=_request(),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(ordinary, temporary, certified, duplicate),
    )

    assert [pack.card.identifier for pack in selection.packs] == [
        "mem_constitution",
        "cand_temp",
        "mem_ordinary",
    ]
    assert selection.duplicate_ids == ("mem_duplicate",)
    assert selection.suppressed_ids == ()


def test_prior_committed_selection_suppresses_only_unchanged_same_context_permanent() -> None:
    module = _assembler()
    snapshot = _snapshot()
    permanent = _pack("mem_decision", last_event_sequence=6)
    temporary = _pack("cand_temp", candidate_kind="temporary")
    prior = module.PriorContextSelection(
        manifest_id="asm_previous",
        committed=True,
        session_id="session_a",
        context_id="startup",
        scope=RetrievalScope(SCOPE),
        epistemic_context=EpistemicContext(),
        canonical_highwater=snapshot.highwater,
        canonical_content_fingerprint=snapshot.content_fingerprint,
        index_stale=False,
        degraded_reasons=(),
        records=(
            module.SelectionIdentity(
                "mem_decision", permanent.card.content_sha256, 6
            ),
            module.SelectionIdentity(
                "cand_temp", temporary.card.content_sha256, 2
            ),
        ),
    )

    unchanged = module.select_evidence_packs(
        request=_request(),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(permanent, temporary),
        prior=prior,
    )
    changed_context = module.select_evidence_packs(
        request=_request(context_id="checkpoint_8"),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(permanent, temporary),
        prior=prior,
    )
    changed_lifecycle = module.select_evidence_packs(
        request=_request(),
        result=_result(_snapshot(_highwater(memory_event_seq=7))),
        snapshot=_snapshot(_highwater(memory_event_seq=7)),
        build=_build(permanent),
        prior=prior,
    )
    changed_fingerprint_snapshot = replace(
        snapshot, content_fingerprint=_hash("replacement-canonical")
    )
    changed_fingerprint = module.select_evidence_packs(
        request=_request(),
        result=_result(changed_fingerprint_snapshot),
        snapshot=changed_fingerprint_snapshot,
        build=_build(permanent),
        prior=prior,
    )
    newly_degraded = module.select_evidence_packs(
        request=_request(),
        result=_result(snapshot, degraded=(DegradedReason.INDEX_STALE,)),
        snapshot=snapshot,
        build=_build(permanent),
        prior=prior,
    )

    assert [pack.card.identifier for pack in unchanged.packs] == ["cand_temp"]
    assert unchanged.suppressed_ids == ("mem_decision",)
    assert [pack.card.identifier for pack in changed_context.packs] == [
        "cand_temp",
        "mem_decision",
    ]
    assert changed_context.suppressed_ids == ()
    assert [pack.card.identifier for pack in changed_lifecycle.packs] == [
        "mem_decision"
    ]
    assert changed_lifecycle.suppressed_ids == ()
    assert [pack.card.identifier for pack in changed_fingerprint.packs] == [
        "mem_decision"
    ]
    assert changed_fingerprint.suppressed_ids == ()
    assert [pack.card.identifier for pack in newly_degraded.packs] == [
        "mem_decision"
    ]
    assert newly_degraded.suppressed_ids == ()


def test_suppression_requires_same_scope_epistemic_context_and_build_health() -> None:
    module = _assembler()
    snapshot = _snapshot()
    permanent = _pack("mem_decision", last_event_sequence=6)
    prior = module.PriorContextSelection(
        manifest_id="asm_previous",
        committed=True,
        session_id="session_a",
        context_id="startup",
        scope=RetrievalScope(SCOPE),
        epistemic_context=EpistemicContext(),
        canonical_highwater=snapshot.highwater,
        canonical_content_fingerprint=snapshot.content_fingerprint,
        index_stale=False,
        degraded_reasons=(),
        records=(
            module.SelectionIdentity(
                "mem_decision", permanent.card.content_sha256, 6
            ),
        ),
    )

    changed_scope = module.select_evidence_packs(
        request=_request(scope=RetrievalScope(SCOPE, ("workspace-other",))),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(permanent),
        prior=prior,
    )
    changed_actor = module.select_evidence_packs(
        request=_request(epistemic_context=EpistemicContext(actor_id="auditor")),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(permanent),
        prior=prior,
    )
    newly_degraded_build = module.select_evidence_packs(
        request=_request(),
        result=_result(snapshot),
        snapshot=snapshot,
        build=EvidencePackBuildResult(
            (permanent,),
            (PackOmission("mem_missing", "evidence_missing"),),
            0,
            ("evidence_missing",),
        ),
        prior=prior,
    )

    assert [pack.card.identifier for pack in changed_scope.packs] == [
        "mem_decision"
    ]
    assert [pack.card.identifier for pack in changed_actor.packs] == [
        "mem_decision"
    ]
    assert [pack.card.identifier for pack in newly_degraded_build.packs] == [
        "mem_decision"
    ]


def test_non_exact_permanent_selection_round_robins_topics_deterministically() -> None:
    module = _assembler()
    snapshot = _snapshot()
    first = _pack("mem_storage_1", topic="storage.engine", sequence=3)
    second = _pack("mem_storage_2", topic="storage.engine", sequence=2)
    distinct = _pack("mem_platform", topic="runtime.platform", sequence=1)
    traces = tuple(
        CandidateTrace(
            pack.card.identifier,
            (CandidateChannel.BM25,),
            (),
            rank,
            True,
            ("selected",),
        )
        for rank, pack in enumerate((first, second, distinct), start=1)
    )

    selection = module.select_evidence_packs(
        request=_request(),
        result=_result(snapshot, trace=traces),
        snapshot=snapshot,
        build=_build(first, second, distinct),
    )

    assert [pack.card.identifier for pack in selection.packs] == [
        "mem_storage_1",
        "mem_platform",
        "mem_storage_2",
    ]


def test_exact_permanent_candidates_keep_retrieval_priority_before_diversity() -> None:
    module = _assembler()
    snapshot = _snapshot()
    exact_first = _pack("mem_exact_1", topic="storage.engine", sequence=3)
    exact_second = _pack("mem_exact_2", topic="storage.engine", sequence=2)
    distinct = _pack("mem_platform", topic="runtime.platform", sequence=1)
    traces = (
        CandidateTrace(
            exact_first.card.identifier,
            (CandidateChannel.EXACT,),
            (),
            1,
            True,
            ("selected",),
        ),
        CandidateTrace(
            exact_second.card.identifier,
            (CandidateChannel.EXACT,),
            (),
            2,
            True,
            ("selected",),
        ),
        CandidateTrace(
            distinct.card.identifier,
            (CandidateChannel.BM25,),
            (),
            3,
            True,
            ("selected",),
        ),
    )

    selection = module.select_evidence_packs(
        request=_request(),
        result=_result(snapshot, trace=traces),
        snapshot=snapshot,
        build=_build(exact_first, exact_second, distinct),
    )

    assert [pack.card.identifier for pack in selection.packs] == [
        "mem_exact_1",
        "mem_exact_2",
        "mem_platform",
    ]


def test_assembly_identity_binds_the_full_epistemic_context() -> None:
    module = _assembler()
    snapshot = _snapshot()
    result = _result(snapshot)
    build = _build(_pack("mem_decision"))
    request = _request()

    codex = module.assemble_evidence_packet(
        request=request,
        result=result,
        snapshot=snapshot,
        build=build,
    )
    auditor = module.assemble_evidence_packet(
        request=replace(
            request,
            epistemic_context=EpistemicContext(actor_id="auditor"),
        ),
        result=result,
        snapshot=snapshot,
        build=build,
    )

    assert codex.packet != auditor.packet
    assert codex.packet_sha256 != auditor.packet_sha256
    assert codex.manifest_id != auditor.manifest_id


def test_duplicate_content_keeps_a_compact_alternative_until_one_pack_fits() -> None:
    module = _assembler()
    snapshot = _snapshot()
    large = _pack("mem_large", content="shared-content", sequence=2)
    large = replace(
        large,
        evidence=tuple(
            SafeEvidence(
                reference=f"https:sha256:{_hash(f'evidence:{index}')}",
                reference_sha256=_hash(f"reference:{index}"),
            )
            for index in range(20)
        ),
    )
    compact = _pack("mem_compact", content="shared-content", sequence=1)

    assembled = module.assemble_evidence_packet(
        request=_request(),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(large, compact),
    )

    assert [item.candidate_id for item in assembled.selected] == ["mem_compact"]
    assert assembled.duplicate_ids == ("mem_large",)
    assert assembled.budget_omitted_ids == ()
    assert "mem_compact" in assembled.packet
    assert "mem_large" not in assembled.packet


def test_permanent_duplicate_fallback_waits_for_remaining_temporary_packs() -> None:
    module = _assembler()
    snapshot = _snapshot()
    oversized_temporary = _pack(
        "cand_oversized",
        candidate_kind="temporary",
        content="shared-content",
        sequence=1,
    )
    oversized_temporary = replace(
        oversized_temporary,
        evidence=tuple(
            SafeEvidence(
                reference=f"https:sha256:{_hash(f'evidence:{index}')}",
                reference_sha256=_hash(f"reference:{index}"),
            )
            for index in range(20)
        ),
    )
    compact_permanent = _pack(
        "mem_compact",
        content="shared-content",
        sequence=3,
    )
    later_temporary = _pack(
        "cand_later",
        candidate_kind="temporary",
        content="different-content",
        sequence=2,
    )
    solo = module.assemble_evidence_packet(
        request=_request(),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(later_temporary),
    )

    assembled = module.assemble_evidence_packet(
        request=_request(max_chars=solo.packet_chars),
        result=_result(snapshot),
        snapshot=snapshot,
        build=_build(oversized_temporary, compact_permanent, later_temporary),
    )

    assert [item.candidate_id for item in assembled.selected] == ["cand_later"]
    assert assembled.duplicate_ids == ("mem_compact",)
    assert assembled.budget_omitted_ids == ("cand_oversized",)


def test_permanent_duplicate_fallback_preserves_exact_candidate_priority() -> None:
    module = _assembler()
    snapshot = _snapshot()
    oversized_temporary = replace(
        _pack(
            "cand_oversized",
            candidate_kind="temporary",
            content="shared-content",
            sequence=1,
        ),
        evidence=tuple(
            SafeEvidence(
                reference=f"https:sha256:{_hash(f'evidence:{index}')}",
                reference_sha256=_hash(f"reference:{index}"),
            )
            for index in range(20)
        ),
    )
    fallback = _pack("mem_plain", content="shared-content", sequence=3)
    exact = _pack("mem_exact", content="exact-content", sequence=2)
    result = _result(
        snapshot,
        trace=(
            CandidateTrace(
                exact.card.identifier,
                (CandidateChannel.EXACT,),
                (),
                1,
                True,
                ("selected",),
            ),
            CandidateTrace(
                fallback.card.identifier,
                (CandidateChannel.BM25,),
                (),
                2,
                True,
                ("selected",),
            ),
        ),
    )
    exact_only = module.assemble_evidence_packet(
        request=_request(),
        result=result,
        snapshot=snapshot,
        build=_build(exact),
    )

    assembled = module.assemble_evidence_packet(
        request=_request(max_chars=exact_only.packet_chars),
        result=result,
        snapshot=snapshot,
        build=_build(oversized_temporary, exact, fallback),
    )

    assert [item.candidate_id for item in assembled.selected] == ["mem_exact"]
    assert assembled.duplicate_ids == ("mem_plain",)
    assert assembled.budget_omitted_ids == ("cand_oversized",)


def test_prior_suppression_applies_to_a_permanent_duplicate_fallback() -> None:
    module = _assembler()
    snapshot = _snapshot()
    oversized_temporary = replace(
        _pack(
            "cand_oversized",
            candidate_kind="temporary",
            content="shared-content",
            sequence=1,
        ),
        evidence=tuple(
            SafeEvidence(
                reference=f"https:sha256:{_hash(f'evidence:{index}')}",
                reference_sha256=_hash(f"reference:{index}"),
            )
            for index in range(20)
        ),
    )
    permanent = _pack(
        "mem_permanent",
        content="shared-content",
        sequence=2,
        last_event_sequence=6,
    )
    request = _request()
    result = _result(snapshot)
    first = module.assemble_evidence_packet(
        request=request,
        result=result,
        snapshot=snapshot,
        build=_build(oversized_temporary, permanent),
    )
    prior = module.PriorContextSelection(
        manifest_id=first.manifest_id,
        committed=True,
        session_id=request.session_id,
        context_id=request.context_id,
        scope=request.scope,
        epistemic_context=request.epistemic_context,
        canonical_highwater=snapshot.highwater,
        canonical_content_fingerprint=snapshot.content_fingerprint,
        index_stale=result.index_stale,
        degraded_reasons=first.degraded_reasons,
        records=first.selected,
    )

    second = module.assemble_evidence_packet(
        request=request,
        result=result,
        snapshot=snapshot,
        build=_build(oversized_temporary, permanent),
        prior=prior,
    )

    assert second.selected == ()
    assert second.suppressed_ids == ("mem_permanent",)
    assert second.duplicate_ids == ()
    assert second.budget_omitted_ids == ("cand_oversized",)


def test_duplicate_selection_accounts_for_prompt_evidence_header_savings() -> None:
    module = _assembler()
    snapshot = _snapshot()
    prompt_uri = "codex-event://session_a/7"
    prompt = SafeEvidence(
        reference=prompt_uri,
        reference_sha256=_hash(prompt_uri),
        event_session_id="session_a",
        event_turn_id="turn_2",
        event_name="UserPromptSubmit",
        payload_blob_sha256=_hash("prompt-payload"),
    )
    other_uri = "codex-event://session_a/6"
    other = replace(
        prompt,
        reference=other_uri,
        reference_sha256=_hash(other_uri),
        event_turn_id="turn_1",
        payload_blob_sha256=_hash("other-payload"),
    )
    without_prompt = replace(
        _pack("mem_without", content="shared-content", sequence=2),
        evidence=(other,),
    )
    with_prompt = replace(
        _pack("mem_current", content="shared-content", sequence=1),
        evidence=(prompt,),
    )
    request = _request(prompt_event_uri=prompt_uri)
    result = _result(snapshot)
    prompt_only = module.assemble_evidence_packet(
        request=request,
        result=result,
        snapshot=snapshot,
        build=EvidencePackBuildResult((with_prompt,), (), 0, (), prompt),
    )

    assembled = module.assemble_evidence_packet(
        request=replace(request, max_chars=prompt_only.packet_chars),
        result=result,
        snapshot=snapshot,
        build=EvidencePackBuildResult(
            (without_prompt, with_prompt), (), 0, (), prompt
        ),
    )

    assert [item.candidate_id for item in assembled.selected] == [
        "mem_current"
    ]
    assert assembled.duplicate_ids == ("mem_without",)
    assert assembled.budget_omitted_ids == ()


def test_nonmandatory_duplicate_falls_back_to_an_alternative_that_fits() -> None:
    module = _assembler()
    snapshot = _snapshot()
    temporary = replace(
        _pack(
            "cand_temporary",
            candidate_kind="temporary",
            content="shared-content",
            sequence=2,
        ),
        evidence=tuple(
            SafeEvidence(
                reference=f"https:sha256:{_hash(f'temporary:{index}')}",
                reference_sha256=_hash(f"temporary-reference:{index}"),
            )
            for index in range(16)
        ),
    )
    permanent = _pack(
        "mem_permanent",
        content="shared-content",
        sequence=1,
    )
    result = _result(snapshot)
    permanent_only = module.assemble_evidence_packet(
        request=_request(),
        result=result,
        snapshot=snapshot,
        build=_build(permanent),
    )

    assembled = module.assemble_evidence_packet(
        request=_request(max_chars=permanent_only.packet_chars),
        result=result,
        snapshot=snapshot,
        build=_build(temporary, permanent),
    )

    assert [item.candidate_id for item in assembled.selected] == [
        "mem_permanent"
    ]
    assert assembled.duplicate_ids == ("cand_temporary",)
    assert assembled.budget_omitted_ids == ()
