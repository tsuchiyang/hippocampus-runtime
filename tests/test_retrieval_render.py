from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import import_module
import json

import pytest

from codex_memory.retrieval.canonical_snapshot import CanonicalSnapshot
from codex_memory.retrieval.contracts import (
    EpistemicContext,
    IndexGeneration,
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
from codex_memory.retrieval.errors import RetrievalValidationError


SCOPE = r"C:\Work\Memory Runtime"


def _assembler():
    module = import_module("codex_memory.retrieval.assembler")
    if not hasattr(module, "assemble_evidence_packet"):
        pytest.fail("P2.4 packet assembly API is missing", pytrace=False)
    return module


def _render():
    try:
        return import_module("codex_memory.retrieval.render")
    except ModuleNotFoundError:
        pytest.fail("P2.4 Protocol v2 renderer is missing", pytrace=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _highwater() -> IndexHighwater:
    return IndexHighwater(3, 6, 2, 1, 2, 4, 2)


def _snapshot() -> CanonicalSnapshot:
    return CanonicalSnapshot(
        schema_version=3,
        highwater=_highwater(),
        content_fingerprint=_hash("canonical-snapshot"),
        source_timestamp="2026-09-13T00:00:00.000000Z",
    )


def _generation() -> IndexGeneration:
    highwater = _highwater()
    return IndexGeneration(
        generation_id="gen_p24",
        state="active",
        canonical_highwater=highwater,
        indexed_highwater=highwater,
        canonical_content_fingerprint=_hash("canonical-snapshot"),
        provider_fingerprint="disabled",
        model_fingerprint="disabled",
        validation_state="validated",
        created_at="2026-09-13T00:00:00.000000Z",
        updated_at="2026-09-13T00:00:00.000000Z",
    )


def _request(*, max_chars: int = 4000, context_id: str = "checkpoint_7") -> RetrievalRequest:
    return RetrievalRequest(
        query="Which memory rules apply?",
        session_id="session_a",
        turn_id="turn_2",
        context_id=context_id,
        prompt_event_uri=None,
        scope=RetrievalScope(SCOPE, ("workspace_shared",)),
        epistemic_context=EpistemicContext(),
        mode=RetrievalMode.OFF,
        max_chars=max_chars,
    )


def _result(snapshot: CanonicalSnapshot) -> RetrievalQueryResult:
    generation = _generation()
    return RetrievalQueryResult(
        status=RetrievalStatus.CURRENT,
        mode=RetrievalMode.OFF,
        generation=generation,
        canonical_highwater=snapshot.highwater,
        indexed_highwater=generation.indexed_highwater,
        index_stale=False,
        degraded_reasons=(),
        candidates=(),
        trace=(),
    )


def _pack(
    identifier: str,
    *,
    candidate_kind: str = "record",
    record_type: str = "decision",
    certified: bool = False,
    statement: str | None = None,
    value: object = None,
    sequence: int = 1,
) -> EvidencePack:
    action = "capture" if candidate_kind == "temporary" else "activate"
    return EvidencePack(
        card=MemoryCard(
            identifier=identifier,
            candidate_kind=candidate_kind,
            record_type=record_type,
            scope_kind="project",
            scope_key=SCOPE,
            topic_key="runtime.packet",
            statement=statement or f"Remember {identifier}.",
            value=value,
            state="temporary" if candidate_kind == "temporary" else "active",
            sequence=sequence,
            content_sha256=_hash(identifier),
            trust_label=(
                "CERTIFIED CONSTITUTION"
                if certified
                else "UNVERIFIED TEMPORARY"
                if candidate_kind == "temporary"
                else "RETRIEVED MEMORY"
            ),
            certified=certified,
            last_event=LedgerEvent(
                f"event_{identifier}",
                sequence + 10,
                action,
                _hash(f"event:{identifier}"),
                "2026-09-13T00:00:00.000000Z",
            ),
        ),
        evidence=(),
    )


def _section(packet: str, opening: str, closing: str) -> list[dict[str, object]]:
    lines = packet.splitlines()
    start = lines.index(opening)
    end = lines.index(closing)
    values = []
    for line in lines[start + 1 : end]:
        parsed = json.loads(line)
        assert json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == line
        values.append(parsed)
    return values


def test_phase2_packet_has_explicit_partitions_metadata_and_deterministic_bytes() -> None:
    module = _assembler()
    snapshot = _snapshot()
    request = _request()
    build = EvidencePackBuildResult(
        packs=(
            _pack("mem_decision", sequence=3),
            _pack("cand_temp", candidate_kind="temporary", sequence=2),
            _pack(
                "mem_constitution",
                record_type="constitution",
                certified=True,
                sequence=1,
            ),
        ),
        omissions=(),
        unsafe_records_omitted=0,
        degraded_reasons=(),
    )

    first = module.assemble_evidence_packet(
        request=request,
        result=_result(snapshot),
        snapshot=snapshot,
        build=build,
    )
    second = module.assemble_evidence_packet(
        request=request,
        result=_result(snapshot),
        snapshot=snapshot,
        build=build,
    )

    assert first == second
    assert first.packet.startswith("[CODEX MEMORY PROTOCOL v2]\n")
    assert first.packet_chars == len(first.packet) <= request.max_chars
    assert first.packet_sha256 == hashlib.sha256(
        first.packet.encode("utf-8")
    ).hexdigest()
    assert first.packet_sha256 == (
        "0a115a76d98a311afd5882f23862a4e1087428200e956b0e5161dc27e9a41686"
    )
    assert first.manifest_id == "asm_28590ed145ed5fa02baa54741aa5d6bf"
    metadata_line = next(
        line for line in first.packet.splitlines() if line.startswith("retrieval_metadata=")
    )
    metadata = json.loads(metadata_line.removeprefix("retrieval_metadata="))
    assert metadata == {
        "canonical_content_fingerprint": snapshot.content_fingerprint,
        "canonical_highwater": snapshot.highwater.to_dict(),
        "degraded_reasons": [],
        "generation_id": "gen_p24",
        "index_stale": False,
        "indexed_highwater": snapshot.highwater.to_dict(),
        "manifest_id": first.manifest_id,
        "mode": "off",
        "status": "current",
    }
    certified = _section(
        first.packet,
        "[CERTIFIED CONSTITUTION DATA]",
        "[/CERTIFIED CONSTITUTION DATA]",
    )
    temporary = _section(
        first.packet,
        "[UNVERIFIED TEMPORARY DATA]",
        "[/UNVERIFIED TEMPORARY DATA]",
    )
    permanent = _section(
        first.packet,
        "[RETRIEVED GROUND-TRUTH MEMORY DATA]",
        "[/RETRIEVED GROUND-TRUTH MEMORY DATA]",
    )
    assert [item["evidence_pack"]["card"]["id"] for item in certified] == [
        "mem_constitution"
    ]
    assert [item["evidence_pack"]["card"]["id"] for item in temporary] == [
        "cand_temp"
    ]
    assert [item["evidence_pack"]["card"]["id"] for item in permanent] == [
        "mem_decision"
    ]
    assert "ground_truth identifies the provenance domain, not perfect correctness" in first.packet


def test_final_evidence_line_scan_rejects_secret_bearing_pack() -> None:
    module = _render()
    unsafe = _pack(
        "mem_unsafe",
        statement="api_key=live-value-that-must-not-render",
    )

    with pytest.raises(module.PacketSafetyError, match="unsafe evidence pack"):
        module.render_evidence_line(unsafe)


@pytest.mark.parametrize(
    "statement",
    (
        'Use token="example" in documentation.',
        'Write api_key="<redacted>" in the sample output.',
    ),
)
def test_final_secret_scan_accepts_safe_placeholders_after_json_serialization(
    statement: str,
) -> None:
    module = _render()
    pack = _pack("mem_placeholder", statement=statement)

    line = module.render_evidence_line(pack)
    parsed = json.loads(line)

    assert parsed["evidence_pack"]["card"]["statement"] == statement


def test_final_secret_scan_still_covers_rendered_request_metadata() -> None:
    module = _assembler()
    snapshot = _snapshot()
    unsafe_request = replace(
        _request(),
        scope=RetrievalScope("api_key=live-value-that-must-not-render"),
    )

    with pytest.raises(
        _render().PacketSafetyError,
        match="unsafe rendered packet",
    ):
        module.assemble_evidence_packet(
            request=unsafe_request,
            result=_result(snapshot),
            snapshot=snapshot,
            build=EvidencePackBuildResult((), (), 0, ()),
        )


def test_final_secret_scan_detects_private_key_split_across_record_fields() -> None:
    unsafe = _pack(
        "mem_split_pem",
        statement=('-' + '----BEGIN PRIVATE KEY-----'),
        value={
            "body": "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=",
            "footer": "-----END PRIVATE KEY-----",
        },
    )

    with pytest.raises(_render().PacketSafetyError, match="unsafe evidence pack"):
        _render().render_evidence_line(unsafe)


def test_final_secret_scan_detects_private_key_split_across_selected_records() -> None:
    snapshot = _snapshot()
    opening = _pack(
        "mem_pem_opening",
        statement=('-' + '----BEGIN PRIVATE KEY-----'),
    )
    closing = _pack(
        "mem_pem_closing",
        value={
            "body": "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=",
            "footer": "-----END PRIVATE KEY-----",
        },
    )

    with pytest.raises(_render().PacketSafetyError, match="unsafe rendered packet"):
        _assembler().assemble_evidence_packet(
            request=_request(),
            result=_result(snapshot),
            snapshot=snapshot,
            build=EvidencePackBuildResult((opening, closing), (), 0, ()),
        )


def test_memory_card_value_cannot_be_mutated_through_a_derived_view() -> None:
    pack = _pack(
        "mem_immutable",
        value={"platforms": ["windows"], "settings": {"mode": "safe"}},
    )
    original_hash = pack.pack_sha256

    exposed = pack.card.value
    exposed["platforms"].append("linux")
    exposed["settings"]["mode"] = "unsafe"
    serialized = pack.to_dict()
    serialized["card"]["value"]["platforms"].append("macos")

    assert pack.card.value == {
        "platforms": ["windows"],
        "settings": {"mode": "safe"},
    }
    assert pack.pack_sha256 == original_hash


def test_pack_hash_binds_audit_metadata_omitted_from_the_compact_packet_view() -> None:
    reference = "codex-event://session_a/17"
    first_evidence = SafeEvidence(
        reference=reference,
        reference_sha256=_hash(reference),
        event_session_id="session_a",
        event_turn_id="turn_2",
        event_name="UserPromptSubmit",
        event_created_at="2026-09-13T00:00:00.000000Z",
        payload_blob_sha256=_hash("prompt-payload"),
    )
    first = replace(_pack("mem_audit_hash"), evidence=(first_evidence,))
    second = replace(
        first,
        evidence=(
            replace(
                first_evidence,
                event_created_at="2026-09-13T00:00:01.000000Z",
            ),
        ),
    )

    first_line = json.loads(_render().render_evidence_line(first))
    second_line = json.loads(_render().render_evidence_line(second))

    assert first_line["evidence_pack"] == second_line["evidence_pack"]
    assert first_line["pack_sha256"] != second_line["pack_sha256"]


def test_packet_uses_exact_character_cap_and_omits_only_complete_records() -> None:
    module = _assembler()
    snapshot = _snapshot()
    packs = (
        _pack("mem_decision", sequence=3),
        _pack("cand_temp", candidate_kind="temporary", sequence=2),
        _pack(
            "mem_constitution",
            record_type="constitution",
            certified=True,
            sequence=1,
        ),
    )
    build = EvidencePackBuildResult(packs, (), 0, ())

    unconstrained = module.assemble_evidence_packet(
        request=_request(max_chars=4000),
        result=_result(snapshot),
        snapshot=snapshot,
        build=build,
    )
    exact_cap = unconstrained.packet_chars
    exact = module.assemble_evidence_packet(
        request=_request(max_chars=exact_cap),
        result=_result(snapshot),
        snapshot=snapshot,
        build=build,
    )
    constrained = module.assemble_evidence_packet(
        request=_request(max_chars=exact_cap - 1),
        result=_result(snapshot),
        snapshot=snapshot,
        build=build,
    )

    assert exact.packet_chars == exact_cap
    assert [item.candidate_id for item in exact.selected] == [
        "mem_constitution",
        "cand_temp",
        "mem_decision",
    ]
    assert constrained.packet_chars <= exact_cap - 1
    assert [item.candidate_id for item in constrained.selected] == [
        "mem_constitution",
        "cand_temp",
    ]
    assert dict(constrained.omission_counts)["budget"] == 1
    assert constrained.budget_omitted_ids == ("mem_decision",)
    assert constrained.build_omissions == ()
    assert "mem_decision" not in constrained.packet
    for opening, closing in (
        ("[CERTIFIED CONSTITUTION DATA]", "[/CERTIFIED CONSTITUTION DATA]"),
        ("[UNVERIFIED TEMPORARY DATA]", "[/UNVERIFIED TEMPORARY DATA]"),
        (
            "[RETRIEVED GROUND-TRUTH MEMORY DATA]",
            "[/RETRIEVED GROUND-TRUTH MEMORY DATA]",
        ),
    ):
        _section(constrained.packet, opening, closing)


def test_packet_fails_closed_when_skeleton_or_certified_data_cannot_fit() -> None:
    module = _assembler()
    snapshot = _snapshot()
    result = _result(snapshot)

    with pytest.raises(module.PacketBudgetError, match="protocol exceeds"):
        module.assemble_evidence_packet(
            request=_request(max_chars=1),
            result=result,
            snapshot=snapshot,
            build=EvidencePackBuildResult((), (), 0, ()),
        )

    certified = _pack(
        "mem_constitution",
        record_type="constitution",
        certified=True,
    )
    with pytest.raises(module.PacketBudgetError, match="Constitution"):
        module.assemble_evidence_packet(
            request=_request(max_chars=2000),
            result=result,
            snapshot=snapshot,
            build=EvidencePackBuildResult((certified,), (), 0, ()),
        )


def test_packet_renders_only_blob_verified_prompt_evidence_metadata() -> None:
    module = _assembler()
    snapshot = _snapshot()
    uri = "codex-event://session_a/17"
    request = RetrievalRequest(
        query="memory query",
        session_id="session_a",
        turn_id="turn_2",
        context_id="checkpoint_7",
        prompt_event_uri=uri,
        scope=RetrievalScope(SCOPE),
        epistemic_context=EpistemicContext(),
        mode=RetrievalMode.OFF,
        max_chars=4000,
    )
    evidence = SafeEvidence(
        reference=uri,
        reference_sha256=_hash(uri),
        event_session_id="session_a",
        event_turn_id="turn_2",
        event_name="UserPromptSubmit",
        event_created_at="2026-09-13T00:00:00.000000Z",
        payload_blob_sha256=_hash("private prompt payload"),
    )

    with pytest.raises(RetrievalValidationError, match="verified prompt evidence"):
        module.assemble_evidence_packet(
            request=request,
            result=_result(snapshot),
            snapshot=snapshot,
            build=EvidencePackBuildResult((), (), 0, ()),
        )

    assembled = module.assemble_evidence_packet(
        request=request,
        result=_result(snapshot),
        snapshot=snapshot,
        build=EvidencePackBuildResult((), (), 0, (), evidence),
    )
    prompt_line = next(
        line for line in assembled.packet.splitlines() if line.startswith("prompt_evidence=")
    )
    assert json.loads(prompt_line.removeprefix("prompt_evidence=")) == {
        "payload_blob_sha256": evidence.payload_blob_sha256,
        "reference": evidence.reference,
    }


def test_omission_counts_do_not_double_count_unsafe_build_omissions() -> None:
    module = _assembler()
    snapshot = _snapshot()
    assembled = module.assemble_evidence_packet(
        request=_request(),
        result=_result(snapshot),
        snapshot=snapshot,
        build=EvidencePackBuildResult(
            (),
            (
                PackOmission("mem_unsafe", "unsafe_content"),
                PackOmission("mem_missing", "evidence_missing"),
            ),
            1,
            ("unsafe_content_omitted", "evidence_missing"),
        ),
    )

    assert dict(assembled.omission_counts) == {
        "budget": 0,
        "duplicate": 0,
        "provenance": 1,
        "suppressed": 0,
        "unsafe": 1,
    }
    assert [item.to_dict() for item in assembled.build_omissions] == [
        {"candidate_id": "mem_unsafe", "reason": "unsafe_content"},
        {"candidate_id": "mem_missing", "reason": "evidence_missing"},
    ]
