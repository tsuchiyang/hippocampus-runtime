from __future__ import annotations

from importlib import import_module
import hashlib
import json

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import capture_candidate
from codex_memory.hook_input import parse_hook_payload
from codex_memory.memory_evidence import event_uri
from codex_memory.memory_review import review_candidate
from codex_memory.retrieval.canonical_snapshot import (
    capture_canonical_snapshot,
    open_verified_canonical_snapshot,
)
from codex_memory.retrieval.contracts import (
    CandidateTrace,
    DegradedReason,
    EpistemicContext,
    RetrievalMode,
    RetrievalQueryResult,
    RetrievalRequest,
    RetrievalScope,
    RetrievalStatus,
    RetrievedCandidate,
)
from codex_memory.retrieval.errors import RetrievalValidationError
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.structured_ledger import apply_action, propose_record


SCOPE = r"C:\Work\Memory Runtime"


def _pack_module():
    try:
        return import_module("codex_memory.retrieval.evidence_pack")
    except ModuleNotFoundError:
        pytest.fail("P2.4 evidence-pack module is missing", pytrace=False)


def _request(
    *,
    session_id: str = "session_a",
    turn_id: str = "turn_2",
    context_id: str = "startup",
    prompt_event_uri: str | None = None,
    scope_key: str = SCOPE,
) -> RetrievalRequest:
    return RetrievalRequest(
        query="Which storage engine is authoritative?",
        session_id=session_id,
        turn_id=turn_id,
        context_id=context_id,
        prompt_event_uri=prompt_event_uri,
        scope=RetrievalScope(scope_key),
        epistemic_context=EpistemicContext(),
        mode=RetrievalMode.OFF,
        max_chars=4000,
    )


def _query_result(snapshot, candidate: RetrievedCandidate) -> RetrievalQueryResult:
    return RetrievalQueryResult(
        status=RetrievalStatus.CURRENT,
        mode=RetrievalMode.OFF,
        generation=None,
        canonical_highwater=snapshot.highwater,
        indexed_highwater=snapshot.highwater,
        index_stale=False,
        degraded_reasons=(),
        candidates=(candidate,),
        trace=(CandidateTrace(candidate.candidate_id, (), (), 1, True, ("selected",)),),
    )


def _capture_and_promote(
    runtime,
    *,
    record_type: str = "decision",
    session_id: str = "session_a",
    turn_id: str = "turn_1",
    context_id: str = "startup",
    scope_key: str = SCOPE,
):
    raw_prompt = "The user explicitly approved SQLite; raw prompt must stay in its blob."
    event = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": session_id,
                "turn_id": turn_id,
                "cwd": scope_key,
                "prompt": raw_prompt,
            }
        )
    )
    evidence = event_uri(session_id, event.event_id)
    candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=scope_key,
        record_type=record_type,
        scope_kind="project",
        scope_key=scope_key,
        topic_key="storage.engine",
        statement="Use SQLite as the durable memory store.",
        value={"engine": "sqlite3"},
        session_id=session_id,
        turn_id=turn_id,
        context_id=context_id,
        idempotency_key=f"capture-{record_type}",
        evidence_uris=(evidence,),
    )
    reviewed = review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=candidate.candidate_id,
        outcome="promote",
        reason="Confirmed by the user and verified against the captured event.",
        idempotency_key=f"promote-{record_type}",
        evidence_uris=(evidence,),
    )
    assert reviewed.permanent_record_id is not None
    return raw_prompt, event, candidate, reviewed


def _retrieved_permanent(
    runtime, record_id: str, *, mandatory: bool
) -> RetrievedCandidate:
    row = runtime.connection.execute(
        "SELECT r.*, c.state FROM memory_records AS r "
        "JOIN memory_current AS c ON c.record_id = r.record_id WHERE r.record_id = ?",
        (record_id,),
    ).fetchone()
    return RetrievedCandidate(
        candidate_kind="record",
        candidate_id=record_id,
        record_type=row["record_type"],
        scope_kind=row["scope_kind"],
        scope_key=row["scope_key"],
        state=row["state"],
        sequence=row["record_seq"],
        content_sha256=row["content_sha256"],
        mandatory=mandatory,
        fused_score_numerator=1,
        fused_score_denominator=61,
        contributions=(),
    )


def test_build_pack_reconstructs_real_promotion_chain_without_blob_payload(runtime) -> None:
    module = _pack_module()
    raw_prompt, source_event, candidate, reviewed = _capture_and_promote(runtime)
    permanent_id = reviewed.permanent_record_id
    row = runtime.connection.execute(
        "SELECT r.*, c.state FROM memory_records AS r "
        "JOIN memory_current AS c ON c.record_id = r.record_id WHERE r.record_id = ?",
        (permanent_id,),
    ).fetchone()
    retrieved = RetrievedCandidate(
        candidate_kind="record",
        candidate_id=permanent_id,
        record_type=row["record_type"],
        scope_kind=row["scope_kind"],
        scope_key=row["scope_key"],
        state=row["state"],
        sequence=row["record_seq"],
        content_sha256=row["content_sha256"],
        mandatory=False,
        fused_score_numerator=1,
        fused_score_denominator=61,
        contributions=(),
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )

    assert result.omissions == ()
    assert result.degraded_reasons == ()
    assert len(result.packs) == 1
    pack = result.packs[0]
    assert pack.card.identifier == permanent_id
    assert pack.card.candidate_kind == "record"
    assert pack.card.state == "active"
    assert pack.card.content_sha256 == candidate.content_sha256
    assert pack.card.last_event.action == "activate"
    assert pack.card.last_event.event_sha256
    assert pack.card.certified is False

    assert len(pack.promotions) == 1
    promotion = pack.promotions[0]
    assert promotion.candidate_id == candidate.candidate_id
    assert promotion.origin.session_id == "session_a"
    assert promotion.origin.turn_id == "turn_1"
    assert promotion.origin.context_id == "startup"
    assert promotion.capture_event.action == "capture"
    assert promotion.review_event.action == "promote"
    assert promotion.review_reason == (
        "Confirmed by the user and verified against the captured event."
    )
    assert promotion.policy_code == "promoted_new"
    assert promotion.permanent_record_created is True
    assert promotion.superseded_record_id is None
    assert promotion.capture_evidence[0].reference == event_uri(
        "session_a", source_event.event_id
    )
    assert promotion.capture_evidence[0].payload_blob_sha256 == (
        source_event.payload_blob_sha256
    )
    assert promotion.review_evidence[0].payload_blob_sha256 == (
        source_event.payload_blob_sha256
    )

    document = pack.to_dict()
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True)
    assert raw_prompt not in encoded
    assert source_event.payload_blob_sha256 in encoded
    assert pack.pack_sha256 == module.evidence_pack_sha256(pack)


@pytest.mark.parametrize(
    ("record_type", "mandatory"),
    (("decision", False), ("constitution", True)),
)
def test_real_promoted_record_fits_the_supported_packet_budget(
    runtime, record_type: str, mandatory: bool
) -> None:
    module = _pack_module()
    session_id = "01a09a85-a36b-7131-a291-a52afbc35800"
    origin_turn = "01a09a85-a36b-7131-a291-a52afbc35801"
    request_turn = "01a09a85-a36b-7131-a291-a52afbc35802"
    context_id = "cp_1f4d76f0a06347f8aef1b54c1b1a8650"
    scope_key = ('C' + ':\\Users\\tsuchi yang\\Documents\\ChatGPT\\memery')
    raw_prompt, source_event, _candidate, reviewed = _capture_and_promote(
        runtime,
        record_type=record_type,
        session_id=session_id,
        turn_id=origin_turn,
        context_id=context_id,
        scope_key=scope_key,
    )
    retrieved = _retrieved_permanent(
        runtime, reviewed.permanent_record_id, mandatory=mandatory
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    request = _request(
        session_id=session_id,
        turn_id=request_turn,
        context_id=context_id,
        prompt_event_uri=event_uri(session_id, source_event.event_id),
        scope_key=scope_key,
    )

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        query_result = _query_result(snapshot, retrieved)
        build = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=request,
            result=query_result,
            snapshot=snapshot,
        )
        assembled = import_module(
            "codex_memory.retrieval.assembler"
        ).assemble_evidence_packet(
            request=request,
            result=query_result,
            snapshot=snapshot,
            build=build,
        )

    assert [item.candidate_id for item in assembled.selected] == [
        reviewed.permanent_record_id
    ]
    assert assembled.packet_chars <= 4000
    assert raw_prompt not in assembled.packet


def test_intermediate_review_evidence_is_preserved_for_temporary_and_promoted_chains(
    runtime,
) -> None:
    module = _pack_module()
    raw_prompt = "The user established this Constitution rule in the source event."
    source = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_review",
                "cwd": SCOPE,
                "prompt": raw_prompt,
            }
        )
    )
    source_uri = event_uri("session_a", source.event_id)
    candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="constitution",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="safety.review-chain",
        statement="Preserve all review evidence in derived provenance.",
        value=None,
        session_id="session_a",
        turn_id="turn_review",
        context_id="startup",
        idempotency_key="capture-intermediate-review",
        evidence_uris=(),
    )
    review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=candidate.candidate_id,
        outcome="defer",
        reason="Keep this pending until the next coherent review step.",
        idempotency_key="defer-with-authority-evidence",
        evidence_uris=(source_uri,),
    )
    current = runtime.connection.execute(
        "SELECT c.*, p.state FROM memory_candidates AS c "
        "JOIN memory_candidate_current AS p ON p.candidate_id = c.candidate_id "
        "WHERE c.candidate_id = ?",
        (candidate.candidate_id,),
    ).fetchone()
    temporary = RetrievedCandidate(
        "temporary",
        candidate.candidate_id,
        current["record_type"],
        current["scope_kind"],
        current["scope_key"],
        current["state"],
        current["candidate_seq"],
        current["content_sha256"],
        False,
        0,
        1,
        (),
        "UNVERIFIED TEMPORARY",
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        temporary_build = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, temporary),
            snapshot=snapshot,
        )

    temporary_pack = temporary_build.packs[0]
    assert temporary_pack.candidate_provenance is not None
    assert temporary_pack.candidate_provenance.origin.session_id == "session_a"
    assert temporary_pack.candidate_provenance.origin.turn_id == "turn_review"
    assert temporary_pack.candidate_provenance.origin.context_id == "startup"
    assert temporary_pack.candidate_provenance.capture_event.action == "capture"
    assert [review.event.action for review in temporary_pack.candidate_reviews] == [
        "defer"
    ]
    assert temporary_pack.candidate_reviews[0].evidence[0].reference == source_uri
    assert temporary_pack.evidence[0].payload_blob_sha256 == source.payload_blob_sha256

    promoted = review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=candidate.candidate_id,
        outcome="promote",
        reason="The earlier user evidence certifies this rule.",
        idempotency_key="promote-after-evidenced-defer",
        evidence_uris=(),
    )
    assert promoted.permanent_record_id is not None
    permanent = _retrieved_permanent(
        runtime, promoted.permanent_record_id, mandatory=True
    )

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        permanent_build = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, permanent),
            snapshot=snapshot,
        )

    permanent_pack = permanent_build.packs[0]
    promotion = permanent_pack.promotions[0]
    assert [review.event.action for review in promotion.reviews] == [
        "defer",
        "promote",
    ]
    assert promotion.reviews[0].evidence[0].reference == source_uri
    assert promotion.reviews[1].evidence == ()
    assert permanent_pack.evidence[0].payload_blob_sha256 == source.payload_blob_sha256
    encoded = json.dumps(permanent_pack.to_dict(), ensure_ascii=False, sort_keys=True)
    assert source_uri in encoded
    assert source.payload_blob_sha256 in encoded
    assert raw_prompt not in encoded


def test_build_pack_keeps_same_session_temporary_memory_explicitly_unverified(runtime) -> None:
    module = _pack_module()
    candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="assumption",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.platform",
        statement="The current target is Windows.",
        value={"platform": "windows"},
        session_id="session_a",
        turn_id="turn_1",
        context_id="startup",
        idempotency_key="temporary-assumption",
    )
    row = runtime.connection.execute(
        "SELECT c.*, current.state FROM memory_candidates AS c "
        "JOIN memory_candidate_current AS current ON current.candidate_id = c.candidate_id "
        "WHERE c.candidate_id = ?",
        (candidate.candidate_id,),
    ).fetchone()
    retrieved = RetrievedCandidate(
        candidate_kind="temporary",
        candidate_id=candidate.candidate_id,
        record_type=row["record_type"],
        scope_kind=row["scope_kind"],
        scope_key=row["scope_key"],
        state=row["state"],
        sequence=row["candidate_seq"],
        content_sha256=row["content_sha256"],
        mandatory=False,
        fused_score_numerator=0,
        fused_score_denominator=1,
        contributions=(),
        trust_label="UNVERIFIED TEMPORARY",
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )
        other_session = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(session_id="session_b"),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )

    assert result.omissions == ()
    assert len(result.packs) == 1
    pack = result.packs[0]
    assert pack.card.identifier == candidate.candidate_id
    assert pack.card.candidate_kind == "temporary"
    assert pack.card.state == "temporary"
    assert pack.card.trust_label == "UNVERIFIED TEMPORARY"
    assert pack.card.certified is False
    assert pack.card.last_event.action == "capture"
    assert pack.promotions == ()
    assert pack.supersession_relations == ()
    assert other_session.packs == ()
    assert [item.to_dict() for item in other_session.omissions] == [
        {
            "candidate_id": candidate.candidate_id,
            "reason": "temporary_not_visible",
        }
    ]


def test_builder_omits_and_counts_secret_material_split_across_record_fields(
    runtime,
) -> None:
    module = _pack_module()
    candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="assumption",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.split-secret",
        statement=('-' + '----BEGIN PRIVATE KEY-----'),
        value={
            "body": "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=",
            "footer": "-----END PRIVATE KEY-----",
        },
        session_id="session_a",
        turn_id="turn_secret",
        context_id="startup",
        idempotency_key="capture-split-secret",
    )
    row = runtime.connection.execute(
        "SELECT c.*, current.state FROM memory_candidates AS c "
        "JOIN memory_candidate_current AS current "
        "ON current.candidate_id = c.candidate_id WHERE c.candidate_id = ?",
        (candidate.candidate_id,),
    ).fetchone()
    retrieved = RetrievedCandidate(
        "temporary",
        candidate.candidate_id,
        row["record_type"],
        row["scope_kind"],
        row["scope_key"],
        row["state"],
        row["candidate_seq"],
        row["content_sha256"],
        False,
        0,
        1,
        (),
        "UNVERIFIED TEMPORARY",
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )

    assert result.packs == ()
    assert [item.to_dict() for item in result.omissions] == [
        {"candidate_id": candidate.candidate_id, "reason": "unsafe_content"}
    ]
    assert result.unsafe_records_omitted == 1
    assert result.degraded_reasons == ("unsafe_content_omitted",)


def test_certified_constitution_keeps_event_identity_and_hashes_external_reference(runtime) -> None:
    module = _pack_module()
    source = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_constitution",
                "cwd": SCOPE,
                "prompt": "Always keep canonical memory append-only.",
            }
        )
    )
    local_uri = event_uri("session_a", source.event_id)
    external_uri = "https://example.invalid/audit/append-only"
    candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="constitution",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="safety.append-only",
        statement="Canonical memory history is append-only.",
        value=None,
        session_id="session_a",
        turn_id="turn_constitution",
        context_id="checkpoint_7",
        idempotency_key="capture-constitution-evidence-pack",
        evidence_uris=(local_uri, external_uri),
    )
    promoted = review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=candidate.candidate_id,
        outcome="promote",
        reason="The user directly established this invariant.",
        idempotency_key="promote-constitution-evidence-pack",
        evidence_uris=(local_uri,),
    )
    row = runtime.connection.execute(
        "SELECT r.*, current.state FROM memory_records AS r "
        "JOIN memory_current AS current ON current.record_id = r.record_id "
        "WHERE r.record_id = ?",
        (promoted.permanent_record_id,),
    ).fetchone()
    retrieved = RetrievedCandidate(
        "record",
        row["record_id"],
        row["record_type"],
        row["scope_kind"],
        row["scope_key"],
        row["state"],
        row["record_seq"],
        row["content_sha256"],
        True,
        0,
        1,
        (),
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(context_id="checkpoint_7"),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )

    assert result.omissions == ()
    pack = result.packs[0]
    assert pack.card.certified is True
    assert pack.card.trust_label == "CERTIFIED CONSTITUTION"
    local = next(item for item in pack.evidence if item.reference == local_uri)
    assert local.event_session_id == "session_a"
    assert local.event_turn_id == "turn_constitution"
    assert local.payload_blob_sha256 == source.payload_blob_sha256
    expected_external = "https:sha256:" + hashlib.sha256(
        external_uri.encode("utf-8")
    ).hexdigest()
    external = next(item for item in pack.evidence if item.reference == expected_external)
    assert external.reference_sha256 == hashlib.sha256(
        expected_external.encode("utf-8")
    ).hexdigest()
    assert external_uri not in json.dumps(pack.to_dict(), sort_keys=True)


def test_missing_event_blob_omits_only_the_affected_pack_as_evidence_unavailable(runtime) -> None:
    module = _pack_module()
    _raw_prompt, source, _candidate, reviewed = _capture_and_promote(runtime)
    row = runtime.connection.execute(
        "SELECT r.*, current.state FROM memory_records AS r "
        "JOIN memory_current AS current ON current.record_id = r.record_id "
        "WHERE r.record_id = ?",
        (reviewed.permanent_record_id,),
    ).fetchone()
    retrieved = RetrievedCandidate(
        "record",
        row["record_id"],
        row["record_type"],
        row["scope_kind"],
        row["scope_key"],
        row["state"],
        row["record_seq"],
        row["content_sha256"],
        False,
        1,
        61,
        (),
    )
    blob_path = (
        runtime.paths.blobs_dir
        / "sha256"
        / source.payload_blob_sha256[:2]
        / f"{source.payload_blob_sha256}.bin"
    )
    blob_path.unlink()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )

    assert result.packs == ()
    assert [item.to_dict() for item in result.omissions] == [
        {"candidate_id": row["record_id"], "reason": "evidence_unavailable"}
    ]
    assert result.degraded_reasons == ("evidence_unavailable",)
    assert result.unsafe_records_omitted == 0


def test_build_verifies_prompt_event_identity_without_copying_prompt_blob(runtime) -> None:
    module = _pack_module()
    raw_prompt = "A private prompt body that must never enter the evidence packet."
    source = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_prompt",
                "cwd": SCOPE,
                "prompt": raw_prompt,
            }
        )
    )
    uri = event_uri("session_a", source.event_id)
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        query_result = RetrievalQueryResult(
            status=RetrievalStatus.CURRENT,
            mode=RetrievalMode.OFF,
            generation=None,
            canonical_highwater=snapshot.highwater,
            indexed_highwater=snapshot.highwater,
            index_stale=False,
            degraded_reasons=(),
            candidates=(),
            trace=(),
        )
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(prompt_event_uri=uri),
            result=query_result,
            snapshot=snapshot,
        )
        assembled = import_module(
            "codex_memory.retrieval.assembler"
        ).assemble_evidence_packet(
            request=_request(prompt_event_uri=uri),
            result=query_result,
            snapshot=snapshot,
            build=result,
        )

    assert result.prompt_evidence is not None
    assert result.prompt_evidence.reference == uri
    assert result.prompt_evidence.event_session_id == "session_a"
    assert result.prompt_evidence.event_turn_id == "turn_prompt"
    assert result.prompt_evidence.event_name == "UserPromptSubmit"
    assert result.prompt_evidence.payload_blob_sha256 == source.payload_blob_sha256
    assert raw_prompt not in json.dumps(
        result.prompt_evidence.to_dict(), ensure_ascii=False, sort_keys=True
    )
    assert raw_prompt not in assembled.packet
    prompt_line = next(
        line
        for line in assembled.packet.splitlines()
        if line.startswith("prompt_evidence=")
    )
    assert json.loads(prompt_line.removeprefix("prompt_evidence=")) == {
        "payload_blob_sha256": source.payload_blob_sha256,
        "reference": uri,
    }


def test_build_rejects_non_prompt_or_cross_session_prompt_evidence(runtime) -> None:
    module = _pack_module()
    tool_event = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "PostToolUse",
                "session_id": "session_a",
                "turn_id": "turn_1",
                "cwd": SCOPE,
                "tool_name": "Read",
                "tool_use_id": "tool_1",
                "tool_input": {"path": "README.md"},
                "tool_response": {"ok": True},
            }
        )
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = RetrievalQueryResult(
            status=RetrievalStatus.CURRENT,
            mode=RetrievalMode.OFF,
            generation=None,
            canonical_highwater=snapshot.highwater,
            indexed_highwater=snapshot.highwater,
            index_stale=False,
            degraded_reasons=(),
            candidates=(),
            trace=(),
        )
        with pytest.raises(
            RetrievalValidationError,
            match="current session user prompt",
        ):
            module.build_evidence_packs(
                connection,
                BlobStore(runtime.paths.blobs_dir),
                request=_request(
                    prompt_event_uri=event_uri("session_a", tool_event.event_id)
                ),
                result=result,
                snapshot=snapshot,
            )
        with pytest.raises(RetrievalValidationError, match="another session"):
            module.build_evidence_packs(
                connection,
                BlobStore(runtime.paths.blobs_dir),
                request=_request(
                    prompt_event_uri=event_uri("session_b", tool_event.event_id)
                ),
                result=result,
                snapshot=snapshot,
            )


def test_build_pack_reconstructs_explicit_supersession_relation(runtime) -> None:
    module = _pack_module()
    old = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="storage.engine",
        statement="Use the old storage engine.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-old-record",
    )
    apply_action(
        runtime.connection,
        record_id=old.record_id,
        action="activate",
        idempotency_key="p24-old-activate",
    )
    source = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_supersede",
                "cwd": SCOPE,
                "prompt": "Replace the old storage engine with SQLite.",
            }
        )
    )
    uri = event_uri("session_a", source.event_id)
    candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="storage.engine",
        statement="Use SQLite as the new storage engine.",
        value=None,
        session_id="session_a",
        turn_id="turn_supersede",
        context_id="checkpoint_9",
        idempotency_key="p24-superseding-candidate",
        evidence_uris=(uri,),
    )
    promoted = review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=candidate.candidate_id,
        outcome="promote",
        reason="The new direct user evidence supersedes the old decision.",
        idempotency_key="p24-superseding-review",
        evidence_uris=(uri,),
        supersedes_record_id=old.record_id,
    )
    assert promoted.action == "promote"
    assert promoted.permanent_record_id is not None
    row = runtime.connection.execute(
        "SELECT r.*, current.state FROM memory_records AS r "
        "JOIN memory_current AS current ON current.record_id = r.record_id "
        "WHERE r.record_id = ?",
        (promoted.permanent_record_id,),
    ).fetchone()
    retrieved = RetrievedCandidate(
        "record",
        row["record_id"],
        row["record_type"],
        row["scope_kind"],
        row["scope_key"],
        row["state"],
        row["record_seq"],
        row["content_sha256"],
        False,
        1,
        61,
        (),
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(context_id="checkpoint_9"),
            result=_query_result(snapshot, retrieved),
            snapshot=snapshot,
        )

    assert result.omissions == ()
    pack = result.packs[0]
    assert pack.promotions[0].candidate_id == candidate.candidate_id
    assert pack.promotions[0].superseded_record_id == old.record_id
    assert len(pack.supersession_relations) == 1
    relation = pack.supersession_relations[0]
    assert relation.relation_type == "superseded_by"
    assert relation.source_record_id == old.record_id
    assert relation.target_record_id == promoted.permanent_record_id
    assert relation.lifecycle_event_id
    assert relation.relation_sha256


def test_stale_retrieval_candidate_cannot_reintroduce_superseded_record(runtime) -> None:
    module = _pack_module()
    old = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.lifecycle-old",
        statement="Old lifecycle decision.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-stale-old",
    )
    replacement = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.lifecycle-new",
        statement="Replacement lifecycle decision.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-stale-replacement",
    )
    apply_action(
        runtime.connection,
        record_id=old.record_id,
        action="activate",
        idempotency_key="p24-stale-old-activate",
    )
    apply_action(
        runtime.connection,
        record_id=old.record_id,
        action="supersede",
        replacement_record_id=replacement.record_id,
        idempotency_key="p24-stale-old-supersede",
    )
    old_sequence = runtime.connection.execute(
        "SELECT record_seq FROM memory_records WHERE record_id = ?",
        (old.record_id,),
    ).fetchone()[0]
    stale_candidate = RetrievedCandidate(
        "record",
        old.record_id,
        old.record_type,
        old.scope_kind,
        old.scope_key,
        "active",
        old_sequence,
        old.content_sha256,
        False,
        1,
        61,
        (),
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        result = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=_query_result(snapshot, stale_candidate),
            snapshot=snapshot,
        )

    assert result.packs == ()
    assert [item.to_dict() for item in result.omissions] == [
        {"candidate_id": old.record_id, "reason": "candidate_not_current"}
    ]


def test_complete_pack_secret_scan_omits_only_cross_provenance_secret(runtime) -> None:
    module = _pack_module()
    source = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_split_secret",
                "cwd": SCOPE,
                "prompt": "Record the reviewed assumption.",
            }
        )
    )
    uri = event_uri("session_a", source.event_id)
    unsafe_candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="assumption",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.unsafe-review",
        statement=('-' + '----BEGIN PRIVATE KEY-----'),
        value=None,
        session_id="session_a",
        turn_id="turn_split_secret",
        context_id="startup",
        idempotency_key="p24-split-secret-capture",
        evidence_uris=(uri,),
    )
    unsafe_review = review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=unsafe_candidate.candidate_id,
        outcome="promote",
        reason=(
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\n"
            "-----END PRIVATE KEY-----"
        ),
        idempotency_key="p24-split-secret-promote",
        evidence_uris=(uri,),
    )
    assert unsafe_review.permanent_record_id is not None
    safe = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.safe-record",
        statement="Use SQLite for durable storage.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-safe-neighbor",
    )
    apply_action(
        runtime.connection,
        record_id=safe.record_id,
        action="activate",
        idempotency_key="p24-safe-neighbor-activate",
    )
    unsafe_retrieved = _retrieved_permanent(
        runtime, unsafe_review.permanent_record_id, mandatory=False
    )
    safe_retrieved = _retrieved_permanent(runtime, safe.record_id, mandatory=False)
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        query_result = RetrievalQueryResult(
            status=RetrievalStatus.CURRENT,
            mode=RetrievalMode.OFF,
            generation=None,
            canonical_highwater=snapshot.highwater,
            indexed_highwater=snapshot.highwater,
            index_stale=False,
            degraded_reasons=(),
            candidates=(unsafe_retrieved, safe_retrieved),
            trace=(),
        )
        build = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=query_result,
            snapshot=snapshot,
        )
        assembled = import_module(
            "codex_memory.retrieval.assembler"
        ).assemble_evidence_packet(
            request=_request(),
            result=query_result,
            snapshot=snapshot,
            build=build,
        )

    assert [pack.card.identifier for pack in build.packs] == [safe.record_id]
    assert [item.to_dict() for item in build.omissions] == [
        {
            "candidate_id": unsafe_review.permanent_record_id,
            "reason": "unsafe_content",
        }
    ]
    assert build.unsafe_records_omitted == 1
    assert build.degraded_reasons == ("unsafe_content_omitted",)
    assert [item.candidate_id for item in assembled.selected] == [safe.record_id]


def test_packet_order_secret_scan_omits_unsafe_temporary_but_keeps_safe_record(
    runtime,
) -> None:
    module = _pack_module()
    unsafe_candidate = capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="assumption",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.packet-order-secret",
        statement=(
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\n"
            "-----END PRIVATE KEY-----"
        ),
        value=None,
        session_id="session_a",
        turn_id="turn_packet_order_secret",
        context_id="startup",
        idempotency_key="p24-packet-order-secret-capture",
    )
    review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        candidate_id=unsafe_candidate.candidate_id,
        outcome="defer",
        reason=('-' + '----BEGIN PRIVATE KEY-----'),
        idempotency_key="p24-packet-order-secret-defer",
    )
    safe = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.packet-order-safe-neighbor",
        statement="Use SQLite for durable storage.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-packet-order-safe-neighbor",
    )
    apply_action(
        runtime.connection,
        record_id=safe.record_id,
        action="activate",
        idempotency_key="p24-packet-order-safe-neighbor-activate",
    )
    unsafe_row = runtime.connection.execute(
        "SELECT c.*, current.state FROM memory_candidates AS c "
        "JOIN memory_candidate_current AS current "
        "ON current.candidate_id = c.candidate_id "
        "WHERE c.candidate_id = ?",
        (unsafe_candidate.candidate_id,),
    ).fetchone()
    unsafe_retrieved = RetrievedCandidate(
        "temporary",
        unsafe_candidate.candidate_id,
        unsafe_row["record_type"],
        unsafe_row["scope_kind"],
        unsafe_row["scope_key"],
        unsafe_row["state"],
        unsafe_row["candidate_seq"],
        unsafe_row["content_sha256"],
        False,
        0,
        1,
        (),
        "UNVERIFIED TEMPORARY",
    )
    safe_retrieved = _retrieved_permanent(runtime, safe.record_id, mandatory=False)
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)

    with open_verified_canonical_snapshot(paths) as (connection, snapshot):
        query_result = RetrievalQueryResult(
            status=RetrievalStatus.CURRENT,
            mode=RetrievalMode.OFF,
            generation=None,
            canonical_highwater=snapshot.highwater,
            indexed_highwater=snapshot.highwater,
            index_stale=False,
            degraded_reasons=(),
            candidates=(unsafe_retrieved, safe_retrieved),
            trace=(),
        )
        build = module.build_evidence_packs(
            connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=query_result,
            snapshot=snapshot,
        )
        assembled = import_module(
            "codex_memory.retrieval.assembler"
        ).assemble_evidence_packet(
            request=_request(),
            result=query_result,
            snapshot=snapshot,
            build=build,
        )

    assert [pack.card.identifier for pack in build.packs] == [safe.record_id]
    assert [item.to_dict() for item in build.omissions] == [
        {"candidate_id": unsafe_candidate.candidate_id, "reason": "unsafe_content"}
    ]
    assert build.unsafe_records_omitted == 1
    assert build.degraded_reasons == ("unsafe_content_omitted",)
    assert [item.candidate_id for item in assembled.selected] == [safe.record_id]


def test_builder_rejects_snapshot_not_bound_to_its_read_transaction(runtime) -> None:
    module = _pack_module()
    record = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="runtime.snapshot-binding",
        statement="Use the verified snapshot transaction.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-snapshot-binding-record",
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key="p24-snapshot-binding-activate",
    )
    retrieved = _retrieved_permanent(runtime, record.record_id, mandatory=False)
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    stale_snapshot = capture_canonical_snapshot(paths)
    stale_result = _query_result(stale_snapshot, retrieved)
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="close",
        idempotency_key="p24-snapshot-binding-close",
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="reopen",
        idempotency_key="p24-snapshot-binding-reopen",
    )

    with pytest.raises(
        RetrievalValidationError,
        match="snapshot does not match read transaction",
    ):
        module.build_evidence_packs(
            runtime.connection,
            BlobStore(runtime.paths.blobs_dir),
            request=_request(),
            result=stale_result,
            snapshot=stale_snapshot,
        )
