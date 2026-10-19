from __future__ import annotations

from importlib import import_module, util
import os
from pathlib import Path
import sqlite3
from threading import Barrier, Thread

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.ledger_common import (
    IdempotencyConflict,
    LedgerError,
    ValidationError,
    VerificationError,
)
from codex_memory.memory_evidence import event_uri
from codex_memory.paths import RuntimePaths
from codex_memory.structured_ledger import apply_action, propose_record, verify_ledger


SCOPE = r"C:\Work\Repo"
MATCHING_SCOPE = SCOPE.swapcase() if os.name == "nt" else SCOPE
REASON = "Reviewed against the latest durable evidence."
LEDGER_TABLES = (
    "memory_records",
    "memory_events",
    "memory_evidence",
    "memory_relations",
    "memory_current",
    "memory_candidates",
    "memory_candidate_evidence",
    "memory_candidate_events",
    "memory_candidate_current",
)


def _review_module():
    spec = util.find_spec("codex_memory.memory_review")
    assert spec is not None, "memory review module is required"
    module = import_module("codex_memory.memory_review")
    assert hasattr(module, "review_candidate"), "review_candidate API is required"
    return module


def _candidate(
    runtime,
    *,
    key: str,
    record_type: str = "decision",
    scope_key: str = SCOPE,
    topic: str = "storage.engine",
    statement: str = "Use SQLite.",
    value=None,
    evidence=(),
    session_id: str = "session_a",
):
    return import_module("codex_memory.candidate_memory").capture_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        project_scope_key=scope_key,
        record_type=record_type,
        scope_kind="project",
        scope_key=scope_key,
        topic_key=topic,
        statement=statement,
        session_id=session_id,
        turn_id="turn_1",
        context_id="startup",
        idempotency_key=key,
        value=value,
        evidence_uris=evidence,
    )


def _permanent(
    runtime,
    *,
    key: str,
    record_type: str = "decision",
    scope_key: str = SCOPE,
    topic: str = "storage.engine",
    statement: str = "Use SQLite.",
    value=None,
    evidence=(),
):
    record = propose_record(
        runtime.connection,
        record_type=record_type,
        scope_kind="project",
        scope_key=scope_key,
        topic_key=topic,
        statement=statement,
        value=value,
        evidence_uris=evidence,
        idempotency_key=key,
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key=f"{key}-activate",
    )
    return record


def _review(runtime, candidate_id: str, *, key: str, **changes):
    arguments = {
        "candidate_id": candidate_id,
        "outcome": "promote",
        "reason": REASON,
        "idempotency_key": key,
        "evidence_uris": (),
        "supersedes_record_id": None,
    }
    arguments.update(changes)
    return _review_module().review_candidate(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        **arguments,
    )


def _prompt(runtime, prompt_envelope_factory, text: str, *, session: str = "session_a"):
    event = runtime.events.append(prompt_envelope_factory(session, text))
    return event, event_uri(session, event.event_id)


def _tool(runtime, tool_envelope_factory, name: str, *, session: str = "session_a"):
    event = runtime.events.append(
        tool_envelope_factory(session, f"tool-{name}", name)
    )
    return event, event_uri(session, event.event_id)


def _corrupt_event_blob(runtime, event_id: int) -> None:
    digest = runtime.connection.execute(
        "SELECT payload_blob_sha256 FROM events WHERE event_id = ?", (event_id,)
    ).fetchone()[0]
    path = runtime.paths.blobs_dir / "sha256" / digest[:2] / f"{digest}.bin"
    path.write_bytes(b"corrupt")


def _snapshot(connection: sqlite3.Connection):
    return {
        table: tuple(
            tuple(row)
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
        )
        for table in LEDGER_TABLES
    }


def _candidate_state(connection: sqlite3.Connection, candidate_id: str):
    return tuple(
        connection.execute(
            """
            SELECT state, permanent_record_id
            FROM memory_candidate_current
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()
    )


def test_exact_active_hash_links_without_changing_permanent_evidence(runtime) -> None:
    permanent_uri = "file:///permanent/original"
    capture_uri = "file:///candidate/capture"
    record = _permanent(
        runtime,
        key="duplicate-permanent",
        evidence=(permanent_uri,),
        value={"engine": "sqlite"},
    )
    candidate = _candidate(
        runtime,
        key="duplicate-candidate",
        evidence=(capture_uri,),
        value={"engine": "sqlite"},
    )
    before_record = tuple(
        runtime.connection.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record.record_id,)
        ).fetchone()
    )

    result = _review(runtime, candidate.candidate_id, key="duplicate-review")

    assert result.action == "promote"
    assert result.policy_code == "linked_duplicate"
    assert result.permanent_record_id == record.record_id
    assert result.permanent_record_created is False
    assert result.superseded_record_id is None
    assert _candidate_state(runtime.connection, candidate.candidate_id) == (
        "promoted",
        record.record_id,
    )
    assert tuple(
        row[0]
        for row in runtime.connection.execute(
            "SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal",
            (record.record_id,),
        )
    ) == (permanent_uri,)
    assert tuple(
        runtime.connection.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record.record_id,)
        ).fetchone()
    ) == before_record


def test_exact_decision_does_not_bypass_a_differing_active_conflict(runtime) -> None:
    exact = _permanent(runtime, key="exact-conflict-decision")
    _permanent(
        runtime,
        key="differing-conflict-decision",
        statement="Use PostgreSQL.",
        value={"engine": "postgresql"},
    )
    candidate = _candidate(
        runtime,
        key="exact-conflict-candidate",
    )

    result = _review(runtime, candidate.candidate_id, key="exact-conflict-review")

    assert result.action == "defer"
    assert result.policy_code == "decision_conflict"
    assert result.permanent_record_id is None
    assert runtime.connection.execute(
        "SELECT COUNT(*) FROM memory_current WHERE state = 'active'"
    ).fetchone()[0] == 2
    assert exact.record_id != result.permanent_record_id


def test_exact_and_differing_authoritative_target_fails_closed(runtime) -> None:
    _permanent(runtime, key="exact-target-conflict")
    differing = _permanent(
        runtime,
        key="differing-target-conflict",
        statement="Use PostgreSQL.",
        value={"engine": "postgresql"},
    )
    candidate = _candidate(runtime, key="exact-target-conflict-candidate")
    before = _snapshot(runtime.connection)

    with pytest.raises(ValidationError, match="exact authoritative duplicate"):
        _review(
            runtime,
            candidate.candidate_id,
            key="exact-target-conflict-review",
            supersedes_record_id=differing.record_id,
        )

    assert _snapshot(runtime.connection) == before


def test_exact_constitution_does_not_certify_while_a_differing_one_is_active(
    runtime, prompt_envelope_factory
) -> None:
    _permanent(
        runtime,
        key="exact-conflict-constitution",
        record_type="constitution",
        topic="policy.conflict",
        statement="Keep reviews local.",
    )
    _permanent(
        runtime,
        key="differing-conflict-constitution",
        record_type="constitution",
        topic="policy.conflict",
        statement="Send reviews remotely.",
    )
    _, prompt_uri = _prompt(
        runtime,
        prompt_envelope_factory,
        "Persist the local review policy.",
    )
    candidate = _candidate(
        runtime,
        key="exact-conflict-constitution-candidate",
        record_type="constitution",
        topic="policy.conflict",
        statement="Keep reviews local.",
        evidence=(prompt_uri,),
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key="exact-conflict-constitution-review",
    )

    assert result.action == "defer"
    assert result.policy_code == "constitution_conflict"
    assert result.permanent_record_id is None


def test_new_promotion_copies_only_capture_evidence_then_activates(runtime) -> None:
    capture = (
        "file:///capture/first",
        "https://example.test/capture/second",
        "file:///capture/first",
    )
    candidate = _candidate(
        runtime,
        key="new-candidate",
        topic="storage.new",
        statement="Use the new durable store.",
        value={"mode": "durable"},
        evidence=capture,
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key="new-review",
        evidence_uris=("file:///review/not-copied",),
    )

    assert result.action == "promote"
    assert result.policy_code == "promoted_new"
    assert result.permanent_record_created is True
    assert result.superseded_record_id is None
    assert tuple(
        row[0]
        for row in runtime.connection.execute(
            "SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal",
            (result.permanent_record_id,),
        )
    ) == capture
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?",
        (result.permanent_record_id,),
    ).fetchone()[0] == "active"
    assert tuple(
        row[0]
        for row in runtime.connection.execute(
            "SELECT idempotency_key FROM memory_events WHERE record_id = ? ORDER BY event_seq",
            (result.permanent_record_id,),
        )
    ) == (
        f"candidate:{candidate.candidate_id}:propose",
        f"candidate:{candidate.candidate_id}:activate",
    )
    assert verify_ledger(runtime.connection).records == 1


def test_differing_decision_without_target_defers_on_normalized_scope(runtime) -> None:
    target = _permanent(
        runtime,
        key="decision-conflict-target",
        statement="Use PostgreSQL.",
    )
    candidate = _candidate(
        runtime,
        key="decision-conflict-candidate",
        scope_key=MATCHING_SCOPE,
        statement="Use SQLite.",
    )

    result = _review(runtime, candidate.candidate_id, key="decision-conflict-review")

    assert result.action == "defer"
    assert result.requested_outcome == "promote"
    assert result.policy_code == "decision_conflict"
    assert result.permanent_record_id is None
    assert _candidate_state(runtime.connection, candidate.candidate_id) == (
        "temporary",
        None,
    )
    assert runtime.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?", (target.record_id,)
    ).fetchone()[0] == "active"


def test_multiple_authoritative_conflicts_defer_without_creating_third_active(runtime) -> None:
    _permanent(runtime, key="multiple-one", statement="Choose option one.")
    _permanent(runtime, key="multiple-two", statement="Choose option two.")
    candidate = _candidate(
        runtime,
        key="multiple-candidate",
        statement="Choose option three.",
    )

    result = _review(runtime, candidate.candidate_id, key="multiple-review")

    assert result.action == "defer"
    assert result.policy_code == "decision_conflict"
    assert runtime.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 2
    assert runtime.connection.execute(
        "SELECT COUNT(*) FROM memory_current WHERE state = 'active'"
    ).fetchone()[0] == 2


def test_explicit_one_of_multiple_authoritative_rows_is_not_a_sole_target(runtime) -> None:
    first = _permanent(runtime, key="multiple-target-one", statement="Option one.")
    _permanent(runtime, key="multiple-target-two", statement="Option two.")
    candidate = _candidate(
        runtime,
        key="multiple-target-candidate",
        statement="Option three.",
    )
    before = _snapshot(runtime.connection)

    with pytest.raises(ValidationError):
        _review(
            runtime,
            candidate.candidate_id,
            key="multiple-target-review",
            supersedes_record_id=first.record_id,
        )

    assert _snapshot(runtime.connection) == before


def test_explicit_target_must_be_the_sole_differing_active_row(runtime) -> None:
    target = _permanent(
        runtime,
        key="sole-target",
        statement="Use the prior engine.",
    )
    unrelated = _permanent(
        runtime,
        key="unrelated-target",
        topic="storage.unrelated",
        statement="Unrelated decision.",
    )
    candidate = _candidate(
        runtime,
        key="wrong-target-candidate",
        statement="Use the replacement engine.",
    )
    before = _snapshot(runtime.connection)

    with pytest.raises(ValidationError):
        _review(
            runtime,
            candidate.candidate_id,
            key="wrong-target-review",
            supersedes_record_id=unrelated.record_id,
        )

    assert _snapshot(runtime.connection) == before
    assert target.record_id != unrelated.record_id


def test_target_without_local_evidence_requires_a_current_local_review_event(
    runtime, prompt_envelope_factory
) -> None:
    target = _permanent(
        runtime,
        key="no-local-target",
        statement="Use the old engine.",
    )
    _, capture_uri = _prompt(runtime, prompt_envelope_factory, "capture evidence")
    candidate = _candidate(
        runtime,
        key="no-local-candidate",
        statement="Use the new engine.",
        evidence=(capture_uri,),
    )

    deferred = _review(
        runtime,
        candidate.candidate_id,
        key="no-local-defer",
        supersedes_record_id=target.record_id,
    )
    assert deferred.action == "defer"
    assert deferred.policy_code == "newer_evidence_required"

    _, review_uri = _prompt(runtime, prompt_envelope_factory, "review evidence")
    promoted = _review(
        runtime,
        candidate.candidate_id,
        key="no-local-promote",
        supersedes_record_id=target.record_id,
        evidence_uris=(review_uri,),
    )
    assert promoted.action == "promote"
    assert promoted.policy_code == "promoted_superseding"
    assert promoted.superseded_record_id == target.record_id
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?", (target.record_id,)
    ).fetchone()[0] == "superseded"
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?",
        (promoted.permanent_record_id,),
    ).fetchone()[0] == "active"
    assert runtime.connection.execute(
        "SELECT idempotency_key FROM memory_events WHERE record_id = ? AND action = 'supersede'",
        (target.record_id,),
    ).fetchone()[0] == f"candidate:{candidate.candidate_id}:supersede"


def test_newer_event_comparison_uses_numeric_event_id(runtime, prompt_envelope_factory) -> None:
    uris = [
        _prompt(runtime, prompt_envelope_factory, f"event {index}")[1]
        for index in range(1, 11)
    ]
    assert uris[8].endswith("/9")
    assert uris[9].endswith("/10")
    target = _permanent(
        runtime,
        key="numeric-target",
        statement="Use event nine policy.",
        evidence=(uris[8],),
    )
    candidate = _candidate(
        runtime,
        key="numeric-candidate",
        statement="Use event ten policy.",
        evidence=(uris[9],),
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key="numeric-review",
        supersedes_record_id=target.record_id,
    )

    assert result.action == "promote"
    assert result.policy_code == "promoted_superseding"


def test_newer_event_must_exceed_maximum_target_local_event_id(
    runtime, prompt_envelope_factory
) -> None:
    _, target_low = _prompt(runtime, prompt_envelope_factory, "target low")
    _, candidate_middle = _prompt(runtime, prompt_envelope_factory, "candidate middle")
    _, target_high = _prompt(runtime, prompt_envelope_factory, "target high")
    target = _permanent(
        runtime,
        key="max-target",
        statement="Use target evidence range.",
        evidence=(target_low, target_high),
    )
    candidate = _candidate(
        runtime,
        key="max-candidate",
        statement="Use candidate middle evidence.",
        evidence=(candidate_middle,),
    )

    deferred = _review(
        runtime,
        candidate.candidate_id,
        key="max-review-defer",
        supersedes_record_id=target.record_id,
    )
    assert deferred.action == "defer"
    assert deferred.policy_code == "newer_evidence_required"

    _, newest = _prompt(runtime, prompt_envelope_factory, "newest review evidence")
    promoted = _review(
        runtime,
        candidate.candidate_id,
        key="max-review-promote",
        supersedes_record_id=target.record_id,
        evidence_uris=(newest,),
    )
    assert promoted.action == "promote"
    assert promoted.policy_code == "promoted_superseding"


@pytest.mark.parametrize("source", ("none", "tool"))
def test_constitution_without_direct_user_event_defers(
    runtime, tool_envelope_factory, source
) -> None:
    evidence = ()
    if source == "tool":
        evidence = (_tool(runtime, tool_envelope_factory, "functions.exec")[1],)
    candidate = _candidate(
        runtime,
        key=f"constitution-no-user-{source}",
        record_type="constitution",
        topic="policy.user-authority",
        statement="Persist this governing policy.",
        evidence=evidence,
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key=f"constitution-no-user-review-{source}",
    )

    assert result.action == "defer"
    assert result.requested_outcome == "promote"
    assert result.policy_code == "newer_evidence_required"
    assert runtime.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0


def test_constitution_direct_review_event_promotes_new_record(
    runtime, prompt_envelope_factory
) -> None:
    candidate = _candidate(
        runtime,
        key="constitution-direct-review",
        record_type="constitution",
        topic="policy.direct-review",
        statement="Persist the direct user policy.",
    )
    _, uri = _prompt(runtime, prompt_envelope_factory, "Persist this policy")

    result = _review(
        runtime,
        candidate.candidate_id,
        key="constitution-direct-review-action",
        evidence_uris=(uri,),
    )

    assert result.action == "promote"
    assert result.policy_code == "promoted_new"
    assert result.permanent_record_created is True


def test_exact_constitution_link_certifies_without_rewriting_legacy_record(
    runtime, prompt_envelope_factory
) -> None:
    record = _permanent(
        runtime,
        key="legacy-constitution",
        record_type="constitution",
        topic="policy.legacy",
        statement="Retain this explicit policy.",
        value={"authority": "user"},
    )
    before = tuple(
        runtime.connection.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record.record_id,)
        ).fetchone()
    )
    _, uri = _prompt(runtime, prompt_envelope_factory, "Retain this policy")
    candidate = _candidate(
        runtime,
        key="legacy-constitution-candidate",
        record_type="constitution",
        topic="policy.legacy",
        statement="Retain this explicit policy.",
        value={"authority": "user"},
        evidence=(uri,),
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key="legacy-constitution-review",
    )

    assert result.policy_code == "linked_duplicate"
    assert result.permanent_record_id == record.record_id
    assert tuple(
        runtime.connection.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record.record_id,)
        ).fetchone()
    ) == before
    packet = import_module("codex_memory.memory_context").build_memory_packet(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        session_id="session_b",
        turn_id="turn_1",
        context_id="startup",
        scope_key=MATCHING_SCOPE,
        prompt_event_uri=None,
    )
    assert "CERTIFIED CONSTITUTION" in packet
    assert record.record_id in packet


def test_constitution_replacement_requires_newer_event_itself_to_be_user_prompt(
    runtime, prompt_envelope_factory, tool_envelope_factory
) -> None:
    _, old_user = _prompt(runtime, prompt_envelope_factory, "old direct instruction")
    _, target_evidence = _prompt(runtime, prompt_envelope_factory, "target instruction")
    _, newer_tool = _tool(runtime, tool_envelope_factory, "functions.exec")
    target = _permanent(
        runtime,
        key="constitution-replacement-target",
        record_type="constitution",
        topic="policy.replacement",
        statement="Keep the older policy.",
        evidence=(target_evidence,),
    )
    candidate = _candidate(
        runtime,
        key="constitution-replacement-candidate",
        record_type="constitution",
        topic="policy.replacement",
        statement="Use the replacement policy.",
        evidence=(old_user, newer_tool),
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key="constitution-replacement-review",
        supersedes_record_id=target.record_id,
    )

    assert result.action == "defer"
    assert result.policy_code == "newer_evidence_required"
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?", (target.record_id,)
    ).fetchone()[0] == "active"


def test_constitution_newer_user_prompt_supersedes_differing_active_record(
    runtime, prompt_envelope_factory
) -> None:
    _, target_uri = _prompt(runtime, prompt_envelope_factory, "older constitution")
    target = _permanent(
        runtime,
        key="constitution-positive-target",
        record_type="constitution",
        topic="policy.positive-replacement",
        statement="Keep the old explicit policy.",
        evidence=(target_uri,),
    )
    _, candidate_uri = _prompt(runtime, prompt_envelope_factory, "new constitution")
    candidate = _candidate(
        runtime,
        key="constitution-positive-candidate",
        record_type="constitution",
        topic="policy.positive-replacement",
        statement="Use the new explicit policy.",
        evidence=(candidate_uri,),
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key="constitution-positive-review",
        supersedes_record_id=target.record_id,
    )

    assert result.action == "promote"
    assert result.policy_code == "promoted_superseding"
    assert result.superseded_record_id == target.record_id


@pytest.mark.parametrize("record_type", ("constitution", "decision"))
@pytest.mark.parametrize("linked_evidence_location", ("capture", "review"))
def test_newer_highwater_includes_promoted_linked_candidate_evidence(
    runtime,
    prompt_envelope_factory,
    record_type,
    linked_evidence_location,
) -> None:
    first_event, old_uri = _prompt(runtime, prompt_envelope_factory, "old U1")
    target = _permanent(
        runtime,
        key=f"linked-highwater-target-{record_type}-{linked_evidence_location}",
        record_type=record_type,
        topic=f"policy.linked-highwater-{record_type}-{linked_evidence_location}",
        statement="Keep the certified target body.",
    )
    second_event, certified_uri = _prompt(
        runtime, prompt_envelope_factory, "certifying U2"
    )
    assert first_event.event_id < second_event.event_id
    certifying = _candidate(
        runtime,
        key=f"linked-highwater-certifying-{record_type}-{linked_evidence_location}",
        record_type=record_type,
        topic=f"policy.linked-highwater-{record_type}-{linked_evidence_location}",
        statement="Keep the certified target body.",
        evidence=(certified_uri,) if linked_evidence_location == "capture" else (),
    )
    linked = _review(
        runtime,
        certifying.candidate_id,
        key=f"linked-highwater-link-{record_type}-{linked_evidence_location}",
        evidence_uris=(certified_uri,)
        if linked_evidence_location == "review"
        else (),
    )
    assert linked.policy_code == "linked_duplicate"
    assert linked.permanent_record_id == target.record_id
    assert runtime.connection.execute(
        "SELECT COUNT(*) FROM memory_evidence WHERE record_id = ?", (target.record_id,)
    ).fetchone()[0] == 0

    replacement = _candidate(
        runtime,
        key=f"linked-highwater-replacement-{record_type}-{linked_evidence_location}",
        record_type=record_type,
        topic=f"policy.linked-highwater-{record_type}-{linked_evidence_location}",
        statement="Use a differing replacement body.",
    )
    stale = _review(
        runtime,
        replacement.candidate_id,
        key=f"linked-highwater-stale-{record_type}-{linked_evidence_location}",
        supersedes_record_id=target.record_id,
        evidence_uris=(old_uri,),
    )

    assert stale.action == "defer"
    assert stale.policy_code == "newer_evidence_required"
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id = ?", (target.record_id,)
    ).fetchone()[0] == "active"
    assert runtime.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1

    third_event, newest_uri = _prompt(runtime, prompt_envelope_factory, "newest U3")
    assert third_event.event_id > second_event.event_id
    promoted = _review(
        runtime,
        replacement.candidate_id,
        key=f"linked-highwater-newest-{record_type}-{linked_evidence_location}",
        supersedes_record_id=target.record_id,
        evidence_uris=(newest_uri,),
    )
    assert promoted.action == "promote"
    assert promoted.policy_code == "promoted_superseding"
    assert promoted.superseded_record_id == target.record_id


@pytest.mark.parametrize("record_type", ("constitution", "decision"))
@pytest.mark.parametrize("linked_evidence_location", ("capture", "review"))
def test_promoted_linked_candidate_highwater_evidence_is_blob_verified(
    runtime,
    prompt_envelope_factory,
    record_type,
    linked_evidence_location,
) -> None:
    event, evidence_uri = _prompt(
        runtime, prompt_envelope_factory, "certifying evidence"
    )
    target = _permanent(
        runtime,
        key=f"linked-integrity-target-{record_type}-{linked_evidence_location}",
        record_type=record_type,
        topic=f"policy.linked-integrity-{record_type}-{linked_evidence_location}",
        statement="Keep the certified target body.",
    )
    certifying = _candidate(
        runtime,
        key=f"linked-integrity-certifying-{record_type}-{linked_evidence_location}",
        record_type=record_type,
        topic=f"policy.linked-integrity-{record_type}-{linked_evidence_location}",
        statement="Keep the certified target body.",
        evidence=(evidence_uri,) if linked_evidence_location == "capture" else (),
    )
    linked = _review(
        runtime,
        certifying.candidate_id,
        key=f"linked-integrity-link-{record_type}-{linked_evidence_location}",
        evidence_uris=(evidence_uri,)
        if linked_evidence_location == "review"
        else (),
    )
    assert linked.policy_code == "linked_duplicate"
    assert linked.permanent_record_id == target.record_id

    _corrupt_event_blob(runtime, event.event_id)
    replacement = _candidate(
        runtime,
        key=f"linked-integrity-replacement-{record_type}-{linked_evidence_location}",
        record_type=record_type,
        topic=f"policy.linked-integrity-{record_type}-{linked_evidence_location}",
        statement="Use a differing replacement body.",
    )
    before = _snapshot(runtime.connection)

    with pytest.raises(VerificationError):
        _review(
            runtime,
            replacement.candidate_id,
            key=f"linked-integrity-review-{record_type}-{linked_evidence_location}",
            supersedes_record_id=target.record_id,
        )

    assert _snapshot(runtime.connection) == before
    assert runtime.connection.in_transaction is False


@pytest.mark.parametrize("record_type", ("assumption", "failure", "rejected_approach"))
def test_non_authoritative_distinct_records_coexist(runtime, record_type) -> None:
    _permanent(
        runtime,
        key=f"coexist-existing-{record_type}",
        record_type=record_type,
        topic="coexist.topic",
        statement="Existing observation.",
    )
    candidate = _candidate(
        runtime,
        key=f"coexist-candidate-{record_type}",
        record_type=record_type,
        topic="coexist.topic",
        statement="Distinct observation.",
    )

    result = _review(
        runtime,
        candidate.candidate_id,
        key=f"coexist-review-{record_type}",
    )

    assert result.action == "promote"
    assert result.policy_code == "promoted_new"
    assert runtime.connection.execute(
        """
        SELECT COUNT(*)
        FROM memory_records AS r
        JOIN memory_current AS c ON c.record_id = r.record_id
        WHERE r.record_type = ? AND r.topic_key = 'coexist.topic' AND c.state = 'active'
        """,
        (record_type,),
    ).fetchone()[0] == 2


@pytest.mark.parametrize(
    ("outcome", "action", "policy", "state"),
    (
        ("defer", "defer", "review_deferred", "temporary"),
        ("reject", "reject", "review_rejected", "rejected"),
        ("expire", "expire", "review_expired", "expired"),
    ),
)
def test_public_review_handles_non_promotion_outcomes(
    runtime, outcome, action, policy, state
) -> None:
    candidate = _candidate(runtime, key=f"non-promote-{outcome}")

    result = _review(
        runtime,
        candidate.candidate_id,
        key=f"non-promote-review-{outcome}",
        outcome=outcome,
        evidence_uris=("file:///review/evidence",),
    )

    assert result.action == action
    assert result.requested_outcome == outcome
    assert result.policy_code == policy
    assert result.state == state
    assert result.permanent_record_id is None
    assert runtime.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0


@pytest.mark.parametrize("outcome", ("defer", "reject", "expire"))
@pytest.mark.parametrize("damage", ("cross_session", "corrupt_blob"))
def test_non_promotion_review_evidence_is_same_session_and_blob_verified(
    runtime, prompt_envelope_factory, outcome, damage
) -> None:
    session = "session_b" if damage == "cross_session" else "session_a"
    event, uri = _prompt(runtime, prompt_envelope_factory, damage, session=session)
    candidate = _candidate(runtime, key=f"non-promotion-evidence-{outcome}-{damage}")
    if damage == "corrupt_blob":
        _corrupt_event_blob(runtime, event.event_id)
    before = _snapshot(runtime.connection)

    with pytest.raises(VerificationError):
        _review(
            runtime,
            candidate.candidate_id,
            key=f"non-promotion-review-{outcome}-{damage}",
            outcome=outcome,
            evidence_uris=(uri,),
        )

    assert _snapshot(runtime.connection) == before


def test_exact_review_retry_precedes_blob_revalidation_and_terminal_check(
    runtime, prompt_envelope_factory
) -> None:
    candidate = _candidate(runtime, key="retry-candidate")
    event, uri = _prompt(runtime, prompt_envelope_factory, "review once")
    first = _review(
        runtime,
        candidate.candidate_id,
        key="retry-review",
        evidence_uris=(uri,),
    )
    before = _snapshot(runtime.connection)
    _corrupt_event_blob(runtime, event.event_id)

    retry = _review(
        runtime,
        candidate.candidate_id,
        key="retry-review",
        evidence_uris=(uri,),
    )

    assert retry == first
    assert _snapshot(runtime.connection) == before


@pytest.mark.parametrize(
    "changes",
    (
        {"reason": "A different rationale."},
        {"evidence_uris": ("file:///different/evidence",)},
        {"outcome": "reject"},
        {"supersedes_record_id": "mem_different"},
    ),
)
def test_same_review_key_with_different_request_conflicts_before_terminal_policy(
    runtime, changes
) -> None:
    candidate = _candidate(runtime, key=f"retry-conflict-{next(iter(changes))}")
    first = _review(runtime, candidate.candidate_id, key="shared-review-key")
    before = _snapshot(runtime.connection)

    with pytest.raises(IdempotencyConflict):
        _review(
            runtime,
            candidate.candidate_id,
            key="shared-review-key",
            **changes,
        )

    assert first.state == "promoted"
    assert _snapshot(runtime.connection) == before


def test_same_review_key_for_another_candidate_conflicts(runtime) -> None:
    first_candidate = _candidate(
        runtime,
        key="shared-key-first-candidate",
        topic="shared.first",
    )
    second_candidate = _candidate(
        runtime,
        key="shared-key-second-candidate",
        topic="shared.second",
    )
    first = _review(runtime, first_candidate.candidate_id, key="cross-candidate-key")
    before = _snapshot(runtime.connection)

    with pytest.raises(IdempotencyConflict):
        _review(runtime, second_candidate.candidate_id, key="cross-candidate-key")

    assert first.state == "promoted"
    assert _snapshot(runtime.connection) == before


def test_deferred_supersession_retry_cannot_drop_original_target(runtime) -> None:
    target = _permanent(
        runtime,
        key="deferred-retry-target",
        statement="Keep the existing choice.",
    )
    candidate = _candidate(
        runtime,
        key="deferred-retry-candidate",
        statement="Replace the existing choice.",
    )
    first = _review(
        runtime,
        candidate.candidate_id,
        key="deferred-retry-review",
        supersedes_record_id=target.record_id,
    )
    assert first.action == "defer"
    assert first.policy_code == "newer_evidence_required"
    before = _snapshot(runtime.connection)

    with pytest.raises(IdempotencyConflict):
        _review(
            runtime,
            candidate.candidate_id,
            key="deferred-retry-review",
            supersedes_record_id=None,
        )

    assert _snapshot(runtime.connection) == before


def test_deferred_supersession_exact_retry_survives_later_promotion(
    runtime, prompt_envelope_factory
) -> None:
    target = _permanent(
        runtime,
        key="deferred-terminal-target",
        statement="Keep the terminal target.",
    )
    candidate = _candidate(
        runtime,
        key="deferred-terminal-candidate",
        statement="Replace the terminal target.",
    )
    deferred = _review(
        runtime,
        candidate.candidate_id,
        key="deferred-terminal-first",
        supersedes_record_id=target.record_id,
    )
    _, newer = _prompt(runtime, prompt_envelope_factory, "newer terminal evidence")
    promoted = _review(
        runtime,
        candidate.candidate_id,
        key="deferred-terminal-promote",
        supersedes_record_id=target.record_id,
        evidence_uris=(newer,),
    )
    before = _snapshot(runtime.connection)

    retry = _review(
        runtime,
        candidate.candidate_id,
        key="deferred-terminal-first",
        supersedes_record_id=target.record_id,
    )

    assert promoted.state == "promoted"
    assert retry == deferred
    assert _snapshot(runtime.connection) == before


def test_deferred_target_retry_uses_original_immutable_conflict_not_new_active_row(
    runtime,
) -> None:
    original = _permanent(
        runtime,
        key="immutable-retry-target-a",
        statement="Original target A.",
    )
    candidate = _candidate(
        runtime,
        key="immutable-retry-candidate",
        statement="Candidate replacement.",
    )
    deferred = _review(
        runtime,
        candidate.candidate_id,
        key="immutable-retry-key",
        supersedes_record_id=original.record_id,
    )
    immediate_retry = _review(
        runtime,
        candidate.candidate_id,
        key="immutable-retry-key",
        supersedes_record_id=original.record_id,
    )
    assert immediate_retry == deferred

    apply_action(
        runtime.connection,
        record_id=original.record_id,
        action="close",
        idempotency_key="immutable-retry-close-a",
    )
    replacement = _permanent(
        runtime,
        key="immutable-retry-target-b",
        statement="Later target B.",
    )
    before = _snapshot(runtime.connection)

    with pytest.raises(IdempotencyConflict):
        _review(
            runtime,
            candidate.candidate_id,
            key="immutable-retry-key",
            supersedes_record_id=replacement.record_id,
        )

    original_retry = _review(
        runtime,
        candidate.candidate_id,
        key="immutable-retry-key",
        supersedes_record_id=original.record_id,
    )
    assert original_retry == deferred
    assert _snapshot(runtime.connection) == before


def test_non_authoritative_candidate_rejects_supersedes_target(runtime) -> None:
    target = _permanent(
        runtime,
        key="assumption-target",
        record_type="assumption",
        topic="assumption.target",
        statement="Existing assumption.",
    )
    candidate = _candidate(
        runtime,
        key="assumption-target-candidate",
        record_type="assumption",
        topic="assumption.target",
        statement="Different assumption.",
    )
    before = _snapshot(runtime.connection)

    with pytest.raises(ValidationError):
        _review(
            runtime,
            candidate.candidate_id,
            key="assumption-target-review",
            supersedes_record_id=target.record_id,
        )

    assert _snapshot(runtime.connection) == before


def test_exact_duplicate_rejects_unrelated_supersedes_target(runtime) -> None:
    duplicate = _permanent(runtime, key="duplicate-with-target")
    unrelated = _permanent(
        runtime,
        key="duplicate-unrelated-target",
        topic="duplicate.unrelated",
        statement="Unrelated row.",
    )
    candidate = _candidate(runtime, key="duplicate-with-target-candidate")
    before = _snapshot(runtime.connection)

    with pytest.raises(ValidationError):
        _review(
            runtime,
            candidate.candidate_id,
            key="duplicate-with-target-review",
            supersedes_record_id=unrelated.record_id,
        )

    assert duplicate.record_id != unrelated.record_id
    assert _snapshot(runtime.connection) == before


@pytest.mark.parametrize(
    ("location", "damage"),
    (
        ("review", "cross_session"),
        ("capture", "corrupt_blob"),
        ("review", "corrupt_blob"),
    ),
)
def test_all_local_capture_and_review_evidence_is_identity_and_blob_verified(
    runtime, prompt_envelope_factory, location, damage
) -> None:
    session = "session_b" if damage == "cross_session" else "session_a"
    event, uri = _prompt(runtime, prompt_envelope_factory, damage, session=session)
    candidate = _candidate(
        runtime,
        key=f"evidence-{location}-{damage}",
        evidence=(uri,) if location == "capture" else (),
    )
    if damage == "corrupt_blob":
        _corrupt_event_blob(runtime, event.event_id)
    before = _snapshot(runtime.connection)

    with pytest.raises(VerificationError):
        _review(
            runtime,
            candidate.candidate_id,
            key=f"evidence-review-{location}-{damage}",
            evidence_uris=(uri,) if location == "review" else (),
        )

    assert _snapshot(runtime.connection) == before
    assert runtime.connection.in_transaction is False


def test_review_uses_exactly_one_immediate_transaction(runtime) -> None:
    candidate = _candidate(runtime, key="one-transaction-candidate")
    traced: list[str] = []
    runtime.connection.set_trace_callback(traced.append)
    try:
        _review(runtime, candidate.candidate_id, key="one-transaction-review")
    finally:
        runtime.connection.set_trace_callback(None)

    begins = [
        statement.strip().upper()
        for statement in traced
        if statement.strip().upper().startswith("BEGIN")
    ]
    assert begins == ["BEGIN IMMEDIATE"]


@pytest.mark.parametrize(
    "failure_point",
    (
        "after_permanent_propose",
        "after_permanent_activate",
        "after_permanent_supersede",
        "after_candidate_promote_event",
        "after_candidate_projection_update",
    ),
)
def test_failure_at_each_promotion_stage_rolls_back_every_ledger_table(
    runtime, prompt_envelope_factory, failure_point
) -> None:
    target = None
    review_evidence = ()
    if failure_point == "after_permanent_supersede":
        target = _permanent(
            runtime,
            key="fault-target",
            statement="Old fault policy.",
        )
        review_evidence = (
            _prompt(runtime, prompt_envelope_factory, "newer fault evidence")[1],
        )
    candidate = _candidate(
        runtime,
        key=f"fault-candidate-{failure_point}",
        statement="New fault policy.",
    )
    before = _snapshot(runtime.connection)

    if failure_point == "after_permanent_propose":
        runtime.connection.execute(
            """
            CREATE TEMP TRIGGER task7_fault
            BEFORE INSERT ON memory_events
            WHEN NEW.action = 'activate'
            BEGIN SELECT RAISE(FAIL, 'after permanent propose'); END
            """
        )
    elif failure_point in {"after_permanent_activate", "after_permanent_supersede"}:
        runtime.connection.execute(
            """
            CREATE TEMP TRIGGER task7_fault
            BEFORE INSERT ON memory_candidate_events
            WHEN NEW.action = 'promote'
            BEGIN SELECT RAISE(FAIL, 'before candidate promote'); END
            """
        )
    elif failure_point == "after_candidate_promote_event":
        runtime.connection.execute(
            """
            CREATE TEMP TRIGGER task7_fault
            BEFORE UPDATE ON memory_candidate_current
            WHEN NEW.state = 'promoted'
            BEGIN SELECT RAISE(FAIL, 'after candidate promote event'); END
            """
        )
    else:
        runtime.connection.execute(
            """
            CREATE TEMP TRIGGER task7_fault
            AFTER UPDATE ON memory_candidate_current
            WHEN NEW.state = 'promoted' AND NEW.updated_at <> 'task7-corrupt'
            BEGIN
                UPDATE memory_candidate_current
                SET updated_at = 'task7-corrupt'
                WHERE candidate_id = NEW.candidate_id;
            END
            """
        )

    with pytest.raises(LedgerError):
        _review(
            runtime,
            candidate.candidate_id,
            key=f"fault-review-{failure_point}",
            supersedes_record_id=target.record_id if target is not None else None,
            evidence_uris=review_evidence,
        )

    assert runtime.connection.in_transaction is False
    assert _snapshot(runtime.connection) == before


def test_two_connections_same_review_key_create_one_identical_result(
    tmp_path: Path,
) -> None:
    database = import_module("codex_memory.database")
    paths = RuntimePaths.for_repo(tmp_path)
    setup = database.connect_database(paths.db_path)
    database.migrate_database(setup)
    candidate = import_module("codex_memory.candidate_memory").capture_candidate(
        setup,
        BlobStore(paths.blobs_dir),
        project_scope_key=SCOPE,
        record_type="decision",
        scope_kind="project",
        scope_key=SCOPE,
        topic_key="concurrency.review",
        statement="Create one permanent result.",
        session_id="session_a",
        turn_id="turn_1",
        context_id="startup",
        idempotency_key="concurrent-candidate",
    )
    setup.close()
    barrier = Barrier(2)
    results = []
    failures = []

    def worker() -> None:
        connection = database.connect_database(paths.db_path)
        try:
            barrier.wait()
            results.append(
                _review_module().review_candidate(
                    connection,
                    BlobStore(paths.blobs_dir),
                    candidate_id=candidate.candidate_id,
                    outcome="promote",
                    reason=REASON,
                    idempotency_key="concurrent-review",
                )
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            connection.close()

    threads = [Thread(target=worker), Thread(target=worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert failures == []
    assert len(results) == 2
    assert results[0] == results[1]
    check = database.connect_database(paths.db_path)
    try:
        assert check.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
        assert check.execute(
            "SELECT COUNT(*) FROM memory_candidate_events WHERE action = 'promote'"
        ).fetchone()[0] == 1
        assert check.execute(
            "SELECT COUNT(DISTINCT permanent_record_id) FROM memory_candidate_current"
        ).fetchone()[0] == 1
        assert tuple(
            row[0]
            for row in check.execute(
                "SELECT idempotency_key FROM memory_events ORDER BY event_seq"
            )
        ) == (
            f"candidate:{candidate.candidate_id}:propose",
            f"candidate:{candidate.candidate_id}:activate",
        )
    finally:
        check.close()
