from __future__ import annotations

import json
import threading
import time

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import capture_candidate
from codex_memory.database import connect_database
from codex_memory.hook_input import parse_hook_payload
from codex_memory.memory_evidence import event_uri
from codex_memory.memory_review import review_candidate
from codex_memory.retrieval import config, isolated_packet
from codex_memory.retrieval.canonical_snapshot import capture_canonical_snapshot
from codex_memory.retrieval.errors import RetrievalValidationError
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from test_p26c_selected_host import selected_host_lab  # fixture dependency
from test_p26d_compact import compact_lab, invoke


def _memory_rows(connection):
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memory_%' ORDER BY name")]
    return {table: tuple(tuple(row) for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
        for table in tables}


@pytest.mark.parametrize("promotion_at", ["before_query", "before_assembly"])
def test_selected_compact_concurrent_promotion_keeps_fresh_legacy(
    compact_lab, monkeypatch, promotion_at,
):
    """Catch stale emission or reuse of Legacy captured before a real promotion.

    A second connection/thread performs the legal review transaction at a
    deterministic boundary; query, assembly, output and storage stay real.
    This is not a host/process timing or probabilistic race certification.
    """
    runtime, marker, _hooks = compact_lab
    assert marker["provider_policy"] == "disabled"
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    scope = str(runtime.paths.repo_root)
    statement = "The approved synthetic handoff marker is PROMOTED-P26D-EXAMPLE."
    event = runtime.events.append(parse_hook_payload({
        "hook_event_name": "UserPromptSubmit", "session_id": "review-session",
        "turn_id": "review-turn", "cwd": scope, "prompt": statement,
    }))
    evidence = event_uri("review-session", event.event_id)
    candidate = capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir),
        project_scope_key=scope, record_type="decision", scope_kind="project", scope_key=scope,
        topic_key="p26d.promotion-race", statement=statement, session_id="review-session",
        turn_id="review-turn", context_id="startup", idempotency_key="capture-promotion-race",
        evidence_uris=(evidence,))
    assert RetrievalService(paths).rebuild().status.value == "current"

    def forbidden_provider(*args, **kwargs):
        pytest.fail("selected compact must not construct a provider")

    monkeypatch.setattr(config, "provider_from_config", forbidden_provider)
    rc, before_output, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="before", prompt="synthetic handoff marker")
    assert rc == 0 and json.loads(err)["delivery"] == "emitted", err
    # A different session's temporary candidate is not visible before promotion.
    assert "PROMOTED-P26D-EXAMPLE" not in before_output
    before_snapshot = capture_canonical_snapshot(paths)
    before_memory = _memory_rows(runtime.connection)
    manifest_root = runtime.paths.runtime_root / "injections"
    saved_manifests = {path.relative_to(manifest_root): path.read_bytes()
        for path in manifest_root.glob("inj_*/manifest.json")}
    assert len(saved_manifests) == 1

    start_review, finished, cancel = threading.Event(), threading.Event(), threading.Event()
    reviewed, memory_after_review, worker_errors, stages = [], [], [], []

    def promote():
        connection = None
        try:
            if not start_review.wait(5) or cancel.is_set():
                return
            connection = connect_database(runtime.paths.db_path, busy_timeout_ms=100)
            reviewed.append(review_candidate(connection, BlobStore(runtime.paths.blobs_dir),
                candidate_id=candidate.candidate_id, outcome="promote",
                reason="The synthetic decision is explicitly approved.",
                idempotency_key="review-promotion-race", evidence_uris=(evidence,)))
            memory_after_review.append(_memory_rows(connection))
            stages.append("promotion_committed")
        except BaseException as error:
            worker_errors.append(error)
        finally:
            if connection is not None:
                connection.close()
            finished.set()

    def complete_review_within(deadline):
        assert not start_review.is_set(), "the fixture must promote exactly once"
        start_review.set()
        # Consume the actual Hook's remaining budget, never grant a fresh one.
        remaining = max(0, deadline.deadline_ns - time.monotonic_ns()) / 1_000_000_000
        assert finished.wait(remaining), "promotion did not finish within the existing deadline"
        assert not worker_errors, worker_errors
        assert len(reviewed) == 1 and reviewed[0].state == "promoted"

    actual_query = RetrievalService.query
    actual_assemble = isolated_packet.build_phase2_packet
    queries, assembly_errors = [], []

    def query(self, request, **kwargs):
        stages.append("query_start")
        if promotion_at == "before_query":
            complete_review_within(kwargs["deadline"])
        result = actual_query(self, request, **kwargs)
        queries.append(result)
        stages.append("query_complete")
        return result

    def assemble(*args, **kwargs):
        assert promotion_at == "before_assembly"
        complete_review_within(kwargs["deadline"])
        try:
            return actual_assemble(*args, **kwargs)
        except RetrievalValidationError as error:
            assembly_errors.append(str(error))
            stages.append("assembly_rejected")
            raise

    monkeypatch.setattr(RetrievalService, "query", query)
    monkeypatch.setattr(isolated_packet, "build_phase2_packet", assemble)
    worker = threading.Thread(target=promote, name="p26d-synthetic-promotion", daemon=True)
    worker.start()
    try:
        rc, output, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
            turn_id="during-promotion", prompt="synthetic handoff marker")
    finally:
        cancel.set()
        start_review.set()
        worker.join(5)  # test cleanup only; never extends the Hook deadline
    assert not worker.is_alive()
    assert finished.is_set() and not worker_errors, worker_errors
    assert len(reviewed) == len(memory_after_review) == len(queries) == 1
    record_id = reviewed[0].permanent_record_id
    assert record_id is not None and reviewed[0].permanent_record_created
    assert runtime.connection.execute(
        "SELECT state FROM memory_current WHERE record_id=?", (record_id,)).fetchone()[0] == "active"
    assert tuple(runtime.connection.execute(
        "SELECT state, permanent_record_id FROM memory_candidate_current WHERE candidate_id=?",
        (candidate.candidate_id,)).fetchone()) == ("promoted", record_id)

    after_snapshot = capture_canonical_snapshot(paths)
    assert after_snapshot.highwater != before_snapshot.highwater
    result = queries[0]
    assert result.indexed_highwater == before_snapshot.highwater
    if promotion_at == "before_query":
        assert stages == ["query_start", "promotion_committed", "query_complete"]
        assert result.status.value == "stale" and result.index_stale
        assert result.canonical_highwater == after_snapshot.highwater
        assert record_id in {item.candidate_id for item in result.candidates}
        assert assembly_errors == []
        expected_reason = "retrieval_degraded"
    else:
        assert stages == ["query_start", "query_complete", "promotion_committed", "assembly_rejected"]
        assert result.status.value == "current" and not result.index_stale
        assert result.canonical_highwater == before_snapshot.highwater
        assert record_id not in {item.candidate_id for item in result.candidates}
        assert assembly_errors == ["retrieval result does not match canonical snapshot"]
        # Preserve the existing public fallback category; no new enum is invented.
        expected_reason = "manifest_unavailable"

    assert rc == 0, err
    metadata = json.loads(err)
    assert metadata["delivery"] == "legacy_fallback"
    assert metadata["reason"] == expected_reason
    assert metadata["cleanup_status"] == "complete"
    assert metadata["injection_id"] is None and metadata["manifest_sha256"] is None
    packet = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert statement in packet and "HOST-P26C-EXAMPLE" in packet and len(packet) <= 4000
    assert {path.relative_to(manifest_root): path.read_bytes()
        for path in manifest_root.glob("inj_*/manifest.json")} == saved_manifests
    assert _memory_rows(runtime.connection) == memory_after_review[0]
    # Existing immutable history remains a prefix; only the real promotion adds rows.
    for table in ("memory_records", "memory_events", "memory_evidence", "memory_relations",
                  "memory_candidates", "memory_candidate_events", "memory_candidate_evidence"):
        assert memory_after_review[0][table][:len(before_memory[table])] == before_memory[table]
    assert [tuple(row) for row in runtime.connection.execute(
        "SELECT session_id, event_name FROM events ORDER BY event_id")] == [
        ("review-session", "UserPromptSubmit"), ("selected-session", "UserPromptSubmit"),
        ("selected-session", "UserPromptSubmit")]
    status = RetrievalService(paths).status()
    assert status.status.value == "stale" and status.indexed_highwater == before_snapshot.highwater
