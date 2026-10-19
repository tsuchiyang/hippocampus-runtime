from __future__ import annotations

from codex_memory.blob_store import BlobStore
from codex_memory.memory_evidence import event_uri
from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
from codex_memory.structured_ledger import apply_action, propose_record


def _request() -> RetrievalRequest:
    return RetrievalRequest(None, "session-a", "turn-a", "context-a", None, RetrievalScope("repo"))


def test_mandatory_keeps_only_certified_constitution_and_temporary_direct_read_never_writes_cache(runtime) -> None:
    """Would fail if unverified Constitution elevated or temporary entered retrieval DB."""
    from codex_memory.retrieval.mandatory import mandatory_candidates, temporary_candidates
    from codex_memory.candidate_memory import capture_candidate, apply_candidate_action_in_transaction

    certified = propose_record(runtime.connection, record_type="constitution", scope_kind="project", scope_key="repo", topic_key="cert", statement="Certified", value=None, evidence_uris=(), idempotency_key="cert")
    apply_action(runtime.connection, record_id=certified.record_id, action="activate", idempotency_key="cert-active")
    unverified = propose_record(runtime.connection, record_type="constitution", scope_kind="project", scope_key="repo", topic_key="uncert", statement="Unverified", value=None, evidence_uris=(), idempotency_key="uncert")
    apply_action(runtime.connection, record_id=unverified.record_id, action="activate", idempotency_key="uncert-active")
    runtime.events.append(__import__("codex_memory.hook_input", fromlist=["parse_hook_payload"]).parse_hook_payload({"hook_event_name":"UserPromptSubmit","session_id":"session-a","turn_id":"turn-a","cwd":".","prompt":"certify"}))
    candidate = capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key="repo", record_type="constitution", scope_kind="project", scope_key="repo", topic_key="cert", statement="Certified", session_id="session-a", turn_id="turn-a", context_id="context-a", idempotency_key="cert-candidate", evidence_uris=(event_uri("session-a", 1),))
    runtime.connection.execute("BEGIN")
    apply_candidate_action_in_transaction(runtime.connection, candidate_id=candidate.candidate_id, action="promote", requested_outcome="promote", reason="review", idempotency_key="cert-promote", evidence_uris=(), policy_code="linked_duplicate", permanent_record_id=certified.record_id, permanent_record_created=False)
    runtime.connection.commit()
    temporary = capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key="repo", record_type="decision", scope_kind="project", scope_key="repo", topic_key="temp", statement="Same session", session_id="session-a", turn_id="turn-a", context_id="context-a", idempotency_key="temp")
    assert mandatory_candidates(runtime.connection, BlobStore(runtime.paths.blobs_dir), _request()).ids == (certified.record_id,)
    assert temporary_candidates(runtime.connection, BlobStore(runtime.paths.blobs_dir), _request()).ids == (temporary.candidate_id,)
    assert runtime.connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'retrieval_documents'").fetchone() is None
