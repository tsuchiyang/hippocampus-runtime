from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import apply_candidate_action, capture_candidate
from codex_memory.hook_input import parse_hook_payload
from codex_memory.injection_store import InjectionPaths, InjectionStore
from codex_memory.memory_evidence import event_uri
from codex_memory.retrieval import cache_database, isolated_packet
from codex_memory.retrieval.contracts import (
    CandidateChannel, DegradedReason, RetrievalRequest, RetrievalScope, RetrievalStatus,
)
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
from codex_memory.retrieval.service import RetrievalService, SemanticQueryInputs
from test_p26c_selected_host import selected_host_lab  # fixture dependency
from test_p26d_compact import compact_lab, invoke
from test_p26d_promotion_race import _memory_rows
from test_retrieval_semantic_acceptance import _historical_generation
from test_retrieval_semantic_index import _active_semantic_metadata


@pytest.mark.parametrize("leaf", ["semantic.sqlite3", "generation-manifest.json"])
def test_missing_active_semantic_leaf_keeps_verified_deterministic_results(
    runtime, monkeypatch, leaf,
):
    """Missing real activated semantic files cannot feed unchecked vectors."""
    paths, provider, record = _historical_generation(runtime, monkeypatch)
    active_id = _active_semantic_metadata(paths)[0]
    generation = SemanticGenerationPaths.for_generation(paths.generations_dir, active_id)
    request = RetrievalRequest("semantic", "s", "t", "startup", None,
        RetrievalScope(str(paths.repo_root)))
    values = SemanticQueryInputs((1.0, 0.0, 0.0, 0.0), (("semantic", 1.0),))
    service = RetrievalService(paths)
    before = service.query(request, semantic_query=values)
    trace = next(item for item in before.trace if item.candidate_id == record.record_id)
    assert CandidateChannel.DENSE in trace.source_channels
    assert CandidateChannel.SPARSE in trace.source_channels
    assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE not in before.degraded_reasons
    canonical_before = _memory_rows(runtime.connection)
    calls_before = dict(provider.call_counts)
    source = generation.directory / leaf
    assert source.is_file() and source.resolve().is_relative_to(paths.runtime_root.resolve())
    source.rename(generation.directory / (leaf + ".held-for-inspection"))
    assert not source.exists()

    after = service.query(request, semantic_query=values)
    assert after.status is RetrievalStatus.CURRENT and not after.index_stale
    assert DegradedReason.SEMANTIC_INDEX_UNAVAILABLE in after.degraded_reasons
    trace = next(item for item in after.trace if item.candidate_id == record.record_id)
    assert CandidateChannel.DENSE not in trace.source_channels
    assert CandidateChannel.SPARSE not in trace.source_channels
    assert trace.source_channels  # actual deterministic candidates still participate
    assert _active_semantic_metadata(paths)[0] == active_id
    assert _memory_rows(runtime.connection) == canonical_before
    assert provider.call_counts == calls_before
    assert not source.exists(), "query must not silently recreate the missing file"


def test_selected_compact_without_fts_emits_audited_other_channels(compact_lab, monkeypatch):
    """An unavailable BM25 channel must not disable a healthy deterministic packet."""
    runtime, _marker, _hooks = compact_lab
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    before = _memory_rows(runtime.connection)
    paths.retrieval_root.rename(runtime.paths.runtime_root / "original-fts-cache-held")
    probes = []

    def unavailable(connection):
        probes.append(True)
        return False

    # Capability seam only; actual schema creation, query and output remain real.
    monkeypatch.setattr(cache_database, "_supports_fts5", unavailable)
    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT
    assert probes
    with closing(sqlite3.connect(paths.database_path)) as cache:
        assert not cache.execute(
            "SELECT 1 FROM sqlite_master WHERE name='retrieval_documents_fts'").fetchone()
    actual_query = RetrievalService.query
    observed = []

    def query(self, request, **kwargs):
        result = actual_query(self, request, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(RetrievalService, "query", query)
    rc, output, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="without-fts", prompt="verification code")
    assert rc == 0, err
    assert len(observed) == 1
    result = observed[0]
    assert result.status is RetrievalStatus.CURRENT and not result.index_stale
    assert DegradedReason.FTS5_UNAVAILABLE in result.degraded_reasons
    assert result.candidates and any(item.source_channels for item in result.trace)
    assert all(CandidateChannel.BM25 not in item.source_channels for item in result.trace)
    metadata = json.loads(err)
    assert metadata["delivery"] == "emitted" and metadata["cleanup_status"] == "complete"
    store = InjectionStore(InjectionPaths.for_repo(paths.repo_root), BlobStore(runtime.paths.blobs_dir))
    verified = store.load(metadata["injection_id"])
    assert verified.serialized_response_bytes == output.encode("utf-8")
    packet = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert "HOST-P26C-EXAMPLE" in packet and len(packet) <= 4000
    assert _memory_rows(runtime.connection) == before


@pytest.mark.parametrize("action,state", [("reject", "rejected"), ("expire", "expired")])
def test_selected_compact_candidate_lifecycle_after_query_uses_fresh_legacy(
    compact_lab, monkeypatch, action, state,
):
    """A temporary candidate removed after selection must not leak into output."""
    runtime, _marker, _hooks = compact_lab
    scope = str(runtime.paths.repo_root)
    statement = "TEMPORARY-P26D-REMOVED is an unverified synthetic handoff note."
    event = runtime.events.append(parse_hook_payload({
        "hook_event_name": "UserPromptSubmit", "session_id": "selected-session",
        "turn_id": "capture", "cwd": scope, "prompt": statement,
    }))
    evidence = event_uri("selected-session", event.event_id)
    candidate = capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir),
        project_scope_key=scope, record_type="decision", scope_kind="project", scope_key=scope,
        topic_key="p26d.temporary-lifecycle", statement=statement, session_id="selected-session",
        turn_id="capture", context_id="startup", idempotency_key="capture-lifecycle",
        evidence_uris=(evidence,))
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild().status is RetrievalStatus.CURRENT
    rc, output, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="before", prompt="synthetic handoff note")
    assert rc == 0 and json.loads(err)["delivery"] == "emitted", err
    assert statement in output and "UNVERIFIED TEMPORARY" in output
    root = runtime.paths.runtime_root / "injections"
    manifests = {path.relative_to(root): path.read_bytes() for path in root.glob("inj_*/manifest.json")}
    actual_query = RetrievalService.query
    actual_assemble = isolated_packet.build_phase2_packet
    states_after_action, observed_errors = [], []

    def query_then_review(self, request, **kwargs):
        result = actual_query(self, request, **kwargs)
        assert candidate.candidate_id in {item.candidate_id for item in result.candidates}
        changed = apply_candidate_action(runtime.connection, BlobStore(runtime.paths.blobs_dir),
            candidate_id=candidate.candidate_id, action=action,
            reason="Explicit synthetic lifecycle change.", idempotency_key="review-lifecycle",
            evidence_uris=(evidence,))
        assert changed.state == state
        states_after_action.append(_memory_rows(runtime.connection))
        return result

    def assemble(*args, **kwargs):
        try:
            return actual_assemble(*args, **kwargs)
        except ValueError as error:
            observed_errors.append(str(error))
            raise

    monkeypatch.setattr(RetrievalService, "query", query_then_review)
    monkeypatch.setattr(isolated_packet, "build_phase2_packet", assemble)
    rc, output, err = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id="after", prompt="synthetic handoff note")
    assert rc == 0, err
    assert len(states_after_action) == 1
    assert observed_errors == ["retrieval result does not match canonical snapshot"]
    metadata = json.loads(err)
    assert metadata["delivery"] == "legacy_fallback" and metadata["injection_id"] is None
    assert metadata["cleanup_status"] == "complete"
    packet = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert "TEMPORARY-P26D-REMOVED" not in packet and "HOST-P26C-EXAMPLE" in packet
    assert len(packet) <= 4000
    assert _memory_rows(runtime.connection) == states_after_action[0]
    assert {path.relative_to(root): path.read_bytes() for path in root.glob("inj_*/manifest.json")} == manifests
