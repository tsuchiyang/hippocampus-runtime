from __future__ import annotations

import hashlib

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.structured_ledger import apply_action, propose_record
from codex_memory.memory_evidence import event_uri


@pytest.fixture
def frozen_clock(monkeypatch):
    values = iter(f"2026-01-01T00:00:{second:02d}.000000Z" for second in range(1, 60))
    def next_timestamp(_connection):
        return next(values)
    monkeypatch.setattr("codex_memory.structured_ledger.next_memory_timestamp", next_timestamp)
    monkeypatch.setattr("codex_memory.candidate_memory.next_memory_timestamp", next_timestamp)
    class Identifier:
        def __init__(self, value): self.hex = value
    identifiers = iter(f"{value:032x}" for value in range(1, 60))
    monkeypatch.setattr("codex_memory.structured_ledger.uuid.uuid4", lambda: Identifier(next(identifiers)))
    monkeypatch.setattr("codex_memory.candidate_memory.uuid.uuid4", lambda: Identifier(next(identifiers)))


def _packet(runtime, *, session: str = "session-a", query: str | None = None) -> str:
    from codex_memory.memory_context import build_memory_packet
    return build_memory_packet(runtime.connection, BlobStore(runtime.paths.blobs_dir), session_id=session, turn_id="turn-a", context_id="startup", scope_key="repo", prompt_event_uri=None, query=query)


def _record(runtime, *, key: str, kind: str = "decision", statement: str = "ordinary", scope: str = "repo"):
    record = propose_record(runtime.connection, record_type=kind, scope_kind="project", scope_key=scope, topic_key=key, statement=statement, value=None, evidence_uris=(), idempotency_key=key)
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key=key + "-active")
    return record


def _temporary(runtime, *, session: str = "session-a"):
    from codex_memory.candidate_memory import capture_candidate
    return capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key="repo", record_type="decision", scope_kind="project", scope_key="repo", topic_key="temp", statement="temporary", session_id=session, turn_id="turn-a", context_id="startup", idempotency_key="temporary")


def _certified(runtime) -> None:
    from codex_memory.candidate_memory import capture_candidate, apply_candidate_action_in_transaction
    from codex_memory.hook_input import parse_hook_payload
    record = _record(runtime, key="constitution", kind="constitution", statement="constitution")
    runtime.events.append(parse_hook_payload({"hook_event_name": "UserPromptSubmit", "session_id": "session-a", "turn_id": "turn-a", "cwd": ".", "prompt": "direct certification"}))
    candidate = capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir), project_scope_key="repo", record_type="constitution", scope_kind="project", scope_key="repo", topic_key="constitution", statement="constitution", session_id="session-a", turn_id="turn-a", context_id="startup", idempotency_key="constitution-candidate", evidence_uris=(event_uri("session-a", 1),))
    runtime.connection.execute("BEGIN")
    apply_candidate_action_in_transaction(runtime.connection, candidate_id=candidate.candidate_id, action="promote", requested_outcome="promote", reason="review", idempotency_key="constitution-promote", evidence_uris=(), policy_code="linked_duplicate", permanent_record_id=record.record_id, permanent_record_created=False)
    runtime.connection.commit()


@pytest.mark.parametrize(("scenario", "length", "digest"), (("empty", 1196, "e90467c38ec429818e2c9346982a3a288dc83e89a3d89698a2ede88986462982"), ("certified", 1458, "7f2e42100571ac8e997cee3c26fc01dde2c250c72633aa92f53fa7abebb5973f"), ("unverified", 1428, "a5e60352f67c2aff6713601286c79f6c72035ddd7160d69c4e7c0492665efe95"), ("ordinary", 1410, "a1dc46932ce84c6c9fb9532eac93fc1a2063128737c5cc8bbcd9c8f295ce0134"), ("unsafe", 1196, "b9de432ffc342f2a7ca34066d89f43b8699865863baf0e1cb9100b619acd8654"), ("temporary", 1418, "0110103fa6735733f1e6c0f1e47d2a161ea107b536930924da145b7524c9af50")))
def test_authority_packet_matches_the_frozen_phase1b_packet_for_every_authority_case(runtime, frozen_clock, scenario: str, length: int, digest: str) -> None:
    """Would fail if extraction changes any legacy packet byte or authority branch."""
    from codex_memory import memory_context

    if scenario == "certified":
        _certified(runtime)
    elif scenario == "unverified":
        _record(runtime, key="unverified", kind="constitution", statement="unverified")
    elif scenario == "ordinary":
        _record(runtime, key="ordinary")
    elif scenario == "unsafe":
        _record(runtime, key="unsafe", statement=('A' + 'KIA1234567890ABCDEF'))
    elif scenario == "temporary":
        _temporary(runtime)

    kwargs = dict(session_id="session-a", turn_id="turn-a", context_id="startup", scope_key="repo", prompt_event_uri=None)
    packet = memory_context.build_memory_packet(runtime.connection, BlobStore(runtime.paths.blobs_dir), **kwargs)
    assert len(packet.encode("utf-8")) == length
    assert hashlib.sha256(packet.encode("utf-8")).hexdigest() == digest


def test_authority_and_context_expose_one_snapshot_with_identical_certification_and_temporary_sets(runtime) -> None:
    """Would fail if the delegates choose different temporary/certified candidates."""
    from codex_memory import memory_authority, memory_context
    from codex_memory.memory_scope import ScopeContext

    _record(runtime, key="ordinary")
    temporary = _temporary(runtime)
    args = dict(connection=runtime.connection, blobs=BlobStore(runtime.paths.blobs_dir), session_id="session-a", scope_context=ScopeContext("repo"), query=None)
    authority_items, _ = memory_authority.select_authorized_items(**args)
    context_items, _ = memory_context.select_authorized_items(**args)
    assert [(item.identifier, item.certified, item.state) for item in authority_items] == [(item.identifier, item.certified, item.state) for item in context_items]
    assert [item.identifier for item in authority_items if item.state == "temporary"] == [temporary.candidate_id]
