from __future__ import annotations

from dataclasses import fields
import hashlib
import importlib.util
from importlib import import_module

import pytest

from codex_memory.blob_store import BlobIntegrityError
from codex_memory.hook_input import parse_hook_payload
from codex_memory.ledger_common import RecordNotFound

def _evidence():
    spec = importlib.util.find_spec("codex_memory.memory_evidence")
    assert spec is not None, "memory evidence module is required"
    module = import_module("codex_memory.memory_evidence")
    for name in (
        "EventEvidence",
        "event_uri",
        "parse_event_uri",
        "list_event_evidence",
        "resolve_event_evidence",
        "safe_evidence_reference",
    ):
        assert hasattr(module, name), f"memory evidence API is missing {name}"
    return module


def test_event_uri_is_canonical_and_rejects_unsafe_session_identifiers(runtime) -> None:
    evidence = _evidence()

    uri = evidence.event_uri("session_a", 7)

    assert uri == "codex-event://session_a/7"
    assert evidence.parse_event_uri(uri) == ("session_a", 7)
    assert evidence.event_uri("550e8400-e29b-41d4-a716-446655440000", 7) == (
        "codex-event://550e8400-e29b-41d4-a716-446655440000/7"
    )
    unsafe_fragments = tuple(":/?#[]@!$&'()*+,;=%\\") + ("%2F", chr(0), chr(0x1F), chr(0x80), " ", "\t", "\n")
    for fragment in unsafe_fragments:
        with pytest.raises(ValueError):
            evidence.event_uri(f"session{fragment}id", 7)
        with pytest.raises(ValueError):
            evidence.list_event_evidence(
                runtime.connection, session_id="session_a", turn_id=f"turn{fragment}id"
            )
    with pytest.raises(ValueError):
        evidence.event_uri("", 7)
    for invalid_uri in (
        "codex-event://session_a/07",
        "codex-event://session_a/0",
        "codex-event://session_a/7?query",
        "codex-event://session_a/7/extra",
    ):
        with pytest.raises(ValueError):
            evidence.parse_event_uri(invalid_uri)


def test_event_evidence_is_bounded_metadata_and_revalidates_blob(runtime, prompt_envelope_factory) -> None:
    evidence = _evidence()
    first = runtime.events.append(prompt_envelope_factory("session_a", "first"))
    second = runtime.events.append(prompt_envelope_factory("session_a", "second"))
    runtime.events.append(prompt_envelope_factory("session_b", "other"))
    listed = evidence.list_event_evidence(runtime.connection, session_id="session_a", limit=1)
    other_turn = runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_2",
                "cwd": ".",
                "prompt": "later",
            }
        )
    )

    assert [item.event_id for item in listed] == [second.event_id]
    assert [item.event_id for item in evidence.list_event_evidence(
        runtime.connection, session_id="session_a", turn_id="turn_1", limit=10
    )] == [second.event_id, first.event_id]
    assert other_turn.event_id not in [item.event_id for item in evidence.list_event_evidence(
        runtime.connection, session_id="session_a", turn_id="turn_1", limit=10
    )]
    for invalid_limit in (0, 101, True):
        with pytest.raises(ValueError):
            evidence.list_event_evidence(runtime.connection, session_id="session_a", limit=invalid_limit)
    assert tuple(field.name for field in fields(evidence.EventEvidence)) == (
        "event_id", "session_id", "turn_id", "event_name", "tool_name", "created_at", "uri"
    )
    resolved = evidence.resolve_event_evidence(
        runtime.connection, runtime.events._blobs, evidence.event_uri("session_a", first.event_id)
    )
    assert resolved.event_id == first.event_id
    assert resolved.session_id == "session_a"
    payload_digest = runtime.connection.execute(
        "SELECT payload_blob_sha256 FROM events WHERE event_id = ?", (first.event_id,)
    ).fetchone()["payload_blob_sha256"]
    (runtime.paths.blobs_dir / "sha256" / payload_digest[:2] / f"{payload_digest}.bin").write_bytes(
        b"corrupt"
    )
    with pytest.raises(BlobIntegrityError):
        evidence.resolve_event_evidence(
            runtime.connection, runtime.events._blobs, evidence.event_uri("session_a", first.event_id)
        )
    with pytest.raises(RecordNotFound) as missing:
        evidence.resolve_event_evidence(
            runtime.connection, runtime.events._blobs, evidence.event_uri("session_b", first.event_id)
        )
    assert "first" not in str(missing.value)


def test_safe_evidence_reference_keeps_local_uris_and_redacts_external_details() -> None:
    evidence = _evidence()
    local = evidence.event_uri("session_a", 7)
    external = ('H' + 'TTPS://viewer:synthetic@example.test/path?user=synthetic#fragment')

    assert evidence.safe_evidence_reference(local) == local
    safe = evidence.safe_evidence_reference(external)
    assert safe == f"https:sha256:{hashlib.sha256(external.encode('utf-8')).hexdigest()}"
    assert "?" not in safe and "#" not in safe and "@" not in safe
