from __future__ import annotations

import json
import hashlib

import pytest


def _manifest_module():
    try:
        from codex_memory import injection_manifest
    except ImportError as exc:
        pytest.fail(f"injection manifest codec is missing: {exc}", pytrace=False)
    return injection_manifest


@pytest.fixture
def manifest_body() -> dict[str, object]:
    zero = "0" * 64
    blob_ref = {"sha256": zero, "byte_size": 0, "relative_path": f"sha256/{zero[:2]}/{zero}.bin"}
    return {
        "format": "codex-memory.injection.v1",
        "source": {"event_id": 1, "event_uri": "codex-event://s/1", "event_blob_sha256": zero, "session_id": "s", "turn_id": "t", "context_id": "prompt", "origin": "offline"},
        "request": {"project_scope_key": "repo", "workspace_scope_keys": [], "actor_id": "codex", "role": "operator", "memory_view_id": "ground_truth.default", "epistemic_policy_id": "default", "truth_domain": "ground_truth", "query_sha256": zero, "max_chars": 4000, "mode": "shadow"},
        "snapshot": {"canonical_highwater": {"record_seq": 0, "memory_event_seq": 0, "evidence_seq": 0, "relation_seq": 0, "candidate_seq": 0, "candidate_event_seq": 0, "candidate_evidence_seq": 0}, "indexed_highwater": {"record_seq": 0, "memory_event_seq": 0, "evidence_seq": 0, "relation_seq": 0, "candidate_seq": 0, "candidate_event_seq": 0, "candidate_evidence_seq": 0}, "canonical_content_fingerprint": zero, "generation": None, "index_stale": False, "degraded_reasons": [], "requirement_version": "p2.5"},
        "identities": {"provider": {"configured": "disabled", "observed": "disabled", "validated": "disabled"}, "model": {"configured": "disabled", "observed": "disabled", "validated": "disabled"}, "preprocessing": {"configured": "disabled", "observed": "disabled", "validated": "disabled"}, "config_sha256": zero, "implementation_sha256": zero},
        "candidates": {"query_result": {"status": "current", "mode": "shadow", "generation": None, "canonical_highwater": {"record_seq": 0, "memory_event_seq": 0, "evidence_seq": 0, "relation_seq": 0, "candidate_seq": 0, "candidate_event_seq": 0, "candidate_evidence_seq": 0}, "indexed_highwater": {"record_seq": 0, "memory_event_seq": 0, "evidence_seq": 0, "relation_seq": 0, "candidate_seq": 0, "candidate_event_seq": 0, "candidate_evidence_seq": 0}, "index_stale": False, "degraded_reasons": [], "candidates": [], "trace": []}, "final_selected": [], "budget_omitted_ids": [], "build_omissions": [], "duplicate_ids": [], "suppressed_ids": [], "omission_counts": {"budget": 0, "duplicate": 0, "provenance": 0, "suppressed": 0, "unsafe": 0}},
        "evidence": {"packs": []},
        "packets": {"prepared_packet_ref": blob_ref, "packet_chars": 0, "assembly_id": "asm_" + "0" * 32, "legacy_additional_context_ref": blob_ref, "serialized_response_ref": blob_ref},
        "timing": {"started_at": "2026-09-14T00:00:00Z", "prepared_at": "2026-09-14T00:00:01Z", "query_duration_ms": 1, "assembly_duration_ms": 1, "prepare_duration_ms": 1, "budget_ms": 1000},
        "delivery": {"v2": "prepared", "legacy": "emitted", "host_acknowledgement": "unknown"},
    }


def test_manifest_round_trips_canonical_bytes_and_self_hash(manifest_body):
    module = _manifest_module()
    manifest = module.InjectionManifest.from_body(manifest_body)
    decoded = json.loads(manifest.to_bytes())
    assert set(decoded) == {"body", "manifest_sha256", "injection_id"}
    assert decoded["injection_id"] == "inj_" + decoded["manifest_sha256"]
    assert module.InjectionManifest.from_bytes(manifest.to_bytes()) == manifest


@pytest.mark.parametrize("raw", [b'{"body":{},"body":{}}', b'{"body":NaN}', b'[]', b'{} trailing'])
def test_manifest_rejects_non_strict_json(raw):
    module = _manifest_module()
    with pytest.raises(module.InjectionAuditError) as raised:
        module.InjectionManifest.from_bytes(raw)
    assert raised.value.reason is module.InjectionAuditReason.INVALID_INPUT


def test_manifest_rejects_unknown_nested_field(manifest_body):
    module = _manifest_module()
    manifest_body["packets"]["wire_response_ref"] = None
    with pytest.raises(module.InjectionAuditError, match="unknown or missing"):
        module.InjectionManifest.from_body(manifest_body)


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("format",), "codex-memory.injection.v2", "unsupported"),
        (("request", "truth_domain"), "personal_guess", "unsupported"),
        (("request", "mode"), "active", "unsupported"),
    ],
)
def test_manifest_rejects_unsupported_contract_values(manifest_body, path, value, reason):
    module = _manifest_module()
    target = manifest_body
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(module.InjectionAuditError) as raised:
        module.InjectionManifest.from_body(manifest_body)
    assert raised.value.reason.value == reason


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("snapshot", "index_stale"), "false", "invalid_input"),
        (("delivery", "legacy"), "banana", "invalid_input"),
        (("source", "event_id"), True, "invalid_input"),
        (("source", "event_uri"), "codex-event://other/1", "source_mismatch"),
        (("packets", "prepared_packet_ref", "relative_path"), "../blob", "invalid_input"),
        (("candidates", "query_result", "trace"), [{"candidate_id": "x"}], "invalid_input"),
    ],
)
def test_manifest_rejects_invalid_nested_values(manifest_body, path, value, reason):
    module = _manifest_module()
    target = manifest_body
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(module.InjectionAuditError) as raised:
        module.InjectionManifest.from_body(manifest_body)
    assert raised.value.reason.value == reason


def test_manifest_wraps_nested_contract_type_errors_as_audit_errors(manifest_body):
    module = _manifest_module()
    manifest_body["candidates"]["query_result"]["candidates"] = [{
        "candidate_kind": [],
        "candidate_id": "mem_1",
        "record_type": "decision",
        "scope_kind": "project",
        "scope_key": "repo",
        "state": "active",
        "sequence": 1,
        "content_sha256": "0" * 64,
        "mandatory": False,
        "fused_score_numerator": 1,
        "fused_score_denominator": 1,
        "contributions": [],
        "trust_label": "canonical",
    }]
    with pytest.raises(module.InjectionAuditError) as raised:
        module.InjectionManifest.from_body(manifest_body)
    assert raised.value.reason is module.InjectionAuditReason.INVALID_INPUT


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("request", "mode"), []),
        (("source", "origin"), {}),
        (("delivery", "legacy"), []),
    ],
)
def test_manifest_membership_fields_never_leak_type_error_from_factory_or_parser(
    manifest_body, path, value
):
    module = _manifest_module()
    target = manifest_body
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(module.InjectionAuditError):
        module.InjectionManifest.from_body(manifest_body)
    body_bytes = json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(body_bytes).hexdigest()
    raw = json.dumps({"body": manifest_body, "manifest_sha256": digest, "injection_id": "inj_" + digest}).encode()
    with pytest.raises(module.InjectionAuditError):
        module.InjectionManifest.from_bytes(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evidence", "packs"), [{"arbitrary_unknown": "accepted"}]),
        (("candidates", "final_selected"), [{"candidate_id": None, "content_sha256": "bad", "last_event_sequence": -1}]),
        (("candidates", "omission_counts"), "not-an-object"),
        (("timing", "started_at"), 123),
    ],
)
def test_manifest_rejects_unvalidated_owned_leaves(manifest_body, path, value):
    module = _manifest_module()
    target = manifest_body
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(module.InjectionAuditError):
        module.InjectionManifest.from_body(manifest_body)


def test_manifest_validates_complete_evidence_pack_and_nested_unknowns(manifest_body):
    module = _manifest_module()
    zero = "0" * 64
    event = {"action": "activate", "created_at": "2026-09-14T00:00:00Z", "event_id": "mev_1", "event_sequence": 1, "event_sha256": zero}
    pack = {
        "card": {"candidate_kind": "record", "certified": False, "content_sha256": zero, "id": "mem_1", "kind": "decision", "last_event": event, "scope": {"key": "repo", "kind": "project"}, "sequence": 1, "state": "active", "statement": "A safe evidence statement.", "topic": "runtime.audit", "trust_label": "canonical", "value": None},
        "evidence": [],
        "pack_version": 1,
        "provenance_status": "complete",
    }
    manifest_body["evidence"]["packs"] = [pack]
    manifest_body["candidates"]["final_selected"] = [{"candidate_id": "mem_1", "content_sha256": zero, "last_event_sequence": 1}]
    assert module.InjectionManifest.from_body(manifest_body).body["evidence"]["packs"][0]["card"]["id"] == "mem_1"
    pack["card"]["unknown"] = "rejected"
    with pytest.raises(module.InjectionAuditError, match="unknown or missing"):
        module.InjectionManifest.from_body(manifest_body)


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_manifest_rejects_non_integer_evidence_pack_version(manifest_body, invalid_version):
    module = _manifest_module()
    zero = "0" * 64
    manifest_body["evidence"]["packs"] = [{
        "card": {"candidate_kind": "record", "certified": False, "content_sha256": zero, "id": "mem_1", "kind": "decision", "last_event": {"action": "activate", "created_at": "2026-09-14T00:00:00Z", "event_id": "mev_1", "event_sequence": 1, "event_sha256": zero}, "scope": {"key": "repo", "kind": "project"}, "sequence": 1, "state": "active", "statement": "Safe.", "topic": "runtime.audit", "trust_label": "canonical", "value": None},
        "evidence": [], "pack_version": invalid_version, "provenance_status": "complete",
    }]
    manifest_body["candidates"]["final_selected"] = [{"candidate_id": "mem_1", "content_sha256": zero, "last_event_sequence": 1}]
    with pytest.raises(module.InjectionAuditError):
        module.InjectionManifest.from_body(manifest_body)


def test_manifest_does_not_treat_unicode_space_as_json_whitespace():
    module = _manifest_module()
    with pytest.raises(module.InjectionAuditError) as raised:
        module.InjectionManifest.from_bytes("\u00a0{}".encode("utf-8"))
    assert raised.value.reason is module.InjectionAuditReason.INVALID_INPUT


def test_manifest_rejects_changed_body_with_old_digest(manifest_body):
    module = _manifest_module()
    document = json.loads(module.InjectionManifest.from_body(manifest_body).to_bytes())
    document["body"]["request"]["max_chars"] = 3999
    with pytest.raises(module.InjectionAuditError, match="tampered"):
        module.InjectionManifest.from_bytes(json.dumps(document).encode("utf-8"))


def test_manifest_body_is_deeply_frozen_and_copied(manifest_body):
    module = _manifest_module()
    manifest = module.InjectionManifest.from_body(manifest_body)
    manifest_body["request"]["max_chars"] = 1
    assert manifest.body["request"]["max_chars"] == 4000
    with pytest.raises(TypeError):
        manifest.body["request"]["max_chars"] = 2
    with pytest.raises(AttributeError):
        manifest.body["candidates"]["final_selected"].append("x")


def test_public_constructor_cannot_bypass_validation_or_deep_freezing(manifest_body):
    module = _manifest_module()
    with pytest.raises(module.InjectionAuditError):
        module.InjectionManifest(manifest_body, "0" * 64, "inj_" + "0" * 64)
    expected = module.InjectionManifest.from_body(manifest_body)
    direct = module.InjectionManifest(manifest_body, expected.manifest_sha256, expected.injection_id)
    manifest_body["request"]["max_chars"] = 1
    assert direct.body["request"]["max_chars"] == 4000
