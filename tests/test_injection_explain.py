from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, replace
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_memory.blob_store import BlobRef
from codex_memory.injection_manifest import InjectionAuditError, InjectionManifest
from codex_memory.ledger_common import canonical_json
from codex_memory.retrieval.assembler import AssembledPacket, PacketAuditSnapshot, _request_audit_binding, _result_audit_metadata
from codex_memory.retrieval.assembler import build_phase2_packet
from codex_memory.retrieval.canonical_snapshot import CanonicalSnapshot
from codex_memory.retrieval.contracts import DegradedReason, IndexGeneration, IndexHighwater, RetrievalMode, RetrievalQueryResult, RetrievalRequest, RetrievalScope, RetrievalStatus
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService


ZERO = "0" * 64


def _fixture():
    highwater = IndexHighwater()
    snapshot = CanonicalSnapshot(3, highwater, ZERO, "2026-09-14T00:00:00Z")
    prompt = {"event_created_at": "2026-09-14T00:00:00Z", "event_name": "UserPromptSubmit", "event_session_id": "session", "event_turn_id": "turn", "payload_blob_sha256": ZERO, "reference": "codex-event://session/7", "reference_sha256": ZERO}
    request = RetrievalRequest("private query text", "session", "turn", "prompt", "codex-event://session/7", RetrievalScope("repo"), mode=RetrievalMode.SHADOW)
    # The existing query service truthfully reports its executed mode as OFF
    # even when a caller requested output-inert SHADOW orchestration.
    result = RetrievalQueryResult(RetrievalStatus.CURRENT, RetrievalMode.OFF, None, highwater, highwater, False, (), (), ())
    audit = PacketAuditSnapshot(snapshot, (), (), prompt, (), (), (), (), (), (), (("budget", 0), ("duplicate", 0), ("provenance", 0), ("suppressed", 0), ("unsafe", 0)), (), _request_audit_binding(request), _result_audit_metadata(result))
    assembled = AssembledPacket("[CODEX MEMORY PROTOCOL v2]\n", "asm_" + "0" * 32, hashlib.sha256(b"[CODEX MEMORY PROTOCOL v2]\n").hexdigest(), 27, (), (), (), (), (), (("budget", 0), ("duplicate", 0), ("provenance", 0), ("suppressed", 0), ("unsafe", 0)), (), audit)
    source = {"event_id": 7, "event_uri": "codex-event://session/7", "event_blob_sha256": ZERO, "session_id": "session", "turn_id": "turn", "context_id": "prompt", "origin": "offline"}
    environment = {"provider": {"configured": "disabled", "observed": "disabled", "validated": "disabled"}, "model": {"configured": "disabled", "observed": "disabled", "validated": "disabled"}, "preprocessing": {"configured": "disabled", "observed": "disabled", "validated": "disabled"}, "config_sha256": ZERO, "implementation_sha256": ZERO, "legacy_delivery": "emitted", "host_acknowledgement": "unknown"}
    timing = {"started_at": "2026-09-14T00:00:00Z", "prepared_at": "2026-09-14T00:00:01Z", "query_duration_ms": 1, "assembly_duration_ms": 2, "prepare_duration_ms": 3, "budget_ms": 1000}
    return request, result, assembled, source, environment, timing


def test_prepare_manifest_uses_exact_frozen_snapshot_and_blobs_without_query_text(monkeypatch):
    from codex_memory.retrieval import injection_audit
    request, result, assembled, source, environment, timing = _fixture()
    captured = []
    def put(_blobs, data, *, deadline):
        captured.append(data)
        digest = hashlib.sha256(data).hexdigest()
        return BlobRef(digest, len(data), f"sha256/{digest[:2]}/{digest}.bin")
    monkeypatch.setattr(injection_audit, "put_audit_blob", put)
    manifest = injection_audit.prepare_shadow_manifest(request=request, result=result, assembled=assembled, source=source, legacy_packet="legacy exact\n", legacy_response_bytes=b'{"hookSpecificOutput":{}}\n', environment=environment, timing=timing, blobs=object(), deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
    assert captured == [assembled.packet.encode(), b"legacy exact\n", b'{"hookSpecificOutput":{}}\n']
    assert manifest.body["request"]["query_sha256"] == hashlib.sha256(canonical_json(request.query).encode()).hexdigest()
    assert "private query text" not in manifest.to_bytes().decode()
    assert manifest.body["delivery"] == {"v2": "prepared", "legacy": "emitted", "host_acknowledgement": "unknown"}
    assert manifest.body["request"]["mode"] == "shadow"
    assert manifest.body["candidates"]["query_result"]["mode"] == "off"


def test_actual_service_preserves_shadow_request_and_reports_executed_off_mode(runtime):
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths)
    assert service.rebuild().status.value == "current"
    request = RetrievalRequest("audit mode truth", "session", "turn", "prompt", None, RetrievalScope(str(runtime.paths.repo_root)), mode=RetrievalMode.SHADOW)
    result = service.query(request)
    assert request.mode is RetrievalMode.SHADOW
    assert result.mode is RetrievalMode.OFF


def test_actual_event_service_and_assembly_snapshot_prepare_normally(monkeypatch, runtime, prompt_envelope_factory):
    from codex_memory.retrieval import injection_audit
    record = runtime.events.append(prompt_envelope_factory("session_actual", "actual audit query"))
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths); service.rebuild()
    request = RetrievalRequest("actual audit query", "session_actual", "turn_1", "prompt", f"codex-event://session_actual/{record.event_id}", RetrievalScope(str(runtime.paths.repo_root)), mode=RetrievalMode.SHADOW)
    result = service.query(request)
    assembled = build_phase2_packet(paths, request=request, result=result)
    captured = []
    def put(_blobs, data, *, deadline):
        digest = hashlib.sha256(data).hexdigest(); captured.append(data)
        return BlobRef(digest, len(data), f"sha256/{digest[:2]}/{digest}.bin")
    monkeypatch.setattr(injection_audit, "put_audit_blob", put)
    _request, _result, _assembled, _source, environment, timing = _fixture()
    source = {"event_id": record.event_id, "event_uri": request.prompt_event_uri, "event_blob_sha256": record.payload_blob_sha256, "session_id": record.session_id, "turn_id": record.turn_id, "context_id": request.context_id, "origin": "offline"}
    manifest = injection_audit.prepare_shadow_manifest(request=request, result=result, assembled=assembled, source=source, legacy_packet="legacy", legacy_response_bytes=b"response", environment=environment, timing=timing, blobs=object(), deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
    assert manifest.body["request"]["mode"] == "shadow"
    assert manifest.body["candidates"]["query_result"]["mode"] == "off"


@pytest.mark.parametrize("changed", ["event_id", "session_id", "turn_id", "event_blob_sha256"])
def test_prepare_manifest_rejects_current_source_mismatch_before_blob_write(monkeypatch, changed):
    from codex_memory.retrieval import injection_audit
    request, result, assembled, source, environment, timing = _fixture()
    source[changed] = 8 if changed == "event_id" else ("f" * 64 if changed == "event_blob_sha256" else "other")
    monkeypatch.setattr(injection_audit, "put_audit_blob", lambda *_a, **_k: pytest.fail("source mismatch must precede blob writes"))
    with pytest.raises(InjectionAuditError) as raised:
        injection_audit.prepare_shadow_manifest(request=request, result=result, assembled=assembled, source=source, legacy_packet="legacy", legacy_response_bytes=b"response", environment=environment, timing=timing, blobs=object(), deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
    assert raised.value.reason.value == "source_mismatch"


def _changed_generation(highwater):
    return IndexGeneration("gen_changed", "active", highwater, highwater, "1" * 64, "provider", "model", "validated", "2026-09-14T00:00:00Z", "2026-09-14T00:00:00Z")


@pytest.mark.parametrize(
    ("request_change", "result_change"),
    [
        (lambda request: replace(request, query="never assembled query"), None),
        (lambda request: replace(request, scope=RetrievalScope("other")), None),
        (lambda request: replace(request, max_chars=3999), None),
        (None, lambda result: replace(result, status=RetrievalStatus.STALE)),
        (None, lambda result: replace(result, index_stale=True)),
        (None, lambda result: replace(result, indexed_highwater=IndexHighwater(record_seq=1))),
        (None, lambda result: replace(result, degraded_reasons=(DegradedReason.INDEX_STALE,))),
        (None, lambda result: replace(result, generation=_changed_generation(result.canonical_highwater))),
    ],
)
def test_prepare_rejects_request_or_result_not_used_by_assembly_before_blob_write(monkeypatch, request_change, result_change):
    from codex_memory.retrieval import injection_audit
    request, result, assembled, source, environment, timing = _fixture()
    if request_change is not None:
        request = request_change(request)
    if result_change is not None:
        result = result_change(result)
    monkeypatch.setattr(injection_audit, "put_audit_blob", lambda *_a, **_k: pytest.fail("binding mismatch must precede blob writes"))
    with pytest.raises(InjectionAuditError):
        injection_audit.prepare_shadow_manifest(request=request, result=result, assembled=assembled, source=source, legacy_packet="legacy", legacy_response_bytes=b"response", environment=environment, timing=timing, blobs=object(), deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))


def test_explain_replays_verified_values_without_packet_text_or_retrieval(monkeypatch):
    from codex_memory.retrieval import injection_audit
    request, result, assembled, source, environment, timing = _fixture()
    refs = []
    def put(_blobs, data, *, deadline):
        digest = hashlib.sha256(data).hexdigest(); refs.append(data)
        return BlobRef(digest, len(data), f"sha256/{digest[:2]}/{digest}.bin")
    monkeypatch.setattr(injection_audit, "put_audit_blob", put)
    manifest = injection_audit.prepare_shadow_manifest(request=request, result=result, assembled=assembled, source=source, legacy_packet="SENSITIVE PACKET", legacy_response_bytes=b"SENSITIVE RESPONSE", environment=environment, timing=timing, blobs=object(), deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
    verified = SimpleNamespace(manifest=manifest, prepared_packet_bytes=refs[0], legacy_additional_context_bytes=refs[1], serialized_response_bytes=refs[2], receipts=())
    output = injection_audit.explain_injection(verified)
    assert output["injection_id"] == manifest.injection_id
    assert output["selected"] == []
    assert output["receipts"] == {"status": "unknown", "items": []}
    assert "SENSITIVE" not in canonical_json(output)
    assert output["packets"]["prepared_packet_sha256"] == manifest.body["packets"]["prepared_packet_ref"]["sha256"]


def test_implementation_fingerprint_is_deterministic_and_honors_deadline():
    from codex_memory.retrieval import injection_audit
    first = injection_audit.implementation_fingerprint(deadline=DeadlineBudget.start(5000, now_ns=time.monotonic_ns))
    second = injection_audit.implementation_fingerprint(deadline=DeadlineBudget.start(5000, now_ns=time.monotonic_ns))
    assert first == second and len(first) == 64
    with pytest.raises(InjectionAuditError) as raised:
        injection_audit.implementation_fingerprint(deadline=DeadlineBudget(0, 0))
    assert raised.value.reason.value == "deadline_exhausted"


def test_implementation_fingerprint_rejects_oversize_before_unbounded_read(monkeypatch, tmp_path):
    from codex_memory.retrieval import injection_audit
    package = tmp_path / "codex_memory"; module_dir = package / "retrieval"; module_dir.mkdir(parents=True)
    module_file = module_dir / "injection_audit.py"; module_file.write_text("# module\n", encoding="utf-8")
    oversized = package / "oversized.py"
    with oversized.open("wb") as stream:
        stream.seek(8 * 1024 * 1024); stream.write(b"x")
    original = Path.read_bytes
    def bounded_read(path):
        if path == oversized:
            pytest.fail("oversized source must be rejected before read_bytes")
        return original(path)
    monkeypatch.setattr(injection_audit, "__file__", str(module_file))
    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    with pytest.raises(InjectionAuditError) as raised:
        injection_audit.implementation_fingerprint(deadline=DeadlineBudget.start(5000, now_ns=time.monotonic_ns))
    assert raised.value.reason.value == "size_limit"


def test_implementation_fingerprint_rejects_redirected_source(monkeypatch, tmp_path):
    from codex_memory.retrieval import injection_audit
    package = tmp_path / "codex_memory"; module_dir = package / "retrieval"; module_dir.mkdir(parents=True)
    module_file = module_dir / "injection_audit.py"; module_file.write_text("# module\n", encoding="utf-8")
    target = tmp_path / "outside.py"; target.write_text("# outside\n", encoding="utf-8")
    redirected = package / "redirected.py"
    try:
        redirected.symlink_to(target)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    monkeypatch.setattr(injection_audit, "__file__", str(module_file))
    with pytest.raises(InjectionAuditError) as raised:
        injection_audit.implementation_fingerprint(deadline=DeadlineBudget.start(5000, now_ns=time.monotonic_ns))
    assert raised.value.reason.value == "path_unsafe"


def test_explain_rejects_tampered_loaded_bytes(monkeypatch):
    from codex_memory.retrieval import injection_audit
    request, result, assembled, source, environment, timing = _fixture()
    values = []
    def put(_blobs, data, *, deadline):
        digest = hashlib.sha256(data).hexdigest(); values.append(data)
        return BlobRef(digest, len(data), f"sha256/{digest[:2]}/{digest}.bin")
    monkeypatch.setattr(injection_audit, "put_audit_blob", put)
    manifest = injection_audit.prepare_shadow_manifest(request=request, result=result, assembled=assembled, source=source, legacy_packet="legacy", legacy_response_bytes=b"response", environment=environment, timing=timing, blobs=object(), deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
    verified = SimpleNamespace(manifest=manifest, prepared_packet_bytes=b"tampered", legacy_additional_context_bytes=values[1], serialized_response_bytes=values[2], receipts=())
    with pytest.raises(InjectionAuditError) as raised:
        injection_audit.explain_injection(verified)
    assert raised.value.reason.value == "tampered"


def test_historical_explain_survives_deleted_retrieval_and_canonical(tmp_path):
    from codex_memory.blob_store import BlobStore
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    from codex_memory.retrieval.injection_audit import explain_injection
    from test_injection_manifest import manifest_body as manifest_body_fixture
    manifest_body = manifest_body_fixture.__wrapped__()
    blobs = BlobStore(tmp_path / ".codex-memory" / "blobs")
    raw_values = (b"packet", b"legacy", b'{"ok":true}\n')
    refs = [blobs.put_bytes(raw) for raw in raw_values]
    for key, ref in zip(("prepared_packet_ref", "legacy_additional_context_ref", "serialized_response_ref"), refs, strict=True):
        manifest_body["packets"][key] = asdict(ref)
    manifest_body["packets"]["packet_chars"] = len("packet")
    manifest = InjectionManifest.from_body(manifest_body)
    store = InjectionStore(InjectionPaths.for_repo(tmp_path), blobs)
    store.commit(manifest, deadline=DeadlineBudget.start(10000, now_ns=time.monotonic_ns))
    retrieval = tmp_path / ".codex-memory" / "retrieval"; retrieval.mkdir()
    canonical = tmp_path / ".codex-memory" / "db"; canonical.mkdir(); (canonical / "memory.sqlite3").write_bytes(b"later invalid")
    shutil.rmtree(retrieval); shutil.rmtree(canonical)
    output = explain_injection(store.load(manifest.injection_id))
    assert output["injection_id"] == manifest.injection_id
    assert output["packets"]["prepared_packet_sha256"] == refs[0].sha256
