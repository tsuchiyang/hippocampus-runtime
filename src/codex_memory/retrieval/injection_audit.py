"""Prepare and explain exact injection evidence without rerunning retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import stat
import time
from typing import Any

from codex_memory.blob_store import BlobRef, BlobStore
from codex_memory.injection_manifest import InjectionAuditError, InjectionManifest
from codex_memory.injection_store import VerifiedInjection, put_audit_blob
from codex_memory.ledger_common import canonical_json
from codex_memory.memory_evidence import parse_event_uri

from .assembler import AssembledPacket, _request_audit_binding, _result_audit_metadata
from .contracts import RetrievalQueryResult, RetrievalRequest
from .deadline import DeadlineBudget


_SOURCE_FIELDS = frozenset({"event_id", "event_uri", "event_blob_sha256", "session_id", "turn_id", "context_id", "origin"})
_ENVIRONMENT_FIELDS = frozenset({"provider", "model", "preprocessing", "config_sha256", "implementation_sha256", "legacy_delivery", "host_acknowledgement"})
_TIMING_FIELDS = frozenset({"started_at", "prepared_at", "query_duration_ms", "assembly_duration_ms", "prepare_duration_ms", "budget_ms"})
_MAX_SOURCE_FILES = 512
_MAX_SOURCE_BYTES = 8 * 1024 * 1024


def _deadline_check(deadline: DeadlineBudget) -> None:
    if not isinstance(deadline, DeadlineBudget):
        raise InjectionAuditError("invalid_input", "deadline is invalid")
    if time.monotonic_ns() >= deadline.deadline_ns:
        raise InjectionAuditError("deadline_exhausted")


def implementation_fingerprint(*, deadline: DeadlineBudget) -> str:
    """Hash the installed package's bounded Python source tree, without Git."""
    module_path = Path(__file__).absolute()
    package_root = module_path.parents[1]
    digest = hashlib.sha256()
    count = total = 0
    try:
        for directory in (package_root, module_path.parent):
            _deadline_check(deadline)
            if directory.is_symlink() or not directory.is_dir():
                raise InjectionAuditError("path_unsafe", "installed source directory is unsafe")
        paths = []
        for path in package_root.rglob("*.py"):
            _deadline_check(deadline)
            paths.append(path)
            if len(paths) > _MAX_SOURCE_FILES:
                raise InjectionAuditError("size_limit", "installed source file limit exceeded")
        paths.sort(key=lambda item: item.relative_to(package_root).as_posix())
        for path in paths:
            _deadline_check(deadline)
            parent = path.parent
            while parent != package_root:
                if parent.is_symlink() or not parent.is_dir():
                    raise InjectionAuditError("path_unsafe", "installed source directory is unsafe")
                parent = parent.parent
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise InjectionAuditError("path_unsafe", "installed source path is unsafe")
            count += 1
            remaining = _MAX_SOURCE_BYTES - total
            if metadata.st_size > remaining:
                raise InjectionAuditError("size_limit", "installed source byte limit exceeded")
            chunks: list[bytes] = []
            observed = 0
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size != metadata.st_size:
                    raise InjectionAuditError("path_unsafe", "installed source changed during fingerprint")
                while True:
                    _deadline_check(deadline)
                    chunk = stream.read(min(64 * 1024, remaining - observed + 1))
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > remaining:
                        raise InjectionAuditError("size_limit", "installed source byte limit exceeded")
                    chunks.append(chunk)
            if observed != metadata.st_size:
                raise InjectionAuditError("unavailable", "installed source changed during fingerprint")
            data = b"".join(chunks)
            total += observed
            relative = path.relative_to(package_root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
            digest.update(len(data).to_bytes(8, "big")); digest.update(data)
            _deadline_check(deadline)
    except InjectionAuditError:
        raise
    except OSError as exc:
        raise InjectionAuditError("unavailable", "installed source cannot be fingerprinted", cause=exc) from exc
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _exact_object(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InjectionAuditError("invalid_input", f"{name} has unknown or missing fields")
    return value


def _validate_environment(value: object) -> dict[str, Any]:
    environment = _exact_object(value, _ENVIRONMENT_FIELDS, "environment")
    identity_fields = frozenset({"configured", "observed", "validated"})
    for name in ("provider", "model", "preprocessing"):
        identity = _exact_object(environment[name], identity_fields, f"environment.{name}")
        if any(not isinstance(item, str) or not item for item in identity.values()):
            raise InjectionAuditError("invalid_input", f"environment.{name} is invalid")
    for name in ("config_sha256", "implementation_sha256"):
        digest = environment[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise InjectionAuditError("invalid_input", f"environment.{name} is invalid")
    delivery = environment["legacy_delivery"]
    if not isinstance(delivery, str) or delivery not in {"emitted", "partial", "unknown"} or environment["host_acknowledgement"] != "unknown":
        raise InjectionAuditError("invalid_input", "environment delivery is invalid")
    return environment


def _validate_timing(value: object) -> dict[str, Any]:
    from codex_memory.injection_manifest import _require_timestamp

    timing = _exact_object(value, _TIMING_FIELDS, "timing")
    for name in ("started_at", "prepared_at"):
        _require_timestamp(timing[name], f"timing.{name}")
    for name in ("query_duration_ms", "assembly_duration_ms", "prepare_duration_ms", "budget_ms"):
        item = timing[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise InjectionAuditError("invalid_input", f"timing.{name} is invalid")
    return timing


def _validate_source(request: RetrievalRequest, assembled: AssembledPacket, source: object) -> dict[str, Any]:
    value = _exact_object(source, _SOURCE_FIELDS, "source")
    if (
        isinstance(value["event_id"], bool)
        or not isinstance(value["event_id"], int)
        or value["event_id"] < 1
        or not isinstance(value["event_blob_sha256"], str)
        or len(value["event_blob_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in value["event_blob_sha256"])
        or not isinstance(value["origin"], str)
        or value["origin"] not in {"hook", "offline"}
    ):
        raise InjectionAuditError("source_mismatch", "source identity is invalid")
    audit = assembled.audit_snapshot
    if audit is None or audit.prompt_evidence is None:
        raise InjectionAuditError("source_mismatch", "verified prompt evidence is missing")
    prompt = _plain(audit.prompt_evidence)
    expected = {
        "event_uri": request.prompt_event_uri,
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "context_id": request.context_id,
    }
    if any(value[key] != expected[key] for key in expected):
        raise InjectionAuditError("source_mismatch", "source does not match request")
    try:
        uri_session, uri_event_id = parse_event_uri(value["event_uri"])
    except (TypeError, ValueError) as exc:
        raise InjectionAuditError("source_mismatch", "source event URI is invalid", cause=exc) from exc
    if uri_session != value["session_id"] or uri_event_id != value["event_id"]:
        raise InjectionAuditError("source_mismatch", "source event identity differs")
    if (
        prompt.get("reference") != value["event_uri"]
        or prompt.get("event_session_id") != value["session_id"]
        or prompt.get("event_turn_id") != value["turn_id"]
        or prompt.get("event_name") != "UserPromptSubmit"
        or prompt.get("payload_blob_sha256") != value["event_blob_sha256"]
    ):
        raise InjectionAuditError("source_mismatch", "source does not match verified prompt evidence")
    return value


def _validate_same_assembly(request: RetrievalRequest, result: RetrievalQueryResult, assembled: AssembledPacket) -> None:
    audit = assembled.audit_snapshot
    if audit is None:
        raise InjectionAuditError("invalid_input", "packet audit snapshot is missing")
    if audit.request_binding is None or _plain(audit.request_binding) != _request_audit_binding(request):
        raise InjectionAuditError("tampered", "request does not match packet audit snapshot")
    if audit.result_metadata is None or _plain(audit.result_metadata) != _result_audit_metadata(result):
        raise InjectionAuditError("tampered", "result metadata does not match packet audit snapshot")
    if _plain(audit.query_candidates) != [item.to_dict() for item in result.candidates] or _plain(audit.query_trace) != [item.to_dict() for item in result.trace]:
        raise InjectionAuditError("tampered", "query result does not match packet audit snapshot")
    if result.canonical_highwater != audit.canonical_snapshot.highwater:
        raise InjectionAuditError("tampered", "query result does not match canonical snapshot")
    if tuple(_plain(audit.final_selected)) != tuple(item.to_dict() for item in assembled.selected):
        raise InjectionAuditError("tampered", "packet selection does not match audit snapshot")
    if assembled.packet_chars != len(assembled.packet) or assembled.packet_sha256 != hashlib.sha256(assembled.packet.encode("utf-8")).hexdigest():
        raise InjectionAuditError("tampered", "prepared packet identity is invalid")
    if request.mode.value == "active" or result.mode.value == "active":
        raise InjectionAuditError("unsupported", "active retrieval audit is unsupported")


def prepare_shadow_manifest(
    *,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    assembled: AssembledPacket,
    source: dict[str, object],
    legacy_packet: str,
    legacy_response_bytes: bytes,
    environment: dict[str, object],
    timing: dict[str, object],
    blobs: BlobStore,
    deadline: DeadlineBudget,
) -> InjectionManifest:
    """Seal already-observed prompt/query/assembly values; never requery state."""
    if not isinstance(request, RetrievalRequest) or not isinstance(result, RetrievalQueryResult) or not isinstance(assembled, AssembledPacket):
        raise InjectionAuditError("invalid_input", "audit inputs are invalid")
    if not isinstance(legacy_packet, str) or not isinstance(legacy_response_bytes, bytes) or not isinstance(deadline, DeadlineBudget):
        raise InjectionAuditError("invalid_input", "audit byte inputs are invalid")
    source_value = _validate_source(request, assembled, source)
    environment_value = _validate_environment(environment)
    timing_value = _validate_timing(timing)
    _validate_same_assembly(request, result, assembled)
    audit = assembled.audit_snapshot

    prepared_ref = put_audit_blob(blobs, assembled.packet.encode("utf-8"), deadline=deadline)
    legacy_ref = put_audit_blob(blobs, legacy_packet.encode("utf-8"), deadline=deadline)
    response_ref = put_audit_blob(blobs, legacy_response_bytes, deadline=deadline)
    query_sha256 = hashlib.sha256(canonical_json(request.query).encode("utf-8")).hexdigest()
    body = {
        "format": "codex-memory.injection.v1",
        "source": dict(source_value),
        "request": {
            "project_scope_key": request.scope.project_scope_key,
            "workspace_scope_keys": list(request.scope.workspace_scope_keys),
            "actor_id": request.epistemic_context.actor_id,
            "role": request.epistemic_context.role,
            "memory_view_id": request.epistemic_context.memory_view_id,
            "epistemic_policy_id": request.epistemic_context.epistemic_policy_id,
            "truth_domain": request.epistemic_context.truth_domain,
            "query_sha256": query_sha256,
            "max_chars": request.max_chars,
            "mode": request.mode.value,
        },
        "snapshot": {
            "canonical_highwater": audit.canonical_snapshot.highwater.to_dict(),
            "indexed_highwater": None if result.indexed_highwater is None else result.indexed_highwater.to_dict(),
            "canonical_content_fingerprint": audit.canonical_snapshot.content_fingerprint,
            "generation": None if result.generation is None else result.generation.to_dict(),
            "index_stale": result.index_stale,
            "degraded_reasons": [reason.value for reason in result.degraded_reasons],
            "requirement_version": "p2.5",
        },
        "identities": {key: environment_value[key] for key in ("provider", "model", "preprocessing", "config_sha256", "implementation_sha256")},
        "candidates": {
            "query_result": result.to_dict(),
            "final_selected": _plain(audit.final_selected),
            "budget_omitted_ids": list(audit.budget_omitted_ids),
            "build_omissions": _plain(audit.build_omissions),
            "duplicate_ids": list(audit.duplicate_ids),
            "suppressed_ids": list(audit.suppressed_ids),
            "omission_counts": dict(audit.omission_counts),
        },
        "evidence": {"packs": _plain(audit.evidence_packs)},
        "packets": {
            "prepared_packet_ref": asdict(prepared_ref),
            "packet_chars": assembled.packet_chars,
            "assembly_id": assembled.manifest_id,
            "legacy_additional_context_ref": asdict(legacy_ref),
            "serialized_response_ref": asdict(response_ref),
        },
        "timing": dict(timing_value),
        "delivery": {"v2": "prepared", "legacy": environment_value["legacy_delivery"], "host_acknowledgement": environment_value["host_acknowledgement"]},
    }
    return InjectionManifest.from_body(body)


def _verified_ref(raw: bytes, ref: Mapping[str, object], name: str) -> None:
    if len(raw) != ref["byte_size"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise InjectionAuditError("tampered", f"{name} does not match manifest")


def explain_injection(verified: VerifiedInjection) -> dict[str, object]:
    """Explain only loaded, verified historical bytes; never access live state."""
    if not hasattr(verified, "manifest") or not isinstance(verified.manifest, InjectionManifest):
        raise InjectionAuditError("invalid_input", "verified injection is invalid")
    body = verified.manifest.body
    packets = body["packets"]
    _verified_ref(verified.prepared_packet_bytes, packets["prepared_packet_ref"], "prepared packet")
    _verified_ref(verified.legacy_additional_context_bytes, packets["legacy_additional_context_ref"], "legacy context")
    _verified_ref(verified.serialized_response_bytes, packets["serialized_response_ref"], "serialized response")
    if len(verified.prepared_packet_bytes.decode("utf-8")) != packets["packet_chars"]:
        raise InjectionAuditError("tampered", "prepared packet character count differs")
    receipts = [_plain(item) for item in verified.receipts]
    return {
        "format": "codex-memory.injection-explain.v1",
        "injection_id": verified.manifest.injection_id,
        "manifest_sha256": verified.manifest.manifest_sha256,
        "source": _plain(body["source"]),
        "request": _plain(body["request"]),
        "snapshot": _plain(body["snapshot"]),
        "identities": _plain(body["identities"]),
        "selected": _plain(body["candidates"]["final_selected"]),
        "trace": _plain(body["candidates"]["query_result"]["trace"]),
        "omissions": {key: _plain(body["candidates"][key]) for key in ("budget_omitted_ids", "build_omissions", "duplicate_ids", "suppressed_ids", "omission_counts")},
        "evidence": _plain(body["evidence"]),
        "packets": {
            "assembly_id": packets["assembly_id"],
            "prepared_packet_sha256": packets["prepared_packet_ref"]["sha256"],
            "prepared_packet_chars": packets["packet_chars"],
            "legacy_additional_context_sha256": packets["legacy_additional_context_ref"]["sha256"],
            "serialized_response_sha256": packets["serialized_response_ref"]["sha256"],
        },
        "delivery": _plain(body["delivery"]),
        "receipts": {"status": "unknown" if not receipts else "verified", "items": receipts},
    }
