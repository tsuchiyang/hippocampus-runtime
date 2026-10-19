"""Shared deterministic packet preparation; callers own opt-in, time and wire."""

from __future__ import annotations

from typing import Callable
import time

from codex_memory.blob_store import BlobStore
from codex_memory.injection_manifest import InjectionAuditError
from codex_memory.injection_store import CommittedInjection, InjectionPaths, InjectionStore, _directories

from .assembler import build_phase2_packet
from .canonical_snapshot import open_verified_canonical_snapshot
from .contracts import DegradedReason, RetrievalRequest
from .deadline import DeadlineBudget
from .injection_audit import _deadline_check, implementation_fingerprint, prepare_shadow_manifest
from .paths import RetrievalPaths
from .service import RetrievalService


class IsolatedPacketFailure(Exception):
    def __init__(self, reason: str, code: int = 1):
        self.reason, self.code = reason, code
        super().__init__(reason)


def prepare_isolated_packet(
    paths: RetrievalPaths, *, request: RetrievalRequest, legacy_packet: str,
    source: dict[str, object], config_sha256: str, deadline: DeadlineBudget,
    started_at: str, prepared_at: Callable[[], str], budget_ms: int,
    encode_wire: Callable[[str, list[str], list[str]], bytes],
    bound_reads: bool = False,
) -> tuple[bytes, CommittedInjection]:
    """Commit/reopen exact bytes before returning; never emit or reset a budget."""
    _deadline_check(deadline)
    audit_paths = InjectionPaths.for_repo(paths.repo_root)
    try:
        # Reject a known unusable destination before expensive derived work.
        # Absence is normal on first use; a successful read-only check is NOT
        # installation authority. Commit/reopen retain their full path guards.
        with _directories(audit_paths.injections_dir, create=False, deadline=deadline):
            pass
    except InjectionAuditError as error:
        if error.reason.value != "missing":
            raise
    _deadline_check(deadline)
    query_started = time.monotonic_ns()
    result = RetrievalService(paths).query(request, **({"deadline": deadline, "bound_reads": True} if bound_reads else {}))
    _deadline_check(deadline)
    if DegradedReason.CANONICAL_INVALID in result.degraded_reasons:
        raise IsolatedPacketFailure("canonical_invalid")
    if DegradedReason.CANONICAL_UNAVAILABLE in result.degraded_reasons:
        raise IsolatedPacketFailure("canonical_unavailable")
    if result.index_stale or result.status.value != "current":
        raise IsolatedPacketFailure("retrieval_degraded")
    query_ms = (time.monotonic_ns() - query_started) // 1_000_000
    assembly_started = time.monotonic_ns()
    assembled = build_phase2_packet(paths, request=request, result=result,
        **({"deadline": deadline} if bound_reads else {}))
    _deadline_check(deadline)
    assembly_ms = (time.monotonic_ns() - assembly_started) // 1_000_000
    raw = encode_wire(assembled.packet, [item.candidate_id for item in assembled.selected],
        list(assembled.degraded_reasons))
    fingerprint = implementation_fingerprint(deadline=deadline)
    identity = {"configured": "disabled", "observed": "disabled", "validated": "disabled"}
    blobs = BlobStore(paths.runtime_root / "blobs")
    manifest = prepare_shadow_manifest(
        request=request, result=result, assembled=assembled, source=source,
        legacy_packet=legacy_packet, legacy_response_bytes=raw,
        environment={"provider": identity, "model": identity, "preprocessing": identity,
            "config_sha256": config_sha256, "implementation_sha256": fingerprint,
            "legacy_delivery": "unknown", "host_acknowledgement": "unknown"},
        timing={"started_at": started_at, "prepared_at": prepared_at(),
            "query_duration_ms": query_ms, "assembly_duration_ms": assembly_ms,
            "prepare_duration_ms": (time.monotonic_ns() - deadline.started_ns) // 1_000_000,
            "budget_ms": budget_ms}, blobs=blobs, deadline=deadline,
    )
    store = InjectionStore(audit_paths, blobs)
    committed = store.commit(manifest, deadline=deadline)
    _deadline_check(deadline)
    verified = store.load(committed.injection_id, deadline=deadline)
    _deadline_check(deadline)
    if verified.manifest.to_bytes() != manifest.to_bytes() or verified.serialized_response_bytes != raw:
        raise InjectionAuditError("tampered")
    with open_verified_canonical_snapshot(paths, **({"deadline": deadline} if bound_reads else {})) as (_connection, latest):
        if (latest.highwater != assembled.audit_snapshot.canonical_snapshot.highwater
                or latest.content_fingerprint != assembled.audit_snapshot.canonical_snapshot.content_fingerprint):
            raise IsolatedPacketFailure("canonical_changed")
    _deadline_check(deadline)
    return raw, committed
