from __future__ import annotations

import hashlib
from importlib import import_module
import sqlite3

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.memory_context import build_memory_packet
from codex_memory.retrieval.contracts import (
    EpistemicContext,
    RetrievalMode,
    RetrievalQueryResult,
    RetrievalRequest,
    RetrievalScope,
    RetrievalStatus,
)
from codex_memory.retrieval.errors import RetrievalValidationError
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from codex_memory.structured_ledger import apply_action, propose_record


def _assembler():
    module = import_module("codex_memory.retrieval.assembler")
    if not hasattr(module, "build_phase2_packet"):
        pytest.fail("P2.4 read-only packet orchestration API is missing", pytrace=False)
    return module


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retrieval_package_exports_offline_phase2_packet_api() -> None:
    retrieval = import_module("codex_memory.retrieval")

    assert retrieval.build_phase2_packet is _assembler().build_phase2_packet
    assert retrieval.AssembledPacket is _assembler().AssembledPacket
    assert retrieval.PacketAuditSnapshot is _assembler().PacketAuditSnapshot
    assert retrieval.PriorContextSelection is _assembler().PriorContextSelection


def test_real_retrieval_result_builds_read_only_packet_and_keeps_legacy_v1_stable(
    runtime,
) -> None:
    scope_key = str(runtime.paths.repo_root)
    marker = "p24-unique-storage-marker"
    record = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=scope_key,
        topic_key="runtime.packet-source",
        statement=f"Use the canonical ledger for {marker}.",
        value={"marker": marker},
        evidence_uris=(),
        idempotency_key="p24-integration-record",
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key="p24-integration-activate",
    )
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths)
    assert service.rebuild().status.value == "current"
    request = RetrievalRequest(
        query=marker,
        session_id="session_p24",
        turn_id="turn_p24",
        context_id="startup",
        prompt_event_uri=None,
        scope=RetrievalScope(scope_key),
        epistemic_context=EpistemicContext(),
        mode=RetrievalMode.OFF,
        max_chars=4000,
    )
    result = service.query(request)
    assert record.record_id in {
        candidate.candidate_id for candidate in result.candidates
    }
    legacy_before = build_memory_packet(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        session_id=request.session_id,
        turn_id=request.turn_id,
        context_id=request.context_id,
        scope_key=scope_key,
        prompt_event_uri=None,
        query=request.query,
        max_chars=request.max_chars,
    )
    canonical_before = _file_sha256(runtime.paths.db_path)

    assembled = _assembler().build_phase2_packet(
        paths,
        request=request,
        result=result,
    )

    canonical_after = _file_sha256(runtime.paths.db_path)
    legacy_after = build_memory_packet(
        runtime.connection,
        BlobStore(runtime.paths.blobs_dir),
        session_id=request.session_id,
        turn_id=request.turn_id,
        context_id=request.context_id,
        scope_key=scope_key,
        prompt_event_uri=None,
        query=request.query,
        max_chars=request.max_chars,
    )
    assert assembled.packet.startswith("[CODEX MEMORY PROTOCOL v2]\n")
    assert record.record_id in assembled.packet
    assert assembled.packet_chars <= request.max_chars
    assert canonical_after == canonical_before
    assert legacy_after == legacy_before
    assert legacy_after.startswith("[CODEX MEMORY PROTOCOL v1]\n")
    assert assembled.audit_snapshot is not None
    assert assembled.audit_snapshot.canonical_snapshot.content_fingerprint
    assert assembled.audit_snapshot.request_binding["session_id"] == request.session_id
    assert "query" not in assembled.audit_snapshot.request_binding
    assert assembled.audit_snapshot.result_metadata["status"] == result.status.value
    assert assembled.audit_snapshot.result_metadata["mode"] == result.mode.value
    assert assembled.audit_snapshot.final_selected == tuple(
        item.to_dict() for item in assembled.selected
    )
    assert tuple(
        pack["card"]["id"]
        for pack in assembled.audit_snapshot.evidence_packs
    ) == tuple(item.candidate_id for item in assembled.selected)


def test_packet_audit_snapshot_stays_bound_after_canonical_changes(runtime) -> None:
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths)
    assert service.rebuild().status.value == "current"
    request = RetrievalRequest(
        query="frozen audit",
        session_id="session_p25",
        turn_id="turn_p25",
        context_id="prompt",
        prompt_event_uri=None,
        scope=RetrievalScope(str(runtime.paths.repo_root)),
        mode=RetrievalMode.OFF,
    )
    result = service.query(request)
    assembled = _assembler().build_phase2_packet(paths, request=request, result=result)
    before = assembled.audit_snapshot.canonical_snapshot
    propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(runtime.paths.repo_root),
        topic_key="runtime.after-audit",
        statement="This change happens after packet assembly.",
        value=None,
        evidence_uris=(),
        idempotency_key="p25-after-audit",
    )
    runtime.connection.commit()
    assert assembled.audit_snapshot.canonical_snapshot == before
    canonical_snapshot = import_module("codex_memory.retrieval.canonical_snapshot")
    assert assembled.audit_snapshot.canonical_snapshot.content_fingerprint != canonical_snapshot.capture_canonical_snapshot(paths).content_fingerprint


def test_packet_build_rejects_retrieval_result_from_an_older_canonical_snapshot(
    runtime,
) -> None:
    scope_key = str(runtime.paths.repo_root)
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths)
    assert service.rebuild().status.value == "current"
    request = RetrievalRequest(
        query="stale packet",
        session_id="session_p24",
        turn_id="turn_p24",
        context_id="startup",
        prompt_event_uri=None,
        scope=RetrievalScope(scope_key),
        mode=RetrievalMode.OFF,
    )
    stale_result = service.query(request)
    propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=scope_key,
        topic_key="runtime.changed-after-query",
        statement="This append advances the canonical snapshot.",
        value=None,
        evidence_uris=(),
        idempotency_key="p24-after-query-change",
    )
    runtime.connection.commit()

    with pytest.raises(
        RetrievalValidationError,
        match="does not match canonical snapshot",
    ):
        _assembler().build_phase2_packet(
            paths,
            request=request,
            result=stale_result,
        )


def test_public_packet_build_classifies_second_schema_meta_io_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    runtime,
) -> None:
    retrieval = import_module("codex_memory.retrieval")
    canonical_snapshot = import_module(
        "codex_memory.retrieval.canonical_snapshot"
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    snapshot = canonical_snapshot.capture_canonical_snapshot(paths)
    request = RetrievalRequest(
        query="snapshot read failure",
        session_id="session_p24",
        turn_id="turn_p24",
        context_id="startup",
        prompt_event_uri=None,
        scope=RetrievalScope(str(runtime.paths.repo_root)),
        mode=RetrievalMode.OFF,
    )
    result = RetrievalQueryResult(
        status=RetrievalStatus.CURRENT,
        mode=RetrievalMode.OFF,
        generation=None,
        canonical_highwater=snapshot.highwater,
        indexed_highwater=snapshot.highwater,
        index_stale=False,
        degraded_reasons=(),
        candidates=(),
        trace=(),
    )
    canonical_before = runtime.paths.db_path.read_bytes()

    class FaultConnection(sqlite3.Connection):
        schema_reads = 0

        def execute(self, sql, parameters=()):
            if (
                sql.lstrip().lower().startswith("select")
                and "schema_meta" in sql.lower()
            ):
                self.schema_reads += 1
                if self.schema_reads == 2:
                    error = sqlite3.OperationalError(
                        "isolated synthetic read I/O failure"
                    )
                    error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                    error.sqlite_errorname = "SQLITE_IOERR"
                    raise error
            return super().execute(sql, parameters)

    def open_fault_connection(path):
        connection = sqlite3.connect(
            path.resolve(strict=True).as_uri() + "?mode=ro",
            uri=True,
            isolation_level=None,
            factory=FaultConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA recursive_triggers=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA query_only=ON")
        return connection

    monkeypatch.setattr(
        canonical_snapshot,
        "connect_readonly_database",
        open_fault_connection,
    )

    with pytest.raises(
        canonical_snapshot.CanonicalSnapshotUnavailableError
    ) as raised:
        retrieval.build_phase2_packet(
            paths,
            request=request,
            result=result,
        )

    assert (
        raised.value.stage
        is canonical_snapshot.CanonicalValidationStage.SCHEMA_QUERY
    )
    chain = []
    cause = raised.value
    while cause is not None and cause not in chain:
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    assert any(
        getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_IOERR
        for error in chain
    )
    assert runtime.paths.db_path.read_bytes() == canonical_before


def test_offline_packet_builder_rejects_active_before_reading_runtime_paths(
    tmp_path,
) -> None:
    request = RetrievalRequest(
        query="must stay offline",
        session_id="session_p24",
        turn_id="turn_p24",
        context_id="startup",
        prompt_event_uri=None,
        scope=RetrievalScope(str(tmp_path)),
        mode=RetrievalMode.ACTIVE,
    )
    result = RetrievalQueryResult(
        status=RetrievalStatus.ABSENT,
        mode=RetrievalMode.ACTIVE,
        generation=None,
        canonical_highwater=None,
        indexed_highwater=None,
        index_stale=True,
        degraded_reasons=(),
        candidates=(),
        trace=(),
    )

    with pytest.raises(RetrievalValidationError, match="active retrieval mode"):
        _assembler().build_phase2_packet(
            RetrievalPaths.for_repo(tmp_path),
            request=request,
            result=result,
        )
