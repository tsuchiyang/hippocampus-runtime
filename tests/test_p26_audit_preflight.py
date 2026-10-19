from __future__ import annotations

import sqlite3

import pytest

from codex_memory.injection_store import InjectionPaths
from codex_memory.retrieval import injection_audit, isolated_cli, isolated_hook
from codex_memory.retrieval.service import RetrievalService
from test_p26b_prompt import _invoke, _memory_contents, _metadata, prompt_lab


def test_known_blocked_audit_root_rejects_before_optional_work(prompt_lab, monkeypatch):
    """Reject-only cost reduction, not a reproduction of a historical timeout."""
    runtime, _record_id = prompt_lab
    root = InjectionPaths.for_repo(runtime.paths.repo_root).injections_dir
    sentinel = b"synthetic blocked audit destination"
    root.write_bytes(sentinel)
    memory_before = _memory_contents(runtime)
    actual_query, actual_put = RetrievalService.query, injection_audit.put_audit_blob
    work = []

    def query(*args, **kwargs):
        work.append("query")
        return actual_query(*args, **kwargs)

    def put(*args, **kwargs):
        work.append("audit_blob")
        return actual_put(*args, **kwargs)

    monkeypatch.setattr(RetrievalService, "query", query)
    monkeypatch.setattr(injection_audit, "put_audit_blob", put)
    code, stdout, stderr = _invoke(prompt_lab)
    metadata = _metadata(stderr)
    assert code == 0, stderr
    assert metadata["delivery"] == "legacy_fallback"
    assert metadata["reason"] == "path_unsafe"
    assert metadata["injection_id"] is None and metadata["manifest_sha256"] is None
    assert metadata["cleanup_status"] == "complete"
    assert "沙箱采用 WAL，保留原子事务。" in stdout
    assert root.read_bytes() == sentinel
    assert _memory_contents(runtime) == memory_before
    assert work == [], "known unsafe destination must not consume query/audit work"


def test_unsafe_audit_child_after_precheck_is_still_rejected_at_commit(prompt_lab, monkeypatch):
    runtime, _record_id = prompt_lab
    root = InjectionPaths.for_repo(runtime.paths.repo_root).injections_dir
    root.mkdir()
    sentinel = b"synthetic exchange after optional query"
    actual_query = RetrievalService.query
    queried = []

    def exchange(*args, **kwargs):
        result = actual_query(*args, **kwargs)
        (root / "temp").write_bytes(sentinel)
        queried.append(True)
        return result

    monkeypatch.setattr(RetrievalService, "query", exchange)
    code, stdout, stderr = _invoke(prompt_lab)
    metadata = _metadata(stderr)
    assert queried == [True]
    assert code == 0 and stdout, stderr
    assert metadata["delivery"] == "legacy_fallback"
    assert metadata["reason"] == "path_unsafe"
    assert metadata["injection_id"] is None and metadata["manifest_sha256"] is None
    assert metadata["cleanup_status"] == "complete"
    assert (root / "temp").read_bytes() == sentinel


@pytest.mark.parametrize("damage", ["schema", "source_io"])
def test_blocked_audit_root_cannot_mask_new_canonical_failure(prompt_lab, monkeypatch, damage):
    runtime, _record_id = prompt_lab
    root = InjectionPaths.for_repo(runtime.paths.repo_root).injections_dir
    root.write_bytes(b"synthetic blocked audit destination")
    actual_legacy = isolated_hook._legacy_and_source
    actual_resolve = isolated_cli.resolve_event_evidence
    source_reads = []
    calls = []

    def resolve(*args, **kwargs):
        source_reads.append(True)
        if damage == "source_io" and len(source_reads) == 2:
            error = sqlite3.OperationalError("PRIVATE_CANONICAL_IO")
            error.sqlite_errorcode = sqlite3.SQLITE_IOERR
            raise error
        return actual_resolve(*args, **kwargs)

    def legacy(*args, **kwargs):
        result = actual_legacy(*args, **kwargs)
        calls.append(True)
        if damage == "schema" and len(calls) == 1:
            trigger = runtime.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND tbl_name='memory_records' ORDER BY name"
            ).fetchone()[0]
            runtime.connection.execute(f'DROP TRIGGER "{trigger}"')
        return result

    monkeypatch.setattr(isolated_cli, "resolve_event_evidence", resolve)
    monkeypatch.setattr(isolated_hook, "_legacy_and_source", legacy)
    code, stdout, stderr = _invoke(prompt_lab)
    metadata = _metadata(stderr)
    assert calls, "the initial real canonical/Legacy path must complete"
    if damage == "source_io":
        assert len(source_reads) == 2, "fault must reach the fallback source read"
    assert code == 1 and stdout == ""
    assert metadata["delivery"] == "not_emitted"
    assert metadata["reason"] == ("canonical_invalid" if damage == "schema" else "canonical_unavailable")
    assert metadata["injection_id"] is None and metadata["manifest_sha256"] is None
    assert metadata["cleanup_status"] == "complete"
    assert "PRIVATE_CANONICAL_IO" not in stderr
