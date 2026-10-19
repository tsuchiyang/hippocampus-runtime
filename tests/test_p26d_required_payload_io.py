from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from codex_memory import atomic, cli
from codex_memory.blob_store import BlobStore
from codex_memory.injection_store import InjectionPaths, InjectionStore
from codex_memory.retrieval import isolated_hook
from test_p26c_selected_host import selected_host_lab  # fixture dependency
from test_p26d_compact import compact_lab, invoke
from test_p26d_promotion_race import _memory_rows


@pytest.mark.parametrize("failure_at", ["fsync", "replace"])
def test_required_payload_io_failure_refuses_output_before_optional_packet_and_recovers(
    compact_lab, monkeypatch, failure_at,
):
    """Storage-seam faults, not a reproduction of the historical slow I/O.

    Catch swallowing a required payload error or applying optional-packet
    Legacy fallback before the source event exists. Keep the actual selected
    CLI, atomic write/cleanup, SQLite, guards and later recovery path real.
    """
    runtime, _marker, _hooks = compact_lab
    root = runtime.paths.repo_root
    turn = "required-payload-" + failure_at
    prompt = "What is the HOST-P26C-EXAMPLE verification code?"
    payload = {
        "hook_event_name": "UserPromptSubmit", "session_id": "selected-session",
        "cwd": str(root), "transcript_path": str(root / "rollout.jsonl"),
        "turn_id": turn, "prompt": prompt,
    }
    expected_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(expected_bytes).hexdigest()
    final_blob = runtime.paths.blobs_dir / "sha256" / digest[:2] / (digest + ".bin")
    temp_pattern = "." + digest + ".bin.*"
    before = tuple(runtime.connection.iterdump())
    memory_before = _memory_rows(runtime.connection)
    assert not final_blob.exists()
    fault_hits = []
    original_fsync, original_replace = atomic.os.fsync, atomic.os.replace
    original_factory = cli._construct_runtime
    opened_connections = []

    def actual_runtime(*args, **kwargs):
        value = original_factory(*args, **kwargs)
        opened_connections.append(value.connection)
        return value

    monkeypatch.setattr(cli, "_construct_runtime", actual_runtime)

    def fail_fsync(descriptor):
        opened = atomic.os.fstat(descriptor)
        for path in final_blob.parent.glob(temp_pattern):
            candidate = path.stat()
            if (candidate.st_dev, candidate.st_ino) == (opened.st_dev, opened.st_ino):
                assert path.read_bytes() == expected_bytes
                fault_hits.append("required_payload_fsync")
                raise OSError("synthetic private payload I/O diagnostic")
        return original_fsync(descriptor)

    def fail_replace(source, destination):
        if destination == final_blob:
            assert source.read_bytes() == expected_bytes
            fault_hits.append("required_payload_replace")
            raise OSError("synthetic private payload I/O diagnostic")
        return original_replace(source, destination)

    def forbidden_optional_packet(*args, **kwargs):
        pytest.fail("optional packet/fallback must not run without a stored source event")

    with monkeypatch.context() as fault:
        fault.setattr(atomic.os, failure_at, fail_fsync if failure_at == "fsync" else fail_replace)
        fault.setattr(isolated_hook, "prepare_isolated_packet", forbidden_optional_packet)
        fault.setattr(isolated_hook, "_legacy_and_source", forbidden_optional_packet)
        code, stdout, stderr = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
            turn_id=turn, prompt=prompt)

    assert fault_hits == ["required_payload_" + failure_at], "the actual target write must fail once"
    assert code == 1 and stdout == ""
    metadata = json.loads(stderr)
    fault_observation = {"exit_status": code,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "metadata": metadata}
    assert metadata["delivery"] == "not_emitted"
    # The held-directory context maps the original OSError to the stable
    # InjectionAuditError('unavailable'); do not broaden the accepted enum.
    assert metadata["reason"] == "unavailable"
    assert metadata["cleanup_status"] == "complete"
    assert len(opened_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened_connections[0].execute("SELECT 1")
    assert metadata["injection_id"] is None and metadata["manifest_sha256"] is None
    assert "synthetic private payload I/O diagnostic" not in stderr
    assert tuple(runtime.connection.iterdump()) == before
    assert not final_blob.exists() and list(final_blob.parent.glob(temp_pattern)) == []
    assert not InjectionPaths.for_repo(root).injections_dir.exists()

    # A new development invocation after removing the injected fault is not
    # a replay/reclassification of the failed historical soak or a host run.
    code, stdout, stderr = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
        turn_id=turn, prompt=prompt)
    metadata = json.loads(stderr)
    assert code == 0 and metadata["delivery"] == "emitted", stderr
    assert len(opened_connections) == 2
    with pytest.raises(sqlite3.ProgrammingError):
        opened_connections[1].execute("SELECT 1")
    verified = InjectionStore(InjectionPaths.for_repo(root), BlobStore(runtime.paths.blobs_dir)).load(
        metadata["injection_id"])
    assert verified.serialized_response_bytes == stdout.encode("utf-8")
    assert "HOST-P26C-EXAMPLE" in stdout
    assert [tuple(row) for row in runtime.connection.execute(
        "SELECT session_id, turn_id, payload_blob_sha256 FROM events")] == [
        ("selected-session", turn, digest)]
    assert final_blob.read_bytes() == expected_bytes
    assert list(final_blob.parent.glob(temp_pattern)) == []
    assert _memory_rows(runtime.connection) == memory_before
    (root / "required-payload-observation.json").write_text(json.dumps({
        "failure_at": failure_at, "fault_hits": fault_hits,
        "fault": fault_observation,
        "recovery_exit_status": code, "recovery_metadata": metadata,
        "payload_sha256": digest,
        "source_event_count": runtime.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "evidence_level": "real in-process selected CLI; target storage/API OSError injection, not OS timing reproduction",
    }, sort_keys=True), encoding="utf-8")
