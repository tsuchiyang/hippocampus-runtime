"""Independent regressions for the delivered Linux/Windows blind-test findings."""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO, TextIOWrapper
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from codex_memory import cli
from codex_memory import structured_ledger
from codex_memory.candidate_memory import capture_candidate
from codex_memory.checkpoint import CheckpointService
from codex_memory.hook_input import HookInputError, parse_hook_payload
from codex_memory.ledger_common import ValidationError, validate_evidence_uris
from codex_memory.paths import RuntimePathError, RuntimePaths, require_safe_file
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.structured_ledger import propose_record, show_record, verify_ledger


def _payload(root: Path, event: str = "UserPromptSubmit", **extra: object) -> dict:
    return {"hook_event_name": event, "cwd": str(root), "session_id": "blind-fix-session",
            "turn_id": "blind-fix-turn", "prompt": "Synthetic recovery objective."} | extra


def _invoke(root: Path, command: str, event: str, **extra: object) -> tuple[int, str, str]:
    out, err = StringIO(), StringIO()
    code = cli.main([command], StringIO(json.dumps(_payload(root, event, **extra))), out, err)
    return code, out.getvalue(), err.getvalue()


def _checkpoint(root: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    assert _invoke(root, "user-prompt-submit", "UserPromptSubmit")[0] == 0
    assert json.loads(_invoke(root, "pre-compact", "PreCompact", trigger="manual")[1]) == {"continue": True}
    assert json.loads(_invoke(root, "post-compact", "PostCompact", trigger="manual")[1]) == {"continue": True}
    db = root / ".codex-memory/db/memory.sqlite3"
    with sqlite3.connect(db) as connection:
        row = connection.execute("SELECT checkpoint_id, recovery_required FROM checkpoints").fetchone()
    assert row[1] == 1
    return db, row[0]


@pytest.mark.parametrize("fault", ["write", "short_write", "flush"])
def test_recovery_transport_failure_keeps_checkpoint_retryable(tmp_path: Path, fault: str) -> None:
    db, checkpoint_id = _checkpoint(tmp_path)

    class FaultyOutput(StringIO):
        def write(self, value: str) -> int:
            if fault == "write":
                raise BrokenPipeError("synthetic disconnected receiver")
            if fault == "short_write":
                return super().write(value[:-1])
            return super().write(value)

        def flush(self) -> None:
            if fault == "flush":
                raise BrokenPipeError("synthetic buffered delivery failure")
            return super().flush()

    output, error = FaultyOutput(), StringIO()
    try:
        code = cli.main(["session-start"], StringIO(json.dumps(_payload(tmp_path, "SessionStart",
                        source="compact"))), output, error)
    except BrokenPipeError:
        code = 1  # Preserve the baseline failure, then inspect actual durable state.
    with sqlite3.connect(db) as connection:
        state = connection.execute("SELECT recovery_required, recovery_delivered_at FROM checkpoints").fetchone()
    assert state == (1, None)
    assert code != 0
    retry = _invoke(tmp_path, "session-start", "SessionStart", source="compact")
    assert retry[0] == 0 and retry[2] == ""
    context = json.loads(retry[1])["hookSpecificOutput"]["additionalContext"]
    assert checkpoint_id in context and "Synthetic recovery objective." in context
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT recovery_required FROM checkpoints").fetchone()[0] == 0


def test_installed_entry_closed_stdout_retains_recovery(tmp_path: Path) -> None:
    db, checkpoint_id = _checkpoint(tmp_path)
    entry = Path(sys.executable).parent / ("codex-memory-hook.exe" if os.name == "nt" else "codex-memory-hook")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("CODEX")}
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen([str(entry), "session-start"], cwd=tmp_path, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stdin is not None and process.stderr is not None
    process.stdout.close()  # Real OS pipe, with no receiver before entry dispatches.
    try:
        process.stdin.write(json.dumps(_payload(tmp_path, "SessionStart", source="compact")).encode())
        process.stdin.close()
        process.wait(timeout=15)
        diagnostic = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()
    assert process.returncode != 0
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT recovery_required, recovery_delivered_at FROM checkpoints").fetchone() == (1, None)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    retry = _invoke(tmp_path, "session-start", "SessionStart", source="compact")
    assert retry[0] == 0 and checkpoint_id in retry[1]
    assert len(diagnostic) < 4096


def test_recovery_packet_uses_the_checkpoint_being_delivered(tmp_path: Path) -> None:
    db, checkpoint_id = _checkpoint(tmp_path)
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT recovery_delivered_at FROM checkpoints"
        ).fetchone() == (None,)
    result = _invoke(tmp_path, "session-start", "SessionStart", source="compact")
    assert result[0] == 0 and result[2] == ""
    context = json.loads(result[1])["hookSpecificOutput"]["additionalContext"]
    protocol = next(line for line in context.splitlines() if line.startswith("context session="))
    assert "window=" + json.dumps(checkpoint_id) in protocol
    assert 'window="startup"' not in protocol


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sqlite_sidecar_disappearing_between_stat_and_resolve_is_not_a_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    paths.ensure()
    leaf = Path(str(paths.db_path) + suffix)
    leaf.write_bytes(b"synthetic ephemeral sidecar")
    resolve = Path.resolve
    removed: list[bool] = []

    def disappear(path: Path, *args, **kwargs):
        if path == leaf and not removed:
            leaf.unlink()
            removed.append(True)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", disappear)
    paths.validate()
    assert removed == [True]
    assert not leaf.exists()


def test_required_leaf_disappearing_still_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    leaf = tmp_path / "required.bin"
    leaf.write_bytes(b"required")
    resolve = Path.resolve

    def disappear(path: Path, *args, **kwargs):
        if path == leaf:
            leaf.unlink(missing_ok=True)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", disappear)
    with pytest.raises(RuntimePathError):
        require_safe_file(leaf, tmp_path, require_exists=True)


@pytest.mark.parametrize("field,event", [("prompt", "UserPromptSubmit"),
                                         ("tool_response", "PostToolUse"),
                                         ("tool_input", "PostToolUse")])
def test_hook_field_limits_reject_oversized_payloads(tmp_path: Path, field: str, event: str) -> None:
    payload = _payload(tmp_path, event, tool_name="synthetic", tool_use_id="synthetic-tool",
                       tool_input={}, tool_response={})
    payload[field] = "x" * (65_537 if field == "prompt" else 262_145)
    with pytest.raises(HookInputError, match="large|limit"):
        parse_hook_payload(payload)


def test_hook_prompt_limit_is_utf8_bytes_not_characters(tmp_path: Path) -> None:
    assert parse_hook_payload(_payload(tmp_path, prompt="x" * 65_536)).prompt == "x" * 65_536
    with pytest.raises(HookInputError, match="large|limit"):
        parse_hook_payload(_payload(tmp_path, prompt="中" * 21_846))


def test_stdin_read_is_bounded_and_rejects_before_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MeasuredInput(StringIO):
        requested: list[int] = []

        def read(self, size: int = -1) -> str:
            self.requested.append(size)
            return super().read(size)

    stream = MeasuredInput(json.dumps(_payload(tmp_path, extra="x" * 1_048_577)))
    attempted: list[bool] = []
    monkeypatch.setattr(cli, "_resolve_repo_root", lambda _: attempted.append(True))
    output, error = StringIO(), StringIO()
    code = cli.main(["user-prompt-submit"], stream, output, error)
    assert stream.requested == [1_048_577]
    assert code == 2 and output.getvalue() == ""
    assert attempted == [] and not (tmp_path / ".codex-memory").exists()


def test_event_store_rechecks_limit_before_any_blob_or_row(runtime) -> None:
    small = _payload(runtime.paths.repo_root)
    envelope = parse_hook_payload(small)
    huge = "x" * 200_000
    envelope = replace(envelope, prompt=huge, raw=small | {"prompt": huge})
    with pytest.raises(HookInputError, match="large|limit"):
        runtime.events.append(envelope)
    assert runtime.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert list(runtime.paths.blob_sha256_dir.rglob("*.bin")) == []


def test_recovery_settlement_failure_keeps_retryable_state_and_one_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, checkpoint_id = _checkpoint(tmp_path)

    def fail_settlement(*_args):
        raise sqlite3.OperationalError("synthetic settlement I/O failure")

    with monkeypatch.context() as scope:
        scope.setattr(CheckpointService, "mark_recovery_delivered", fail_settlement)
        failed = _invoke(tmp_path, "session-start", "SessionStart", source="compact")
    assert failed[0] == 1 and failed[1].count("\n") == 1
    assert checkpoint_id in json.loads(failed[1])["hookSpecificOutput"]["additionalContext"]
    assert failed[2] == "codex-memory-hook: recovery settlement failed\n"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT recovery_required, recovery_delivered_at FROM checkpoints").fetchone() == (1, None)
    retry = _invoke(tmp_path, "session-start", "SessionStart", source="compact")
    assert retry[0] == 0 and checkpoint_id in retry[1]


@pytest.mark.parametrize("fault", ["redirect", "dangling_redirect", "permission", "io"])
def test_sidecar_disappearance_exception_does_not_excuse_other_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    paths.ensure()
    leaf = Path(str(paths.db_path) + "-wal")
    leaf.write_bytes(b"sidecar")
    outside = tmp_path / "external.bin"
    if fault == "redirect":
        outside.write_bytes(b"sentinel")
    resolve = Path.resolve
    attacked: list[bool] = []

    def fail(path: Path, *args, **kwargs):
        if path == leaf and not attacked:
            attacked.append(True)
            if fault in {"redirect", "dangling_redirect"}:
                leaf.unlink()
                try:
                    leaf.symlink_to(outside)
                except OSError as error:
                    pytest.skip(f"symlink capability unavailable: winerror={getattr(error, 'winerror', None)}")
            elif fault == "permission":
                raise PermissionError("synthetic denied resolution")
            else:
                raise OSError(5, "synthetic I/O failure")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail)
    with pytest.raises(RuntimePathError):
        paths.validate()
    assert attacked == [True]
    if fault == "redirect":
        assert outside.read_bytes() == b"sentinel"


def test_eight_process_writes_and_reads_need_no_blanket_retry(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    entry = Path(sys.executable).parent / ("codex-memory-hook.exe" if os.name == "nt" else "codex-memory-hook")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("CODEX")}

    def invoke(index: int):
        args = ["record", "propose", "--type", "decision", "--scope", "project",
                "--scope-key", "synthetic", "--topic-key", f"parallel.item{index}",
                "--statement", f"Synthetic record {index}", "--idempotency-key", f"parallel-{index}"]
        return subprocess.run([str(entry), *args], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=20)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(24)))
    assert [result.returncode for result in results] == [0] * 24
    assert all(not result.stderr for result in results)
    with sqlite3.connect(tmp_path / ".codex-memory/db/memory.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 24
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("uri", ["javascript:alert(1)", "JaVaScRiPt:alert(1)",
                                "vbscript:msgbox(1)", "data:text/html,<script>alert(1)</script>"])
def test_new_proposal_rejects_executable_evidence_without_changing_history(runtime, uri: str) -> None:
    # Historical syntax validation stays compatible; the new-input policy is separate.
    assert validate_evidence_uris((uri,)) == (uri,)
    before = verify_ledger(runtime.connection)
    with pytest.raises(ValidationError, match="evidence"):
        propose_record(runtime.connection, record_type="decision", scope_kind="project",
            scope_key="synthetic", topic_key="evidence.policy", statement="Synthetic decision.",
            value=None, evidence_uris=(uri,), idempotency_key="unsafe-evidence")
    assert verify_ledger(runtime.connection) == before


@pytest.mark.parametrize("name", ["a" * 70_000, "a" * 256, "中" * 128,
                                 "a/" * 2100 + "file.py"],
                         ids=["oversized-ascii", "component-boundary", "utf8-component", "total-path"])
def test_portable_name_rejects_overlong_component_or_relative_path(name: str) -> None:
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("portable_verify_length", Path(__file__).resolve().parents[1] / "portable/verify.py")
    assert spec is not None and spec.loader is not None
    verifier = module_from_spec(spec)
    spec.loader.exec_module(verifier)
    with pytest.raises(ValueError, match="name|length"):
        verifier.safe_name(name)


def test_existing_evidence_remains_verifiable_under_new_ingress_policy(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    # Seed a historically valid record with the old syntax-only ingress. The
    # actual ledger hashing/inserts/verification remain the real implementation.
    with monkeypatch.context() as fixture:
        fixture.setattr(structured_ledger, "validate_new_evidence_uris", validate_evidence_uris)
        record = propose_record(runtime.connection, record_type="decision", scope_kind="project",
            scope_key="synthetic", topic_key="legacy.evidence", statement="Legacy synthetic record.",
            value=[1, 2], evidence_uris=("javascript:alert(1)",), idempotency_key="legacy-evidence")
    assert verify_ledger(runtime.connection).records == 1
    assert show_record(runtime.connection, record.record_id).value == [1, 2]
    with pytest.raises(ValidationError):
        propose_record(runtime.connection, record_type="decision", scope_kind="project",
            scope_key="synthetic", topic_key="new.evidence", statement="New synthetic record.",
            value=None, evidence_uris=("javascript:alert(1)",), idempotency_key="new-unsafe")
    assert verify_ledger(runtime.connection).records == 1


def test_candidate_capture_rejects_executable_evidence_before_persistence(runtime) -> None:
    from codex_memory.blob_store import BlobStore
    with pytest.raises(ValidationError, match="evidence"):
        capture_candidate(runtime.connection, BlobStore(runtime.paths.blobs_dir),
            project_scope_key="synthetic", scope_kind="project", scope_key="synthetic",
            record_type="assumption", topic_key="candidate.safety", statement="Synthetic assumption.",
            session_id="synthetic-session", turn_id="synthetic-turn", context_id="startup",
            idempotency_key="unsafe-candidate", evidence_uris=("javascript:alert(1)",))
    assert verify_ledger(runtime.connection).candidates == 0


@pytest.mark.parametrize("case", ["oversized-prompt", "crlf-wire-bytes"])
def test_oversized_input_is_rejected_by_installed_entry_without_runtime_creation(tmp_path: Path, case: str) -> None:
    entry = Path(sys.executable).parent / ("codex-memory-hook.exe" if os.name == "nt" else "codex-memory-hook")
    raw = (json.dumps(_payload(tmp_path, prompt="x" * 200_000)).encode("utf-8")
           if case == "oversized-prompt" else b"\r\n" * 600_000 + json.dumps(_payload(tmp_path)).encode("utf-8"))
    result = subprocess.run([str(entry), "user-prompt-submit"], cwd=tmp_path,
        input=raw, capture_output=True, timeout=10)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr.replace(b"\r\n", b"\n") == b"codex-memory-hook: invalid hook input\n"
    assert not (tmp_path / ".codex-memory").exists()


def test_missing_canonical_query_stays_fail_closed_and_does_not_initialize(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    entry = Path(sys.executable).parent / ("codex-memory-hook.exe" if os.name == "nt" else "codex-memory-hook")
    result = subprocess.run([str(entry), "retrieval", "query", "--session-id", "synthetic",
        "--turn-id", "synthetic", "--context-id", "startup", "--query", "synthetic"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    document = json.loads(result.stdout)
    assert result.returncode == 1 and document["candidates"] == []
    assert "canonical_unavailable" in document["degraded_reasons"]
    assert document["canonical_highwater"] is None
    assert not (tmp_path / ".codex-memory").exists()


@pytest.mark.parametrize("family", ["canonical", "retrieval"])
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_retrieval_validation_handles_real_sidecar_disappearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, suffix: str,
) -> None:
    runtime_paths = RuntimePaths.for_repo(tmp_path)
    runtime_paths.ensure()
    runtime_paths.db_path.write_bytes(b"synthetic path-validation fixture, not opened by SQLite")
    paths = RetrievalPaths.for_repo(tmp_path)
    if family == "retrieval":
        paths.retrieval_root.mkdir()
    database = paths.canonical_database_path if family == "canonical" else paths.database_path
    leaf = Path(str(database) + suffix)
    leaf.write_bytes(b"synthetic temporary sidecar")
    resolve = Path.resolve
    removed: list[bool] = []

    def disappear(path: Path, *args, **kwargs):
        if path == leaf and not removed:
            leaf.unlink()
            removed.append(True)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", disappear)
    paths.validate()
    assert removed == [True]


@pytest.mark.parametrize("raw_case", ["crlf-overflow", "exact-byte-limit", "invalid-utf8"])
def test_real_binary_stdin_counts_bytes_before_newline_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_case: str,
) -> None:
    base = json.dumps(_payload(tmp_path)).encode("utf-8")
    if raw_case == "crlf-overflow":
        raw = b"\r\n" * 600_000 + base
    elif raw_case == "exact-byte-limit":
        raw = b" " * (1_048_576 - len(base)) + base
    else:
        raw = base.replace(b"Synthetic recovery objective.", b"\xff")
    attempted: list[bool] = []

    def stop_before_runtime(_cwd):
        attempted.append(True)
        raise RuntimeError("readonly pre-initialization sentinel")

    monkeypatch.setattr(cli, "_resolve_repo_root", stop_before_runtime)
    with TextIOWrapper(BytesIO(raw), encoding="latin1", newline=None) as stdin:
        code = cli.main(["user-prompt-submit"], stdin, StringIO(), StringIO())
    assert code == (1 if raw_case == "exact-byte-limit" else 2)
    assert attempted == ([True] if raw_case == "exact-byte-limit" else [])
    assert not (tmp_path / ".codex-memory").exists()
