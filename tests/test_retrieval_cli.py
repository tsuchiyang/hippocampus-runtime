from __future__ import annotations

import json
import sqlite3
from io import StringIO
from pathlib import Path
import os
import shutil
import subprocess
import hashlib

import pytest

from codex_memory.database import connect_database, migrate_database
from codex_memory.paths import RuntimePaths
from codex_memory.paths import RuntimePathError


def _repository(tmp_path: Path, name: str = "Repository With Spaces") -> Path:
    repository = tmp_path / name
    repository.mkdir()
    subprocess.run(
        ["git", "init", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = RuntimePaths.for_repo(repository)
    paths.ensure()
    connection = connect_database(paths.db_path)
    try:
        migrate_database(connection)
    finally:
        connection.close()
    return repository


def _run(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("codex-memory-hook")
    assert executable is not None, "installed codex-memory-hook entry point is required"
    return subprocess.run(
        [executable, *arguments],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )


def _canonical_object(stdout: str) -> dict[str, object]:
    assert stdout.endswith("\n") and stdout.count("\n") == 1
    value = json.loads(stdout)
    assert isinstance(value, dict)
    assert stdout == json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    return value


def test_real_entry_point_exposes_all_five_canonical_retrieval_commands(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    absent = _run(repository, "retrieval", "status")
    initialized = _run(repository, "retrieval", "init")
    verified = _run(repository, "retrieval", "verify")
    rebuilt = _run(repository, "retrieval", "rebuild")
    caught_up = _run(repository, "retrieval", "catch-up")

    assert (absent.returncode, absent.stderr) == (0, "")
    assert _canonical_object(absent.stdout)["status"] == "absent"
    documents = [_canonical_object(result.stdout) for result in (initialized, verified, rebuilt, caught_up)]
    assert [(result.returncode, result.stderr) for result in (initialized, verified, rebuilt, caught_up)] == [
        (0, ""),
        (0, ""),
        (0, ""),
        (0, ""),
    ]
    assert [document["status"] for document in documents] == [
        "current",
        "current",
        "current",
        "current",
    ]
    assert documents[0] == documents[1] == documents[2] == documents[3]
    assert set(documents[0]) == {
        "status",
        "mode",
        "schema_version",
        "active_generation",
        "canonical_highwater",
        "indexed_highwater",
        "canonical_content_fingerprint",
        "provider_fingerprint",
        "model_fingerprint",
        "validation_state",
        "degraded_reason",
        "capabilities",
    }
    assert documents[0]["mode"] == "off"
    # CR-01 added append-only provider-attempt audit tables without changing
    # the flat output shape; the emitted version must report the v4 substrate.
    assert documents[0]["schema_version"] == 4
    assert documents[0]["capabilities"] == []


def test_status_and_verify_are_read_only_and_have_distinct_absent_exit_codes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    retrieval_root = repository / ".codex-memory" / "retrieval"

    status = _run(repository, "retrieval", "status")
    verified = _run(repository, "retrieval", "verify")

    assert status.returncode == 0
    assert verified.returncode == 1
    assert status.stdout == verified.stdout
    assert _canonical_object(status.stdout)["degraded_reason"] == "index_absent"
    assert status.stderr == verified.stderr == ""
    assert not retrieval_root.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("retrieval",),
        ("retrieval", "unknown"),
        ("retrieval", "status", "extra"),
    ],
)
def test_invalid_retrieval_arguments_have_one_bounded_error(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    repository = _repository(tmp_path)

    result = _run(repository, *arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: invalid retrieval input\n"
    assert "Traceback" not in result.stderr


def test_query_malformed_arguments_have_one_bounded_error(tmp_path: Path) -> None:
    """Would fail if the flat query parser leaked argparse diagnostics or accepted bad caps."""
    repository = _repository(tmp_path)

    result = _run(repository, "retrieval", "query", "--session-id", "s", "--turn-id", "t", "--context-id", "c", "--query", "x", "--max-chars", "0")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: invalid retrieval input\n"


def test_semantic_cli_flags_are_exact_and_do_not_allow_abbreviation(
    tmp_path: Path,
) -> None:
    """P2.3 options are opt-in and must not be accepted by prefix."""
    repository = _repository(tmp_path)

    accepted = _run(repository, "retrieval", "rebuild", "--semantic")
    abbreviated = _run(repository, "retrieval", "rebuild", "--sem")
    query_abbreviated = _run(
        repository,
        "retrieval",
        "query",
        "--session-id", "session",
        "--turn-id", "turn",
        "--context-id", "context",
        "--query", "safe query",
        "--disable-den",
    )

    # The complete option is recognized even when no shadow semantic profile is
    # configured; it must take the bounded no-provider path rather than being
    # parsed as a legacy command.
    assert accepted.returncode in {0, 1}
    assert (abbreviated.returncode, abbreviated.stdout, abbreviated.stderr) == (
        2, "", "codex-memory-hook: invalid retrieval input\n",
    )
    assert (query_abbreviated.returncode, query_abbreviated.stdout, query_abbreviated.stderr) == (
        2, "", "codex-memory-hook: invalid retrieval input\n",
    )


def test_semantic_query_disable_flags_are_parsed_independently(
    tmp_path: Path,
) -> None:
    """Would fail until the P2.3 query parser exposes all three exact flags."""
    from codex_memory.retrieval import cli as retrieval_cli

    values = retrieval_cli._parse_query_arguments(
        [
            "--session-id", "session", "--turn-id", "turn", "--context-id", "context",
            "--query", "safe query", "--disable-dense", "--disable-sparse",
            "--disable-reranker", "--explain",
        ]
    )

    assert values["disable_dense"] is True
    assert values["disable_sparse"] is True
    assert values["disable_reranker"] is True
    assert values["explain"] is True


def test_query_rejects_active_mode_without_falling_back_to_ground_truth(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    codex = repository / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text("[retrieval]\nmode = 'active'\n", encoding="utf-8")

    result = _run(
        repository,
        "retrieval", "query", "--session-id", "session", "--turn-id", "turn",
        "--context-id", "context", "--query", "safe query",
    )

    assert (result.returncode, result.stdout, result.stderr) == (
        2, "", "codex-memory-hook: invalid retrieval input\n",
    )


def test_semantic_maintenance_without_valid_shadow_config_never_creates_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_memory.retrieval import cli as retrieval_cli

    repository = _repository(tmp_path)
    monkeypatch.setattr(
        retrieval_cli,
        "provider_from_config",
        lambda _config: pytest.fail("provider must not be constructed"),
    )
    stdout = StringIO()
    stderr = StringIO()

    code = retrieval_cli.run_retrieval_command(
        ["retrieval", "rebuild", "--semantic"], stdout, stderr, lambda _cwd: repository
    )

    assert code == 1
    assert stderr.getvalue() == ""
    assert _canonical_object(stdout.getvalue())["status"] == "absent"


def test_historical_explain_dispatches_before_retrieval_service_or_provider(monkeypatch, tmp_path):
    from codex_memory.retrieval import cli as retrieval_cli
    manifest_id = "inj_" + "a" * 64
    monkeypatch.setattr(retrieval_cli, "RetrievalService", lambda *_a: pytest.fail("service must not be constructed"))
    monkeypatch.setattr(retrieval_cli, "provider_from_config", lambda *_a: pytest.fail("provider must not be constructed"))
    monkeypatch.setattr(retrieval_cli, "_load_historical_injection", lambda root, value: {"injection_id": value, "selected": []})
    stdout, stderr = StringIO(), StringIO()
    code = retrieval_cli.run_retrieval_command(["retrieval", "explain", manifest_id], stdout, stderr, lambda _cwd: tmp_path)
    assert (code, stderr.getvalue()) == (0, "")
    assert _canonical_object(stdout.getvalue()) == {"injection_id": manifest_id, "selected": []}


@pytest.mark.parametrize("manifest_id", ["../manifest", "inj_ABC", "inj_" + "a" * 63])
def test_historical_explain_rejects_unsafe_or_malformed_id_before_store(monkeypatch, tmp_path, manifest_id):
    from codex_memory.retrieval import cli as retrieval_cli
    monkeypatch.setattr(retrieval_cli, "_load_historical_injection", lambda *_a: pytest.fail("invalid ID must not reach store"), raising=False)
    stdout, stderr = StringIO(), StringIO()
    code = retrieval_cli.run_retrieval_command(["retrieval", "explain", manifest_id], stdout, stderr, lambda _cwd: tmp_path)
    assert (code, stdout.getvalue(), stderr.getvalue()) == (2, "", "codex-memory-hook: invalid retrieval input\n")


def test_historical_explain_reports_invalid_evidence_without_service(monkeypatch, tmp_path):
    from codex_memory.injection_manifest import InjectionAuditError
    from codex_memory.retrieval import cli as retrieval_cli
    monkeypatch.setattr(retrieval_cli, "RetrievalService", lambda *_a: pytest.fail("service must not be constructed"))
    monkeypatch.setattr(retrieval_cli, "_load_historical_injection", lambda *_a: (_ for _ in ()).throw(InjectionAuditError("tampered")))
    stdout, stderr = StringIO(), StringIO()
    code = retrieval_cli.run_retrieval_command(["retrieval", "explain", "inj_" + "a" * 64], stdout, stderr, lambda _cwd: tmp_path)
    assert (code, stderr.getvalue()) == (1, "")
    assert _canonical_object(stdout.getvalue()) == {"status": "error", "reason": "tampered"}


def test_real_query_cli_is_deterministic_redacted_explainable_and_legacy_exact(tmp_path: Path) -> None:
    """Would fail if the public query CLI drifted from the canonical read-only contract."""
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.blob_store import BlobStore
    from codex_memory.memory_context import build_memory_packet

    repository = _repository(tmp_path)
    paths = RuntimePaths.for_repo(repository)
    connection = connect_database(paths.db_path)
    record = propose_record(connection, record_type="decision", scope_kind="project", scope_key=str(repository), topic_key="needle", statement="needle safe result", value=None, evidence_uris=(), idempotency_key="cli-query-record")
    apply_action(connection, record_id=record.record_id, action="activate", idempotency_key="cli-query-active")
    connection.commit()
    expected_packet = build_memory_packet(connection, BlobStore(paths.blobs_dir), session_id="session", turn_id="turn", context_id="context", scope_key=str(repository), prompt_event_uri=None, query="needle", max_chars=4000)
    connection.close()

    assert _run(repository, "retrieval", "rebuild").returncode == 0
    default = _run(repository, "retrieval", "query", "--session-id", "session", "--turn-id", "turn", "--context-id", "context", "--query", "needle")
    arguments = ("retrieval", "query", "--session-id", "session", "--turn-id", "turn", "--context-id", "context", "--query", "needle", "--explain", "--legacy")
    first = _run(repository, *arguments); second = _run(repository, *arguments)

    assert (first.returncode, first.stderr, first.stdout) == (0, "", second.stdout)
    document = _canonical_object(first.stdout)
    assert document["status"] == "current"
    assert [candidate["candidate_id"] for candidate in document["candidates"]] == [record.record_id]
    assert document["candidates"][0]["scope_key"] == "<local-scope>"
    assert "needle" not in default.stdout.replace(record.record_id, "")
    assert str(repository) not in default.stdout
    assert {trace["candidate_id"] for trace in document["trace"] if trace["selected"]} == {record.record_id}
    assert document["legacy"] == {"packet": expected_packet, "sha256": hashlib.sha256(expected_packet.encode("utf-8")).hexdigest(), "length": len(expected_packet)}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_real_cli_accepts_windows_case_variation_in_a_repository_with_spaces(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    invocation_root = Path(str(repository).swapcase()) if os.name == "nt" else repository

    result = _run(invocation_root, "retrieval", "status")

    assert result.returncode == 0
    assert result.stderr == ""
    assert _canonical_object(result.stdout)["status"] == "absent"


def _logical_rows(database_path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    with sqlite3.connect(database_path) as connection:
        tables = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        )
        return {
            table: tuple(
                tuple(row)
                for row in connection.execute(
                    f'SELECT * FROM "{table}" ORDER BY rowid'
                )
            )
            for table in tables
        }


def test_init_and_rebuild_preserve_canonical_database_bytes_and_logical_rows(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    database_path = RuntimePaths.for_repo(repository).db_path
    before_bytes = database_path.read_bytes()
    before_rows = _logical_rows(database_path)

    initialized = _run(repository, "retrieval", "init")
    rebuilt = _run(repository, "retrieval", "rebuild")

    assert initialized.returncode == rebuilt.returncode == 0
    assert database_path.read_bytes() == before_bytes
    assert _logical_rows(database_path) == before_rows
    assert all(not name.startswith("retrieval") for name in before_rows)


def test_status_and_verify_do_not_create_an_absent_runtime_tree(tmp_path: Path) -> None:
    repository = tmp_path / "git only repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )

    status = _run(repository, "retrieval", "status")
    verified = _run(repository, "retrieval", "verify")

    assert status.returncode == 0
    assert verified.returncode == 1
    assert _canonical_object(status.stdout)["degraded_reason"] == "canonical_unavailable"
    assert status.stdout == verified.stdout
    assert status.stderr == verified.stderr == ""
    assert not (repository / ".codex-memory").exists()


def test_corrupt_state_is_bounded_and_rebuild_repairs_without_leaking_bytes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    retrieval = repository / ".codex-memory" / "retrieval"
    retrieval.mkdir()
    secret = "SENSITIVE_CORRUPT_RETRIEVAL_BYTES"
    (retrieval / "retrieval.sqlite3").write_text(secret, encoding="utf-8")

    status = _run(repository, "retrieval", "status")
    verified = _run(repository, "retrieval", "verify")
    initialized = _run(repository, "retrieval", "init")
    rebuilt = _run(repository, "retrieval", "rebuild")

    for result, exit_code in ((status, 0), (verified, 1), (initialized, 1)):
        assert result.returncode == exit_code
        assert result.stderr == ""
        assert _canonical_object(result.stdout)["status"] == "corrupt"
        assert secret not in result.stdout
    assert rebuilt.returncode == 0
    assert rebuilt.stderr == ""
    assert _canonical_object(rebuilt.stdout)["status"] == "current"


def test_repository_resolution_failure_is_generic_and_bounded(tmp_path: Path) -> None:
    outside_git = tmp_path / "SENSITIVE_repository_name"
    outside_git.mkdir()

    result = _run(outside_git, "retrieval", "status")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: runtime unavailable\n"
    assert "SENSITIVE" not in result.stderr


@pytest.mark.parametrize(
    "error,reason",
    [
        (PermissionError("SENSITIVE denied path"), "path_unavailable"),
        (RuntimePathError("SENSITIVE redirected path"), "path_unsafe"),
    ],
)
def test_operational_failure_emits_only_a_bounded_status_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
) -> None:
    from codex_memory.retrieval import cli as retrieval_cli

    class FailingService:
        def rebuild(self):
            raise error

    monkeypatch.setattr(retrieval_cli, "RetrievalService", lambda paths: FailingService())
    stdout = StringIO()
    stderr = StringIO()

    code = retrieval_cli.run_retrieval_command(
        ["retrieval", "rebuild"], stdout, stderr, lambda _cwd: tmp_path
    )

    assert code == 1
    assert stderr.getvalue() == ""
    document = _canonical_object(stdout.getvalue())
    assert document["status"] == "corrupt"
    assert document["degraded_reason"] == reason
    assert "SENSITIVE" not in stdout.getvalue()


def test_canonical_unavailable_exception_serializes_stable_reason_without_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import canonical_snapshot, cli as retrieval_cli

    stage_enum = getattr(canonical_snapshot, "CanonicalValidationStage", None)
    assert stage_enum is not None
    unavailable = canonical_snapshot.CanonicalSnapshotUnavailableError(
        "canonical database is unavailable",
        stage=stage_enum.SCHEMA_QUERY,
    )

    class FailingService:
        def status(self):
            raise unavailable

    monkeypatch.setattr(retrieval_cli, "RetrievalService", lambda paths: FailingService())
    stdout = StringIO()
    stderr = StringIO()

    code = retrieval_cli.run_retrieval_command(
        ["retrieval", "status"], stdout, stderr, lambda _cwd: tmp_path
    )

    assert code == 1
    assert stderr.getvalue() == ""
    document = _canonical_object(stdout.getvalue())
    assert document["status"] == "corrupt"
    assert document["degraded_reason"] == "canonical_unavailable"
    assert "canonical database is unavailable" not in stdout.getvalue()
    assert str(tmp_path) not in stdout.getvalue()


def test_canonical_invalid_exception_serializes_stable_reason_without_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import canonical_snapshot, cli as retrieval_cli

    stage_enum = getattr(canonical_snapshot, "CanonicalValidationStage", None)
    assert stage_enum is not None
    invalid = canonical_snapshot.CanonicalSnapshotInvalidError(
        "canonical database is invalid",
        stage=stage_enum.DDL_READ,
    )

    class FailingService:
        def verify(self):
            raise invalid

    monkeypatch.setattr(retrieval_cli, "RetrievalService", lambda paths: FailingService())
    stdout = StringIO()
    stderr = StringIO()

    code = retrieval_cli.run_retrieval_command(
        ["retrieval", "verify"], stdout, stderr, lambda _cwd: tmp_path
    )

    assert code == 1
    assert stderr.getvalue() == ""
    document = _canonical_object(stdout.getvalue())
    assert document["status"] == "corrupt"
    assert document["degraded_reason"] == "canonical_invalid"
    assert "canonical database is invalid" not in stdout.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
