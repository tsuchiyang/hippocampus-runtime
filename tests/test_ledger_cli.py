from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest


def _repo(tmp_path: Path, name: str = "repo with spaces") -> Path:
    repository = tmp_path / name
    repository.mkdir()
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    return repository


def _run(
    repository: Path,
    arguments: list[str],
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "codex_memory.cli", *arguments],
        cwd=repository,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _propose(
    repository: Path,
    *,
    topic: str = "storage.engine",
    statement: str = "Use SQLite.",
    key: str = "proposal-1",
) -> subprocess.CompletedProcess[str]:
    return _run(
        repository,
        [
            "record",
            "propose",
            "--type",
            "decision",
            "--scope",
            "project",
            "--scope-key",
            "example",
            "--topic-key",
            topic,
            "--statement",
            statement,
            "--value-json",
            '{"name":"sqlite3"}',
            "--evidence-uri",
            "https://example.com/design",
            "--idempotency-key",
            key,
        ],
    )


def _assert_canonical_object(stdout: str) -> dict[str, object]:
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


def test_record_propose_activate_show_and_verify_real_entry_point(tmp_path: Path) -> None:
    repository = _repo(tmp_path)

    proposed = _propose(repository)

    assert proposed.returncode == 0 and proposed.stderr == ""
    proposal = _assert_canonical_object(proposed.stdout)
    assert proposal["state"] == "proposed"
    assert proposal["value"] == {"name": "sqlite3"}
    assert proposal["evidence_uris"] == ["https://example.com/design"]
    record_id = proposal["record_id"]

    activated = _run(
        repository,
        ["record", "activate", record_id, "--idempotency-key", "activate-1"],
    )
    assert activated.returncode == 0 and activated.stderr == ""
    assert _assert_canonical_object(activated.stdout)["state"] == "active"

    shown = _run(repository, ["record", "show", record_id])
    assert shown.returncode == 0 and shown.stderr == ""
    show = _assert_canonical_object(shown.stdout)
    assert show["record_id"] == record_id
    assert show["state"] == "active"

    verified = _run(repository, ["ledger", "verify"])
    assert verified.returncode == 0 and verified.stderr == ""
    assert _assert_canonical_object(verified.stdout) == {
        "current": 1,
        "events": 2,
        "evidence": 1,
        "records": 1,
        "relations": 0,
        "schema_version": 3,
        "candidates": 0,
        "candidate_events": 0,
        "candidate_evidence": 0,
        "candidate_current": 0,
    }


def test_ledger_verify_reports_nonzero_candidate_counts_canonically(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    database = import_module("codex_memory.database")
    candidates = import_module("codex_memory.candidate_memory")
    blob_store = import_module("codex_memory.blob_store")
    paths_module = import_module("codex_memory.paths")
    paths = paths_module.RuntimePaths.for_repo(repository)
    connection = database.connect_database(paths.db_path)
    database.migrate_database(connection)
    candidates.capture_candidate(
        connection,
        blob_store.BlobStore(paths.blobs_dir),
        project_scope_key=str(repository.resolve()),
        record_type="decision",
        scope_kind="project",
        scope_key=str(repository.resolve()),
        topic_key="cli.candidate",
        statement="The CLI must report candidate rows.",
        session_id="cli-session",
        turn_id="cli-turn",
        context_id="startup",
        idempotency_key="cli-candidate",
        evidence_uris=("https://example.test/cli",),
    )
    connection.close()

    verified = _run(repository, ["ledger", "verify"])

    assert verified.returncode == 0 and verified.stderr == ""
    assert _assert_canonical_object(verified.stdout) == {
        "candidate_current": 1,
        "candidate_events": 1,
        "candidate_evidence": 1,
        "candidates": 1,
        "current": 0,
        "events": 0,
        "evidence": 0,
        "records": 0,
        "relations": 0,
        "schema_version": 3,
    }


def test_record_list_is_deterministic_and_filterable(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    first = json.loads(_propose(repository, topic="topic.one", key="proposal-1").stdout)
    second = json.loads(_propose(repository, topic="topic.two", key="proposal-2").stdout)

    listed = _run(repository, ["record", "list", "--limit", "2"])
    filtered = _run(
        repository, ["record", "list", "--topic-key", "topic.two", "--limit", "10"]
    )

    assert listed.returncode == 0 and listed.stderr == ""
    assert [item["record_id"] for item in _assert_canonical_object(listed.stdout)["records"]] == [
        first["record_id"],
        second["record_id"],
    ]
    assert [
        item["record_id"] for item in _assert_canonical_object(filtered.stdout)["records"]
    ] == [second["record_id"]]


def test_record_propose_defaults_omitted_value_to_json_null(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    result = _run(
        repository,
        [
            "record",
            "propose",
            "--type",
            "assumption",
            "--scope",
            "workspace",
            "--scope-key",
            "workspace-a",
            "--topic-key",
            "build.environment",
            "--statement",
            "Use the current Python runtime.",
            "--idempotency-key",
            "proposal-null",
        ],
    )

    assert result.returncode == 0 and result.stderr == ""
    assert _assert_canonical_object(result.stdout)["value"] is None


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("invalidate", []),
        ("close", []),
        ("reopen", []),
        ("supersede", ["--replacement-record-id", "mem_replacement"]),
    ],
)
def test_cli_rejects_python_only_lifecycle_routes_as_unsupported(
    tmp_path: Path, command: str, extra: list[str]
) -> None:
    repository = _repo(tmp_path)
    proposal = json.loads(_propose(repository).stdout)

    result = _run(
        repository,
        [
            "record",
            command,
            proposal["record_id"],
            *extra,
            "--idempotency-key",
            f"{command}-1",
        ],
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: unsupported command\n"
    assert "Traceback" not in result.stderr
    shown = _run(repository, ["record", "show", proposal["record_id"]])
    assert json.loads(shown.stdout)["state"] == "proposed"


def test_invalid_cli_json_and_validation_fail_without_traceback(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    result = _run(
        repository,
        [
            "record",
            "propose",
            "--type",
            "decision",
            "--scope",
            "project",
            "--scope-key",
            "example",
            "--topic-key",
            "storage.engine",
            "--statement",
            "Use SQLite.",
            "--idempotency-key",
            "proposal-invalid",
            "--value-json",
            '{"duplicate":1,"duplicate":2}',
        ],
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: invalid ledger input\n"
    assert "Traceback" not in result.stderr
    assert not (repository / ".codex-memory").exists()


def test_deep_cli_json_is_bounded_without_traceback_or_runtime_creation(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    deeply_nested = "[" * 1500 + "null" + "]" * 1500

    result = _run(
        repository,
        [
            "record",
            "propose",
            "--type",
            "decision",
            "--scope",
            "project",
            "--scope-key",
            "example",
            "--topic-key",
            "storage.engine",
            "--statement",
            "Use SQLite.",
            "--idempotency-key",
            "proposal-deep",
            "--value-json",
            deeply_nested,
        ],
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: invalid ledger input\n"
    assert "Traceback" not in result.stderr
    assert not (repository / ".codex-memory").exists()


def test_idempotency_conflict_is_bounded_and_does_not_leak_input(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    assert _propose(repository).returncode == 0

    conflict = _propose(
        repository,
        statement="Sensitive alternate statement.",
        key="proposal-1",
    )

    assert conflict.returncode == 1
    assert conflict.stdout == ""
    assert conflict.stderr == "codex-memory-hook: idempotency conflict\n"
    assert "Sensitive" not in conflict.stderr
    assert "Traceback" not in conflict.stderr


def test_cross_operation_idempotency_conflicts_are_bounded_and_atomic(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    first = json.loads(_propose(repository, key="proposal-first").stdout)
    activated = _run(
        repository,
        [
            "record",
            "activate",
            first["record_id"],
            "--idempotency-key",
            "action-owned",
        ],
    )
    assert activated.returncode == 0

    action_then_proposal = _propose(
        repository,
        topic="topic.second",
        statement="Second statement.",
        key="action-owned",
    )
    proposal_then_action = _run(
        repository,
        [
            "record",
            "activate",
            first["record_id"],
            "--idempotency-key",
            "proposal-first",
        ],
    )

    for result in (action_then_proposal, proposal_then_action):
        assert result.returncode == 1
        assert result.stdout == ""
        assert result.stderr == "codex-memory-hook: idempotency conflict\n"
        assert "Traceback" not in result.stderr
    listed = _run(repository, ["record", "list"])
    assert len(json.loads(listed.stdout)["records"]) == 1


def test_ledger_verify_requires_existing_database_without_creating_paths(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)

    result = _run(repository, ["ledger", "verify"])

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: ledger verification failed\n"
    assert not (repository / ".codex-memory").exists()


def test_phase_zero_unsupported_command_contract_is_unchanged(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    result = _run(repository, ["unknown"], "{}")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: unsupported command\n"


def test_phase_zero_post_tool_hook_stdin_and_output_contract_is_unchanged(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    payload = json.dumps(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "cwd": str(repository),
            "tool_name": "exec_command",
            "tool_use_id": "tool-1",
            "tool_input": {"command": "git status --short"},
            "tool_response": {"output": ""},
        }
    )

    result = _run(repository, ["post-tool-use"], payload)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    "arguments",
    (
        ["record", "list", "--limit", "true"],
        ["record", "show"],
        ["ledger", "verify", "--unknown"],
    ),
)
def test_legacy_invalid_argv_and_return_code_contract_remains_byte_exact(
    tmp_path: Path, arguments: list[str]
) -> None:
    repository = _repo(tmp_path)

    result = _run(repository, arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: invalid ledger input\n"
    assert "Traceback" not in result.stderr
    assert not (repository / ".codex-memory").exists()
