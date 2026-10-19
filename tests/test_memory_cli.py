from __future__ import annotations

from dataclasses import asdict as dataclass_asdict
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
from typing import Callable

import pytest


def _memory_cli():
    try:
        return import_module("codex_memory.memory_cli")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Task 8 memory CLI module is missing: {exc}")


def _repo(tmp_path: Path, name: str = "repo with spaces") -> Path:
    repository = tmp_path / name
    repository.mkdir()
    return repository


def _invoke(
    repository: Path,
    arguments: list[object],
    *,
    resolver: Callable[[Path], Path] | None = None,
) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = _memory_cli().run_memory_command(
        arguments,
        stdout,
        stderr,
        (lambda _cwd: repository) if resolver is None else resolver,
    )
    return code, stdout.getvalue(), stderr.getvalue()


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


def _capture_arguments(
    repository: Path,
    *,
    session_id: str = "session-a",
    turn_id: str = "turn-a",
    context_id: str = "startup",
    topic_key: str = "runtime.storage",
    statement: str = "Use the local SQLite runtime.",
    idempotency_key: str = "capture-a",
    evidence: tuple[str, ...] = (),
) -> list[object]:
    arguments: list[object] = [
        "memory",
        "capture",
        "--session-id",
        session_id,
        "--turn-id",
        turn_id,
        "--context-id",
        context_id,
        "--type",
        "decision",
        "--scope",
        "project",
        "--scope-key",
        str(repository.resolve()),
        "--topic-key",
        topic_key,
        "--statement",
        statement,
        "--idempotency-key",
        idempotency_key,
        "--value-json",
        '{"z":2,"a":1}',
    ]
    for uri in evidence:
        arguments.extend(("--evidence-uri", uri))
    return arguments


def _capture(repository: Path, **changes: object) -> dict[str, object]:
    code, stdout, stderr = _invoke(
        repository, _capture_arguments(repository, **changes)
    )
    assert (code, stderr) == (0, "")
    return _assert_canonical_object(stdout)


def _append_prompt_event(
    repository: Path, *, session_id: str = "session-a", turn_id: str = "turn-a"
) -> object:
    paths_module = import_module("codex_memory.paths")
    database = import_module("codex_memory.database")
    blob_store = import_module("codex_memory.blob_store")
    event_store = import_module("codex_memory.event_store")
    hook_input = import_module("codex_memory.hook_input")
    paths = paths_module.RuntimePaths.for_repo(repository)
    paths.ensure()
    connection = database.connect_database(paths.db_path)
    try:
        database.migrate_database(connection)
        envelope = hook_input.parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": session_id,
                "turn_id": turn_id,
                "cwd": str(repository),
                "prompt": "sensitive prompt body must not be listed",
            }
        )
        return event_store.EventStore(
            connection, blob_store.BlobStore(paths.blobs_dir)
        ).append(envelope)
    finally:
        connection.close()


def test_capture_and_pending_emit_exact_canonical_results_and_preserve_duplicates(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    uri = "https://example.test/evidence"

    captured = _capture(repository, evidence=(uri, uri))

    assert set(captured) == {
        "candidate_id",
        "record_type",
        "scope_kind",
        "scope_key",
        "topic_key",
        "statement",
        "value",
        "session_id",
        "turn_id",
        "context_id",
        "evidence_uris",
        "state",
        "permanent_record_id",
        "content_sha256",
        "created_at",
        "updated_at",
    }
    assert captured["value"] == {"a": 1, "z": 2}
    assert captured["evidence_uris"] == [uri, uri]
    assert captured["scope_key"] == str(repository.resolve())
    assert captured["state"] == "temporary"

    code, stdout, stderr = _invoke(
        repository, ["memory", "pending", "--session-id", "session-a"]
    )
    other = _invoke(
        repository, ["memory", "pending", "--session-id", "session-b"]
    )

    assert (code, stderr) == (0, "")
    pending = _assert_canonical_object(stdout)
    assert list(pending) == ["candidates"]
    assert pending["candidates"] == [captured]
    assert _assert_canonical_object(other[1]) == {"candidates": []}
    assert (other[0], other[2]) == (0, "")


def test_evidence_exposes_only_bounded_event_identity_fields(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    event = _append_prompt_event(repository)

    code, stdout, stderr = _invoke(
        repository,
        [
            "memory",
            "evidence",
            "--session-id",
            "session-a",
            "--turn-id",
            "turn-a",
            "--limit",
            "1",
        ],
    )

    assert (code, stderr) == (0, "")
    result = _assert_canonical_object(stdout)
    assert list(result) == ["evidence"]
    assert len(result["evidence"]) == 1
    listed = result["evidence"][0]
    assert set(listed) == {
        "event_id",
        "session_id",
        "turn_id",
        "event_name",
        "tool_name",
        "created_at",
        "uri",
    }
    assert listed == {
        "event_id": event.event_id,
        "session_id": "session-a",
        "turn_id": "turn-a",
        "event_name": "UserPromptSubmit",
        "tool_name": None,
        "created_at": event.created_at,
        "uri": f"codex-event://session-a/{event.event_id}",
    }
    assert "sensitive prompt body" not in stdout
    assert "payload" not in stdout


def test_review_and_expire_emit_exact_results_and_expire_is_convenience_review(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    first = _capture(repository, idempotency_key="capture-review")

    reviewed = _invoke(
        repository,
        [
            "memory",
            "review",
            first["candidate_id"],
            "--outcome",
            "reject",
            "--reason",
            "No longer applies.",
            "--idempotency-key",
            "review-reject",
        ],
    )

    assert (reviewed[0], reviewed[2]) == (0, "")
    rejected = _assert_canonical_object(reviewed[1])
    assert set(rejected) == {
        "candidate_id",
        "event_id",
        "action",
        "requested_outcome",
        "policy_code",
        "state",
        "permanent_record_id",
        "permanent_record_created",
        "superseded_record_id",
        "created_at",
    }
    assert (
        rejected["candidate_id"],
        rejected["action"],
        rejected["requested_outcome"],
        rejected["policy_code"],
        rejected["state"],
    ) == (
        first["candidate_id"],
        "reject",
        "reject",
        "review_rejected",
        "rejected",
    )

    second = _capture(
        repository,
        topic_key="runtime.expire",
        idempotency_key="capture-expire",
    )
    expired_call = _invoke(
        repository,
        [
            "memory",
            "expire",
            second["candidate_id"],
            "--reason",
            "Insufficient evidence.",
            "--idempotency-key",
            "review-expire",
        ],
    )

    assert (expired_call[0], expired_call[2]) == (0, "")
    expired = _assert_canonical_object(expired_call[1])
    assert (
        expired["candidate_id"],
        expired["action"],
        expired["requested_outcome"],
        expired["policy_code"],
        expired["state"],
    ) == (
        second["candidate_id"],
        "expire",
        "expire",
        "review_expired",
        "expired",
    )
    assert "Insufficient evidence" not in expired_call[1]


def test_recall_uses_normalized_scope_context_and_null_prompt_without_query_render(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    _capture(repository)
    query = "UNIQUE_QUERY_MUST_NOT_RENDER"

    code, stdout, stderr = _invoke(
        repository,
        [
            "memory",
            "recall",
            "--session-id",
            "session-a",
            "--query",
            query,
            "--max-chars",
            "4000",
        ],
    )

    assert (code, stderr) == (0, "")
    result = _assert_canonical_object(stdout)
    assert list(result) == ["packet"]
    packet = result["packet"]
    assert isinstance(packet, str) and len(packet) <= 4000
    assert (
        "context session=\"session-a\" turn=\"cli-recall\" "
        "window=\"startup\" scope="
        + _memory_cli().canonical_json(str(repository.resolve()))
    ) in packet
    assert "prompt_evidence=null" in packet
    assert query not in packet
    assert "UNVERIFIED TEMPORARY" in packet


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["memory"], "codex-memory-hook: unsupported command\n"),
        (["memory", "unknown"], "codex-memory-hook: unsupported command\n"),
        (
            ["memory", "pending", "--session-id", "s", "--unknown", "x"],
            "codex-memory-hook: invalid memory input\n",
        ),
        (
            ["memory", "pending", "--sess", "s"],
            "codex-memory-hook: invalid memory input\n",
        ),
        (
            ["memory", "review", "cand", "--outcome", "other"],
            "codex-memory-hook: invalid memory input\n",
        ),
    ),
)
def test_unknown_subcommands_and_flags_have_exact_exit_two_errors(
    tmp_path: Path, arguments: list[object], message: str
) -> None:
    repository = _repo(tmp_path)
    resolved = False

    def forbidden_resolver(_cwd: Path) -> Path:
        nonlocal resolved
        resolved = True
        raise AssertionError("invalid arguments must not resolve a repository")

    code, stdout, stderr = _invoke(
        repository, arguments, resolver=forbidden_resolver
    )

    assert (code, stdout, stderr) == (2, "", message)
    assert resolved is False
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    ("full_flag", "abbreviation"),
    (("--evidence-uri", "--evidence"), ("--value-json", "--value")),
)
def test_capture_rejects_long_option_abbreviations_before_repository_lookup(
    tmp_path: Path, full_flag: str, abbreviation: str
) -> None:
    repository = _repo(tmp_path)
    arguments = _capture_arguments(
        repository, evidence=("https://example.test/evidence",)
    )
    arguments[arguments.index(full_flag)] = abbreviation
    resolved = False

    def forbidden_resolver(_cwd: Path) -> Path:
        nonlocal resolved
        resolved = True
        raise AssertionError("abbreviated flags must not resolve a repository")

    result = _invoke(repository, arguments, resolver=forbidden_resolver)

    assert result == (2, "", "codex-memory-hook: invalid memory input\n")
    assert resolved is False
    assert not (repository / ".codex-memory").exists()


@pytest.mark.parametrize(
    "arguments",
    (
        ["memory", "evidence", "--session-id", "s", "--limit", "0"],
        ["memory", "evidence", "--session-id", "s", "--limit", "101"],
        ["memory", "pending", "--session-id", "s", "--limit", "0"],
        ["memory", "pending", "--session-id", "s", "--limit", "101"],
        ["memory", "recall", "--session-id", "s", "--max-chars", "2047"],
        ["memory", "recall", "--session-id", "s", "--max-chars", "4001"],
        ["memory", "pending", "--session-id", "s", "--limit", True],
    ),
)
def test_parser_rejects_limit_boundaries_and_non_string_bool(
    tmp_path: Path, arguments: list[object]
) -> None:
    repository = _repo(tmp_path)
    assert _invoke(repository, arguments) == (
        2,
        "",
        "codex-memory-hook: invalid memory input\n",
    )
    assert not (repository / ".codex-memory").exists()


@pytest.mark.parametrize(
    "arguments",
    (
        ["memory", "evidence", "--session-id", "s", "--limit", "1"],
        ["memory", "evidence", "--session-id", "s", "--limit", "100"],
        ["memory", "pending", "--session-id", "s", "--limit", "1"],
        ["memory", "pending", "--session-id", "s", "--limit", "100"],
        ["memory", "recall", "--session-id", "s", "--max-chars", "2048"],
        ["memory", "recall", "--session-id", "s", "--max-chars", "4000"],
    ),
)
def test_parser_accepts_frozen_limit_boundaries(arguments: list[object]) -> None:
    parsed = _memory_cli()._parse_arguments(arguments)
    assert parsed.command in {"evidence", "pending", "recall"}


def test_parser_freezes_read_command_default_limits() -> None:
    assert _memory_cli()._parse_arguments(
        ["memory", "evidence", "--session-id", "s"]
    ).limit == 50
    assert _memory_cli()._parse_arguments(
        ["memory", "pending", "--session-id", "s"]
    ).limit == 100
    assert _memory_cli()._parse_arguments(
        ["memory", "recall", "--session-id", "s"]
    ).max_chars == 4000


def test_invalid_json_is_rejected_before_runtime_and_never_echoed(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    secret = ('s' + 'k-secret-value-MUST-NOT-ECHO')
    arguments = _capture_arguments(repository)
    value_index = arguments.index("--value-json") + 1
    arguments[value_index] = '{"secret":"' + secret + '","duplicate":1,"duplicate":2}'

    code, stdout, stderr = _invoke(repository, arguments)

    assert (code, stdout, stderr) == (
        2,
        "",
        "codex-memory-hook: invalid memory input\n",
    )
    assert secret not in stderr and "Traceback" not in stderr
    assert not (repository / ".codex-memory").exists()


def test_operation_errors_redact_statement_reason_uri_value_and_traceback(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    _capture(repository, idempotency_key="shared-capture")
    sensitive_statement = "PRIVATE_STATEMENT_MUST_NOT_ECHO"
    conflict = _invoke(
        repository,
        _capture_arguments(
            repository,
            statement=sensitive_statement,
            idempotency_key="shared-capture",
            evidence=(('h' + 'ttps://user:password@example.test/private?token=secret'),),
        ),
    )
    candidate = _capture(
        repository,
        topic_key="runtime.review-redaction",
        idempotency_key="redaction-review-capture",
    )
    first = _invoke(
        repository,
        [
            "memory",
            "review",
            candidate["candidate_id"],
            "--outcome",
            "reject",
            "--reason",
            "first reason",
            "--idempotency-key",
            "shared-review",
        ],
    )
    sensitive_reason = "PRIVATE_REASON_MUST_NOT_ECHO"
    review_conflict = _invoke(
        repository,
        [
            "memory",
            "review",
            candidate["candidate_id"],
            "--outcome",
            "reject",
            "--reason",
            sensitive_reason,
            "--idempotency-key",
            "shared-review",
        ],
    )
    sensitive_value = "sk-value-MUST-NOT-ECHO"
    value_arguments = _capture_arguments(
        repository,
        topic_key="runtime.secret-value",
        idempotency_key="secret-value-capture",
    )
    value_arguments[value_arguments.index("--value-json") + 1] = (
        '{"api_key":"' + sensitive_value + '"}'
    )
    value_rejected = _invoke(repository, value_arguments)

    assert first[0] == 0
    for result in (conflict, review_conflict):
        assert result == (1, "", "codex-memory-hook: idempotency conflict\n")
        assert "Traceback" not in result[2]
    assert value_rejected == (
        2,
        "",
        "codex-memory-hook: invalid memory input\n",
    )
    assert sensitive_statement not in conflict[2]
    assert "password" not in conflict[2] and "token" not in conflict[2]
    assert sensitive_reason not in review_conflict[2]
    assert sensitive_value not in value_rejected[2]


def test_review_forwards_duplicate_evidence_and_supersedes_while_expire_clears_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    memory_cli = _memory_cli()
    candidate_module = import_module("codex_memory.candidate_memory")
    paths_module = import_module("codex_memory.paths")
    calls: list[dict[str, object]] = []

    def review_spy(connection: object, blobs: object, **kwargs: object):
        calls.append({"connection": connection, "blobs": blobs, **kwargs})
        return candidate_module.CandidateActionResult(
            candidate_id=str(kwargs["candidate_id"]),
            event_id="cevt-result",
            action=str(kwargs["outcome"]),
            requested_outcome=str(kwargs["outcome"]),
            policy_code="review_expired",
            state="expired",
            permanent_record_id=None,
            permanent_record_created=None,
            superseded_record_id=None,
            created_at="2026-09-01T00:00:00.000000Z",
        )

    monkeypatch.setattr(memory_cli, "review_candidate", review_spy)
    connection = object()
    paths = paths_module.RuntimePaths.for_repo(repository)
    duplicate = "https://example.test/review"
    review = memory_cli._parse_arguments(
        [
            "memory",
            "review",
            "cand-review",
            "--outcome",
            "promote",
            "--reason",
            "Reviewed.",
            "--idempotency-key",
            "review-key",
            "--evidence-uri",
            duplicate,
            "--evidence-uri",
            duplicate,
            "--supersedes",
            "mem-old",
        ]
    )
    expire = memory_cli._parse_arguments(
        [
            "memory",
            "expire",
            "cand-expire",
            "--reason",
            "Expired.",
            "--idempotency-key",
            "expire-key",
        ]
    )

    memory_cli._dispatch_write(review, connection, paths)
    memory_cli._dispatch_write(expire, connection, paths)

    assert calls[0]["evidence_uris"] == (duplicate, duplicate)
    assert calls[0]["supersedes_record_id"] == "mem-old"
    assert calls[0]["outcome"] == "promote"
    assert calls[1]["evidence_uris"] == ()
    assert calls[1]["supersedes_record_id"] is None
    assert calls[1]["outcome"] == "expire"


def test_recall_dispatch_passes_exact_normalized_context_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    memory_cli = _memory_cli()
    paths = import_module("codex_memory.paths").RuntimePaths.for_repo(repository)
    observed: dict[str, object] = {}

    def context_spy(connection: object, *, session_id: str) -> str:
        observed["context"] = (connection, session_id)
        return "cp-latest"

    def packet_spy(connection: object, blobs: object, **kwargs: object) -> str:
        observed["packet"] = {"connection": connection, "blobs": blobs, **kwargs}
        return "packet-result"

    monkeypatch.setattr(memory_cli, "current_context_id", context_spy)
    monkeypatch.setattr(memory_cli, "build_memory_packet", packet_spy)
    command = memory_cli._parse_arguments(
        [
            "memory",
            "recall",
            "--session-id",
            "session-a",
            "--query",
            "selector only",
            "--max-chars",
            "3000",
        ]
    )
    connection = object()

    result = memory_cli._dispatch_read(command, connection, paths)

    assert result.packet == "packet-result"
    assert observed["context"] == (connection, "session-a")
    packet = observed["packet"]
    assert packet["connection"] is connection
    assert packet["session_id"] == "session-a"
    assert packet["turn_id"] == "cli-recall"
    assert packet["context_id"] == "cp-latest"
    assert packet["scope_key"] == str(paths.repo_root)
    assert packet["prompt_event_uri"] is None
    assert packet["query"] == "selector only"
    assert packet["max_chars"] == 3000


def test_recall_dispatch_passes_configured_workspace_scope_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    codex = repository / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text(
        '[scope]\nworkspace_keys = ["team-notes"]\n', encoding="utf-8"
    )
    memory_cli = _memory_cli()
    paths = import_module("codex_memory.paths").RuntimePaths.for_repo(repository)
    observed: dict[str, object] = {}
    monkeypatch.setattr(memory_cli, "current_context_id", lambda *args, **kwargs: "startup")

    def packet(connection: object, blobs: object, **kwargs: object) -> str:
        observed.update(kwargs)
        return "packet-result"

    monkeypatch.setattr(memory_cli, "build_memory_packet", packet)
    command = memory_cli._parse_arguments(
        ["memory", "recall", "--session-id", "session-a"]
    )

    result = memory_cli._dispatch_read(command, object(), paths)

    assert result.packet == "packet-result"
    assert observed["workspace_scope_keys"] == ("team-notes",)


def test_recall_with_malformed_memory_config_reports_diagnostic_without_failing(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    _capture(repository)
    codex = repository / ".codex"
    codex.mkdir(exist_ok=True)
    (codex / "memory.toml").write_text("[scope\ninvalid", encoding="utf-8")

    code, stdout, stderr = _invoke(
        repository, ["memory", "recall", "--session-id", "session-a"]
    )

    assert code == 0
    assert stdout.startswith('{"packet":"')
    assert stderr == (
        "codex-memory-hook: invalid memory config; workspace recall disabled\n"
    )


@pytest.mark.parametrize("command", ("review", "expire"))
def test_missing_candidate_error_is_exact_bounded_and_redacted(
    tmp_path: Path, command: str
) -> None:
    repository = _repo(tmp_path)
    _capture(repository)
    arguments: list[object] = ["memory", command, "cand_missing"]
    if command == "review":
        arguments.extend(("--outcome", "reject"))
    arguments.extend(
        (
            "--reason",
            "PRIVATE_MISSING_REASON",
            "--idempotency-key",
            f"missing-{command}",
        )
    )

    result = _invoke(repository, arguments)

    assert result == (1, "", "codex-memory-hook: candidate not found\n")
    assert "PRIVATE_MISSING_REASON" not in result[2]
    assert "Traceback" not in result[2]


@pytest.mark.parametrize(
    "arguments",
    (
        ["memory", "evidence", "--session-id", "session-a"],
        ["memory", "pending", "--session-id", "session-a"],
        ["memory", "recall", "--session-id", "session-a"],
    ),
)
def test_read_commands_require_existing_runtime_without_creating_any_path(
    tmp_path: Path, arguments: list[object]
) -> None:
    repository = _repo(tmp_path)

    result = _invoke(repository, arguments)

    assert result == (1, "", "codex-memory-hook: memory operation failed\n")
    assert not (repository / ".codex-memory").exists()


def test_runtime_path_derivation_failure_is_bounded_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    memory_cli = _memory_cli()
    secret = "C:\\private\\SECRET_RUNTIME_PATH"
    monkeypatch.setattr(
        memory_cli.RuntimePaths,
        "for_repo",
        staticmethod(
            lambda _root: (_ for _ in ()).throw(OSError(secret))
        ),
    )

    result = _invoke(
        repository, ["memory", "pending", "--session-id", "session-a"]
    )

    assert result == (1, "", "codex-memory-hook: runtime unavailable\n")
    assert secret not in result[2]
    assert "Traceback" not in result[2]


def test_read_commands_use_mode_ro_query_only_and_never_ensure_or_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    _capture(repository)
    memory_cli = _memory_cli()
    structured = import_module("codex_memory.structured_ledger")
    original_sqlite_connect = structured.sqlite3.connect
    observed: dict[str, object] = {}

    def connect_spy(database: object, *args: object, **kwargs: object):
        observed["database"] = database
        observed["uri"] = kwargs.get("uri")
        return original_sqlite_connect(database, *args, **kwargs)

    original_readonly = memory_cli.connect_readonly_database

    def readonly_spy(path: Path):
        connection = original_readonly(path)
        observed["query_only"] = connection.execute("PRAGMA query_only").fetchone()[0]
        return connection

    monkeypatch.setattr(structured.sqlite3, "connect", connect_spy)
    monkeypatch.setattr(memory_cli, "connect_readonly_database", readonly_spy)
    monkeypatch.setattr(
        memory_cli.RuntimePaths,
        "ensure",
        lambda self: pytest.fail("read command must not ensure runtime paths"),
    )
    monkeypatch.setattr(
        memory_cli,
        "migrate_database",
        lambda connection: pytest.fail("read command must not migrate"),
    )

    result = _invoke(
        repository, ["memory", "pending", "--session-id", "session-a"]
    )

    assert result[0] == 0 and result[2] == ""
    assert str(observed["database"]).endswith("?mode=ro")
    assert observed["uri"] is True
    assert observed["query_only"] == 1


def test_capture_ensures_writable_runtime_and_migrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    memory_cli = _memory_cli()
    calls: list[str] = []
    original_ensure = memory_cli.RuntimePaths.ensure
    original_connect = memory_cli.connect_database
    original_migrate = memory_cli.migrate_database

    def ensure_spy(paths: object) -> None:
        calls.append("ensure")
        original_ensure(paths)

    def connect_spy(path: Path):
        calls.append("connect")
        return original_connect(path)

    def migrate_spy(connection: object) -> None:
        calls.append("migrate")
        original_migrate(connection)

    monkeypatch.setattr(memory_cli.RuntimePaths, "ensure", ensure_spy)
    monkeypatch.setattr(memory_cli, "connect_database", connect_spy)
    monkeypatch.setattr(memory_cli, "migrate_database", migrate_spy)

    result = _invoke(repository, _capture_arguments(repository))

    assert result[0] == 0 and result[2] == ""
    assert calls == ["ensure", "connect", "migrate"]


def test_dataclass_serialization_calls_asdict_and_canonical_json_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_cli = _memory_cli()
    evidence_module = import_module("codex_memory.memory_evidence")
    value = evidence_module.EventEvidence(
        event_id=1,
        session_id="session-a",
        turn_id="turn-a",
        event_name="UserPromptSubmit",
        tool_name=None,
        created_at="2026-09-01T00:00:00.000000Z",
        uri="codex-event://session-a/1",
    )
    calls = {"asdict": 0, "canonical": 0}
    real_canonical = memory_cli.canonical_json

    def asdict_spy(document: object) -> dict[str, object]:
        calls["asdict"] += 1
        return dataclass_asdict(document)

    def canonical_spy(document: object) -> str:
        calls["canonical"] += 1
        return real_canonical(document)

    monkeypatch.setattr(memory_cli, "asdict", asdict_spy)
    monkeypatch.setattr(memory_cli, "canonical_json", canonical_spy)

    serialized = memory_cli._serialize(value)

    assert calls == {"asdict": 1, "canonical": 1}
    assert _assert_canonical_object(serialized)["uri"] == "codex-event://session-a/1"
