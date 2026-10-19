from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
from typing import Callable, Sequence, TextIO

from codex_memory.blob_store import BlobIntegrityError, BlobStore
from codex_memory.candidate_memory import CandidateView, capture_candidate, list_pending_candidates
from codex_memory.database import connect_database, migrate_database
from codex_memory.ledger_common import (
    IdempotencyConflict,
    LedgerError,
    RecordNotFound,
    ValidationError,
    VerificationError,
    canonical_json,
    parse_json_value,
)
from codex_memory.memory_context import build_memory_packet, current_context_id
from codex_memory.memory_evidence import EventEvidence, list_event_evidence
from codex_memory.memory_review import review_candidate
from codex_memory.paths import RuntimePathError, RuntimePaths
from codex_memory.project_memory_config import load_project_memory_config
from codex_memory.structured_ledger import connect_readonly_database, verify_ledger


_COMMANDS = frozenset({"capture", "evidence", "pending", "recall", "review", "expire"})
_READ_COMMANDS = frozenset({"evidence", "pending", "recall"})
_CLI_RECALL_TURN_ID = "cli-recall"


@dataclass(frozen=True, slots=True)
class _EvidenceResult:
    evidence: tuple[EventEvidence, ...]


@dataclass(frozen=True, slots=True)
class _PendingResult:
    candidates: tuple[CandidateView, ...]


@dataclass(frozen=True, slots=True)
class _RecallResult:
    packet: str


class _MemoryArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        raise ValidationError("invalid memory arguments")


class _UnsupportedMemoryCommand(Exception):
    pass


def run_memory_command(
    arguments: Sequence[str],
    stdout: TextIO,
    stderr: TextIO,
    resolve_repo_root: Callable[[Path], Path],
) -> int:
    """Parse and execute one bounded automatic-memory command."""
    try:
        command = _parse_arguments(arguments)
    except _UnsupportedMemoryCommand:
        stderr.write("codex-memory-hook: unsupported command\n")
        return 2
    except (ValidationError, ValueError, TypeError):
        stderr.write("codex-memory-hook: invalid memory input\n")
        return 2

    try:
        repo_root = resolve_repo_root(Path.cwd())
        paths = RuntimePaths.for_repo(repo_root)
    except Exception:
        stderr.write("codex-memory-hook: runtime unavailable\n")
        return 1

    connection: sqlite3.Connection | None = None
    read_transaction = False
    try:
        if command.command in _READ_COMMANDS:
            paths.validate()
            connection = connect_readonly_database(paths.db_path)
            connection.execute("BEGIN")
            read_transaction = True
            result = _dispatch_read(command, connection, paths, stderr=stderr)
        else:
            paths.ensure()
            connection = connect_database(paths.db_path)
            migrate_database(connection)
            result = _dispatch_write(command, connection, paths)
        serialized = _serialize(result)
        stdout.write(serialized)
        return 0
    except IdempotencyConflict:
        stderr.write("codex-memory-hook: idempotency conflict\n")
        return 1
    except RecordNotFound:
        stderr.write("codex-memory-hook: candidate not found\n")
        return 1
    except (ValidationError, ValueError, TypeError):
        stderr.write("codex-memory-hook: invalid memory input\n")
        return 2
    except (
        VerificationError,
        LedgerError,
        BlobIntegrityError,
        RuntimePathError,
        sqlite3.Error,
        OSError,
    ):
        stderr.write("codex-memory-hook: memory operation failed\n")
        return 1
    except Exception:
        stderr.write("codex-memory-hook: runtime unavailable\n")
        return 1
    finally:
        if connection is not None:
            try:
                if read_transaction and connection.in_transaction:
                    connection.rollback()
            finally:
                try:
                    connection.close()
                except Exception:
                    pass


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    tokens = list(arguments)
    if any(type(token) is not str for token in tokens):
        raise ValidationError("invalid memory arguments")
    if len(tokens) < 2 or tokens[0] != "memory" or tokens[1] not in _COMMANDS:
        raise _UnsupportedMemoryCommand

    parser = _MemoryArgumentParser(add_help=False, exit_on_error=False)
    prefixes = parser.add_subparsers(dest="prefix", required=True)
    memory = prefixes.add_parser("memory", add_help=False, exit_on_error=False)
    commands = memory.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", add_help=False, exit_on_error=False)
    capture.add_argument("--session-id", required=True)
    capture.add_argument("--turn-id", required=True)
    capture.add_argument("--context-id", required=True)
    capture.add_argument("--type", dest="record_type", required=True)
    capture.add_argument("--scope", dest="scope_kind", required=True)
    capture.add_argument("--scope-key", required=True)
    capture.add_argument("--topic-key", required=True)
    capture.add_argument("--statement", required=True)
    capture.add_argument("--idempotency-key", required=True)
    capture.add_argument("--value-json")
    capture.add_argument("--evidence-uri", action="append", default=[])

    evidence = commands.add_parser("evidence", add_help=False, exit_on_error=False)
    evidence.add_argument("--session-id", required=True)
    evidence.add_argument("--turn-id")
    evidence.add_argument("--limit", type=_bounded_integer(1, 100), default=50)

    pending = commands.add_parser("pending", add_help=False, exit_on_error=False)
    pending.add_argument("--session-id", required=True)
    pending.add_argument("--limit", type=_bounded_integer(1, 100), default=100)

    recall = commands.add_parser("recall", add_help=False, exit_on_error=False)
    recall.add_argument("--session-id", required=True)
    recall.add_argument("--query")
    recall.add_argument("--max-chars", type=_bounded_integer(2048, 4000), default=4000)

    review = commands.add_parser("review", add_help=False, exit_on_error=False)
    review.add_argument("candidate_id")
    review.add_argument(
        "--outcome", choices=("promote", "reject", "defer", "expire"), required=True
    )
    review.add_argument("--reason", required=True)
    review.add_argument("--idempotency-key", required=True)
    review.add_argument("--evidence-uri", action="append", default=[])
    review.add_argument("--supersedes", dest="supersedes_record_id")

    expire = commands.add_parser("expire", add_help=False, exit_on_error=False)
    expire.add_argument("candidate_id")
    expire.add_argument("--reason", required=True)
    expire.add_argument("--idempotency-key", required=True)

    try:
        parsed = parser.parse_args(tokens)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValidationError("invalid memory arguments") from exc
    if parsed.command == "capture":
        parsed.value = (
            None if parsed.value_json is None else parse_json_value(parsed.value_json)
        )
    return parsed


def _bounded_integer(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        if type(raw) is not str:
            raise argparse.ArgumentTypeError("invalid bounded integer")
        try:
            value = int(raw, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("invalid bounded integer") from exc
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError("invalid bounded integer")
        return value

    return parse


def _dispatch_read(
    command: argparse.Namespace,
    connection: sqlite3.Connection,
    paths: RuntimePaths,
    *,
    stderr: TextIO | None = None,
) -> object:
    if command.command == "evidence":
        verify_ledger(connection)
        return _EvidenceResult(
            list_event_evidence(
                connection,
                session_id=command.session_id,
                turn_id=command.turn_id,
                limit=command.limit,
            )
        )
    if command.command == "pending":
        verify_ledger(connection)
        return _PendingResult(
            list_pending_candidates(
                connection,
                session_id=command.session_id,
                limit=command.limit,
            )
        )

    context_id = current_context_id(connection, session_id=command.session_id)
    packet = build_memory_packet(
        connection,
        BlobStore(paths.blobs_dir),
        session_id=command.session_id,
        turn_id=_CLI_RECALL_TURN_ID,
        context_id=context_id,
        prompt_event_uri=None,
        query=command.query,
        max_chars=command.max_chars,
        **_memory_scope_kwargs(paths.repo_root, diagnostic_stream=stderr),
    )
    return _RecallResult(packet)


def _dispatch_write(
    command: argparse.Namespace,
    connection: sqlite3.Connection,
    paths: RuntimePaths,
) -> object:
    blobs = BlobStore(paths.blobs_dir)
    if command.command == "capture":
        return capture_candidate(
            connection,
            blobs,
            project_scope_key=load_project_memory_config(paths.repo_root).project_key_for(paths.repo_root),
            record_type=command.record_type,
            scope_kind=command.scope_kind,
            scope_key=command.scope_key,
            topic_key=command.topic_key,
            statement=command.statement,
            session_id=command.session_id,
            turn_id=command.turn_id,
            context_id=command.context_id,
            idempotency_key=command.idempotency_key,
            value=command.value,
            evidence_uris=tuple(command.evidence_uri),
        )
    if command.command == "review":
        return review_candidate(
            connection,
            blobs,
            candidate_id=command.candidate_id,
            outcome=command.outcome,
            reason=command.reason,
            idempotency_key=command.idempotency_key,
            evidence_uris=tuple(command.evidence_uri),
            supersedes_record_id=command.supersedes_record_id,
        )
    return review_candidate(
        connection,
        blobs,
        candidate_id=command.candidate_id,
        outcome="expire",
        reason=command.reason,
        idempotency_key=command.idempotency_key,
        evidence_uris=(),
        supersedes_record_id=None,
    )


def _serialize(value: object) -> str:
    return canonical_json(asdict(value)) + "\n"


def _memory_scope_kwargs(
    repo_root: Path, *, diagnostic_stream: TextIO | None = None
) -> dict[str, object]:
    config = load_project_memory_config(repo_root)
    if config.diagnostic is not None and diagnostic_stream is not None:
        diagnostic_stream.write(config.diagnostic)
    result = {"scope_key": config.project_key_for(repo_root)}
    if config.workspace_scope_keys:
        result["workspace_scope_keys"] = config.workspace_scope_keys
    return result
