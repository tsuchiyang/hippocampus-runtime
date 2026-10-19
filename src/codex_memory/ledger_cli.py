from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sqlite3
from typing import Callable, Sequence, TextIO

from codex_memory.database import connect_database, migrate_database
from codex_memory.paths import RuntimePaths
from codex_memory.structured_ledger import (
    IdempotencyConflict,
    InvalidTransition,
    LedgerError,
    RecordNotFound,
    ValidationError,
    VerificationError,
    apply_action,
    canonical_json,
    list_records,
    parse_json_value,
    propose_record,
    show_record,
    verify_database,
)


class _LedgerArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValidationError("invalid ledger arguments")


class _UnsupportedLedgerCommand(Exception):
    pass


def run_ledger_command(
    arguments: Sequence[str],
    stdout: TextIO,
    stderr: TextIO,
    resolve_repo_root: Callable[[Path], Path],
) -> int:
    """Parse and run one structured-ledger command with bounded diagnostics."""
    try:
        command = _parse_arguments(arguments)
    except _UnsupportedLedgerCommand:
        stderr.write("codex-memory-hook: unsupported command\n")
        return 2
    except (ValidationError, ValueError, TypeError):
        stderr.write("codex-memory-hook: invalid ledger input\n")
        return 2

    try:
        repo_root = resolve_repo_root(Path.cwd())
    except Exception:
        stderr.write("codex-memory-hook: runtime unavailable\n")
        return 1

    paths = RuntimePaths.for_repo(repo_root)
    if command.prefix == "ledger":
        try:
            result = verify_database(paths.db_path)
            stdout.write(_serialize(result))
            return 0
        except Exception:
            stderr.write("codex-memory-hook: ledger verification failed\n")
            return 1

    connection: sqlite3.Connection | None = None
    try:
        paths.ensure()
        connection = connect_database(paths.db_path)
        migrate_database(connection)
        result = _dispatch_record(command, connection)
        stdout.write(_serialize(result))
        return 0
    except IdempotencyConflict:
        stderr.write("codex-memory-hook: idempotency conflict\n")
        return 1
    except RecordNotFound:
        stderr.write("codex-memory-hook: record not found\n")
        return 1
    except InvalidTransition:
        stderr.write("codex-memory-hook: invalid ledger action\n")
        return 1
    except ValidationError:
        stderr.write("codex-memory-hook: invalid ledger input\n")
        return 2
    except (VerificationError, LedgerError, sqlite3.Error, OSError):
        stderr.write("codex-memory-hook: ledger operation failed\n")
        return 1
    except Exception:
        stderr.write("codex-memory-hook: runtime unavailable\n")
        return 1
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    tokens = list(arguments)
    if (
        len(tokens) >= 2
        and tokens[0] == "record"
        and tokens[1] in {"invalidate", "close", "reopen", "supersede"}
    ):
        raise _UnsupportedLedgerCommand

    parser = _LedgerArgumentParser(add_help=False, exit_on_error=False)
    prefixes = parser.add_subparsers(dest="prefix", required=True)
    record = prefixes.add_parser("record", add_help=False, exit_on_error=False)
    record_commands = record.add_subparsers(dest="command", required=True)

    propose = record_commands.add_parser("propose", add_help=False, exit_on_error=False)
    propose.add_argument("--type", dest="record_type", required=True)
    propose.add_argument("--scope", dest="scope_kind", required=True)
    propose.add_argument("--scope-key", required=True)
    propose.add_argument("--topic-key", required=True)
    propose.add_argument("--statement", required=True)
    propose.add_argument("--idempotency-key", required=True)
    propose.add_argument("--value-json")
    propose.add_argument("--evidence-uri", action="append", default=[])

    activate = record_commands.add_parser(
        "activate", add_help=False, exit_on_error=False
    )
    activate.add_argument("record_id")
    activate.add_argument("--idempotency-key", required=True)

    show = record_commands.add_parser("show", add_help=False, exit_on_error=False)
    show.add_argument("record_id")
    listing = record_commands.add_parser("list", add_help=False, exit_on_error=False)
    listing.add_argument("--scope", dest="scope_kind")
    listing.add_argument("--scope-key")
    listing.add_argument("--topic-key")
    listing.add_argument("--type", dest="record_type")
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)

    ledger = prefixes.add_parser("ledger", add_help=False, exit_on_error=False)
    ledger_commands = ledger.add_subparsers(dest="command", required=True)
    ledger_commands.add_parser("verify", add_help=False, exit_on_error=False)

    try:
        parsed = parser.parse_args(tokens)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValidationError("invalid ledger arguments") from exc
    if parsed.prefix == "record" and parsed.command == "propose":
        parsed.value = (
            None if parsed.value_json is None else parse_json_value(parsed.value_json)
        )
    return parsed


def _dispatch_record(command: argparse.Namespace, connection: sqlite3.Connection) -> object:
    if command.command == "propose":
        return propose_record(
            connection,
            record_type=command.record_type,
            scope_kind=command.scope_kind,
            scope_key=command.scope_key,
            topic_key=command.topic_key,
            statement=command.statement,
            idempotency_key=command.idempotency_key,
            value=command.value,
            evidence_uris=tuple(command.evidence_uri),
        )
    if command.command == "show":
        return show_record(connection, command.record_id)
    if command.command == "list":
        records = list_records(
            connection,
            scope_kind=command.scope_kind,
            scope_key=command.scope_key,
            topic_key=command.topic_key,
            record_type=command.record_type,
            state=command.state,
            limit=command.limit,
        )
        return {"records": [asdict(record) for record in records]}
    return apply_action(
        connection,
        record_id=command.record_id,
        action=command.command,
        idempotency_key=command.idempotency_key,
        replacement_record_id=getattr(command, "replacement_record_id", None),
    )


def _serialize(value: object) -> str:
    document = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return canonical_json(document) + "\n"
