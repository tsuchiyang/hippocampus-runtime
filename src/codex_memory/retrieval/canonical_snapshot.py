from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import sqlite3
from typing import Iterator

from codex_memory.ledger_common import VerificationError, canonical_json
from codex_memory.paths import RuntimePathError
from codex_memory.structured_ledger import connect_readonly_database, verify_ledger

from .contracts import IndexHighwater
from .errors import RetrievalValidationError
from .paths import RetrievalPaths


_IMMUTABLE_TABLES = (
    ("memory_records", "record_seq"),
    ("memory_events", "event_seq"),
    ("memory_evidence", "evidence_seq"),
    ("memory_relations", "relation_seq"),
    ("memory_candidates", "candidate_seq"),
    ("memory_candidate_events", "event_seq"),
    ("memory_candidate_evidence", "evidence_seq"),
)
_EMPTY_TIMESTAMP = "1970-01-01T00:00:00.000000Z"


class CanonicalValidationStage(StrEnum):
    OPEN = "open"
    SCHEMA_QUERY = "schema_query"
    DDL_READ = "ddl_read"
    HIGHWATER_READ = "highwater_read"
    INVARIANT_VALIDATION = "invariant_validation"


class CanonicalFailureKind(StrEnum):
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class _CanonicalSnapshotError(VerificationError):
    kind: CanonicalFailureKind
    _message = "canonical database is unavailable"

    def __init__(
        self,
        message: str | None = None,
        *,
        stage: CanonicalValidationStage = CanonicalValidationStage.OPEN,
    ) -> None:
        del message
        super().__init__(self._message)
        self.stage = CanonicalValidationStage(stage)


class CanonicalSnapshotUnavailableError(_CanonicalSnapshotError):
    """The canonical database could not be opened or read reliably."""

    kind = CanonicalFailureKind.UNAVAILABLE
    _message = "canonical database is unavailable"


class CanonicalSnapshotInvalidError(_CanonicalSnapshotError):
    """The opened canonical database failed schema or ledger validation."""

    kind = CanonicalFailureKind.INVALID
    _message = "canonical database is invalid"


@dataclass(frozen=True, slots=True)
class CanonicalSnapshot:
    schema_version: int
    highwater: IndexHighwater
    content_fingerprint: str
    source_timestamp: str


@contextmanager
def open_verified_canonical_snapshot(
    paths: RetrievalPaths,
    *, deadline=None, observer=None,
) -> Iterator[tuple[sqlite3.Connection, CanonicalSnapshot]]:
    """Yield one verified read-only canonical transaction and its snapshot."""
    stage = CanonicalValidationStage.OPEN
    connection: sqlite3.Connection | None = None
    bounded = deadline is not None and observer is None
    if bounded:
        from .injection_audit import _deadline_check
        import time
        _deadline_check(deadline)

    def set_stage(value: str) -> None:
        nonlocal stage
        stage = CanonicalValidationStage(value)

    try:
        paths.validate_canonical()
    except RuntimePathError as exc:
        raise CanonicalSnapshotUnavailableError(
            stage=stage
        ) from exc
    try:
        connection = connect_readonly_database(paths.canonical_database_path)
    except (VerificationError, sqlite3.Error, OSError, ValueError, TypeError) as exc:
        _raise_classified_failure(exc, stage=stage)
    try:
        # This boundary is deliberately outside every canonical classifier.
        # A control-channel failure says nothing about the database.
        if observer is not None:
            observer.canonical_opened(paths, deadline, in_transaction=connection.in_transaction)
        before_read = None
        if observer is not None:
            def before_read():
                # A failed observer callback must bypass both verifier wrapping
                # and canonical classification, even if it raises ValueError.
                from codex_memory.acceptance_observer import AcceptanceObserverError
                try:
                    observer.validation_read_started(deadline, stage=stage.value,
                        in_transaction=connection.in_transaction)
                except Exception as exc:
                    raise AcceptanceObserverError("validation_observer_failure") from exc
        try:
            if bounded:
                _deadline_check(deadline)
                connection.execute("PRAGMA busy_timeout=0")
                connection.set_progress_handler(lambda: int(time.monotonic_ns() >= deadline.deadline_ns), 100)
            connection.execute("BEGIN")
            verification = verify_ledger(connection, _validation_stage=set_stage,
                **({"_before_first_snapshot_read": before_read,
                    "_after_first_snapshot_read": observer.validation_finished} if observer is not None else {}))
            if observer is not None:
                observer.validation_finished()
            if verification.schema_version != 3:
                raise VerificationError("canonical schema version is unsupported")
        except (VerificationError, sqlite3.Error, OSError, ValueError, TypeError) as exc:
            if observer is not None:
                observer.validation_finished()
            _raise_classified_failure(exc, stage=stage)
        try:
            set_stage(CanonicalValidationStage.HIGHWATER_READ.value)
            highwater = _highwater(connection)
            set_stage(CanonicalValidationStage.INVARIANT_VALIDATION.value)
            content_fingerprint = _fingerprint(connection)
            source_timestamp = _source_timestamp(connection)
            snapshot = CanonicalSnapshot(
                schema_version=3,
                highwater=highwater,
                content_fingerprint=content_fingerprint,
                source_timestamp=source_timestamp,
            )
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            _raise_classified_failure(exc, stage=stage)
        try:
            yield connection, snapshot
        except (sqlite3.Error, OSError) as exc:
            raise CanonicalSnapshotUnavailableError(stage=stage) from exc
    finally:
        if connection is not None:
            if connection.in_transaction:
                connection.rollback()
            connection.close()


def capture_canonical_snapshot(paths: RetrievalPaths) -> CanonicalSnapshot:
    """Return one verified, read-only view of canonical memory state."""
    with open_verified_canonical_snapshot(paths) as (_connection, snapshot):
        return snapshot


def require_matching_canonical_snapshot(
    connection: sqlite3.Connection,
    expected: CanonicalSnapshot,
) -> None:
    """Prove that ``expected`` describes the caller's current read transaction."""
    if not isinstance(connection, sqlite3.Connection):
        raise RetrievalValidationError("canonical connection is invalid")
    if not isinstance(expected, CanonicalSnapshot):
        raise RetrievalValidationError("canonical snapshot is invalid")
    if not connection.in_transaction:
        raise RetrievalValidationError(
            "canonical snapshot binding requires an active transaction"
        )
    stage = CanonicalValidationStage.INVARIANT_VALIDATION

    def set_stage(value: str) -> None:
        nonlocal stage
        stage = CanonicalValidationStage(value)

    try:
        verification = verify_ledger(connection, _validation_stage=set_stage)
        set_stage(CanonicalValidationStage.HIGHWATER_READ.value)
        highwater = _highwater(connection)
        set_stage(CanonicalValidationStage.INVARIANT_VALIDATION.value)
        content_fingerprint = _fingerprint(connection)
        source_timestamp = _source_timestamp(connection)
    except (VerificationError, sqlite3.Error, OSError, ValueError, TypeError) as exc:
        _raise_classified_failure(exc, stage=stage)
    actual = CanonicalSnapshot(
        schema_version=verification.schema_version,
        highwater=highwater,
        content_fingerprint=content_fingerprint,
        source_timestamp=source_timestamp,
    )
    if actual != expected:
        raise RetrievalValidationError(
            "canonical snapshot does not match read transaction"
        )


_UNAVAILABLE_SQLITE_CODES = frozenset(
    {
        sqlite3.SQLITE_IOERR,
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
        sqlite3.SQLITE_CANTOPEN,
        sqlite3.SQLITE_READONLY,
        sqlite3.SQLITE_FULL,
        sqlite3.SQLITE_PERM,
        sqlite3.SQLITE_AUTH,
        sqlite3.SQLITE_PROTOCOL,
        sqlite3.SQLITE_INTERRUPT,
        sqlite3.SQLITE_NOMEM,
    }
)
_INVALID_SQLITE_CODES = frozenset(
    {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}
)


def _exception_chain(error: BaseException):
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _sqlite_primary_code(error: BaseException) -> int | None:
    code = getattr(error, "sqlite_errorcode", None)
    if type(code) is int:
        return code & 0xFF
    return None


def _classify_canonical_failure(error: BaseException) -> CanonicalFailureKind:
    """Classify a failure conservatively, preferring structured SQLite codes."""
    nodes = tuple(_exception_chain(error))
    codes = tuple(
        code
        for node in nodes
        if (code := _sqlite_primary_code(node)) is not None
    )
    if any(code in _UNAVAILABLE_SQLITE_CODES for code in codes):
        return CanonicalFailureKind.UNAVAILABLE
    if any(isinstance(node, OSError) for node in nodes):
        return CanonicalFailureKind.UNAVAILABLE
    if any(code in _INVALID_SQLITE_CODES for code in codes):
        return CanonicalFailureKind.INVALID
    sqlite_errors = tuple(node for node in nodes if isinstance(node, sqlite3.Error))
    if sqlite_errors:
        if any(_sqlite_primary_code(node) is None for node in sqlite_errors):
            return CanonicalFailureKind.UNAVAILABLE
        # SQLite reports a missing schema object as SQLITE_ERROR.  The real
        # ledger verifier wraps that read as a semantic VerificationError, so
        # retain invalid for that proven non-target/schema-mismatch path while
        # keeping an unwrapped generic SQLITE_ERROR conservative.
        if (
            any(code == sqlite3.SQLITE_ERROR for code in codes)
            and any(isinstance(node, VerificationError) for node in nodes)
        ):
            return CanonicalFailureKind.INVALID
        return CanonicalFailureKind.UNAVAILABLE
    if any(isinstance(node, VerificationError | ValueError | TypeError) for node in nodes):
        return CanonicalFailureKind.INVALID
    return CanonicalFailureKind.UNAVAILABLE


def _raise_classified_failure(
    error: BaseException, *, stage: CanonicalValidationStage
) -> None:
    error_type = (
        CanonicalSnapshotInvalidError
        if _classify_canonical_failure(error) is CanonicalFailureKind.INVALID
        else CanonicalSnapshotUnavailableError
    )
    raise error_type(stage=stage) from error


def _highwater(connection: sqlite3.Connection) -> IndexHighwater:
    values = []
    for table, sequence in _IMMUTABLE_TABLES:
        row = connection.execute(f"SELECT MAX({sequence}) FROM {table}").fetchone()
        values.append(0 if row[0] is None else row[0])
    return IndexHighwater(
        record_seq=values[0],
        memory_event_seq=values[1],
        evidence_seq=values[2],
        relation_seq=values[3],
        candidate_seq=values[4],
        candidate_event_seq=values[5],
        candidate_evidence_seq=values[6],
    )


def _fingerprint(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    _hash_json(digest, {"format": "codex-memory.canonical-snapshot.v1", "schema_version": 3})
    for table, sequence in _IMMUTABLE_TABLES:
        columns = tuple(row["name"] for row in connection.execute(f"PRAGMA table_info({table})"))
        _hash_json(digest, {"table": table, "columns": columns})
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY {sequence}"):
            _hash_json(digest, {column: row[column] for column in columns})
    return digest.hexdigest()


def _hash_json(digest: hashlib._Hash, value: object) -> None:
    digest.update(canonical_json(value).encode("utf-8"))
    digest.update(b"\n")


def _source_timestamp(connection: sqlite3.Connection) -> str:
    timestamps = []
    for table, _sequence in _IMMUTABLE_TABLES:
        value = connection.execute(f"SELECT MAX(created_at) FROM {table}").fetchone()[0]
        if value is not None:
            timestamps.append(value)
    return max(timestamps, default=_EMPTY_TIMESTAMP)
