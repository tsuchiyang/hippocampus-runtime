from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path


def next_memory_timestamp(connection: sqlite3.Connection) -> str:
    """Return a UTC timestamp strictly after every structured-memory event."""
    current = datetime.now(UTC)
    latest_values = [
        connection.execute("SELECT MAX(created_at) FROM memory_events").fetchone()[0]
    ]
    candidate_events_exist = connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'memory_candidate_events'"
    ).fetchone()
    if candidate_events_exist is not None:
        latest_values.append(
            connection.execute(
                "SELECT MAX(created_at) FROM memory_candidate_events"
            ).fetchone()[0]
        )
    latest = max((value for value in latest_values if value is not None), default=None)
    if latest is not None:
        latest_time = datetime.fromisoformat(latest.replace("Z", "+00:00"))
        if current <= latest_time:
            current = latest_time + timedelta(microseconds=1)
    return current.isoformat(timespec="microseconds").replace("+00:00", "Z")


_VERSION_ONE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_key TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL,
        turn_id TEXT,
        event_name TEXT NOT NULL,
        tool_use_id TEXT,
        tool_name TEXT,
        prompt_text TEXT,
        created_at TEXT NOT NULL,
        payload_blob_sha256 TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS events_session_event_id_desc ON events(session_id, event_id DESC)",
    """
    CREATE INDEX IF NOT EXISTS events_session_name_event_id_desc
        ON events(session_id, event_name, event_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        checkpoint_id TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        trigger TEXT NOT NULL CHECK (trigger IN ('manual', 'auto')),
        status TEXT NOT NULL CHECK (status IN ('PREPARING', 'COMMITTED')),
        created_at TEXT NOT NULL,
        committed_at TEXT,
        raw_event_highwater INTEGER,
        manifest_relpath TEXT,
        manifest_sha256 TEXT,
        recovery_required INTEGER NOT NULL DEFAULT 0 CHECK (recovery_required IN (0, 1)),
        recovery_delivered_at TEXT,
        CHECK (
            status = 'PREPARING' OR (
                committed_at IS NOT NULL
                AND raw_event_highwater IS NOT NULL
                AND manifest_relpath IS NOT NULL
                AND manifest_sha256 IS NOT NULL
            )
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoints_session_committed_seq_desc
        ON checkpoints(session_id, checkpoint_seq DESC)
        WHERE status = 'COMMITTED'
    """,
)

_VERSION_TWO_STATEMENTS = (
    """
    CREATE TABLE memory_records (
        record_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL UNIQUE,
        record_type TEXT NOT NULL CHECK (
            record_type IN (
                'constitution', 'decision', 'assumption',
                'rejected_approach', 'failure'
            )
        ),
        scope_kind TEXT NOT NULL CHECK (scope_kind IN ('project', 'workspace')),
        scope_key TEXT NOT NULL,
        topic_key TEXT NOT NULL,
        statement TEXT NOT NULL,
        value_json TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        content_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE memory_events (
        event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        record_id TEXT NOT NULL REFERENCES memory_records(record_id),
        action TEXT NOT NULL CHECK (
            action IN ('propose', 'activate', 'supersede', 'invalidate', 'close', 'reopen')
        ),
        payload_json TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        event_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE memory_evidence (
        evidence_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_id TEXT NOT NULL UNIQUE,
        record_id TEXT NOT NULL REFERENCES memory_records(record_id),
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        uri TEXT NOT NULL,
        evidence_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(record_id, ordinal)
    )
    """,
    """
    CREATE TABLE memory_relations (
        relation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        relation_id TEXT NOT NULL UNIQUE,
        event_id TEXT NOT NULL REFERENCES memory_events(event_id),
        relation_type TEXT NOT NULL CHECK (relation_type IN ('superseded_by')),
        source_record_id TEXT NOT NULL REFERENCES memory_records(record_id),
        target_record_id TEXT NOT NULL REFERENCES memory_records(record_id),
        relation_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE memory_current (
        record_id TEXT PRIMARY KEY REFERENCES memory_records(record_id),
        state TEXT NOT NULL CHECK (
            state IN ('proposed', 'active', 'superseded', 'invalidated', 'closed')
        ),
        last_event_seq INTEGER NOT NULL,
        last_event_id TEXT NOT NULL REFERENCES memory_events(event_id),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX memory_records_scope_topic_type
        ON memory_records(scope_kind, scope_key, topic_key, record_type, record_seq)
    """,
    """
    CREATE INDEX memory_events_record_event_seq
        ON memory_events(record_id, event_seq)
    """,
    """
    CREATE INDEX memory_evidence_record_ordinal
        ON memory_evidence(record_id, ordinal)
    """,
    """
    CREATE INDEX memory_relations_source
        ON memory_relations(source_record_id, relation_seq)
    """,
    """
    CREATE INDEX memory_relations_target
        ON memory_relations(target_record_id, relation_seq)
    """,
    """
    CREATE INDEX memory_current_state_record_id
        ON memory_current(state, record_id)
    """,
)

_VERSION_THREE_STATEMENTS = (
    """
    CREATE TABLE memory_candidates (
        candidate_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT NOT NULL UNIQUE,
        record_type TEXT NOT NULL CHECK (
            record_type IN (
                'constitution', 'decision', 'assumption',
                'rejected_approach', 'failure'
            )
        ),
        scope_kind TEXT NOT NULL CHECK (scope_kind IN ('project', 'workspace')),
        scope_key TEXT NOT NULL,
        topic_key TEXT NOT NULL,
        statement TEXT NOT NULL,
        value_json TEXT NOT NULL,
        session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        context_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        content_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE memory_candidate_evidence (
        evidence_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_id TEXT NOT NULL UNIQUE,
        candidate_id TEXT NOT NULL REFERENCES memory_candidates(candidate_id),
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        uri TEXT NOT NULL,
        evidence_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(candidate_id, ordinal)
    )
    """,
    """
    CREATE TABLE memory_candidate_events (
        event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        candidate_id TEXT NOT NULL REFERENCES memory_candidates(candidate_id),
        action TEXT NOT NULL CHECK (
            action IN ('capture', 'defer', 'promote', 'reject', 'expire')
        ),
        payload_json TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        event_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE memory_candidate_current (
        candidate_id TEXT PRIMARY KEY REFERENCES memory_candidates(candidate_id),
        state TEXT NOT NULL CHECK (
            state IN ('temporary', 'promoted', 'rejected', 'expired')
        ),
        last_event_seq INTEGER NOT NULL,
        last_event_id TEXT NOT NULL REFERENCES memory_candidate_events(event_id),
        permanent_record_id TEXT REFERENCES memory_records(record_id),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX memory_candidates_scope_topic_type
        ON memory_candidates(scope_kind, scope_key, topic_key, record_type, candidate_seq)
    """,
    """
    CREATE INDEX memory_candidates_session_seq
        ON memory_candidates(session_id, candidate_seq)
    """,
    """
    CREATE INDEX memory_candidate_events_candidate_event_seq
        ON memory_candidate_events(candidate_id, event_seq)
    """,
    """
    CREATE INDEX memory_candidate_evidence_candidate_ordinal
        ON memory_candidate_evidence(candidate_id, ordinal)
    """,
    """
    CREATE INDEX memory_candidate_current_state_candidate_id
        ON memory_candidate_current(state, candidate_id)
    """,
)

STRUCTURED_TABLE_STATEMENTS = _VERSION_TWO_STATEMENTS[:5]
STRUCTURED_INDEX_STATEMENTS = _VERSION_TWO_STATEMENTS[5:]

_VERSION_TWO_APPEND_ONLY_TABLES = (
    "memory_records",
    "memory_events",
    "memory_evidence",
    "memory_relations",
)

_VERSION_THREE_APPEND_ONLY_TABLES = (
    "memory_candidates",
    "memory_candidate_evidence",
    "memory_candidate_events",
)

APPEND_ONLY_UNIQUE_KEYS = {
    "memory_records": (
        ("record_seq",),
        ("record_id",),
        ("idempotency_key",),
    ),
    "memory_events": (
        ("event_seq",),
        ("event_id",),
        ("idempotency_key",),
    ),
    "memory_evidence": (
        ("evidence_seq",),
        ("evidence_id",),
        ("record_id", "ordinal"),
    ),
    "memory_relations": (
        ("relation_seq",),
        ("relation_id",),
    ),
    "memory_candidates": (
        ("candidate_seq",),
        ("candidate_id",),
        ("idempotency_key",),
    ),
    "memory_candidate_evidence": (
        ("evidence_seq",),
        ("evidence_id",),
        ("candidate_id", "ordinal"),
    ),
    "memory_candidate_events": (
        ("event_seq",),
        ("event_id",),
        ("idempotency_key",),
    ),
}


def append_only_trigger_statement(table: str, operation: str) -> str:
    return f"""
        CREATE TRIGGER {table}_reject_{operation}
        BEFORE {operation.upper()} ON {table}
        BEGIN
            SELECT RAISE(ABORT, 'append-only table');
        END
    """


def append_only_insert_guard_statement(
    table: str, unique_keys: tuple[tuple[str, ...], ...]
) -> str:
    conflicts = " OR ".join(
        "(" + " AND ".join(f"{column} = NEW.{column}" for column in columns) + ")"
        for columns in unique_keys
    )
    return f"""
        CREATE TRIGGER {table}_guard_insert_conflict
        BEFORE INSERT ON {table}
        WHEN EXISTS (
            SELECT 1 FROM {table} WHERE {conflicts}
        )
        BEGIN
            SELECT RAISE(ABORT, 'append-only table');
        END
    """


def connect_database(path: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open a caller-owned SQLite connection configured for durable storage."""
    if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or not 0 <= busy_timeout_ms <= 5000:
        raise ValueError("busy_timeout_ms must be between zero and 5000")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None, timeout=busy_timeout_ms / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA recursive_triggers=ON")
    return connection


def migrate_database(connection: sqlite3.Connection) -> None:
    """Atomically migrate the runtime database to schema version three."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        version_row = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if version_row is None:
            version = 0
        elif version_row[0] in {"1", "2", "3"}:
            version = int(version_row[0])
        else:
            raise ValueError(f"unsupported schema version: {version_row[0]!r}")

        for statement in _VERSION_ONE_STATEMENTS:
            connection.execute(statement)
        if version == 0:
            connection.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1')"
            )
            version = 1

        if version == 1:
            _migrate_to_version_two(connection)
            connection.execute(
                "UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'"
            )
            version = 2
        if version == 2:
            _migrate_to_version_three(connection)
            connection.execute(
                "UPDATE schema_meta SET value = '3' WHERE key = 'schema_version'"
            )

        _ensure_append_only_insert_guards(connection)

        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_to_version_two(connection: sqlite3.Connection) -> None:
    for statement in _VERSION_TWO_STATEMENTS:
        connection.execute(statement)
    for table in _VERSION_TWO_APPEND_ONLY_TABLES:
        for operation in ("update", "delete"):
            connection.execute(append_only_trigger_statement(table, operation))


def _migrate_to_version_three(connection: sqlite3.Connection) -> None:
    for statement in _VERSION_THREE_STATEMENTS:
        connection.execute(statement)
    for table in _VERSION_THREE_APPEND_ONLY_TABLES:
        for operation in ("update", "delete"):
            connection.execute(append_only_trigger_statement(table, operation))


def _ensure_append_only_insert_guards(connection: sqlite3.Connection) -> None:
    for table, unique_keys in APPEND_ONLY_UNIQUE_KEYS.items():
        name = f"{table}_guard_insert_conflict"
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (name,),
        ).fetchone()
        if exists is None:
            connection.execute(append_only_insert_guard_statement(table, unique_keys))
