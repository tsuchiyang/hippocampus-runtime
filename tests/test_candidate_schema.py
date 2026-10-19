from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sqlite3

import pytest

from codex_memory.paths import RuntimePaths


CANDIDATE_TABLES = {
    "memory_candidates",
    "memory_candidate_evidence",
    "memory_candidate_events",
    "memory_candidate_current",
}
CANDIDATE_INDEXES = {
    "memory_candidates_scope_topic_type",
    "memory_candidates_session_seq",
    "memory_candidate_events_candidate_event_seq",
    "memory_candidate_evidence_candidate_ordinal",
    "memory_candidate_current_state_candidate_id",
}
EXPECTED_CANDIDATE_COLUMNS = {
    "memory_candidates": (
        ("candidate_seq", "INTEGER", 0, None, 1),
        ("candidate_id", "TEXT", 1, None, 0),
        ("record_type", "TEXT", 1, None, 0),
        ("scope_kind", "TEXT", 1, None, 0),
        ("scope_key", "TEXT", 1, None, 0),
        ("topic_key", "TEXT", 1, None, 0),
        ("statement", "TEXT", 1, None, 0),
        ("value_json", "TEXT", 1, None, 0),
        ("session_id", "TEXT", 1, None, 0),
        ("turn_id", "TEXT", 1, None, 0),
        ("context_id", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("content_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_candidate_evidence": (
        ("evidence_seq", "INTEGER", 0, None, 1),
        ("evidence_id", "TEXT", 1, None, 0),
        ("candidate_id", "TEXT", 1, None, 0),
        ("ordinal", "INTEGER", 1, None, 0),
        ("uri", "TEXT", 1, None, 0),
        ("evidence_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_candidate_events": (
        ("event_seq", "INTEGER", 0, None, 1),
        ("event_id", "TEXT", 1, None, 0),
        ("candidate_id", "TEXT", 1, None, 0),
        ("action", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("event_sha256", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_candidate_current": (
        ("candidate_id", "TEXT", 0, None, 1),
        ("state", "TEXT", 1, None, 0),
        ("last_event_seq", "INTEGER", 1, None, 0),
        ("last_event_id", "TEXT", 1, None, 0),
        ("permanent_record_id", "TEXT", 0, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ),
}
EXPECTED_CANDIDATE_INDEX_COLUMNS = {
    "memory_candidates_scope_topic_type": (
        "scope_kind", "scope_key", "topic_key", "record_type", "candidate_seq",
    ),
    "memory_candidates_session_seq": ("session_id", "candidate_seq"),
    "memory_candidate_events_candidate_event_seq": ("candidate_id", "event_seq"),
    "memory_candidate_evidence_candidate_ordinal": ("candidate_id", "ordinal"),
    "memory_candidate_current_state_candidate_id": ("state", "candidate_id"),
}
EXPECTED_CANDIDATE_INDEX_TABLES = {
    "memory_candidates_scope_topic_type": "memory_candidates",
    "memory_candidates_session_seq": "memory_candidates",
    "memory_candidate_events_candidate_event_seq": "memory_candidate_events",
    "memory_candidate_evidence_candidate_ordinal": "memory_candidate_evidence",
    "memory_candidate_current_state_candidate_id": "memory_candidate_current",
}
EXPECTED_UNIQUE_COLUMNS = {
    "memory_candidates": frozenset({("candidate_id",), ("idempotency_key",)}),
    "memory_candidate_evidence": frozenset(
        {("evidence_id",), ("candidate_id", "ordinal")}
    ),
    "memory_candidate_events": frozenset({("event_id",), ("idempotency_key",)}),
    "memory_candidate_current": frozenset({("candidate_id",)}),
}
EXPECTED_FOREIGN_KEYS = {
    "memory_candidates": frozenset(),
    "memory_candidate_evidence": frozenset(
        {("candidate_id", "memory_candidates", "candidate_id", "NO ACTION", "NO ACTION", "NONE")}
    ),
    "memory_candidate_events": frozenset(
        {("candidate_id", "memory_candidates", "candidate_id", "NO ACTION", "NO ACTION", "NONE")}
    ),
    "memory_candidate_current": frozenset(
        {
            ("candidate_id", "memory_candidates", "candidate_id", "NO ACTION", "NO ACTION", "NONE"),
            ("last_event_id", "memory_candidate_events", "event_id", "NO ACTION", "NO ACTION", "NONE"),
            ("permanent_record_id", "memory_records", "record_id", "NO ACTION", "NO ACTION", "NONE"),
        }
    ),
}
IMMUTABLE_CANDIDATE_TABLES = (
    "memory_candidates",
    "memory_candidate_evidence",
    "memory_candidate_events",
)
PHASE_ONE_TABLES = (
    "events",
    "checkpoints",
    "memory_records",
    "memory_events",
    "memory_evidence",
    "memory_relations",
    "memory_current",
)


def _database():
    return import_module("codex_memory.database")


def _connect_version_two(tmp_path: Path) -> sqlite3.Connection:
    database = _database()
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        for statement in database._VERSION_ONE_STATEMENTS:
            connection.execute(statement)
        for statement in database._VERSION_TWO_STATEMENTS:
            connection.execute(statement)
        for table in database._VERSION_TWO_APPEND_ONLY_TABLES:
            for operation in ("update", "delete"):
                connection.execute(database.append_only_trigger_statement(table, operation))
        connection.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', '2')"
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    assert _version(connection) == "2"
    return connection


def _version(connection: sqlite3.Connection) -> str:
    return connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()[0]


def _snapshot_phase_one(connection: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    return {
        table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]
        for table in PHASE_ONE_TABLES
    }


def _populate_version_two(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO events "
        "(event_key, session_id, turn_id, event_name, created_at, payload_blob_sha256) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "candidate-schema-event",
            "session-1",
            "turn-1",
            "UserPromptSubmit",
            "2026-09-01T00:00:00.000000Z",
            "a" * 64,
        ),
    )
    ledger = import_module("codex_memory.structured_ledger")
    ledger.propose_record(
        connection,
        record_type="decision",
        scope_kind="project",
        scope_key="example",
        topic_key="candidate.schema",
        statement="Preserve the Phase 1A ledger.",
        value={"schema": 2},
        evidence_uris=("https://example.com/schema",),
        idempotency_key="candidate-schema-proposal",
    )


def _insert_candidate_fixture(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO memory_candidates "
        "(candidate_id, record_type, scope_kind, scope_key, topic_key, statement, "
        "value_json, session_id, turn_id, context_id, idempotency_key, "
        "content_sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "candidate-1",
            "decision",
            "project",
            "example",
            "candidate.schema",
            "Keep candidates append-only.",
            "{}",
            "session-1",
            "turn-1",
            "context-1",
            "candidate-capture-1",
            "b" * 64,
            "2026-09-01T00:00:00.000000Z",
        ),
    )
    connection.execute(
        "INSERT INTO memory_candidate_evidence "
        "(evidence_id, candidate_id, ordinal, uri, evidence_sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "candidate-evidence-1",
            "candidate-1",
            0,
            "https://example.com/candidate",
            "c" * 64,
            "2026-09-01T00:00:00.000000Z",
        ),
    )
    connection.execute(
        "INSERT INTO memory_candidate_events "
        "(event_id, candidate_id, action, payload_json, idempotency_key, "
        "event_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "candidate-event-1",
            "candidate-1",
            "capture",
            "{}",
            "candidate-event-key-1",
            "d" * 64,
            "2026-09-01T00:00:00.000000Z",
        ),
    )
    connection.execute(
        "INSERT INTO memory_candidate_current "
        "(candidate_id, state, last_event_seq, last_event_id, permanent_record_id, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "candidate-1",
            "temporary",
            1,
            "candidate-event-1",
            None,
            "2026-09-01T00:00:00.000000Z",
        ),
    )


def test_schema_two_to_three_preserves_populated_phase_zero_and_one_rows(
    tmp_path: Path,
) -> None:
    connection = _connect_version_two(tmp_path)
    _populate_version_two(connection)
    before = _snapshot_phase_one(connection)

    _database().migrate_database(connection)

    assert _version(connection) == "3"
    assert _snapshot_phase_one(connection) == before
    connection.close()


def test_schema_three_is_a_no_op_and_candidate_contract_is_complete(tmp_path: Path) -> None:
    connection = _connect_version_two(tmp_path)
    _database().migrate_database(connection)
    before = list(
        connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        )
    )

    _database().migrate_database(connection)

    assert _version(connection) == "3"
    assert list(
        connection.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
    ) == before
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    indexes = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert CANDIDATE_TABLES <= tables
    assert CANDIDATE_INDEXES <= indexes
    for table, columns in EXPECTED_CANDIDATE_COLUMNS.items():
        assert tuple(
            (row["name"], row["type"].upper(), row["notnull"], row["dflt_value"], row["pk"])
            for row in connection.execute(f"PRAGMA table_info({table})")
        ) == columns
        unique_columns = frozenset(
            tuple(row["name"] for row in connection.execute(f"PRAGMA index_info({index['name']})"))
            for index in connection.execute(f"PRAGMA index_list({table})")
            if index["unique"] == 1
        )
        assert unique_columns == EXPECTED_UNIQUE_COLUMNS[table]
    for index, columns in EXPECTED_CANDIDATE_INDEX_COLUMNS.items():
        owner = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        assert owner is not None
        assert owner["tbl_name"] == EXPECTED_CANDIDATE_INDEX_TABLES[index]
        index_definition = next(
            row
            for row in connection.execute(f"PRAGMA index_list({owner['tbl_name']})")
            if row["name"] == index
        )
        assert (
            index_definition["unique"],
            index_definition["origin"],
            index_definition["partial"],
        ) == (0, "c", 0)
        assert tuple(
            row["name"] for row in connection.execute(f"PRAGMA index_info({index})")
        ) == columns
    foreign_keys = {
        table: {
            (
                row["from"], row["table"], row["to"], row["on_update"],
                row["on_delete"], row["match"],
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }
        for table in CANDIDATE_TABLES
    }
    assert foreign_keys == EXPECTED_FOREIGN_KEYS
    candidate_triggers = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'memory_candidate_%_reject_%'"
        )
    }
    expected_triggers = {
        f"{table}_reject_{operation}"
        for table in IMMUTABLE_CANDIDATE_TABLES
        for operation in ("update", "delete")
    }
    assert candidate_triggers == expected_triggers
    assert not any(name.startswith("memory_candidate_current_reject_") for name in candidate_triggers)
    connection.close()


def test_schema_three_candidate_constraints_and_append_only_guards(tmp_path: Path) -> None:
    connection = _connect_version_two(tmp_path)
    _database().migrate_database(connection)
    _insert_candidate_fixture(connection)

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO memory_candidates "
            "(candidate_id, record_type, scope_kind, scope_key, topic_key, statement, "
            "value_json, session_id, turn_id, context_id, idempotency_key, "
            "content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "candidate-invalid",
                "note",
                "project",
                "example",
                "candidate.schema",
                "invalid",
                "{}",
                "session-1",
                "turn-1",
                "context-1",
                "candidate-invalid-key",
                "e" * 64,
                "2026-09-01T00:00:00.000000Z",
            ),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO memory_candidates "
            "(candidate_id, record_type, scope_kind, scope_key, topic_key, statement, "
            "value_json, session_id, turn_id, context_id, idempotency_key, "
            "content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "candidate-invalid-scope",
                "decision",
                "global",
                "example",
                "candidate.schema",
                "invalid scope",
                "{}",
                "session-1",
                "turn-1",
                "context-1",
                "candidate-invalid-scope-key",
                "e" * 64,
                "2026-09-01T00:00:00.000000Z",
            ),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO memory_candidate_evidence "
            "(evidence_id, candidate_id, ordinal, uri, evidence_sha256, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("negative-ordinal", "candidate-1", -1, "https://example.com", "f" * 64, "now"),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO memory_candidate_events "
            "(event_id, candidate_id, action, payload_json, idempotency_key, "
            "event_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("invalid-action", "candidate-1", "approve", "{}", "invalid-action-key", "f" * 64, "now"),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE memory_candidate_current SET state = 'invalid' WHERE candidate_id = ?",
            ("candidate-1",),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO memory_candidate_evidence "
            "(evidence_id, candidate_id, ordinal, uri, evidence_sha256, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("orphan-evidence", "missing", 0, "https://example.com", "e" * 64, "now"),
        )
    for table in IMMUTABLE_CANDIDATE_TABLES:
        with pytest.raises(sqlite3.IntegrityError, match="append-only table"):
            connection.execute(f"UPDATE {table} SET created_at = created_at")
        with pytest.raises(sqlite3.IntegrityError, match="append-only table"):
            connection.execute(f"DELETE FROM {table}")
    connection.execute("PRAGMA recursive_triggers=OFF")
    for table in IMMUTABLE_CANDIDATE_TABLES:
        row = connection.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
        columns = tuple(row.keys())
        with pytest.raises(sqlite3.IntegrityError, match="append-only table"):
            connection.execute(
                f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(row[column] for column in columns),
            )
    connection.execute("PRAGMA recursive_triggers=ON")
    connection.execute(
        "UPDATE memory_candidate_current SET updated_at = ? WHERE candidate_id = ?",
        ("2026-09-01T00:00:01.000000Z", "candidate-1"),
    )
    connection.execute("DELETE FROM memory_candidate_current WHERE candidate_id = ?", ("candidate-1",))
    connection.close()


def test_schema_three_migration_rolls_back_all_candidate_ddl(tmp_path: Path) -> None:
    connection = _connect_version_two(tmp_path)

    def deny_candidate_event_table(action, name, _column, _database_name, _trigger):
        if action == sqlite3.SQLITE_CREATE_TABLE and name == "memory_candidate_events":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_candidate_event_table)
    with pytest.raises(sqlite3.DatabaseError):
        _database().migrate_database(connection)
    connection.set_authorizer(None)

    assert _version(connection) == "2"
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert not (CANDIDATE_TABLES & tables)
    connection.close()


@pytest.mark.parametrize("version", ["4", "-1", "one", "true"])
def test_schema_three_rejects_unsupported_stored_versions(tmp_path: Path, version: str) -> None:
    connection = _connect_version_two(tmp_path)
    connection.execute("UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (version,))

    with pytest.raises(ValueError, match="schema version"):
        _database().migrate_database(connection)
    connection.close()


def test_schema_three_rejects_corrupt_existing_metadata(tmp_path: Path) -> None:
    connection = _connect_version_two(tmp_path)
    connection.execute("DELETE FROM schema_meta WHERE key = 'schema_version'")

    with pytest.raises(sqlite3.Error):
        _database().migrate_database(connection)
    connection.close()
