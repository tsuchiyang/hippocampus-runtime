from __future__ import annotations

import sqlite3
import hashlib
import importlib.util
from importlib import import_module
from pathlib import Path
from queue import Queue
from threading import Barrier, Thread

import pytest

from codex_memory.paths import RuntimePaths
from codex_memory.structured_ledger import _normalize_schema_sql


REQUIRED_TABLES = {
    "schema_meta",
    "events",
    "checkpoints",
    "memory_records",
    "memory_events",
    "memory_evidence",
    "memory_relations",
    "memory_current",
    "memory_candidates",
    "memory_candidate_evidence",
    "memory_candidate_events",
    "memory_candidate_current",
}
REQUIRED_INDEXES = {
    "events_session_event_id_desc",
    "events_session_name_event_id_desc",
    "checkpoints_session_committed_seq_desc",
    "memory_records_scope_topic_type",
    "memory_events_record_event_seq",
    "memory_evidence_record_ordinal",
    "memory_relations_source",
    "memory_relations_target",
    "memory_current_state_record_id",
    "memory_candidates_scope_topic_type",
    "memory_candidates_session_seq",
    "memory_candidate_events_candidate_event_seq",
    "memory_candidate_evidence_candidate_ordinal",
    "memory_candidate_current_state_candidate_id",
}
REQUIRED_TRIGGERS = {
    name
    for table in (
        "memory_records",
        "memory_events",
        "memory_evidence",
        "memory_relations",
        "memory_candidates",
        "memory_candidate_evidence",
        "memory_candidate_events",
    )
    for name in (
        f"{table}_reject_update",
        f"{table}_reject_delete",
        f"{table}_guard_insert_conflict",
    )
}

CANDIDATE_COLUMN_SHAPES = {
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
CANDIDATE_UNIQUE_COLUMNS = {
    "memory_candidates": frozenset({("candidate_id",), ("idempotency_key",)}),
    "memory_candidate_evidence": frozenset(
        {("evidence_id",), ("candidate_id", "ordinal")}
    ),
    "memory_candidate_events": frozenset({("event_id",), ("idempotency_key",)}),
    "memory_candidate_current": frozenset({("candidate_id",)}),
}
CANDIDATE_FOREIGN_KEYS = {
    "memory_candidates": frozenset(),
    "memory_candidate_evidence": frozenset(
        {
            (
                "candidate_id",
                "memory_candidates",
                "candidate_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            )
        }
    ),
    "memory_candidate_events": frozenset(
        {
            (
                "candidate_id",
                "memory_candidates",
                "candidate_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            )
        }
    ),
    "memory_candidate_current": frozenset(
        {
            (
                "candidate_id",
                "memory_candidates",
                "candidate_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
            (
                "last_event_id",
                "memory_candidate_events",
                "event_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
            (
                "permanent_record_id",
                "memory_records",
                "record_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        }
    ),
}
CANDIDATE_INDEXES = {
    "memory_candidates_scope_topic_type": (
        "memory_candidates",
        ("scope_kind", "scope_key", "topic_key", "record_type", "candidate_seq"),
    ),
    "memory_candidates_session_seq": (
        "memory_candidates",
        ("session_id", "candidate_seq"),
    ),
    "memory_candidate_events_candidate_event_seq": (
        "memory_candidate_events",
        ("candidate_id", "event_seq"),
    ),
    "memory_candidate_evidence_candidate_ordinal": (
        "memory_candidate_evidence",
        ("candidate_id", "ordinal"),
    ),
    "memory_candidate_current_state_candidate_id": (
        "memory_candidate_current",
        ("state", "candidate_id"),
    ),
}


def _connect(tmp_path: Path) -> sqlite3.Connection:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    database.migrate_database(connection)
    return connection


def _ledger():
    return import_module("codex_memory.structured_ledger")


@pytest.mark.parametrize("separator", ["\u00a0", "\u2003", "\u2007", "\u202f", "\u3000"])
def test_schema_sql_normalization_does_not_fold_unicode_separator(
    separator: str,
) -> None:
    assert _normalize_schema_sql(
        f"CREATE{separator}TABLE x(a TEXT)"
    ) != _normalize_schema_sql("CREATE TABLE x(a TEXT)")


@pytest.mark.parametrize("separator", ["\t", "\n", "\f", "\r", " "])
def test_schema_sql_normalization_accepts_sqlite_tokenizer_whitespace(
    separator: str,
) -> None:
    assert _normalize_schema_sql(f"CREATE{separator}TABLE x(a TEXT)") == _normalize_schema_sql(
        "CREATE TABLE x(a TEXT)"
    )


@pytest.mark.parametrize(
    "quoted",
    ["'a\u00a0b'", '"a\u00a0b"', "`a\u00a0b`", "[a\u00a0b]"],
)
def test_schema_sql_normalization_preserves_unicode_separator_inside_quotes(
    quoted: str,
) -> None:
    normalized = _normalize_schema_sql(f"CHECK (value = {quoted})")
    without_separator = _normalize_schema_sql(f"CHECK (value = {quoted.replace(chr(0x00A0), '')})")

    assert "\u00a0" in normalized
    assert normalized != without_separator


def test_canonical_schema_validation_rejects_unicode_separator_in_trigger_body(
    tmp_path: Path,
) -> None:
    database = import_module("codex_memory.database")
    connection = _connect(tmp_path)
    database_path = RuntimePaths.for_repo(tmp_path).db_path
    connection.close()
    with sqlite3.connect(database_path) as tamper:
        tamper.execute("PRAGMA writable_schema=ON")
        tamper.execute(
            """UPDATE sqlite_master
            SET sql = replace(sql, 'BEFORE UPDATE', 'BEFORE\u00a0UPDATE')
            WHERE type = 'trigger' AND name = 'memory_records_reject_update'"""
        )
        tamper.execute("PRAGMA writable_schema=OFF")

    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA recursive_triggers=ON")
    try:
        with pytest.raises(_ledger().VerificationError, match="ledger verification failed|trigger|schema|shape"):
            _ledger().verify_ledger(connection)
    finally:
        connection.close()


def _propose(connection: sqlite3.Connection, **changes):
    arguments = {
        "record_type": "decision",
        "scope_kind": "project",
        "scope_key": "example",
        "topic_key": "storage.engine",
        "statement": "Use SQLite.",
        "value": {"name": "sqlite3"},
        "evidence_uris": ("https://example.com/design",),
        "idempotency_key": "proposal-1",
    }
    arguments.update(changes)
    return _ledger().propose_record(connection, **arguments)


def _seed_version_one(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1');
        CREATE TABLE events (
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
        );
        CREATE INDEX events_session_event_id_desc
            ON events(session_id, event_id DESC);
        CREATE INDEX events_session_name_event_id_desc
            ON events(session_id, event_name, event_id DESC);
        CREATE TABLE checkpoints (
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
        );
        CREATE INDEX checkpoints_session_committed_seq_desc
            ON checkpoints(session_id, checkpoint_seq DESC)
            WHERE status = 'COMMITTED';
        INSERT INTO events (
            event_key, session_id, turn_id, event_name, created_at,
            payload_blob_sha256
        ) VALUES (
            'event-key', 'session-1', 'turn-1', 'UserPromptSubmit',
            '2026-08-31T00:00:00.000000Z', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        );
        INSERT INTO checkpoints (
            checkpoint_id, session_id, turn_id, trigger, status, created_at,
            committed_at, raw_event_highwater, manifest_relpath,
            manifest_sha256
        ) VALUES (
            'checkpoint-1', 'session-1', 'turn-1', 'manual', 'COMMITTED',
            '2026-08-31T00:00:01.000000Z', '2026-08-31T00:00:02.000000Z',
            1, 'session/checkpoint.json',
            'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        );
        """
    )


def _object_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (object_type,)
        )
    }


def test_fresh_migration_creates_complete_version_two_schema(tmp_path: Path) -> None:
    connection = _connect(tmp_path)

    assert connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()[0] == "3"
    assert REQUIRED_TABLES <= _object_names(connection, "table")
    assert REQUIRED_INDEXES <= _object_names(connection, "index")
    assert REQUIRED_TRIGGERS == (
        REQUIRED_TRIGGERS & _object_names(connection, "trigger")
    )
    connection.close()


def test_version_one_migration_preserves_phase_zero_rows(tmp_path: Path) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    _seed_version_one(connection)

    database.migrate_database(connection)

    assert connection.execute("SELECT value FROM schema_meta").fetchone()[0] == "3"
    assert dict(connection.execute("SELECT * FROM events").fetchone()) == {
        "event_id": 1,
        "event_key": "event-key",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "event_name": "UserPromptSubmit",
        "tool_use_id": None,
        "tool_name": None,
        "prompt_text": None,
        "created_at": "2026-08-31T00:00:00.000000Z",
        "payload_blob_sha256": "a" * 64,
    }
    checkpoint = connection.execute("SELECT * FROM checkpoints").fetchone()
    assert checkpoint["checkpoint_id"] == "checkpoint-1"
    assert checkpoint["raw_event_highwater"] == 1
    assert checkpoint["manifest_sha256"] == "b" * 64
    connection.close()


def test_version_one_migration_preserves_phase_zero_schema_self_healing(
    tmp_path: Path,
) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    _seed_version_one(connection)
    connection.execute("DROP INDEX events_session_event_id_desc")

    database.migrate_database(connection)

    assert "events_session_event_id_desc" in _object_names(connection, "index")
    assert connection.execute("SELECT value FROM schema_meta").fetchone()[0] == "3"
    connection.close()


@pytest.mark.parametrize("damage", ["table", "index"])
def test_version_two_migration_preserves_phase_zero_schema_self_healing(
    tmp_path: Path, damage: str
) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    database.migrate_database(connection)
    if damage == "index":
        connection.execute(
            "INSERT INTO events "
            "(event_key, session_id, event_name, created_at, payload_blob_sha256) "
            "VALUES ('preserved-event', 'session-1', 'UserPromptSubmit', "
            "'2026-08-31T00:00:00.000000Z', ?)",
            ("a" * 64,),
        )
        connection.execute("DROP INDEX events_session_event_id_desc")
    else:
        connection.execute("DROP TABLE events")

    database.migrate_database(connection)

    assert "events" in _object_names(connection, "table")
    assert "events_session_event_id_desc" in _object_names(connection, "index")
    assert connection.execute("SELECT value FROM schema_meta").fetchone()[0] == "3"
    expected_rows = 1 if damage == "index" else 0
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == expected_rows
    connection.close()


@pytest.mark.parametrize(
    ("table", "column", "invalid"),
    [
        ("memory_records", "record_type", "note"),
        ("memory_records", "scope_kind", "global"),
        ("memory_events", "action", "approve"),
        ("memory_relations", "relation_type", "depends_on"),
        ("memory_current", "state", "deleted"),
    ],
)
def test_version_two_schema_rejects_invalid_enums(
    tmp_path: Path, table: str, column: str, invalid: str
) -> None:
    connection = _connect(tmp_path)
    fixtures = {
        "memory_records": (
            "INSERT INTO memory_records "
            "(record_id, record_type, scope_kind, scope_key, topic_key, statement, "
            "value_json, idempotency_key, content_sha256, created_at) "
            "VALUES ('r1', 'decision', 'project', 'p', 'topic', 'statement', "
            "'null', 'proposal-key', ?, '2026-08-31T00:00:00.000000Z')",
            ("a" * 64,),
        ),
        "memory_events": (
            "INSERT INTO memory_events "
            "(event_id, record_id, action, payload_json, idempotency_key, "
            "event_sha256, created_at) VALUES "
            "('e1', 'r1', 'propose', '{}', 'event-key', ?, "
            "'2026-08-31T00:00:00.000000Z')",
            ("b" * 64,),
        ),
        "memory_relations": (
            "INSERT INTO memory_relations "
            "(relation_id, event_id, relation_type, source_record_id, "
            "target_record_id, relation_sha256, created_at) VALUES "
            "('l1', 'e1', 'superseded_by', 'r1', 'r2', ?, "
            "'2026-08-31T00:00:00.000000Z')",
            ("c" * 64,),
        ),
        "memory_current": (
            "INSERT INTO memory_current "
            "(record_id, state, last_event_seq, last_event_id, updated_at) VALUES "
            "('r1', 'proposed', 1, 'e1', '2026-08-31T00:00:00.000000Z')",
            (),
        ),
    }
    record_sql, record_params = fixtures["memory_records"]
    connection.execute(record_sql, record_params)
    if table == "memory_relations":
        connection.execute(
            record_sql.replace("'r1'", "'r2'", 1).replace("'proposal-key'", "'proposal-key-2'"),
            record_params,
        )
    if table in {"memory_relations", "memory_current"}:
        event_sql, event_params = fixtures["memory_events"]
        connection.execute(event_sql, event_params)

    sql, params = fixtures[table]
    sql = sql.replace(f"'{fixtures_value(table, column)}'", f"'{invalid}'", 1)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(sql, params)
    connection.close()


def fixtures_value(table: str, column: str) -> str:
    return {
        ("memory_records", "record_type"): "decision",
        ("memory_records", "scope_kind"): "project",
        ("memory_events", "action"): "propose",
        ("memory_relations", "relation_type"): "superseded_by",
        ("memory_current", "state"): "proposed",
    }[(table, column)]


def test_version_one_to_two_migration_rolls_back_atomically(tmp_path: Path) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    _seed_version_one(connection)

    def deny_memory_events(action, table, _column, _database, _trigger):
        if action == sqlite3.SQLITE_CREATE_TABLE and table == "memory_events":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_memory_events)
    with pytest.raises(sqlite3.DatabaseError):
        database.migrate_database(connection)
    connection.set_authorizer(None)

    assert connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()[0] == "1"
    assert not ({"memory_records", "memory_events"} & _object_names(connection, "table"))
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 1
    connection.close()


@pytest.mark.parametrize(
    "record_type",
    ["constitution", "decision", "assumption", "rejected_approach", "failure"],
)
@pytest.mark.parametrize("scope_kind", ["project", "workspace"])
def test_propose_stores_every_record_type_and_scope(
    tmp_path: Path, record_type: str, scope_kind: str
) -> None:
    connection = _connect(tmp_path)

    result = _propose(
        connection,
        record_type=record_type,
        scope_kind=scope_kind,
        idempotency_key=f"{record_type}-{scope_kind}",
    )

    stored = connection.execute(
        "SELECT * FROM memory_records WHERE record_id = ?", (result.record_id,)
    ).fetchone()
    assert stored["record_type"] == record_type
    assert stored["scope_kind"] == scope_kind
    assert stored["scope_key"] == "example"
    assert result.state == "proposed"
    connection.close()


def test_canonical_json_and_content_hash_match_hand_derived_fixture(tmp_path: Path) -> None:
    connection = _connect(tmp_path)

    result = _propose(connection)

    stored = connection.execute(
        "SELECT value_json, content_sha256 FROM memory_records WHERE record_id = ?",
        (result.record_id,),
    ).fetchone()
    assert stored["value_json"] == '{"name":"sqlite3"}'
    assert stored["content_sha256"] == (
        "11f5883994e52d3e83ee3f67ced7439cb006e0aebb14b38b8aace3438dcda262"
    )
    assert result.content_sha256 == stored["content_sha256"]
    connection.close()


def test_shared_record_content_identity_matches_phase_1a_proposal(tmp_path: Path) -> None:
    spec = importlib.util.find_spec("codex_memory.ledger_common")
    assert spec is not None, "shared ledger primitives module is required"
    ledger_common = import_module("codex_memory.ledger_common")
    assert hasattr(ledger_common, "validate_record_content")
    assert hasattr(ledger_common, "scope_keys_equal")

    connection = _connect(tmp_path)
    result = _propose(connection)
    content = ledger_common.validate_record_content(
        record_type="decision",
        scope_kind="project",
        scope_key="example",
        topic_key="storage.engine",
        statement="Use SQLite.",
        value={"name": "sqlite3"},
    )

    assert content.value_json == '{"name":"sqlite3"}'
    assert content.content_sha256 == result.content_sha256
    assert _ledger().canonical_json is ledger_common.canonical_json
    assert _ledger().ValidationError is ledger_common.ValidationError
    assert ledger_common.validate_evidence_uris(("https://example.com/proof",), maximum=1) == (
        "https://example.com/proof",
    )
    assert ledger_common.scope_keys_equal("Repo", "repo", windows=True)
    assert not ledger_common.scope_keys_equal("Repo", "repo", windows=False)
    connection.close()


def test_proposal_preserves_evidence_order_and_duplicates(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    evidence = (
        "https://example.com/two",
        "https://example.com/one",
        "https://example.com/two",
    )

    result = _propose(connection, evidence_uris=evidence)

    rows = connection.execute(
        "SELECT ordinal, uri, evidence_sha256 FROM memory_evidence "
        "WHERE record_id = ? ORDER BY ordinal",
        (result.record_id,),
    ).fetchall()
    assert [(row["ordinal"], row["uri"]) for row in rows] == list(enumerate(evidence))
    first_preimage = (
        f'{{"record_id":"{result.record_id}","uri":"https://example.com/two"}}'
    ).encode("utf-8")
    assert rows[0]["evidence_sha256"] == hashlib.sha256(first_preimage).hexdigest()
    assert result.evidence_uris == evidence
    connection.close()


def test_equivalent_proposal_retry_returns_original_result(tmp_path: Path) -> None:
    connection = _connect(tmp_path)

    first = _propose(connection)
    second = _propose(connection)

    assert second == first
    assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 1
    connection.close()


@pytest.mark.parametrize(
    "change",
    [
        {"statement": "Use PostgreSQL."},
        {"value": {"name": "apsw"}},
        {"evidence_uris": ("https://example.com/other",)},
        {"evidence_uris": ("https://example.com/design", "https://example.com/design")},
    ],
)
def test_proposal_key_reuse_with_different_input_conflicts(
    tmp_path: Path, change: dict[str, object]
) -> None:
    connection = _connect(tmp_path)
    _propose(connection)

    with pytest.raises(_ledger().IdempotencyConflict):
        _propose(connection, **change)

    assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
    connection.close()


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/a%20b?x=1#part",
        "file:///C:/repo/design.md",
        "urn:isbn:0451450523",
        "git+ssh://example.com/repo.git",
        "codex:checkpoint/123",
    ],
)
def test_evidence_uri_accepts_absolute_well_formed_values(
    tmp_path: Path, uri: str
) -> None:
    connection = _connect(tmp_path)
    result = _propose(connection, evidence_uris=(uri,))

    assert result.evidence_uris == (uri,)
    connection.close()


@pytest.mark.parametrize(
    "uri",
    [
        "relative/path",
        "https:",
        "https://",
        "https:?query-only",
        "https://?query-only",
        "https:///missing-authority",
        "://missing-scheme",
        "1http://example.com",
        "https://example.com/with space",
        "https://example.com/with\\backslash",
        "https://example.com/%",
        "https://example.com/%zz",
        "https://example.com/line\nfeed",
        "https://example.com/\x00control",
        "https://example.com/\x7fcontrol",
    ],
)
def test_evidence_uri_rejects_hostile_or_relative_values(tmp_path: Path, uri: str) -> None:
    connection = _connect(tmp_path)

    with pytest.raises(_ledger().ValidationError):
        _propose(connection, evidence_uris=(uri,))

    assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    connection.close()


@pytest.mark.parametrize(
    "topic",
    ["a", "storage.engine", "storage/engine", "a_b.c-1", "a/b.c_d-e2"],
)
def test_topic_key_accepts_frozen_format(tmp_path: Path, topic: str) -> None:
    connection = _connect(tmp_path)
    result = _propose(connection, topic_key=topic)

    assert result.topic_key == topic
    connection.close()


@pytest.mark.parametrize(
    "topic",
    ["", ".a", "a.", "a..b", "a//b", "UPPER", "a b", "a:b", "é"],
)
def test_topic_key_rejects_invalid_format(tmp_path: Path, topic: str) -> None:
    connection = _connect(tmp_path)

    with pytest.raises(_ledger().ValidationError):
        _propose(connection, topic_key=topic)

    connection.close()


@pytest.mark.parametrize(
    ("change", "valid"),
    [
        ({"scope_key": "s" * 256}, True),
        ({"scope_key": "é" * 128}, True),
        ({"scope_key": "s" * 257}, False),
        ({"scope_key": " \t "}, False),
        ({"statement": "x" * 8192}, True),
        ({"statement": "é" * 4096}, True),
        ({"statement": "x" * 8193}, False),
        ({"statement": "\n\t"}, False),
        ({"value": "x" * 32766}, True),
        ({"value": "x" * 32767}, False),
        ({"idempotency_key": "k" * 128}, True),
        ({"idempotency_key": "k" * 129}, False),
        ({"idempotency_key": "bad key"}, False),
        ({"evidence_uris": ("x:" + "a" * 2046,)}, True),
        ({"evidence_uris": ("x:" + "a" * 2047,)}, False),
    ],
)
def test_utf8_byte_and_idempotency_boundaries(
    tmp_path: Path, change: dict[str, object], valid: bool
) -> None:
    connection = _connect(tmp_path)
    if valid:
        assert _propose(connection, **change).record_id
    else:
        with pytest.raises(_ledger().ValidationError):
            _propose(connection, **change)
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    connection.close()


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), {1: "non-string"}, {"nested": {2: "key"}}, "\ud800"],
)
def test_canonical_value_rejects_nonfinite_nonstring_keys_and_unicode_errors(
    tmp_path: Path, value: object
) -> None:
    connection = _connect(tmp_path)

    with pytest.raises(_ledger().ValidationError):
        _propose(connection, value=value)

    connection.close()


def test_canonical_value_rejects_circular_python_container(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    value: list[object] = []
    value.append(value)

    with pytest.raises(_ledger().ValidationError):
        _propose(connection, value=value)

    connection.close()


def test_canonical_value_rejects_deep_python_container_without_recursion_leak(
    tmp_path: Path,
) -> None:
    connection = _connect(tmp_path)
    value: object = None
    for _ in range(1500):
        value = [value]

    with pytest.raises(_ledger().ValidationError):
        _propose(connection, value=value)

    assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    connection.close()


@pytest.mark.parametrize(
    "change",
    [
        {"record_type": []},
        {"scope_kind": []},
    ],
)
def test_enum_inputs_raise_bounded_validation_for_non_strings(
    tmp_path: Path, change: dict[str, object]
) -> None:
    connection = _connect(tmp_path)
    with pytest.raises(_ledger().ValidationError):
        _propose(connection, **change)
    connection.close()


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":1,"a":2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1} trailing',
        '',
    ],
)
def test_cli_value_json_parser_rejects_duplicates_nonfinite_and_trailing(
    raw: str,
) -> None:
    with pytest.raises(_ledger().ValidationError):
        _ledger().parse_json_value(raw)


def test_cli_value_json_parser_accepts_any_single_json_value() -> None:
    assert _ledger().parse_json_value(' {"z":"é","a":[1,true,null]} \n') == {
        "z": "é",
        "a": [1, True, None],
    }


def test_source_tables_reject_direct_update_and_delete(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    result = _propose(connection)
    event_id = connection.execute(
        "SELECT event_id FROM memory_events WHERE record_id = ?", (result.record_id,)
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO memory_records "
        "(record_id, record_type, scope_kind, scope_key, topic_key, statement, "
        "value_json, idempotency_key, content_sha256, created_at) VALUES "
        "('r-target', 'decision', 'project', 'example', 'target', 'Target.', "
        "'null', 'target-proposal', ?, '2026-08-31T00:00:00.000000Z')",
        ("d" * 64,),
    )
    connection.execute(
        "INSERT INTO memory_relations "
        "(relation_id, event_id, relation_type, source_record_id, target_record_id, "
        "relation_sha256, created_at) VALUES "
        "('relation-1', ?, 'superseded_by', ?, 'r-target', ?, "
        "'2026-08-31T00:00:00.000000Z')",
        (event_id, result.record_id, "e" * 64),
    )

    keys = {
        "memory_records": ("record_id", result.record_id),
        "memory_events": ("event_id", event_id),
        "memory_evidence": ("record_id", result.record_id),
        "memory_relations": ("relation_id", "relation-1"),
    }
    for table, (column, value) in keys.items():
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                f"UPDATE {table} SET created_at = created_at WHERE {column} = ?", (value,)
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(f"DELETE FROM {table} WHERE {column} = ?", (value,))
    connection.close()


def _insert_replace_fixture(connection: sqlite3.Connection, table: str) -> str:
    record_sql = (
        "INSERT INTO memory_records "
        "(record_id, record_type, scope_kind, scope_key, topic_key, statement, "
        "value_json, idempotency_key, content_sha256, created_at) VALUES "
        "(?, 'decision', 'project', 'example', ?, 'Statement.', 'null', ?, ?, ?)"
    )
    created_at = "2026-08-31T00:00:00.000000Z"
    if table == "memory_records":
        connection.execute(
            record_sql,
            ("replace-record", "replace.record", "replace-record-key", "a" * 64, created_at),
        )
        return "replace-record"

    connection.execute(
        record_sql,
        ("parent-record", "parent.record", "parent-record-key", "b" * 64, created_at),
    )
    if table == "memory_events":
        connection.execute(
            "INSERT INTO memory_events "
            "(event_id, record_id, action, payload_json, idempotency_key, "
            "event_sha256, created_at) VALUES "
            "('replace-event', 'parent-record', 'propose', '{}', "
            "'replace-event-key', ?, ?)",
            ("c" * 64, created_at),
        )
        return "replace-event"
    if table == "memory_evidence":
        connection.execute(
            "INSERT INTO memory_evidence "
            "(evidence_id, record_id, ordinal, uri, evidence_sha256, created_at) "
            "VALUES ('replace-evidence', 'parent-record', 0, 'urn:test:evidence', ?, ?)",
            ("d" * 64, created_at),
        )
        return "replace-evidence"

    connection.execute(
        record_sql,
        ("target-record", "target.record", "target-record-key", "e" * 64, created_at),
    )
    connection.execute(
        "INSERT INTO memory_events "
        "(event_id, record_id, action, payload_json, idempotency_key, "
        "event_sha256, created_at) VALUES "
        "('relation-event', 'parent-record', 'supersede', ?, "
        "'relation-event-key', ?, ?)",
        ('{"replacement_record_id":"target-record"}', "f" * 64, created_at),
    )
    connection.execute(
        "INSERT INTO memory_relations "
        "(relation_id, event_id, relation_type, source_record_id, target_record_id, "
        "relation_sha256, created_at) VALUES "
        "('replace-relation', 'relation-event', 'superseded_by', 'parent-record', "
        "'target-record', ?, ?)",
        ("1" * 64, created_at),
    )
    return "replace-relation"


@pytest.mark.parametrize(
    ("table", "id_column"),
    [
        ("memory_records", "record_id"),
        ("memory_events", "event_id"),
        ("memory_evidence", "evidence_id"),
        ("memory_relations", "relation_id"),
    ],
)
def test_source_tables_reject_insert_or_replace(
    tmp_path: Path, table: str, id_column: str
) -> None:
    connection = _connect(tmp_path)
    identity = _insert_replace_fixture(connection, table)
    row = connection.execute(
        f"SELECT * FROM {table} WHERE {id_column} = ?", (identity,)
    ).fetchone()
    columns = tuple(row.keys())
    replacement = dict(row)
    replacement["created_at"] = "2026-08-31T23:59:59.000000Z"
    connection.execute("PRAGMA recursive_triggers=OFF")

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            tuple(replacement[column] for column in columns),
        )

    assert connection.execute(
        f"SELECT created_at FROM {table} WHERE {id_column} = ?", (identity,)
    ).fetchone()[0] == "2026-08-31T00:00:00.000000Z"
    connection.close()


def test_verify_requires_recursive_triggers_for_replace_protection(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    assert connection.execute("PRAGMA recursive_triggers").fetchone()[0] == 1
    connection.execute("PRAGMA recursive_triggers=OFF")

    with pytest.raises(_ledger().VerificationError):
        _ledger().verify_ledger(connection)
    connection.close()


def test_proposal_rolls_back_record_event_evidence_and_projection(tmp_path: Path) -> None:
    connection = _connect(tmp_path)

    def deny_evidence_insert(action, table, _column, _database, _trigger):
        if action == sqlite3.SQLITE_INSERT and table == "memory_evidence":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_evidence_insert)
    with pytest.raises(sqlite3.DatabaseError):
        _propose(connection)
    connection.set_authorizer(None)

    for table in ("memory_records", "memory_events", "memory_evidence", "memory_current"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    connection.close()


def _act(
    connection: sqlite3.Connection,
    record_id: str,
    action: str,
    key: str,
    replacement_record_id: str | None = None,
):
    return _ledger().apply_action(
        connection,
        record_id=record_id,
        action=action,
        idempotency_key=key,
        replacement_record_id=replacement_record_id,
    )


def test_all_six_actions_follow_the_frozen_transition_table(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    source = _propose(connection)
    replacement = _propose(
        connection, topic_key="storage.replacement", idempotency_key="proposal-2"
    )

    activated = _act(connection, source.record_id, "activate", "activate-1")
    closed = _act(connection, source.record_id, "close", "close-1")
    reopened = _act(connection, source.record_id, "reopen", "reopen-1")
    superseded = _act(
        connection,
        source.record_id,
        "supersede",
        "supersede-1",
        replacement.record_id,
    )

    assert (source.state, activated.state, closed.state, reopened.state, superseded.state) == (
        "proposed",
        "active",
        "closed",
        "active",
        "superseded",
    )
    proposed_invalid = _propose(
        connection, topic_key="invalid.proposed", idempotency_key="proposal-3"
    )
    active_invalid = _propose(
        connection, topic_key="invalid.active", idempotency_key="proposal-4"
    )
    _act(connection, active_invalid.record_id, "activate", "activate-2")
    assert _act(
        connection, proposed_invalid.record_id, "invalidate", "invalidate-1"
    ).state == "invalidated"
    assert _act(
        connection, active_invalid.record_id, "invalidate", "invalidate-2"
    ).state == "invalidated"
    assert [
        row[0]
        for row in connection.execute(
            "SELECT action FROM memory_events WHERE record_id = ? ORDER BY event_seq",
            (source.record_id,),
        )
    ] == ["propose", "activate", "close", "reopen", "supersede"]
    connection.close()


def test_exact_action_retry_survives_later_state_changes(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    first = _act(connection, record.record_id, "activate", "activate-stable")
    _act(connection, record.record_id, "close", "close-after")

    replayed = _act(connection, record.record_id, "activate", "activate-stable")

    assert replayed == first
    assert replayed.state == "active"
    assert _ledger().show_record(connection, record.record_id).state == "closed"
    assert connection.execute(
        "SELECT COUNT(*) FROM memory_events WHERE record_id = ?", (record.record_id,)
    ).fetchone()[0] == 3
    connection.close()


@pytest.mark.parametrize(
    ("setup", "action"),
    [
        ((), "close"),
        ((), "reopen"),
        (("activate",), "activate"),
        (("activate", "close"), "invalidate"),
        (("invalidate",), "activate"),
    ],
)
def test_invalid_transition_appends_no_partial_event(
    tmp_path: Path, setup: tuple[str, ...], action: str
) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    for index, prior in enumerate(setup):
        _act(connection, record.record_id, prior, f"setup-{index}")
    before = connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]

    with pytest.raises(_ledger().InvalidTransition):
        _act(connection, record.record_id, action, "rejected-action")

    assert connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == before
    connection.close()


def test_action_idempotency_conflict_does_not_append(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    first = _propose(connection)
    second = _propose(connection, topic_key="second", idempotency_key="proposal-2")
    _act(connection, first.record_id, "activate", "shared-action")

    with pytest.raises(_ledger().IdempotencyConflict):
        _act(connection, second.record_id, "activate", "shared-action")

    assert connection.execute(
        "SELECT COUNT(*) FROM memory_events WHERE idempotency_key = 'shared-action'"
    ).fetchone()[0] == 1
    connection.close()


def test_action_owned_key_rejected_by_proposal_without_partial_record(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    _act(connection, record.record_id, "activate", "action-owned")
    before = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("memory_records", "memory_events", "memory_evidence", "memory_current")
    }

    with pytest.raises(_ledger().IdempotencyConflict):
        _propose(
            connection,
            topic_key="topic.second",
            idempotency_key="action-owned",
        )

    assert {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in before
    } == before
    connection.close()


def test_proposal_owned_key_rejected_by_action_without_partial_event(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection, idempotency_key="proposal-owned")

    with pytest.raises(_ledger().IdempotencyConflict):
        _act(connection, record.record_id, "activate", "proposal-owned")

    assert connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1
    assert _ledger().show_record(connection, record.record_id).state == "proposed"
    connection.close()


def test_supersede_rejects_self_and_missing_replacement_without_partial_rows(
    tmp_path: Path,
) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    _act(connection, record.record_id, "activate", "activate-1")
    before = connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]

    with pytest.raises(_ledger().InvalidTransition):
        _act(connection, record.record_id, "supersede", "self", record.record_id)
    with pytest.raises(_ledger().RecordNotFound):
        _act(connection, record.record_id, "supersede", "missing", "mem_missing")

    assert connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == before
    assert connection.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0] == 0
    connection.close()


def test_supersede_accepts_nonactive_replacement_and_hashes_relation(
    tmp_path: Path,
) -> None:
    connection = _connect(tmp_path)
    source = _propose(connection)
    target = _propose(connection, topic_key="target", idempotency_key="proposal-target")
    _act(connection, source.record_id, "activate", "activate-source")

    event = _act(
        connection,
        source.record_id,
        "supersede",
        "supersede-source",
        target.record_id,
    )

    relation = connection.execute("SELECT * FROM memory_relations").fetchone()
    preimage = (
        '{"event_id":"'
        + event.event_id
        + '","relation_type":"superseded_by","source_record_id":"'
        + source.record_id
        + '","target_record_id":"'
        + target.record_id
        + '"}'
    ).encode("utf-8")
    assert relation["relation_sha256"] == hashlib.sha256(preimage).hexdigest()
    assert relation["source_record_id"] == source.record_id
    assert relation["target_record_id"] == target.record_id
    assert _ledger().show_record(connection, target.record_id).state == "proposed"
    connection.close()


def test_show_and_list_are_deterministic_filtered_and_bounded(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    first = _propose(connection, topic_key="topic.one", idempotency_key="proposal-1")
    second = _propose(
        connection,
        scope_kind="workspace",
        scope_key="workspace-a",
        topic_key="topic.two",
        idempotency_key="proposal-2",
    )
    third = _propose(connection, topic_key="topic.three", idempotency_key="proposal-3")
    _act(connection, first.record_id, "activate", "activate-1")

    assert _ledger().show_record(connection, first.record_id).state == "active"
    assert [item.record_id for item in _ledger().list_records(connection, limit=2)] == [
        first.record_id,
        second.record_id,
    ]
    assert [
        item.record_id
        for item in _ledger().list_records(
            connection, scope_kind="project", state="proposed", limit=100
        )
    ] == [third.record_id]
    assert [
        item.record_id
        for item in _ledger().list_records(connection, topic_key="topic.two", limit=100)
    ] == [second.record_id]
    with pytest.raises(_ledger().ValidationError):
        _ledger().list_records(connection, limit=0)
    with pytest.raises(_ledger().ValidationError):
        _ledger().list_records(connection, limit=101)
    with pytest.raises(_ledger().RecordNotFound):
        _ledger().show_record(connection, "mem_missing")
    connection.close()


def test_verify_accepts_valid_hashes_schema_triggers_and_projection(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    _act(connection, record.record_id, "activate", "activate-1")

    result = _ledger().verify_ledger(connection)

    assert result.schema_version == 3
    assert (result.records, result.events, result.evidence, result.relations, result.current) == (
        1,
        2,
        1,
        0,
        1,
    )
    connection.close()


def test_candidate_schema_pragmas_freeze_all_tables_indexes_keys_and_triggers(
    tmp_path: Path,
) -> None:
    connection = _connect(tmp_path)

    for table, expected_columns in CANDIDATE_COLUMN_SHAPES.items():
        assert tuple(
            (
                row["name"],
                row["type"].upper(),
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in connection.execute(f"PRAGMA table_info({table})")
        ) == expected_columns
        indexes = connection.execute(f"PRAGMA index_list({table})").fetchall()
        assert frozenset(
            tuple(
                item["name"]
                for item in connection.execute(f"PRAGMA index_info({row['name']})")
            )
            for row in indexes
            if row["unique"] == 1
        ) == CANDIDATE_UNIQUE_COLUMNS[table]
        assert frozenset(
            (
                row["from"],
                row["table"],
                row["to"],
                row["on_update"],
                row["on_delete"],
                row["match"],
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        ) == CANDIDATE_FOREIGN_KEYS[table]

    for index, (owner, expected_columns) in CANDIDATE_INDEXES.items():
        stored = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        assert stored["tbl_name"] == owner
        definition = next(
            row
            for row in connection.execute(f"PRAGMA index_list({owner})")
            if row["name"] == index
        )
        assert (
            definition["unique"],
            definition["origin"],
            definition["partial"],
        ) == (0, "c", 0)
        assert tuple(
            row["name"] for row in connection.execute(f"PRAGMA index_info({index})")
        ) == expected_columns

    candidate_trigger_names = {
        f"{table}_reject_{operation}"
        for table in (
            "memory_candidates",
            "memory_candidate_evidence",
            "memory_candidate_events",
        )
        for operation in ("update", "delete")
    }
    assert candidate_trigger_names <= _object_names(connection, "trigger")
    assert _ledger().verify_ledger(connection).candidate_current == 0
    connection.close()


@pytest.mark.parametrize(
    ("table", "before", "after"),
    [
        (
            "memory_candidates",
            "idempotency_key TEXT NOT NULL UNIQUE",
            "idempotency_key TEXT NOT NULL",
        ),
        (
            "memory_candidate_evidence",
            "ordinal INTEGER NOT NULL CHECK (ordinal >= 0)",
            "ordinal INTEGER NOT NULL",
        ),
        (
            "memory_candidate_events",
            "candidate_id TEXT NOT NULL REFERENCES memory_candidates(candidate_id)",
            "candidate_id TEXT NOT NULL",
        ),
        (
            "memory_candidate_current",
            "permanent_record_id TEXT REFERENCES memory_records(record_id)",
            "permanent_record_id TEXT",
        ),
    ],
)
def test_verify_rejects_each_weakened_candidate_table_shape(
    tmp_path: Path, table: str, before: str, after: str
) -> None:
    connection = _connect(tmp_path)
    _rewrite_empty_table(connection, table, before, after)

    with pytest.raises(_ledger().VerificationError, match="candidate|table|schema|key"):
        _ledger().verify_ledger(connection)
    connection.close()


@pytest.mark.parametrize(
    "damage", ("columns", "unique", "partial", "owner", "collation_direction")
)
def test_verify_rejects_candidate_index_shape_metadata_and_owner(
    tmp_path: Path, damage: str
) -> None:
    connection = _connect(tmp_path)
    name = "memory_candidates_session_seq"
    connection.execute(f"DROP INDEX {name}")
    if damage == "columns":
        connection.execute(f"CREATE INDEX {name} ON memory_candidates(topic_key)")
    elif damage == "unique":
        connection.execute(
            f"CREATE UNIQUE INDEX {name} ON memory_candidates(session_id, candidate_seq)"
        )
    elif damage == "partial":
        connection.execute(
            f"CREATE INDEX {name} ON memory_candidates(session_id, candidate_seq) "
            "WHERE session_id != ''"
        )
    elif damage == "owner":
        connection.execute("CREATE TABLE candidate_decoy (session_id TEXT, candidate_seq INTEGER)")
        connection.execute(
            f"CREATE INDEX {name} ON candidate_decoy(session_id, candidate_seq)"
        )
    else:
        connection.execute(
            f"CREATE INDEX {name} ON memory_candidates("
            "session_id COLLATE NOCASE, candidate_seq DESC)"
        )

    with pytest.raises(
        _ledger().VerificationError, match="candidate|index|schema|unique"
    ):
        _ledger().verify_ledger(connection)
    connection.close()


def test_verify_rejects_weakened_candidate_append_only_trigger(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    connection.execute("DROP TRIGGER memory_candidate_events_reject_update")
    connection.execute(
        """
        CREATE TRIGGER memory_candidate_events_reject_update
        BEFORE UPDATE ON memory_candidate_events
        WHEN OLD.event_id != NEW.event_id
        BEGIN
            SELECT RAISE(ABORT, 'append-only table');
        END
        """
    )

    with pytest.raises(_ledger().VerificationError, match="candidate|trigger|schema"):
        _ledger().verify_ledger(connection)
    connection.close()


@pytest.mark.parametrize(
    "damage", ["hash", "canonical", "index", "index_shape", "trigger", "projection"]
)
def test_verify_detects_hash_schema_trigger_and_projection_damage(
    tmp_path: Path, damage: str
) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    if damage in {"hash", "canonical"}:
        connection.execute("DROP TRIGGER memory_records_reject_update")
        if damage == "hash":
            connection.execute(
                "UPDATE memory_records SET content_sha256 = ? WHERE record_id = ?",
                ("0" * 64, record.record_id),
            )
        else:
            connection.execute(
                "UPDATE memory_records SET value_json = ? WHERE record_id = ?",
                ('{ "name": "sqlite3" }', record.record_id),
            )
    elif damage == "index":
        connection.execute("DROP INDEX memory_events_record_event_seq")
    elif damage == "index_shape":
        connection.execute("DROP INDEX memory_events_record_event_seq")
        connection.execute(
            "CREATE INDEX memory_events_record_event_seq ON memory_events(action)"
        )
    elif damage == "trigger":
        connection.execute("DROP TRIGGER memory_evidence_reject_delete")
    else:
        connection.execute("DELETE FROM memory_current")

    with pytest.raises(_ledger().VerificationError):
        _ledger().verify_ledger(connection)
    connection.close()


def test_verify_rejects_missing_insert_conflict_guard(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    connection.execute("DROP TRIGGER memory_candidate_events_guard_insert_conflict")

    with pytest.raises(_ledger().VerificationError, match="trigger|schema|guard"):
        _ledger().verify_ledger(connection)
    connection.close()


@pytest.mark.parametrize(
    "damage",
    ("uppercase_topic", "blank_statement", "oversize_scope", "oversize_value"),
)
def test_verify_rejects_invalid_record_contract_with_recomputed_content_hash(
    tmp_path: Path, damage: str
) -> None:
    connection = _connect(tmp_path)
    result = _propose(connection)
    record = dict(
        connection.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (result.record_id,)
        ).fetchone()
    )
    if damage == "uppercase_topic":
        record["topic_key"] = "UPPER"
    elif damage == "blank_statement":
        record["statement"] = " \t"
    elif damage == "oversize_scope":
        record["scope_key"] = "s" * 257
    else:
        record["value_json"] = _ledger().canonical_json("v" * 32768)
    value = _ledger().parse_json_value(record["value_json"])
    content_sha256 = import_module("codex_memory.ledger_common")._hash_preimage(
        {
            "record_type": record["record_type"],
            "scope": {"kind": record["scope_kind"], "key": record["scope_key"]},
            "statement": record["statement"],
            "topic_key": record["topic_key"],
            "value": value,
        }
    )
    connection.execute("DROP TRIGGER memory_records_reject_update")
    connection.execute(
        """
        UPDATE memory_records
        SET scope_key = ?, topic_key = ?, statement = ?, value_json = ?,
            content_sha256 = ?
        WHERE record_id = ?
        """,
        (
            record["scope_key"],
            record["topic_key"],
            record["statement"],
            record["value_json"],
            content_sha256,
            result.record_id,
        ),
    )
    connection.execute(
        import_module("codex_memory.database").append_only_trigger_statement(
            "memory_records", "update"
        )
    )

    with pytest.raises(_ledger().VerificationError, match="record content") as captured:
        _ledger().verify_ledger(connection)

    assert isinstance(captured.value.__cause__, _ledger().ValidationError)
    connection.close()


def test_verify_rejects_invalid_evidence_uri_with_recomputed_hash(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    result = _propose(connection)
    invalid_uri = "not a URI"
    evidence_sha256 = import_module("codex_memory.ledger_common")._hash_preimage(
        {"record_id": result.record_id, "uri": invalid_uri}
    )
    connection.execute("DROP TRIGGER memory_evidence_reject_update")
    connection.execute(
        """
        UPDATE memory_evidence
        SET uri = ?, evidence_sha256 = ?
        WHERE record_id = ?
        """,
        (invalid_uri, evidence_sha256, result.record_id),
    )
    connection.execute(
        import_module("codex_memory.database").append_only_trigger_statement(
            "memory_evidence", "update"
        )
    )

    with pytest.raises(_ledger().VerificationError, match="evidence URI") as captured:
        _ledger().verify_ledger(connection)

    assert isinstance(captured.value.__cause__, _ledger().ValidationError)
    connection.close()


def test_verify_detects_relation_event_disagreement(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    source = _propose(connection)
    target = _propose(connection, topic_key="target", idempotency_key="proposal-target")
    _act(connection, source.record_id, "activate", "activate-source")
    _act(connection, source.record_id, "supersede", "supersede-source", target.record_id)
    connection.execute("DROP TRIGGER memory_relations_reject_update")
    connection.execute(
        "UPDATE memory_relations SET event_id = "
        "(SELECT event_id FROM memory_events WHERE action = 'activate')"
    )

    with pytest.raises(_ledger().VerificationError):
        _ledger().verify_ledger(connection)
    connection.close()


def _rewrite_empty_table(
    connection: sqlite3.Connection, table: str, before: str, after: str
) -> None:
    original = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()[0]
    assert before in original
    replacement = original.replace(before, after, 1)
    artifacts = [
        row[0]
        for row in connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE tbl_name = ? AND type IN ('index', 'trigger') AND sql IS NOT NULL",
            (table,),
        )
    ]
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute(f"DROP TABLE {table}")
    connection.execute(replacement)
    for statement in artifacts:
        connection.execute(statement)
    connection.execute("PRAGMA foreign_keys=ON")


@pytest.mark.parametrize(
    ("category", "table", "before", "after"),
    [
        (
            "primary_key",
            "memory_current",
            "record_id TEXT PRIMARY KEY REFERENCES memory_records(record_id)",
            "record_id TEXT REFERENCES memory_records(record_id)",
        ),
        (
            "not_null",
            "memory_current",
            "state TEXT NOT NULL CHECK",
            "state TEXT CHECK",
        ),
        (
            "unique",
            "memory_relations",
            "relation_id TEXT NOT NULL UNIQUE",
            "relation_id TEXT NOT NULL",
        ),
        (
            "check",
            "memory_current",
            "state IN ('proposed', 'active', 'superseded', 'invalidated', 'closed')",
            "state IN ('proposed', 'active', 'superseded', 'invalidated', 'closed', 'deleted')",
        ),
        (
            "foreign_key",
            "memory_current",
            "last_event_id TEXT NOT NULL REFERENCES memory_events(event_id)",
            "last_event_id TEXT NOT NULL",
        ),
    ],
)
def test_verify_rejects_weakened_table_constraints(
    tmp_path: Path,
    category: str,
    table: str,
    before: str,
    after: str,
) -> None:
    connection = _connect(tmp_path)
    _rewrite_empty_table(connection, table, before, after)

    with pytest.raises(_ledger().VerificationError, match="schema|table|constraint|key"):
        _ledger().verify_ledger(connection)
    connection.close()


def test_verify_rejects_required_index_with_wrong_uniqueness(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    connection.execute("DROP INDEX memory_events_record_event_seq")
    connection.execute(
        "CREATE UNIQUE INDEX memory_events_record_event_seq "
        "ON memory_events(record_id, event_seq)"
    )

    with pytest.raises(_ledger().VerificationError, match="index|unique"):
        _ledger().verify_ledger(connection)
    connection.close()


def test_verify_rejects_trigger_that_only_blocks_some_updates(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    connection.execute("DROP TRIGGER memory_records_reject_update")
    connection.execute(
        """
        CREATE TRIGGER memory_records_reject_update
        BEFORE UPDATE ON memory_records
        WHEN OLD.record_id != NEW.record_id
        BEGIN
            SELECT RAISE(ABORT, 'append-only table');
        END
        """
    )

    with pytest.raises(_ledger().VerificationError, match="trigger"):
        _ledger().verify_ledger(connection)
    connection.close()


def test_verify_preserves_quoted_check_literal_case(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    _rewrite_empty_table(connection, "memory_records", "'decision'", "'DECISION'")

    with pytest.raises(_ledger().VerificationError, match="constraint"):
        _ledger().verify_ledger(connection)
    connection.close()


def test_verify_binds_required_index_to_expected_owner_table(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    connection.execute("DROP INDEX memory_events_record_event_seq")
    connection.execute("CREATE TABLE decoy_events (record_id TEXT, event_seq INTEGER)")
    connection.execute(
        "CREATE INDEX memory_events_record_event_seq "
        "ON decoy_events(record_id, event_seq)"
    )

    with pytest.raises(_ledger().VerificationError, match="index"):
        _ledger().verify_ledger(connection)
    connection.close()


def test_rebuild_restores_deleted_projection_without_changing_sources(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    first = _propose(connection)
    second = _propose(connection, topic_key="two", idempotency_key="proposal-2")
    _act(connection, first.record_id, "activate", "activate-1")
    expected = [dict(row) for row in connection.execute("SELECT * FROM memory_current ORDER BY record_id")]
    source_counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("memory_records", "memory_events", "memory_evidence", "memory_relations")
    }
    connection.execute("DELETE FROM memory_current")

    rebuilt = _ledger().rebuild_projection(connection)

    assert rebuilt == 2
    assert [dict(row) for row in connection.execute("SELECT * FROM memory_current ORDER BY record_id")] == expected
    assert {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in source_counts
    } == source_counts
    assert _ledger().show_record(connection, second.record_id).state == "proposed"
    connection.close()


def test_rebuild_failure_rolls_back_to_prior_projection(tmp_path: Path) -> None:
    connection = _connect(tmp_path)
    record = _propose(connection)
    before = [dict(row) for row in connection.execute("SELECT * FROM memory_current")]
    connection.execute("DROP TRIGGER memory_events_reject_update")
    connection.execute(
        "UPDATE memory_events SET event_sha256 = ? WHERE record_id = ?",
        ("0" * 64, record.record_id),
    )

    with pytest.raises(_ledger().VerificationError):
        _ledger().rebuild_projection(connection)

    assert [dict(row) for row in connection.execute("SELECT * FROM memory_current")] == before
    connection.close()


def test_verify_database_uses_existing_read_only_version_two_file(tmp_path: Path) -> None:
    path = RuntimePaths.for_repo(tmp_path).db_path
    connection = _connect(tmp_path)
    _propose(connection)
    connection.close()

    result = _ledger().verify_database(path)

    assert result.records == 1
    readonly = _ledger().connect_readonly_database(path)
    assert readonly.execute("PRAGMA query_only").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError):
        readonly.execute("DELETE FROM memory_current")
    readonly.close()


def test_verification_uses_one_snapshot_when_writer_commits_after_first_read(
    tmp_path: Path,
) -> None:
    database = import_module("codex_memory.database")
    path = RuntimePaths.for_repo(tmp_path).db_path
    setup = database.connect_database(path)
    database.migrate_database(setup)
    setup.close()
    verifier = _ledger().connect_readonly_database(path)
    writer = database.connect_database(path)
    callback_calls = 0

    def commit_after_first_snapshot_read() -> None:
        nonlocal callback_calls
        callback_calls += 1
        _propose(writer, idempotency_key="writer-proposal")

    result = _ledger().verify_ledger(
        verifier,
        _after_first_snapshot_read=commit_after_first_snapshot_read,
    )

    assert callback_calls == 1
    assert (result.records, result.events, result.current) == (0, 0, 0)
    assert writer.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
    verifier.close()
    writer.close()


def test_verification_candidate_counts_share_snapshot_with_concurrent_writer(
    tmp_path: Path,
) -> None:
    database = import_module("codex_memory.database")
    candidate = import_module("codex_memory.candidate_memory")
    blob_store = import_module("codex_memory.blob_store")
    paths = RuntimePaths.for_repo(tmp_path)
    setup = database.connect_database(paths.db_path)
    database.migrate_database(setup)
    setup.close()
    verifier = _ledger().connect_readonly_database(paths.db_path)
    writer = database.connect_database(paths.db_path)
    callback_calls = 0

    def capture_after_first_snapshot_read() -> None:
        nonlocal callback_calls
        callback_calls += 1
        candidate.capture_candidate(
            writer,
            blob_store.BlobStore(paths.blobs_dir),
            project_scope_key=str(tmp_path),
            record_type="decision",
            scope_kind="project",
            scope_key=str(tmp_path),
            topic_key="snapshot.candidate",
            statement="This candidate committed after the verifier snapshot.",
            session_id="snapshot-session",
            turn_id="snapshot-turn",
            context_id="startup",
            idempotency_key="snapshot-candidate",
        )

    result = _ledger().verify_ledger(
        verifier,
        _after_first_snapshot_read=capture_after_first_snapshot_read,
    )

    assert callback_calls == 1
    assert (
        result.candidates,
        result.candidate_events,
        result.candidate_evidence,
        result.candidate_current,
    ) == (0, 0, 0, 0)
    assert writer.execute("SELECT COUNT(*) FROM memory_candidates").fetchone()[0] == 1
    verifier.close()
    writer.close()


def test_verify_database_never_creates_missing_paths_or_migrates(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "memory.sqlite3"
    with pytest.raises(_ledger().VerificationError):
        _ledger().verify_database(missing)
    assert not missing.parent.exists()

    version_one = tmp_path / "version-one.sqlite3"
    connection = sqlite3.connect(version_one, isolation_level=None)
    connection.row_factory = sqlite3.Row
    _seed_version_one(connection)
    connection.close()
    with pytest.raises(_ledger().VerificationError):
        _ledger().verify_database(version_one)
    with sqlite3.connect(version_one) as check:
        assert check.execute("SELECT value FROM schema_meta").fetchone()[0] == "1"


def test_readonly_connection_closes_when_configuration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_file = tmp_path / "memory.sqlite3"
    database_file.write_bytes(b"placeholder")

    class FailingConnection:
        row_factory = None
        closed = False

        def execute(self, _statement: str) -> None:
            raise sqlite3.OperationalError("forced pragma failure")

        def close(self) -> None:
            self.closed = True

    connection = FailingConnection()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connection)

    with pytest.raises(_ledger().VerificationError):
        _ledger().connect_readonly_database(database_file)

    assert connection.closed is True


def test_concurrent_equivalent_proposals_create_one_logical_result(tmp_path: Path) -> None:
    database = import_module("codex_memory.database")
    path = RuntimePaths.for_repo(tmp_path).db_path
    setup = database.connect_database(path)
    database.migrate_database(setup)
    setup.close()
    barrier = Barrier(3)
    results: Queue[object] = Queue()

    def propose_from_independent_connection() -> None:
        connection = database.connect_database(path)
        try:
            barrier.wait()
            results.put(_propose(connection))
        except BaseException as exc:
            results.put(exc)
        finally:
            connection.close()

    threads = [
        Thread(target=propose_from_independent_connection),
        Thread(target=propose_from_independent_connection),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    outcomes = [results.get_nowait(), results.get_nowait()]
    assert not [item for item in outcomes if isinstance(item, BaseException)]
    assert outcomes[0].record_id == outcomes[1].record_id
    check = database.connect_database(path)
    assert check.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
    assert check.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1
    assert check.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 1
    check.close()


def test_transaction_helpers_require_and_preserve_caller_transaction(
    tmp_path: Path,
) -> None:
    connection = _connect(tmp_path)
    ledger = _ledger()
    assert hasattr(ledger, "propose_record_in_transaction")
    assert hasattr(ledger, "apply_action_in_transaction")

    proposal = {
        "record_type": "decision",
        "scope_kind": "project",
        "scope_key": "repo",
        "topic_key": "transaction.owner",
        "statement": "The caller owns this transaction.",
        "idempotency_key": "transaction-owner-propose",
        "evidence_uris": ("file:///transaction-owner",),
    }
    with pytest.raises(ledger.LedgerError):
        ledger.propose_record_in_transaction(connection, **proposal)
    with pytest.raises(ledger.LedgerError):
        ledger.apply_action_in_transaction(
            connection,
            record_id="mem_missing",
            action="activate",
            idempotency_key="transaction-owner-activate",
        )

    connection.execute("BEGIN IMMEDIATE")
    traced: list[str] = []
    connection.set_trace_callback(traced.append)
    try:
        record = ledger.propose_record_in_transaction(connection, **proposal)
        retry = ledger.propose_record_in_transaction(connection, **proposal)
        activated = ledger.apply_action_in_transaction(
            connection,
            record_id=record.record_id,
            action="activate",
            idempotency_key="transaction-owner-activate",
        )
        action_retry = ledger.apply_action_in_transaction(
            connection,
            record_id=record.record_id,
            action="activate",
            idempotency_key="transaction-owner-activate",
        )
        assert connection.in_transaction is True
        assert retry == record
        assert action_retry == activated
        assert not any(
            statement.strip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
            for statement in traced
        )
    finally:
        connection.set_trace_callback(None)
        connection.rollback()

    assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0
    connection.close()


def test_transaction_helper_failure_never_rolls_back_callers_work(
    tmp_path: Path,
) -> None:
    connection = _connect(tmp_path)
    ledger = _ledger()
    assert hasattr(ledger, "propose_record_in_transaction")
    assert hasattr(ledger, "apply_action_in_transaction")
    connection.execute("BEGIN IMMEDIATE")
    try:
        record = ledger.propose_record_in_transaction(
            connection,
            record_type="decision",
            scope_kind="project",
            scope_key="repo",
            topic_key="transaction.failure",
            statement="Keep caller work after an inner failure.",
            idempotency_key="transaction-failure-propose",
        )
        ledger.apply_action_in_transaction(
            connection,
            record_id=record.record_id,
            action="activate",
            idempotency_key="transaction-failure-activate",
        )
        with pytest.raises(ledger.InvalidTransition):
            ledger.apply_action_in_transaction(
                connection,
                record_id=record.record_id,
                action="activate",
                idempotency_key="transaction-failure-invalid",
            )
        assert connection.in_transaction is True
        assert connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 2
    finally:
        connection.rollback()
        connection.close()
