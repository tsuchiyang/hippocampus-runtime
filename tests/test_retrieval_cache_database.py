from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codex_memory.retrieval.cache_database import (
    _normal_sql,
    CacheCorruptError,
    CacheUnsupportedError,
    append_generation,
    append_generation_event,
    connect_cache_database,
    deterministic_generation_id,
    inspect_cache_schema_version,
    initialize_schema,
    initialize_schema_v2,
    rebuild_projections,
    verify_cache_database,
    verify_legacy_schema_v1,
)
from codex_memory.retrieval.contracts import IndexHighwater


def _initialize_legacy_schema_v1(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE retrieval_schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE retrieval_generations (
            generation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id TEXT NOT NULL UNIQUE,
            canonical_highwater_json TEXT NOT NULL,
            indexed_highwater_json TEXT NOT NULL,
            canonical_content_fingerprint TEXT NOT NULL,
            provider_fingerprint TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_generation_events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            action TEXT NOT NULL CHECK (action IN ('create', 'validate', 'activate', 'retire', 'fail')),
            payload_json TEXT NOT NULL,
            event_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_generation_current (
            generation_id TEXT PRIMARY KEY REFERENCES retrieval_generations(generation_id),
            state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')),
            last_event_seq INTEGER NOT NULL,
            last_event_id TEXT NOT NULL REFERENCES retrieval_generation_events(event_id),
            validation_state TEXT NOT NULL,
            degraded_reason TEXT,
            failure_metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_generation_id TEXT REFERENCES retrieval_generations(generation_id),
            validation_state TEXT NOT NULL,
            degraded_reason TEXT,
            failure_metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX retrieval_generation_events_generation_seq ON retrieval_generation_events(generation_id, event_seq);
        CREATE INDEX retrieval_generation_current_state_generation ON retrieval_generation_current(state, generation_id);
        CREATE TRIGGER retrieval_generations_reject_update
            BEFORE UPDATE ON retrieval_generations
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generations_reject_delete
            BEFORE DELETE ON retrieval_generations
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generations_guard_insert_conflict
            BEFORE INSERT ON retrieval_generations
            WHEN EXISTS (
                SELECT 1 FROM retrieval_generations
                WHERE (generation_seq = NEW.generation_seq) OR (generation_id = NEW.generation_id)
            )
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generation_events_reject_update
            BEFORE UPDATE ON retrieval_generation_events
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generation_events_reject_delete
            BEFORE DELETE ON retrieval_generation_events
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        CREATE TRIGGER retrieval_generation_events_guard_insert_conflict
            BEFORE INSERT ON retrieval_generation_events
            WHEN EXISTS (
                SELECT 1 FROM retrieval_generation_events
                WHERE (event_seq = NEW.event_seq) OR (event_id = NEW.event_id)
            )
            BEGIN
                SELECT RAISE(ABORT, 'append-only table');
            END;
        INSERT INTO retrieval_schema_meta(key, value) VALUES ('schema_version', '1');
        INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', NULL, '{}',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z');
        """
    )


def _cache(path: Path) -> sqlite3.Connection:
    connection = connect_cache_database(path)
    initialize_schema(connection)
    return connection


def _generation(connection: sqlite3.Connection, record_seq: int = 1) -> str:
    highwater = IndexHighwater(record_seq=record_seq)
    generation_id = deterministic_generation_id(highwater, "a" * 64)
    append_generation(
        connection,
        generation_id=generation_id,
        canonical_highwater=highwater,
        indexed_highwater=highwater,
        canonical_content_fingerprint="a" * 64,
        provider_fingerprint="unconfigured:p2.1",
        model_fingerprint="unconfigured:p2.1",
        created_at="2026-01-02T03:04:05.000000Z",
    )
    return generation_id


def test_legacy_schema_v1_is_exact_compatibility_fixture(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.sqlite3"
    connection = connect_cache_database(path)
    _initialize_legacy_schema_v1(connection)
    try:
        assert inspect_cache_schema_version(connection) == 1
        verify_legacy_schema_v1(connection)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if row[0] != "sqlite_sequence"
            and (row[0] == "retrieval_documents_fts" or not row[0].startswith("retrieval_documents_fts_"))
        }
        assert tables == {
            "retrieval_schema_meta",
            "retrieval_generations",
            "retrieval_generation_events",
            "retrieval_generation_current",
            "retrieval_state",
        }
        assert [row[1] for row in connection.execute("PRAGMA table_info(retrieval_schema_meta)")] == [
            "key",
            "value",
        ]
        assert [row[1] for row in connection.execute("PRAGMA table_info(retrieval_generations)")] == [
            "generation_seq", "generation_id", "canonical_highwater_json", "indexed_highwater_json",
            "canonical_content_fingerprint", "provider_fingerprint", "model_fingerprint", "created_at",
        ]
        assert [row[1] for row in connection.execute("PRAGMA table_info(retrieval_generation_events)")] == [
            "event_seq", "event_id", "generation_id", "action", "payload_json", "event_sha256", "created_at",
        ]
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(retrieval_generation_events)")}
        assert "retrieval_generation_events_generation_seq" in indexes
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(retrieval_generation_current)")}
        assert "retrieval_generation_current_state_generation" in indexes
        foreign_keys = {(row[3], row[2], row[4]) for row in connection.execute("PRAGMA foreign_key_list(retrieval_generation_events)")}
        assert foreign_keys == {("generation_id", "retrieval_generations", "generation_id")}
        triggers = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
        assert {
            "retrieval_generations_reject_update", "retrieval_generations_reject_delete",
            "retrieval_generations_guard_insert_conflict", "retrieval_generation_events_reject_update",
            "retrieval_generation_events_reject_delete", "retrieval_generation_events_guard_insert_conflict",
        } <= triggers
        with pytest.raises(CacheUnsupportedError):
            verify_cache_database(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("fts5_available", [False, True])
def test_schema_v2_has_exact_tables_and_optional_fts_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fts5_available: bool
) -> None:
    path = tmp_path / f"retrieval-v2-{fts5_available}.sqlite3"
    connection = connect_cache_database(path)
    monkeypatch.setattr(
        "codex_memory.retrieval.cache_database._supports_fts5",
        lambda _connection: fts5_available,
        raising=False,
    )
    initialize_schema_v2(connection)
    try:
        assert inspect_cache_schema_version(connection) == 2
        assert verify_cache_database(connection) is None
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if row[0] != "sqlite_sequence"
            and (row[0] == "retrieval_documents_fts" or not row[0].startswith("retrieval_documents_fts_"))
        }
        base_tables = {
            "retrieval_schema_meta",
            "retrieval_generations",
            "retrieval_generation_events",
            "retrieval_generation_current",
            "retrieval_state",
            "retrieval_generation_capabilities",
            "retrieval_documents",
            "retrieval_document_sources",
            "retrieval_symbols",
            "retrieval_runs",
            "retrieval_candidate_traces",
        }
        if fts5_available:
            assert tables == base_tables | {"retrieval_documents_fts"}
        else:
            assert tables == base_tables
        assert connection.execute(
            "SELECT value FROM retrieval_schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "2"
        assert connection.execute(
            "SELECT COUNT(*) FROM retrieval_generation_capabilities"
        ).fetchone()[0] == 0
        assert [row[1] for row in connection.execute("PRAGMA table_info(retrieval_documents)")] == [
            "document_seq",
            "document_id",
            "generation_id",
            "record_id",
            "record_seq",
            "record_type",
            "scope_kind",
            "scope_key",
            "topic_key",
            "statement",
            "value_json",
            "search_text",
            "state",
            "created_at",
            "last_memory_event_seq",
            "evidence_count",
            "content_sha256",
            "document_sha256",
        ]
        if fts5_available:
            assert [row[1] for row in connection.execute("PRAGMA table_info(retrieval_documents_fts)")] == [
                "document_id",
                "generation_id",
                "search_text",
            ]
            assert [row[1] for row in connection.execute("PRAGMA table_info(retrieval_generation_capabilities)")] == [
                "generation_id",
                "fts5_state",
                "sqlite_version",
                "index_policy_fingerprint",
                "projection_manifest_sha256",
                "projected_document_count",
                "omitted_unsafe_count",
                "created_at",
            ]
        else:
            assert not connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
            ).fetchone()
    finally:
        connection.close()


def test_initialization_is_idempotent_and_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.sqlite3"
    connection = _cache(path)
    try:
        initialize_schema(connection)
        assert connection.execute("SELECT value FROM retrieval_schema_meta WHERE key = 'schema_version'").fetchone()[0] == "4"
    finally:
        connection.close()
    with sqlite3.connect(tmp_path / "future.sqlite3") as future:
        future.execute("CREATE TABLE retrieval_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        future.execute("INSERT INTO retrieval_schema_meta VALUES ('schema_version', '99')")
    future = connect_cache_database(tmp_path / "future.sqlite3")
    with pytest.raises(CacheUnsupportedError, match="unsupported"):
        initialize_schema(future)
    future.close()


def test_schema_v2_refuses_to_overwrite_v1_or_unknown_versions(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.sqlite3"
    legacy = connect_cache_database(legacy_path)
    _initialize_legacy_schema_v1(legacy)
    legacy.close()
    future_path = tmp_path / "future.sqlite3"
    with sqlite3.connect(future_path) as future:
        future.execute("CREATE TABLE retrieval_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        future.execute("INSERT INTO retrieval_schema_meta VALUES ('schema_version', '99')")

    with pytest.raises(CacheUnsupportedError):
        initialize_schema_v2(connect_cache_database(legacy_path))
    with pytest.raises(CacheUnsupportedError):
        initialize_schema_v2(connect_cache_database(future_path))


def test_generation_and_event_history_are_append_only_even_for_replace(tmp_path: Path) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        generation_id = _generation(connection)
        append_generation_event(connection, generation_id=generation_id, action="create", created_at="2026-01-02T03:04:05.000000Z")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE retrieval_generations SET created_at = 'x'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM retrieval_generation_events")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("INSERT OR REPLACE INTO retrieval_generations SELECT * FROM retrieval_generations")
    finally:
        connection.close()


def test_replay_requires_complete_valid_lifecycle_before_an_active_generation(tmp_path: Path) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        generation_id = _generation(connection)
        append_generation_event(connection, generation_id=generation_id, action="create", created_at="2026-01-02T03:04:05.000000Z")
        append_generation_event(connection, generation_id=generation_id, action="validate", created_at="2026-01-02T03:04:05.000000Z")
        append_generation_event(connection, generation_id=generation_id, action="activate", created_at="2026-01-02T03:04:05.000000Z")
        rebuild_projections(connection)
        assert connection.execute("SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1").fetchone()[0] == generation_id
        assert connection.execute("SELECT state FROM retrieval_generation_current").fetchone()[0] == "active"
    finally:
        connection.close()


@pytest.mark.parametrize("events", [("create", "activate"), ("create", "fail", "activate")])
def test_replay_rejects_incomplete_or_failed_generation_activation(tmp_path: Path, events: tuple[str, ...]) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        generation_id = _generation(connection)
        for action in events:
            append_generation_event(connection, generation_id=generation_id, action=action, created_at="2026-01-02T03:04:05.000000Z")
        with pytest.raises(CacheCorruptError, match="lifecycle"):
            rebuild_projections(connection)
    finally:
        connection.close()


def test_safe_active_switch_retires_the_old_generation_before_new_activation(tmp_path: Path) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        first = _generation(connection, 1)
        second = _generation(connection, 2)
        for action, generation in (("create", first), ("validate", first), ("activate", first), ("retire", first), ("create", second), ("validate", second), ("activate", second)):
            append_generation_event(connection, generation_id=generation, action=action, created_at="2026-01-02T03:04:05.000000Z")
        rebuild_projections(connection)
        assert dict(connection.execute("SELECT generation_id, state FROM retrieval_generation_current")) == {first: "retired", second: "active"}
        verify_cache_database(connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mutate",
    [
        "extra_index",
        "no_op_trigger",
        "missing_schema_version",
    ],
)
def test_verification_rejects_schema_objects_that_only_have_the_expected_names(
    tmp_path: Path, mutate: str
) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        if mutate == "extra_index":
            connection.execute("CREATE INDEX unexpected_generation_index ON retrieval_generations(created_at)")
        elif mutate == "no_op_trigger":
            connection.execute("DROP TRIGGER retrieval_generations_reject_update")
            connection.execute("""CREATE TRIGGER retrieval_generations_reject_update
                BEFORE UPDATE ON retrieval_generations BEGIN SELECT 1; END""")
        else:
            connection.execute("DELETE FROM retrieval_schema_meta WHERE key = 'schema_version'")
        with pytest.raises(CacheCorruptError):
            verify_cache_database(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("mutation", ["arbitrary_id", "different_indexed_highwater", "provider", "no_events"])
def test_verification_rejects_generation_metadata_that_cannot_be_rederived(
    tmp_path: Path, mutation: str
) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        generation_id = "gen_not_a_deterministic_generation" if mutation == "arbitrary_id" else "gen_00000000000000000000000000000000"
        append_generation(
            connection,
            generation_id=generation_id,
            canonical_highwater=IndexHighwater(record_seq=1),
            indexed_highwater=IndexHighwater(record_seq=2) if mutation == "different_indexed_highwater" else IndexHighwater(record_seq=1),
            canonical_content_fingerprint="a" * 64,
            provider_fingerprint="other-provider" if mutation == "provider" else "unconfigured:p2.1",
            model_fingerprint="unconfigured:p2.1",
            created_at="2026-01-02T03:04:05.000000Z",
        )
        if mutation != "no_events":
            for action in ("create", "validate", "activate"):
                append_generation_event(connection, generation_id=generation_id, action=action, created_at="2026-01-02T03:04:05.000000Z")
        if mutation == "no_events":
            with pytest.raises(CacheCorruptError):
                rebuild_projections(connection)
        else:
            rebuild_projections(connection)
            with pytest.raises(CacheCorruptError):
                verify_cache_database(connection)
    finally:
        connection.close()


def test_event_payload_cannot_override_derived_projection_state(tmp_path: Path) -> None:
    connection = _cache(tmp_path / "retrieval.sqlite3")
    try:
        generation_id = _generation(connection)
        append_generation_event(connection, generation_id=generation_id, action="create", created_at="2026-01-02T03:04:05.000000Z")
        append_generation_event(
            connection,
            generation_id=generation_id,
            action="validate",
            created_at="2026-01-02T03:04:05.000000Z",
            payload={"validation_state": "failed"},
        )
        append_generation_event(connection, generation_id=generation_id, action="activate", created_at="2026-01-02T03:04:05.000000Z")
        with pytest.raises(CacheCorruptError, match="payload"):
            rebuild_projections(connection)
    finally:
        connection.close()


def test_schema_validation_preserves_case_inside_sql_string_literals(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.sqlite3"
    connection = _cache(path)
    connection.close()
    with sqlite3.connect(path) as tamper:
        tamper.execute("PRAGMA writable_schema=ON")
        tamper.execute(
            """UPDATE sqlite_master
            SET sql = replace(sql, '''building''', '''BUILDING''')
            WHERE type = 'table' AND name = 'retrieval_generation_current'"""
        )
        tamper.execute("PRAGMA writable_schema=OFF")
    connection = connect_cache_database(path)
    try:
        with pytest.raises(CacheCorruptError, match="schema"):
            verify_cache_database(connection)
    finally:
        connection.close()


def test_schema_validation_does_not_fold_unicode_long_s_in_unquoted_identifiers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retrieval.sqlite3"
    connection = _cache(path)
    connection.close()
    with sqlite3.connect(path) as tamper:
        tamper.execute("PRAGMA writable_schema=ON")
        tamper.execute(
            """UPDATE sqlite_master
            SET sql = replace(sql, 'NEW.generation_seq', 'NEW.generation_ſeq')
            WHERE type = 'trigger'
              AND name = 'retrieval_generations_guard_insert_conflict'"""
        )
        changed = tamper.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?",
            ("retrieval_generations_guard_insert_conflict",),
        ).fetchone()[0]
        tamper.execute("PRAGMA writable_schema=OFF")
    assert "NEW.generation_ſeq" in changed

    connection = connect_cache_database(path)
    try:
        with pytest.raises(CacheCorruptError, match="schema"):
            verify_cache_database(connection)
    finally:
        connection.close()


def test_schema_validation_rejects_trigger_when_condition_mutation(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.sqlite3"
    connection = _cache(path)
    connection.close()
    with sqlite3.connect(path) as tamper:
        tamper.execute("PRAGMA writable_schema=ON")
        tamper.execute(
            """UPDATE sqlite_master
            SET sql = replace(sql, 'WHEN EXISTS', 'WHEN NOT EXISTS')
            WHERE type = 'trigger'
              AND name = 'retrieval_generations_guard_insert_conflict'"""
        )
        tamper.execute("PRAGMA writable_schema=OFF")

    connection = connect_cache_database(path)
    try:
        with pytest.raises(CacheCorruptError, match="schema"):
            verify_cache_database(connection)
    finally:
        connection.close()


def test_sql_normalization_accepts_keyword_formatting_but_preserves_all_quoted_content() -> None:
    assert _normal_sql("CREATE   TABLE x ( state TEXT CHECK (state IN ('building')) )") == _normal_sql(
        "create table x ( state text check (state in ('building')) )"
    )
    assert _normal_sql("CHECK (value = 'a  b')") != _normal_sql("check (value = 'a b')")
    assert _normal_sql("CHECK (state = 'building')") != _normal_sql("check (state = 'BUILDING')")
    assert _normal_sql('CREATE TABLE "MiX ed" (`VaLue`, [BrAck et])') != _normal_sql(
        'create table "mix ed" (`value`, [brack et])'
    )
    assert _normal_sql("CREATE TABLE x(a TEXT, CHECK(action IN ('create', 'validate')))") == _normal_sql(
        "create table x ( a text , check ( action in ( 'create' , 'validate' ) ) )"
    )
    assert _normal_sql("CREATE TABLE x(a.b TEXT CHECK(a.b!=1 AND a.b>=0))") == _normal_sql(
        "create table x ( a . b text check ( a . b != 1 and a . b >= 0 ) )"
    )
    assert _normal_sql("CHECK(value != 1)") != _normal_sql("check(value ! = 1)")
    assert _normal_sql("CHECK(value >= 1)") != _normal_sql("check(value > = 1)")


@pytest.mark.parametrize("separator", ["\u00a0", "\u2003", "\u2007", "\u202f", "\u3000"])
def test_sql_normalization_does_not_treat_unicode_separator_as_sqlite_whitespace(
    separator: str,
) -> None:
    assert _normal_sql(
        f"SELECT RAISE{separator}(ABORT, 'append-only table')"
    ) != _normal_sql("SELECT RAISE(ABORT, 'append-only table')")


@pytest.mark.parametrize("separator", ["\t", "\n", "\f", "\r", " "])
def test_sql_normalization_accepts_only_sqlite_tokenizer_whitespace(
    separator: str,
) -> None:
    assert _normal_sql(f"CREATE{separator}TABLE x(a TEXT)") == _normal_sql(
        "CREATE TABLE x(a TEXT)"
    )


@pytest.mark.parametrize(
    "quoted",
    ["'a\u00a0b'", '"a\u00a0b"', "`a\u00a0b`", "[a\u00a0b]"],
)
def test_sql_normalization_preserves_unicode_separator_inside_all_quote_forms(
    quoted: str,
) -> None:
    normalized = _normal_sql(f"CHECK (value = {quoted})")
    without_separator = _normal_sql(f"CHECK (value = {quoted.replace(chr(0x00A0), '')})")

    assert "\u00a0" in normalized
    assert normalized != without_separator


def test_schema_validation_rejects_unicode_separator_in_trigger_body(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.sqlite3"
    connection = _cache(path)
    connection.close()
    with sqlite3.connect(path) as tamper:
        tamper.execute("PRAGMA writable_schema=ON")
        tamper.execute(
            """UPDATE sqlite_master
            SET sql = replace(sql, 'RAISE(', 'RAISE\u00a0(')
            WHERE type = 'trigger' AND name = 'retrieval_generations_reject_update'"""
        )
        tamper.execute("PRAGMA writable_schema=OFF")

    connection = connect_cache_database(path)
    try:
        with pytest.raises(CacheCorruptError, match="schema"):
            verify_cache_database(connection)
    finally:
        connection.close()
