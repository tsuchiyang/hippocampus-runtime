from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
from importlib import import_module

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.hook_input import parse_hook_payload
from codex_memory.paths import RuntimePaths


class CommitFailingConnection(sqlite3.Connection):
    fail_next_commit = False

    def commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("forced commit failure")
        super().commit()


def test_database_and_event_store_modules_are_available() -> None:
    assert importlib.util.find_spec("codex_memory.database") is not None
    assert importlib.util.find_spec("codex_memory.event_store") is not None


def test_replayed_payload_is_idempotent(runtime, prompt_envelope) -> None:
    first = runtime.events.append(prompt_envelope)
    second = runtime.events.append(prompt_envelope)

    assert first.event_id == second.event_id
    assert runtime.events.highwater(prompt_envelope.session_id) == first.event_id


def test_replayed_payload_returns_original_id_from_new_connection(runtime, prompt_envelope) -> None:
    first = runtime.events.append(prompt_envelope)
    runtime.connection.close()

    database = import_module("codex_memory.database")
    event_store = import_module("codex_memory.event_store")
    connection = database.connect_database(runtime.paths.db_path)
    database.migrate_database(connection)
    replayed = event_store.EventStore(connection, BlobStore(runtime.paths.blobs_dir)).append(prompt_envelope)

    assert replayed.event_id == first.event_id
    connection.close()


def test_key_order_only_payload_change_is_idempotent(runtime) -> None:
    first = parse_hook_payload(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "session_a",
            "turn_id": "turn_1",
            "cwd": ".",
            "prompt": "hello",
            "metadata": {"a": 1, "b": 2},
        }
    )
    reordered = parse_hook_payload(
        {
            "metadata": {"b": 2, "a": 1},
            "prompt": "hello",
            "cwd": ".",
            "turn_id": "turn_1",
            "session_id": "session_a",
            "hook_event_name": "UserPromptSubmit",
        }
    )

    assert runtime.events.append(first).event_id == runtime.events.append(reordered).event_id


def test_event_payload_blob_and_key_use_exact_canonical_bytes(runtime) -> None:
    envelope = parse_hook_payload(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "session_a",
            "turn_id": "turn_1",
            "cwd": ".",
            "prompt": "hello",
            "metadata": {"z": "é", "a": [1, 2]},
        }
    )
    expected_payload = (
        b'{"cwd":".","hook_event_name":"UserPromptSubmit","metadata":{"a":[1,2],'
        b'"z":"\xc3\xa9"},"prompt":"hello","session_id":"session_a","turn_id":"turn_1"}'
    )
    key_parts = (
        expected_payload,
        b"UserPromptSubmit",
        b"session_a",
        b"turn_1",
        b"",
    )
    expected_key_digest = hashlib.sha256(
        b"".join(len(part).to_bytes(8, "big") + part for part in key_parts)
    ).hexdigest()

    record = runtime.events.append(envelope)
    payload_digest = hashlib.sha256(expected_payload).hexdigest()

    assert record.event_key == f"sha256:{expected_key_digest}"
    assert record.payload_blob_sha256 == payload_digest
    assert (
        runtime.paths.blobs_dir / "sha256" / payload_digest[:2] / f"{payload_digest}.bin"
    ).read_bytes() == expected_payload


def test_two_tool_use_ids_in_same_turn_are_distinct(runtime, tool_envelope_factory) -> None:
    one = runtime.events.append(tool_envelope_factory("session_a", "tool_1", "Bash"))
    two = runtime.events.append(tool_envelope_factory("session_a", "tool_2", "Bash"))

    assert one.event_id != two.event_id


def test_latest_prompt_and_recent_tools_are_session_scoped(
    runtime,
    prompt_envelope_factory,
    tool_envelope_factory,
) -> None:
    runtime.events.append(prompt_envelope_factory("session_a", "first"))
    runtime.events.append(prompt_envelope_factory("session_b", "other"))
    runtime.events.append(
        parse_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session_a",
                "turn_id": "turn_2",
                "cwd": ".",
                "prompt": "latest",
            }
        )
    )
    runtime.events.append(tool_envelope_factory("session_a", "tool_1", "Bash"))
    runtime.events.append(tool_envelope_factory("session_a", "tool_2", "apply_patch"))
    runtime.events.append(tool_envelope_factory("session_b", "tool_3", "Bash"))

    assert runtime.events.latest_user_prompt("session_a") == "latest"
    assert runtime.events.recent_tool_names("session_a", limit=10) == ["apply_patch", "Bash"]
    assert runtime.events.recent_tool_names("session_a", limit=0) == []
    with pytest.raises(ValueError, match="limit"):
        runtime.events.recent_tool_names("session_a", limit=-1)


def test_database_pragmas_and_migration_are_effective(tmp_path) -> None:
    database = import_module("codex_memory.database")
    path = RuntimePaths.for_repo(tmp_path).db_path
    connection = database.connect_database(path)
    database.migrate_database(connection)
    database.migrate_database(connection)

    assert connection.row_factory is sqlite3.Row
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert connection.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0] == "3"
    connection.close()


@pytest.mark.parametrize("schema_version", ["4", "one"])
def test_migration_rejects_unsupported_or_malformed_schema_version(tmp_path, schema_version) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    connection.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)", (schema_version,)
    )

    with pytest.raises(ValueError, match="schema version"):
        database.migrate_database(connection)
    connection.close()


def test_migration_rolls_back_all_schema_changes_when_version_insert_fails(tmp_path) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)

    def deny_schema_version_insert(action, table, _column, _database, _trigger):
        if action == sqlite3.SQLITE_INSERT and table == "schema_meta":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_schema_version_insert)
    with pytest.raises(sqlite3.DatabaseError):
        database.migrate_database(connection)
    connection.set_authorizer(None)

    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'events'"
    ).fetchone() is None
    connection.close()


def test_migration_commit_failure_rolls_back_and_connection_is_reusable(tmp_path) -> None:
    database = import_module("codex_memory.database")
    connection = sqlite3.connect(
        tmp_path / "runtime.sqlite3",
        isolation_level=None,
        factory=CommitFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.fail_next_commit = True

    with pytest.raises(sqlite3.OperationalError, match="forced commit failure"):
        database.migrate_database(connection)

    assert connection.in_transaction is False
    database.migrate_database(connection)
    assert connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()[0] == "3"
    connection.close()


def test_event_append_commit_failure_rolls_back_and_connection_is_reusable(
    tmp_path, prompt_envelope
) -> None:
    database = import_module("codex_memory.database")
    event_store = import_module("codex_memory.event_store")
    connection = sqlite3.connect(
        tmp_path / "runtime.sqlite3",
        isolation_level=None,
        factory=CommitFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    database.migrate_database(connection)
    events = event_store.EventStore(connection, BlobStore(tmp_path / "blobs"))
    connection.fail_next_commit = True

    with pytest.raises(sqlite3.OperationalError, match="forced commit failure"):
        events.append(prompt_envelope)

    assert connection.in_transaction is False
    stored = events.append(prompt_envelope)
    assert events.highwater(prompt_envelope.session_id) == stored.event_id
    connection.close()


def test_checkpoints_reject_invalid_status(tmp_path) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    database.migrate_database(connection)

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO checkpoints (checkpoint_id, session_id, turn_id, trigger, status, created_at) "
            "VALUES ('checkpoint_1', 'session_a', 'turn_1', 'manual', 'INVALID', '2026-08-30T00:00:00.000000Z')"
        )
    connection.close()


def test_checkpoints_reject_incomplete_committed_rows(tmp_path) -> None:
    database = import_module("codex_memory.database")
    connection = database.connect_database(RuntimePaths.for_repo(tmp_path).db_path)
    database.migrate_database(connection)

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO checkpoints (checkpoint_id, session_id, turn_id, trigger, status, created_at) "
            "VALUES ('checkpoint_1', 'session_a', 'turn_1', 'manual', 'COMMITTED', "
            "'2026-08-30T00:00:00.000000Z')"
        )
    connection.close()
