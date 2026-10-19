from __future__ import annotations

import importlib
from pathlib import Path
import sqlite3

import pytest

from codex_memory import database, structured_ledger
from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import capture_candidate
from codex_memory.paths import RuntimePaths
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.canonical_snapshot import capture_canonical_snapshot


def _canonical_database(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    repo = tmp_path / "projection repository"
    repo.mkdir()
    runtime = RuntimePaths.for_repo(repo)
    runtime.ensure()
    connection = database.connect_database(runtime.db_path)
    database.migrate_database(connection)
    return repo, connection


def _propose(connection: sqlite3.Connection, repo: Path, key: str, *, evidence: tuple[str, ...] = ()):
    return structured_ledger.propose_record(
        connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(repo),
        topic_key=f"projection.{key}",
        statement=f"Project permanent record {key}.",
        value={"key": key},
        evidence_uris=evidence,
        idempotency_key=f"projection-{key}",
    )


def test_projection_uses_one_verified_snapshot_and_excludes_candidates_and_blobs(tmp_path: Path) -> None:
    snapshot_module = importlib.import_module("codex_memory.retrieval.canonical_snapshot")
    projector = importlib.import_module("codex_memory.retrieval.projector")
    repo, connection = _canonical_database(tmp_path)
    first = _propose(
        connection,
        repo,
        "first",
        evidence=(('H' + 'TTPS://viewer:secret@example.test/path?token=secret#fragment'),),
    )
    second = _propose(connection, repo, "second")
    structured_ledger.apply_action(connection, record_id=first.record_id, action="activate", idempotency_key="activate-first")
    structured_ledger.apply_action(
        connection,
        record_id=first.record_id,
        action="supersede",
        replacement_record_id=second.record_id,
        idempotency_key="supersede-first",
    )
    capture_candidate(
        connection,
        BlobStore(RuntimePaths.for_repo(repo).blobs_dir),
        project_scope_key=str(repo),
        record_type="decision",
        scope_kind="project",
        scope_key=str(repo),
        topic_key="projection.candidate",
        statement="Temporary rows never enter a retrieval generation.",
        session_id="session",
        turn_id="turn",
        context_id="context",
        idempotency_key="candidate",
    )
    connection.close()

    paths = RetrievalPaths.for_repo(repo)
    with snapshot_module.open_verified_canonical_snapshot(paths) as (canonical, snapshot):
        first_batch = projector.project_full_snapshot(canonical, snapshot)
        second_batch = projector.project_full_snapshot(canonical, snapshot)

    assert first_batch == second_batch
    assert [document.record_id for document in first_batch.documents] == [first.record_id, second.record_id]
    assert [document.state for document in first_batch.documents] == ["superseded", "proposed"]
    assert first_batch.documents[0].relation_endpoints == (second.record_id,)
    assert first_batch.documents[1].relation_endpoints == (first.record_id,)
    assert first_batch.documents[0].safe_references == (
        "https:sha256:29b582b3d132ae02a24ae2db353f68476ebf9dfc65a56c0aad31c8ac836bd3f5",
    )
    assert all("secret" not in source.safe_reference for source in first_batch.sources)
    assert first_batch.omitted_unsafe_count == 0
    assert all("candidate" not in document.topic_key for document in first_batch.documents)


def test_projection_omits_a_record_when_a_safe_reference_cannot_be_derived(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot_module = importlib.import_module("codex_memory.retrieval.canonical_snapshot")
    projector = importlib.import_module("codex_memory.retrieval.projector")
    repo, connection = _canonical_database(tmp_path)
    _propose(connection, repo, "unsafe", evidence=("https://example.test/evidence",))
    connection.close()
    monkeypatch.setattr(
        projector,
        "safe_evidence_reference",
        lambda _uri: (_ for _ in ()).throw(ValueError("unsafe")),
    )

    with snapshot_module.open_verified_canonical_snapshot(RetrievalPaths.for_repo(repo)) as (
        canonical,
        snapshot,
    ):
        batch = projector.project_full_snapshot(canonical, snapshot)

    assert batch.documents == ()
    assert batch.sources == ()
    assert batch.symbols == ()
    assert batch.omitted_unsafe_count == 1


def test_stale_delta_tracks_lifecycle_evidence_and_relation_changes_but_suppresses_inactive_rows(
    tmp_path: Path,
) -> None:
    """Would fail if stale candidates were drawn from stale index state."""
    projector = importlib.import_module("codex_memory.retrieval.projector")
    repo, connection = _canonical_database(tmp_path)
    first = _propose(connection, repo, "first", evidence=("https://example.test/first",))
    second = _propose(connection, repo, "second")
    structured_ledger.apply_action(
        connection, record_id=first.record_id, action="activate", idempotency_key="activate-first"
    )
    since = capture_canonical_snapshot(RetrievalPaths.for_repo(repo)).highwater
    structured_ledger.apply_action(
        connection,
        record_id=first.record_id,
        action="supersede",
        replacement_record_id=second.record_id,
        idempotency_key="supersede-first",
    )

    changed = projector.changed_record_ids(connection, since)
    delta = projector.project_stale_delta(connection, since=since, limit=8)

    assert changed == (first.record_id, second.record_id)
    assert first.record_id not in delta.record_ids
    assert delta.record_ids == (second.record_id,)
    assert delta.overflowed is False


def test_stale_delta_reports_overflow_without_claiming_a_complete_candidate_set(tmp_path: Path) -> None:
    """Would fail if a truncated stale delta were treated as trusted/current."""
    projector = importlib.import_module("codex_memory.retrieval.projector")
    repo, connection = _canonical_database(tmp_path)
    since = capture_canonical_snapshot(RetrievalPaths.for_repo(repo)).highwater
    first = _propose(connection, repo, "overflow-one")
    second = _propose(connection, repo, "overflow-two")

    delta = projector.project_stale_delta(connection, since=since, limit=1)

    assert delta.record_ids == ()
    assert delta.overflowed is True
    assert projector.changed_record_ids(connection, since) == (first.record_id, second.record_id)


class _SmallParameterConnection:
    """Makes an accidental large SQLite IN binding observable in a small test."""

    def __init__(self, connection: sqlite3.Connection, maximum: int) -> None:
        self._connection = connection
        self._maximum = maximum

    def execute(self, sql: str, parameters: object = ()):
        if " IN (" in sql and len(tuple(parameters)) > self._maximum:
            raise AssertionError("changed-ID scan exceeded its bounded parameter budget")
        return self._connection.execute(sql, parameters)


def test_changed_ids_and_stale_delta_use_ordered_bounded_scans(tmp_path: Path) -> None:
    """Would fail if either path assembled all changed IDs into one IN query."""
    projector = importlib.import_module("codex_memory.retrieval.projector")
    repo, connection = _canonical_database(tmp_path)
    since = capture_canonical_snapshot(RetrievalPaths.for_repo(repo)).highwater
    records = tuple(_propose(connection, repo, f"bounded-{number}") for number in range(7))
    bounded = _SmallParameterConnection(connection, maximum=6)

    assert projector.changed_record_ids(bounded, since) == tuple(record.record_id for record in records)
    assert projector.project_stale_delta(bounded, since=since, limit=2).overflowed is True


@pytest.mark.parametrize(
    ("action", "expected_state", "visible"),
    [
        ("activate", "active", True),
        ("invalidate", "invalidated", False),
        ("close", "closed", False),
        ("reopen", "active", True),
    ],
)
def test_stale_delta_tracks_each_lifecycle_transition(
    tmp_path: Path, action: str, expected_state: str, visible: bool
) -> None:
    """Would fail if a lifecycle event did not invalidate/correct stale state."""
    projector = importlib.import_module("codex_memory.retrieval.projector")
    repo, connection = _canonical_database(tmp_path)
    record = _propose(connection, repo, f"lifecycle-{action}")
    if action != "activate":
        structured_ledger.apply_action(
            connection, record_id=record.record_id, action="activate", idempotency_key=f"before-{action}"
        )
    if action == "reopen":
        structured_ledger.apply_action(connection, record_id=record.record_id, action="close", idempotency_key="close-reopen")
    since = capture_canonical_snapshot(RetrievalPaths.for_repo(repo)).highwater
    structured_ledger.apply_action(
        connection, record_id=record.record_id, action=action, idempotency_key=f"after-{action}"
    )

    assert projector.changed_record_ids(connection, since) == (record.record_id,)
    delta = projector.project_stale_delta(connection, since=since, limit=1)
    assert connection.execute("SELECT state FROM memory_current WHERE record_id = ?", (record.record_id,)).fetchone()[0] == expected_state
    assert (record.record_id in delta.record_ids) is visible
