from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import time
from typing import Callable

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.git_capture import GitSnapshot
from codex_memory.hook_input import HookEnvelope, parse_hook_payload
from codex_memory.transcript import TranscriptCapture, TranscriptCaptureError

try:
    import codex_memory.checkpoint as checkpoint_module
except ModuleNotFoundError:
    checkpoint_module = None


def _precompact(
    session_id: str = "session_a", *, cwd: Path | str = ".", transcript_path: Path | None = None
) -> HookEnvelope:
    payload: dict[str, object] = {
        "hook_event_name": "PreCompact",
        "session_id": session_id,
        "turn_id": "turn_7",
        "cwd": str(cwd),
        "trigger": "manual",
    }
    if transcript_path is not None:
        payload["transcript_path"] = str(transcript_path)
    return parse_hook_payload(payload)


def _postcompact(session_id: str = "session_a", *, cwd: Path | str = ".") -> HookEnvelope:
    return parse_hook_payload(
        {
            "hook_event_name": "PostCompact",
            "session_id": session_id,
            "turn_id": "turn_8",
            "cwd": str(cwd),
            "trigger": "auto",
        }
    )


def _blob_path(blobs_dir: Path, digest: str) -> Path:
    return blobs_dir / "sha256" / digest[:2] / f"{digest}.bin"


def _checkpoint_path(paths, session_id: str, checkpoint_id: str) -> Path:
    session_digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return paths.checkpoints_dir / session_digest / f"{checkpoint_id}.json"


def _captured_transcript(_path: Path | None, blobs: BlobStore) -> TranscriptCapture:
    data = b'{"message":"opaque transcript"}\n'
    ref = blobs.put_bytes(data)
    return TranscriptCapture("captured", "C:\\transcripts\\session.jsonl", ref.sha256, len(data))


def _unavailable_transcript(_path: Path | None, _blobs: BlobStore) -> TranscriptCapture:
    return TranscriptCapture("unavailable", None, None, None)


def _git_snapshot(repo_root: Path, blobs: BlobStore) -> GitSnapshot:
    status = blobs.put_bytes(b" M src/example.py\0")
    working = blobs.put_bytes(b"diff --git a/src/example.py b/src/example.py\nworking\n")
    staged = blobs.put_bytes(b"diff --git a/staged.txt b/staged.txt\nstaged\n")
    return GitSnapshot(
        repo_root=str(repo_root.resolve()),
        branch="master",
        head="a" * 40,
        is_unborn=False,
        status_blob_sha256=status.sha256,
        working_diff_blob_sha256=working.sha256,
        staged_diff_blob_sha256=staged.sha256,
        untracked_paths=["notes/next.txt"],
    )


@pytest.fixture
def service_factory(runtime):
    if checkpoint_module is None:
        pytest.skip("checkpoint module is not implemented yet")

    def make_service(
        *,
        transcript_capture: Callable = _captured_transcript,
        git_capture: Callable = _git_snapshot,
        service_type=None,
    ):
        service_class = service_type or checkpoint_module.CheckpointService
        return service_class(
            runtime.paths,
            runtime.events,
            BlobStore(runtime.paths.blobs_dir),
            transcript_capture=transcript_capture,
            git_capture=git_capture,
        )

    return make_service


def _checkpoint_rows(runtime):
    return runtime.connection.execute(
        "SELECT * FROM checkpoints ORDER BY checkpoint_seq"
    ).fetchall()


class _CommitFailureProxy:
    def __init__(self, connection: sqlite3.Connection, fail_on: int) -> None:
        self._connection = connection
        self._fail_on = fail_on
        self._commit_calls = 0

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def commit(self) -> None:
        self._commit_calls += 1
        if self._commit_calls == self._fail_on:
            raise sqlite3.OperationalError("injected commit failure")
        self._connection.commit()


def _assert_connection_reusable(connection: sqlite3.Connection) -> None:
    assert connection.in_transaction is False
    connection.execute("BEGIN IMMEDIATE")
    connection.rollback()


def _create_directory_redirect(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return
    pytest.skip("directory symlink/junction creation is unavailable")


def _remove_directory_redirect(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif getattr(link, "is_junction", lambda: False)():
        link.rmdir()


def test_checkpoint_module_is_available() -> None:
    if checkpoint_module is None:
        pytest.fail("codex_memory.checkpoint is not implemented")


def test_precompact_commits_canonical_self_hashed_manifest_and_metadata(
    runtime, service_factory
) -> None:
    envelope = _precompact(cwd=runtime.paths.repo_root)
    service = service_factory()

    manifest = service.create_precompact(envelope)

    rows = _checkpoint_rows(runtime)
    assert len(rows) == 1
    row = rows[0]
    assert row["checkpoint_id"] == manifest.checkpoint_id
    assert row["status"] == "COMMITTED"
    assert row["session_id"] == "session_a"
    assert row["turn_id"] == "turn_7"
    assert row["trigger"] == "manual"
    assert row["committed_at"].endswith("Z")
    assert row["raw_event_highwater"] == manifest.raw_event_highwater
    assert row["manifest_sha256"] == manifest.manifest_sha256
    assert row["recovery_required"] == 0

    event = runtime.connection.execute(
        "SELECT event_id, event_name FROM events WHERE session_id = ? ORDER BY event_id DESC LIMIT 1",
        ("session_a",),
    ).fetchone()
    assert (event["event_id"], event["event_name"]) == (
        manifest.raw_event_highwater,
        "PreCompact",
    )

    expected_path = _checkpoint_path(
        runtime.paths, manifest.session_id, manifest.checkpoint_id
    )
    assert row["manifest_relpath"] == expected_path.relative_to(
        runtime.paths.checkpoints_dir
    ).as_posix()
    raw = expected_path.read_bytes()
    document = json.loads(raw)
    assert raw == json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    stored_digest = document.pop("manifest_sha256")
    expected_digest = hashlib.sha256(
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert stored_digest == expected_digest == manifest.manifest_sha256
    assert document["status"] == "COMMITTED"
    assert document["recovery_required"] is False
    assert service.latest_committed("session_a") == manifest


def test_null_transcript_commits_explicit_degraded_checkpoint(runtime, service_factory) -> None:
    service = service_factory(transcript_capture=_unavailable_transcript)

    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    assert manifest.transcript == TranscriptCapture("unavailable", None, None, None)
    assert service.latest_committed("session_a") == manifest


@pytest.mark.parametrize("failed_capture", ["transcript", "git"])
def test_capture_failure_leaves_only_preparing_state(
    runtime, service_factory, failed_capture: str
) -> None:
    def fail_transcript(_path, _blobs):
        raise TranscriptCaptureError("unstable transcript")

    def fail_git(_path, _blobs):
        raise RuntimeError("git failed")

    service = service_factory(
        transcript_capture=fail_transcript if failed_capture == "transcript" else _captured_transcript,
        git_capture=fail_git if failed_capture == "git" else _git_snapshot,
    )

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    assert [row["status"] for row in _checkpoint_rows(runtime)] == ["PREPARING"]
    assert service.latest_committed("session_a") is None


def test_writer_failure_before_replace_leaves_preparing_and_releases_lock(
    runtime, service_factory
) -> None:
    class FailingWriter(checkpoint_module.CheckpointService):
        def _write_manifest(self, path, document):
            raise OSError("injected write failure")

    service = service_factory(service_type=FailingWriter)
    envelope = _precompact(cwd=runtime.paths.repo_root)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(envelope)

    row = _checkpoint_rows(runtime)[0]
    assert row["status"] == "PREPARING"
    assert not _checkpoint_path(runtime.paths, envelope.session_id, row["checkpoint_id"]).exists()
    with service._session_lock(envelope.session_id):
        pass


def test_failure_after_replace_before_db_transition_leaves_orphan_manifest_preparing(
    runtime, service_factory
) -> None:
    class FailingCommit(checkpoint_module.CheckpointService):
        def _commit_prepared(self, *args, **kwargs):
            raise RuntimeError("injected crash after replace")

    service = service_factory(service_type=FailingCommit)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    row = _checkpoint_rows(runtime)[0]
    assert row["status"] == "PREPARING"
    assert _checkpoint_path(runtime.paths, "session_a", row["checkpoint_id"]).is_file()
    assert service.latest_committed("session_a") is None


@pytest.mark.parametrize("damage", ["invalid-json", "noncanonical"])
def test_manifest_re_read_rejects_corrupt_or_noncanonical_bytes_before_commit(
    runtime, service_factory, damage: str
) -> None:
    class DamagingWriter(checkpoint_module.CheckpointService):
        def _write_manifest(self, path, document):
            super()._write_manifest(path, document)
            if damage == "invalid-json":
                path.write_bytes(b"{not-json")
            else:
                path.write_bytes(path.read_bytes() + b" ")

    service = service_factory(service_type=DamagingWriter)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    assert [row["status"] for row in _checkpoint_rows(runtime)] == ["PREPARING"]


@pytest.mark.parametrize(
    ("reference", "damage"),
    [
        ("transcript", "missing"),
        ("transcript", "tampered"),
        ("status", "missing"),
        ("status", "tampered"),
        ("working", "missing"),
        ("working", "tampered"),
        ("staged", "missing"),
        ("staged", "tampered"),
    ],
)
def test_latest_committed_fails_closed_for_missing_or_tampered_referenced_blob(
    runtime, service_factory, reference: str, damage: str
) -> None:
    service = service_factory()
    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    digests = {
        "transcript": manifest.transcript.blob_sha256,
        "status": manifest.git.status_blob_sha256,
        "working": manifest.git.working_diff_blob_sha256,
        "staged": manifest.git.staged_diff_blob_sha256,
    }
    digest = digests[reference]
    assert digest is not None
    path = _blob_path(runtime.paths.blobs_dir, digest)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(checkpoint_module.CheckpointError):
        service.latest_committed("session_a")


def test_guarded_commit_rejects_lost_preparing_row(runtime, service_factory) -> None:
    class DeletedPreparation(checkpoint_module.CheckpointService):
        def _commit_prepared(self, checkpoint_id, *args, **kwargs):
            self.events._connection.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            )
            return super()._commit_prepared(checkpoint_id, *args, **kwargs)

    service = service_factory(service_type=DeletedPreparation)

    with pytest.raises(checkpoint_module.CheckpointError, match="transition"):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    assert _checkpoint_rows(runtime) == []


def test_latest_committed_selects_newest_exact_session_only(runtime, service_factory) -> None:
    service = service_factory()
    first_a = service.create_precompact(_precompact("session_a", cwd=runtime.paths.repo_root))
    only_b = service.create_precompact(_precompact("session_b", cwd=runtime.paths.repo_root))
    newest_a = service.create_precompact(_precompact("session_a", cwd=runtime.paths.repo_root))

    assert service.latest_committed("session_a") == newest_a
    assert service.latest_committed("session_b") == only_b
    assert service.latest_committed("missing_session") is None
    assert service.latest_committed("session_a") != first_a


def test_postcompact_appends_event_then_marks_only_newest_same_session(
    runtime, service_factory
) -> None:
    service = service_factory()
    first_a = service.create_precompact(_precompact("session_a", cwd=runtime.paths.repo_root))
    only_b = service.create_precompact(_precompact("session_b", cwd=runtime.paths.repo_root))
    newest_a = service.create_precompact(_precompact("session_a", cwd=runtime.paths.repo_root))
    runtime.connection.execute(
        """
        CREATE TRIGGER require_postcompact_event_before_recovery
        BEFORE UPDATE OF recovery_required ON checkpoints
        WHEN NEW.recovery_required = 1
        BEGIN
            SELECT CASE WHEN NOT EXISTS (
                SELECT 1 FROM events
                WHERE session_id = NEW.session_id
                  AND event_name = 'PostCompact'
                  AND event_id > OLD.raw_event_highwater
            ) THEN RAISE(ABORT, 'PostCompact event missing before recovery update') END;
        END
        """
    )

    marked = service.mark_postcompact(_postcompact("session_a", cwd=runtime.paths.repo_root))

    assert marked is not None
    assert marked.checkpoint_id == newest_a.checkpoint_id
    assert marked.recovery_required is True
    flags = {
        row["checkpoint_id"]: row["recovery_required"] for row in _checkpoint_rows(runtime)
    }
    assert flags == {
        first_a.checkpoint_id: 0,
        only_b.checkpoint_id: 0,
        newest_a.checkpoint_id: 1,
    }
    post_event = runtime.connection.execute(
        "SELECT event_name, session_id FROM events ORDER BY event_id DESC LIMIT 1"
    ).fetchone()
    assert (post_event["event_name"], post_event["session_id"]) == (
        "PostCompact",
        "session_a",
    )
    manifest_path = _checkpoint_path(runtime.paths, "session_a", newest_a.checkpoint_id)
    assert json.loads(manifest_path.read_bytes())["recovery_required"] is False


def test_postcompact_without_committed_checkpoint_returns_none_after_event(
    runtime, service_factory
) -> None:
    service = service_factory()

    assert service.mark_postcompact(_postcompact(cwd=runtime.paths.repo_root)) is None
    row = runtime.connection.execute(
        "SELECT event_name FROM events ORDER BY event_id DESC LIMIT 1"
    ).fetchone()
    assert row["event_name"] == "PostCompact"


def test_recovery_delivered_guarded_transition_is_same_session_and_manifest_immutable(
    runtime, service_factory
) -> None:
    service = service_factory()
    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    service.mark_postcompact(_postcompact(cwd=runtime.paths.repo_root))
    manifest_path = _checkpoint_path(runtime.paths, "session_a", manifest.checkpoint_id)
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(checkpoint_module.CheckpointError):
        service.mark_recovery_delivered(manifest.checkpoint_id, "session_b")
    service.mark_recovery_delivered(manifest.checkpoint_id, "session_a")

    row = _checkpoint_rows(runtime)[0]
    assert row["recovery_required"] == 0
    assert row["recovery_delivered_at"].endswith("Z")
    assert manifest_path.read_bytes() == original_bytes
    with pytest.raises(checkpoint_module.CheckpointError):
        service.mark_recovery_delivered(manifest.checkpoint_id, "session_a")


@pytest.mark.parametrize(
    "invalid_envelope",
    [
        _postcompact(),
        replace(_precompact(), turn_id=None),
        replace(_precompact(), trigger="unsupported"),
    ],
)
def test_create_precompact_rejects_invalid_envelopes_without_side_effects(
    runtime, service_factory, invalid_envelope: HookEnvelope
) -> None:
    service = service_factory()

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(invalid_envelope)

    assert runtime.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert _checkpoint_rows(runtime) == []


def test_mark_postcompact_rejects_wrong_envelope_without_appending(runtime, service_factory) -> None:
    service = service_factory()

    with pytest.raises(checkpoint_module.CheckpointError):
        service.mark_postcompact(_precompact(cwd=runtime.paths.repo_root))

    assert runtime.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_session_names_are_hashed_for_checkpoint_and_lock_paths(runtime, service_factory) -> None:
    session_id = "..\\..//outside/危険 session"
    service = service_factory()

    manifest = service.create_precompact(
        _precompact(session_id, cwd=runtime.paths.repo_root)
    )

    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    manifest_path = _checkpoint_path(runtime.paths, session_id, manifest.checkpoint_id)
    lock_path = runtime.paths.checkpoints_dir / ".locks" / f"{digest}.lock"
    assert manifest_path.is_file()
    assert manifest_path.parent.name == digest
    assert lock_path.is_file()
    assert manifest_path.resolve().is_relative_to(runtime.paths.checkpoints_dir.resolve())
    assert json.loads(manifest_path.read_bytes())["session_id"] == session_id


@pytest.mark.skipif(os.name != "nt", reason="Windows advisory lock contract")
def test_windows_lock_contention_is_bounded_and_lock_is_released(
    runtime, service_factory, monkeypatch
) -> None:
    import msvcrt

    session_id = "contended-session"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    lock_path = runtime.paths.checkpoints_dir / ".locks" / f"{digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    service = service_factory()
    monkeypatch.setattr(checkpoint_module, "_LOCK_ATTEMPTS", 2)
    monkeypatch.setattr(checkpoint_module, "_LOCK_DELAY_SECONDS", 0.01)

    with lock_path.open("a+b") as contender:
        contender.seek(0)
        msvcrt.locking(contender.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            started = time.monotonic()
            with pytest.raises(checkpoint_module.CheckpointError, match="lock"):
                service.create_precompact(
                    _precompact(session_id, cwd=runtime.paths.repo_root)
                )
            assert time.monotonic() - started < 1.0
            events = runtime.connection.execute(
                "SELECT event_name FROM events ORDER BY event_id"
            ).fetchall()
            assert [row["event_name"] for row in events] == ["PreCompact"]
            assert _checkpoint_rows(runtime) == []
        finally:
            contender.seek(0)
            msvcrt.locking(contender.fileno(), msvcrt.LK_UNLCK, 1)

    manifest = service.create_precompact(_precompact(session_id, cwd=runtime.paths.repo_root))
    assert manifest.session_id == session_id
    with lock_path.open("a+b") as probe:
        probe.seek(0)
        msvcrt.locking(probe.fileno(), msvcrt.LK_NBLCK, 1)
        probe.seek(0)
        msvcrt.locking(probe.fileno(), msvcrt.LK_UNLCK, 1)


def test_preparing_commit_failure_rolls_back_and_connection_remains_reusable(
    runtime, service_factory
) -> None:
    service = service_factory()
    runtime.events._connection = _CommitFailureProxy(runtime.connection, fail_on=2)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    assert _checkpoint_rows(runtime) == []
    assert runtime.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    _assert_connection_reusable(runtime.connection)


def test_committed_transition_commit_failure_reverts_to_preparing(
    runtime, service_factory
) -> None:
    service = service_factory()
    runtime.events._connection = _CommitFailureProxy(runtime.connection, fail_on=3)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    rows = _checkpoint_rows(runtime)
    assert len(rows) == 1
    assert rows[0]["status"] == "PREPARING"
    assert rows[0]["committed_at"] is None
    assert rows[0]["manifest_relpath"] is None
    _assert_connection_reusable(runtime.connection)


def test_postcompact_commit_failure_rolls_back_recovery_mutation(
    runtime, service_factory
) -> None:
    service = service_factory()
    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    runtime.events._connection = _CommitFailureProxy(runtime.connection, fail_on=2)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.mark_postcompact(_postcompact(cwd=runtime.paths.repo_root))

    row = runtime.connection.execute(
        "SELECT recovery_required FROM checkpoints WHERE checkpoint_id = ?",
        (manifest.checkpoint_id,),
    ).fetchone()
    assert row["recovery_required"] == 0
    assert runtime.connection.execute(
        "SELECT COUNT(*) FROM events WHERE event_name = 'PostCompact'"
    ).fetchone()[0] == 1
    _assert_connection_reusable(runtime.connection)


def test_recovery_delivery_commit_failure_rolls_back_and_remains_required(
    runtime, service_factory
) -> None:
    service = service_factory()
    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    service.mark_postcompact(_postcompact(cwd=runtime.paths.repo_root))
    runtime.events._connection = _CommitFailureProxy(runtime.connection, fail_on=1)

    with pytest.raises(checkpoint_module.CheckpointError):
        service.mark_recovery_delivered(manifest.checkpoint_id, manifest.session_id)

    row = runtime.connection.execute(
        "SELECT recovery_required, recovery_delivered_at FROM checkpoints WHERE checkpoint_id = ?",
        (manifest.checkpoint_id,),
    ).fetchone()
    assert row["recovery_required"] == 1
    assert row["recovery_delivered_at"] is None
    _assert_connection_reusable(runtime.connection)


def test_checkpoint_rejects_real_lock_directory_redirect_without_external_write(
    runtime, service_factory
) -> None:
    session_id = "redirected-session"
    checkpoints = runtime.paths.checkpoints_dir
    checkpoints.mkdir(parents=True, exist_ok=True)
    external = runtime.paths.repo_root / "external-locks"
    external.mkdir()
    link = checkpoints / ".locks"
    _create_directory_redirect(link, external)
    service = service_factory()

    try:
        with pytest.raises(checkpoint_module.CheckpointError, match="path|redirect|safe"):
            service.create_precompact(
                _precompact(session_id, cwd=runtime.paths.repo_root)
            )
        assert list(external.iterdir()) == []
    finally:
        _remove_directory_redirect(link)


def test_latest_committed_rejects_real_session_directory_redirect(runtime, service_factory) -> None:
    service = service_factory()
    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    original = _checkpoint_path(runtime.paths, manifest.session_id, manifest.checkpoint_id)
    original_bytes = original.read_bytes()
    external = runtime.paths.repo_root / "external-session"
    external.mkdir()
    external_manifest = external / original.name
    external_manifest.write_bytes(original_bytes)
    session_directory = original.parent
    original.unlink()
    session_directory.rmdir()
    _create_directory_redirect(session_directory, external)

    try:
        with pytest.raises(checkpoint_module.CheckpointError, match="path|redirect|safe"):
            service.latest_committed(manifest.session_id)
        assert external_manifest.read_bytes() == original_bytes
    finally:
        _remove_directory_redirect(session_directory)


@pytest.mark.parametrize("redirected_leaf", ["lock", "manifest"])
def test_checkpoint_consults_redirect_guard_for_lock_and_manifest_leaf(
    runtime, service_factory, monkeypatch, redirected_leaf: str
) -> None:
    session_id = "guarded-leaf"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    fixed_hex = "1" * 32
    lock_path = runtime.paths.checkpoints_dir / ".locks" / f"{digest}.lock"
    manifest_path = (
        runtime.paths.checkpoints_dir / digest / f"cp_{fixed_hex}.json"
    )
    guarded_path = lock_path if redirected_leaf == "lock" else manifest_path
    original_guard = getattr(checkpoint_module, "_path_is_redirect", lambda _path: False)
    monkeypatch.setattr(
        checkpoint_module,
        "_path_is_redirect",
        lambda path: path == guarded_path or original_guard(path),
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint_module, "uuid4", lambda: type("FixedUuid", (), {"hex": fixed_hex})()
    )
    service = service_factory()

    with pytest.raises(checkpoint_module.CheckpointError, match="path|redirect|safe"):
        service.create_precompact(_precompact(session_id, cwd=runtime.paths.repo_root))

    rows = _checkpoint_rows(runtime)
    if redirected_leaf == "lock":
        assert rows == []
    else:
        assert len(rows) == 1 and rows[0]["status"] == "PREPARING"


def test_latest_committed_consults_redirect_guard_for_manifest_leaf(
    runtime, service_factory, monkeypatch
) -> None:
    service = service_factory()
    manifest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    manifest_path = _checkpoint_path(
        runtime.paths, manifest.session_id, manifest.checkpoint_id
    )
    original_guard = checkpoint_module._path_is_redirect
    monkeypatch.setattr(
        checkpoint_module,
        "_path_is_redirect",
        lambda path: path == manifest_path or original_guard(path),
    )

    with pytest.raises(checkpoint_module.CheckpointError, match="path|redirect|safe"):
        service.latest_committed(manifest.session_id)


def test_redirect_guard_rejects_generic_windows_reparse_attribute() -> None:
    class ReparseOnlyPath:
        @staticmethod
        def is_symlink() -> bool:
            return False

        @staticmethod
        def is_junction() -> bool:
            return False

        @staticmethod
        def lstat():
            return type(
                "ReparseStat",
                (),
                {"st_file_attributes": stat.FILE_ATTRIBUTE_REPARSE_POINT},
            )()

    assert checkpoint_module._path_is_redirect(ReparseOnlyPath()) is True


def test_manifest_re_read_rejects_duplicate_json_keys_before_commit(
    runtime, service_factory
) -> None:
    class DuplicateKeyWriter(checkpoint_module.CheckpointService):
        def _write_manifest(self, path, document):
            super()._write_manifest(path, document)
            raw = path.read_text(encoding="utf-8")
            path.write_text(
                raw.replace(
                    '"checkpoint_id":',
                    '"checkpoint_id":"cp_00000000000000000000000000000000","checkpoint_id":',
                    1,
                ),
                encoding="utf-8",
            )

    service = service_factory(service_type=DuplicateKeyWriter)

    with pytest.raises(checkpoint_module.CheckpointError, match="duplicate key"):
        service.create_precompact(_precompact(cwd=runtime.paths.repo_root))

    assert [row["status"] for row in _checkpoint_rows(runtime)] == ["PREPARING"]


def test_corrupt_newest_committed_checkpoint_fails_without_older_fallback(
    runtime, service_factory
) -> None:
    service = service_factory()
    older = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    newest = service.create_precompact(_precompact(cwd=runtime.paths.repo_root))
    newest_path = _checkpoint_path(runtime.paths, newest.session_id, newest.checkpoint_id)
    newest_path.write_bytes(newest_path.read_bytes() + b" ")

    with pytest.raises(checkpoint_module.CheckpointError):
        service.latest_committed("session_a")

    assert older.checkpoint_id != newest.checkpoint_id
