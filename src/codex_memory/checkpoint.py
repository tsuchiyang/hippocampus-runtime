"""Atomic, self-verifying compaction checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Iterator, Literal
from uuid import uuid4

from codex_memory.atomic import atomic_write_json
from codex_memory.blob_store import BlobStore
from codex_memory.event_store import EventStore
from codex_memory.git_capture import GitSnapshot, capture_git_snapshot
from codex_memory.hook_input import HookEnvelope, HookEvent
from codex_memory.paths import RuntimePaths
from codex_memory.transcript import TranscriptCapture, capture_transcript


_LOCK_ATTEMPTS = 20
_LOCK_DELAY_SECONDS = 0.05
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_ID_PATTERN = re.compile(r"cp_[0-9a-f]{32}\Z")


class CheckpointError(RuntimeError):
    """Raised when checkpoint evidence cannot be committed or verified safely."""


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    checkpoint_id: str
    status: Literal["COMMITTED"]
    session_id: str
    turn_id: str
    trigger: Literal["manual", "auto"]
    created_at: str
    raw_event_highwater: int
    transcript: TranscriptCapture
    git: GitSnapshot
    recovery_required: bool
    manifest_sha256: str


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _session_digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise CheckpointError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"{field} must be a non-empty string")
    return value


def _require_optional_string(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise CheckpointError(f"{field} must be a string or null")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"manifest contains duplicate key: {key}")
        result[key] = value
    return result


def _path_is_redirect(path: Path) -> bool:
    """Return true for symlinks and Windows junction/reparse directory redirects."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & reparse_flag)
    except FileNotFoundError:
        return False
    except OSError:
        return True


@contextmanager
def _immediate_transaction(connection: Any) -> Iterator[None]:
    """Commit a short transaction or leave the connection fully rolled back."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


class CheckpointService:
    def __init__(
        self,
        paths: RuntimePaths,
        events: EventStore,
        blobs: BlobStore,
        transcript_capture=capture_transcript,
        git_capture=capture_git_snapshot,
    ) -> None:
        self.paths = paths
        self.events = events
        self.blobs = blobs
        self.transcript_capture = transcript_capture
        self.git_capture = git_capture

    def create_precompact(self, envelope: HookEnvelope) -> CheckpointManifest:
        self._validate_precompact(envelope)
        try:
            self.events.append(envelope)
            with self._session_lock(envelope.session_id):
                checkpoint_id = f"cp_{uuid4().hex}"
                created_at = _utc_timestamp()
                self._insert_preparing(checkpoint_id, envelope, created_at)

                transcript = self.transcript_capture(envelope.transcript_path, self.blobs)
                self._verify_transcript(transcript)
                git = self.git_capture(envelope.cwd, self.blobs)
                self._verify_git(git)
                highwater = self.events.highwater(envelope.session_id)
                if isinstance(highwater, bool) or not isinstance(highwater, int) or highwater < 1:
                    raise CheckpointError("PreCompact event highwater is unavailable")

                body = self._manifest_body(
                    checkpoint_id, envelope, created_at, highwater, transcript, git
                )
                manifest_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
                document = {**body, "manifest_sha256": manifest_sha256}
                manifest_path = self._manifest_path(envelope.session_id, checkpoint_id)
                manifest_relpath = manifest_path.relative_to(
                    self.paths.checkpoints_dir
                ).as_posix()
                self._write_manifest(manifest_path, document)
                manifest = self._load_and_verify_manifest(
                    manifest_path,
                    expected_checkpoint_id=checkpoint_id,
                    expected_session_id=envelope.session_id,
                    expected_manifest_sha256=manifest_sha256,
                )
                self._commit_prepared(
                    checkpoint_id,
                    envelope.session_id,
                    highwater,
                    manifest_relpath,
                    manifest_sha256,
                )
                return manifest
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError("pre-compaction checkpoint creation failed") from exc

    def latest_committed(self, session_id: str) -> CheckpointManifest | None:
        _require_string(session_id, "session_id")
        try:
            row = self.events._connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE session_id = ? AND status = 'COMMITTED'
                ORDER BY checkpoint_seq DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return self._manifest_from_row(row)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError("committed checkpoint lookup failed") from exc

    def mark_postcompact(self, envelope: HookEnvelope) -> CheckpointManifest | None:
        self._validate_postcompact(envelope)
        try:
            self.events.append(envelope)
            connection = self.events._connection
            checkpoint_id: str | None = None
            with _immediate_transaction(connection):
                row = connection.execute(
                    """
                    SELECT checkpoint_id FROM checkpoints
                    WHERE session_id = ? AND status = 'COMMITTED'
                    ORDER BY checkpoint_seq DESC
                    LIMIT 1
                    """,
                    (envelope.session_id,),
                ).fetchone()
                if row is not None:
                    checkpoint_id = row["checkpoint_id"]
                    cursor = connection.execute(
                        """
                        UPDATE checkpoints
                        SET recovery_required = 1, recovery_delivered_at = NULL
                        WHERE checkpoint_id = ? AND session_id = ? AND status = 'COMMITTED'
                        """,
                        (checkpoint_id, envelope.session_id),
                    )
                    if cursor.rowcount != 1:
                        raise CheckpointError("PostCompact checkpoint transition was not exact")

            if checkpoint_id is None:
                return None

            marked_row = connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
            if marked_row is None:
                raise CheckpointError("marked checkpoint disappeared")
            return self._manifest_from_row(marked_row)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError("PostCompact checkpoint update failed") from exc

    def mark_recovery_delivered(self, checkpoint_id: str, session_id: str) -> None:
        if _CHECKPOINT_ID_PATTERN.fullmatch(
            _require_string(checkpoint_id, "checkpoint_id")
        ) is None:
            raise CheckpointError("checkpoint_id is invalid")
        _require_string(session_id, "session_id")
        connection = self.events._connection
        try:
            with _immediate_transaction(connection):
                cursor = connection.execute(
                    """
                    UPDATE checkpoints
                    SET recovery_required = 0, recovery_delivered_at = ?
                    WHERE checkpoint_id = ?
                      AND session_id = ?
                      AND status = 'COMMITTED'
                      AND recovery_required = 1
                    """,
                    (_utc_timestamp(), checkpoint_id, session_id),
                )
                if cursor.rowcount != 1:
                    raise CheckpointError("recovery delivery transition was not exact")
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError("recovery delivery transition failed") from exc

    @contextmanager
    def _session_lock(self, session_id: str) -> Iterator[None]:
        digest = _session_digest(_require_string(session_id, "session_id"))
        checkpoint_root = self._checkpoint_root(create=True)
        locks_directory = self._safe_child_directory(
            checkpoint_root, ".locks", create=True
        )
        lock_path = locks_directory / f"{digest}.lock"
        self._safe_leaf(lock_path, locks_directory, require_exists=False)
        with lock_path.open("a+b") as stream:
            self._safe_leaf(lock_path, locks_directory, require_exists=True)
            acquired = False
            for attempt in range(_LOCK_ATTEMPTS):
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    if attempt < _LOCK_ATTEMPTS - 1:
                        time.sleep(_LOCK_DELAY_SECONDS)
                else:
                    acquired = True
                    break
            if not acquired:
                raise CheckpointError("session checkpoint lock is contended")
            try:
                yield
            finally:
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                except OSError as exc:
                    raise CheckpointError("session checkpoint lock release failed") from exc

    def _insert_preparing(
        self, checkpoint_id: str, envelope: HookEnvelope, created_at: str
    ) -> None:
        connection = self.events._connection
        with _immediate_transaction(connection):
            connection.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, session_id, turn_id, trigger, status, created_at
                ) VALUES (?, ?, ?, ?, 'PREPARING', ?)
                """,
                (
                    checkpoint_id,
                    envelope.session_id,
                    envelope.turn_id,
                    envelope.trigger,
                    created_at,
                ),
            )

    def _commit_prepared(
        self,
        checkpoint_id: str,
        session_id: str,
        highwater: int,
        manifest_relpath: str,
        manifest_sha256: str,
    ) -> None:
        connection = self.events._connection
        with _immediate_transaction(connection):
            cursor = connection.execute(
                """
                UPDATE checkpoints
                SET status = 'COMMITTED',
                    committed_at = ?,
                    raw_event_highwater = ?,
                    manifest_relpath = ?,
                    manifest_sha256 = ?
                WHERE checkpoint_id = ? AND session_id = ? AND status = 'PREPARING'
                """,
                (
                    _utc_timestamp(),
                    highwater,
                    manifest_relpath,
                    manifest_sha256,
                    checkpoint_id,
                    session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise CheckpointError("PREPARING to COMMITTED transition was not exact")

    def _manifest_body(
        self,
        checkpoint_id: str,
        envelope: HookEnvelope,
        created_at: str,
        highwater: int,
        transcript: TranscriptCapture,
        git: GitSnapshot,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "status": "COMMITTED",
            "session_id": envelope.session_id,
            "turn_id": envelope.turn_id,
            "trigger": envelope.trigger,
            "created_at": created_at,
            "raw_event_highwater": highwater,
            "transcript": {
                "status": transcript.status,
                "source_path": transcript.source_path,
                "blob_sha256": transcript.blob_sha256,
                "byte_size": transcript.byte_size,
            },
            "git": {
                "repo_root": git.repo_root,
                "branch": git.branch,
                "head": git.head,
                "is_unborn": git.is_unborn,
                "status_blob_sha256": git.status_blob_sha256,
                "working_diff_blob_sha256": git.working_diff_blob_sha256,
                "staged_diff_blob_sha256": git.staged_diff_blob_sha256,
                "untracked_paths": git.untracked_paths,
            },
            "recovery_required": False,
        }

    def _checkpoint_root(self, *, create: bool) -> Path:
        try:
            repository_root = self.paths.repo_root.resolve(strict=True)
        except OSError as exc:
            raise CheckpointError("unsafe checkpoint path: repository root is unavailable") from exc
        runtime_root = self.paths.runtime_root
        if runtime_root.parent != self.paths.repo_root:
            raise CheckpointError("unsafe checkpoint path: runtime root is misplaced")
        self._validate_directory(runtime_root, repository_root)

        checkpoint_root = self.paths.checkpoints_dir
        if checkpoint_root.parent != runtime_root:
            raise CheckpointError("unsafe checkpoint path: checkpoint root is misplaced")
        if _path_is_redirect(checkpoint_root):
            raise CheckpointError("unsafe checkpoint path redirect: checkpoint root")
        if not checkpoint_root.exists():
            if not create:
                raise CheckpointError("unsafe checkpoint path: checkpoint root is missing")
            try:
                checkpoint_root.mkdir()
            except OSError as exc:
                raise CheckpointError(
                    "unsafe checkpoint path: checkpoint root could not be created"
                ) from exc
        self._validate_directory(checkpoint_root, runtime_root.resolve(strict=True))
        return checkpoint_root

    def _safe_child_directory(
        self, parent: Path, name: str, *, create: bool
    ) -> Path:
        child = parent / name
        if child.parent != parent or _path_is_redirect(child):
            raise CheckpointError(f"unsafe checkpoint path redirect: {name}")
        if not child.exists():
            if not create:
                raise CheckpointError(f"unsafe checkpoint path is missing: {name}")
            try:
                child.mkdir()
            except OSError as exc:
                raise CheckpointError(
                    f"unsafe checkpoint path could not be created: {name}"
                ) from exc
        self._validate_directory(child, parent.resolve(strict=True))
        return child

    @staticmethod
    def _validate_directory(path: Path, trusted_parent: Path) -> None:
        if _path_is_redirect(path):
            raise CheckpointError(f"unsafe checkpoint path redirect: {path.name}")
        if not path.is_dir():
            raise CheckpointError(f"unsafe checkpoint path is not a directory: {path.name}")
        try:
            resolved = path.resolve(strict=True)
            parent = trusted_parent.resolve(strict=True)
        except OSError as exc:
            raise CheckpointError(f"unsafe checkpoint path cannot be resolved: {path.name}") from exc
        if resolved.parent != parent or not resolved.is_relative_to(parent):
            raise CheckpointError(f"unsafe checkpoint path escaped its parent: {path.name}")

    @staticmethod
    def _safe_leaf(path: Path, parent: Path, *, require_exists: bool) -> Path:
        if path.parent != parent or _path_is_redirect(path):
            raise CheckpointError(f"unsafe checkpoint path redirect: {path.name}")
        try:
            trusted_parent = parent.resolve(strict=True)
        except OSError as exc:
            raise CheckpointError("unsafe checkpoint parent is unavailable") from exc
        if not path.exists():
            if require_exists:
                raise CheckpointError(f"unsafe checkpoint path is missing: {path.name}")
            resolved = path.resolve(strict=False)
            if resolved.parent != trusted_parent:
                raise CheckpointError(f"unsafe checkpoint path escaped its parent: {path.name}")
            return path
        if not path.is_file():
            raise CheckpointError(f"unsafe checkpoint path is not a regular file: {path.name}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise CheckpointError(f"unsafe checkpoint path cannot be resolved: {path.name}") from exc
        if resolved.parent != trusted_parent or not resolved.is_relative_to(trusted_parent):
            raise CheckpointError(f"unsafe checkpoint path escaped its parent: {path.name}")
        return resolved

    def _safe_manifest_path(
        self, path: Path, *, create_parent: bool, require_exists: bool
    ) -> Path:
        checkpoint_root = self._checkpoint_root(create=create_parent)
        session_directory = path.parent
        if (
            session_directory.parent != checkpoint_root
            or _DIGEST_PATTERN.fullmatch(session_directory.name) is None
            or not path.name.endswith(".json")
            or _CHECKPOINT_ID_PATTERN.fullmatch(path.stem) is None
        ):
            raise CheckpointError("unsafe checkpoint manifest path")
        safe_session_directory = self._safe_child_directory(
            checkpoint_root, session_directory.name, create=create_parent
        )
        return self._safe_leaf(
            safe_session_directory / path.name,
            safe_session_directory,
            require_exists=require_exists,
        )

    def _manifest_path(self, session_id: str, checkpoint_id: str) -> Path:
        if _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id) is None:
            raise CheckpointError("checkpoint_id is invalid")
        return (
            self.paths.checkpoints_dir
            / _session_digest(session_id)
            / f"{checkpoint_id}.json"
        )

    def _write_manifest(self, path: Path, document: dict[str, object]) -> None:
        safe_path = self._safe_manifest_path(
            path, create_parent=True, require_exists=False
        )
        atomic_write_json(safe_path, document)
        self._safe_manifest_path(path, create_parent=False, require_exists=True)

    def _load_and_verify_manifest(
        self,
        path: Path,
        *,
        expected_checkpoint_id: str,
        expected_session_id: str,
        expected_manifest_sha256: str,
    ) -> CheckpointManifest:
        try:
            safe_path = self._safe_manifest_path(
                path, create_parent=False, require_exists=True
            )
            raw = safe_path.read_bytes()
            document = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError("manifest could not be read as canonical UTF-8 JSON") from exc
        if not isinstance(document, dict):
            raise CheckpointError("manifest root must be an object")
        if raw != _canonical_json(document):
            raise CheckpointError("manifest bytes are not canonical JSON")
        expected_keys = {
            "schema_version",
            "checkpoint_id",
            "status",
            "session_id",
            "turn_id",
            "trigger",
            "created_at",
            "raw_event_highwater",
            "transcript",
            "git",
            "recovery_required",
            "manifest_sha256",
        }
        if set(document) != expected_keys:
            raise CheckpointError("manifest fields do not match schema version 1")
        stored_digest = _require_digest(document["manifest_sha256"], "manifest_sha256")
        body = dict(document)
        del body["manifest_sha256"]
        actual_digest = hashlib.sha256(_canonical_json(body)).hexdigest()
        if stored_digest != actual_digest or stored_digest != expected_manifest_sha256:
            raise CheckpointError("manifest self-hash verification failed")
        manifest = self._parse_manifest(document)
        if manifest.checkpoint_id != expected_checkpoint_id:
            raise CheckpointError("manifest checkpoint identity mismatch")
        if manifest.session_id != expected_session_id:
            raise CheckpointError("manifest session identity mismatch")
        self._verify_transcript(manifest.transcript)
        self._verify_git(manifest.git)
        return manifest

    def _parse_manifest(self, document: dict[str, Any]) -> CheckpointManifest:
        if document["schema_version"] != 1 or isinstance(document["schema_version"], bool):
            raise CheckpointError("unsupported manifest schema version")
        checkpoint_id = _require_string(document["checkpoint_id"], "checkpoint_id")
        if _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id) is None:
            raise CheckpointError("checkpoint_id is invalid")
        if document["status"] != "COMMITTED":
            raise CheckpointError("manifest is not committed")
        session_id = _require_string(document["session_id"], "session_id")
        turn_id = _require_string(document["turn_id"], "turn_id")
        trigger = document["trigger"]
        if trigger not in {"manual", "auto"}:
            raise CheckpointError("manifest trigger is invalid")
        created_at = _require_string(document["created_at"], "created_at")
        if not created_at.endswith("Z"):
            raise CheckpointError("created_at is not a UTC timestamp")
        highwater = document["raw_event_highwater"]
        if isinstance(highwater, bool) or not isinstance(highwater, int) or highwater < 1:
            raise CheckpointError("raw_event_highwater is invalid")
        if document["recovery_required"] is not False:
            raise CheckpointError("immutable manifest recovery_required must be false")

        transcript = self._parse_transcript(document["transcript"])
        git = self._parse_git(document["git"])
        return CheckpointManifest(
            schema_version=1,
            checkpoint_id=checkpoint_id,
            status="COMMITTED",
            session_id=session_id,
            turn_id=turn_id,
            trigger=trigger,
            created_at=created_at,
            raw_event_highwater=highwater,
            transcript=transcript,
            git=git,
            recovery_required=False,
            manifest_sha256=_require_digest(
                document["manifest_sha256"], "manifest_sha256"
            ),
        )

    def _parse_transcript(self, value: object) -> TranscriptCapture:
        if not isinstance(value, dict) or set(value) != {
            "status",
            "source_path",
            "blob_sha256",
            "byte_size",
        }:
            raise CheckpointError("transcript manifest fields are invalid")
        status = value["status"]
        source_path = _require_optional_string(value["source_path"], "transcript.source_path")
        if status == "captured":
            digest = _require_digest(value["blob_sha256"], "transcript.blob_sha256")
            byte_size = value["byte_size"]
            if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
                raise CheckpointError("transcript.byte_size is invalid")
            return TranscriptCapture("captured", source_path, digest, byte_size)
        if status == "unavailable":
            if value["blob_sha256"] is not None or value["byte_size"] is not None:
                raise CheckpointError("unavailable transcript must not reference a blob")
            return TranscriptCapture("unavailable", source_path, None, None)
        raise CheckpointError("transcript status is invalid")

    def _parse_git(self, value: object) -> GitSnapshot:
        expected_keys = {
            "repo_root",
            "branch",
            "head",
            "is_unborn",
            "status_blob_sha256",
            "working_diff_blob_sha256",
            "staged_diff_blob_sha256",
            "untracked_paths",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise CheckpointError("Git manifest fields are invalid")
        repo_root = _require_string(value["repo_root"], "git.repo_root")
        branch = _require_optional_string(value["branch"], "git.branch")
        head = _require_optional_string(value["head"], "git.head")
        is_unborn = value["is_unborn"]
        if not isinstance(is_unborn, bool):
            raise CheckpointError("git.is_unborn must be a boolean")
        if is_unborn != (head is None):
            raise CheckpointError("Git HEAD and unborn state are inconsistent")
        untracked = value["untracked_paths"]
        if not isinstance(untracked, list) or any(
            not isinstance(path, str) for path in untracked
        ):
            raise CheckpointError("git.untracked_paths must be a string list")
        return GitSnapshot(
            repo_root=repo_root,
            branch=branch,
            head=head,
            is_unborn=is_unborn,
            status_blob_sha256=_require_digest(
                value["status_blob_sha256"], "git.status_blob_sha256"
            ),
            working_diff_blob_sha256=_require_digest(
                value["working_diff_blob_sha256"], "git.working_diff_blob_sha256"
            ),
            staged_diff_blob_sha256=_require_digest(
                value["staged_diff_blob_sha256"], "git.staged_diff_blob_sha256"
            ),
            untracked_paths=list(untracked),
        )

    def _verify_transcript(self, transcript: object) -> None:
        if not isinstance(transcript, TranscriptCapture):
            raise CheckpointError("transcript capture returned an invalid value")
        if transcript.status == "unavailable":
            if transcript.blob_sha256 is not None or transcript.byte_size is not None:
                raise CheckpointError("unavailable transcript must not reference a blob")
            return
        if transcript.status != "captured":
            raise CheckpointError("transcript capture status is invalid")
        digest = _require_digest(transcript.blob_sha256, "transcript.blob_sha256")
        if (
            isinstance(transcript.byte_size, bool)
            or not isinstance(transcript.byte_size, int)
            or transcript.byte_size < 0
        ):
            raise CheckpointError("transcript.byte_size is invalid")
        self._verify_blob(digest, transcript.byte_size)

    def _verify_git(self, git: object) -> None:
        if not isinstance(git, GitSnapshot):
            raise CheckpointError("Git capture returned an invalid value")
        if not isinstance(git.repo_root, str) or not git.repo_root:
            raise CheckpointError("Git repository root is invalid")
        if not isinstance(git.is_unborn, bool) or git.is_unborn != (git.head is None):
            raise CheckpointError("Git HEAD and unborn state are inconsistent")
        if not isinstance(git.untracked_paths, list) or any(
            not isinstance(path, str) for path in git.untracked_paths
        ):
            raise CheckpointError("Git untracked paths are invalid")
        for field, digest_value in (
            ("git.status_blob_sha256", git.status_blob_sha256),
            ("git.working_diff_blob_sha256", git.working_diff_blob_sha256),
            ("git.staged_diff_blob_sha256", git.staged_diff_blob_sha256),
        ):
            self._verify_blob(_require_digest(digest_value, field))

    def _verify_blob(self, digest: str, byte_size: int | None = None) -> None:
        root = self.paths.blobs_dir.resolve()
        path = self.paths.blobs_dir / "sha256" / digest[:2] / f"{digest}.bin"
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise CheckpointError(f"referenced blob is unavailable: {digest}") from exc
        if path.is_symlink() or not path.is_file() or not resolved.is_relative_to(root):
            raise CheckpointError(f"referenced blob path is unsafe: {digest}")
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            raise CheckpointError(f"referenced blob cannot be read: {digest}") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise CheckpointError(f"referenced blob hash mismatch: {digest}")
        if byte_size is not None and len(data) != byte_size:
            raise CheckpointError(f"referenced blob size mismatch: {digest}")

    def _manifest_from_row(self, row: Any) -> CheckpointManifest:
        checkpoint_id = row["checkpoint_id"]
        session_id = row["session_id"]
        expected_path = self._manifest_path(session_id, checkpoint_id)
        expected_relpath = expected_path.relative_to(
            self.paths.checkpoints_dir
        ).as_posix()
        if row["manifest_relpath"] != expected_relpath:
            raise CheckpointError("checkpoint manifest path does not match its identity")
        row_digest = _require_digest(row["manifest_sha256"], "manifest_sha256")
        manifest = self._load_and_verify_manifest(
            expected_path,
            expected_checkpoint_id=checkpoint_id,
            expected_session_id=session_id,
            expected_manifest_sha256=row_digest,
        )
        if (
            row["status"] != "COMMITTED"
            or manifest.turn_id != row["turn_id"]
            or manifest.trigger != row["trigger"]
            or manifest.created_at != row["created_at"]
            or manifest.raw_event_highwater != row["raw_event_highwater"]
        ):
            raise CheckpointError("checkpoint manifest metadata does not match SQLite")
        recovery_required = row["recovery_required"]
        if recovery_required not in {0, 1}:
            raise CheckpointError("checkpoint recovery state is invalid")
        return replace(manifest, recovery_required=bool(recovery_required))

    @staticmethod
    def _validate_precompact(envelope: object) -> None:
        if (
            not isinstance(envelope, HookEnvelope)
            or envelope.event is not HookEvent.PRE_COMPACT
            or not envelope.turn_id
            or envelope.trigger not in {"manual", "auto"}
        ):
            raise CheckpointError("create_precompact requires a valid PreCompact envelope")

    @staticmethod
    def _validate_postcompact(envelope: object) -> None:
        if (
            not isinstance(envelope, HookEnvelope)
            or envelope.event is not HookEvent.POST_COMPACT
            or not envelope.turn_id
            or envelope.trigger not in {"manual", "auto"}
        ):
            raise CheckpointError("mark_postcompact requires a valid PostCompact envelope")
