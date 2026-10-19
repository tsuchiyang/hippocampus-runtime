from __future__ import annotations

import math
import hashlib
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_memory.blob_store import BlobRef, BlobStore
from codex_memory.paths import RuntimePaths


def _transcript_api():
    try:
        return import_module("codex_memory.transcript")
    except ModuleNotFoundError as error:
        pytest.fail(f"Transcript capture module is not implemented: {error}")


def _store(tmp_path: Path) -> BlobStore:
    return BlobStore(RuntimePaths.for_repo(tmp_path).blobs_dir)


def _stat(size: int, mtime_ns: int) -> SimpleNamespace:
    return SimpleNamespace(st_size=size, st_mtime_ns=mtime_ns)


def test_none_path_is_unavailable() -> None:
    transcript = _transcript_api()

    captured = transcript.capture_transcript(None, _store(Path.cwd()))

    assert captured.status == "unavailable"
    assert captured.source_path is None
    assert captured.blob_sha256 is None
    assert captured.byte_size is None


def test_initially_missing_path_is_unavailable_and_preserves_spelling(tmp_path: Path) -> None:
    transcript = _transcript_api()
    path = tmp_path / "absent transcript.jsonl"

    captured = transcript.capture_transcript(path, _store(tmp_path))

    assert captured.status == "unavailable"
    assert captured.source_path == str(path)
    assert captured.blob_sha256 is None
    assert captured.byte_size is None


def test_invalid_utf8_nul_and_crlf_bytes_are_preserved_exactly(tmp_path: Path) -> None:
    transcript = _transcript_api()
    path = tmp_path / "rollout.jsonl"
    payload = b"\xff\x00first\r\nsecond\x80\r\n"
    path.write_bytes(payload)
    store = _store(tmp_path)

    captured = transcript.capture_transcript(path, store)

    assert captured.status == "captured"
    assert captured.source_path == str(path)
    assert captured.byte_size == 18
    assert captured.blob_sha256 == hashlib.sha256(payload).hexdigest()
    assert store.read(
        BlobRef(
            sha256=captured.blob_sha256,
            byte_size=captured.byte_size,
            relative_path=f"sha256/{captured.blob_sha256[:2]}/{captured.blob_sha256}.bin",
        )
    ) == payload


@pytest.mark.parametrize("changed_field", ["size", "mtime"])
def test_changed_metadata_retries_then_captures_stable_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_field: str
) -> None:
    transcript = _transcript_api()
    path = tmp_path / "rollout.jsonl"
    payload = b"stable bytes"
    path.write_bytes(payload)
    original_stat = Path.stat
    first_after = _stat(len(payload) + 1, 10) if changed_field == "size" else _stat(len(payload), 11)
    observations = iter([_stat(len(payload), 10), first_after, _stat(len(payload), 12), _stat(len(payload), 12)])

    def changing_stat(self: Path, *args, **kwargs):
        if self == path:
            return next(observations)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", changing_stat)

    captured = transcript.capture_transcript(path, _store(tmp_path), delay_seconds=0)

    assert captured.status == "captured"
    assert captured.byte_size == len(payload)


def test_unstable_source_exhausts_retries_and_sleeps_only_between_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _transcript_api()
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(b"body")
    original_stat = Path.stat
    calls = 0
    sleeps: list[float] = []

    def unstable_stat(self: Path, *args, **kwargs):
        nonlocal calls
        if self == path:
            calls += 1
            return _stat(4, calls)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unstable_stat)
    monkeypatch.setattr("time.sleep", sleeps.append)

    with pytest.raises(transcript.TranscriptCaptureError):
        transcript.capture_transcript(path, _store(tmp_path), attempts=3, delay_seconds=0.125)

    assert sleeps == [0.125, 0.125]


def test_sharing_violation_retries_then_captures_stable_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = _transcript_api()
    path = tmp_path / "locked.jsonl"
    payload = b"released after sharing violation"
    path.write_bytes(payload)
    store = _store(tmp_path)
    original_stat = Path.stat
    calls = 0
    sleeps: list[float] = []

    def temporarily_denied_stat(self: Path, *args, **kwargs):
        nonlocal calls
        if self == path:
            calls += 1
            if calls == 1:
                raise PermissionError("sharing violation")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", temporarily_denied_stat)
    monkeypatch.setattr("time.sleep", sleeps.append)

    captured = transcript.capture_transcript(path, store, attempts=2, delay_seconds=0.125)

    assert captured.status == "captured"
    assert captured.byte_size == len(payload)
    assert captured.blob_sha256 == hashlib.sha256(payload).hexdigest()
    assert sleeps == [0.125]


def test_permission_failure_then_disappearance_exhausts_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _transcript_api()
    path = tmp_path / "locked-then-missing.jsonl"
    path.write_bytes(b"body")
    original_stat = Path.stat
    stat_attempts = 0
    sleeps: list[float] = []

    def denied_then_missing_stat(self: Path, *args, **kwargs):
        nonlocal stat_attempts
        if self == path:
            stat_attempts += 1
            if stat_attempts == 1:
                raise PermissionError("sharing violation")
            raise FileNotFoundError("gone after sharing violation")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_then_missing_stat)
    monkeypatch.setattr("time.sleep", sleeps.append)

    with pytest.raises(transcript.TranscriptCaptureError, match="could not be captured"):
        transcript.capture_transcript(path, _store(tmp_path), attempts=3, delay_seconds=0.125)

    assert stat_attempts == 3
    assert sleeps == [0.125, 0.125]


def test_disappearance_after_initial_observation_blocks_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _transcript_api()
    path = tmp_path / "vanishing.jsonl"
    path.write_bytes(b"body")
    original_stat = Path.stat
    original_read_bytes = Path.read_bytes
    stat_calls = 0

    def vanishing_stat(self: Path, *args, **kwargs):
        nonlocal stat_calls
        if self == path:
            stat_calls += 1
            if stat_calls == 1:
                return _stat(4, 1)
            raise FileNotFoundError("gone after observation")
        return original_stat(self, *args, **kwargs)

    def vanishing_read(self: Path) -> bytes:
        if self == path:
            raise FileNotFoundError("gone after observation")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "stat", vanishing_stat)
    monkeypatch.setattr(Path, "read_bytes", vanishing_read)

    with pytest.raises(transcript.TranscriptCaptureError, match="could not be captured"):
        transcript.capture_transcript(path, _store(tmp_path), attempts=2, delay_seconds=0)


def test_directory_read_failure_blocks_capture(tmp_path: Path) -> None:
    transcript = _transcript_api()
    directory = tmp_path / "transcript-directory"
    directory.mkdir()

    with pytest.raises(transcript.TranscriptCaptureError, match="could not be captured"):
        transcript.capture_transcript(directory, _store(tmp_path), attempts=1)


def test_blob_storage_failure_after_stable_read_blocks_capture(tmp_path: Path) -> None:
    transcript = _transcript_api()
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(b"stable")

    class FailingStore:
        def put_bytes(self, data: bytes):
            raise RuntimeError("blob write failed")

    with pytest.raises(transcript.TranscriptCaptureError, match="blob"):
        transcript.capture_transcript(path, FailingStore())


@pytest.mark.parametrize(
    ("attempts", "delay_seconds"),
    [
        (False, 0),
        (0, 0),
        (-1, 0),
        (1.5, 0),
        (1, False),
        (1, -0.1),
        (1, math.inf),
        (1, -math.inf),
        (1, math.nan),
    ],
)
def test_invalid_retry_options_are_rejected(attempts: object, delay_seconds: object) -> None:
    transcript = _transcript_api()

    with pytest.raises(ValueError):
        transcript.capture_transcript(None, _store(Path.cwd()), attempts=attempts, delay_seconds=delay_seconds)
