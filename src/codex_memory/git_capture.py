"""Bounded, byte-preserving Git state capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from codex_memory.blob_store import BlobStore


_GIT_TIMEOUT_SECONDS = 10
_ERROR_DETAIL_LIMIT = 2_000


class GitCaptureError(RuntimeError):
    """Raised when the repository state cannot be captured safely."""


@dataclass(frozen=True, slots=True)
class GitSnapshot:
    repo_root: str
    branch: str | None
    head: str | None
    is_unborn: bool
    status_blob_sha256: str
    working_diff_blob_sha256: str
    staged_diff_blob_sha256: str
    untracked_paths: list[str]


def capture_git_snapshot(repo_root: Path, blobs: BlobStore) -> GitSnapshot:
    """Capture Git metadata and raw diff bytes for one repository."""
    repository = Path(repo_root)
    if not repository.is_dir():
        raise GitCaptureError(f"repository path is not a directory: {repository}")

    top_level = _run_git(repository, ("rev-parse", "--show-toplevel"))
    _require_success(top_level, ("rev-parse", "--show-toplevel"))
    root_text = _decode_metadata(top_level.stdout)
    if not root_text:
        raise GitCaptureError("git rev-parse --show-toplevel returned an empty repository root")
    actual_root = Path(root_text).resolve()

    branch_result = _run_git(repository, ("symbolic-ref", "--short", "-q", "HEAD"))
    if branch_result.returncode == 0:
        branch = _decode_metadata(branch_result.stdout)
    elif branch_result.returncode == 1 and not branch_result.stdout and not branch_result.stderr:
        branch = None
    else:
        _require_success(branch_result, ("symbolic-ref", "--short", "-q", "HEAD"))
        raise AssertionError("unreachable")

    head_result = _run_git(repository, ("rev-parse", "--verify", "HEAD"))
    if head_result.returncode == 0:
        head = _decode_metadata(head_result.stdout)
        is_unborn = False
    elif head_result.returncode == 128:
        head = None
        is_unborn = True
    else:
        _require_success(head_result, ("rev-parse", "--verify", "HEAD"))
        raise AssertionError("unreachable")

    status = _successful_output(repository, ("status", "--porcelain=v1", "-z"))
    working_diff = _successful_output(repository, ("diff", "--binary", "--no-ext-diff"))
    staged_diff = _successful_output(
        repository, ("diff", "--cached", "--binary", "--no-ext-diff")
    )
    untracked_output = _successful_output(
        repository, ("ls-files", "--others", "--exclude-standard", "-z")
    )

    return GitSnapshot(
        repo_root=str(actual_root),
        branch=branch,
        head=head,
        is_unborn=is_unborn,
        status_blob_sha256=blobs.put_bytes(status).sha256,
        working_diff_blob_sha256=blobs.put_bytes(working_diff).sha256,
        staged_diff_blob_sha256=blobs.put_bytes(staged_diff).sha256,
        untracked_paths=_untracked_paths(untracked_output),
    )


def _run_git(repo_root: Path, arguments: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    command = ["git", *arguments]
    try:
        return subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            check=False,
            shell=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _bounded_detail(exc.stderr)
        raise GitCaptureError(_failure_message(command, "timed out", detail)) from exc
    except OSError as exc:
        detail = _bounded_detail(str(exc))
        raise GitCaptureError(_failure_message(command, "could not start", detail)) from exc


def _successful_output(repo_root: Path, arguments: tuple[str, ...]) -> bytes:
    result = _run_git(repo_root, arguments)
    _require_success(result, arguments)
    return result.stdout


def _require_success(result: subprocess.CompletedProcess[bytes], arguments: tuple[str, ...]) -> None:
    if result.returncode != 0:
        raise GitCaptureError(
            _failure_message(["git", *arguments], f"exited {result.returncode}", result.stderr)
        )


def _failure_message(command: list[str], reason: str, detail: bytes | str | None) -> str:
    suffix = _bounded_detail(detail)
    if suffix:
        return f"{' '.join(command)} {reason}: {suffix}"
    return f"{' '.join(command)} {reason}"


def _bounded_detail(detail: bytes | str | None) -> str:
    if detail is None:
        return ""
    if isinstance(detail, bytes):
        text = detail.decode("utf-8", errors="surrogateescape")
    else:
        text = detail
    return text.rstrip("\r\n")[:_ERROR_DETAIL_LIMIT]


def _decode_metadata(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogateescape").rstrip("\r\n")


def _untracked_paths(data: bytes) -> list[str]:
    return sorted(
        _decode_metadata(path).replace("\\", "/")
        for path in data.split(b"\0")
        if path
    )
