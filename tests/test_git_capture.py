from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess

import pytest

from codex_memory.blob_store import BlobStore


def _capture_module():
    try:
        return import_module("codex_memory.git_capture")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Git capture implementation is missing: {exc}")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout


def _init_repo(repo: Path, branch: str = "main") -> None:
    _git(repo.parent, "init", "--initial-branch", branch, str(repo))


def _commit(repo: Path, message: str) -> None:
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        message,
    )


def _blob_bytes(root: Path, sha256: str) -> bytes:
    return (root / "sha256" / sha256[:2] / f"{sha256}.bin").read_bytes()


def test_unborn_custom_branch_in_path_with_spaces_preserves_raw_status(tmp_path: Path) -> None:
    repo = tmp_path / "repository with spaces"
    repo.mkdir()
    _init_repo(repo, "topic/phase-0")
    (repo / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    blobs_root = tmp_path / "blobs"

    snapshot = _capture_module().capture_git_snapshot(repo, BlobStore(blobs_root))

    assert snapshot.repo_root == str(repo.resolve())
    assert snapshot.branch == "topic/phase-0"
    assert snapshot.head is None
    assert snapshot.is_unborn is True
    assert _blob_bytes(blobs_root, snapshot.status_blob_sha256) == _git(
        repo, "status", "--porcelain=v1", "-z"
    )


def test_dirty_repository_preserves_distinct_raw_diffs_and_sorted_untracked_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "dirty repo"
    repo.mkdir()
    _init_repo(repo)
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _commit(repo, "initial")
    tracked.write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    tracked.write_text("working\n", encoding="utf-8")
    (repo / "zeta.txt").write_text("z\n", encoding="utf-8")
    nested = repo / "nested"
    nested.mkdir()
    (nested / "entry.txt").write_text("nested\n", encoding="utf-8")
    (repo / "unicode-\u00df.txt").write_text("unicode\n", encoding="utf-8")
    blobs_root = tmp_path / "blobs"

    snapshot = _capture_module().capture_git_snapshot(repo, BlobStore(blobs_root))

    expected_status = _git(repo, "status", "--porcelain=v1", "-z")
    expected_working = _git(repo, "diff", "--binary", "--no-ext-diff")
    expected_staged = _git(repo, "diff", "--cached", "--binary", "--no-ext-diff")
    assert snapshot.head == _git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    assert snapshot.is_unborn is False
    assert _blob_bytes(blobs_root, snapshot.status_blob_sha256) == expected_status
    assert _blob_bytes(blobs_root, snapshot.working_diff_blob_sha256) == expected_working
    assert _blob_bytes(blobs_root, snapshot.staged_diff_blob_sha256) == expected_staged
    assert snapshot.working_diff_blob_sha256 != snapshot.staged_diff_blob_sha256
    assert snapshot.untracked_paths == ["nested/entry.txt", "unicode-\u00df.txt", "zeta.txt"]


def test_detached_head_has_no_branch_but_keeps_head(tmp_path: Path) -> None:
    repo = tmp_path / "detached"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _commit(repo, "initial")
    expected_head = _git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    _git(repo, "checkout", "--detach")

    snapshot = _capture_module().capture_git_snapshot(repo, BlobStore(tmp_path / "blobs"))

    assert snapshot.branch is None
    assert snapshot.head == expected_head
    assert snapshot.is_unborn is False


@pytest.mark.parametrize("path_kind", ["missing", "file", "directory"])
def test_non_repository_or_file_path_raises_git_capture_error(
    tmp_path: Path, path_kind: str
) -> None:
    path = tmp_path / "not a repository"
    if path_kind == "file":
        path.write_text("not a directory", encoding="utf-8")
    elif path_kind == "directory":
        path.mkdir()

    module = _capture_module()
    with pytest.raises(module.GitCaptureError):
        module.capture_git_snapshot(path, BlobStore(tmp_path / "blobs"))


def test_timeout_and_unexpected_return_errors_bound_diagnostic_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    module = _capture_module()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr=b"x" * 2_500)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(module.GitCaptureError) as timeout_error:
        module.capture_git_snapshot(repo, BlobStore(tmp_path / "timeout-blobs"))
    assert "x" * 2_000 in str(timeout_error.value)
    assert "x" * 2_001 not in str(timeout_error.value)

    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 3, stdout=b"", stderr=b"z" * 2_500)

    monkeypatch.setattr(module.subprocess, "run", failed)
    with pytest.raises(module.GitCaptureError) as return_error:
        module.capture_git_snapshot(repo, BlobStore(tmp_path / "return-blobs"))
    assert "z" * 2_000 in str(return_error.value)
    assert "z" * 2_001 not in str(return_error.value)


def test_symbolic_ref_error_with_stderr_is_not_treated_as_detached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    module = _capture_module()
    outputs = [
        subprocess.CompletedProcess(
            ["git", "rev-parse", "--show-toplevel"],
            0,
            stdout=str(repo.resolve()).encode("utf-8") + b"\n",
            stderr=b"",
        ),
        subprocess.CompletedProcess(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            1,
            stdout=b"",
            stderr=b"fatal symbolic-ref error",
        ),
    ]

    def controlled_run(command, **kwargs):
        return outputs.pop(0)

    monkeypatch.setattr(module.subprocess, "run", controlled_run)
    with pytest.raises(module.GitCaptureError, match="fatal symbolic-ref error"):
        module.capture_git_snapshot(repo, BlobStore(tmp_path / "blobs"))


def test_non_utf8_payloads_are_stored_without_decoding_and_git_calls_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    module = _capture_module()
    expected_calls = [
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "symbolic-ref", "--short", "-q", "HEAD"],
        ["git", "rev-parse", "--verify", "HEAD"],
        ["git", "status", "--porcelain=v1", "-z"],
        ["git", "diff", "--binary", "--no-ext-diff"],
        ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    ]
    outputs = [
        str(repo.resolve()).encode("utf-8") + b"\n",
        b"main\n",
        b"a" * 40 + b"\n",
        b"\xff\x00",
        b"\x80working\n",
        b"\xfestaged\n",
        b"nested\\bad-\xff.txt\x00",
    ]
    observed: list[tuple[list[str], dict[str, object]]] = []

    def controlled_run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=outputs[len(observed) - 1], stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", controlled_run)
    blobs_root = tmp_path / "blobs"

    snapshot = module.capture_git_snapshot(repo, BlobStore(blobs_root))

    assert [command for command, _ in observed] == expected_calls
    assert all(
        kwargs == {
            "cwd": repo,
            "capture_output": True,
            "check": False,
            "shell": False,
            "timeout": 10,
        }
        for _, kwargs in observed
    )
    assert _blob_bytes(blobs_root, snapshot.status_blob_sha256) == b"\xff\x00"
    assert _blob_bytes(blobs_root, snapshot.working_diff_blob_sha256) == b"\x80working\n"
    assert _blob_bytes(blobs_root, snapshot.staged_diff_blob_sha256) == b"\xfestaged\n"
    assert snapshot.untracked_paths == ["nested/bad-\udcff.txt"]
