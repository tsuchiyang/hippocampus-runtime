from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from codex_memory.paths import RuntimePathError
from codex_memory.retrieval.paths import RetrievalPaths


def _canonical_repo(repo: Path) -> Path:
    db = repo / ".codex-memory" / "db"
    db.mkdir(parents=True)
    (db / "memory.sqlite3").write_bytes(b"SQLite format 3\x00")
    return repo


def _directory_redirect(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (NotImplementedError, OSError):
        pass
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip("Windows cannot create a symlink or junction in this environment")


def _remove_redirect(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif getattr(path, "is_junction", lambda: False)():
        path.rmdir()


def _remove_fixture_canonical_layout(paths: RetrievalPaths) -> None:
    """Remove only the empty canonical layout created by _canonical_repo."""
    paths.canonical_database_path.unlink()
    paths.canonical_database_path.parent.rmdir()
    paths.runtime_root.rmdir()


def test_for_repo_preserves_spaced_root_and_creates_nothing(tmp_path: Path) -> None:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()

    paths = RetrievalPaths.for_repo(repo)

    assert paths.repo_root == repo.resolve()
    assert paths.retrieval_root == repo / ".codex-memory" / "retrieval"
    assert not (repo / ".codex-memory").exists()


def test_for_repo_uses_platform_case_semantics_for_repository_paths(tmp_path: Path) -> None:
    repo = tmp_path / "MiXeD Case Repo"
    repo.mkdir()
    differently_cased = tmp_path / "MIXED CASE REPO"

    actual = RetrievalPaths.for_repo(repo)
    alternate = RetrievalPaths.for_repo(differently_cased)

    if os.name == "nt":
        assert alternate.repo_root == actual.repo_root
        assert alternate.canonical_database_path == actual.canonical_database_path
    else:
        assert alternate.repo_root != actual.repo_root
        assert alternate.canonical_database_path != actual.canonical_database_path


def test_validate_existing_canonical_never_creates_retrieval_state(tmp_path: Path) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)

    paths.validate()

    assert not paths.retrieval_root.exists()


@pytest.mark.parametrize(
    "sidecar",
    ["memory.sqlite3-wal", "memory.sqlite3-shm", "memory.sqlite3-journal"],
)
def test_validation_rejects_nonregular_canonical_sidecars_before_sqlite_open(
    tmp_path: Path, sidecar: str
) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)
    (paths.canonical_database_path.parent / sidecar).mkdir()

    with pytest.raises(RuntimePathError, match="file|path|unsafe"):
        paths.validate()
    assert not paths.retrieval_root.exists()


def test_prepare_for_write_creates_only_the_retrieval_root_and_reserved_children(
    tmp_path: Path,
) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)

    paths.prepare_for_write()

    assert sorted(path.name for path in paths.retrieval_root.iterdir()) == [
        "generations",
        "temp",
    ]
    assert not paths.database_path.exists()


@pytest.mark.parametrize("component", ["runtime", "retrieval", "generations", "temp"])
def test_validation_rejects_directory_redirects_without_creating_external_content(
    tmp_path: Path, component: str
) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)
    external = tmp_path / "external"
    external.mkdir()
    if component == "runtime":
        _remove_fixture_canonical_layout(paths)
        link = paths.runtime_root
    elif component == "retrieval":
        paths.runtime_root.mkdir(exist_ok=True)
        link = paths.retrieval_root
    else:
        paths.runtime_root.mkdir(exist_ok=True)
        paths.retrieval_root.mkdir()
        link = getattr(paths, f"{component}_dir")
    _directory_redirect(link, external)
    try:
        with pytest.raises(RuntimePathError, match="unsafe|redirect|path"):
            paths.validate()
        assert list(external.iterdir()) == []
    finally:
        _remove_redirect(link)


@pytest.mark.parametrize("leaf", ["retrieval.sqlite3", "retrieval.sqlite3-wal", "retrieval.sqlite3-shm", "retrieval.sqlite3-journal"])
def test_validation_rejects_redirected_or_nonregular_retrieval_leaves(
    tmp_path: Path, leaf: str
) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)
    paths.prepare_for_write()
    target = tmp_path / "outside.sqlite3"
    target.write_bytes(b"outside")
    unsafe = paths.retrieval_root / leaf
    is_link = True
    try:
        unsafe.symlink_to(target)
    except (NotImplementedError, OSError):
        is_link = False
        unsafe.mkdir()
    try:
        with pytest.raises(RuntimePathError, match="unsafe|redirect|file"):
            paths.validate()
        assert target.read_bytes() == b"outside"
    finally:
        if is_link:
            unsafe.unlink()
        else:
            unsafe.rmdir()


def test_validation_rejects_a_non_directory_retrieval_root(tmp_path: Path) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)
    paths.runtime_root.mkdir(exist_ok=True)
    paths.retrieval_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimePathError, match="directory"):
        paths.validate()


def test_generic_reparse_predicate_is_rejected_without_platform_filesystem_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _canonical_repo(tmp_path / "repository")
    paths = RetrievalPaths.for_repo(repo)
    paths.runtime_root.mkdir(exist_ok=True)

    monkeypatch.setattr("codex_memory.retrieval.paths.path_is_redirect", lambda path: path == paths.retrieval_root)

    with pytest.raises(RuntimePathError, match="unsafe|redirect"):
        paths.validate()
