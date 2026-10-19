from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

from codex_memory import paths as paths_module
from codex_memory.blob_store import BlobIntegrityError, BlobRef, BlobStore
from codex_memory.cli import _construct_runtime


RuntimePaths = paths_module.RuntimePaths
RuntimePathError = paths_module.RuntimePathError


def _create_directory_redirect(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (NotImplementedError, OSError):
        if os.name != "nt":
            pytest.skip("directory symlinks are unavailable")

    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
        shell=False,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("directory junctions are unavailable")


def _remove_directory_redirect(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif getattr(link, "is_junction", lambda: False)():
        link.rmdir()


@pytest.mark.parametrize(
    "redirected_component",
    ["runtime", "database", "blobs", "blob-sha256", "checkpoints", "emergency"],
)
def test_runtime_construction_rejects_preexisting_directory_redirect_before_sqlite_open(
    tmp_path: Path, redirected_component: str
) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    external = tmp_path / f"external-{redirected_component}"
    external.mkdir()
    if redirected_component == "runtime":
        link = paths.runtime_root
    else:
        paths.runtime_root.mkdir()
        components = {
            "database": paths.db_path.parent,
            "blobs": paths.blobs_dir,
            "checkpoints": paths.checkpoints_dir,
            "emergency": paths.emergency_dir,
        }
        if redirected_component == "blob-sha256":
            paths.blobs_dir.mkdir()
            link = paths.blob_sha256_dir
        else:
            link = components[redirected_component]
    _create_directory_redirect(link, external)

    runtime = None
    try:
        try:
            runtime = _construct_runtime(tmp_path)
        except Exception as exc:
            assert isinstance(exc, paths_module.RuntimePathError)
            pass
        else:
            pytest.fail("runtime construction followed a pre-existing redirect")
        assert list(external.iterdir()) == []
    finally:
        if runtime is not None:
            runtime.connection.close()
        _remove_directory_redirect(link)


def test_runtime_construction_rejects_preexisting_database_leaf_redirect(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    paths.ensure()
    external = tmp_path / "external-database-file"
    external.mkdir()
    target = external / "target.sqlite3"
    original = b"not a runtime database"
    target.write_bytes(original)
    directory_redirect = False
    try:
        paths.db_path.symlink_to(target)
    except (NotImplementedError, OSError):
        target.unlink()
        _create_directory_redirect(paths.db_path, external)
        directory_redirect = True

    try:
        with pytest.raises(RuntimePathError, match="file|redirect|path"):
            _construct_runtime(tmp_path)
        if directory_redirect:
            assert list(external.iterdir()) == []
        else:
            assert target.read_bytes() == original
            assert list(external.iterdir()) == [target]
    finally:
        if directory_redirect:
            _remove_directory_redirect(paths.db_path)
        else:
            paths.db_path.unlink()


@pytest.mark.parametrize("redirected_level", ["sha256", "prefix"])
def test_blob_write_rejects_real_digest_prefix_redirect_without_external_write(
    tmp_path: Path, redirected_level: str
) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    paths.ensure()
    sha_root = paths.blobs_dir / "sha256"
    sha_root.mkdir(exist_ok=True)
    data = b"sensitive payload"
    digest = hashlib.sha256(data).hexdigest()
    external = tmp_path / "external-blobs"
    external.mkdir()
    if redirected_level == "sha256":
        sha_root.rmdir()
        link = sha_root
    else:
        link = sha_root / digest[:2]
    _create_directory_redirect(link, external)
    store = BlobStore(paths.blobs_dir)

    try:
        with pytest.raises(BlobIntegrityError, match="unsafe|redirect|path"):
            store.put_bytes(data)
        assert list(external.iterdir()) == []
    finally:
        _remove_directory_redirect(link)


def test_blob_read_rejects_real_leaf_redirect(tmp_path: Path) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    paths.ensure()
    data = b"external sensitive payload"
    digest = hashlib.sha256(data).hexdigest()
    ref = BlobRef(
        sha256=digest,
        byte_size=len(data),
        relative_path=f"sha256/{digest[:2]}/{digest}.bin",
    )
    prefix = paths.blob_sha256_dir / digest[:2]
    prefix.mkdir()
    external = tmp_path / "external-blob-file"
    external.mkdir()
    target = external / "target.bin"
    target.write_bytes(data)
    leaf = prefix / f"{digest}.bin"
    directory_redirect = False
    try:
        leaf.symlink_to(target)
    except (NotImplementedError, OSError):
        target.unlink()
        _create_directory_redirect(leaf, external)
        directory_redirect = True

    try:
        with pytest.raises(BlobIntegrityError, match="unsafe|redirect|path"):
            BlobStore(paths.blobs_dir).read(ref)
        if directory_redirect:
            assert list(external.iterdir()) == []
        else:
            assert target.read_bytes() == data
    finally:
        if directory_redirect:
            _remove_directory_redirect(leaf)
        else:
            leaf.unlink()


def test_blob_reference_must_match_its_content_addressed_path(tmp_path: Path) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    paths.ensure()
    outside = paths.runtime_root / "outside.bin"
    outside.write_bytes(b"")
    ref = BlobRef(
        sha256="0" * 64,
        byte_size=0,
        relative_path="../outside.bin",
    )

    with pytest.raises(BlobIntegrityError, match="path"):
        BlobStore(paths.blobs_dir).read(ref)


def test_runtime_paths_reject_generic_windows_reparse_attribute() -> None:
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
                {"st_file_attributes": 0x400},
            )()

    assert paths_module.path_is_redirect(ReparseOnlyPath()) is True
