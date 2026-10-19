from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any


class RuntimePathError(RuntimeError):
    """Raised when durable runtime storage is redirected or escapes its root."""


def path_is_redirect(path: Any) -> bool:
    """Return true for symlinks and Windows junction/reparse redirects."""
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


def require_safe_directory(
    path: Path, trusted_parent: Path | None = None
) -> Path:
    """Resolve an ordinary directory and optionally prove exact parent containment."""
    if path_is_redirect(path):
        raise RuntimePathError(f"unsafe runtime directory redirect: {path.name}")
    try:
        if not path.is_dir():
            raise RuntimePathError(f"unsafe runtime path is not a directory: {path.name}")
        resolved = path.resolve(strict=True)
        if trusted_parent is not None:
            if path.parent != trusted_parent:
                raise RuntimePathError(f"unsafe runtime directory placement: {path.name}")
            parent = trusted_parent.resolve(strict=True)
            if resolved.parent != parent or not resolved.is_relative_to(parent):
                raise RuntimePathError(f"runtime directory escaped its parent: {path.name}")
        return resolved
    except RuntimePathError:
        raise
    except OSError as exc:
        raise RuntimePathError(
            f"runtime directory cannot be resolved safely: {path.name}"
        ) from exc


def ensure_safe_child_directory(parent: Path, child: Path) -> Path:
    """Create one exact child without recursively following an existing redirect."""
    require_safe_directory(parent)
    if child.parent != parent or path_is_redirect(child):
        raise RuntimePathError(f"unsafe runtime directory placement: {child.name}")
    try:
        child.mkdir()
    except FileExistsError:
        pass
    except OSError as exc:
        raise RuntimePathError(
            f"runtime directory could not be created safely: {child.name}"
        ) from exc
    return require_safe_directory(child, parent)


def require_safe_file(
    path: Path, trusted_parent: Path, *, require_exists: bool,
    allow_disappearance: bool = False,
) -> Path:
    """Validate a regular leaf without following a redirect outside its parent."""
    if allow_disappearance and require_exists:
        raise ValueError("a required file cannot allow disappearance")
    require_safe_directory(trusted_parent)
    if path.parent != trusted_parent or path_is_redirect(path):
        raise RuntimePathError(f"unsafe runtime file redirect: {path.name}")
    try:
        details = path.lstat()
    except FileNotFoundError:
        if require_exists:
            raise RuntimePathError(f"runtime file is missing: {path.name}")
        return path
    except OSError as exc:
        raise RuntimePathError(f"runtime file is unavailable: {path.name}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise RuntimePathError(f"unsafe runtime path is not a file: {path.name}")
    try:
        resolved = path.resolve(strict=True)
        parent = trusted_parent.resolve(strict=True)
    except FileNotFoundError as exc:
        if allow_disappearance and _missing_ephemeral_leaf(path, trusted_parent):
            return path
        raise RuntimePathError(f"runtime file cannot be resolved: {path.name}") from exc
    except OSError as exc:
        raise RuntimePathError(f"runtime file cannot be resolved: {path.name}") from exc
    if resolved.parent != parent or not resolved.is_relative_to(parent):
        if allow_disappearance and _missing_ephemeral_leaf(path, trusted_parent):
            return path
        raise RuntimePathError(f"runtime file escaped its parent: {path.name}")
    return resolved


def _missing_ephemeral_leaf(path: Path, trusted_parent: Path) -> bool:
    """Only an actually absent SQLite sidecar may disappear during validation."""
    try:
        path.lstat()  # A dangling redirect still exists and is not excused.
    except FileNotFoundError:
        require_safe_directory(trusted_parent)
        return True
    except OSError as exc:
        raise RuntimePathError(f"runtime file is unavailable: {path.name}") from exc
    return False


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    repo_root: Path
    runtime_root: Path
    db_path: Path
    blobs_dir: Path
    checkpoints_dir: Path
    emergency_dir: Path

    @classmethod
    def for_repo(cls, repo_root: Path) -> "RuntimePaths":
        root = repo_root.resolve()
        runtime = root / ".codex-memory"
        return cls(
            repo_root=root,
            runtime_root=runtime,
            db_path=runtime / "db" / "memory.sqlite3",
            blobs_dir=runtime / "blobs",
            checkpoints_dir=runtime / "checkpoints",
            emergency_dir=runtime / "emergency",
        )

    @property
    def blob_sha256_dir(self) -> Path:
        return self.blobs_dir / "sha256"

    def ensure(self) -> None:
        require_safe_directory(self.repo_root)
        ensure_safe_child_directory(self.repo_root, self.runtime_root)
        ensure_safe_child_directory(self.runtime_root, self.db_path.parent)
        self._validate_database_leaves()
        ensure_safe_child_directory(self.runtime_root, self.blobs_dir)
        ensure_safe_child_directory(self.blobs_dir, self.blob_sha256_dir)
        ensure_safe_child_directory(self.runtime_root, self.checkpoints_dir)
        ensure_safe_child_directory(self.runtime_root, self.emergency_dir)
        self.validate()

    def validate(self) -> None:
        require_safe_directory(self.repo_root)
        require_safe_directory(self.runtime_root, self.repo_root)
        require_safe_directory(self.db_path.parent, self.runtime_root)
        self._validate_database_leaves()
        require_safe_directory(self.blobs_dir, self.runtime_root)
        require_safe_directory(self.blob_sha256_dir, self.blobs_dir)
        require_safe_directory(self.checkpoints_dir, self.runtime_root)
        require_safe_directory(self.emergency_dir, self.runtime_root)

    def _validate_database_leaves(self) -> None:
        database_directory = self.db_path.parent
        require_safe_directory(database_directory, self.runtime_root)
        require_safe_file(self.db_path, database_directory, require_exists=False)
        for candidate in (
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
            Path(f"{self.db_path}-journal"),
        ):
            require_safe_file(candidate, database_directory, require_exists=False,
                              allow_disappearance=True)
