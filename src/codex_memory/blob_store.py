from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import re

from codex_memory.atomic import atomic_write_bytes
from codex_memory.paths import (
    RuntimePathError,
    ensure_safe_child_directory,
    require_safe_directory,
    require_safe_file,
)


_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class BlobRef:
    sha256: str
    byte_size: int
    relative_path: str


class BlobIntegrityError(RuntimeError):
    """Raised when an on-disk blob does not match its reference."""


class BlobStore:
    def __init__(self, blobs_dir: Path) -> None:
        self._blobs_dir = blobs_dir

    def put_bytes(self, data: bytes) -> BlobRef:
        digest = hashlib.sha256(data).hexdigest()
        ref = BlobRef(
            sha256=digest,
            byte_size=len(data),
            relative_path=f"sha256/{digest[:2]}/{digest}.bin",
        )
        path = self._safe_blob_path(ref, create_parents=True, require_exists=False)
        if path.exists():
            self.read(ref)
            return ref

        atomic_write_bytes(path, data)
        self.read(ref)
        return ref

    def path_for(self, ref: BlobRef) -> Path:
        if (
            not isinstance(ref.sha256, str)
            or _DIGEST_PATTERN.fullmatch(ref.sha256) is None
            or isinstance(ref.byte_size, bool)
            or not isinstance(ref.byte_size, int)
            or ref.byte_size < 0
        ):
            raise BlobIntegrityError("blob reference metadata is invalid")
        expected = f"sha256/{ref.sha256[:2]}/{ref.sha256}.bin"
        if ref.relative_path != expected:
            raise BlobIntegrityError("blob reference path does not match its digest")
        return self._blobs_dir / "sha256" / ref.sha256[:2] / f"{ref.sha256}.bin"

    def read(self, ref: BlobRef) -> bytes:
        path = self._safe_blob_path(ref, create_parents=False, require_exists=True)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        errors: list[str] = []
        if len(data) != ref.byte_size:
            errors.append(f"byte size mismatch: expected {ref.byte_size}, got {len(data)}")
        if digest != ref.sha256:
            errors.append(f"SHA-256 mismatch: expected {ref.sha256}, got {digest}")
        if errors:
            raise BlobIntegrityError(f"blob {'; '.join(errors)}")
        return data

    def read_sha256(self, digest: str) -> bytes:
        if _DIGEST_PATTERN.fullmatch(digest) is None:
            raise BlobIntegrityError("blob digest is invalid")
        path = self._blobs_dir / "sha256" / digest[:2] / f"{digest}.bin"
        size = path.stat().st_size
        return self.read(
            BlobRef(
                sha256=digest,
                byte_size=size,
                relative_path=f"sha256/{digest[:2]}/{digest}.bin",
            )
        )

    def _safe_blob_path(
        self, ref: BlobRef, *, create_parents: bool, require_exists: bool
    ) -> Path:
        path = self.path_for(ref)
        try:
            container = self._safe_blob_container(create=create_parents)
            sha_root = self._child_directory(
                container, "sha256", create=create_parents
            )
            prefix = self._child_directory(
                sha_root, ref.sha256[:2], create=create_parents
            )
            require_safe_file(path, prefix, require_exists=require_exists)
            return path
        except RuntimePathError as exc:
            raise BlobIntegrityError("unsafe blob storage path") from exc

    def _safe_blob_container(self, *, create: bool) -> Path:
        parent = self._blobs_dir.parent
        if parent.name == ".codex-memory":
            repository = parent.parent
            require_safe_directory(repository)
            runtime = self._child_directory(repository, parent.name, create=create)
            return self._child_directory(runtime, self._blobs_dir.name, create=create)
        require_safe_directory(parent)
        return self._child_directory(parent, self._blobs_dir.name, create=create)

    @staticmethod
    def _child_directory(parent: Path, name: str, *, create: bool) -> Path:
        child = parent / name
        if create:
            ensure_safe_child_directory(parent, child)
        else:
            require_safe_directory(child, parent)
        return child
