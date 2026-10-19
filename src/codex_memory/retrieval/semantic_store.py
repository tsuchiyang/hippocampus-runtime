"""Immutable, self-verifying storage for one semantic generation.

This module deliberately has no provider dependency.  It owns only derived
SQLite bytes and their manifest, so deleting a completed generation is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
import uuid
import ctypes
from typing import Any, Callable, Iterable

from codex_memory.ledger_common import canonical_json
from codex_memory.paths import RuntimePathError, path_is_redirect, require_safe_directory, require_safe_file

from .semantic_paths import SemanticGenerationPaths
from .contracts import IndexHighwater


SEMANTIC_SCHEMA_VERSION = 1
_MAX_SEMANTIC_DATABASE_BYTES = 256 * 1024 * 1024
_MAX_SEMANTIC_MANIFEST_BYTES = 1024 * 1024
_MAX_SEMANTIC_SNAPSHOT_BYTES = _MAX_SEMANTIC_DATABASE_BYTES + _MAX_SEMANTIC_MANIFEST_BYTES
_SEMANTIC_SNAPSHOT_DEADLINE_NS = 5_000_000_000
_HASH = __import__("re").compile(r"\A[0-9a-f]{64}\Z")
_SEMANTIC_DDL = (
    "CREATE TABLE semantic_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE semantic_documents (
        ordinal INTEGER PRIMARY KEY, document_id TEXT NOT NULL UNIQUE, record_id TEXT NOT NULL,
        content_sha256 TEXT NOT NULL, text_sha256 TEXT NOT NULL)""",
    """CREATE TABLE semantic_dense_vectors (
        vector_seq INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL REFERENCES semantic_documents(document_id),
        content_sha256 TEXT NOT NULL, model_fingerprint TEXT NOT NULL, preprocessing_fingerprint TEXT NOT NULL,
        dimension INTEGER NOT NULL, vector_format TEXT NOT NULL CHECK (vector_format = 'f32le'),
        normalized INTEGER NOT NULL CHECK (normalized = 1), vector_blob BLOB NOT NULL, vector_sha256 TEXT NOT NULL,
        UNIQUE (document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, dimension))""",
    """CREATE TABLE semantic_sparse_postings (
        posting_seq INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL REFERENCES semantic_documents(document_id),
        content_sha256 TEXT NOT NULL, model_fingerprint TEXT NOT NULL, preprocessing_fingerprint TEXT NOT NULL,
        term TEXT NOT NULL, weight_hex TEXT NOT NULL, posting_sha256 TEXT NOT NULL,
        UNIQUE (document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, term))""",
    "CREATE INDEX semantic_dense_vectors_document_seq ON semantic_dense_vectors(document_id, vector_seq)",
    "CREATE INDEX semantic_sparse_postings_term_seq ON semantic_sparse_postings(term, posting_seq)",
)


class SemanticStoreError(RuntimeError):
    """The disposable semantic generation is unsafe, incomplete, or corrupt."""


_OWNED_STAGING: dict[Path, tuple[str, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int] | None]] = {}
_OWNED_STAGING_LOCK = threading.Lock()


def deterministic_semantic_generation_id(
    projection_generation_id: str,
    capabilities: Any,
    preprocessing_fingerprint: str,
    build_config_sha256: str,
) -> str:
    """Bind derived semantic bytes to an exact provider capability profile."""
    fingerprint = getattr(capabilities, "fingerprint", None)
    if not isinstance(projection_generation_id, str) or not projection_generation_id:
        raise SemanticStoreError("invalid projection generation")
    if not all(isinstance(value, str) and _HASH.fullmatch(value) for value in (fingerprint, preprocessing_fingerprint, build_config_sha256)):
        raise SemanticStoreError("invalid semantic generation fingerprint")
    return _semantic_generation_id_from_fingerprint(
        projection_generation_id, fingerprint, preprocessing_fingerprint, build_config_sha256
    )


def _semantic_generation_id_from_fingerprint(
    projection_generation_id: str, provider_fingerprint: str,
    preprocessing_fingerprint: str, build_config_sha256: str,
) -> str:
    preimage = {
        "build_config_sha256": build_config_sha256,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "projection_generation_id": projection_generation_id,
        "provider_capabilities_fingerprint": provider_fingerprint,
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
    }
    return "sem_" + _sha256_bytes(canonical_json(preimage).encode("utf-8"))[:32]


def _trigger(table: str, operation: str) -> str:
    return f"CREATE TRIGGER {table}_reject_{operation} BEFORE {operation.upper()} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only table'); END"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular(path: Path, parent: Path) -> None:
    require_safe_file(path, parent, require_exists=True)
    details = path.lstat()
    if path_is_redirect(path) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise SemanticStoreError("semantic generation leaf is unsafe")


def _leaf_identity(path: Path) -> tuple[int, int, int, int]:
    details = path.lstat()
    if path_is_redirect(path) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise SemanticStoreError("semantic generation leaf is unsafe")
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _directory_identity(path: Path) -> tuple[int, int, int, int]:
    details = path.lstat()
    if path_is_redirect(path) or not stat.S_ISDIR(details.st_mode):
        raise SemanticStoreError("semantic generation directory is unsafe")
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _directory_location_identity(path: Path) -> tuple[int, int]:
    """Return only stable directory identity fields for parent TOCTOU checks."""
    details = path.lstat()
    if path_is_redirect(path) or not stat.S_ISDIR(details.st_mode):
        raise SemanticStoreError("semantic generation directory is unsafe")
    return details.st_dev, details.st_ino


class _WindowsFileTime(ctypes.Structure):
    _fields_ = (("low", ctypes.c_ulong), ("high", ctypes.c_ulong))


class _WindowsFileInformation(ctypes.Structure):
    _fields_ = (
        ("attributes", ctypes.c_ulong),
        ("creation_time", _WindowsFileTime),
        ("last_access_time", _WindowsFileTime),
        ("last_write_time", _WindowsFileTime),
        ("volume_serial", ctypes.c_ulong),
        ("size_high", ctypes.c_ulong),
        ("size_low", ctypes.c_ulong),
        ("links", ctypes.c_ulong),
        ("file_index_high", ctypes.c_ulong),
        ("file_index_low", ctypes.c_ulong),
    )


@dataclass(frozen=True, slots=True)
class _HeldSemanticGeneration:
    parent_handle: Any
    directory_handle: Any
    database_handle: Any
    manifest_handle: Any
    parent_state: tuple[int, ...]
    directory_state: tuple[int, ...]
    database_state: tuple[int, ...]
    manifest_state: tuple[int, ...]
    database_bytes: bytearray
    manifest_bytes: bytearray


@cache
def _windows_kernel32() -> Any:
    """Reuse completed typed API setup, never handles, file state, or last-error.

    Publication happens only after every signature is configured. Concurrent
    cold calls may initialize separate complete objects; no I/O or validation
    result is cached, and a failed initialization remains retryable.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    )
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.GetFileInformationByHandle.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsFileInformation),
    )
    kernel32.GetFileInformationByHandle.restype = ctypes.c_int
    kernel32.GetFileSizeEx.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong))
    kernel32.GetFileSizeEx.restype = ctypes.c_int
    kernel32.SetFilePointerEx.argtypes = (
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_ulong,
    )
    kernel32.SetFilePointerEx.restype = ctypes.c_int
    kernel32.ReadFile.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
    )
    kernel32.ReadFile.restype = ctypes.c_int
    return kernel32


def _windows_handle_state(handle: Any) -> tuple[int, ...]:
    kernel32 = _windows_kernel32()
    information = _WindowsFileInformation()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        raise SemanticStoreError("semantic acceptance object identity cannot be read") from ctypes.WinError(ctypes.get_last_error())
    return (
        int(information.attributes),
        int(information.volume_serial),
        (int(information.file_index_high) << 32) | int(information.file_index_low),
        (int(information.size_high) << 32) | int(information.size_low),
        (int(information.last_write_time.high) << 32) | int(information.last_write_time.low),
        int(information.links),
    )


def _open_windows_acceptance_handle(path: Path, *, directory: bool) -> tuple[Any, tuple[int, ...]]:
    """Deny WRITE/DELETE sharing; this does not prevent new external hardlinks."""
    kernel32 = _windows_kernel32()
    desired_access = 0x80000000  # GENERIC_READ, including held directories
    share_mode = 0x00000001  # READ only: deny WRITE and DELETE for every held object
    flags = 0x00200000 | (0x02000000 if directory else 0)  # OPEN_REPARSE_POINT (+ BACKUP_SEMANTICS)
    handle = kernel32.CreateFileW(str(path), desired_access, share_mode, None, 3, flags, None)
    if handle == ctypes.c_void_p(-1).value:
        raise SemanticStoreError(f"semantic acceptance object cannot be held: {path.name}") from ctypes.WinError(ctypes.get_last_error())
    try:
        state = _windows_handle_state(handle)
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    attributes = state[0]
    is_directory = bool(attributes & 0x00000010)
    is_reparse = bool(attributes & 0x00000400)
    if is_reparse or is_directory != directory or (not directory and state[-1] != 1):
        kernel32.CloseHandle(handle)
        raise SemanticStoreError(f"semantic acceptance object is unsafe: {path.name}")
    return handle, state


def _read_windows_handle(handle: Any, *, max_bytes: int, deadline_ns: int) -> bytearray:
    """Read bytes from the already-held object; never reopen its pathname."""
    kernel32 = _windows_kernel32()
    size = ctypes.c_longlong()
    if not kernel32.GetFileSizeEx(handle, ctypes.byref(size)) or size.value < 0:
        raise SemanticStoreError("semantic acceptance leaf size cannot be read") from ctypes.WinError(ctypes.get_last_error())
    if size.value > max_bytes:
        raise SemanticStoreError("semantic acceptance leaf exceeds snapshot limit")
    position = ctypes.c_longlong()
    if not kernel32.SetFilePointerEx(handle, 0, ctypes.byref(position), 0):
        raise SemanticStoreError("semantic acceptance leaf cannot be rewound") from ctypes.WinError(ctypes.get_last_error())
    remaining = size.value
    offset = 0
    data = bytearray(size.value)
    while remaining:
        if time.monotonic_ns() > deadline_ns:
            raise SemanticStoreError("semantic acceptance snapshot deadline exceeded")
        requested = min(remaining, 1_048_576)
        buffer = (ctypes.c_char * requested).from_buffer(data, offset)
        count = ctypes.c_ulong()
        if not kernel32.ReadFile(handle, buffer, requested, ctypes.byref(count), None):
            raise SemanticStoreError("semantic acceptance leaf cannot be read") from ctypes.WinError(ctypes.get_last_error())
        if time.monotonic_ns() > deadline_ns:
            raise SemanticStoreError("semantic acceptance snapshot deadline exceeded")
        if count.value == 0:
            raise SemanticStoreError("semantic acceptance leaf changed while reading")
        offset += count.value
        remaining -= count.value
    return data


def _assert_held_semantic_generation_unchanged(held: _HeldSemanticGeneration) -> None:
    """Recheck the same objects immediately before the cache transaction commits."""
    observed = (
        _windows_handle_state(held.parent_handle),
        _windows_handle_state(held.directory_handle),
        _windows_handle_state(held.database_handle),
        _windows_handle_state(held.manifest_handle),
    )
    expected = (
        held.parent_state,
        held.directory_state,
        held.database_state,
        held.manifest_state,
    )
    if observed != expected or observed[2][-1] != 1 or observed[3][-1] != 1:
        raise SemanticStoreError("semantic generation changed before active commit")


@contextmanager
def _hold_semantic_generation_for_acceptance(paths: SemanticGenerationPaths):
    """Hold parents against DELETE and leaves against WRITE/DELETE until commit."""
    if os.name != "nt":
        raise SemanticStoreError("protected semantic generation acceptance is unavailable")
    handles: list[Any] = []
    try:
        parent_handle, parent_state = _open_windows_acceptance_handle(paths.generations_dir, directory=True)
        handles.append(parent_handle)
        directory_handle, directory_state = _open_windows_acceptance_handle(paths.directory, directory=True)
        handles.append(directory_handle)
        database_handle, database_state = _open_windows_acceptance_handle(paths.database_path, directory=False)
        handles.append(database_handle)
        manifest_handle, manifest_state = _open_windows_acceptance_handle(paths.manifest_path, directory=False)
        handles.append(manifest_handle)
        if {entry.name for entry in paths.directory.iterdir()} != {"semantic.sqlite3", "generation-manifest.json"}:
            raise SemanticStoreError("semantic generation has unexpected leaves")
        deadline_ns = time.monotonic_ns() + _SEMANTIC_SNAPSHOT_DEADLINE_NS
        manifest_bytes = _read_windows_handle(
            manifest_handle,
            max_bytes=_MAX_SEMANTIC_MANIFEST_BYTES,
            deadline_ns=deadline_ns,
        )
        database_budget = _MAX_SEMANTIC_SNAPSHOT_BYTES - len(manifest_bytes)
        if database_budget < 0:
            raise SemanticStoreError("semantic acceptance snapshot exceeds aggregate limit")
        database_bytes = _read_windows_handle(
            database_handle,
            max_bytes=min(_MAX_SEMANTIC_DATABASE_BYTES, database_budget),
            deadline_ns=deadline_ns,
        )
        if len(manifest_bytes) + len(database_bytes) > _MAX_SEMANTIC_SNAPSHOT_BYTES:
            raise SemanticStoreError("semantic acceptance snapshot exceeds aggregate limit")
        yield _HeldSemanticGeneration(
            parent_handle=parent_handle,
            directory_handle=directory_handle,
            database_handle=database_handle,
            manifest_handle=manifest_handle,
            parent_state=parent_state,
            directory_state=directory_state,
            database_state=database_state,
            manifest_state=manifest_state,
            database_bytes=database_bytes,
            manifest_bytes=manifest_bytes,
        )
    finally:
        if handles:
            kernel32 = _windows_kernel32()
            close_error = False
            for handle in reversed(handles):
                close_error = not bool(kernel32.CloseHandle(handle)) or close_error
            if close_error:
                raise SemanticStoreError("semantic acceptance handle cannot be closed")


def _connect_semantic_database(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
        connection.execute("PRAGMA query_only=ON")
    else:
        connection = sqlite3.connect(path, isolation_level=None)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def connect_semantic_database(paths: SemanticGenerationPaths, *, readonly: bool = False) -> sqlite3.Connection:
    """Open only a validated fixed semantic generation leaf; never arbitrary SQLite."""
    path = _generation_database_path(paths, require_exists=readonly)
    return _connect_semantic_database(path, readonly=readonly)


def _generation_database_path(paths: SemanticGenerationPaths, *, require_exists: bool = False) -> Path:
    """Accept only the fixed semantic.sqlite3 leaf of a generation object."""
    if not isinstance(paths, SemanticGenerationPaths):
        raise SemanticStoreError("semantic database requires generation paths")
    expected_directory = paths.generations_dir / paths.semantic_generation_id
    expected_marker = paths.generations_dir / f".{paths.semantic_generation_id}.owner"
    if (
        paths.directory != expected_directory
        or paths.ownership_marker_path != expected_marker
        or paths.ownership_marker_path.parent != paths.generations_dir
        or paths.database_path.name != "semantic.sqlite3" or paths.manifest_path.name != "generation-manifest.json"
    ):
        raise SemanticStoreError("invalid semantic generation paths")
    if paths.database_path.parent != paths.directory or paths.manifest_path.parent != paths.directory:
        raise SemanticStoreError("invalid semantic generation paths")
    if paths.database_path.name == "memory.sqlite3":
        raise SemanticStoreError("canonical database is not semantic storage")
    try:
        require_safe_directory(paths.generations_dir)
        require_safe_directory(paths.directory, paths.generations_dir)
        require_safe_file(paths.database_path, paths.directory, require_exists=require_exists)
    except RuntimePathError as exc:
        raise SemanticStoreError("semantic generation path is unsafe") from exc
    return paths.database_path


def initialize_semantic_database(paths: SemanticGenerationPaths) -> None:
    """Create an empty exact semantic database in one short write transaction."""
    path = _generation_database_path(paths)
    marker = paths.ownership_marker_path
    if marker.exists() or path_is_redirect(marker):
        if marker.exists():
            _safe_regular(marker, paths.generations_dir)
        raise SemanticStoreError("semantic staging marker already exists")
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.write(descriptor, token.encode("ascii")); os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise SemanticStoreError("semantic staging marker cannot be created") from exc
    connection = _connect_semantic_database(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SEMANTIC_DDL:
                connection.execute(statement)
            for table in ("semantic_schema_meta", "semantic_documents", "semantic_dense_vectors", "semantic_sparse_postings"):
                for operation in ("update", "delete"):
                    connection.execute(_trigger(table, operation))
            connection.execute("INSERT INTO semantic_schema_meta VALUES ('schema_version', ?)", (str(SEMANTIC_SCHEMA_VERSION),))
            connection.commit()
        except BaseException:
            connection.rollback(); raise
    finally:
        connection.close()
    with _OWNED_STAGING_LOCK:
        _OWNED_STAGING[paths.directory.resolve()] = (
            token, _directory_identity(paths.directory), _leaf_identity(marker), _leaf_identity(path), None,
        )


def append_semantic_document(connection: sqlite3.Connection, *, ordinal: int, document_id: str, record_id: str, content_sha256: str, text_sha256: str) -> None:
    if type(ordinal) is not int or ordinal < 0 or not all(isinstance(x, str) and x for x in (document_id, record_id)) or not _HASH.fullmatch(content_sha256) or not _HASH.fullmatch(text_sha256):
        raise SemanticStoreError("invalid semantic document")
    connection.execute("INSERT INTO semantic_documents VALUES (?, ?, ?, ?, ?)", (ordinal, document_id, record_id, content_sha256, text_sha256))


def append_dense_vector(connection: sqlite3.Connection, *, document_id: str, content_sha256: str, model_fingerprint: str, preprocessing_fingerprint: str, dimension: int, vector_blob: bytes, vector_sha256: str) -> None:
    if type(dimension) is not int or dimension < 1 or not isinstance(vector_blob, bytes) or len(vector_blob) != dimension * 4 or _sha256_bytes(vector_blob) != vector_sha256:
        raise SemanticStoreError("invalid dense vector")
    if not all(_HASH.fullmatch(value) for value in (content_sha256, model_fingerprint, preprocessing_fingerprint, vector_sha256)):
        raise SemanticStoreError("invalid dense vector hash")
    import struct
    values = struct.unpack(f"<{dimension}f", vector_blob)
    if not all(math.isfinite(value) for value in values):
        raise SemanticStoreError("invalid dense vector values")
    connection.execute("""INSERT INTO semantic_dense_vectors(document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, dimension, vector_format, normalized, vector_blob, vector_sha256) VALUES (?, ?, ?, ?, ?, 'f32le', 1, ?, ?)""", (document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, dimension, vector_blob, vector_sha256))


def append_sparse_posting(connection: sqlite3.Connection, *, document_id: str, content_sha256: str, model_fingerprint: str, preprocessing_fingerprint: str, term: str, weight_hex: str, posting_sha256: str) -> None:
    try: value = float.fromhex(weight_hex)
    except (TypeError, ValueError) as exc: raise SemanticStoreError("invalid sparse weight") from exc
    if not math.isfinite(value) or not isinstance(term, str) or not term or not all(_HASH.fullmatch(item) for item in (content_sha256, model_fingerprint, preprocessing_fingerprint, posting_sha256)):
        raise SemanticStoreError("invalid sparse posting")
    expected = _sha256_bytes(canonical_json({"content_sha256": content_sha256, "document_id": document_id, "model_fingerprint": model_fingerprint, "preprocessing_fingerprint": preprocessing_fingerprint, "term": term, "weight_hex": weight_hex}).encode("utf-8"))
    if expected != posting_sha256: raise SemanticStoreError("sparse posting hash mismatch")
    connection.execute("INSERT INTO semantic_sparse_postings(document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, term, weight_hex, posting_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)", (document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, term, weight_hex, posting_sha256))


@dataclass(frozen=True, slots=True)
class SemanticGenerationManifest:
    semantic_schema_version: int
    semantic_generation_id: str
    projection_generation_id: str
    canonical_highwater: dict[str, Any]
    indexed_highwater: dict[str, Any]
    document_map_sha256: str
    component_counts: dict[str, int]
    component_digests: dict[str, str]
    provider_fingerprint: str
    dense_model_fingerprint: str | None
    sparse_model_fingerprint: str | None
    reranker_model_fingerprint: str | None
    preprocessing_fingerprint: str
    build_config_sha256: str
    semantic_database_sha256: str
    validation_state: str
    created_at: str
    manifest_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {name: getattr(self, name) for name in self.__dataclass_fields__ if include_hash or name != "manifest_sha256"}
        return value

    def to_json_bytes(self) -> bytes:
        return canonical_json(self.to_dict()).encode("utf-8")


def _digest_rows(connection: sqlite3.Connection, table: str, columns: str, order_by: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(f"SELECT {columns} FROM {table} ORDER BY {order_by}"):
        digest.update(canonical_json(dict(row)).encode("utf-8")); digest.update(b"\n"); count += 1
    return count, digest.hexdigest()


def _build_semantic_manifest_from_connection(
    *,
    paths: SemanticGenerationPaths,
    connection: sqlite3.Connection,
    database_sha256: str,
    projection_generation_id: str,
    canonical_highwater: dict[str, Any],
    indexed_highwater: dict[str, Any],
    provider_fingerprint: str,
    dense_model_fingerprint: str | None,
    sparse_model_fingerprint: str | None,
    reranker_model_fingerprint: str | None,
    preprocessing_fingerprint: str,
    build_config_sha256: str,
    created_at: str,
) -> SemanticGenerationManifest:
    verify_semantic_database(connection)
    document_count, document_digest = _digest_rows(connection, "semantic_documents", "ordinal, document_id, record_id, content_sha256, text_sha256", "ordinal")
    dense_count, dense_digest = _digest_rows(connection, "semantic_dense_vectors", "vector_seq, document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, dimension, vector_format, normalized, vector_sha256", "vector_seq")
    sparse_count, sparse_digest = _digest_rows(connection, "semantic_sparse_postings", "posting_seq, document_id, content_sha256, model_fingerprint, preprocessing_fingerprint, term, weight_hex, posting_sha256", "posting_seq")
    fields = dict(semantic_schema_version=SEMANTIC_SCHEMA_VERSION, semantic_generation_id=paths.semantic_generation_id, projection_generation_id=projection_generation_id, canonical_highwater=_exact_highwater(canonical_highwater), indexed_highwater=_exact_highwater(indexed_highwater), document_map_sha256=document_digest, component_counts={"documents": document_count, "dense_vectors": dense_count, "sparse_postings": sparse_count}, component_digests={"dense_vectors": dense_digest, "sparse_postings": sparse_digest}, provider_fingerprint=provider_fingerprint, dense_model_fingerprint=dense_model_fingerprint, sparse_model_fingerprint=sparse_model_fingerprint, reranker_model_fingerprint=reranker_model_fingerprint, preprocessing_fingerprint=preprocessing_fingerprint, build_config_sha256=build_config_sha256, semantic_database_sha256=database_sha256, validation_state="validated", created_at=created_at)
    manifest_sha256 = _sha256_bytes(canonical_json(fields).encode("utf-8"))
    return SemanticGenerationManifest(**fields, manifest_sha256=manifest_sha256)


def build_semantic_manifest(*, paths: SemanticGenerationPaths, projection_generation_id: str, canonical_highwater: dict[str, Any], indexed_highwater: dict[str, Any], provider_fingerprint: str, dense_model_fingerprint: str | None, sparse_model_fingerprint: str | None, reranker_model_fingerprint: str | None, preprocessing_fingerprint: str, build_config_sha256: str, created_at: str) -> SemanticGenerationManifest:
    _safe_regular(paths.database_path, paths.directory)
    connection = _connect_semantic_database(paths.database_path, readonly=True)
    try:
        return _build_semantic_manifest_from_connection(
            paths=paths,
            connection=connection,
            database_sha256=_sha256_file(paths.database_path),
            projection_generation_id=projection_generation_id,
            canonical_highwater=canonical_highwater,
            indexed_highwater=indexed_highwater,
            provider_fingerprint=provider_fingerprint,
            dense_model_fingerprint=dense_model_fingerprint,
            sparse_model_fingerprint=sparse_model_fingerprint,
            reranker_model_fingerprint=reranker_model_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            build_config_sha256=build_config_sha256,
            created_at=created_at,
        )
    finally:
        connection.close()


def verify_semantic_database(connection: sqlite3.Connection) -> None:
    try:
        expected = {item.strip() for item in _SEMANTIC_DDL} | {_trigger(table, operation).strip() for table in ("semantic_schema_meta", "semantic_documents", "semantic_dense_vectors", "semantic_sparse_postings") for operation in ("update", "delete")}
        actual = {row["sql"].strip() for row in connection.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name != 'sqlite_sequence'")}
        if actual != expected or connection.execute("SELECT value FROM semantic_schema_meta WHERE key = 'schema_version'").fetchone()[0] != str(SEMANTIC_SCHEMA_VERSION): raise SemanticStoreError("semantic database schema is invalid")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or list(connection.execute("PRAGMA foreign_key_check")): raise SemanticStoreError("semantic database integrity is invalid")
        documents: dict[str, sqlite3.Row] = {}
        expected_ordinal = 0
        for row in connection.execute("SELECT * FROM semantic_documents ORDER BY ordinal"):
            if row["ordinal"] != expected_ordinal or not isinstance(row["document_id"], str) or not row["document_id"] or not isinstance(row["record_id"], str) or not row["record_id"] or not _HASH.fullmatch(row["content_sha256"]) or not _HASH.fullmatch(row["text_sha256"]):
                raise SemanticStoreError("semantic document identity is invalid")
            documents[row["document_id"]] = row; expected_ordinal += 1
        for row in connection.execute("SELECT * FROM semantic_dense_vectors ORDER BY vector_seq"):
            document = documents.get(row["document_id"])
            if document is None or row["content_sha256"] != document["content_sha256"] or row["vector_format"] != "f32le" or row["normalized"] != 1:
                raise SemanticStoreError("semantic dense identity is invalid")
            append_dense_vector_check(row)
        for row in connection.execute("SELECT * FROM semantic_sparse_postings ORDER BY posting_seq"):
            document = documents.get(row["document_id"])
            if document is None or row["content_sha256"] != document["content_sha256"] or not isinstance(row["term"], str) or not row["term"]:
                raise SemanticStoreError("semantic sparse identity is invalid")
            try:
                weight = float.fromhex(row["weight_hex"])
            except (TypeError, ValueError) as exc:
                raise SemanticStoreError("invalid sparse weight") from exc
            if not math.isfinite(weight) or weight.hex() != row["weight_hex"]:
                raise SemanticStoreError("invalid sparse weight")
            expected = _sha256_bytes(canonical_json({"content_sha256": row["content_sha256"], "document_id": row["document_id"], "model_fingerprint": row["model_fingerprint"], "preprocessing_fingerprint": row["preprocessing_fingerprint"], "term": row["term"], "weight_hex": row["weight_hex"]}).encode("utf-8"))
            if expected != row["posting_sha256"] or not all(_HASH.fullmatch(row[name]) for name in ("content_sha256", "model_fingerprint", "preprocessing_fingerprint", "posting_sha256")):
                raise SemanticStoreError("semantic sparse hash is invalid")
    except (sqlite3.Error, IndexError, TypeError, ValueError) as exc: raise SemanticStoreError("semantic database is corrupt") from exc


def append_dense_vector_check(row: sqlite3.Row) -> None:
    try:
        import struct
        values = struct.unpack(f"<{row['dimension']}f", row["vector_blob"])
    except (TypeError, ValueError, struct.error) as exc:
        raise SemanticStoreError("invalid dense vector") from exc
    valid = (
        type(row["dimension"]) is int and row["dimension"] > 0
        and len(row["vector_blob"]) == row["dimension"] * 4
        and _sha256_bytes(row["vector_blob"]) == row["vector_sha256"]
        and all(math.isfinite(value) for value in values)
        and math.isclose(math.fsum(value * value for value in values), 1.0, rel_tol=1e-5, abs_tol=1e-5)
        and all(_HASH.fullmatch(row[name]) for name in ("content_sha256", "model_fingerprint", "preprocessing_fingerprint", "vector_sha256"))
    )
    if not valid: raise SemanticStoreError("invalid dense vector")


def verify_semantic_generation(paths: SemanticGenerationPaths) -> SemanticGenerationManifest:
    try:
        paths.validate_leaves(); _safe_regular(paths.database_path, paths.directory); _safe_regular(paths.manifest_path, paths.directory)
        database_identity = _leaf_identity(paths.database_path)
        manifest_identity = _leaf_identity(paths.manifest_path)
        if {entry.name for entry in paths.directory.iterdir()} != {"semantic.sqlite3", "generation-manifest.json"}: raise SemanticStoreError("semantic generation has unexpected leaves")
        raw = paths.manifest_path.read_bytes(); value = json.loads(raw.decode("utf-8"))
        if canonical_json(value).encode("utf-8") != raw: raise SemanticStoreError("semantic manifest is not canonical")
        manifest = SemanticGenerationManifest(**value)
        _validate_manifest(manifest)
        if _sha256_bytes(canonical_json(manifest.to_dict(include_hash=False)).encode("utf-8")) != manifest.manifest_sha256: raise SemanticStoreError("semantic manifest self hash mismatch")
        rebuilt = build_semantic_manifest(paths=paths, projection_generation_id=manifest.projection_generation_id, canonical_highwater=manifest.canonical_highwater, indexed_highwater=manifest.indexed_highwater, provider_fingerprint=manifest.provider_fingerprint, dense_model_fingerprint=manifest.dense_model_fingerprint, sparse_model_fingerprint=manifest.sparse_model_fingerprint, reranker_model_fingerprint=manifest.reranker_model_fingerprint, preprocessing_fingerprint=manifest.preprocessing_fingerprint, build_config_sha256=manifest.build_config_sha256, created_at=manifest.created_at)
        if rebuilt != manifest: raise SemanticStoreError("semantic manifest database hash or count mismatch")
        if _leaf_identity(paths.database_path) != database_identity or _leaf_identity(paths.manifest_path) != manifest_identity:
            raise SemanticStoreError("semantic generation changed during verification")
        with _OWNED_STAGING_LOCK:
            owned = _OWNED_STAGING.get(paths.directory.resolve())
            if owned is not None:
                token, _, marker_identity, _, _ = owned
                if _leaf_identity(paths.ownership_marker_path) != marker_identity or paths.ownership_marker_path.read_text("ascii") != token:
                    raise SemanticStoreError("semantic staging marker changed")
                _OWNED_STAGING[paths.directory.resolve()] = (
                    token, _directory_identity(paths.directory), marker_identity,
                    database_identity, manifest_identity,
                )
        return manifest
    except (RuntimePathError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc: raise SemanticStoreError("semantic generation is unsafe or corrupt") from exc


def _verify_held_semantic_generation(
    paths: SemanticGenerationPaths,
    held: _HeldSemanticGeneration,
) -> SemanticGenerationManifest:
    """Verify manifest and SQLite content read from the exact held leaf objects."""
    raw = held.manifest_bytes
    value = json.loads(raw.decode("utf-8"))
    if canonical_json(value).encode("utf-8") != raw:
        raise SemanticStoreError("semantic manifest is not canonical")
    manifest = SemanticGenerationManifest(**value)
    _validate_manifest(manifest)
    if manifest.semantic_generation_id != paths.semantic_generation_id:
        raise SemanticStoreError("semantic manifest path identity mismatch")
    if _sha256_bytes(canonical_json(manifest.to_dict(include_hash=False)).encode("utf-8")) != manifest.manifest_sha256:
        raise SemanticStoreError("semantic manifest self hash mismatch")

    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.deserialize(held.database_bytes)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA query_only=ON")
        rebuilt = _build_semantic_manifest_from_connection(
            paths=paths,
            connection=connection,
            database_sha256=_sha256_bytes(held.database_bytes),
            projection_generation_id=manifest.projection_generation_id,
            canonical_highwater=manifest.canonical_highwater,
            indexed_highwater=manifest.indexed_highwater,
            provider_fingerprint=manifest.provider_fingerprint,
            dense_model_fingerprint=manifest.dense_model_fingerprint,
            sparse_model_fingerprint=manifest.sparse_model_fingerprint,
            reranker_model_fingerprint=manifest.reranker_model_fingerprint,
            preprocessing_fingerprint=manifest.preprocessing_fingerprint,
            build_config_sha256=manifest.build_config_sha256,
            created_at=manifest.created_at,
        )
    finally:
        connection.close()
    if rebuilt != manifest:
        raise SemanticStoreError("semantic manifest database hash or count mismatch")
    return manifest


def accept_semantic_generation(
    paths: SemanticGenerationPaths,
    acceptance_callback: Callable[
        [SemanticGenerationManifest, _ReadOnlySnapshotConnection, Callable[[], None]], None
    ],
) -> SemanticGenerationManifest:
    """Consume and hand off one exact verified snapshot while its objects are held."""
    _require_semantic_acceptance_capability()
    connection = None
    try:
        paths.validate_leaves()
        require_safe_directory(paths.generations_dir)
        require_safe_directory(paths.directory, paths.generations_dir)
        with _hold_semantic_generation_for_acceptance(paths) as held:
            manifest = _verify_held_semantic_generation(paths, held)
            connection = _connection_from_snapshot_bytes(held.database_bytes)
            acceptance_callback(
                manifest,
                connection,
                lambda: _assert_held_semantic_generation_unchanged(held),
            )
            return manifest
    except (RuntimePathError, OSError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
        raise SemanticStoreError("semantic generation acceptance is unsafe or corrupt") from exc
    finally:
        if connection is not None:
            connection.close()


def _require_semantic_acceptance_capability() -> None:
    """Require the reviewed Win32 stable-handle snapshot implementation."""
    if os.name != "nt":
        raise SemanticStoreError("semantic generation acceptance capability is unavailable")


class _ReadOnlySnapshotCursor:
    """Expose row consumption without leaking the underlying SQLite connection."""

    __slots__ = ("__cursor",)

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self.__cursor = cursor

    def fetchone(self):  # type: ignore[no-untyped-def]
        return self.__cursor.fetchone()

    def fetchall(self):  # type: ignore[no-untyped-def]
        return self.__cursor.fetchall()

    def __iter__(self) -> _ReadOnlySnapshotCursor:
        return self

    def __next__(self):  # type: ignore[no-untyped-def]
        return next(self.__cursor)

    def close(self) -> None:
        self.__cursor.close()


class _ReadOnlySnapshotConnection:
    """Narrow facade that is not a `sqlite3.Connection` dispatch target."""

    __slots__ = ("__connection",)

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.__connection = connection

    def execute(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> _ReadOnlySnapshotCursor:
        return _ReadOnlySnapshotCursor(self.__connection.execute(sql, parameters))

    def close(self) -> None:
        self.__connection.close()


def _connection_from_snapshot_bytes(
    database_bytes: bytes | bytearray,
) -> _ReadOnlySnapshotConnection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.deserialize(database_bytes)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA query_only=ON")
        denied_actions = {
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_DETACH,
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_INDEX,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_INDEX,
            sqlite3.SQLITE_CREATE_TEMP_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
            sqlite3.SQLITE_CREATE_TEMP_VIEW,
            sqlite3.SQLITE_CREATE_TRIGGER,
            sqlite3.SQLITE_CREATE_VIEW,
            sqlite3.SQLITE_DROP_INDEX,
            sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_INDEX,
            sqlite3.SQLITE_DROP_TEMP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_TRIGGER,
            sqlite3.SQLITE_DROP_TEMP_VIEW,
            sqlite3.SQLITE_DROP_TRIGGER,
            sqlite3.SQLITE_DROP_VIEW,
            sqlite3.SQLITE_ALTER_TABLE,
            sqlite3.SQLITE_REINDEX,
            sqlite3.SQLITE_ANALYZE,
        }
        read_only_pragmas = frozenset(
            {"database_list", "journal_mode", "query_only", "user_version"}
        )

        def read_only_authorizer(
            action: int,
            _argument1: str | None,
            argument2: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action in denied_actions:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA:
                pragma = (_argument1 or "").casefold()
                if argument2 is None and pragma in read_only_pragmas:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            if (
                action == sqlite3.SQLITE_FUNCTION
                and (argument2 or "").casefold() == "load_extension"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(read_only_authorizer)
        return _ReadOnlySnapshotConnection(connection)
    except BaseException:
        connection.close()
        raise


@contextmanager
def open_verified_semantic_snapshot(paths: SemanticGenerationPaths):
    """Read an existing generation from the exact verified held byte snapshot.

    This protects the data consumed by readers, not new-generation activation.
    The readonly in-memory database never reopens the leaf pathname.
    """
    connection = None
    try:
        _generation_database_path(paths, require_exists=True)
        paths.validate_leaves()
        with _hold_semantic_generation_for_acceptance(paths) as held:
            manifest = _verify_held_semantic_generation(paths, held)
            _assert_held_semantic_generation_unchanged(held)
            connection = _connection_from_snapshot_bytes(held.database_bytes)
            yield manifest, connection
    except (RuntimePathError, OSError, TypeError, ValueError, sqlite3.Error) as exc:
        raise SemanticStoreError("semantic snapshot is unsafe or corrupt") from exc
    finally:
        if connection is not None:
            connection.close()


def _validate_manifest(manifest: SemanticGenerationManifest) -> None:
    if manifest.semantic_schema_version != SEMANTIC_SCHEMA_VERSION or not isinstance(manifest.semantic_generation_id, str) or not manifest.semantic_generation_id.startswith("sem_") or not isinstance(manifest.projection_generation_id, str) or not manifest.projection_generation_id.startswith("gen_"):
        raise SemanticStoreError("semantic manifest identity is invalid")
    _exact_highwater(manifest.canonical_highwater); _exact_highwater(manifest.indexed_highwater)
    if manifest.validation_state != "validated" or not isinstance(manifest.created_at, str) or not manifest.created_at.endswith("Z"):
        raise SemanticStoreError("semantic manifest state is invalid")
    if set(manifest.component_counts) != {"documents", "dense_vectors", "sparse_postings"} or any(type(value) is not int or value < 0 for value in manifest.component_counts.values()):
        raise SemanticStoreError("semantic manifest counts are invalid")
    if set(manifest.component_digests) != {"dense_vectors", "sparse_postings"}:
        raise SemanticStoreError("semantic manifest digests are invalid")
    hashes = [manifest.document_map_sha256, manifest.provider_fingerprint, manifest.preprocessing_fingerprint, manifest.build_config_sha256, manifest.semantic_database_sha256, manifest.manifest_sha256, *manifest.component_digests.values()]
    hashes.extend(value for value in (manifest.dense_model_fingerprint, manifest.sparse_model_fingerprint, manifest.reranker_model_fingerprint) if value is not None)
    if any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in hashes):
        raise SemanticStoreError("semantic manifest hash is invalid")
    if manifest.semantic_generation_id != _semantic_generation_id_from_fingerprint(
        manifest.projection_generation_id, manifest.provider_fingerprint,
        manifest.preprocessing_fingerprint, manifest.build_config_sha256,
    ):
        raise SemanticStoreError("semantic manifest generation id is invalid")


def _exact_highwater(value: Any) -> dict[str, int]:
    try:
        if not isinstance(value, dict):
            raise ValueError
        highwater = IndexHighwater.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise SemanticStoreError("semantic manifest highwater is invalid") from exc
    if highwater.to_dict() != value:
        raise SemanticStoreError("semantic manifest highwater is invalid")
    return highwater.to_dict()


def _reuse_existing_semantic_generation(
    paths: SemanticGenerationPaths, expected: SemanticGenerationManifest,
) -> None:
    """Validate identical held bytes for store-only reuse, without a handoff."""
    with open_verified_semantic_snapshot(paths) as (existing, _connection):
        if existing != expected:
            raise SemanticStoreError("refusing to overwrite semantic generation")


def install_semantic_generation(
    staged: SemanticGenerationPaths, final: SemanticGenerationPaths, *,
    failure_injector: Any | None = None,
    acceptance_callback: Callable[
        [SemanticGenerationManifest, _ReadOnlySnapshotConnection, Callable[[], None]], None
    ] | None = None,
) -> None:
    """Atomically install a verified owned generation, never overwriting a peer."""
    if acceptance_callback is not None:
        _require_semantic_acceptance_capability()
    if staged.semantic_generation_id != final.semantic_generation_id or staged.directory.parent == final.directory.parent:
        raise SemanticStoreError("invalid semantic generation installation")
    staged_manifest = verify_semantic_generation(staged)
    with _OWNED_STAGING_LOCK:
        owned = _OWNED_STAGING.get(staged.directory.resolve())
    if owned is None:
        raise SemanticStoreError("semantic staging directory is not owned")
    token, directory_identity, marker_identity, database_identity, manifest_identity = owned
    if manifest_identity is None or (
        _directory_identity(staged.directory) != directory_identity
        or _leaf_identity(staged.ownership_marker_path) != marker_identity
        or _leaf_identity(staged.database_path) != database_identity
        or _leaf_identity(staged.manifest_path) != manifest_identity
        or staged.ownership_marker_path.read_text("ascii") != token
    ):
        raise SemanticStoreError("semantic staging identity changed")
    try:
        require_safe_directory(final.generations_dir)
        final_parent_identity = _directory_location_identity(final.generations_dir)
    except RuntimePathError as exc:
        raise SemanticStoreError("final semantic generation parent is unsafe") from exc
    if final.directory.exists() or path_is_redirect(final.directory):
        def accept_existing(
            existing: SemanticGenerationManifest,
            snapshot: _ReadOnlySnapshotConnection,
            guard: Callable[[], None],
        ) -> None:
            if existing != staged_manifest:
                raise SemanticStoreError("refusing to overwrite semantic generation")
            if acceptance_callback is not None:
                acceptance_callback(existing, snapshot, guard)

        if acceptance_callback is None:
            _reuse_existing_semantic_generation(final, staged_manifest)
        else:
            accept_semantic_generation(final, accept_existing)
        return
    staged_identity = _directory_identity(staged.directory)
    for leaf in (staged.database_path, staged.manifest_path):
        descriptor = os.open(leaf, os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(staged.directory)
    _fsync_directory(staged.generations_dir)
    if (
        _directory_identity(staged.directory) != directory_identity
        or _leaf_identity(staged.ownership_marker_path) != marker_identity
        or _leaf_identity(staged.database_path) != database_identity
        or _leaf_identity(staged.manifest_path) != manifest_identity
        or staged.ownership_marker_path.read_text("ascii") != token
    ):
        raise SemanticStoreError("semantic staging identity changed before rename")
    if failure_injector is not None:
        failure_injector("before_atomic_noreplace_rename")
    try:
        require_safe_directory(final.generations_dir)
        if _directory_location_identity(final.generations_dir) != final_parent_identity:
            raise SemanticStoreError("final semantic generation parent identity changed")
    except RuntimePathError as exc:
        raise SemanticStoreError("final semantic generation parent is unsafe") from exc
    if (
        _directory_identity(staged.directory) != directory_identity
        or _leaf_identity(staged.ownership_marker_path) != marker_identity
        or _leaf_identity(staged.database_path) != database_identity
        or _leaf_identity(staged.manifest_path) != manifest_identity
        or staged.ownership_marker_path.read_text("ascii") != token
    ):
        raise SemanticStoreError("semantic staging identity changed before rename")
    if final.directory.exists() or path_is_redirect(final.directory):
        def accept_racing_final(
            existing: SemanticGenerationManifest,
            snapshot: _ReadOnlySnapshotConnection,
            guard: Callable[[], None],
        ) -> None:
            if existing != staged_manifest:
                raise SemanticStoreError("refusing to overwrite semantic generation")
            if acceptance_callback is not None:
                acceptance_callback(existing, snapshot, guard)

        try:
            if acceptance_callback is None:
                _reuse_existing_semantic_generation(final, staged_manifest)
            else:
                accept_semantic_generation(final, accept_racing_final)
        except SemanticStoreError as exc:
            raise SemanticStoreError("foreign final semantic generation appeared") from exc
        return
    _atomic_directory_rename_noreplace(staged.directory, final.directory)
    try:
        with _hold_semantic_generation_for_acceptance(final) as held:
            if failure_injector is not None:
                failure_injector("after_final_handles_acquired")
            _fsync_directory(final.generations_dir)
            if _directory_location_identity(final.generations_dir) != final_parent_identity:
                raise SemanticStoreError("final semantic generation parent identity changed during installation")
            installed_manifest = _verify_held_semantic_generation(final, held)
            if _directory_location_identity(final.generations_dir) != final_parent_identity:
                raise SemanticStoreError(
                    "final semantic generation parent identity changed after verification"
                )
            if _directory_identity(final.directory) != staged_identity or installed_manifest != staged_manifest:
                raise SemanticStoreError("semantic generation changed during installation")
            if _directory_location_identity(final.generations_dir) != final_parent_identity:
                raise SemanticStoreError(
                    "final semantic generation parent identity changed at acceptance"
                )
            if _leaf_identity(staged.ownership_marker_path) != marker_identity or staged.ownership_marker_path.read_text("ascii") != token:
                raise SemanticStoreError("semantic staging marker changed during installation")
            staged.ownership_marker_path.unlink()
            _fsync_directory(staged.generations_dir)
            if _directory_location_identity(final.generations_dir) != final_parent_identity:
                raise SemanticStoreError(
                    "final semantic generation parent identity changed after cleanup"
                )
            with _OWNED_STAGING_LOCK:
                _OWNED_STAGING.pop(staged.directory.resolve(), None)
            if acceptance_callback is not None:
                snapshot = _connection_from_snapshot_bytes(held.database_bytes)
                try:
                    acceptance_callback(
                        installed_manifest,
                        snapshot,
                        lambda: _assert_held_semantic_generation_unchanged(held),
                    )
                finally:
                    snapshot.close()
    except RuntimePathError as exc:
        raise SemanticStoreError("final semantic generation parent is unsafe during installation") from exc
    except (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
        raise SemanticStoreError("final semantic generation is unsafe or corrupt during installation") from exc
    except OSError as exc:
        raise SemanticStoreError("final semantic generation parent changed during installation") from exc


def _fsync_directory(path: Path) -> None:
    """Persist directory metadata where the platform exposes a directory fd."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_directory_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a same-volume directory while refusing an existing target.

    Windows `MoveFileW` without `MOVEFILE_REPLACE_EXISTING` supplies the required
    no-replace operation.  Other platforms fail closed until a platform-specific
    renameat2-equivalent is available; a check-then-rename fallback is unsafe.
    """
    if os.name != "nt":
        raise SemanticStoreError("atomic no-replace directory install is unavailable")
    move_file = ctypes.windll.kernel32.MoveFileW
    move_file.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
    move_file.restype = ctypes.c_int
    if not move_file(str(source), str(destination)):
        raise SemanticStoreError("atomic no-replace semantic generation install failed")
