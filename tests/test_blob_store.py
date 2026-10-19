import hashlib

import pytest

from codex_memory.atomic import atomic_write_bytes, atomic_write_json
from codex_memory.blob_store import BlobIntegrityError, BlobStore
from codex_memory.paths import RuntimePaths


def test_identical_bytes_are_deduplicated(tmp_path) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    store = BlobStore(paths.blobs_dir)
    first = store.put_bytes(b"same")
    second = store.put_bytes(b"same")
    assert first == second
    assert store.read(first) == b"same"
    assert first.sha256 == hashlib.sha256(b"same").hexdigest()


def test_corrupt_blob_is_rejected(tmp_path) -> None:
    paths = RuntimePaths.for_repo(tmp_path)
    store = BlobStore(paths.blobs_dir)
    ref = store.put_bytes(b"expected")
    store.path_for(ref).write_bytes(b"corrupt")
    with pytest.raises(BlobIntegrityError, match="SHA-256"):
        store.read(ref)


def test_read_sha256_reuses_existing_integrity_checks(tmp_path) -> None:
    store = BlobStore(RuntimePaths.for_repo(tmp_path).blobs_dir)
    ref = store.put_bytes(b"expected")

    assert hasattr(store, "read_sha256"), "digest-addressed read is required"
    assert store.read_sha256(ref.sha256) == b"expected"

    store.path_for(ref).write_bytes(b"corrupt")
    with pytest.raises(BlobIntegrityError, match="SHA-256"):
        store.read_sha256(ref.sha256)
    with pytest.raises(BlobIntegrityError, match="digest is invalid"):
        store.read_sha256("not-a-digest")


def test_existing_corrupt_blob_is_rejected_without_rewriting(tmp_path) -> None:
    store = BlobStore(RuntimePaths.for_repo(tmp_path).blobs_dir)
    ref = store.put_bytes(b"expected")
    blob_path = store.path_for(ref)
    blob_path.write_bytes(b"corrupt")

    with pytest.raises(BlobIntegrityError, match="byte size"):
        store.put_bytes(b"expected")

    assert blob_path.read_bytes() == b"corrupt"


def test_atomic_write_json_writes_canonical_utf8_and_creates_parents(tmp_path) -> None:
    path = tmp_path / "nested" / "record.json"

    written = atomic_write_json(path, {"z": "é", "a": [1, 2]})

    assert written == b'{"a":[1,2],"z":"\xc3\xa9"}'
    assert path.read_bytes() == written


def test_atomic_write_cleans_up_temporary_file_when_replace_fails(tmp_path, monkeypatch) -> None:
    path = tmp_path / "payload.bin"

    def fail_replace(source, destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("codex_memory.atomic.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_bytes(path, b"payload")

    assert list(tmp_path.iterdir()) == []
