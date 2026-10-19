import hashlib
import json
from pathlib import Path
from dataclasses import replace

import pytest

from codex_memory.retrieval.contracts import IndexHighwater
from codex_memory.ledger_common import canonical_json


def _highwater() -> dict[str, int]:
    return IndexHighwater(record_seq=1).to_dict()


def _prepared_install_pair(tmp_path: Path):  # type: ignore[no-untyped-def]
    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        build_semantic_manifest,
        deterministic_semantic_generation_id,
        initialize_semantic_database,
        verify_semantic_generation,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id(
        "gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64
    )
    staged = SemanticGenerationPaths.for_generation(
        tmp_path / "temp" / "generations", generation_id
    )
    final = SemanticGenerationPaths.for_generation(
        tmp_path / "retrieval" / "generations", generation_id
    )
    staged.directory.mkdir(parents=True)
    final.generations_dir.mkdir(parents=True)
    initialize_semantic_database(staged)
    manifest = build_semantic_manifest(
        paths=staged,
        projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(),
        indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint,
        dense_model_fingerprint=None,
        sparse_model_fingerprint=None,
        reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64,
        build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    staged.manifest_path.write_bytes(manifest.to_json_bytes())
    verify_semantic_generation(staged)
    return staged, final, manifest


def test_semantic_store_writes_and_verifies_a_manifest(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        build_semantic_manifest,
        initialize_semantic_database,
        verify_semantic_generation,
        deterministic_semantic_generation_id,
    )
    from codex_memory.retrieval.provider import DisabledProvider

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id("gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64)
    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", generation_id)
    paths.directory.mkdir(parents=True)
    initialize_semantic_database(paths)
    manifest = build_semantic_manifest(
        paths=paths,
        projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(),
        indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint,
        dense_model_fingerprint=None,
        sparse_model_fingerprint=None,
        reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64,
        build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    paths.manifest_path.write_bytes(manifest.to_json_bytes())

    assert verify_semantic_generation(paths).semantic_generation_id == paths.semantic_generation_id


def test_semantic_store_rejects_manifest_file_hash_mismatch(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError,
        build_semantic_manifest,
        initialize_semantic_database,
        verify_semantic_generation,
        deterministic_semantic_generation_id,
    )
    from codex_memory.retrieval.provider import DisabledProvider

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id("gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64)
    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", generation_id)
    paths.directory.mkdir(parents=True)
    initialize_semantic_database(paths)
    manifest = build_semantic_manifest(
        paths=paths, projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(), indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint, dense_model_fingerprint=None,
        sparse_model_fingerprint=None, reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64, build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    paths.manifest_path.write_bytes(manifest.to_json_bytes())
    paths.database_path.write_bytes(b"foreign replacement")

    with pytest.raises(SemanticStoreError, match="hash|database"):
        verify_semantic_generation(paths)


@pytest.mark.parametrize("highwater_field", ["canonical_highwater", "indexed_highwater"])
@pytest.mark.parametrize(
    "bad_kind",
    ["missing", "extra", "negative", "boolean", "string", "float", "null"],
)
def test_semantic_manifest_rejects_invalid_highwater_shape_and_value(
    tmp_path: Path, highwater_field: str, bad_kind: str
) -> None:
    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError, build_semantic_manifest, initialize_semantic_database,
        verify_semantic_generation, deterministic_semantic_generation_id,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id(
        "gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64
    )
    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", generation_id)
    paths.directory.mkdir(parents=True)
    initialize_semantic_database(paths)
    manifest = build_semantic_manifest(
        paths=paths,
        projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(),
        indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint,
        dense_model_fingerprint=None,
        sparse_model_fingerprint=None,
        reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64,
        build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    payload = manifest.to_dict()
    bad_highwater: dict[str, object] = dict(_highwater())
    if bad_kind == "missing":
        bad_highwater.pop("record_seq")
    elif bad_kind == "extra":
        bad_highwater["unexpected"] = 0
    else:
        bad_highwater["record_seq"] = {
            "negative": -1,
            "boolean": True,
            "string": "1",
            "float": 1.0,
            "null": None,
        }[bad_kind]
    payload[highwater_field] = bad_highwater
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    paths.manifest_path.write_bytes(canonical_json(payload).encode("utf-8"))

    with pytest.raises(SemanticStoreError, match="highwater"):
        verify_semantic_generation(paths)


def test_semantic_manifest_rejects_noncanonical_highwater_value_representation(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError, build_semantic_manifest, initialize_semantic_database,
        verify_semantic_generation, deterministic_semantic_generation_id,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id(
        "gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64
    )
    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", generation_id)
    paths.directory.mkdir(parents=True)
    initialize_semantic_database(paths)
    manifest = build_semantic_manifest(
        paths=paths,
        projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(),
        indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint,
        dense_model_fingerprint=None,
        sparse_model_fingerprint=None,
        reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64,
        build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    raw = manifest.to_json_bytes().decode("utf-8")
    raw = raw.replace('"memory_event_seq":0', '"memory_event_seq":-0', 1)
    assert json.loads(raw)["canonical_highwater"]["memory_event_seq"] == 0
    paths.manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(SemanticStoreError, match="canonical"):
        verify_semantic_generation(paths)


def test_semantic_database_initializer_refuses_an_arbitrary_or_canonical_path(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_store import SemanticStoreError, initialize_semantic_database

    canonical = tmp_path / ".codex-memory" / "db" / "memory.sqlite3"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical-bytes")
    before = canonical.read_bytes()

    with pytest.raises(SemanticStoreError, match="paths|generation|path"):
        initialize_semantic_database(canonical)  # type: ignore[arg-type]

    assert canonical.read_bytes() == before


def test_public_semantic_connector_refuses_a_canonical_path_before_sqlite_opens(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_store import SemanticStoreError, connect_semantic_database

    canonical = tmp_path / ".codex-memory" / "db" / "memory.sqlite3"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical-bytes")
    before = canonical.read_bytes()

    with pytest.raises(SemanticStoreError, match="paths|generation|path"):
        connect_semantic_database(canonical)  # type: ignore[arg-type]

    assert canonical.read_bytes() == before


def test_forged_generation_paths_cannot_redirect_the_ownership_marker(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import SemanticStoreError, initialize_semantic_database

    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", "sem_0123456789abcdef")
    paths.directory.mkdir(parents=True)
    external = tmp_path / "external.owner"
    forged = replace(paths, ownership_marker_path=external)

    with pytest.raises(SemanticStoreError, match="marker|paths|unsafe"):
        initialize_semantic_database(forged)

    assert not external.exists()


def test_install_requires_owned_verified_staging_and_reuses_identical_final(tmp_path: Path) -> None:
    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        build_semantic_manifest, deterministic_semantic_generation_id,
        initialize_semantic_database, install_semantic_generation, verify_semantic_generation,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    identifier = deterministic_semantic_generation_id("gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64)
    staged = SemanticGenerationPaths.for_generation(tmp_path / "temp" / "generations", identifier)
    final = SemanticGenerationPaths.for_generation(tmp_path / "retrieval" / "generations", identifier)
    staged.directory.mkdir(parents=True)
    final.generations_dir.parent.mkdir(parents=True)
    final.generations_dir.mkdir()
    initialize_semantic_database(staged)
    manifest = build_semantic_manifest(
        paths=staged, projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(), indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint, dense_model_fingerprint=None,
        sparse_model_fingerprint=None, reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64, build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    staged.manifest_path.write_bytes(manifest.to_json_bytes())
    verify_semantic_generation(staged)

    install_semantic_generation(staged, final)
    assert verify_semantic_generation(final) == manifest
    assert not final.ownership_marker_path.exists()


def test_install_refuses_a_foreign_final_directory_created_at_the_rename_boundary(tmp_path: Path) -> None:
    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError, build_semantic_manifest, deterministic_semantic_generation_id,
        initialize_semantic_database, install_semantic_generation, verify_semantic_generation,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    identifier = deterministic_semantic_generation_id("gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64)
    staged = SemanticGenerationPaths.for_generation(tmp_path / "temp" / "generations", identifier)
    final = SemanticGenerationPaths.for_generation(tmp_path / "retrieval" / "generations", identifier)
    staged.directory.mkdir(parents=True)
    final.generations_dir.mkdir(parents=True)
    initialize_semantic_database(staged)
    manifest = build_semantic_manifest(paths=staged, projection_generation_id="gen_0123456789abcdef", canonical_highwater=_highwater(), indexed_highwater=_highwater(), provider_fingerprint=capabilities.fingerprint, dense_model_fingerprint=None, sparse_model_fingerprint=None, reranker_model_fingerprint=None, preprocessing_fingerprint="b" * 64, build_config_sha256="c" * 64, created_at="2026-09-05T00:00:00.000000Z")
    staged.manifest_path.write_bytes(manifest.to_json_bytes())
    verify_semantic_generation(staged)

    with pytest.raises(SemanticStoreError, match="final|overwrite|target"):
        install_semantic_generation(staged, final, failure_injector=lambda _stage: final.directory.mkdir())

    assert final.directory.is_dir()
    assert staged.directory.is_dir()


def test_install_reuses_identical_existing_final_without_activation_callback(tmp_path: Path) -> None:
    """A store-only retry must verify/reuse the existing bytes without activation."""
    from codex_memory.retrieval.semantic_store import install_semantic_generation, verify_semantic_generation

    staged, final, expected = _prepared_install_pair(tmp_path)
    final.directory.mkdir()
    final.database_path.write_bytes(staged.database_path.read_bytes())
    final.manifest_path.write_bytes(staged.manifest_path.read_bytes())
    before = {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in (final.database_path, final.manifest_path)
    }

    install_semantic_generation(staged, final)
    install_semantic_generation(staged, final)

    assert verify_semantic_generation(final) == expected
    assert {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in (final.database_path, final.manifest_path)
    } == before
    assert staged.database_path.is_file()
    assert staged.ownership_marker_path.is_file()


def test_install_reuses_identical_racing_final_without_activation_callback(tmp_path: Path) -> None:
    """A same-content peer appearing before rename is safe to reuse, never overwrite."""
    from codex_memory.retrieval.semantic_store import install_semantic_generation, verify_semantic_generation

    staged, final, expected = _prepared_install_pair(tmp_path)
    before = {}

    def create_peer(stage: str) -> None:
        assert stage == "before_atomic_noreplace_rename"
        final.directory.mkdir()
        final.database_path.write_bytes(staged.database_path.read_bytes())
        final.manifest_path.write_bytes(staged.manifest_path.read_bytes())
        before.update({
            path.name: (path.stat().st_ino, path.read_bytes())
            for path in (final.database_path, final.manifest_path)
        })

    install_semantic_generation(staged, final, failure_injector=create_peer)

    assert verify_semantic_generation(final) == expected
    assert {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in (final.database_path, final.manifest_path)
    } == before
    assert staged.database_path.is_file()
    assert staged.ownership_marker_path.is_file()


@pytest.mark.parametrize("exchange_kind", ["database_leaf", "staged_directory"])
def test_install_rejects_staging_exchange_before_rename_without_installing_foreign_bytes(
    tmp_path: Path, exchange_kind: str
) -> None:
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError,
        install_semantic_generation,
    )

    staged, final, _ = _prepared_install_pair(tmp_path)
    saved = tmp_path / f"saved-{exchange_kind}"

    def exchange(stage: str) -> None:
        assert stage == "before_atomic_noreplace_rename"
        if exchange_kind == "database_leaf":
            staged.database_path.replace(saved)
        else:
            staged.directory.replace(saved)
            staged.directory.mkdir()
        staged.database_path.write_bytes(b"foreign-db")
        if exchange_kind == "staged_directory":
            staged.manifest_path.write_bytes(b"foreign-manifest")

    with pytest.raises(SemanticStoreError, match="identity|changed|unsafe"):
        install_semantic_generation(staged, final, failure_injector=exchange)

    assert not final.directory.exists()
    assert staged.database_path.read_bytes() == b"foreign-db"


def test_install_rejects_final_generations_parent_exchange_before_rename(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError,
        install_semantic_generation,
    )

    staged, final, _ = _prepared_install_pair(tmp_path)
    original_parent = final.generations_dir.with_name("generations-original")

    def exchange(stage: str) -> None:
        assert stage == "before_atomic_noreplace_rename"
        final.generations_dir.replace(original_parent)
        final.generations_dir.mkdir()

    with pytest.raises(SemanticStoreError, match="parent|identity|changed|unsafe"):
        install_semantic_generation(staged, final, failure_injector=exchange)

    assert not final.directory.exists()
    assert original_parent.is_dir()


def test_install_rejects_final_generations_parent_exchange_inside_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent exchanged at the rename boundary must not be reported as success."""
    import codex_memory.retrieval.semantic_store as semantic_store

    staged, final, _ = _prepared_install_pair(tmp_path)
    original_parent = final.generations_dir.with_name("generations-original")
    real_rename = semantic_store._atomic_directory_rename_noreplace

    def exchange_parent(source: Path, destination: Path) -> None:
        final.generations_dir.replace(original_parent)
        final.generations_dir.mkdir()
        real_rename(source, destination)

    monkeypatch.setattr(semantic_store, "_atomic_directory_rename_noreplace", exchange_parent)

    with pytest.raises(semantic_store.SemanticStoreError, match="parent|changed|unsafe"):
        semantic_store.install_semantic_generation(staged, final)

    assert original_parent.is_dir()
    assert (final.generations_dir / final.semantic_generation_id).is_dir()


def test_install_holds_final_generations_parent_after_protected_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generations parent remains non-exchangeable after held-byte verification."""
    import codex_memory.retrieval.semantic_store as semantic_store

    staged, final, manifest = _prepared_install_pair(tmp_path)
    original_parent = final.generations_dir.with_name("generations-original")
    real_verify = semantic_store._verify_held_semantic_generation
    attack_error: OSError | None = None

    def verify_and_exchange(paths, held):  # type: ignore[no-untyped-def]
        nonlocal attack_error
        result = real_verify(paths, held)
        try:
            final.generations_dir.replace(original_parent)
        except OSError as exc:
            attack_error = exc
        return result

    monkeypatch.setattr(semantic_store, "_verify_held_semantic_generation", verify_and_exchange)

    semantic_store.install_semantic_generation(staged, final)

    assert isinstance(attack_error, OSError)
    assert not original_parent.exists()
    assert semantic_store.verify_semantic_generation(final) == manifest


def test_install_holds_final_directory_after_protected_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final directory remains non-exchangeable after held-byte verification."""
    import codex_memory.retrieval.semantic_store as semantic_store

    staged, final, manifest = _prepared_install_pair(tmp_path)
    displaced = final.directory.with_name("displaced-final-generation")
    real_verify = semantic_store._verify_held_semantic_generation
    attack_error: OSError | None = None

    def verify_and_exchange(paths, held):  # type: ignore[no-untyped-def]
        nonlocal attack_error
        result = real_verify(paths, held)
        try:
            final.directory.replace(displaced)
        except OSError as exc:
            attack_error = exc
        return result

    monkeypatch.setattr(semantic_store, "_verify_held_semantic_generation", verify_and_exchange)

    semantic_store.install_semantic_generation(staged, final)

    assert isinstance(attack_error, OSError)
    assert not displaced.exists()
    assert semantic_store.verify_semantic_generation(final) == manifest


def test_install_rejects_parent_exchange_after_the_final_parent_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final acceptance seam must keep the parent stable, not merely re-stat it."""
    import codex_memory.retrieval.semantic_store as semantic_store

    staged, final, _ = _prepared_install_pair(tmp_path)
    original_parent = final.generations_dir.with_name("generations-original")
    real_fsync = semantic_store._fsync_directory
    real_identity = semantic_store._directory_location_identity
    arm_exchange = False
    exchanged = False

    def fsync_then_arm(path):  # type: ignore[no-untyped-def]
        nonlocal arm_exchange
        real_fsync(path)
        if path == staged.generations_dir and final.directory.exists():
            arm_exchange = True

    def identity_then_exchange(path):  # type: ignore[no-untyped-def]
        nonlocal exchanged
        result = real_identity(path)
        if arm_exchange and path == final.generations_dir and not exchanged:
            exchanged = True
            final.generations_dir.replace(original_parent)
            final.generations_dir.mkdir()
        return result

    monkeypatch.setattr(semantic_store, "_fsync_directory", fsync_then_arm)
    monkeypatch.setattr(semantic_store, "_directory_location_identity", identity_then_exchange)

    with pytest.raises(semantic_store.SemanticStoreError, match="parent|changed|unsafe"):
        semantic_store.install_semantic_generation(staged, final)

    assert exchanged


def test_install_holds_final_generation_directory_during_store_acceptance(
    tmp_path: Path,
) -> None:
    """The final directory itself cannot be exchanged while its handle is held."""
    from codex_memory.retrieval.semantic_store import install_semantic_generation, verify_semantic_generation

    staged, final, manifest = _prepared_install_pair(tmp_path)
    displaced = final.directory.with_name("displaced-final-generation")
    saw_handle_stage = False

    def exchange_after_handles(stage: str) -> None:
        nonlocal saw_handle_stage
        if stage != "after_final_handles_acquired":
            return
        saw_handle_stage = True
        with pytest.raises(OSError):
            final.directory.replace(displaced)

    install_semantic_generation(staged, final, failure_injector=exchange_after_handles)

    assert saw_handle_stage
    assert verify_semantic_generation(final) == manifest
    assert not displaced.exists()


@pytest.mark.parametrize("leaf_name", ["database_path", "manifest_path"])
def test_semantic_generation_verification_rejects_hardlinked_leaf_without_touching_peer(
    tmp_path: Path, leaf_name: str
) -> None:
    import os

    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError, build_semantic_manifest, initialize_semantic_database,
        verify_semantic_generation, deterministic_semantic_generation_id,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id("gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64)
    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", generation_id)
    paths.directory.mkdir(parents=True)
    initialize_semantic_database(paths)
    manifest = build_semantic_manifest(
        paths=paths, projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(), indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint, dense_model_fingerprint=None,
        sparse_model_fingerprint=None, reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64, build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    paths.manifest_path.write_bytes(manifest.to_json_bytes())
    leaf = getattr(paths, leaf_name)
    peer = tmp_path / f"peer-{leaf.name}"
    try:
        os.link(leaf, peer)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    try:
        with pytest.raises(SemanticStoreError, match="unsafe|hard"):
            verify_semantic_generation(paths)
        assert peer.exists()
    finally:
        peer.unlink(missing_ok=True)


@pytest.mark.parametrize("leaf_name", ["database_path", "manifest_path"])
def test_semantic_generation_verification_rejects_reparse_leaf_at_the_primitive_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leaf_name: str
) -> None:
    from codex_memory.retrieval.provider import DisabledProvider
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import (
        SemanticStoreError, build_semantic_manifest, initialize_semantic_database,
        verify_semantic_generation, deterministic_semantic_generation_id,
    )

    capabilities = DisabledProvider().capabilities(deadline_ms=1).capabilities
    assert capabilities is not None
    generation_id = deterministic_semantic_generation_id("gen_0123456789abcdef", capabilities, "b" * 64, "c" * 64)
    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", generation_id)
    paths.directory.mkdir(parents=True)
    initialize_semantic_database(paths)
    manifest = build_semantic_manifest(
        paths=paths, projection_generation_id="gen_0123456789abcdef",
        canonical_highwater=_highwater(), indexed_highwater=_highwater(),
        provider_fingerprint=capabilities.fingerprint, dense_model_fingerprint=None,
        sparse_model_fingerprint=None, reranker_model_fingerprint=None,
        preprocessing_fingerprint="b" * 64, build_config_sha256="c" * 64,
        created_at="2026-09-05T00:00:00.000000Z",
    )
    paths.manifest_path.write_bytes(manifest.to_json_bytes())
    leaf = getattr(paths, leaf_name)
    monkeypatch.setattr(
        "codex_memory.retrieval.semantic_store.path_is_redirect",
        lambda path: Path(path) == leaf,
    )
    with pytest.raises(SemanticStoreError, match="unsafe|redirect"):
        verify_semantic_generation(paths)
