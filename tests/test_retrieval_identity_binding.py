from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

from codex_memory import database, structured_ledger
from codex_memory.paths import RuntimePaths
from codex_memory.retrieval.config import SemanticConfig
from codex_memory.retrieval.provider import (
    CapabilitiesResponse,
    DenseResponse,
    ModelComponent,
    ProviderCall,
    ProviderCallStatus,
    SparseResponse,
    ProviderOperation,
    checked_dense_response,
    checked_sparse_response,
)
from codex_memory.retrieval.provider_control import ProviderCallController
from codex_memory.retrieval.provider import ProviderInput
from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.service import RetrievalService
from codex_memory.retrieval.paths import RetrievalPaths
from retrieval_provider_fakes import FakeProvider


def _config(provider: FakeProvider) -> SemanticConfig:
    capabilities = provider.capabilities(deadline_ms=100).capabilities
    assert capabilities is not None
    return SemanticConfig(
        1, "local_http", "http://127.0.0.1:11434", "audited-local-v1",
        1_000, 10_000, 100, 100, True, True, False, 16, 16, 8, 2,
        1, 0, 2, 30_000, capabilities.fingerprint,
        capabilities.dense.identity.fingerprint if capabilities.dense.identity else None,
        capabilities.sparse.identity.fingerprint if capabilities.sparse.identity else None,
        None,
    )


def _repo(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    repo = tmp_path / "identity repository"
    repo.mkdir()
    paths = RuntimePaths.for_repo(repo)
    paths.ensure()
    connection = database.connect_database(paths.db_path)
    database.migrate_database(connection)
    record = structured_ledger.propose_record(
        connection,
        record_type="decision",
        scope_kind="project",
        scope_key=str(repo),
        topic_key="identity.binding",
        statement="identity binding fixture",
        idempotency_key="identity-binding-fixture",
    )
    structured_ledger.apply_action(
        connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key="identity-binding-fixture-active",
    )
    connection.commit()
    return repo, connection


def test_dense_and_sparse_validators_reject_same_shape_wrong_identity() -> None:
    provider = FakeProvider()
    capabilities = provider.capabilities(deadline_ms=100).capabilities
    assert capabilities is not None
    item = ProviderInput.from_text("q", 0, "query")
    dense = provider.embed_query(item, deadline_ms=100)
    sparse = provider.sparse_query(item, deadline_ms=100)
    assert checked_dense_response(
        dense, (item,), dimension=4,
        expected_provider_fingerprint="0" * 64,
        expected_model_fingerprint="1" * 64,
    ).call.status is ProviderCallStatus.FINGERPRINT_MISMATCH
    assert checked_sparse_response(
        sparse, (item,),
        expected_provider_fingerprint="0" * 64,
        expected_model_fingerprint="1" * 64,
    ).call.status is ProviderCallStatus.FINGERPRINT_MISMATCH


def test_controller_rejects_capability_payload_identity_mismatch(tmp_path: Path) -> None:
    provider = FakeProvider()
    expected = provider.capabilities(deadline_ms=100).capabilities
    assert expected is not None
    controller = ProviderCallController(
        provider,
        DeadlineBudget.start(500, now_ns=__import__("time").monotonic_ns),
        None,
        "0" * 64,
    )
    response = controller.capabilities()
    assert response.call.status is ProviderCallStatus.FINGERPRINT_MISMATCH
    assert response.capabilities is None


def test_build_identity_change_between_handshake_and_documents_is_not_indexed(tmp_path: Path) -> None:
    repo, connection = _repo(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    provider = FakeProvider()
    config = _config(provider)

    class Drift(FakeProvider):
        def embed_documents(self, items, *, deadline_ms):  # type: ignore[no-untyped-def]
            capabilities = self._capabilities
            assert capabilities.dense.identity is not None
            changed = replace(
                capabilities.dense.identity,
                revision="drifted-revision",
            )
            self._capabilities = type(capabilities)(
                capabilities.protocol_version,
                capabilities.provider_type,
                capabilities.provider_version,
                capabilities.provider_digest,
                type(capabilities.dense)(ModelComponent.DENSE, capabilities.dense.state, changed, None),
                capabilities.sparse,
                capabilities.reranker,
            )
            return super().embed_documents(items, deadline_ms=deadline_ms)

    drift = Drift()
    # Keep configured identity from the original provider, while the provider
    # changes its observed model identity after capability negotiation.
    result = RetrievalService(paths).rebuild_semantic(drift, config)
    assert result.status.value == "current"
    with sqlite3.connect(paths.database_path) as cache:
        row = cache.execute(
            "SELECT status, indexed_document_count, coverage_highwater_json FROM retrieval_semantic_components WHERE component='dense'"
        ).fetchone()
    assert row is not None
    assert row[0:2] == ("failed", 0)


def test_non_ok_capability_handshake_never_authorizes_model_build(tmp_path: Path) -> None:
    """A payload on a failed handshake is observed data, not build authorization."""
    repo, connection = _repo(tmp_path)
    connection.close()
    paths = RetrievalPaths.for_repo(repo)
    base = FakeProvider()
    config = _config(base)

    class FailedHandshake(FakeProvider):
        def capabilities(self, *, deadline_ms: int):  # type: ignore[no-untyped-def]
            observed = super().capabilities(deadline_ms=deadline_ms)
            return CapabilitiesResponse(
                ProviderCall(
                    ProviderOperation.CAPABILITIES,
                    None,
                    ProviderCallStatus.UNAVAILABLE,
                    observed.call.provider_fingerprint,
                    None,
                    observed.call.elapsed_ms,
                    ProviderCallStatus.UNAVAILABLE.value,
                ),
                observed.capabilities,
            )

    provider = FailedHandshake()
    result = RetrievalService(paths).rebuild_semantic(provider, config)

    assert result.status.value == "current"
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] == 0
    assert provider.call_counts[ProviderOperation.SPARSE_DOCUMENTS] == 0
