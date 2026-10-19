"""Build disposable semantic generations from the verified P2.2 projection.

The canonical ledger is deliberately never opened here.  The deterministic
retrieval generation is the materialized, verified boundary: we take its
permanent documents, close every SQLite handle, then call the optional model
provider.  A semantic build can therefore fail without affecting recovery or
the active deterministic generation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import hashlib
import json
import math
import sqlite3
import struct
import time
from types import SimpleNamespace
import unicodedata

from codex_memory.ledger_common import canonical_json

from .cache_database import (
    append_semantic_component,
    append_semantic_generation,
    append_semantic_generation_event,
    connect_cache_database,
    immediate_transaction,
    rebuild_semantic_projections,
    verify_cache_database,
)
from .config import SemanticConfig
from .contracts import (
    IndexHighwater,
    RetrievalResult,
    RetrievalStatus,
    canonical_contract_sha256,
)
from .paths import RetrievalPaths
from .provider import (
    CapabilityState,
    ModelComponent,
    ProviderCallStatus,
    ProviderInput,
    RetrievalModelProvider,
    checked_dense_response,
    checked_sparse_response,
)
from .deadline import DeadlineBudget
from .provider_control import ProviderCallController
from .provider_health import ProviderHealthStore
from .semantic_paths import SemanticGenerationPaths
from .semantic_store import (
    SemanticStoreError,
    accept_semantic_generation,
    append_dense_vector,
    append_semantic_document,
    append_sparse_posting,
    build_semantic_manifest,
    connect_semantic_database,
    deterministic_semantic_generation_id,
    initialize_semantic_database,
    install_semantic_generation,
    open_verified_semantic_snapshot,
    verify_semantic_generation,
)


SemanticRetrievalConfig = SemanticConfig

_COMPONENT_OUTCOMES_META_KEY = "component_outcomes_v1"
_COMPONENT_STATUSES = frozenset(
    {"disabled", "ready", "partial", "unsupported", "unavailable", "failed"}
)
_COMPONENT_REASONS = frozenset(
    {status.value for status in ProviderCallStatus if status is not ProviderCallStatus.OK}
    | {
        "provider_fingerprint_mismatch",
        "fingerprint_mismatch",
        "task3_no_rerank_build",
    }
)
_CAPABILITY_UNAVAILABLE_STATUSES = frozenset(
    {
        ProviderCallStatus.UNAVAILABLE,
        ProviderCallStatus.BUSY,
        ProviderCallStatus.TIMEOUT,
        ProviderCallStatus.CIRCUIT_OPEN,
        ProviderCallStatus.DEADLINE_EXHAUSTED,
    }
)


@dataclass(frozen=True, slots=True)
class _ComponentOutcome:
    component: ModelComponent
    status: str
    identity: object | None
    eligible_count: int
    indexed_count: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class _SemanticActivationFence:
    projection_generation_id: str
    active_semantic_generation_id: str | None
    semantic_event_seq: int


def rebuild_semantic_index(
    paths: RetrievalPaths,
    provider: RetrievalModelProvider,
    config: SemanticRetrievalConfig,
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RetrievalResult:
    """Rebuild deterministic projection, then independently build semantic bytes."""
    from .service import RetrievalService

    deterministic = RetrievalService(paths).rebuild()
    return _build_from_current(paths, provider, config, deterministic, clock=clock)


def catch_up_semantic_index(
    paths: RetrievalPaths,
    provider: RetrievalModelProvider,
    config: SemanticRetrievalConfig,
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RetrievalResult:
    """Catch up deterministic projection before building a distinct semantic generation."""
    from .service import RetrievalService

    deterministic = RetrievalService(paths).catch_up()
    return _build_from_current(paths, provider, config, deterministic, clock=clock)


def _build_from_current(
    paths: RetrievalPaths,
    provider: RetrievalModelProvider,
    config: SemanticConfig,
    deterministic: RetrievalResult,
    *,
    clock: Callable[[], int],
) -> RetrievalResult:
    if not isinstance(config, SemanticConfig):
        raise TypeError("semantic config must be SemanticConfig")
    if deterministic.status is not RetrievalStatus.CURRENT or deterministic.generation is None:
        return deterministic

    # This read uses only the already verified derived projection.  It excludes
    # candidates, blobs and transcripts by construction and closes before IO.
    documents = _read_permanent_projection(paths, deterministic.generation.generation_id)
    build_budget = DeadlineBudget.start(config.build_deadline_ms, now_ns=clock)
    controller = ProviderCallController(
        provider,
        build_budget,
        ProviderHealthStore(
            paths.database_path,
            failure_threshold=config.circuit_failure_threshold,
            cooldown_ms=config.circuit_cooldown_ms,
            max_inflight=config.max_inflight,
            queue_limit=config.queue_limit,
        ),
        config.expected_provider_fingerprint,
        now_ns=clock,
        expected_dense_fingerprint=config.expected_dense_fingerprint,
        expected_sparse_fingerprint=config.expected_sparse_fingerprint,
        expected_reranker_fingerprint=config.expected_reranker_fingerprint,
    )
    capabilities_response = controller.capabilities()
    capabilities = capabilities_response.capabilities
    capability_failure_status = capabilities_response.call.status
    if capabilities_response.call.status is not ProviderCallStatus.OK or capabilities is None:
        # Preserve the deterministic floor while producing a durable, empty
        # semantic generation whose components explicitly record unavailability.
        if capabilities is None:
            unavailable = SimpleNamespace(state=CapabilityState.UNAVAILABLE, identity=None)
            capabilities = SimpleNamespace(
                fingerprint=config.expected_provider_fingerprint,
                dense=unavailable, sparse=unavailable, reranker=unavailable,
            )
        if capabilities_response.call.status is ProviderCallStatus.OK:
            capability_failure_status = ProviderCallStatus.INVALID_RESPONSE
    # The controller validates both the response envelope and its capability
    # fingerprint.  Do not let the placeholder/observed metadata turn an
    # identity failure into an availability failure.
    provider_identity_matches = (
        capabilities_response.call.status is ProviderCallStatus.OK
        and capabilities.fingerprint == config.expected_provider_fingerprint
    )

    preprocessing_fingerprint = _preprocessing_fingerprint()
    build_config_sha256 = canonical_contract_sha256(config.to_dict())
    generation_id = deterministic_semantic_generation_id(
        deterministic.generation.generation_id,
        capabilities,
        preprocessing_fingerprint,
        build_config_sha256,
    )
    activation_fence = _capture_semantic_activation_fence(
        paths, deterministic.generation.generation_id
    )
    final = SemanticGenerationPaths.for_generation(paths.generations_dir, generation_id)
    if final.directory.exists():
        def accept_existing(manifest, snapshot, acceptance_guard):  # type: ignore[no-untyped-def]
            _handoff_verified_generation(
                paths,
                manifest,
                _outcomes_from_connection(snapshot, manifest),
                verified_document_bindings=_verified_document_bindings(snapshot),
                acceptance_guard=acceptance_guard,
                expected_active_semantic_generation_id=activation_fence.active_semantic_generation_id,
                expected_semantic_event_seq=activation_fence.semantic_event_seq,
                expected_projection_generation_id=activation_fence.projection_generation_id,
            )

        accept_semantic_generation(final, accept_existing)
        return deterministic

    dense_identity, dense_initial = _component_identity(
        config, capabilities, ModelComponent.DENSE, provider_identity_matches,
        capability_call_status=capability_failure_status,
    )
    sparse_identity, sparse_initial = _component_identity(
        config, capabilities, ModelComponent.SPARSE, provider_identity_matches,
        capability_call_status=capability_failure_status,
    )
    # Provider/network work is complete before the semantic SQLite database is
    # created, so it cannot run inside either a canonical or derived write tx.
    dense_rows, dense_outcome = _dense_rows(
        paths,
        controller,
        documents,
        dense_identity,
        dense_initial,
        config,
        preprocessing_fingerprint,
    )
    sparse_rows, sparse_outcome = _sparse_rows(
        paths,
        controller,
        documents,
        sparse_identity,
        sparse_initial,
        config,
        preprocessing_fingerprint,
    )
    reranker_outcome = _ComponentOutcome(
        ModelComponent.RERANKER,
        "disabled" if not config.reranker else "failed",
        None,
        0 if not config.reranker else len(documents),
        0,
        "disabled" if not config.reranker else "task3_no_rerank_build",
    )
    outcomes = _validate_component_outcomes(
        (dense_outcome, sparse_outcome, reranker_outcome)
    )

    paths.prepare_for_write()
    staged = SemanticGenerationPaths.for_generation(paths.temp_dir, generation_id)
    if staged.directory.exists():
        raise SemanticStoreError("semantic staging generation already exists")
    staged.directory.mkdir()
    try:
        initialize_semantic_database(staged)
        connection = connect_semantic_database(staged)
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for ordinal, document in enumerate(documents):
                    append_semantic_document(
                        connection,
                        ordinal=ordinal,
                        document_id=document["document_id"],
                        record_id=document["record_id"],
                        content_sha256=document["content_sha256"],
                        text_sha256=ProviderInput.sha256_text(
                            _semantic_input_text(document["search_text"])
                        ),
                    )
                for row in dense_rows:
                    append_dense_vector(connection, **row)
                for row in sparse_rows:
                    append_sparse_posting(connection, **row)
                _append_component_outcomes(connection, outcomes)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        finally:
            connection.close()
        manifest = build_semantic_manifest(
            paths=staged,
            projection_generation_id=deterministic.generation.generation_id,
            canonical_highwater=deterministic.canonical_highwater.to_dict(),
            indexed_highwater=deterministic.indexed_highwater.to_dict(),
            provider_fingerprint=capabilities.fingerprint,
            dense_model_fingerprint=None if dense_identity is None else dense_identity.fingerprint,
            sparse_model_fingerprint=None if sparse_identity is None else sparse_identity.fingerprint,
            reranker_model_fingerprint=None,
            preprocessing_fingerprint=preprocessing_fingerprint,
            build_config_sha256=build_config_sha256,
            created_at=deterministic.generation.created_at,
        )
        staged.manifest_path.write_bytes(manifest.to_json_bytes())
        verify_semantic_generation(staged)
        persisted_outcomes = _outcomes_from_manifest(staged, manifest)
        if _component_outcomes_value(persisted_outcomes) != _component_outcomes_value(outcomes):
            raise SemanticStoreError("semantic component outcomes changed before installation")
        def accept_installed(installed_manifest, snapshot, acceptance_guard):  # type: ignore[no-untyped-def]
            installed_outcomes = _outcomes_from_connection(snapshot, installed_manifest)
            if _component_outcomes_value(installed_outcomes) != _component_outcomes_value(persisted_outcomes):
                raise SemanticStoreError("semantic component outcomes changed during installation")
            _handoff_verified_generation(
                paths,
                installed_manifest,
                installed_outcomes,
                verified_document_bindings=_verified_document_bindings(snapshot),
                acceptance_guard=acceptance_guard,
                expected_active_semantic_generation_id=activation_fence.active_semantic_generation_id,
                expected_semantic_event_seq=activation_fence.semantic_event_seq,
                expected_projection_generation_id=activation_fence.projection_generation_id,
            )

        install_semantic_generation(staged, final, acceptance_callback=accept_installed)
        # Preserve the explicit clock seam for build callers; it never controls
        # identities or observable contents, which must remain deterministic.
        clock()
    except BaseException:
        # A failed staged generation is never activated.  The owned staging
        # object is intentionally retained for forensic inspection; it is not a
        # semantic final and cannot replace a previous valid generation.
        raise
    return deterministic


def _read_permanent_projection(paths: RetrievalPaths, generation_id: str) -> tuple[dict[str, str], ...]:
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        verify_cache_database(connection)
        rows = connection.execute(
            """SELECT document_id, record_id, content_sha256, search_text
            FROM retrieval_documents WHERE generation_id = ?
            ORDER BY document_seq""",
            (generation_id,),
        ).fetchall()
        return tuple({key: row[key] for key in row.keys()} for row in rows)
    finally:
        connection.close()


def _component_identity(
    config: SemanticConfig,
    capabilities: object,
    component: ModelComponent,
    provider_identity_matches: bool,
    *,
    capability_call_status: ProviderCallStatus,
):
    enabled = getattr(config, component.value)
    capability = getattr(capabilities, component.value)
    expected = getattr(config, f"expected_{component.value}_fingerprint")
    if not enabled:
        return None, _ComponentOutcome(component, "disabled", None, 0, 0, "disabled")
    if capability_call_status is not ProviderCallStatus.OK:
        if capability_call_status in _CAPABILITY_UNAVAILABLE_STATUSES:
            return None, _ComponentOutcome(
                component, "unavailable", None, 0, 0, capability_call_status.value
            )
        if capability_call_status is ProviderCallStatus.UNSUPPORTED:
            return None, _ComponentOutcome(component, "unsupported", None, 0, 0, "unsupported")
        return None, _ComponentOutcome(component, "failed", None, 0, 0, capability_call_status.value)
    if capability.state is not CapabilityState.AVAILABLE or capability.identity is None:
        status = capability.state.value
        reported_reason = getattr(capability, "reason", None)
        reason = reported_reason if reported_reason in _COMPONENT_REASONS else status
        return None, _ComponentOutcome(component, status, None, 0, 0, reason)
    if not provider_identity_matches:
        return None, _ComponentOutcome(component, "failed", None, 0, 0, "provider_fingerprint_mismatch")
    if capability.identity.fingerprint != expected:
        return None, _ComponentOutcome(component, "failed", None, 0, 0, "fingerprint_mismatch")
    return capability.identity, None


def _chunks(values: tuple[dict[str, str], ...], size: int) -> Iterable[tuple[dict[str, str], ...]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _provider_inputs(documents: tuple[dict[str, str], ...], max_chars: int | None) -> tuple[ProviderInput, ...] | None:
    texts = tuple(_semantic_input_text(item["search_text"]) for item in documents)
    if max_chars is not None and any(len(text) > max_chars for text in texts):
        return None
    return tuple(
        ProviderInput.from_text(item["document_id"], position, text)
        for position, (item, text) in enumerate(zip(documents, texts, strict=True))
    )


def _dense_rows(
    paths: RetrievalPaths,
    provider: ProviderCallController,
    documents: tuple[dict[str, str], ...],
    identity,
    initial: _ComponentOutcome | None,
    config: SemanticConfig,
    preprocessing_fingerprint: str,
) -> tuple[tuple[dict[str, object], ...], _ComponentOutcome]:
    if identity is None:
        assert initial is not None
        eligible_count = 0 if initial.status == "disabled" else len(documents)
        return (), replace(initial, eligible_count=eligible_count)
    rows = list(
        _reusable_dense_rows(
            paths,
            documents,
            identity.fingerprint,
            dimension=identity.dimension,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
    )
    reused = {row["document_id"] for row in rows}
    reason: str | None = None
    for batch in _chunks(tuple(item for item in documents if item["document_id"] not in reused), config.max_batch_items):
        inputs = _provider_inputs(batch, identity.max_input_chars)
        if inputs is None:
            reason = "input_too_long"
            continue
        response = provider.dense_documents(inputs, dimension=identity.dimension)
        if response.call.status is not ProviderCallStatus.OK:
            reason = response.call.status.value
            continue
        for item, source in zip(response.items, batch, strict=True):
            vector = _normal_f32le(item.values)
            rows.append({
                "document_id": source["document_id"], "content_sha256": source["content_sha256"],
                "model_fingerprint": identity.fingerprint,
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "dimension": len(item.values), "vector_blob": vector,
                "vector_sha256": hashlib.sha256(vector).hexdigest(),
            })
    status = "ready" if len(rows) == len(documents) else "partial" if rows else "failed"
    return tuple(rows), _ComponentOutcome(ModelComponent.DENSE, status, identity, len(documents), len(rows), reason)


def _sparse_rows(
    paths: RetrievalPaths,
    controller: ProviderCallController,
    documents: tuple[dict[str, str], ...],
    identity,
    initial: _ComponentOutcome | None,
    config: SemanticConfig,
    preprocessing_fingerprint: str,
) -> tuple[tuple[dict[str, object], ...], _ComponentOutcome]:
    if identity is None:
        assert initial is not None
        eligible_count = 0 if initial.status == "disabled" else len(documents)
        return (), replace(initial, eligible_count=eligible_count)
    rows = list(
        _reusable_sparse_rows(
            paths,
            documents,
            identity.fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
    )
    reused = {row["document_id"] for row in rows}
    reason: str | None = None
    for batch in _chunks(tuple(item for item in documents if item["document_id"] not in reused), config.max_batch_items):
        inputs = _provider_inputs(batch, identity.max_input_chars)
        if inputs is None:
            reason = "input_too_long"
            continue
        response = controller.sparse_documents(inputs)
        if response.call.status is not ProviderCallStatus.OK:
            reason = response.call.status.value
            continue
        for item, source in zip(response.items, batch, strict=True):
            for term, weight in item.weights:
                weight_hex = float(weight).hex()
                digest = hashlib.sha256(canonical_json({
                    "content_sha256": source["content_sha256"], "document_id": source["document_id"],
                    "model_fingerprint": identity.fingerprint,
                    "preprocessing_fingerprint": preprocessing_fingerprint,
                    "term": term, "weight_hex": weight_hex,
                }).encode("utf-8")).hexdigest()
                rows.append({
                    "document_id": source["document_id"], "content_sha256": source["content_sha256"],
                    "model_fingerprint": identity.fingerprint,
                    "preprocessing_fingerprint": preprocessing_fingerprint, "term": term,
                    "weight_hex": weight_hex, "posting_sha256": digest,
                })
    indexed_documents = {row["document_id"] for row in rows}
    status = "ready" if len(indexed_documents) == len(documents) else "partial" if indexed_documents else "failed"
    return tuple(rows), _ComponentOutcome(ModelComponent.SPARSE, status, identity, len(documents), len(indexed_documents), reason)


def _validate_component_outcomes(
    outcomes: tuple[_ComponentOutcome, ...],
) -> tuple[_ComponentOutcome, ...]:
    by_component: dict[ModelComponent, _ComponentOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, _ComponentOutcome) or outcome.component in by_component:
            raise SemanticStoreError("semantic component outcomes are invalid")
        if (
            not isinstance(outcome.status, str)
            or outcome.status not in _COMPONENT_STATUSES
            or type(outcome.eligible_count) is not int
            or type(outcome.indexed_count) is not int
            or not 0 <= outcome.indexed_count <= outcome.eligible_count
            or outcome.reason is not None
            and (
                not isinstance(outcome.reason, str)
                or outcome.reason not in _COMPONENT_REASONS
            )
        ):
            raise SemanticStoreError("semantic component outcome is invalid")
        fingerprint = None if outcome.identity is None else getattr(outcome.identity, "fingerprint", None)
        dimension = None if outcome.identity is None else getattr(outcome.identity, "dimension", None)
        if outcome.identity is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise SemanticStoreError("semantic component identity is invalid")
        if outcome.component is ModelComponent.DENSE and outcome.identity is not None and (
            type(dimension) is not int or dimension < 1
        ):
            raise SemanticStoreError("semantic component dimension is invalid")
        if outcome.component is not ModelComponent.DENSE and dimension is not None:
            raise SemanticStoreError("semantic component dimension is invalid")
        if outcome.status == "ready" and (
            outcome.indexed_count != outcome.eligible_count or outcome.reason is not None
        ):
            raise SemanticStoreError("ready semantic component outcome is invalid")
        if outcome.status == "partial" and (
            not 0 < outcome.indexed_count < outcome.eligible_count or outcome.reason is None
        ):
            raise SemanticStoreError("partial semantic component outcome is invalid")
        if outcome.status in {"ready", "partial"} and (
            outcome.component is ModelComponent.RERANKER or outcome.identity is None
        ):
            raise SemanticStoreError("semantic component identity is missing")
        if outcome.status == "disabled" and (
            outcome.indexed_count != 0 or outcome.reason != "disabled" or outcome.identity is not None
        ):
            raise SemanticStoreError("disabled semantic component outcome is invalid")
        if outcome.status == "unsupported" and (
            outcome.indexed_count != 0 or outcome.reason != "unsupported" or outcome.identity is not None
        ):
            raise SemanticStoreError("unsupported semantic component outcome is invalid")
        if outcome.status == "unavailable" and (
            outcome.indexed_count != 0
            or outcome.reason not in {
                "unavailable",
                "busy",
                "timeout",
                "circuit_open",
                "deadline_exhausted",
            }
            or outcome.identity is not None
        ):
            raise SemanticStoreError("unavailable semantic component outcome is invalid")
        if outcome.status == "failed" and outcome.reason is None:
            raise SemanticStoreError("failed semantic component outcome is invalid")
        by_component[outcome.component] = outcome
    if set(by_component) != set(ModelComponent):
        raise SemanticStoreError("semantic component outcomes are incomplete")
    return tuple(by_component[component] for component in ModelComponent)


def _component_outcomes_value(outcomes: tuple[_ComponentOutcome, ...]) -> dict[str, object]:
    validated = _validate_component_outcomes(outcomes)
    return {
        outcome.component.value: {
            "eligible_document_count": outcome.eligible_count,
            "indexed_document_count": outcome.indexed_count,
            "model_fingerprint": (
                None if outcome.identity is None else outcome.identity.fingerprint
            ),
            "reason": outcome.reason,
            "status": outcome.status,
            "vector_dimension": (
                None if outcome.identity is None else outcome.identity.dimension
            ),
        }
        for outcome in validated
    }


def _append_component_outcomes(
    connection: sqlite3.Connection, outcomes: tuple[_ComponentOutcome, ...]
) -> None:
    value = canonical_json(_component_outcomes_value(outcomes))
    connection.execute(
        "INSERT INTO semantic_schema_meta(key, value) VALUES (?, ?)",
        (_COMPONENT_OUTCOMES_META_KEY, value),
    )


def _component_reason_payload(outcomes: tuple[_ComponentOutcome, ...]) -> dict[str, object]:
    validated = _validate_component_outcomes(outcomes)
    mapping = {
        outcome.component.value: outcome.reason
        for outcome in validated
    }
    degraded = sorted(
        {
            outcome.reason
            for outcome in validated
            if outcome.status not in {"ready", "disabled"} and outcome.reason is not None
        }
    )
    return {
        "component_degraded_reasons": mapping,
        "degraded_reasons": degraded,
    }


def _handoff_verified_generation(
    paths: RetrievalPaths,
    manifest,
    outcomes: tuple[_ComponentOutcome, ...],
    *,
    verified_document_bindings: tuple[tuple[str, str, str, str], ...],
    acceptance_guard: Callable[[], None] | None = None,
    expected_active_semantic_generation_id: str | None,
    expected_semantic_event_seq: int,
    expected_projection_generation_id: str,
) -> None:
    """Append verified semantic lifecycle/coverage metadata in one short cache tx."""
    # Match RetrievalService.rebuild/catch_up lock ordering.  The process lock
    # is the one existing cross-process replacement linearization primitive;
    # semantic activation must not race a cache-file exchange based on older
    # semantic history.
    from .service import _lock_for, process_rebuild_lock

    with _lock_for(paths.database_path):
        paths.prepare_for_write()
        with process_rebuild_lock(paths.temp_dir):
            connection = connect_cache_database(paths.database_path)
            try:
                with immediate_transaction(connection, before_commit=acceptance_guard):
                    current_projection = connection.execute(
                        "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
                    ).fetchone()[0]
                    current_semantic = connection.execute(
                        "SELECT active_semantic_generation_id FROM retrieval_semantic_state WHERE singleton_id = 1"
                    ).fetchone()[0]
                    current_event_seq = connection.execute(
                        "SELECT COALESCE(MAX(event_seq), 0) FROM retrieval_semantic_generation_events"
                    ).fetchone()[0]
                    if (
                        manifest.projection_generation_id != expected_projection_generation_id
                        or current_projection != expected_projection_generation_id
                        or current_semantic != expected_active_semantic_generation_id
                        or current_event_seq != expected_semantic_event_seq
                    ):
                        raise SemanticStoreError("semantic activation fence changed")
                    projection_bindings = tuple(
                        (
                            row["document_id"],
                            row["record_id"],
                            row["content_sha256"],
                            ProviderInput.sha256_text(
                                _semantic_input_text(row["search_text"])
                            ),
                        )
                        for row in connection.execute(
                            """SELECT document_id, record_id, content_sha256, search_text
                            FROM retrieval_documents WHERE generation_id = ?
                            ORDER BY document_seq""",
                            (manifest.projection_generation_id,),
                        )
                    )
                    if projection_bindings != verified_document_bindings:
                        raise SemanticStoreError("semantic catalog binding mismatch")
                    existing = connection.execute(
                        "SELECT 1 FROM retrieval_semantic_generations WHERE semantic_generation_id = ?",
                        (manifest.semantic_generation_id,),
                    ).fetchone()
                    if existing is None:
                        append_semantic_generation(
                            connection,
                            semantic_generation_id=manifest.semantic_generation_id,
                            projection_generation_id=manifest.projection_generation_id,
                            provider_fingerprint=manifest.provider_fingerprint,
                            dense_model_fingerprint=manifest.dense_model_fingerprint,
                            sparse_model_fingerprint=manifest.sparse_model_fingerprint,
                            reranker_model_fingerprint=manifest.reranker_model_fingerprint,
                            preprocessing_fingerprint=manifest.preprocessing_fingerprint,
                            build_config_sha256=manifest.build_config_sha256,
                            manifest_relative_path=f"generations/{manifest.semantic_generation_id}/generation-manifest.json",
                            manifest_sha256=manifest.manifest_sha256,
                            semantic_database_sha256=manifest.semantic_database_sha256,
                            created_at=manifest.created_at,
                        )
                        for outcome in outcomes:
                            append_semantic_component(
                                connection,
                                semantic_generation_id=manifest.semantic_generation_id,
                                component=outcome.component.value,
                                status=outcome.status,
                                model_fingerprint=None if outcome.identity is None else outcome.identity.fingerprint,
                                preprocessing_fingerprint=manifest.preprocessing_fingerprint,
                                vector_dimension=None if outcome.identity is None else outcome.identity.dimension,
                                eligible_document_count=outcome.eligible_count,
                                indexed_document_count=outcome.indexed_count,
                                coverage_highwater=_component_coverage_highwater(
                                    manifest.indexed_highwater, outcome
                                ),
                                component_manifest_sha256=_component_manifest_sha256(manifest, outcome),
                                created_at=manifest.created_at,
                            )
                        append_semantic_generation_event(
                            connection,
                            semantic_generation_id=manifest.semantic_generation_id,
                            action="create",
                            created_at=manifest.created_at,
                            payload=_component_reason_payload(outcomes),
                        )
                        append_semantic_generation_event(connection, semantic_generation_id=manifest.semantic_generation_id, action="validate", created_at=manifest.created_at)
                        active = connection.execute("SELECT active_semantic_generation_id FROM retrieval_semantic_state WHERE singleton_id = 1").fetchone()[0]
                        if active is not None:
                            append_semantic_generation_event(connection, semantic_generation_id=active, action="retire", created_at=manifest.created_at)
                        append_semantic_generation_event(connection, semantic_generation_id=manifest.semantic_generation_id, action="activate", created_at=manifest.created_at)
                        rebuild_semantic_projections(connection)
                    verify_cache_database(connection)
                    if acceptance_guard is not None:
                        acceptance_guard()
            finally:
                connection.close()


def _verified_document_bindings(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str], ...]:
    """Bind the exact verified semantic bytes to the deterministic catalog."""
    return tuple(
        tuple(row)
        for row in connection.execute(
            """SELECT document_id, record_id, content_sha256, text_sha256
            FROM semantic_documents ORDER BY ordinal"""
        )
    )


def _capture_semantic_activation_fence(
    paths: RetrievalPaths, projection_generation_id: str
) -> _SemanticActivationFence:
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        verify_cache_database(connection)
        projection = connection.execute(
            "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
        ).fetchone()[0]
        if projection != projection_generation_id:
            raise SemanticStoreError("semantic activation projection changed")
        active = connection.execute(
            "SELECT active_semantic_generation_id FROM retrieval_semantic_state WHERE singleton_id = 1"
        ).fetchone()[0]
        event_seq = connection.execute(
            "SELECT COALESCE(MAX(event_seq), 0) FROM retrieval_semantic_generation_events"
        ).fetchone()[0]
        return _SemanticActivationFence(projection, active, event_seq)
    finally:
        connection.close()


def _component_coverage_highwater(
    indexed_highwater: dict[str, object], outcome: _ComponentOutcome
) -> dict[str, int]:
    """Return a proven contiguous boundary, never a partial row-count guess.

    A ready component covers the complete deterministic projection and may
    claim that generation's highwater.  Partial or unavailable output is not
    enough to prove any contiguous ledger prefix, so it is represented by the
    explicit all-zero (absent) highwater.
    """
    if outcome.status == "ready":
        return IndexHighwater.from_dict(indexed_highwater).to_dict()
    return IndexHighwater().to_dict()


def _component_manifest_sha256(manifest, outcome: _ComponentOutcome) -> str:
    return hashlib.sha256(canonical_json({
        "component": outcome.component.value,
        "indexed_document_count": outcome.indexed_count,
        "model_fingerprint": None if outcome.identity is None else outcome.identity.fingerprint,
        "reason": outcome.reason,
        "semantic_generation_id": manifest.semantic_generation_id,
        "status": outcome.status,
    }).encode("utf-8")).hexdigest()


def _outcomes_from_manifest(final: SemanticGenerationPaths, manifest) -> tuple[_ComponentOutcome, ...]:
    """Replay the immutable build outcomes; never infer a reason during retry."""
    with open_verified_semantic_snapshot(final) as (verified, connection):
        if verified != manifest:
            raise SemanticStoreError("semantic component manifest changed before consumption")
        return _outcomes_from_connection(connection, verified)


def _outcomes_from_connection(
    connection: sqlite3.Connection, manifest
) -> tuple[_ComponentOutcome, ...]:
    """Replay outcomes from the already verified, readonly content snapshot."""
    row = connection.execute(
            "SELECT value FROM semantic_schema_meta WHERE key = ?",
            (_COMPONENT_OUTCOMES_META_KEY,),
        ).fetchone()
    if row is None:
        raise SemanticStoreError("semantic component outcomes are missing")
    raw = row[0]
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SemanticStoreError("semantic component outcomes are invalid") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise SemanticStoreError("semantic component outcomes are not canonical")
    outcomes = _outcomes_from_value(value)
    _verify_persisted_outcomes(connection, manifest, outcomes)
    return outcomes


def _outcomes_from_value(value: dict[str, object]) -> tuple[_ComponentOutcome, ...]:
    if set(value) != {component.value for component in ModelComponent}:
        raise SemanticStoreError("semantic component outcomes are incomplete")
    outcomes: list[_ComponentOutcome] = []
    fields = {
        "eligible_document_count",
        "indexed_document_count",
        "model_fingerprint",
        "reason",
        "status",
        "vector_dimension",
    }
    for component in ModelComponent:
        item = value[component.value]
        if not isinstance(item, dict) or set(item) != fields:
            raise SemanticStoreError("semantic component outcome is invalid")
        fingerprint = item["model_fingerprint"]
        dimension = item["vector_dimension"]
        identity = None
        if fingerprint is not None:
            identity = SimpleNamespace(fingerprint=fingerprint, dimension=dimension)
        outcomes.append(
            _ComponentOutcome(
                component,
                item["status"],
                identity,
                item["eligible_document_count"],
                item["indexed_document_count"],
                item["reason"],
            )
        )
    return _validate_component_outcomes(tuple(outcomes))


def _verify_persisted_outcomes(
    connection: sqlite3.Connection, manifest, outcomes: tuple[_ComponentOutcome, ...]
) -> None:
    documents = manifest.component_counts["documents"]
    dense = connection.execute(
        """SELECT COUNT(DISTINCT document_id), MIN(dimension), MAX(dimension),
        COUNT(DISTINCT model_fingerprint), MIN(model_fingerprint),
        COUNT(DISTINCT preprocessing_fingerprint), MIN(preprocessing_fingerprint)
        FROM semantic_dense_vectors"""
    ).fetchone()
    sparse = connection.execute(
        """SELECT COUNT(DISTINCT document_id), COUNT(DISTINCT model_fingerprint),
        MIN(model_fingerprint), COUNT(DISTINCT preprocessing_fingerprint),
        MIN(preprocessing_fingerprint) FROM semantic_sparse_postings"""
    ).fetchone()
    by_component = {outcome.component: outcome for outcome in outcomes}
    dense_outcome = by_component[ModelComponent.DENSE]
    sparse_outcome = by_component[ModelComponent.SPARSE]
    reranker_outcome = by_component[ModelComponent.RERANKER]
    for outcome in outcomes:
        expected_eligible = 0 if outcome.status == "disabled" else documents
        if outcome.eligible_count != expected_eligible:
            raise SemanticStoreError("semantic component eligibility is invalid")
    if (
        dense_outcome.indexed_count != dense[0]
        or sparse_outcome.indexed_count != sparse[0]
        or reranker_outcome.indexed_count != 0
        or _outcome_fingerprint(dense_outcome) != manifest.dense_model_fingerprint
        or _outcome_fingerprint(sparse_outcome) != manifest.sparse_model_fingerprint
        or _outcome_fingerprint(reranker_outcome) != manifest.reranker_model_fingerprint
    ):
        raise SemanticStoreError("semantic component coverage is invalid")
    if dense[0] and (
        dense[1] != dense[2]
        or dense[3] != 1
        or dense[4] != _outcome_fingerprint(dense_outcome)
        or dense[5] != 1
        or dense[6] != manifest.preprocessing_fingerprint
        or dense_outcome.identity is None
        or dense_outcome.identity.dimension != dense[1]
    ):
        raise SemanticStoreError("semantic dense outcome is invalid")
    if sparse[0] and (
        sparse[1] != 1
        or sparse[2] != _outcome_fingerprint(sparse_outcome)
        or sparse[3] != 1
        or sparse[4] != manifest.preprocessing_fingerprint
        or sparse_outcome.identity is None
    ):
        raise SemanticStoreError("semantic sparse outcome is invalid")


def _outcome_fingerprint(outcome: _ComponentOutcome) -> str | None:
    return None if outcome.identity is None else outcome.identity.fingerprint


def _preprocessing_fingerprint() -> str:
    return canonical_contract_sha256(
        {
            "encoding": "utf-8",
            "identity": "semantic-input-v1",
            "input_source": "retrieval_documents.search_text",
            "newlines": "crlf-and-cr-to-lf",
            "terminal_lf": "remove-exactly-one",
            "truncation": "forbidden",
            "unicode": "NFKC",
        }
    )


def _semantic_input_text(value: str) -> str:
    if not isinstance(value, str):
        raise SemanticStoreError("semantic input must be text")
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return normalized[:-1] if normalized.endswith("\n") else normalized


def _reusable_dense_rows(
    paths: RetrievalPaths,
    documents: tuple[dict[str, str], ...],
    model_fingerprint: str,
    *,
    dimension: int,
    preprocessing_fingerprint: str,
) -> tuple[dict[str, object], ...]:
    """Copy only exact immutable identity matches from an old verified generation."""
    expected = {
        item["document_id"]: (
            item["content_sha256"],
            ProviderInput.sha256_text(_semantic_input_text(item["search_text"])),
        )
        for item in documents
    }
    for generation in sorted(paths.generations_dir.glob("sem_*")):
        try:
            semantic = SemanticGenerationPaths.for_generation(paths.generations_dir, generation.name)
            with open_verified_semantic_snapshot(semantic) as (manifest, connection):
                if (
                    manifest.dense_model_fingerprint != model_fingerprint
                    or manifest.preprocessing_fingerprint != preprocessing_fingerprint
                ):
                    continue
                rows = connection.execute(
                    """SELECT v.document_id, v.content_sha256, d.text_sha256,
                    v.model_fingerprint, v.preprocessing_fingerprint, v.dimension,
                    v.vector_blob, v.vector_sha256
                    FROM semantic_dense_vectors AS v JOIN semantic_documents AS d
                      ON d.document_id = v.document_id
                    ORDER BY v.vector_seq"""
                ).fetchall()
            reusable = tuple(
                {
                    key: row[key]
                    for key in row.keys()
                    if key != "text_sha256"
                }
                for row in rows
                if expected.get(row["document_id"])
                == (row["content_sha256"], row["text_sha256"])
                and row["model_fingerprint"] == model_fingerprint
                and row["preprocessing_fingerprint"] == preprocessing_fingerprint
                and row["dimension"] == dimension
            )
            if reusable:
                return reusable
        except (SemanticStoreError, sqlite3.Error, OSError):
            continue
    return ()


def _reusable_sparse_rows(
    paths: RetrievalPaths,
    documents: tuple[dict[str, str], ...],
    model_fingerprint: str,
    *,
    preprocessing_fingerprint: str,
) -> tuple[dict[str, object], ...]:
    expected = {
        item["document_id"]: (
            item["content_sha256"],
            ProviderInput.sha256_text(_semantic_input_text(item["search_text"])),
        )
        for item in documents
    }
    for generation in sorted(paths.generations_dir.glob("sem_*")):
        try:
            semantic = SemanticGenerationPaths.for_generation(paths.generations_dir, generation.name)
            with open_verified_semantic_snapshot(semantic) as (manifest, connection):
                if (
                    manifest.sparse_model_fingerprint != model_fingerprint
                    or manifest.preprocessing_fingerprint != preprocessing_fingerprint
                ):
                    continue
                rows = connection.execute(
                    """SELECT p.document_id, p.content_sha256, d.text_sha256,
                    p.model_fingerprint, p.preprocessing_fingerprint, p.term,
                    p.weight_hex, p.posting_sha256
                    FROM semantic_sparse_postings AS p JOIN semantic_documents AS d
                      ON d.document_id = p.document_id
                    ORDER BY p.posting_seq"""
                ).fetchall()
            reusable = tuple(
                {
                    key: row[key]
                    for key in row.keys()
                    if key != "text_sha256"
                }
                for row in rows
                if expected.get(row["document_id"])
                == (row["content_sha256"], row["text_sha256"])
                and row["model_fingerprint"] == model_fingerprint
                and row["preprocessing_fingerprint"] == preprocessing_fingerprint
            )
            if reusable:
                return reusable
        except (SemanticStoreError, sqlite3.Error, OSError):
            continue
    return ()


def _normal_f32le(values: tuple[float, ...]) -> bytes:
    magnitude = math.sqrt(math.fsum(float(value) * float(value) for value in values))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise SemanticStoreError("dense vector is not normalizable")
    normalized = tuple(float(value) / magnitude for value in values)
    return struct.pack(f"<{len(normalized)}f", *normalized)
