from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
import unicodedata
import uuid
from typing import Callable, Mapping

from codex_memory.blob_store import BlobStore
from codex_memory.memory_scope import ScopeContext
from codex_memory.ledger_common import VerificationError
from codex_memory.paths import (
    RuntimePathError,
    path_is_redirect,
    require_safe_directory,
    require_safe_file,
)

from .cache_database import (
    CacheCorruptError,
    CacheUnsupportedError,
    append_generation,
    append_generation_event,
    connect_cache_database,
    copy_history,
    copy_semantic_history,
    copy_v2_immutable_rows,
    deterministic_generation_id as _cache_generation_id,
    _event_values as _cache_event_values,
    inspect_cache_schema_version,
    immediate_transaction,
    initialize_schema,
    read_history_for_rebuild,
    read_semantic_history_for_rebuild,
    read_v2_immutable_rows,
    rebuild_projections,
    verify_cache_database,
    verify_provider_attempt_history,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
)
from .canonical_snapshot import (
    CanonicalSnapshot,
    CanonicalSnapshotInvalidError,
    CanonicalSnapshotUnavailableError,
    capture_canonical_snapshot,
    open_verified_canonical_snapshot,
)
from .contracts import (
    CandidateChannel,
    CandidateTrace,
    DegradedReason,
    IndexGeneration,
    IndexHighwater,
    RetrievalMode,
    RetrievalPolicy,
    RetrievalQueryResult,
    RetrievalRequest,
    RetrievedCandidate,
    RetrievalResult,
    RetrievalStatus,
)
from .errors import RetrievalValidationError
from .deadline import DeadlineBudget
from .fusion import FusionInput, fuse_candidates
from .provider import ProviderCallStatus, ProviderInput, RetrievalModelProvider
from .provider_health import ProviderHealthStore
from .provider_control import ProviderCallController
from .rerank import RerankResult, rerank_candidates
from .mandatory import mandatory_candidates, temporary_candidates
from .exact import _scope, _visible, exact_candidates, symbol_candidates
from .bm25 import bm25_candidates
from .recent import recent_candidates
from .dense import dense_candidates
from .sparse import sparse_candidates
from .paths import RetrievalPaths
from .semantic_paths import SemanticGenerationPaths
from .semantic_store import SemanticStoreError, open_verified_semantic_snapshot
from .projector import (
    ProjectionBatch,
    ProjectionDigest,
    ProjectionDigestBuilder,
    changed_record_ids,
    project_full_snapshot,
    project_stale_delta,
    streaming_projection_digest,
)
from . import index_store


_PROVIDER = "unconfigured:p2.1"
_MODEL = "unconfigured:p2.1"
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_LOCK_MAGIC = b"codex-memory-retrieval-lock-v1\n"
# Catch-up is deliberately bounded: large backlogs require an explicit full
# rebuild instead of allowing a partial generation to claim the new highwater.
MAX_CATCH_UP_AFFECTED_RECORDS = 4_096
CATCH_UP_COPY_BATCH_SIZE = 256


@dataclass(frozen=True, slots=True)
class _OwnedTemporary:
    path: Path
    identity: tuple[int, int]
    link_count: int = 1


@dataclass(frozen=True, slots=True)
class _VerifiedBackup:
    replace_source: _OwnedTemporary
    recovery_guard: _OwnedTemporary


@dataclass(slots=True)
class _InstallHandoff:
    source: _OwnedTemporary
    previous: _VerifiedBackup | None
    retain_previous: bool = False


def deterministic_generation_id(snapshot: CanonicalSnapshot) -> str:
    """Derive the P2.1 generation name from canonical, never wall-clock, data."""
    return _cache_generation_id(
        snapshot.highwater, snapshot.content_fingerprint, schema_version=SCHEMA_VERSION_V3
    )


@dataclass(frozen=True, slots=True)
class SemanticQueryInputs:
    """Already-produced semantic query values for the explicit offline seam.

    This object deliberately carries no provider, endpoint, deadline, prompt,
    or document text.  Task 5 may produce these values after its reliability
    checks; ordinary P2.2 calls pass ``None`` and remain byte-compatible.
    """

    dense_vector: tuple[float, ...] | None = None
    sparse_weights: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class RerankQueryResult:
    """Explicit shadow/offline rerank observation; legacy query bytes stay intact."""

    result: RetrievalQueryResult
    rerank: RerankResult | None


class RetrievalService:
    """Read, compare, and atomically replace the disposable retrieval cache."""

    def __init__(self, paths: RetrievalPaths):
        self.paths = paths

    def query(
        self,
        request: RetrievalRequest,
        *,
        policy: RetrievalPolicy = RetrievalPolicy(),
        semantic_query: SemanticQueryInputs | None = None,
        deadline=None,
        observer=None,
        bound_reads: bool = False,
    ) -> RetrievalQueryResult:
        """Read a verified snapshot and disposable cache; this never rebuilds or writes."""
        if not isinstance(request, RetrievalRequest) or not isinstance(policy, RetrievalPolicy):
            raise RetrievalValidationError("invalid retrieval query")
        if semantic_query is not None and not isinstance(semantic_query, SemanticQueryInputs):
            raise RetrievalValidationError("invalid semantic query inputs")
        if request.mode is RetrievalMode.ACTIVE:
            raise RetrievalValidationError("active retrieval mode is unavailable")
        # ``SHADOW`` is an explicitly output-inert observation mode.  It may
        # use the same disposable read path as OFF, but it never makes ACTIVE
        # context injection available.
        if request.mode not in {RetrievalMode.OFF, RetrievalMode.SHADOW}:
            raise RetrievalValidationError("retrieval mode is unavailable")
        try:
            with open_verified_canonical_snapshot(self.paths, **(
                {"deadline": deadline, "observer": observer} if observer is not None else (
                    {"deadline": deadline} if bound_reads else {})
            )) as (canonical, snapshot):
                direct, direct_trace = _direct_prefix(canonical, self.paths, request, policy)
                try:
                    self._validate_retrieval_paths()
                except RuntimePathError:
                    return _query_result(RetrievalStatus.CORRUPT, snapshot, None, True,
                                         (DegradedReason.PATH_UNSAFE,), direct, direct_trace)
                except OSError:
                    return _query_result(RetrievalStatus.CORRUPT, snapshot, None, True,
                                         (DegradedReason.PATH_UNAVAILABLE,), direct, direct_trace)
                try:
                    if not self.paths.database_path.exists():
                        return _query_result(RetrievalStatus.ABSENT, snapshot, None, True,
                                             (DegradedReason.INDEX_ABSENT,), direct, direct_trace)
                    cache = connect_cache_database(self.paths.database_path, readonly=True,
                        **({"busy_timeout_ms": 0} if bound_reads and observer is None else {}))
                except (sqlite3.Error, OSError, ValueError):
                    return _query_result(RetrievalStatus.CORRUPT, snapshot, None, True,
                                         (DegradedReason.INDEX_CORRUPT,), direct, direct_trace)
                try:
                    if bound_reads and observer is None:
                        from .injection_audit import _deadline_check
                        import time
                        _deadline_check(deadline)
                        cache.set_progress_handler(lambda: int(time.monotonic_ns() >= deadline.deadline_ns), 100)
                    verify_cache_database(cache)
                    generation = _active_generation(cache)
                    if generation is None:
                        raise CacheCorruptError("active generation missing")
                    stale = generation.canonical_highwater != snapshot.highwater or generation.canonical_content_fingerprint != snapshot.content_fingerprint
                    changed_ids = changed_record_ids(canonical, generation.indexed_highwater, limit=policy.stale_delta_limit + 1) if stale else ()
                    if len(changed_ids) > policy.stale_delta_limit:
                        return _query_result(RetrievalStatus.STALE, snapshot, generation, True,
                                             (DegradedReason.INDEX_STALE, DegradedReason.STALE_DELTA_EXCEEDED), direct, direct_trace)
                    changed = project_stale_delta(canonical, since=generation.indexed_highwater, limit=policy.stale_delta_limit) if stale else None
                    # Direct authority prefixes are already selected and must
                    # not re-enter ordinary fusion under a second slot.
                    excluded = set(changed_ids) | {
                        candidate.candidate_id
                        for candidate in direct
                        if candidate.candidate_kind == "record"
                    }
                    inputs, reasons = _cache_inputs(cache, request, generation.generation_id, policy, excluded)
                    if semantic_query is not None and (
                        semantic_query.dense_vector is not None or semantic_query.sparse_weights is not None
                    ):
                        semantic_inputs, semantic_reasons = _active_semantic_fusion_inputs(
                            self.paths,
                            cache,
                            generation.generation_id,
                            _eligible_rows(cache, request, generation.generation_id, excluded),
                            semantic_query,
                            policy,
                        )
                        inputs.extend(semantic_inputs)
                        reasons.extend(semantic_reasons)
                    if changed and changed.record_ids:
                        direct_record_ids = {
                            candidate.candidate_id
                            for candidate in direct
                            if candidate.candidate_kind == "record"
                        }
                        delta = _delta_rows(
                            canonical, snapshot, request,
                            tuple(identifier for identifier in changed.record_ids if identifier not in direct_record_ids),
                        )
                        if delta:
                            inputs.append(FusionInput(CandidateChannel.STALE_DELTA, delta))
                            reasons.append(DegradedReason.STALE_DELTA_USED)
                    fused = fuse_candidates(tuple(inputs), policy)
                    return _query_result(
                        RetrievalStatus.STALE if stale else RetrievalStatus.CURRENT, snapshot, generation, stale,
                        tuple(([DegradedReason.INDEX_STALE] if stale else []) + reasons),
                        direct + fused.candidates, direct_trace + fused.trace,
                    )
                except CacheUnsupportedError:
                    return _query_result(RetrievalStatus.UNSUPPORTED, snapshot, None, True, (DegradedReason.SCHEMA_UNSUPPORTED,), direct, direct_trace)
                except (CacheCorruptError, sqlite3.Error, OSError, ValueError):
                    return _query_result(RetrievalStatus.CORRUPT, snapshot, None, True, (DegradedReason.INDEX_CORRUPT,), direct, direct_trace)
                finally:
                    cache.close()
        except CanonicalSnapshotUnavailableError as exc:
            if observer is not None:
                observer.canonical_failure(exc)
            return _query_result(RetrievalStatus.CORRUPT, None, None, True, (DegradedReason.CANONICAL_UNAVAILABLE,), (), ())
        except CanonicalSnapshotInvalidError as exc:
            if observer is not None:
                observer.canonical_failure(exc)
            return _query_result(RetrievalStatus.CORRUPT, None, None, True, (DegradedReason.CANONICAL_INVALID,), (), ())

    def query_with_rerank(
        self,
        request: RetrievalRequest,
        *,
        provider: RetrievalModelProvider,
        documents: Mapping[str, ProviderInput],
        deadline_ms: int,
        health_store: ProviderHealthStore,
        policy: RetrievalPolicy = RetrievalPolicy(),
        expected_provider_fingerprint: str | None = None,
        expected_model_fingerprint: str | None = None,
        deadline: DeadlineBudget | None = None,
        now_ns=None,
        provider_controller: object | None = None,
        base_result: RetrievalQueryResult | None = None,
    ) -> RerankQueryResult:
        """Run optional reranking after the immutable P2.2 query has completed.

        The returned ``result`` is exactly the legacy deterministic result. The
        separate rerank observation makes shadow use output-inert while letting
        callers inspect the safe reordered candidate slots.  No canonical or
        cache transaction remains open when the provider is called.
        """
        if now_ns is None:
            now_ns = time.monotonic_ns
        if not callable(now_ns):
            raise RetrievalValidationError("now_ns must be callable")
        if deadline is not None and not isinstance(deadline, DeadlineBudget):
            raise RetrievalValidationError("deadline must be DeadlineBudget or null")
        budget = deadline or DeadlineBudget.start(deadline_ms, now_ns=now_ns)
        if base_result is not None and not isinstance(base_result, RetrievalQueryResult):
            raise RetrievalValidationError("base_result must be RetrievalQueryResult or null")
        result = base_result if base_result is not None else self.query(request, policy=policy)
        if not result.candidates:
            return RerankQueryResult(result, None)
        if budget.remaining_ms(now_ns=now_ns, cap_ms=deadline_ms) == 0:
            return RerankQueryResult(result, RerankResult(result.candidates, ProviderCallStatus.DEADLINE_EXHAUSTED, None))
        controller = provider_controller
        if controller is None:
            controller = ProviderCallController(
                provider,
                budget,
                health_store,
                expected_provider_fingerprint,
                now_ns=now_ns,
                expected_reranker_fingerprint=expected_model_fingerprint,
            )
        try:
            query = ProviderInput.from_text("query", 0, request.query or "")
            reranked = rerank_candidates(
                provider, query, result.candidates, documents,
                top_n=policy.rerank_top_n, deadline_ms=deadline_ms, deadline=budget, now_ns=now_ns,
                expected_provider_fingerprint=expected_provider_fingerprint,
                expected_model_fingerprint=expected_model_fingerprint,
                reserve_ms=40,
                controller=controller,
            )
            return RerankQueryResult(result, reranked)
        except Exception:
            return RerankQueryResult(result, RerankResult(result.candidates, ProviderCallStatus.UNAVAILABLE, None))

    def status(self) -> RetrievalResult:
        try:
            self._validate_retrieval_paths()
        except RuntimePathError:
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.PATH_UNSAFE)
        except OSError:
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.PATH_UNAVAILABLE)
        snapshot, canonical_reason = self._snapshot_or_none()
        if snapshot is None:
            return _empty_result(RetrievalStatus.CORRUPT, canonical_reason)
        try:
            database_exists = self.paths.database_path.exists()
        except OSError:
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.PATH_UNAVAILABLE, snapshot)
        if not database_exists:
            return _empty_result(RetrievalStatus.ABSENT, DegradedReason.INDEX_ABSENT, snapshot)
        try:
            connection = connect_cache_database(self.paths.database_path, readonly=True)
        except (sqlite3.Error, OSError, ValueError):
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.PATH_UNAVAILABLE, snapshot)
        try:
            try:
                verify_cache_database(connection)
                generation = _active_generation(connection)
            finally:
                connection.close()
        except CacheUnsupportedError:
            return _empty_result(RetrievalStatus.UNSUPPORTED, DegradedReason.SCHEMA_UNSUPPORTED, snapshot)
        except CacheCorruptError:
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.INDEX_CORRUPT, snapshot)
        except (sqlite3.Error, OSError):
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.PATH_UNAVAILABLE, snapshot)
        if generation is None:
            return _empty_result(RetrievalStatus.CORRUPT, DegradedReason.INDEX_CORRUPT, snapshot)
        status = (
            RetrievalStatus.CURRENT
            if generation.canonical_highwater == snapshot.highwater
            and generation.canonical_content_fingerprint == snapshot.content_fingerprint
            else RetrievalStatus.STALE
        )
        reason = None if status is RetrievalStatus.CURRENT else DegradedReason.INDEX_STALE
        return RetrievalResult(
            status=status,
            mode=RetrievalMode.OFF,
            generation=generation,
            canonical_highwater=snapshot.highwater,
            indexed_highwater=generation.indexed_highwater,
            degraded_reason=reason,
        )

    def verify(self) -> RetrievalResult:
        """Read-only verification boundary; callers decide the exit-code policy."""
        return self.status()

    def init(self) -> RetrievalResult:
        existing = self.status()
        if existing.status is not RetrievalStatus.ABSENT:
            return existing
        return self.rebuild()

    def rebuild_semantic(self, provider, config, *, clock=time.monotonic_ns) -> RetrievalResult:
        """Build optional semantic bytes without changing legacy retrieval behavior."""
        from .semantic_index import rebuild_semantic_index

        return rebuild_semantic_index(self.paths, provider, config, clock=clock)

    def catch_up_semantic(self, provider, config, *, clock=time.monotonic_ns) -> RetrievalResult:
        from .semantic_index import catch_up_semantic_index

        return catch_up_semantic_index(self.paths, provider, config, clock=clock)

    def rebuild(self, *, failure_injector: Callable[[str], None] | None = None) -> RetrievalResult:
        """Safely construct cache schema one; a failure never replaces a valid DB."""
        lock = _lock_for(self.paths.database_path)
        with lock:
            self.paths.prepare_for_write()
            with process_rebuild_lock(self.paths.temp_dir):
                existing = self.status()
                source_schema_version: int | None = None
                if self.paths.database_path.exists():
                    try:
                        connection = connect_cache_database(self.paths.database_path, readonly=True)
                        try:
                            version = inspect_cache_schema_version(connection)
                            if version == SCHEMA_VERSION_V4:
                                try:
                                    verify_provider_attempt_history(connection)
                                except CacheCorruptError:
                                    return _empty_result(
                                        RetrievalStatus.CORRUPT,
                                        DegradedReason.INDEX_CORRUPT,
                                    )
                        finally:
                            connection.close()
                    except (CacheCorruptError, OSError, ValueError, sqlite3.Error):
                        version = None
                    if version in {1, 2, 3}:
                        source_schema_version = version
                    elif existing.status is RetrievalStatus.UNSUPPORTED:
                        return existing
                if (
                    existing.status is RetrievalStatus.CURRENT
                    and failure_injector is None
                    and source_schema_version is None
                ):
                    return existing
                if existing.degraded_reason in {
                    DegradedReason.CANONICAL_UNAVAILABLE,
                    DegradedReason.CANONICAL_INVALID,
                }:
                    return existing
                for attempt in range(2):
                    try:
                        # Retain the existing typed preflight seam.  The snapshot used
                        # for generation identity and projection is the context below.
                        self._capture_for_rebuild()
                        with open_verified_canonical_snapshot(self.paths) as (
                            canonical_connection,
                            snapshot,
                        ):
                            batch = project_full_snapshot(canonical_connection, snapshot)
                    except CanonicalSnapshotUnavailableError:
                        return _empty_result(
                            RetrievalStatus.CORRUPT,
                            DegradedReason.CANONICAL_UNAVAILABLE,
                        )
                    except CanonicalSnapshotInvalidError:
                        return _empty_result(
                            RetrievalStatus.CORRUPT,
                            DegradedReason.CANONICAL_INVALID,
                        )
                    history, prior_active = self._valid_history_or_empty()
                    semantic_history = self._semantic_history_or_empty()
                    if source_schema_version == 3 and prior_active is not None:
                        # v4 is a single-file control-plane extension over the
                        # v3 retrieval generation contract, so the active
                        # deterministic generation id is unchanged. Rebuild
                        # that active projection in the new file while retaining
                        # all retired immutable history and control history.
                        generations, events = history
                        history = (
                            [
                                row for row in generations
                                if row["generation_id"] != prior_active
                            ],
                            [
                                row for row in events
                                if row["generation_id"] != prior_active
                            ],
                        )
                        prior_active = None
                    v2_immutable_rows = (
                        self._v2_immutable_rows_or_empty()
                        if source_schema_version == 2 else {}
                    )
                    if existing.status is RetrievalStatus.UNSUPPORTED:
                        pass
                    temporary = self._owned_temporary_path()
                    installation: _InstallHandoff | None = None
                    try:
                        connection = connect_cache_database(temporary.path)
                        try:
                            with immediate_transaction(connection):
                                self._inject(failure_injector, "ddl")
                                initialize_schema(connection, in_transaction=True)
                                copy_history(
                                    connection, *history,
                                    source_schema_version=source_schema_version or SCHEMA_VERSION_V3,
                                )
                                copy_semantic_history(connection, semantic_history)
                                if v2_immutable_rows:
                                    copy_v2_immutable_rows(connection, v2_immutable_rows)
                                self._inject(failure_injector, "write")
                                if prior_active is not None:
                                    append_generation_event(
                                        connection,
                                        generation_id=prior_active,
                                        action="retire",
                                        created_at=snapshot.source_timestamp,
                                    )
                                target = deterministic_generation_id(snapshot)
                                append_generation(
                                    connection,
                                    generation_id=target,
                                    canonical_highwater=snapshot.highwater,
                                    indexed_highwater=snapshot.highwater,
                                    canonical_content_fingerprint=snapshot.content_fingerprint,
                                    provider_fingerprint=_PROVIDER,
                                    model_fingerprint=_MODEL,
                                    created_at=snapshot.source_timestamp,
                                )
                                _write_projection(
                                    connection,
                                    generation_id=target,
                                    batch=batch,
                                    created_at=snapshot.source_timestamp,
                                )
                                self._inject(failure_injector, "projection")
                                for action in ("create", "validate", "activate"):
                                    append_generation_event(
                                        connection,
                                        generation_id=target,
                                        action=action,
                                        created_at=snapshot.source_timestamp,
                                    )
                                rebuild_projections(connection)
                        finally:
                            connection.close()
                        self._inject(failure_injector, "verify")
                        self._verify_temporary(temporary.path)
                        after = self._capture_for_rebuild()
                        if after.highwater != snapshot.highwater or after.content_fingerprint != snapshot.content_fingerprint:
                            if attempt == 0:
                                continue
                            return _empty_result(RetrievalStatus.STALE, DegradedReason.INDEX_STALE, after)
                        self._inject(failure_injector, "replace")
                        if not _same_file(temporary.path, self.paths.temp_dir, temporary.identity):
                            raise CacheCorruptError("temporary path changed before replace")
                        installation = self._verified_install_handoff(
                            temporary,
                            preserve_previous=existing.status in {
                                RetrievalStatus.CURRENT,
                                RetrievalStatus.STALE,
                            },
                        )
                        return self._install_and_verify(installation)
                    finally:
                        if installation is not None:
                            self._remove_owned_temporary(installation.source)
                            if installation.previous is not None and not installation.retain_previous:
                                self._remove_owned_temporary(
                                    installation.previous.replace_source
                                )
                                self._remove_owned_temporary(
                                    installation.previous.recovery_guard
                                )
                        self._remove_owned_temporary(temporary)
                raise AssertionError("unreachable rebuild retry")

    def catch_up(self, *, failure_injector: Callable[[str], None] | None = None) -> RetrievalResult:
        """Install a fresh generation after reprojecting only canonical changes."""
        lock = _lock_for(self.paths.database_path)
        with lock:
            self.paths.prepare_for_write()
            with process_rebuild_lock(self.paths.temp_dir):
                existing = self.status()
                if existing.status is RetrievalStatus.CURRENT and failure_injector is None:
                    return existing
                if existing.status is RetrievalStatus.ABSENT:
                    return existing
                if (
                    existing.status is not RetrievalStatus.STALE
                    or existing.generation is None
                    or existing.degraded_reason in {
                        DegradedReason.CANONICAL_UNAVAILABLE,
                        DegradedReason.CANONICAL_INVALID,
                    }
                ):
                    return existing
                for attempt in range(2):
                    try:
                        self._capture_for_rebuild()
                        with open_verified_canonical_snapshot(self.paths) as (
                            canonical_connection,
                            snapshot,
                        ):
                            affected = changed_record_ids(
                                canonical_connection,
                                existing.generation.indexed_highwater,
                                limit=MAX_CATCH_UP_AFFECTED_RECORDS + 1,
                            )
                            if len(affected) > MAX_CATCH_UP_AFFECTED_RECORDS:
                                return existing
                            history, prior_active = self._valid_history_or_empty()
                            semantic_history = self._semantic_history_or_empty()
                            if prior_active != existing.generation.generation_id:
                                return existing
                            temporary = self._owned_temporary_path()
                            installation: _InstallHandoff | None = None
                            try:
                                connection = connect_cache_database(temporary.path)
                                try:
                                    with immediate_transaction(connection):
                                        self._inject(failure_injector, "ddl")
                                        initialize_schema(connection, in_transaction=True)
                                        copy_history(connection, *history)
                                        copy_semantic_history(connection, semantic_history)
                                        self._inject(failure_injector, "write")
                                        append_generation_event(
                                            connection,
                                            generation_id=prior_active,
                                            action="retire",
                                            created_at=snapshot.source_timestamp,
                                        )
                                        target = deterministic_generation_id(snapshot)
                                        append_generation(
                                            connection,
                                            generation_id=target,
                                            canonical_highwater=snapshot.highwater,
                                            indexed_highwater=snapshot.highwater,
                                            canonical_content_fingerprint=snapshot.content_fingerprint,
                                            provider_fingerprint=_PROVIDER,
                                            model_fingerprint=_MODEL,
                                            created_at=snapshot.source_timestamp,
                                        )
                                        self._copy_unaffected_projection(
                                            connection,
                                            source_generation=prior_active,
                                            target_generation=target,
                                            affected_record_ids=affected,
                                        )
                                        affected_set = frozenset(affected)
                                        affected_document_ids: set[str] = set()

                                        def consume(kind: str, value: object | None) -> None:
                                            if kind == "document":
                                                document = value
                                                if document.record_id in affected_set:
                                                    affected_document_ids.add(document.document_id)
                                                    _append_projected_document(
                                                        connection, target, document
                                                    )
                                            elif kind == "source":
                                                source = value
                                                if source.document_id in affected_document_ids:
                                                    _append_projected_source(connection, target, source)
                                            elif kind == "symbol":
                                                symbol = value
                                                if symbol.document_id in affected_document_ids:
                                                    _append_projected_symbol(connection, target, symbol)

                                        projection = streaming_projection_digest(
                                            canonical_connection,
                                            snapshot,
                                            row_consumer=consume,
                                        )
                                        _write_projection_capability(
                                            connection,
                                            generation_id=target,
                                            batch=projection,
                                            created_at=snapshot.source_timestamp,
                                        )
                                        _validate_final_projection(
                                            connection,
                                            generation_id=target,
                                            expected=projection,
                                            snapshot=snapshot,
                                        )
                                        self._inject(failure_injector, "projection")
                                        for action in ("create", "validate", "activate"):
                                            append_generation_event(
                                                connection,
                                                generation_id=target,
                                                action=action,
                                                created_at=snapshot.source_timestamp,
                                            )
                                        rebuild_projections(connection)
                                finally:
                                    connection.close()
                                self._inject(failure_injector, "verify")
                                self._verify_temporary(temporary.path)
                                after = self._capture_for_rebuild()
                                if (
                                    after.highwater != snapshot.highwater
                                    or after.content_fingerprint != snapshot.content_fingerprint
                                ):
                                    if attempt == 0:
                                        continue
                                    return _empty_result(
                                        RetrievalStatus.STALE,
                                        DegradedReason.INDEX_STALE,
                                        after,
                                    )
                                self._inject(failure_injector, "replace")
                                if not _same_file(
                                    temporary.path, self.paths.temp_dir, temporary.identity
                                ):
                                    raise CacheCorruptError("temporary path changed before replace")
                                installation = self._verified_install_handoff(
                                    temporary, preserve_previous=True
                                )
                                return self._install_and_verify(installation)
                            finally:
                                if installation is not None:
                                    self._remove_owned_temporary(installation.source)
                                    if (
                                        installation.previous is not None
                                        and not installation.retain_previous
                                    ):
                                        self._remove_owned_temporary(
                                            installation.previous.replace_source
                                        )
                                        self._remove_owned_temporary(
                                            installation.previous.recovery_guard
                                        )
                                self._remove_owned_temporary(temporary)
                    except CanonicalSnapshotUnavailableError:
                        return _empty_result(
                            RetrievalStatus.CORRUPT,
                            DegradedReason.CANONICAL_UNAVAILABLE,
                        )
                    except CanonicalSnapshotInvalidError:
                        return _empty_result(
                            RetrievalStatus.CORRUPT,
                            DegradedReason.CANONICAL_INVALID,
                        )
                raise AssertionError("unreachable catch-up retry")

    def _copy_unaffected_projection(
        self,
        destination: sqlite3.Connection,
        *,
        source_generation: str,
        target_generation: str,
        affected_record_ids: tuple[str, ...],
    ) -> None:
        """Copy only stable rows from the active cache into a new generation."""
        source = connect_cache_database(self.paths.database_path, readonly=True)
        try:
            verify_cache_database(source)
            affected = set(affected_record_ids)
            documents = source.execute(
                "SELECT * FROM retrieval_documents WHERE generation_id = ? ORDER BY document_seq",
                (source_generation,),
            )
            for rows in _batches(documents):
                for row in rows:
                    if row["record_id"] in affected:
                        continue
                    index_store.append_document(
                        destination,
                        document_id=row["document_id"], generation_id=target_generation,
                        record_id=row["record_id"], record_seq=row["record_seq"], record_type=row["record_type"],
                        scope_kind=row["scope_kind"], scope_key=row["scope_key"], topic_key=row["topic_key"],
                        statement=row["statement"], value_json=row["value_json"], search_text=row["search_text"],
                        state=row["state"], created_at=row["created_at"],
                        last_memory_event_seq=row["last_memory_event_seq"], evidence_count=row["evidence_count"],
                        content_sha256=row["content_sha256"], document_sha256=row["document_sha256"],
                    )
            sources = source.execute(
                """SELECT s.*, d.record_id FROM retrieval_document_sources AS s
                JOIN retrieval_documents AS d ON d.document_id = s.document_id
                WHERE d.generation_id = ? ORDER BY s.source_seq""",
                (source_generation,),
            )
            for rows in _batches(sources):
                for row in rows:
                    if row["record_id"] in affected:
                        continue
                    index_store.append_document_source(
                        destination, source_id=row["source_id"], generation_id=target_generation,
                        document_id=row["document_id"], ordinal=row["ordinal"],
                        safe_reference=row["safe_reference"], source_sha256=row["source_sha256"],
                    )
            symbols = source.execute(
                """SELECT s.*, d.record_id FROM retrieval_symbols AS s
                JOIN retrieval_documents AS d ON d.document_id = s.document_id
                WHERE d.generation_id = ? ORDER BY s.symbol_seq""",
                (source_generation,),
            )
            for rows in _batches(symbols):
                for row in rows:
                    if row["record_id"] in affected:
                        continue
                    index_store.append_symbol(
                        destination, symbol_id=row["symbol_id"], generation_id=target_generation,
                        document_id=row["document_id"], ordinal=row["ordinal"], symbol_kind=row["symbol_kind"],
                        normalized_symbol=row["normalized_symbol"], display_symbol=row["display_symbol"],
                        symbol_sha256=row["symbol_sha256"],
                    )
        finally:
            source.close()

    def _snapshot_or_none(self) -> tuple[CanonicalSnapshot | None, DegradedReason]:
        canonical = self.paths.canonical_database_path
        try:
            if path_is_redirect(self.paths.runtime_root):
                return None, DegradedReason.CANONICAL_INVALID
            if not self.paths.runtime_root.exists():
                return None, DegradedReason.CANONICAL_UNAVAILABLE
            require_safe_directory(self.paths.runtime_root, self.paths.repo_root)
            canonical_dir = canonical.parent
            if path_is_redirect(canonical_dir):
                return None, DegradedReason.CANONICAL_INVALID
            if not canonical_dir.exists():
                return None, DegradedReason.CANONICAL_UNAVAILABLE
            require_safe_directory(canonical_dir, self.paths.runtime_root)
            require_safe_file(canonical, canonical_dir, require_exists=False)
        except RuntimePathError:
            return None, DegradedReason.CANONICAL_INVALID
        except OSError:
            return None, DegradedReason.CANONICAL_UNAVAILABLE
        try:
            if not canonical.exists():
                return None, DegradedReason.CANONICAL_UNAVAILABLE
        except OSError:
            return None, DegradedReason.CANONICAL_UNAVAILABLE
        try:
            return capture_canonical_snapshot(self.paths), DegradedReason.CANONICAL_INVALID
        except CanonicalSnapshotUnavailableError:
            return None, DegradedReason.CANONICAL_UNAVAILABLE
        except CanonicalSnapshotInvalidError:
            return None, DegradedReason.CANONICAL_INVALID
        except sqlite3.Error:
            return None, DegradedReason.CANONICAL_UNAVAILABLE
        except VerificationError as exc:
            if _has_io_cause(exc):
                return None, DegradedReason.CANONICAL_UNAVAILABLE
            return None, DegradedReason.CANONICAL_INVALID
        except (RuntimePathError, ValueError):
            return None, DegradedReason.CANONICAL_INVALID
        except OSError:
            return None, DegradedReason.CANONICAL_UNAVAILABLE

    def _validate_retrieval_paths(self) -> None:
        require_safe_directory(self.paths.repo_root)
        if path_is_redirect(self.paths.runtime_root):
            raise RuntimePathError("unsafe retrieval runtime root")
        if not self.paths.runtime_root.exists():
            return
        require_safe_directory(self.paths.runtime_root, self.paths.repo_root)
        self.paths._validate_existing_retrieval_state()

    def _capture_for_rebuild(self) -> CanonicalSnapshot:
        self.paths.prepare_for_write()
        try:
            return capture_canonical_snapshot(self.paths)
        except (CanonicalSnapshotUnavailableError, CanonicalSnapshotInvalidError):
            raise
        except VerificationError as exc:
            raise CacheCorruptError("canonical database is unavailable") from exc

    def _rebase_legacy_history_for_v2(
        self,
        history: tuple[list[sqlite3.Row], list[sqlite3.Row]],
        prior_active: str | None,
    ) -> tuple[tuple[list[tuple[object, ...]], list[tuple[object, ...]]], str | None]:
        generations, events = history
        generation_id_map: dict[str, str] = {}
        rebased_generations: list[tuple[object, ...]] = []
        for row in generations:
            canonical = IndexHighwater.from_dict(json.loads(row["canonical_highwater_json"]))
            new_generation_id = _cache_generation_id(
                canonical, row["canonical_content_fingerprint"], schema_version=SCHEMA_VERSION_V3
            )
            generation_id_map[row["generation_id"]] = new_generation_id
            rebased_generations.append(
                (
                    row["generation_seq"],
                    new_generation_id,
                    row["canonical_highwater_json"],
                    row["indexed_highwater_json"],
                    row["canonical_content_fingerprint"],
                    row["provider_fingerprint"],
                    row["model_fingerprint"],
                    SCHEMA_VERSION_V3,
                    row["created_at"],
                )
            )
        rebased_events: list[tuple[object, ...]] = []
        for row in events:
            new_generation_id = generation_id_map[row["generation_id"]]
            event_id, event_sha256 = _cache_event_values(
                new_generation_id, row["action"], row["payload_json"], row["created_at"]
            )
            rebased_events.append(
                (
                    row["event_seq"],
                    event_id,
                    new_generation_id,
                    row["action"],
                    row["payload_json"],
                    event_sha256,
                    row["created_at"],
                )
            )
        return (rebased_generations, rebased_events), (
            generation_id_map[prior_active] if prior_active is not None else None
        )

    def _valid_history_or_empty(self) -> tuple[tuple[list, list], str | None]:
        if not self.paths.database_path.exists():
            return ([], []), None
        try:
            connection = connect_cache_database(self.paths.database_path, readonly=True)
            try:
                history = read_history_for_rebuild(connection)
                active = connection.execute(
                    "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
                ).fetchone()[0]
                return history, active
            finally:
                connection.close()
        except CacheUnsupportedError:
            raise
        except (CacheCorruptError, OSError, ValueError, sqlite3.Error):

            return ([], []), None

    def _semantic_history_or_empty(self) -> dict[str, tuple]:
        if not self.paths.database_path.exists():
            return {}
        try:
            connection = connect_cache_database(self.paths.database_path, readonly=True)
            try:
                return read_semantic_history_for_rebuild(connection)
            finally:
                connection.close()
        except (CacheCorruptError, OSError, ValueError, sqlite3.Error):
            return {}

    def _v2_immutable_rows_or_empty(self) -> dict[str, tuple[sqlite3.Row, ...]]:
        try:
            connection = connect_cache_database(self.paths.database_path, readonly=True)
            try:
                return read_v2_immutable_rows(connection)
            finally:
                connection.close()
        except (CacheCorruptError, OSError, ValueError, sqlite3.Error):
            return {}

    def _owned_temporary_path(self) -> _OwnedTemporary:
        self.paths.prepare_for_write()
        temporary = self.paths.temp_dir / f"rebuild-{uuid.uuid4().hex}.sqlite3"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
        require_safe_file(temporary, self.paths.temp_dir, require_exists=True)
        return _OwnedTemporary(temporary, _safe_file_identity(temporary, self.paths.temp_dir))

    def _verify_temporary(self, temporary: Path) -> None:
        connection = connect_cache_database(temporary, readonly=True)
        try:
            verify_cache_database(connection)
        finally:
            connection.close()

    def _verified_install_handoff(
        self, temporary: _OwnedTemporary, *, preserve_previous: bool
    ) -> _InstallHandoff:
        """Bind final installation to the descriptor of the verified temporary file."""
        previous = self._preserve_previous_cache() if preserve_previous else None
        try:
            descriptor = os.open(temporary.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                if _descriptor_file_identity(descriptor) != temporary.identity:
                    raise CacheCorruptError("temporary path changed before replace")
                handoff = self.paths.temp_dir / f"install-{uuid.uuid4().hex}.sqlite3"
                try:
                    os.link(temporary.path, handoff)
                except OSError as exc:
                    raise CacheCorruptError("temporary installation handoff is unavailable") from exc
                identity = _safe_file_identity(handoff, self.paths.temp_dir, link_count=2)
                if identity != _descriptor_file_identity(descriptor, link_count=2):
                    self._remove_owned_temporary(_OwnedTemporary(handoff, identity, 2))
                    raise CacheCorruptError("temporary path changed before replace")
                return _InstallHandoff(_OwnedTemporary(handoff, identity, 2), previous)
            finally:
                os.close(descriptor)
        except BaseException:
            if previous is not None:
                self._remove_owned_temporary(previous.replace_source)
                self._remove_owned_temporary(previous.recovery_guard)
            raise

    def _preserve_previous_cache(self) -> _VerifiedBackup:
        """Create a durable verified copy plus a separate atomic-restore alias."""
        previous_path = self.paths.database_path
        descriptor = os.open(
            previous_path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        guard: _OwnedTemporary | None = None
        replace_source: _OwnedTemporary | None = None
        try:
            previous_identity = _descriptor_file_identity(descriptor)
            if not _same_file(
                previous_path,
                previous_path.parent,
                previous_identity,
            ):
                raise CacheCorruptError("previous retrieval cache changed before install")
            guard_path = self.paths.temp_dir / f"preserve-{uuid.uuid4().hex}.sqlite3"
            guard_descriptor = os.open(
                guard_path,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                guard_identity = _descriptor_file_identity(guard_descriptor)
                guard = _OwnedTemporary(guard_path, guard_identity)
                os.lseek(descriptor, 0, os.SEEK_SET)
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    pending = memoryview(chunk)
                    while pending:
                        written = os.write(guard_descriptor, pending)
                        if written <= 0:
                            raise OSError("short retrieval backup write")
                        pending = pending[written:]
                os.fsync(guard_descriptor)
            finally:
                os.close(guard_descriptor)
            if (
                not _same_file(guard.path, self.paths.temp_dir, guard.identity)
                or not _same_file(
                    previous_path,
                    previous_path.parent,
                    previous_identity,
                )
            ):
                raise CacheCorruptError("previous retrieval cache changed before install")
            self._verify_temporary(guard.path)
            if not _same_file(guard.path, self.paths.temp_dir, guard.identity):
                raise CacheCorruptError("previous retrieval cache backup changed")

            replace_path = self.paths.temp_dir / f"rollback-{uuid.uuid4().hex}.sqlite3"
            try:
                os.link(guard.path, replace_path)
            except OSError as exc:
                raise CacheCorruptError(
                    "previous retrieval cache cannot be preserved"
                ) from exc
            replace_source = _OwnedTemporary(replace_path, guard.identity, 2)
            linked_identity = _safe_file_identity(
                replace_path,
                self.paths.temp_dir,
                link_count=2,
            )
            guard = _OwnedTemporary(guard.path, guard.identity, 2)
            if (
                linked_identity != guard.identity
                or not _same_file(
                    guard.path,
                    self.paths.temp_dir,
                    guard.identity,
                    link_count=2,
                )
            ):
                raise CacheCorruptError("previous retrieval cache backup changed")
            return _VerifiedBackup(replace_source, guard)
        except BaseException:
            if replace_source is not None:
                self._remove_owned_temporary(replace_source)
            if guard is not None:
                self._remove_owned_temporary(guard)
            raise
        finally:
            os.close(descriptor)

    def _install_and_verify(self, installation: _InstallHandoff) -> RetrievalResult:
        """Install only the verified inode, restoring the prior valid cache on failure."""
        replaced = False
        installed_file: _OwnedTemporary | None = None
        try:
            os.replace(installation.source.path, self.paths.database_path)
            replaced = True
            installed_file = _observed_owned_file(
                self.paths.database_path, self.paths.database_path.parent
            )
            if not _same_file(
                self.paths.database_path,
                self.paths.database_path.parent,
                installation.source.identity,
                link_count=2,
            ):
                raise CacheCorruptError("install source changed during replacement")
            installed_result = self.status()
            if installed_result.status is not RetrievalStatus.CURRENT:
                raise CacheCorruptError("installed retrieval cache did not verify")
            return installed_result
        except BaseException:
            if replaced and installation.previous is not None:
                try:
                    restored = self._restore_previous_cache(
                        installation.previous, installed_file
                    )
                except (CacheCorruptError, OSError):
                    restored = False
                    installation.retain_previous = True
                if not restored:
                    installation.retain_previous = True
                    if (
                        installed_file is not None
                        and installed_file.identity != installation.source.identity
                    ):
                        self._remove_owned_temporary(installed_file)
            elif replaced and installed_file is not None:
                self._remove_owned_temporary(installed_file)
            raise

    def _restore_previous_cache(
        self, previous: _VerifiedBackup, installed_file: _OwnedTemporary | None
    ) -> bool:
        """Atomically return the prior verified cache to its canonical cache pathname."""
        if installed_file is None or not _same_file(
            self.paths.database_path,
            self.paths.database_path.parent,
            installed_file.identity,
            link_count=installed_file.link_count,
        ):
            return False
        replace_source = previous.replace_source
        recovery_guard = previous.recovery_guard
        if not _same_file(
            replace_source.path,
            replace_source.path.parent,
            replace_source.identity,
            link_count=2,
        ) or not _same_file(
            recovery_guard.path,
            recovery_guard.path.parent,
            recovery_guard.identity,
            link_count=2,
        ):
            raise CacheCorruptError("previous retrieval cache cannot be restored")
        os.replace(replace_source.path, self.paths.database_path)
        if _same_owned_file(
            self.paths.database_path,
            self.paths.database_path.parent,
            recovery_guard.identity,
        ):
            return True
        return self._restore_previous_from_guard(recovery_guard)

    def _restore_previous_from_guard(self, guard: _OwnedTemporary) -> bool:
        """Recover after the replace-source pathname itself was exchanged."""
        try:
            observed = _observed_owned_file(
                self.paths.database_path,
                self.paths.database_path.parent,
            )
        except CacheCorruptError:
            return False
        if observed.identity == guard.identity:
            return True
        if not self._remove_owned_temporary(observed) or not _path_is_absent(
            self.paths.database_path
        ):
            return False
        try:
            descriptor = os.open(
                guard.path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                if (
                    not _descriptor_matches_identity(descriptor, guard.identity)
                    or not _same_owned_file(
                        guard.path,
                        guard.path.parent,
                        guard.identity,
                    )
                ):
                    raise CacheCorruptError(
                        "previous retrieval cache guard changed"
                    )
                os.link(guard.path, self.paths.database_path)
                restored = (
                    _descriptor_matches_identity(descriptor, guard.identity)
                    and _same_owned_file(
                        guard.path,
                        guard.path.parent,
                        guard.identity,
                    )
                    and _same_owned_file(
                        self.paths.database_path,
                        self.paths.database_path.parent,
                        guard.identity,
                    )
                )
            finally:
                os.close(descriptor)
        except (CacheCorruptError, OSError):
            return False
        if not restored:
            try:
                candidate = _observed_owned_file(
                    self.paths.database_path,
                    self.paths.database_path.parent,
                )
            except CacheCorruptError:
                return False
            if candidate.identity != guard.identity:
                self._remove_owned_temporary(candidate)
            return False
        return True

    @staticmethod
    def _inject(injector: Callable[[str], None] | None, step: str) -> None:
        if injector is not None:
            injector(step)

    @staticmethod
    def _remove_owned_temporary(temporary: _OwnedTemporary) -> bool:
        link_counts = tuple(dict.fromkeys((temporary.link_count, 1, 2, 3)))
        for link_count in link_counts:
            if _same_file(temporary.path, temporary.path.parent, temporary.identity, link_count=link_count):
                try:
                    temporary.path.unlink()
                except OSError:
                    return False
                return True
        return False


def _lock_for(path: Path) -> threading.Lock:
    key = path.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


@contextmanager
def process_rebuild_lock(
    temp_dir: Path, *, timeout_ms: int | None = None,
    deadline: DeadlineBudget | None = None,
    now_ns: Callable[[], int] | None = None,
):
    """Hold a fixed, validated advisory lock leaf for the rebuild critical section."""
    if (
        timeout_ms is not None
        and (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms < 0
        )
    ):
        raise RetrievalValidationError("rebuild lock timeout must be non-negative")
    if deadline is not None and not isinstance(deadline, DeadlineBudget):
        raise RetrievalValidationError("rebuild lock deadline must be DeadlineBudget or null")
    if now_ns is not None and not callable(now_ns):
        raise RetrievalValidationError("now_ns must be callable")
    wait_seconds = min(5.0, 5.0 if timeout_ms is None else timeout_ms / 1000)
    wait_started = None
    if deadline is not None:
        # Anchor the physical wait before preparation, using only what remains
        # of the caller's absolute budget at context entry. A frozen injected
        # budget clock must not remove the existing physical lock-wait cap.
        wait_started = time.monotonic()
        wait_seconds = min(wait_seconds, deadline.remaining_ms(
            now_ns=now_ns if now_ns is not None else time.monotonic_ns,
        ) / 1000)
    require_safe_directory(temp_dir)
    lock_path = temp_dir / "rebuild.lock"
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    locked = False
    lock_deadline = (time.monotonic() if wait_started is None else wait_started) + wait_seconds
    try:
        require_safe_file(lock_path, temp_dir, require_exists=False)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            os.write(descriptor, _LOCK_MAGIC)
            identity = _safe_file_identity(lock_path, temp_dir)
        except FileExistsError:
            while True:
                remaining = lock_deadline - time.monotonic()
                if remaining <= 0:
                    raise CacheCorruptError("retrieval rebuild lock is unavailable")
                try:
                    identity = _safe_file_identity(lock_path, temp_dir)
                    descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                    break
                except PermissionError:
                    remaining = lock_deadline - time.monotonic()
                    if remaining <= 0:
                        raise CacheCorruptError("retrieval rebuild lock is unavailable")
                    time.sleep(min(0.05, remaining))
        if identity is None or _descriptor_file_identity(descriptor) != identity:
            raise CacheCorruptError("retrieval rebuild lock path is unsafe")
        os.lseek(descriptor, 0, os.SEEK_SET)
        _lock_descriptor(descriptor, deadline=lock_deadline)
        locked = True
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(_LOCK_MAGIC) + 1) != _LOCK_MAGIC:
            raise CacheCorruptError("retrieval rebuild lock path is unsafe")
        if not _same_file(lock_path, temp_dir, identity):
            raise CacheCorruptError("retrieval rebuild lock path is unsafe")
        yield
    finally:
        if descriptor is not None:
            if locked:
                _unlock_descriptor(descriptor)
            os.close(descriptor)


def _safe_directory_identity(path: Path) -> tuple[int, int]:
    if path_is_redirect(path):
        raise CacheCorruptError("retrieval rebuild lock path is unsafe")
    try:
        details = path.lstat()
    except OSError as exc:
        raise CacheCorruptError("retrieval rebuild lock is unavailable") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise CacheCorruptError("retrieval rebuild lock path is unsafe")
    return details.st_dev, details.st_ino


def _safe_file_identity(path: Path, parent: Path, *, link_count: int = 1) -> tuple[int, int]:
    if path.parent != parent or path_is_redirect(path):
        raise CacheCorruptError("retrieval path is unsafe")
    try:
        details = path.lstat()
    except OSError as exc:
        raise CacheCorruptError("retrieval path is unavailable") from exc
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != link_count:
        raise CacheCorruptError("retrieval path is unsafe")
    return details.st_dev, details.st_ino


def _observed_owned_file(path: Path, parent: Path) -> _OwnedTemporary:
    """Capture the exact regular file just installed before any removal decision."""
    if path.parent != parent or path_is_redirect(path):
        raise CacheCorruptError("retrieval path is unsafe")
    try:
        details = path.lstat()
    except OSError as exc:
        raise CacheCorruptError("retrieval path is unavailable") from exc
    if not stat.S_ISREG(details.st_mode) or details.st_nlink < 1:
        raise CacheCorruptError("retrieval path is unsafe")
    return _OwnedTemporary(path, (details.st_dev, details.st_ino), details.st_nlink)


def _descriptor_file_identity(descriptor: int, *, link_count: int = 1) -> tuple[int, int]:
    try:
        details = os.fstat(descriptor)
    except OSError as exc:
        raise CacheCorruptError("retrieval path is unavailable") from exc
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != link_count:
        raise CacheCorruptError("retrieval path is unsafe")
    return details.st_dev, details.st_ino


def _same_file(path: Path, parent: Path, identity: tuple[int, int], *, link_count: int = 1) -> bool:
    try:
        return _safe_file_identity(path, parent, link_count=link_count) == identity
    except CacheCorruptError:
        return False


def _same_owned_file(path: Path, parent: Path, identity: tuple[int, int]) -> bool:
    """Match a regular file identity without assuming its current hard-link count."""
    try:
        if path.parent != parent or path_is_redirect(path):
            return False
        details = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(details.st_mode) and (details.st_dev, details.st_ino) == identity


def _descriptor_matches_identity(descriptor: int, identity: tuple[int, int]) -> bool:
    try:
        details = os.fstat(descriptor)
    except OSError:
        return False
    return stat.S_ISREG(details.st_mode) and (details.st_dev, details.st_ino) == identity


def _path_is_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _has_io_cause(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (sqlite3.Error, OSError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _lock_descriptor(descriptor: int, *, deadline: float | None = None) -> None:
    lock_deadline = time.monotonic() + 5 if deadline is None else deadline
    if os.name == "nt":
        import msvcrt

        while True:
            remaining = lock_deadline - time.monotonic()
            if remaining <= 0:
                raise CacheCorruptError("retrieval rebuild lock is unavailable")
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                remaining = lock_deadline - time.monotonic()
                if remaining <= 0:
                    raise CacheCorruptError("retrieval rebuild lock is unavailable")
                time.sleep(min(0.05, remaining))
    else:
        import fcntl

        while True:
            remaining = lock_deadline - time.monotonic()
            if remaining <= 0:
                raise CacheCorruptError("retrieval rebuild lock is unavailable")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                remaining = lock_deadline - time.monotonic()
                if remaining <= 0:
                    raise CacheCorruptError("retrieval rebuild lock is unavailable")
                time.sleep(min(0.05, remaining))


def _unlock_descriptor(descriptor: int) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        pass


def _safe_marker_identity(marker: Path, parent: Path) -> tuple[int, int]:
    if marker.parent != parent or path_is_redirect(marker):
        raise CacheCorruptError("retrieval rebuild lock path is unsafe")
    try:
        details = marker.lstat()
    except OSError as exc:
        raise CacheCorruptError("retrieval rebuild lock is unavailable") from exc
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise CacheCorruptError("retrieval rebuild lock path is unsafe")
    return details.st_dev, details.st_ino


def _read_lock_owner(marker: Path, parent: Path) -> tuple[str, tuple[int, int]]:
    before = _safe_marker_identity(marker, parent)
    try:
        descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            owner = os.read(descriptor, 33).decode("ascii")
        finally:
            os.close(descriptor)
    except (OSError, UnicodeError) as exc:
        raise CacheCorruptError("retrieval rebuild lock is unavailable") from exc
    after = _safe_marker_identity(marker, parent)
    if before != after:
        raise CacheCorruptError("retrieval rebuild lock path is unsafe")
    if len(owner) != 32 or any(character not in "0123456789abcdef" for character in owner):
        raise CacheCorruptError("retrieval rebuild lock path is unsafe")
    return owner, after


def _same_directory(path: Path, identity: tuple[int, int]) -> bool:
    try:
        return _safe_directory_identity(path) == identity
    except CacheCorruptError:
        return False


def _same_marker(marker: Path, parent: Path, identity: tuple[int, int]) -> bool:
    try:
        return _safe_marker_identity(marker, parent) == identity
    except CacheCorruptError:
        return False


def _write_projection(
    connection: sqlite3.Connection,
    *,
    generation_id: str,
    batch: ProjectionBatch,
    created_at: str,
) -> None:
    _write_projection_rows(connection, generation_id=generation_id, batch=batch)
    _write_projection_capability(
        connection, generation_id=generation_id, batch=batch, created_at=created_at
    )


def _write_projection_rows(
    connection: sqlite3.Connection,
    *,
    generation_id: str,
    batch: ProjectionBatch,
) -> None:
    for document in batch.documents:
        _append_projected_document(connection, generation_id, document)
    for source in batch.sources:
        _append_projected_source(connection, generation_id, source)
    for symbol in batch.symbols:
        _append_projected_symbol(connection, generation_id, symbol)
    # FTS is rebuilt once from the final generation document set, after any
    # unchanged rows have been copied and changed rows have been reprojected.


def _append_projected_document(
    connection: sqlite3.Connection, generation_id: str, document: object
) -> None:
    index_store.append_document(
        connection,
        document_id=document.document_id,
        generation_id=generation_id,
        record_id=document.record_id,
        record_seq=document.record_seq,
        record_type=document.record_type,
        scope_kind=document.scope_kind,
        scope_key=document.scope_key,
        topic_key=document.topic_key,
        statement=document.statement,
        value_json=document.value_json,
        search_text=document.search_text,
        state=document.state,
        created_at=document.created_at,
        last_memory_event_seq=document.last_memory_event_seq,
        evidence_count=document.evidence_count,
        content_sha256=document.content_sha256,
        document_sha256=document.document_sha256,
    )


def _append_projected_source(
    connection: sqlite3.Connection, generation_id: str, source: object
) -> None:
    index_store.append_document_source(
        connection,
        source_id=source.source_id,
        generation_id=generation_id,
        document_id=source.document_id,
        ordinal=source.ordinal,
        safe_reference=source.safe_reference,
        source_sha256=source.source_sha256,
    )


def _append_projected_symbol(
    connection: sqlite3.Connection, generation_id: str, symbol: object
) -> None:
    index_store.append_symbol(
        connection,
        symbol_id=symbol.symbol_id,
        generation_id=generation_id,
        document_id=symbol.document_id,
        ordinal=symbol.ordinal,
        symbol_kind=symbol.symbol_kind,
        normalized_symbol=symbol.normalized_symbol,
        display_symbol=symbol.display_symbol,
        symbol_sha256=symbol.symbol_sha256,
    )


def _write_projection_capability(
    connection: sqlite3.Connection,
    *,
    generation_id: str,
    batch: ProjectionBatch | ProjectionDigest,
    created_at: str,
) -> None:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
    ).fetchone() is not None:
        connection.execute(
            "INSERT INTO retrieval_documents_fts(document_id, generation_id, search_text) "
            "SELECT document_id, generation_id, search_text FROM retrieval_documents "
            "WHERE generation_id = ? ORDER BY document_seq",
            (generation_id,),
        )
        fts5_state = "available"
    else:
        fts5_state = "unavailable"
    connection.execute(
        """INSERT INTO retrieval_generation_capabilities(
            generation_id, fts5_state, sqlite_version, index_policy_fingerprint,
            projection_manifest_sha256, projected_document_count,
            omitted_unsafe_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            generation_id,
            fts5_state,
            sqlite3.sqlite_version,
            batch.index_policy_fingerprint,
            batch.manifest_sha256,
            len(batch.documents) if isinstance(batch, ProjectionBatch) else batch.document_count,
            batch.omitted_unsafe_count,
            created_at,
        ),
    )


def _validate_final_projection(
    connection: sqlite3.Connection,
    *,
    generation_id: str,
    expected: ProjectionDigest,
    snapshot: CanonicalSnapshot,
) -> None:
    """Fail closed using ordered fixed-batch digests, never global row sets."""
    capability = connection.execute(
        """SELECT fts5_state, projected_document_count, omitted_unsafe_count,
        projection_manifest_sha256, index_policy_fingerprint
        FROM retrieval_generation_capabilities WHERE generation_id = ?""",
        (generation_id,),
    ).fetchone()
    if capability is None or tuple(capability)[1:] != (
        expected.document_count,
        expected.omitted_unsafe_count,
        expected.manifest_sha256,
        expected.index_policy_fingerprint,
    ):
        raise CacheCorruptError("final retrieval projection capability does not match canonical snapshot")
    actual_builder = ProjectionDigestBuilder(
        snapshot, omitted_unsafe_count=capability["omitted_unsafe_count"]
    )
    try:
        documents = connection.execute(
            """SELECT document_id, record_id, record_seq, record_type, scope_kind,
            scope_key, topic_key, statement, value_json, search_text, state,
            created_at, last_memory_event_seq, evidence_count, content_sha256,
            document_sha256 FROM retrieval_documents WHERE generation_id = ?
            ORDER BY record_seq""",
            (generation_id,),
        )
        for rows in _batches(documents):
            for row in rows:
                actual_builder.add_document_row(
                    row["document_sha256"], tuple(row), include_fts=False
                )
        sources = connection.execute(
            """SELECT s.source_id, s.document_id, s.ordinal, s.safe_reference,
            s.source_sha256 FROM retrieval_document_sources AS s
            JOIN retrieval_documents AS d ON d.document_id = s.document_id
            WHERE s.generation_id = ? ORDER BY d.record_seq, s.ordinal""",
            (generation_id,),
        )
        for rows in _batches(sources):
            for row in rows:
                actual_builder.add_source_row(row["source_sha256"], tuple(row))
        symbols = connection.execute(
            """SELECT s.symbol_id, s.document_id, s.ordinal, s.symbol_kind,
            s.normalized_symbol, s.display_symbol, s.symbol_sha256
            FROM retrieval_symbols AS s
            JOIN retrieval_documents AS d ON d.document_id = s.document_id
            WHERE s.generation_id = ? ORDER BY d.record_seq, s.ordinal""",
            (generation_id,),
        )
        for rows in _batches(symbols):
            for row in rows:
                actual_builder.add_symbol_row(row["symbol_sha256"], tuple(row))
        fts = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
        ).fetchone()
        if fts is not None:
            fts_rows = connection.execute(
                """SELECT f.document_id, f.search_text
                FROM retrieval_documents_fts AS f
                JOIN retrieval_documents AS d
                  ON d.document_id = f.document_id AND d.generation_id = f.generation_id
                WHERE f.generation_id = ? ORDER BY d.record_seq""",
                (generation_id,),
            )
            for rows in _batches(fts_rows):
                for row in rows:
                    actual_builder.add_fts_row(tuple(row))
        actual = actual_builder.finish()
    except BaseException:
        actual_builder.close()
        raise
    if (
        actual.document_count,
        actual.source_count,
        actual.symbol_count,
        actual.manifest_sha256,
        actual.document_rows_sha256,
        actual.source_rows_sha256,
        actual.symbol_rows_sha256,
    ) != (
        expected.document_count,
        expected.source_count,
        expected.symbol_count,
        expected.manifest_sha256,
        expected.document_rows_sha256,
        expected.source_rows_sha256,
        expected.symbol_rows_sha256,
    ):
        raise CacheCorruptError("final retrieval projection does not match canonical snapshot")
    if fts is not None and (
        actual.fts_count,
        actual.fts_rows_sha256,
    ) != (expected.fts_count, expected.fts_rows_sha256):
        raise CacheCorruptError("final retrieval FTS projection does not match canonical snapshot")


def _unaffected_where(affected_record_ids: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if not affected_record_ids:
        return "", ()
    return (
        " AND record_id NOT IN (" + ", ".join("?" for _ in affected_record_ids) + ")",
        affected_record_ids,
    )


def _batches(cursor: sqlite3.Cursor):
    """Yield bounded ordered SQLite cursor batches for catch-up copying."""
    while rows := cursor.fetchmany(CATCH_UP_COPY_BATCH_SIZE):
        yield rows


def _active_generation(connection) -> IndexGeneration | None:
    active = connection.execute(
        "SELECT active_generation_id FROM retrieval_state WHERE singleton_id = 1"
    ).fetchone()[0]
    if active is None:
        return None
    row = connection.execute(
        """SELECT g.*, c.state, c.validation_state, c.degraded_reason, c.updated_at
        FROM retrieval_generations AS g
        JOIN retrieval_generation_current AS c ON c.generation_id = g.generation_id
        WHERE g.generation_id = ?""",
        (active,),
    ).fetchone()
    if row is None or row["state"] != "active":
        raise CacheCorruptError("active generation projection is invalid")
    try:
        degraded = None if row["degraded_reason"] is None else DegradedReason(row["degraded_reason"])
        return IndexGeneration(
            generation_id=row["generation_id"],
            state=row["state"],
            canonical_highwater=IndexHighwater.from_dict(json.loads(row["canonical_highwater_json"])),
            indexed_highwater=IndexHighwater.from_dict(json.loads(row["indexed_highwater_json"])),
            canonical_content_fingerprint=row["canonical_content_fingerprint"],
            provider_fingerprint=row["provider_fingerprint"],
            model_fingerprint=row["model_fingerprint"],
            validation_state=row["validation_state"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            degraded_reason=degraded,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise CacheCorruptError("active generation is invalid") from exc


def _empty_result(
    status: RetrievalStatus,
    reason: DegradedReason,
    snapshot: CanonicalSnapshot | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        status=status,
        mode=RetrievalMode.OFF,
        generation=None,
        canonical_highwater=None if snapshot is None else snapshot.highwater,
        indexed_highwater=None,
        degraded_reason=reason,
    )


def _query_result(status, snapshot, generation, index_stale, reasons, candidates, trace) -> RetrievalQueryResult:
    return RetrievalQueryResult(
        status=status, mode=RetrievalMode.OFF, generation=generation,
        canonical_highwater=None if snapshot is None else snapshot.highwater,
        indexed_highwater=None if generation is None else generation.indexed_highwater,
        index_stale=index_stale, degraded_reasons=tuple(dict.fromkeys(reasons)),
        candidates=tuple(candidates), trace=tuple(trace),
    )


def _direct_prefix(canonical, paths, request, policy):
    """Read authority-selected prefix records directly from canonical truth."""
    blobs = BlobStore(paths.runtime_root / "blobs")
    mandatory_ids = mandatory_candidates(canonical, blobs, request, limit=policy.per_channel_limit).ids
    temporary_ids = temporary_candidates(canonical, blobs, request, limit=policy.per_channel_limit).ids
    result = []
    trace = []
    for identifier in mandatory_ids:
        row = canonical.execute("""SELECT r.*, c.state FROM memory_records AS r
            JOIN memory_current AS c ON c.record_id = r.record_id WHERE r.record_id = ?""", (identifier,)).fetchone()
        if row is None:
            continue
        candidate = RetrievedCandidate("record", identifier, row["record_type"], row["scope_kind"], row["scope_key"], row["state"], row["record_seq"], row["content_sha256"], True, 0, 1, ())
        result.append(candidate)
        trace.append(CandidateTrace(identifier, (CandidateChannel.MANDATORY_CONSTITUTION,), (), None, True, ("mandatory_prefix",)))
    for identifier in temporary_ids:
        row = canonical.execute("""SELECT c.*, p.state FROM memory_candidates AS c
            JOIN memory_candidate_current AS p ON p.candidate_id = c.candidate_id WHERE c.candidate_id = ?""", (identifier,)).fetchone()
        if row is None:
            continue
        candidate = RetrievedCandidate("temporary", identifier, row["record_type"], row["scope_kind"], row["scope_key"], row["state"], row["candidate_seq"], row["content_sha256"], False, 0, 1, (), "UNVERIFIED TEMPORARY")
        result.append(candidate)
        trace.append(CandidateTrace(identifier, (CandidateChannel.TEMPORARY,), (), None, True, ("temporary_prefix", "unverified_temporary")))
    return tuple(result), tuple(trace)


def _cache_inputs(cache, request, generation_id, policy, excluded):
    rows = cache.execute("SELECT * FROM retrieval_documents WHERE generation_id = ?", (generation_id,)).fetchall()
    by_id = {row["record_id"]: _cache_row(row) for row in rows if row["record_id"] not in excluded}
    channels = (
        (CandidateChannel.EXACT, exact_candidates(cache, request, generation_id=generation_id, limit=policy.per_channel_limit, excluded_ids=frozenset(excluded))),
        (CandidateChannel.SYMBOL, symbol_candidates(cache, request, generation_id=generation_id, limit=policy.per_channel_limit, excluded_ids=frozenset(excluded))),
        (CandidateChannel.BM25, bm25_candidates(cache, request, generation_id=generation_id, limit=policy.per_channel_limit, excluded_ids=frozenset(excluded))),
        (CandidateChannel.RECENT, recent_candidates(cache, request, generation_id=generation_id, limit=policy.per_channel_limit, excluded_ids=frozenset(excluded))),
    )
    inputs = []
    reasons = []
    for channel, result in channels:
        if "fts5_unavailable" in result.reason_codes:
            reasons.append(DegradedReason.FTS5_UNAVAILABLE)
        inputs.append(FusionInput(channel, tuple(by_id[identifier] for identifier in result.ids if identifier in by_id)))
    return inputs, reasons


def _eligible_rows(cache, request, generation_id, excluded):
    """Return only active, scope-visible deterministic rows before semantic scoring."""
    scope = _scope(request)
    return {
        row["record_id"]: _cache_row(row)
        for row in cache.execute(
            "SELECT * FROM retrieval_documents WHERE generation_id = ?", (generation_id,)
        )
        if row["record_id"] not in excluded and _visible(row, scope)
    }


def _active_semantic_fusion_inputs(
    paths: RetrievalPaths,
    cache: sqlite3.Connection,
    deterministic_generation_id: str,
    eligible: Mapping[str, Mapping[str, object]],
    query: SemanticQueryInputs,
    policy: RetrievalPolicy,
) -> tuple[list[FusionInput], list[DegradedReason]]:
    """Open one verified immutable semantic generation, or safely degrade it.

    A semantic generation may lag the deterministic generation.  Eligibility is
    intentionally built from the current deterministic rows above, so changed,
    removed, closed, or cross-scope records cannot become visible through the
    older derived file.
    """
    state = cache.execute(
        "SELECT active_semantic_generation_id, validation_state FROM retrieval_semantic_state WHERE singleton_id = 1"
    ).fetchone()
    semantic_id = None if state is None else state["active_semantic_generation_id"]
    if not isinstance(semantic_id, str) or state["validation_state"] != "validated":
        return [], [DegradedReason.SEMANTIC_INDEX_UNAVAILABLE]
    try:
        semantic_paths = SemanticGenerationPaths.for_generation(paths.generations_dir, semantic_id)
        with open_verified_semantic_snapshot(semantic_paths) as (manifest, semantic):
            components = _bound_semantic_components(
                cache, semantic, semantic_id, manifest, deterministic_generation_id
            )
            dense_fingerprint = _ready_semantic_component(
                components.get("dense"), manifest.dense_model_fingerprint,
                manifest.preprocessing_fingerprint,
            )
            sparse_fingerprint = _ready_semantic_component(
                components.get("sparse"), manifest.sparse_model_fingerprint,
                manifest.preprocessing_fingerprint,
            )
            inputs, raw_reasons = _semantic_fusion_inputs(
                semantic,
                eligible,
                semantic_generation_id=semantic_id,
                dense_query_vector=query.dense_vector if dense_fingerprint is not None else None,
                dense_model_fingerprint=dense_fingerprint,
                sparse_query_weights=query.sparse_weights if sparse_fingerprint is not None else None,
                sparse_model_fingerprint=sparse_fingerprint,
                preprocessing_fingerprint=manifest.preprocessing_fingerprint,
                policy=policy,
            )
    except (SemanticStoreError, sqlite3.Error, OSError, ValueError):
        return [], [DegradedReason.SEMANTIC_INDEX_UNAVAILABLE]
    reasons: list[DegradedReason] = []
    if query.dense_vector is not None and dense_fingerprint is None and _component_is_not_disabled(components.get("dense")):
        reasons.append(DegradedReason.SEMANTIC_DENSE_UNAVAILABLE)
    if query.sparse_weights is not None and sparse_fingerprint is None and _component_is_not_disabled(components.get("sparse")):
        reasons.append(DegradedReason.SEMANTIC_SPARSE_UNAVAILABLE)
    if any(reason.startswith("dense_") for reason in raw_reasons):
        reasons.append(DegradedReason.SEMANTIC_DENSE_UNAVAILABLE)
    if any(reason.startswith("sparse_") for reason in raw_reasons):
        reasons.append(DegradedReason.SEMANTIC_SPARSE_UNAVAILABLE)
    if manifest.projection_generation_id != deterministic_generation_id:
        reasons.append(DegradedReason.SEMANTIC_INDEX_STALE)
    return inputs, list(dict.fromkeys(reasons))


def _bound_semantic_components(cache, semantic, semantic_id, manifest, deterministic_generation_id):
    """Bind the verified immutable directory to its active append-only cache row."""
    row = cache.execute(
        """SELECT g.*, c.state, c.validation_state AS current_validation_state
             FROM retrieval_semantic_generations AS g
             JOIN retrieval_semantic_generation_current AS c
               ON c.semantic_generation_id = g.semantic_generation_id
            WHERE g.semantic_generation_id = ?""",
        (semantic_id,),
    ).fetchone()
    if row is None:
        raise SemanticStoreError("semantic metadata is incomplete")
    projection = cache.execute(
        """SELECT canonical_highwater_json, indexed_highwater_json
             FROM retrieval_generations WHERE generation_id = ?""",
        (row["projection_generation_id"],),
    ).fetchone()
    current = cache.execute(
        "SELECT 1 FROM retrieval_generations WHERE generation_id = ?",
        (deterministic_generation_id,),
    ).fetchone()
    if projection is None or current is None:
        raise SemanticStoreError("semantic metadata is incomplete")
    if row["state"] != "active" or row["current_validation_state"] != "validated":
        raise SemanticStoreError("semantic metadata is not active")
    expected_path = f"generations/{semantic_id}/generation-manifest.json"
    expected = {
        "semantic_generation_id": row["semantic_generation_id"],
        "projection_generation_id": row["projection_generation_id"],
        "provider_fingerprint": row["provider_fingerprint"],
        "dense_model_fingerprint": row["dense_model_fingerprint"],
        "sparse_model_fingerprint": row["sparse_model_fingerprint"],
        "reranker_model_fingerprint": row["reranker_model_fingerprint"],
        "preprocessing_fingerprint": row["preprocessing_fingerprint"],
        "build_config_sha256": row["build_config_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "semantic_database_sha256": row["semantic_database_sha256"],
        "created_at": row["created_at"],
    }
    if (
        row["manifest_relative_path"] != expected_path
        or any(getattr(manifest, field) != value for field, value in expected.items())
        or manifest.canonical_highwater != json.loads(projection["canonical_highwater_json"])
        or manifest.indexed_highwater != json.loads(projection["indexed_highwater_json"])
    ):
        raise SemanticStoreError("semantic manifest does not match active metadata")
    components = {
        item["component"]: item
        for item in cache.execute(
            "SELECT * FROM retrieval_semantic_components WHERE semantic_generation_id = ?",
            (semantic_id,),
        )
    }
    if set(components) != {"dense", "sparse", "reranker"}:
        raise SemanticStoreError("semantic components are incomplete")
    indexed_highwater = projection["indexed_highwater_json"]
    absent_highwater = IndexHighwater().to_dict()
    actual_documents, actual_indexed = _semantic_component_actual_counts(semantic, manifest)
    manifest_fingerprints = {
        "dense": manifest.dense_model_fingerprint,
        "sparse": manifest.sparse_model_fingerprint,
        "reranker": manifest.reranker_model_fingerprint,
    }
    for name, component in components.items():
        try:
            coverage = json.loads(component["coverage_highwater_json"])
        except (TypeError, ValueError) as exc:
            raise SemanticStoreError("semantic component coverage is invalid") from exc
        expected_coverage = json.loads(indexed_highwater) if component["status"] == "ready" else absent_highwater
        expected_eligible = 0 if component["status"] == "disabled" else actual_documents
        expected_indexed = actual_indexed[name]
        if (
            coverage != expected_coverage
            or component["preprocessing_fingerprint"] != manifest.preprocessing_fingerprint
            or component["model_fingerprint"] != manifest_fingerprints[name]
            or component["eligible_document_count"] != expected_eligible
            or component["indexed_document_count"] != expected_indexed
        ):
            raise SemanticStoreError("semantic component coverage is stale")
    return components


def _semantic_component_actual_counts(semantic, manifest):
    """Read exact derived counts after the immutable store has been verified."""
    documents = semantic.execute("SELECT COUNT(*) FROM semantic_documents").fetchone()[0]
    dense_rows, dense_documents = semantic.execute(
        "SELECT COUNT(*), COUNT(DISTINCT document_id) FROM semantic_dense_vectors"
    ).fetchone()
    sparse_rows, sparse_documents = semantic.execute(
        "SELECT COUNT(*), COUNT(DISTINCT document_id) FROM semantic_sparse_postings"
    ).fetchone()
    if (
        documents != manifest.component_counts["documents"]
        or dense_rows != manifest.component_counts["dense_vectors"]
        or sparse_rows != manifest.component_counts["sparse_postings"]
    ):
        raise SemanticStoreError("semantic manifest counts do not match the derived store")
    return documents, {"dense": dense_documents, "sparse": sparse_documents, "reranker": 0}


def _ready_semantic_component(component, manifest_fingerprint, preprocessing_fingerprint):
    if component is None or component["status"] not in {"ready", "partial"}:
        return None
    fingerprint = component["model_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or fingerprint != manifest_fingerprint
        or component["preprocessing_fingerprint"] != preprocessing_fingerprint
    ):
        return None
    return fingerprint


def _component_is_not_disabled(component) -> bool:
    return component is None or component["status"] != "disabled"


def _semantic_fusion_inputs(
    semantic: sqlite3.Connection,
    eligible: Mapping[str, Mapping[str, object]],
    *,
    semantic_generation_id: str,
    dense_query_vector: tuple[float, ...] | None,
    dense_model_fingerprint: str | None,
    sparse_query_weights: tuple[tuple[str, float], ...] | None,
    sparse_model_fingerprint: str | None,
    preprocessing_fingerprint: str,
    policy: RetrievalPolicy,
) -> tuple[list[FusionInput], tuple[str, ...]]:
    """Build optional semantic RRF inputs without changing deterministic visibility.

    The caller obtains provider output only after deterministic eligibility has
    been projected.  This helper neither opens canonical state nor invokes a
    provider; Task 5 supplies deadline/health-controlled provider orchestration.
    """
    inputs: list[FusionInput] = []
    reasons: list[str] = []
    for channel, query, fingerprint, limit, function in (
        (CandidateChannel.DENSE, dense_query_vector, dense_model_fingerprint, policy.dense_top_k, dense_candidates),
        (CandidateChannel.SPARSE, sparse_query_weights, sparse_model_fingerprint, policy.sparse_top_k, sparse_candidates),
    ):
        if query is None or fingerprint is None:
            continue
        result = function(
            semantic, query, eligible,
            generation_id=semantic_generation_id,
            model_fingerprint=fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            limit=limit,
        )
        if result.ids:
            inputs.append(
                FusionInput(channel, tuple(eligible[item] for item in result.ids if item in eligible), dict(result.raw_score_hex))
            )
        if result.reason_codes and result.reason_codes != (channel.value,):
            reasons.extend(result.reason_codes)
    return inputs, tuple(dict.fromkeys(reasons))


def _cache_row(row):
    return {"candidate_kind": "record", "candidate_id": row["record_id"], "record_type": row["record_type"],
            "scope_kind": row["scope_kind"], "scope_key": row["scope_key"], "state": row["state"],
            "sequence": row["record_seq"], "last_lifecycle_event_sequence": row["last_memory_event_seq"],
            "content_sha256": row["content_sha256"], "document_id": row["document_id"]}


def _delta_rows(canonical, _snapshot, request, identifiers):
    """Project only bounded, active affected rows and rank as the exact channel does."""
    from .exact import _terms
    terms = _terms(request.query)
    scope = ScopeContext(request.scope.project_scope_key, request.scope.workspace_scope_keys)
    values = []
    for identifier in identifiers:
        row = canonical.execute("""SELECT r.*, c.state, c.last_event_seq
            FROM memory_records AS r JOIN memory_current AS c ON c.record_id = r.record_id
            WHERE r.record_id = ? AND c.state = 'active'""", (identifier,)).fetchone()
        if row is None:
            continue
        if not scope.contains(scope_kind=row["scope_kind"], scope_key=row["scope_key"]):
            continue
        endpoints = tuple(endpoint[0] for endpoint in canonical.execute(
            "SELECT CASE WHEN source_record_id = ? THEN target_record_id ELSE source_record_id END FROM memory_relations WHERE source_record_id = ? OR target_record_id = ? ORDER BY relation_seq",
            (identifier, identifier, identifier),
        ))
        search = " ".join((row["record_type"], row["scope_kind"], row["scope_key"], row["topic_key"], row["statement"], row["value_json"], row["state"], row["record_id"], *endpoints))
        haystack = unicodedata.normalize("NFKC", " ".join((row["record_id"], row["content_sha256"], row["topic_key"], row["statement"], search))).casefold()
        match = sum(term in haystack for term in terms)
        values.append((match, row["last_event_seq"], row["record_id"], row))
    values.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple({"candidate_kind": "record", "candidate_id": item[3]["record_id"], "record_type": item[3]["record_type"],
                  "scope_kind": item[3]["scope_kind"], "scope_key": item[3]["scope_key"], "state": item[3]["state"],
                  "sequence": item[3]["record_seq"], "last_lifecycle_event_sequence": item[3]["last_event_seq"],
                  "content_sha256": item[3]["content_sha256"]} for item in values)
