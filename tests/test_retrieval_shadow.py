from __future__ import annotations

import sqlite3
import time
from dataclasses import replace

import pytest

from codex_memory.retrieval.shadow import ShadowComparison, shadow_run_id
from codex_memory.retrieval.provider import ProviderCallStatus


def test_shadow_run_identity_is_deterministic_and_does_not_include_prompt() -> None:
    first = shadow_run_id(
        session_id="session-1",
        turn_id="turn-1",
        hook_event="UserPromptSubmit",
        query="secret prompt must not be stored",
    )
    second = shadow_run_id(
        session_id="session-1",
        turn_id="turn-1",
        hook_event="UserPromptSubmit",
        query="secret prompt must not be stored",
    )

    assert first == second
    assert len(first) == 64
    assert "secret" not in first


def test_shadow_comparison_is_output_inert() -> None:
    comparison = ShadowComparison.skipped("cache_absent")

    assert comparison.response_content is None
    assert comparison.reason == "cache_absent"


def test_shadow_comparison_reports_whole_process_wall_clock_without_request_text() -> None:
    """Audit timing is a bounded diagnostic, not a Hook response field."""
    comparison = ShadowComparison.skipped("provider_offline")

    assert isinstance(comparison.wall_clock_ms, int)
    assert comparison.wall_clock_ms >= 0
    assert "provider_offline" not in repr(comparison.response_content)


def test_semantic_provider_factory_runs_after_the_shared_deadline_starts(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold adapter initialization belongs to the bounded semantic query."""
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths

    events: list[str] = []
    original_start = shadow_module.DeadlineBudget.start

    def observed_start(total_ms, *, now_ns):  # type: ignore[no-untyped-def]
        events.append("deadline")
        return original_start(total_ms, now_ns=now_ns)

    def provider_factory():
        events.append("provider")
        raise RuntimeError("stop after observing construction order")

    monkeypatch.setattr(shadow_module.DeadlineBudget, "start", observed_start)
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, False, False, 8, 8, 8, 8, 1, 0, 2, 30_000,
        "a" * 64, "b" * 64, None, None,
    )
    comparison = shadow_module.compare(
        paths=RetrievalPaths.for_repo(tmp_path),
        request=RetrievalRequest(
            "bounded factory", "session", "turn", "context", None,
            RetrievalScope(str(tmp_path)), mode=RetrievalMode.SHADOW,
        ),
        hook_event="UserPromptSubmit",
        semantic_provider_factory=provider_factory,
        semantic_config=config,
    )

    assert events == ["deadline", "provider"]
    assert comparison.reason == "query_failure"


@pytest.mark.parametrize(
    (
        "deadline_ms",
        "now_ns",
        "optional_deadline_ns",
        "expected_deadline_ns",
    ),
    [
        (1_000, 17_400_000_000, 18_100_000_000, 17_850_000_000),
        (250, 17_000_000_000, 17_600_000_000, 17_250_000_000),
    ],
)
def test_shadow_intersects_configured_and_process_absolute_deadlines_without_rebasing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    deadline_ms: int,
    now_ns: int,
    optional_deadline_ns: int,
    expected_deadline_ns: int,
) -> None:
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import (
        RetrievalMode,
        RetrievalRequest,
        RetrievalScope,
    )
    from codex_memory.retrieval.paths import RetrievalPaths

    observed: dict[str, object] = {}

    def bounded_call(callback, budget):  # type: ignore[no-untyped-def]
        observed["budget"] = budget
        raise shadow_module._DeadlineExpired

    monkeypatch.setattr(shadow_module.time, "monotonic_ns", lambda: now_ns)
    monkeypatch.setattr(shadow_module, "_bounded_call", bounded_call)
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, False, False, 8, 8, 8, 8, 1, 0, 2, 30_000,
        "a" * 64, "b" * 64, None, None,
    )

    comparison = shadow_module.compare(
        paths=RetrievalPaths.for_repo(tmp_path),
        request=RetrievalRequest(
            "bounded factory", "session", "turn", "context", None,
            RetrievalScope(str(tmp_path)), mode=RetrievalMode.SHADOW,
        ),
        hook_event="UserPromptSubmit",
        semantic_provider_factory=lambda: pytest.fail("bounded call must intercept"),
        semantic_config=config,
        deadline_ms=deadline_ms,
        optional_deadline_ns=optional_deadline_ns,
    )

    budget = observed["budget"]
    assert budget.started_ns == now_ns
    assert budget.deadline_ns == expected_deadline_ns
    assert comparison.reason == "deadline_exhausted"


def test_expired_process_absolute_deadline_never_constructs_provider(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import (
        RetrievalMode,
        RetrievalRequest,
        RetrievalScope,
    )
    from codex_memory.retrieval.paths import RetrievalPaths

    now_ns = 19_000_000_000
    monkeypatch.setattr(shadow_module.time, "monotonic_ns", lambda: now_ns)
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, False, False, 8, 8, 8, 8, 1, 0, 2, 30_000,
        "a" * 64, "b" * 64, None, None,
    )

    comparison = shadow_module.compare(
        paths=RetrievalPaths.for_repo(tmp_path),
        request=RetrievalRequest(
            "expired factory", "session", "turn", "context", None,
            RetrievalScope(str(tmp_path)), mode=RetrievalMode.SHADOW,
        ),
        hook_event="UserPromptSubmit",
        semantic_provider_factory=lambda: pytest.fail("expired work must not start"),
        semantic_config=config,
        deadline_ms=1_000,
        optional_deadline_ns=now_ns,
    )

    assert comparison.reason == "deadline_exhausted"


def test_caller_allowance_can_exhaust_optional_shadow_before_provider_start(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approved caller headroom may skip ordinary shadow, never Observer work."""
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import (
        RetrievalMode,
        RetrievalRequest,
        RetrievalScope,
    )
    from codex_memory.retrieval.paths import RetrievalPaths

    now_ns = 19_500_000_000
    monkeypatch.setattr(shadow_module.time, "monotonic_ns", lambda: now_ns)
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 1_000, 1_000,
        100, 100, True, False, False, 8, 8, 8, 8, 1, 0, 2, 30_000,
        "a" * 64, "b" * 64, None, None,
    )

    comparison = shadow_module.compare(
        paths=RetrievalPaths.for_repo(tmp_path),
        request=RetrievalRequest(
            "caller allowance", "session", "turn", "context", None,
            RetrievalScope(str(tmp_path)), mode=RetrievalMode.SHADOW,
        ),
        hook_event="UserPromptSubmit",
        semantic_provider_factory=lambda: pytest.fail("provider must not start"),
        semantic_config=config,
        deadline_ms=1_000,
        optional_deadline_ns=now_ns + 200_000_000,
    )

    assert comparison.reason == "deadline_exhausted"


def test_rerank_uses_a_local_child_budget_without_narrowing_base_query(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.retrieval.contracts import (
        RetrievalMode,
        RetrievalRequest,
        RetrievalScope,
    )
    from codex_memory.retrieval.paths import RetrievalPaths

    from types import SimpleNamespace

    clock = {"now_ns": 21_000_000_000}
    observed: dict[str, object] = {}

    def bounded_call(callback, budget):  # type: ignore[no-untyped-def]
        observed.setdefault("base_budgets", []).append(budget)
        return callback()

    result = SimpleNamespace(
        candidates=(),
        canonical_highwater=None,
        indexed_highwater=None,
        generation=None,
        degraded_reasons=(),
        status=SimpleNamespace(value="absent"),
    )

    class FakeService:
        def __init__(self, paths):  # type: ignore[no-untyped-def]
            observed["paths"] = paths

        def query(self, request, **kwargs):  # type: ignore[no-untyped-def]
            clock["now_ns"] += 200_000_000
            return result

        def query_with_rerank(self, request, **kwargs):  # type: ignore[no-untyped-def]
            observed["rerank_budget"] = kwargs["deadline"]
            return SimpleNamespace(result=result, rerank=None)

    class FakeHealthStore:
        def __init__(self, path):  # type: ignore[no-untyped-def]
            observed["health_path"] = path

    monkeypatch.setattr(
        shadow_module.time, "monotonic_ns", lambda: clock["now_ns"]
    )
    monkeypatch.setattr(shadow_module, "_bounded_call", bounded_call)
    monkeypatch.setattr(
        shadow_module, "_load_retrieval_service", lambda: FakeService
    )
    monkeypatch.setattr(
        shadow_module, "_load_provider_health_store", lambda: FakeHealthStore
    )

    comparison = shadow_module.compare(
        paths=RetrievalPaths.for_repo(tmp_path),
        request=RetrievalRequest(
            "bounded rerank", "session", "turn", "context", None,
            RetrievalScope(str(tmp_path)), mode=RetrievalMode.SHADOW,
        ),
        hook_event="UserPromptSubmit",
        rerank_provider=object(),
        rerank_deadline_ms=500,
        optional_deadline_ns=22_000_000_000,
    )

    base_budget = observed["base_budgets"][0]
    rerank_budget = observed["rerank_budget"]
    assert base_budget.started_ns == 21_000_000_000
    assert base_budget.deadline_ns == 21_750_000_000
    assert rerank_budget.started_ns == 21_200_000_000
    assert rerank_budget.deadline_ns == 21_700_000_000
    assert comparison.reason == "absent"


def test_slow_semantic_provider_factory_cannot_outlive_shared_deadline(tmp_path) -> None:
    """Counting cold construction is insufficient unless waiting is bounded."""
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import DisabledProvider

    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 100, 1_000,
        100, 100, True, False, False, 8, 8, 8, 8, 1, 0, 2, 30_000,
        "a" * 64, "b" * 64, None, None,
    )

    def slow_factory():
        time.sleep(0.5)
        return DisabledProvider()

    started = time.perf_counter()
    comparison = shadow_module.compare(
        paths=RetrievalPaths.for_repo(tmp_path),
        request=RetrievalRequest(
            "bounded factory", "session", "turn", "context", None,
            RetrievalScope(str(tmp_path)), mode=RetrievalMode.SHADOW,
        ),
        hook_event="UserPromptSubmit",
        semantic_provider_factory=slow_factory,
        semantic_config=config,
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000

    assert elapsed_ms < 300
    assert comparison.reason == "deadline_exhausted"


def test_shadow_checks_deterministic_floor_before_calling_provider(runtime) -> None:
    """Missing/corrupt canonical or cache state must not invoke semantic providers."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import ProviderOperation
    from codex_memory.retrieval.shadow import query_shadow
    from retrieval_provider_fakes import FakeProvider

    provider = FakeProvider()
    capabilities = provider.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    capability_calls_before = provider.call_counts[ProviderOperation.CAPABILITIES]
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        capabilities.fingerprint,
        capabilities.dense.identity.fingerprint if capabilities.dense.identity else None,
        capabilities.sparse.identity.fingerprint if capabilities.sparse.identity else None,
        capabilities.reranker.identity.fingerprint if capabilities.reranker.identity else None,
    )
    request = RetrievalRequest(
        "semantic", "session", "turn", "context", None,
        RetrievalScope(str(runtime.paths.repo_root)), mode=RetrievalMode.SHADOW,
    )
    query_shadow(
        paths=RetrievalPaths.for_repo(runtime.paths.repo_root), request=request,
        provider=provider, semantic_config=config,
    )
    assert sum(provider.call_counts.values()) == capability_calls_before


def _shadow_component_case(runtime):
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="shadow.component.flags", statement="safe semantic record", value=None, evidence_uris=(), idempotency_key="shadow-component-flags")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="shadow-component-flags-active")
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild().status.value == "current"
    reference = FakeProvider()
    capabilities = reference.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        capabilities.fingerprint,
        capabilities.dense.identity.fingerprint if capabilities.dense.identity else None,
        capabilities.sparse.identity.fingerprint if capabilities.sparse.identity else None,
        capabilities.reranker.identity.fingerprint if capabilities.reranker.identity else None,
    )
    request = RetrievalRequest(
        "semantic", "session", "turn", "context", None, RetrievalScope(scope),
        mode=RetrievalMode.SHADOW,
    )
    return paths, config, request


def test_explicit_shadow_query_obeys_component_disable_switches(
    runtime, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Component selection under a healthy logical budget, not a latency test."""
    from codex_memory.retrieval import shadow
    from codex_memory.retrieval.cache_database import connect_cache_database
    from codex_memory.retrieval.provider import ProviderOperation
    from retrieval_provider_fakes import FakeProvider

    paths, config, request = _shadow_component_case(runtime)
    # Keep real query, SQLite admission, settlement, workers and validators.
    # This unit's contract is component selection. Its unchanged 500ms budget
    # must not accidentally turn it into another machine-load benchmark.
    monkeypatch.setattr(shadow.time, "monotonic_ns", lambda: 41_000_000_000)
    enabled = FakeProvider()
    enabled_result = shadow.query_shadow(
        paths=paths, request=request, provider=enabled, semantic_config=config,
    )
    disabled = FakeProvider()
    disabled_result = shadow.query_shadow(
        paths=paths, request=request, provider=disabled, semantic_config=config,
        disable_dense=True, disable_sparse=True, disable_reranker=True,
    )

    assert enabled.call_counts[ProviderOperation.DENSE_QUERY] == 1
    assert enabled.call_counts[ProviderOperation.SPARSE_QUERY] == 1
    assert enabled.call_counts[ProviderOperation.RERANK] == 1
    assert sum(disabled.call_counts.values()) == 0
    assert enabled_result.status.value == disabled_result.status.value == "current"
    assert enabled_result.candidates == disabled_result.candidates
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        outcomes = connection.execute(
            "SELECT outcome, provider_status FROM retrieval_provider_attempt_results ORDER BY result_seq"
        ).fetchall()
        assert [tuple(row) for row in outcomes] == [("success", "ok")] * 3
    finally:
        connection.close()


def test_enabled_shadow_components_exhausting_budget_keep_deterministic_floor(
    runtime, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled is not permission to dispatch after reserved budget is exhausted."""
    from codex_memory.retrieval import shadow
    from codex_memory.retrieval.cache_database import connect_cache_database
    from codex_memory.retrieval.provider import ProviderOperation
    from codex_memory.retrieval.provider_control import ProviderCallController
    from codex_memory.retrieval.service import RetrievalService
    from retrieval_provider_fakes import FakeProvider

    paths, config, request = _shadow_component_case(runtime)
    floor = RetrievalService(paths).query(request)
    clock = {"now_ns": 43_000_000_000}
    monkeypatch.setattr(shadow.time, "monotonic_ns", lambda: clock["now_ns"])
    original_dense = ProviderCallController.dense_query

    def completed_dense(controller, *args, **kwargs):
        result = original_dense(controller, *args, **kwargs)
        # A reachable deadline state: only 50ms of the original 500ms remain,
        # less than the existing provider/settlement/return reservation.
        # This is a deterministic state-machine seam, not old wall-clock proof.
        clock["now_ns"] = controller.deadline.deadline_ns - 50_000_000
        return result

    monkeypatch.setattr(ProviderCallController, "dense_query", completed_dense)
    provider = FakeProvider()
    observation = shadow.query_shadow_observation(
        paths=paths, request=request, provider=provider, semantic_config=config,
    )

    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 1
    assert provider.call_counts[ProviderOperation.SPARSE_QUERY] == 0
    assert provider.call_counts[ProviderOperation.RERANK] == 0
    assert observation.provider_trace == (
        {"component": "dense", "operation": "dense_query", "status": "ok"},
        {"component": "sparse", "operation": "sparse_query", "status": "deadline_exhausted"},
        {"component": "reranker", "operation": "rerank", "status": "timeout"},
    )
    assert observation.result.status == floor.status
    assert observation.result.candidates == floor.candidates
    assert observation.result.canonical_highwater == floor.canonical_highwater
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        outcomes = connection.execute(
            "SELECT outcome, provider_status FROM retrieval_provider_attempt_results ORDER BY result_seq"
        ).fetchall()
        assert [tuple(row) for row in outcomes] == [("success", "ok")]
    finally:
        connection.close()


def test_active_shadow_request_is_rejected_before_any_provider_operation(runtime) -> None:
    """ACTIVE is not a provider-enabled alias for the shadow observation mode."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.errors import RetrievalValidationError
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.shadow import compare, query_shadow
    from retrieval_provider_fakes import FakeProvider

    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        "a" * 64, "b" * 64, "c" * 64, "d" * 64,
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    request = RetrievalRequest(
        "safe", "session", "turn", "context", None,
        RetrievalScope(str(runtime.paths.repo_root)), mode=RetrievalMode.ACTIVE,
    )
    direct = FakeProvider()
    with pytest.raises(RetrievalValidationError, match="active retrieval mode is unavailable"):
        query_shadow(paths=paths, request=request, provider=direct, semantic_config=config)
    compared = FakeProvider()
    comparison = compare(
        paths=paths, request=request, hook_event="UserPromptSubmit",
        semantic_provider=compared, semantic_config=config,
    )

    assert sum(direct.call_counts.values()) == 0
    assert sum(compared.call_counts.values()) == 0
    assert (comparison.recorded, comparison.reason) == (False, "active_rejected")


def test_wrong_same_dimension_semantic_fingerprints_contribute_nothing_or_rerank(runtime) -> None:
    """Response shape alone cannot authorize a model output for semantic fusion."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import CandidateChannel, RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import ProviderOperation
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import _shadow_query
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    class WrongQueryFingerprintProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            response = super().embed_query(item, deadline_ms=deadline_ms)
            return replace(response, call=replace(response.call, model_fingerprint="0" * 64))

        def sparse_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            response = super().sparse_query(item, deadline_ms=deadline_ms)
            return replace(response, call=replace(response.call, model_fingerprint="1" * 64))

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="shadow.fingerprint", statement="semantic fingerprint boundary", value=None, evidence_uris=(), idempotency_key="shadow-fingerprint")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="shadow-fingerprint-active")
    runtime.connection.commit()
    builder = FakeProvider()
    capabilities = builder.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        capabilities.fingerprint,
        capabilities.dense.identity.fingerprint if capabilities.dense.identity else None,
        capabilities.sparse.identity.fingerprint if capabilities.sparse.identity else None,
        capabilities.reranker.identity.fingerprint if capabilities.reranker.identity else None,
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    service = RetrievalService(paths)
    assert service.rebuild_semantic(builder, config).status.value == "current"
    wrong = WrongQueryFingerprintProvider()
    outcome = _shadow_query(
        paths=paths,
        request=RetrievalRequest("semantic", "session", "turn", "context", None, RetrievalScope(scope), mode=RetrievalMode.SHADOW),
        provider=wrong,
        semantic_config=config,
        disable_dense=False, disable_sparse=False, disable_reranker=False,
    )

    assert wrong.call_counts[ProviderOperation.DENSE_QUERY] == 1
    assert wrong.call_counts[ProviderOperation.SPARSE_QUERY] == 1
    assert wrong.call_counts[ProviderOperation.RERANK] == 0
    assert all(
        CandidateChannel.DENSE not in trace.source_channels
        and CandidateChannel.SPARSE not in trace.source_channels
        for trace in outcome.result.trace
    )


def test_active_generation_provider_identity_mismatch_blocks_semantic_calls(runtime) -> None:
    """Component fingerprints alone cannot authorize a different provider."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import ProviderCapabilities, ProviderOperation
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import _shadow_query
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    scope = str(runtime.paths.repo_root)
    record = propose_record(
        runtime.connection, record_type="decision", scope_kind="project",
        scope_key=scope, topic_key="shadow.provider.identity",
        statement="provider identity must bind to active semantic generation",
        value=None, evidence_uris=(), idempotency_key="shadow-provider-identity",
    )
    apply_action(
        runtime.connection, record_id=record.record_id, action="activate",
        idempotency_key="shadow-provider-identity-active",
    )
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    first = FakeProvider()
    first_caps = first.capabilities(deadline_ms=500).capabilities
    assert first_caps is not None
    first_config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        first_caps.fingerprint,
        first_caps.dense.identity.fingerprint if first_caps.dense.identity else None,
        first_caps.sparse.identity.fingerprint if first_caps.sparse.identity else None,
        first_caps.reranker.identity.fingerprint if first_caps.reranker.identity else None,
    )
    assert RetrievalService(paths).rebuild_semantic(first, first_config).status.value == "current"

    second = FakeProvider()
    second._capabilities = ProviderCapabilities(
        first_caps.protocol_version, first_caps.provider_type,
        first_caps.provider_version, "e" * 64,
        first_caps.dense, first_caps.sparse, first_caps.reranker,
    )
    second_caps = second.capabilities(deadline_ms=500).capabilities
    assert second_caps is not None
    second_config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        second_caps.fingerprint,
        second_caps.dense.identity.fingerprint if second_caps.dense.identity else None,
        second_caps.sparse.identity.fingerprint if second_caps.sparse.identity else None,
        second_caps.reranker.identity.fingerprint if second_caps.reranker.identity else None,
    )
    outcome = _shadow_query(
        paths=paths,
        request=RetrievalRequest(
            "provider identity", "session", "turn", "context", None,
            RetrievalScope(scope), mode=RetrievalMode.SHADOW,
        ),
        provider=second, semantic_config=second_config,
        disable_dense=False, disable_sparse=False, disable_reranker=False,
    )
    assert second.call_counts[ProviderOperation.DENSE_QUERY] == 0
    assert second.call_counts[ProviderOperation.SPARSE_QUERY] == 0
    assert any(item["status"] == "fingerprint_mismatch" for item in outcome.provider_trace)


@pytest.mark.parametrize(
    "status",
    tuple(value for value in ProviderCallStatus if value is not ProviderCallStatus.OK),
)
def test_dense_non_ok_provider_status_is_preserved_in_shadow_trace(runtime, status) -> None:  # type: ignore[no-untyped-def]
    """An unavailable/busy/timeout result is not evidence of an identity mismatch."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import CandidateChannel, RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import ModelComponent
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import _shadow_query
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key=f"shadow.status.{status.value}", statement="provider status boundary", value=None, evidence_uris=(), idempotency_key=f"shadow-status-{status.value}")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key=f"shadow-status-{status.value}-active")
    runtime.connection.commit()
    builder = FakeProvider()
    capabilities = builder.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000,
        100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000,
        capabilities.fingerprint,
        capabilities.dense.identity.fingerprint if capabilities.dense.identity else None,
        capabilities.sparse.identity.fingerprint if capabilities.sparse.identity else None,
        capabilities.reranker.identity.fingerprint if capabilities.reranker.identity else None,
    )
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild_semantic(builder, config).status.value == "current"
    faulty = FakeProvider()
    faulty.statuses[ModelComponent.DENSE] = status
    outcome = _shadow_query(
        paths=paths,
        request=RetrievalRequest("provider", "session", "turn", "context", None, RetrievalScope(scope), mode=RetrievalMode.SHADOW),
        provider=faulty,
            semantic_config=config,
            disable_dense=False,
            disable_sparse=True,
            disable_reranker=True,
    )

    assert outcome.provider_trace == ({
        "component": "dense", "operation": "dense_query", "status": status.value,
    },)
    assert all(CandidateChannel.DENSE not in trace.source_channels for trace in outcome.result.trace)


@pytest.mark.parametrize("component", ("sparse", "reranker"))
def test_non_ok_sparse_and_reranker_degrade_independently(
    runtime, monkeypatch: pytest.MonkeyPatch, component: str,
) -> None:
    """A failed optional component leaves sibling semantic paths observable."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import ModelComponent, ProviderCallStatus
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval import shadow as shadow_module
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key=f"shadow.independent.{component}", statement="independent provider boundary", value=None, evidence_uris=(), idempotency_key=f"shadow-independent-{component}")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key=f"shadow-independent-{component}-active")
    runtime.connection.commit()
    builder = FakeProvider(); capabilities = builder.capabilities(deadline_ms=500).capabilities
    assert capabilities is not None
    config = SemanticConfig(1, "local_http", "http://127.0.0.1:1", "audited-local-v1", 500, 1_000, 100, 100, True, True, True, 8, 8, 8, 8, 1, 0, 2, 30_000, capabilities.fingerprint, capabilities.dense.identity.fingerprint if capabilities.dense.identity else None, capabilities.sparse.identity.fingerprint if capabilities.sparse.identity else None, capabilities.reranker.identity.fingerprint if capabilities.reranker.identity else None)
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild_semantic(builder, config).status.value == "current"
    monkeypatch.setattr(
        shadow_module.time,
        "monotonic_ns",
        lambda: 10_000_000_000,
    )
    selected = ModelComponent(component)
    faulty = FakeProvider()
    faulty.statuses[selected] = ProviderCallStatus.UNAVAILABLE
    outcome = shadow_module._shadow_query(paths=paths, request=RetrievalRequest("independent provider boundary", "session", "turn", "context", None, RetrievalScope(scope), mode=RetrievalMode.SHADOW), provider=faulty, semantic_config=config, disable_dense=False, disable_sparse=False, disable_reranker=False)

    trace = {entry["component"]: entry["status"] for entry in outcome.provider_trace}
    assert trace[component] == "unavailable"
    assert trace["dense"] == "ok"
    if component == "sparse":
        assert trace["reranker"] == "ok"


def test_shadow_optional_rerank_is_called_but_never_changes_hook_output(runtime) -> None:
    from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.provider import ProviderInput, ProviderOperation
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import compare
    from codex_memory.structured_ledger import apply_action, propose_record
    from retrieval_provider_fakes import FakeProvider

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="shadow.rerank", statement="safe record", value=None, evidence_uris=(), idempotency_key="shadow-rerank")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="shadow-rerank-active")
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root); assert RetrievalService(paths).rebuild().status.value == "current"
    request = RetrievalRequest("safe", "session-rerank", "turn-rerank", "context-rerank", None, RetrievalScope(scope))
    provider = FakeProvider()
    documents = {record.record_id: ProviderInput.from_text(record.record_id, 0, record.statement)}

    comparison = compare(paths=paths, request=request, hook_event="UserPromptSubmit", rerank_provider=provider, rerank_documents=documents, rerank_deadline_ms=500)

    assert comparison.response_content is None
    assert comparison.recorded is True
    assert provider.call_counts[ProviderOperation.RERANK] == 1


def test_shadow_budget_skips_late_audit_mutation(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import compare
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval import cache_database

    scope = str(runtime.paths.repo_root)
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=scope, topic_key="shadow.budget", statement="safe record", value=None, evidence_uris=(), idempotency_key="shadow-budget")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="shadow-budget-active"); runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root); assert RetrievalService(paths).rebuild().status.value == "current"
    original = cache_database.verify_cache_database
    def blocked(connection):  # type: ignore[no-untyped-def]
        time.sleep(0.03); return original(connection)
    monkeypatch.setattr(cache_database, "verify_cache_database", blocked)

    comparison = compare(paths=paths, request=RetrievalRequest("safe", "session-budget", "turn-budget", "context-budget", None, RetrievalScope(scope)), hook_event="UserPromptSubmit", deadline_ms=10)

    assert comparison.recorded is False
    assert comparison.reason == "deadline_exhausted"
    with sqlite3.connect(paths.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM retrieval_runs").fetchone()[0] == 0


def test_shadow_compare_appends_one_hashed_idempotent_audit_without_touching_canonical(
    runtime,
) -> None:
    from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.structured_ledger import apply_action, propose_record
    from codex_memory.retrieval.shadow import compare

    scope = str(runtime.paths.repo_root)
    record = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=scope,
        topic_key="shadow.audit",
        statement="retrieval record",
        value=None,
        evidence_uris=(),
        idempotency_key="shadow-record",
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key="shadow-active",
    )
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild().status.value == "current"
    before = paths.canonical_database_path.read_bytes()
    request = RetrievalRequest(
        "shadow-query-not-for-storage",
        "session-shadow",
        "turn-shadow",
        "context-shadow",
        None,
        RetrievalScope(scope),
    )

    first = compare(paths=paths, request=request, hook_event="UserPromptSubmit")
    second = compare(paths=paths, request=request, hook_event="UserPromptSubmit")

    assert first.recorded is second.recorded is True
    assert paths.canonical_database_path.read_bytes() == before
    with sqlite3.connect(paths.database_path) as connection:
        run_count = connection.execute("SELECT count(*) FROM retrieval_runs").fetchone()[0]
        trace_count = connection.execute(
            "SELECT count(*) FROM retrieval_candidate_traces"
        ).fetchone()[0]
        run = connection.execute(
            "SELECT run_id, query_sha256, mode, wall_clock_ms, provider_trace_json FROM retrieval_runs"
        ).fetchone()
    assert run_count == 1
    assert trace_count >= 1
    assert run is not None and run[2] == "shadow"
    assert isinstance(run[3], int) and run[3] >= 0
    assert run[4] == "[]"
    assert "shadow-query-not-for-storage" not in "".join(str(value) for value in run)


@pytest.mark.parametrize("failure", ("absent", "audit"))
def test_shadow_failures_are_bounded_and_never_change_canonical(
    runtime, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from codex_memory.retrieval.contracts import RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import compare
    from codex_memory.structured_ledger import apply_action, propose_record

    scope = str(runtime.paths.repo_root)
    record = propose_record(
        runtime.connection,
        record_type="decision",
        scope_kind="project",
        scope_key=scope,
        topic_key="shadow.failure",
        statement="safe record",
        value=None,
        evidence_uris=(),
        idempotency_key=f"shadow-{failure}",
    )
    apply_action(
        runtime.connection,
        record_id=record.record_id,
        action="activate",
        idempotency_key=f"shadow-{failure}-active",
    )
    runtime.connection.commit()
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    if failure == "audit":
        assert RetrievalService(paths).rebuild().status.value == "current"
        from codex_memory.retrieval import index_store

        monkeypatch.setattr(
            index_store,
            "append_run",
            lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("audit")),
        )
    before = paths.canonical_database_path.read_bytes()
    request = RetrievalRequest("needle", "session", "turn", "context", None, RetrievalScope(scope))

    comparison = compare(paths=paths, request=request, hook_event="UserPromptSubmit")

    assert comparison.recorded is False
    assert comparison.response_content is None
    assert paths.canonical_database_path.read_bytes() == before
    if failure == "absent":
        assert comparison.reason == "index_absent"
        assert not paths.database_path.exists()
    else:
        assert comparison.reason == "audit_failure"
        with sqlite3.connect(paths.database_path) as connection:
            assert connection.execute("SELECT count(*) FROM retrieval_runs").fetchone()[0] == 0
