from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from dataclasses import replace
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest

from codex_memory.retrieval.config import SemanticConfig
from codex_memory.retrieval.provider import (
    DenseResponse, ModelComponent, ProviderCallStatus, ProviderOperation, RerankItem,
    RerankResponse,
)
from retrieval_provider_fakes import FakeProvider


_FIXTURE = Path(__file__).with_name("fixtures") / "retrieval_p23_benchmark.jsonl"
_CANONICAL = Path(__file__).with_name("fixtures") / "retrieval_p23_canonical_fixture.json"


def _config(provider: FakeProvider) -> SemanticConfig:
    return SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1",
        1_000, 10_000, 100, 100, True, True, True, 16, 16, 8, 16, 1, 0, 2, 30_000,
        provider.expected_provider_fingerprint,
        provider.component_fingerprint(ModelComponent.DENSE),
        provider.component_fingerprint(ModelComponent.SPARSE),
        provider.component_fingerprint(ModelComponent.RERANKER),
    )


def test_fake_pipeline_uses_legal_ledger_projection_semantic_build_and_real_queries(tmp_path: Path) -> None:
    """Would fail if a benchmark could obtain final ranks from relevance labels."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import run_benchmark_pipeline

    provider = FakeProvider()
    result = run_benchmark_pipeline(
        load_benchmark_fixture(_FIXTURE), _CANONICAL, tmp_path,
        provider_factory=lambda: provider, config_factory=_config,
    )

    assert [run.name for run in result.runs] == [
        "p2_2", "p2_2_dense", "p2_2_dense_reranker", "p2_2_dense_sparse_reranker",
    ]
    assert provider.call_counts[ProviderOperation.CAPABILITIES] > 0
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] > 0
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] > 0
    assert provider.call_counts[ProviderOperation.SPARSE_QUERY] > 0
    assert provider.call_counts[ProviderOperation.RERANK] > 0
    # The fixture reaches the official provider boundary with multiple real
    # document and query inputs; it is not selecting benchmark labels directly.
    assert len(set(provider.call_inputs[ProviderOperation.DENSE_DOCUMENTS][0])) > 1
    assert len({inputs[0] for inputs in provider.call_inputs[ProviderOperation.DENSE_QUERY]}) > 1
    assert len({inputs[0] for inputs in provider.call_inputs[ProviderOperation.SPARSE_QUERY]}) > 1
    # The pipeline returns a text-free audit summary so reports can prove input
    # variation without serializing fixture prompts or document content.
    assert result.provider_input_audit["dense_query"] == {
        "call_count": provider.call_counts[ProviderOperation.DENSE_QUERY],
        "distinct_input_sha256_count": len(
            {inputs[0] for inputs in provider.call_inputs[ProviderOperation.DENSE_QUERY]}
        ),
    }
    assert result.provider_input_audit["capabilities"]["call_count"] == (
        provider.call_counts[ProviderOperation.CAPABILITIES]
    )
    assert result.provider_call_count == sum(
        entry["admission_count"] for entry in result.provider_attempt_audit.values()
    )
    for operation in ProviderOperation:
        audit = result.provider_attempt_audit[operation.value]
        assert audit == {
            "admission_count": provider.call_counts[operation],
            "result_count": provider.call_counts[operation],
            "success_count": provider.call_counts[operation],
            "failure_count": 0,
            "unresolved_count": 0,
        }
    assert "mem_exact_record" in result.runs[0].selected_ids["exact-record-id"]
    assert "mem_other_scope" not in result.runs[0].selected_ids["exact-record-id"]
    assert "mem_superseded" not in result.runs[0].selected_ids["exact-record-id"]
    assert all(run.selected_ids for run in result.runs)
    case_ids = {case.case_id for case in load_benchmark_fixture(_FIXTURE)}
    for run in result.runs:
        assert set(run.query_status or ()) == case_ids
        assert set(run.candidate_trace or ()) == case_ids
        assert set((run.query_status or {}).values()) == {"current"}
        for case_id in case_ids:
            selected_from_trace = tuple(
                item["candidate_id"]
                for item in (run.candidate_trace or {})[case_id]
                if item["selected"]
            )
            assert selected_from_trace == tuple(run.selected_ids[case_id])


def test_fixture_repository_rejects_a_different_git_root(tmp_path: Path, monkeypatch) -> None:
    """Fixture scope keys must never be rebound to another repository."""
    from codex_memory.retrieval import benchmark_pipeline

    repo = tmp_path / "fixture"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()

    def completed(command, **kwargs):  # type: ignore[no-untyped-def]
        if command[-2:] == ["init", "--quiet"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, str(other), "")

    monkeypatch.setattr(benchmark_pipeline.subprocess, "run", completed)
    with pytest.raises(benchmark_pipeline.PipelineBlocked) as caught:
        benchmark_pipeline._initialize_fixture_repository(repo)

    assert caught.value.status == "fixture_git_root"


def test_pipeline_query_override_changes_real_selected_candidates_not_relevance_labels(tmp_path: Path) -> None:
    """Would fail if changing a query left a relevance-derived result untouched."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import run_benchmark_pipeline

    cases = load_benchmark_fixture(_FIXTURE)
    normal = run_benchmark_pipeline(cases, _CANONICAL, tmp_path / "normal", provider_factory=FakeProvider, config_factory=_config)
    changed = run_benchmark_pipeline(
        cases, _CANONICAL, tmp_path / "changed", provider_factory=FakeProvider,
        config_factory=_config,
        query_overrides={"exact-record-id": "synthetic English semantic paraphrase query"},
    )

    assert normal.runs[0].selected_ids["exact-record-id"] != changed.runs[0].selected_ids["exact-record-id"]
    assert normal.runs[0].candidate_trace["exact-record-id"] != changed.runs[0].candidate_trace["exact-record-id"]


def test_prepared_fixture_scope_matches_fresh_child_git_root(tmp_path: Path) -> None:
    """Windows short/long temp aliases must not change the project scope key."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import prepare_benchmark_pipeline

    with TemporaryDirectory(prefix="codex-memory-short-alias-") as temporary:
        prepared = prepare_benchmark_pipeline(
            load_benchmark_fixture(_FIXTURE), _CANONICAL, Path(temporary),
            provider_factory=FakeProvider, config_factory=_config,
        )
        subprocess.run(
            ["git", "-C", str(prepared.repo), "init", "--quiet"],
            check=True,
        )
        root = subprocess.run(
            ["git", "-C", str(prepared.repo), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        assert str(Path(root)).replace("/", "\\").casefold() == str(prepared.repo).casefold()


def test_prepared_fixture_repeats_real_pipeline_deterministically_and_query_reaches_provider(
    tmp_path: Path,
) -> None:
    """One established fixture, not two rebuilt databases, proves real E2E repeatability."""
    from codex_memory.retrieval.benchmark import evaluate_profiles, load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import (
        _run_profile,
        prepare_benchmark_pipeline,
    )

    cases = load_benchmark_fixture(_FIXTURE)
    provider = FakeProvider()
    prepared = prepare_benchmark_pipeline(
        cases, _CANONICAL, tmp_path, provider_factory=lambda: provider, config_factory=_config,
    )
    reverse = {record_id: label for label, record_id in prepared.label_to_record_id.items()}

    def run_dense(overrides):  # type: ignore[no-untyped-def]
        return _run_profile(
            "p2_2_dense", False, True, True, cases, overrides,
            prepared.paths, prepared.service, prepared.provider, prepared.config,
            prepared.repo, prepared.workspace, reverse,
        )[0]

    first = run_dense({})
    second = run_dense({})

    def stable(run):  # type: ignore[no-untyped-def]
        return (
            run.name, run.selected_ids, run.status, run.component_status,
            run.query_status, run.candidate_trace,
        )

    assert stable(first) == stable(second)
    # The evaluator sees the same genuine selected IDs and trace.  Real timing
    # is intentionally excluded from this determinism check.
    def without_latency(value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            return {
                key: without_latency(item)
                for key, item in value.items()
                if key not in {"latency_ms", "latency_p50_ms", "latency_p95_ms"}
            }
        if isinstance(value, list):
            return [without_latency(item) for item in value]
        return value

    assert without_latency(evaluate_profiles(
        cases, (first,), clock_ms=0, evidence_label="A: framework/FakeProvider",
    )) == without_latency(evaluate_profiles(
        cases, (second,), clock_ms=0, evidence_label="A: framework/FakeProvider",
    ))

    calls_before = len(provider.call_inputs[ProviderOperation.DENSE_QUERY])
    changed = run_dense({"exact-record-id": "synthetic English semantic paraphrase query"})
    changed_dense_inputs = provider.call_inputs[ProviderOperation.DENSE_QUERY][calls_before:]
    assert changed_dense_inputs
    assert changed_dense_inputs[0] != provider.call_inputs[ProviderOperation.DENSE_QUERY][0]
    assert second.selected_ids["exact-record-id"] != changed.selected_ids["exact-record-id"]
    assert second.candidate_trace["exact-record-id"] != changed.candidate_trace["exact-record-id"]


def test_prepared_fixture_temporary_candidates_are_session_scoped_and_terminal_candidates_hidden(
    tmp_path: Path,
) -> None:
    """Candidate capture/review uses the append-only API, not a test-only table write."""
    from codex_memory import database
    from codex_memory.blob_store import BlobStore
    from codex_memory.candidate_memory import apply_candidate_action, capture_candidate
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import prepare_benchmark_pipeline
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope

    cases = load_benchmark_fixture(_FIXTURE)
    prepared = prepare_benchmark_pipeline(
        cases, _CANONICAL, tmp_path, provider_factory=FakeProvider, config_factory=_config,
    )
    connection = database.connect_database(prepared.runtime.db_path)
    try:
        blobs = BlobStore(prepared.runtime.blobs_dir)
        visible = capture_candidate(
            connection, blobs, project_scope_key=str(prepared.repo), record_type="decision",
            scope_kind="project", scope_key=str(prepared.repo), topic_key="benchmark.temporary.visible",
            statement="synthetic same session temporary candidate", session_id="temporary-session",
            turn_id="temporary-turn", context_id="temporary-context",
            idempotency_key="benchmark-temporary-visible",
        )
        terminal = capture_candidate(
            connection, blobs, project_scope_key=str(prepared.repo), record_type="decision",
            scope_kind="project", scope_key=str(prepared.repo), topic_key="benchmark.temporary.terminal",
            statement="synthetic terminal temporary candidate", session_id="terminal-session",
            turn_id="terminal-turn", context_id="terminal-context",
            idempotency_key="benchmark-temporary-terminal",
        )
        terminal_action = apply_candidate_action(
            connection, blobs, candidate_id=terminal.candidate_id, action="expire",
            reason="benchmark terminal candidate", idempotency_key="benchmark-temporary-expire",
        )
    finally:
        connection.close()
    assert terminal_action.state == "expired"

    def request(session_id: str) -> RetrievalRequest:
        return RetrievalRequest(
            "synthetic temporary query", session_id, "temporary-query-turn",
            "temporary-query-context", None, RetrievalScope(str(prepared.repo)), mode=RetrievalMode.OFF,
        )

    same_session = prepared.service.query(request("temporary-session"))
    other_session = prepared.service.query(request("other-session"))
    terminal_session = prepared.service.query(request("terminal-session"))
    assert visible.candidate_id in {candidate.candidate_id for candidate in same_session.candidates}
    assert visible.candidate_id not in {candidate.candidate_id for candidate in other_session.candidates}
    assert terminal.candidate_id not in {candidate.candidate_id for candidate in terminal_session.candidates}


def test_prepared_fixture_catches_up_unvectorized_record_without_semantic_rows(tmp_path: Path) -> None:
    """A post-build permanent record remains deterministically visible without vectors."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import prepare_benchmark_pipeline
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths
    from codex_memory.retrieval.semantic_store import open_verified_semantic_snapshot

    cases = load_benchmark_fixture(_FIXTURE)
    prepared = prepare_benchmark_pipeline(
        cases, _CANONICAL, tmp_path, provider_factory=FakeProvider, config_factory=_config,
    )
    record_id = prepared.label_to_record_id["mem_new_unembedded"]
    result = prepared.service.query(RetrievalRequest(
        "synthetic newly promoted query", "unembedded-session", "unembedded-turn",
        "unembedded-context", None, RetrievalScope(str(prepared.repo)), mode=RetrievalMode.OFF,
    ))
    assert record_id in {candidate.candidate_id for candidate in result.candidates}
    assert record_id in {item.candidate_id for item in result.trace}

    with sqlite3.connect(prepared.paths.database_path) as cache:
        active = cache.execute(
            "SELECT active_semantic_generation_id FROM retrieval_semantic_state WHERE singleton_id = 1"
        ).fetchone()[0]
    semantic_paths = SemanticGenerationPaths.for_generation(prepared.paths.generations_dir, active)
    with open_verified_semantic_snapshot(semantic_paths) as (_manifest, semantic):
        assert semantic.execute(
            "SELECT COUNT(*) FROM semantic_dense_vectors AS v JOIN semantic_documents AS d "
            "ON d.document_id = v.document_id WHERE d.record_id = ?", (record_id,),
        ).fetchone()[0] == 0
        assert semantic.execute(
            "SELECT COUNT(*) FROM semantic_sparse_postings AS p JOIN semantic_documents AS d "
            "ON d.document_id = p.document_id WHERE d.record_id = ?", (record_id,),
        ).fetchone()[0] == 0


def test_pipeline_benchmark_consumes_actual_reranked_order_and_trace(tmp_path: Path) -> None:
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import run_benchmark_pipeline

    class ReverseRerankProvider(FakeProvider):
        def rerank(self, query, documents, *, deadline_ms):  # type: ignore[no-untyped-def]
            call = self._call(ProviderOperation.RERANK, ModelComponent.RERANKER)
            return RerankResponse(
                call,
                tuple(
                    RerankItem(
                        document.item_id,
                        document.position,
                        float(document.position),
                        False,
                    )
                    for document in documents
                ),
            )

    provider = ReverseRerankProvider()
    result = run_benchmark_pipeline(
        load_benchmark_fixture(_FIXTURE), _CANONICAL, tmp_path,
        provider_factory=lambda: provider, config_factory=_config,
    )
    dense = next(run for run in result.runs if run.name == "p2_2_dense")
    reranked = next(run for run in result.runs if run.name == "p2_2_dense_reranker")
    assert any(
        dense.selected_ids[case_id] != reranked.selected_ids[case_id]
        for case_id in dense.selected_ids
    )
    assert any(
        "reranked" in item["reason_codes"]
        for items in reranked.candidate_trace.values()
        for item in items
    )


def test_pipeline_component_status_uses_query_time_provider_trace(tmp_path: Path) -> None:
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import run_benchmark_pipeline

    class QueryTimeoutProvider(FakeProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            call = replace(
                self._call(ProviderOperation.DENSE_QUERY, None),
                component=ModelComponent.DENSE,
                status=ProviderCallStatus.TIMEOUT,
                reason=ProviderCallStatus.TIMEOUT.value,
            )
            return DenseResponse(call, ())

    provider = QueryTimeoutProvider()
    result = run_benchmark_pipeline(
        load_benchmark_fixture(_FIXTURE), _CANONICAL, tmp_path,
        provider_factory=lambda: provider, config_factory=_config,
    )
    dense = next(run for run in result.runs if run.name == "p2_2_dense")
    assert dense.status == "DEGRADED"
    assert dense.component_status["dense"] == "DEGRADED"


def test_missing_configured_factory_is_blocked_without_fake_fallback(tmp_path: Path) -> None:
    """Would fail if configured mode silently replaced a missing live adapter with FakeProvider."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import PipelineBlocked, run_benchmark_pipeline

    try:
        run_benchmark_pipeline(load_benchmark_fixture(_FIXTURE), _CANONICAL, tmp_path, provider_factory=None, config_factory=None)
    except PipelineBlocked as error:
        assert error.status == "BLOCKED"
    else:
        raise AssertionError("missing configured provider must block")


def test_pipeline_settlement_failure_discards_semantic_results_and_keeps_p2_2_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A post-call health settlement failure cannot change offline selections."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import prepare_benchmark_pipeline
    from codex_memory.retrieval.contracts import (
        CandidateChannel,
        RetrievalMode,
        RetrievalRequest,
        RetrievalScope,
    )
    from codex_memory.retrieval.provider_health import HealthWriteOutcome
    from codex_memory.retrieval.service import RetrievalService
    from codex_memory.retrieval.shadow import query_shadow_observation

    settlement_calls = []

    def not_persisted(self, decision, *args, **kwargs):  # type: ignore[no-untyped-def]
        settlement_calls.append(decision.operation)
        return HealthWriteOutcome.NOT_PERSISTED

    builder = FakeProvider()
    prepared = prepare_benchmark_pipeline(
        load_benchmark_fixture(_FIXTURE),
        _CANONICAL,
        tmp_path,
        provider_factory=lambda: builder,
        config_factory=_config,
    )

    # Install the result-write failure only after a real generation was built
    # and activated through the normal pipeline.
    from codex_memory.retrieval.provider_health import ProviderHealthStore
    monkeypatch.setattr(ProviderHealthStore, "record_success", not_persisted)
    repo = prepared.repo
    paths = prepared.paths
    request = RetrievalRequest(
        "synthetic exact identifier query",
        "settlement-session",
        "settlement-turn",
        "settlement-context",
        None,
        RetrievalScope(str(repo)),
        mode=RetrievalMode.SHADOW,
    )
    baseline = RetrievalService(paths).query(request)
    provider = FakeProvider()
    observation = query_shadow_observation(
        paths=paths,
        request=request,
        provider=provider,
        semantic_config=_config(provider),
        disable_sparse=True,
        disable_reranker=True,
    )

    assert provider.call_counts[ProviderOperation.DENSE_QUERY] == 1
    assert settlement_calls == [ProviderOperation.DENSE_QUERY]
    assert observation.result.candidates == baseline.candidates
    assert any(
        entry == {
            "component": "dense",
            "operation": "dense_query",
            "status": "unavailable",
        }
        for entry in observation.provider_trace
    )
    assert all(
        CandidateChannel.DENSE not in item.source_channels
        for item in observation.result.trace
    )


def test_pipeline_fixture_remains_available_for_verified_semantic_reopen_in_child_process(
    tmp_path: Path,
) -> None:
    """A child uses the installed runtime, not an in-process cached result."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import run_benchmark_pipeline

    result = run_benchmark_pipeline(
        load_benchmark_fixture(_FIXTURE),
        _CANONICAL,
        tmp_path,
        provider_factory=FakeProvider,
        config_factory=_config,
    )
    assert result.runs
    child = tmp_path / "verified_semantic_reopen.py"
    child.write_text(
        """
import json
from pathlib import Path
import sys

from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService, SemanticQueryInputs

repo = Path(sys.argv[1]) / 'benchmark-repository'
request = RetrievalRequest(
    'synthetic exact identifier query', 'child-session', 'child-turn',
    'child-context', None, RetrievalScope(str(repo)), mode=RetrievalMode.SHADOW,
)
result = RetrievalService(RetrievalPaths.for_repo(repo)).query(
    request, semantic_query=SemanticQueryInputs((1.0, 2.0, 3.0, 4.0)),
)
print(json.dumps({
    'status': result.status.value,
    'generation': None if result.generation is None else result.generation.generation_id,
    'has_dense_trace': any(
        any(channel.value == 'dense' for channel in entry.source_channels)
        for entry in result.trace
    ),
}, sort_keys=True))
""".strip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(child), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["status"] == "current"
    assert isinstance(document["generation"], str) and document["generation"]
    assert document["has_dense_trace"] is True
