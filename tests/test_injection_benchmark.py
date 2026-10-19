from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import json
from pathlib import Path

import pytest

from codex_memory import database, structured_ledger
from codex_memory.retrieval.benchmark import BenchmarkCase
from codex_memory.retrieval.config import (RetrievalConfig, RetrievalConfigStatus,
                                           SemanticConfig, SemanticConfigStatus)
from codex_memory.retrieval.provider import ModelComponent, ProviderOperation
from retrieval_provider_fakes import FakeProvider


ROOT = Path(__file__).resolve().parents[1]


def runner():
    try:
        return import_module("codex_memory.retrieval.p25_benchmark")
    except ModuleNotFoundError:
        pytest.fail("P2.5 real offline benchmark runner is missing", pytrace=False)


def case():
    return BenchmarkCase("unit_01", "unit amber transport", "BENCH_PROJECT", (),
                         {"amber": 3}, (), (), (), "independent runner test")


def record(**changes):
    return {"label": "amber", "record_type": "decision", "scope": "project",
            "topic_key": "unit.amber", "statement": "Use unit amber transport.",
            "lifecycle": "active", "provenance": "direct", **changes}


def scenario(**changes):
    return {"group": "unit", "split": "development", "record_labels": ["amber"],
            "max_chars": 4000, "expected_outcome": "complete", **changes}


def semantic_config(provider: FakeProvider) -> SemanticConfig:
    return SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1",
        1_000, 10_000, 100, 100, True, False, False, 16, 16, 8, 16, 1, 0, 2, 30_000,
        provider.expected_provider_fingerprint,
        provider.component_fingerprint(ModelComponent.DENSE), None, None,
    )


def test_fixed_workload_retains_all_groups_splits_and_budget_failure():
    module = runner()
    workload = module.load_workload(ROOT / "tests/fixtures/retrieval_p25_queries.jsonl")
    assert len(workload.cases) == 24
    assert [len([s for s in workload.scenarios.values() if s["group"] == group])
            for group in ("exact", "symbol", "semantic", "state", "scope", "provenance")] == [4] * 6
    assert workload.scenarios["provenance_04"]["expected_outcome"] == "safe_budget_fallback"
    assert workload.cases[-1].mandatory == ("constitution_large",)


def test_missing_mandatory_remains_in_evaluator_denominator():
    module = runner()
    required = replace(case(), mandatory=("amber",))
    metrics = module.evaluate_case(required, (), latency_ms=0)
    assert metrics["mandatory_miss_count"] == 1
    assert metrics["mandatory_recall"] == 0.0
    assert metrics["recall_at_5"] == 0.0


def test_real_pipeline_commits_and_replays_in_installed_process(tmp_path, monkeypatch):
    module = runner()
    prepared = module.prepare_case(case(), scenario(), [record()], tmp_path / "repository")
    calls = {name: 0 for name in ("query", "assembly", "prepare", "commit", "explain")}
    boundaries = ((module.RetrievalService, "query", "query"),
                  (module, "build_phase2_packet", "assembly"),
                  (module, "prepare_shadow_manifest", "prepare"),
                  (module.InjectionStore, "commit", "commit"),
                  (module, "explain_in_child", "explain"))
    for owner, attribute, name in boundaries:
        original = getattr(owner, attribute)
        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, attribute, counted)
    result = module.query_case(prepared, installed_entry=module.installed_entry())
    assert result["status"] == "complete", result
    assert result["phase2_selected_labels"] == ["amber"]
    assert result["legacy_selected_labels"] == ["amber"]
    assert calls == {name: 1 for name in calls}
    assert result["calls"] == calls
    assert result["explain"]["return_code"] == 0
    assert result["explain"]["document"]["source"]["origin"] == "offline"
    assert result["explain"]["document"]["source"]["event_id"] == prepared.source.event_id
    assert result["explain"]["document"]["delivery"]["host_acknowledgement"] == "unknown"
    assert result["implementation_sha256_before"] == result["implementation_sha256_after"]


def test_legacy_phase2_and_explain_latencies_are_independently_measured(tmp_path, monkeypatch):
    module = runner()
    prepared = module.prepare_case(case(), scenario(), [record()], tmp_path / "repository")
    original_legacy = module.build_memory_packet
    original_query = module.RetrievalService.query
    original_explain = module.explain_in_child

    def delayed_legacy(*args, **kwargs):
        module.time.sleep(0.015)
        return original_legacy(*args, **kwargs)

    def delayed_query(*args, **kwargs):
        module.time.sleep(0.045)
        return original_query(*args, **kwargs)

    def delayed_explain(*args, **kwargs):
        module.time.sleep(0.080)
        return original_explain(*args, **kwargs)

    monkeypatch.setattr(module, "build_memory_packet", delayed_legacy)
    monkeypatch.setattr(module.RetrievalService, "query", delayed_query)
    monkeypatch.setattr(module, "explain_in_child", delayed_explain)
    result = module.query_case(prepared, installed_entry=module.installed_entry())
    assert result["status"] == "complete", result
    assert result["legacy_metrics"]["latency_ms"] == result["legacy_duration_ms"]
    assert result["phase2_metrics"]["latency_ms"] == result["phase2_pipeline_duration_ms"]
    assert result["legacy_duration_ms"] < result["phase2_pipeline_duration_ms"]
    assert result["phase2_pipeline_duration_ms"] < result["wall_clock_ms"]
    assert result["explain"]["wall_clock_ms"] >= 80


def test_configured_query_uses_formal_semantic_build_validation_and_query(tmp_path, monkeypatch):
    module = runner()
    prepared = module.prepare_case(case(), scenario(), [record()], tmp_path / "repository")
    provider = FakeProvider()
    config = semantic_config(provider)
    calls = {"build": 0, "validation": 0, "query": 0}
    original_build = module.RetrievalService.rebuild_semantic
    original_ready = module._enabled_semantic_components_ready
    original_query = module._shadow_query

    def counted_build(service, actual_provider, actual_config):
        calls["build"] += 1
        assert service.paths == prepared.paths
        assert actual_provider is provider and actual_config is config
        return original_build(service, actual_provider, actual_config)

    def counted_ready(paths, actual_config):
        calls["validation"] += 1
        assert paths == prepared.paths and actual_config is config
        return original_ready(paths, actual_config)

    def counted_query(**kwargs):
        calls["query"] += 1
        assert kwargs["paths"] == prepared.paths
        assert kwargs["provider"] is provider and kwargs["semantic_config"] is config
        assert kwargs["deadline"].deadline_ns - kwargs["deadline"].started_ns == 1_000_000_000
        return original_query(**kwargs)

    monkeypatch.setattr(module.RetrievalService, "rebuild_semantic", counted_build)
    monkeypatch.setattr(module, "_enabled_semantic_components_ready", counted_ready)
    monkeypatch.setattr(module, "_shadow_query", counted_query)
    result = module.query_case(
        prepared, installed_entry=module.installed_entry(),
        provider=provider, semantic_config=config, profile="configured",
    )
    assert result["status"] == "complete", result
    assert calls == {"build": 1, "validation": 1, "query": 1}
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] > 0
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] > 0
    assert result["provider_call_count"] is None
    identities = result["environment"]["provider"]
    assert identities["configured"] == provider.expected_provider_fingerprint
    assert identities["observed"] == identities["validated"]
    assert identities["observed"] == provider.expected_provider_fingerprint
    assert result["environment"]["model"]["observed"] == result["environment"]["model"]["configured"]
    assert result["semantic_identity_evidence"]["provider_fingerprint"] == provider.expected_provider_fingerprint
    assert result["provider_admission_attempt_count"] == sum(provider.call_counts.values())
    assert result["provider_http_request_count"] is None
    assert result["provider_trace_count"] < result["provider_admission_attempt_count"]
    preprocessing = result["environment"]["preprocessing"]
    assert preprocessing["configured"] == "not_configured"
    assert preprocessing["observed"] == preprocessing["validated"]
    assert len(preprocessing["observed"]) == 64


def test_configured_query_reuses_outer_budget_instead_of_restart(tmp_path):
    module = runner()
    prepared = module.prepare_case(case(), scenario(), [record()], tmp_path / "repository")
    class ObservedProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.query_deadlines = []
        def embed_query(self, item, *, deadline_ms):
            self.query_deadlines.append(deadline_ms)
            return super().embed_query(item, deadline_ms=deadline_ms)
    provider = ObservedProvider()
    result = module.query_case(prepared, installed_entry=module.installed_entry(), profile="configured",
                               provider=provider, semantic_config=replace(semantic_config(provider), query_deadline_ms=14000))
    assert result["status"] == "complete", result
    assert provider.query_deadlines and max(provider.query_deadlines) <= 1000


def test_failed_query_keeps_elapsed_latency_and_missing_mandatory(tmp_path, monkeypatch):
    module = runner()
    prepared = module.prepare_case(replace(case(), mandatory=("amber",)), scenario(), [record()], tmp_path / "repository")
    original = module.RetrievalService.query
    def delayed(service, request):
        module.time.sleep(0.03)
        return original(service, request)
    monkeypatch.setattr(module.RetrievalService, "query", delayed)
    result = module.query_case(prepared, installed_entry=module.installed_entry(), budget_ms=10)
    assert result["status"] == "failed"
    assert result["phase2_pipeline_duration_ms"] >= 30
    assert result["phase2_metrics"]["latency_ms"] >= 30
    assert result["phase2_metrics"]["mandatory_miss_count"] == 1


def test_stage_not_started_is_unknown_latency_without_dropping_case(tmp_path, monkeypatch):
    module = runner()
    prepared = module.prepare_case(replace(case(), mandatory=("amber",)), scenario(), [record()], tmp_path / "repository")
    provider = FakeProvider()
    def unavailable_build(*args, **kwargs):
        raise OSError("controlled semantic build failure")
    monkeypatch.setattr(module.RetrievalService, "rebuild_semantic", unavailable_build)
    result = module.query_case(prepared, installed_entry=module.installed_entry(), profile="configured",
                               provider=provider, semantic_config=semantic_config(provider))
    assert result["status"] == "failed"
    assert result["phase2_pipeline_duration_ms"] is None
    assert result["phase2_metrics"]["latency_ms"] is None
    aggregate = module._aggregate_observations([result])
    assert aggregate["case_count"] == 1
    assert aggregate["phase2"]["mandatory_miss_count"] == 1
    assert aggregate["phase2"]["latency_p50_ms"] is None
    assert aggregate["phase2"]["latency_unavailable_count"] == 1


def test_same_query_changed_lifecycle_changes_real_trace(tmp_path):
    module = runner()
    prepared = module.prepare_case(case(), scenario(), [record()], tmp_path / "repository")
    before = module.query_case(prepared, installed_entry=module.installed_entry())
    assert before["status"] == "complete", before
    connection = database.connect_database(prepared.runtime.db_path)
    try:
        structured_ledger.apply_action(connection, record_id=prepared.label_to_record_id["amber"],
                                       action="invalidate", idempotency_key="unit-invalidate")
    finally:
        connection.close()
    assert module.RetrievalService(prepared.paths).catch_up().status.value == "current"
    after = module.query_case(prepared, installed_entry=module.installed_entry())
    assert after["status"] == "complete", after
    assert before["query_sha256"] == after["query_sha256"]
    assert before["phase2_selected_labels"] == ["amber"]
    assert after["phase2_selected_labels"] == []
    assert before["trace_sha256"] != after["trace_sha256"]
    assert before["manifest_id"] != after["manifest_id"]


def test_certified_oversize_keeps_failure_and_missing_mandatory(tmp_path):
    module = runner()
    required = replace(case(), mandatory=("amber",))
    prepared = module.prepare_case(required, scenario(expected_outcome="safe_budget_fallback"),
                                   [record(record_type="constitution", provenance="certified",
                                           statement="All mandatory evidence stays complete. " * 62)],
                                   tmp_path / "repository")
    result = module.query_case(prepared, installed_entry=module.installed_entry())
    assert result["status"] == "safe_budget_fallback", result
    assert result["phase2_metrics"]["mandatory_miss_count"] == 1
    assert result["phase2_metrics"]["mandatory_recall"] == 0.0
    assert result["calls"] == {"query": 1, "assembly": 1, "prepare": 0, "commit": 0, "explain": 0}
    assert result["manifest_id"] is None


def test_configured_without_explicit_provider_retains_blocked_artifact(tmp_path):
    module = runner()
    output = tmp_path / ".codex-memory/p25-evidence/configured"
    report = module.run_benchmark(profile="configured",
                                  fixture=ROOT / "tests/fixtures/retrieval_p25_queries.jsonl",
                                  output_dir=output)
    assert report["status"] == "BLOCKED"
    assert report["provider_call_count"] == 0
    assert report["cases"] == []
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["status"] == "BLOCKED"


def test_configured_run_uses_selected_valid_provider_instead_of_unimplemented(tmp_path, monkeypatch):
    module = runner()
    full = module.load_workload(ROOT / "tests/fixtures/retrieval_p25_queries.jsonl")
    one = replace(full, cases=(full.cases[0],))
    provider = FakeProvider()
    config = semantic_config(provider)
    loaded = RetrievalConfig(RetrievalConfigStatus.VALID, "shadow", SemanticConfigStatus.VALID,
                             config, None, "a" * 64)
    monkeypatch.setattr(module, "load_workload", lambda _fixture: one)
    monkeypatch.setattr(module, "load_retrieval_config", lambda _repo: loaded)
    monkeypatch.setattr(module, "provider_from_config", lambda actual: provider if actual is config else None)
    output = tmp_path / ".codex-memory/p25-evidence/configured-valid"
    report = module.run_benchmark(profile="configured",
                                  fixture=ROOT / "tests/fixtures/retrieval_p25_queries.jsonl",
                                  output_dir=output)
    assert report["status"] in {"PASS", "FAIL"}, report
    assert len(report["cases"]) == 1
    assert report["cases"][0]["profile"] == "configured"
    assert provider.call_counts[ProviderOperation.DENSE_DOCUMENTS] > 0
    assert provider.call_counts[ProviderOperation.DENSE_QUERY] > 0


def test_evidence_output_cannot_overwrite_or_escape_root(tmp_path):
    module = runner()
    with pytest.raises(ValueError, match="p25-evidence"):
        module.run_benchmark(profile="configured", fixture=ROOT / "tests/fixtures/retrieval_p25_queries.jsonl",
                             output_dir=tmp_path / "arbitrary")
    output = tmp_path / ".codex-memory/p25-evidence/existing"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.run_benchmark(profile="configured", fixture=ROOT / "tests/fixtures/retrieval_p25_queries.jsonl",
                             output_dir=output)


def test_parent_traversal_cannot_escape_evidence_directory(tmp_path):
    module = runner()
    output = tmp_path / ".codex-memory/p25-evidence/../../escaped"
    with pytest.raises(ValueError, match="p25-evidence"):
        module.run_benchmark(profile="configured", fixture=ROOT / "tests/fixtures/retrieval_p25_queries.jsonl",
                             output_dir=output)
    assert not (tmp_path / "escaped").exists()


def test_unavailable_fixture_preserves_failure_artifact(tmp_path):
    module = runner()
    output = tmp_path / ".codex-memory/p25-evidence/missing-fixture"
    with pytest.raises(ValueError):
        module.run_benchmark(profile="deterministic", fixture=tmp_path / "missing.jsonl", output_dir=output)
    artifact = output / "setup-failure.json"
    assert artifact.is_file()
    failure = json.loads(artifact.read_text(encoding="utf-8"))
    assert failure["status"] == "FAIL"
    assert failure["failure_stage"] == "fixture_load"
    assert failure["started_at"] <= failure["ended_at"]
