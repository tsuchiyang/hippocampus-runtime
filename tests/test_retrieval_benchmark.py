from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest


_FIXTURE = Path(__file__).with_name("fixtures") / "retrieval_p23_benchmark.jsonl"
_GENERATION_ID = "gen_0123456789abcdef"


def _benchmark_script_module():  # type: ignore[no-untyped-def]
    """Load the developer-only command as a unit-test module."""
    path = Path(__file__).parents[1] / "scripts" / "benchmark_retrieval_p23.py"
    spec = importlib.util.spec_from_file_location("benchmark_retrieval_p23_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_fixture_has_all_required_case_kinds_and_strict_schema() -> None:
    """Would fail if the P2.3 fixed fixture drifted or accepted unknown fields."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture

    cases = load_benchmark_fixture(_FIXTURE)

    assert [case.case_id for case in cases] == [
        "exact-record-id", "exact-topic-hash", "python-dotted-symbol",
        "path-command-symbol", "zh-semantic-decision", "en-semantic-decision",
        "mixed-language", "lexical-distractor", "near-duplicate-lifecycle",
        "scope-isolation", "unembedded-promoted", "mandatory-constitution",
    ]
    assert len(cases) == 12


def test_metrics_report_hit_false_recall_leaks_and_no_gain_stably() -> None:
    """Would fail if benchmark arithmetic or isolation gates were not real metrics."""
    from codex_memory.retrieval.benchmark import BenchmarkProfileRun, evaluate_profiles, load_benchmark_fixture

    cases = load_benchmark_fixture(_FIXTURE)
    baseline = {case.case_id: tuple(dict.fromkeys((*case.relevant, *case.mandatory))) for case in cases}
    semantic = dict(baseline)
    semantic["lexical-distractor"] = ("mem_lexical_distractor", "mem_lexical_target")
    semantic["mandatory-constitution"] = ()
    report = evaluate_profiles(
        cases,
        (
            BenchmarkProfileRun("p2_2", baseline, "AVAILABLE", {"dense": "DISABLED", "sparse": "DISABLED", "reranker": "DISABLED"}),
            BenchmarkProfileRun("p2_2_dense", semantic, "AVAILABLE", {"dense": "AVAILABLE", "sparse": "DISABLED", "reranker": "DISABLED"}),
        ),
        clock_ms=17,
        evidence_label="A: framework/FakeProvider",
    )

    profiles = report["framework"]["profiles"]
    assert [profile["name"] for profile in profiles] == ["p2_2", "p2_2_dense"]
    dense_case = next(case for case in profiles[1]["cases"] if case["case_id"] == "lexical-distractor")
    assert dense_case["hit_ids"] == ["mem_lexical_target"]
    assert dense_case["false_recall_ids"] == ["mem_lexical_distractor"]
    assert profiles[1]["aggregate"]["forbidden_cross_scope_count"] == 1
    assert profiles[1]["aggregate"]["mandatory_miss_count"] == 1
    assert "lexical-distractor" in profiles[1]["no_gain_or_regression_case_ids"]


def test_output_is_canonical_fixed_clock_and_does_not_expose_input_text_or_local_paths(tmp_path: Path) -> None:
    """Would fail if benchmark output became nondeterministic or exposed fixture-only inputs."""
    from codex_memory.retrieval.benchmark import BenchmarkProfileRun, evaluate_profiles, load_benchmark_fixture, serialize_benchmark_report

    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({
        "case_id": "safe", "query": "TOP_SECRET_PROMPT", "project_scope_key": r"C:\\Users\\local-only",
        "workspace_scope_keys": [], "relevant": {"record_safe": 3}, "mandatory": [],
        "forbidden": [], "excluded": [], "notes": "TOP_SECRET_NOTES",
    }) + "\n", encoding="utf-8")
    cases = load_benchmark_fixture(fixture)
    run = BenchmarkProfileRun("p2_2", {"safe": ("record_safe",)}, "AVAILABLE", {"dense": "DISABLED", "sparse": "DISABLED", "reranker": "DISABLED"})

    first = serialize_benchmark_report(evaluate_profiles(cases, (run,), clock_ms=123, evidence_label="A: framework/FakeProvider"))
    second = serialize_benchmark_report(evaluate_profiles(cases, (run,), clock_ms=123, evidence_label="A: framework/FakeProvider"))

    assert first == second
    assert "TOP_SECRET" not in first
    assert "C:\\\\Users" not in first
    assert json.loads(first)["framework"]["evidence_label"] == "A: framework/FakeProvider"


def test_live_unavailable_is_structurally_separate_and_never_filled_with_fake_results() -> None:
    """Would fail if framework metrics could masquerade as target-machine evidence."""
    from codex_memory.retrieval.benchmark import BenchmarkProfileRun, build_benchmark_envelope, evaluate_profiles

    framework = evaluate_profiles((), (BenchmarkProfileRun("p2_2", {}, "AVAILABLE", {"dense": "DISABLED", "sparse": "DISABLED", "reranker": "DISABLED"}),), clock_ms=0, evidence_label="A: framework/FakeProvider")
    envelope = build_benchmark_envelope(framework=framework, live_status="BLOCKED")

    assert envelope["framework"]["evidence_label"] == "A: framework/FakeProvider"
    assert envelope["live"] == {"status": "BLOCKED", "profiles": [], "components": {"dense": "BLOCKED", "sparse": "UNSUPPORTED", "reranker": "UNSUPPORTED"}}


def test_profile_order_is_canonical_even_when_input_order_is_not() -> None:
    """Would fail if metric output depended on invocation order rather than profile identity."""
    from codex_memory.retrieval.benchmark import BenchmarkProfileRun, evaluate_profiles

    disabled = {"dense": "DISABLED", "sparse": "DISABLED", "reranker": "DISABLED"}
    report = evaluate_profiles(
        (),
        (BenchmarkProfileRun("p2_2_dense", {}, "AVAILABLE", disabled), BenchmarkProfileRun("p2_2", {}, "AVAILABLE", disabled)),
        clock_ms=0,
        evidence_label="A: framework/FakeProvider",
    )

    assert [profile["name"] for profile in report["framework"]["profiles"]] == ["p2_2", "p2_2_dense"]


def test_fake_script_executes_the_pipeline_instead_of_using_relevance_as_final_ranks() -> None:
    """Would fail if the command reintroduced fabricated benchmark selections."""
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark_retrieval_p23.py", "--provider", "fake", "--fixture", str(_FIXTURE)],
        check=False, capture_output=True, text=True, encoding="utf-8",
    )

    assert completed.returncode == 0
    document = json.loads(completed.stdout)
    assert document["framework"]["pipeline_provider_call_count"] > 0
    input_audit = document["framework"]["pipeline_provider_input_audit"]
    assert input_audit["dense_query"]["call_count"] > 0
    assert input_audit["dense_query"]["distinct_input_sha256_count"] > 1
    assert document["framework"]["profiles"][0]["status"] == "AVAILABLE"
    first_case = document["framework"]["profiles"][0]["cases"][0]
    assert first_case["retrieval_status"] == "current"
    assert first_case["candidate_trace_audit"]["selected_ids"]
    assert len(first_case["candidate_trace_audit"]["sha256"]) == 64
    audit = document["framework"]["mandatory_slot_audit"]
    assert audit["p2_2_dense_reranker"]["mandatory-constitution"] == {
        "certification_action": "promote",
        "certification_policy_code": "linked_duplicate",
        "mandatory": True,
        "raw_slot": 0,
        "reranked_slot": 0,
    }


def test_pipeline_certifies_mandatory_constitution_and_preserves_its_rerank_slot(
    tmp_path: Path,
    provider_phase_trace,
) -> None:
    """The fixed fixture must exercise the real certification and protected-slot paths."""
    from codex_memory.retrieval.benchmark import load_benchmark_fixture
    from codex_memory.retrieval.benchmark_pipeline import run_benchmark_pipeline
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.provider import ModelComponent
    from retrieval_provider_fakes import FakeProvider

    def config(provider):
        return SemanticConfig(
            1, "local_http", "http://127.0.0.1:1", "audited-local-v1",
            1_000, 10_000, 100, 100, True, True, True, 16, 16, 8, 16, 1, 0, 2, 30_000,
            provider.expected_provider_fingerprint,
            provider.component_fingerprint(ModelComponent.DENSE),
            provider.component_fingerprint(ModelComponent.SPARSE),
            provider.component_fingerprint(ModelComponent.RERANKER),
        )

    result = run_benchmark_pipeline(
        load_benchmark_fixture(_FIXTURE),
        _FIXTURE.with_name("retrieval_p23_canonical_fixture.json"),
        tmp_path,
        provider_factory=FakeProvider,
        config_factory=config,
    )

    for profile, audit in result.mandatory_slot_audit.items():
        entry = audit["mandatory-constitution"]
        assert entry["certification_action"] == "promote", profile
        assert entry["certification_policy_code"] == "linked_duplicate", profile
        assert entry["mandatory"] is True, profile
        assert entry["raw_slot"] == entry["reranked_slot"], profile


@pytest.mark.parametrize(
    "fixture_state,config_state",
    (
        ("valid", "omitted"),
        ("valid", "missing"),
        ("valid", "invalid"),
        ("omitted", "invalid"),
        ("missing", "invalid"),
        ("invalid", "invalid"),
    ),
)
def test_configured_script_blocks_missing_or_invalid_fixture_without_fake_fallback(
    tmp_path: Path,
    fixture_state: str,
    config_state: str,
) -> None:
    """Would fail if unusable configured input were mislabeled or replaced by FakeProvider."""
    fixture = _FIXTURE
    if fixture_state == "missing":
        fixture = tmp_path / "missing.jsonl"
    elif fixture_state == "invalid":
        fixture = tmp_path / "invalid.jsonl"
        fixture.write_text("{}\n", encoding="utf-8")

    command = [
        sys.executable,
        "scripts/benchmark_retrieval_p23.py",
        "--provider",
        "configured",
    ]
    if fixture_state != "omitted":
        command.extend(("--fixture", str(fixture)))
    if config_state == "missing":
        command.extend(("--config-file", str(tmp_path / "missing.toml")))
    elif config_state == "invalid":
        config = tmp_path / "invalid.toml"
        config.write_text("[retrieval]\nmode = 'off'\n", encoding="utf-8")
        command.extend(("--config-file", str(config)))

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {"live": {"status": "BLOCKED"}}
    assert completed.stderr == ""


def test_configured_fresh_child_uses_installed_cli_and_reconstructed_safe_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """R4 evidence must reopen through the real CLI without copying source/config comments."""
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.local_http_provider import (
        AUDITED_DENSE_FINGERPRINT,
        AUDITED_PROVIDER_FINGERPRINT,
    )
    from codex_memory.retrieval import benchmark_pipeline

    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:11434", "audited-local-v1",
        14_000, 3_600_000, 500, 4_000, True, False, False,
        16, 16, 8, 16, 1, 0, 2, 30_000,
        AUDITED_PROVIDER_FINGERPRINT, AUDITED_DENSE_FINGERPRINT, None, None,
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def completed(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((list(command), dict(kwargs)))
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "", "")
        stdout = json.dumps({
            "status": "current",
            "generation": {"generation_id": _GENERATION_ID},
            "degraded_reasons": [],
            "trace": [{"source_channels": ["dense"]}],
        })
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(benchmark_pipeline.subprocess, "run", completed)
    evidence = benchmark_pipeline.run_fresh_child_query(
        tmp_path, config, "synthetic fixture query"
    )

    assert calls[0][0][:3] == ["git", "-C", str(tmp_path)]
    assert calls[1][0][:3] == [sys.executable, "-m", "codex_memory.cli"]
    assert calls[1][1]["cwd"] == tmp_path
    assert "env" not in calls[1][1]
    assert SemanticConfig.from_dict(
        tomllib.loads(
            (tmp_path / ".codex" / "memory.toml").read_text(encoding="utf-8")
        )["retrieval"]["semantic"]
    ) == config
    assert evidence["return_code"] == 0
    assert evidence["status"] == "current"
    assert evidence["generation_id"] == _GENERATION_ID
    assert evidence["dense_trace_count"] == 1
    assert evidence["stderr_empty"] is True


def test_fresh_child_dense_trace_block_keeps_sanitized_child_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A blocked child query must retain enough output to audit fallback state."""
    from codex_memory.retrieval import benchmark_pipeline
    from codex_memory.retrieval.config import SemanticConfig
    from codex_memory.retrieval.local_http_provider import (
        AUDITED_DENSE_FINGERPRINT,
        AUDITED_PROVIDER_FINGERPRINT,
    )

    config = SemanticConfig(
        1, "local_http", "http://127.0.0.1:11434", "audited-local-v1",
        14_000, 3_600_000, 500, 4_000, True, False, False,
        16, 16, 8, 16, 1, 0, 2, 30_000,
        AUDITED_PROVIDER_FINGERPRINT, AUDITED_DENSE_FINGERPRINT, None, None,
    )

    def completed(command, **kwargs):  # type: ignore[no-untyped-def]
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({
                "status": "current",
                "generation": {"generation_id": _GENERATION_ID},
                "degraded_reasons": ["semantic_dense_unavailable"],
                "trace": [{"source_channels": ["exact"]}],
            }),
            "",
        )

    monkeypatch.setattr(benchmark_pipeline.subprocess, "run", completed)
    with pytest.raises(benchmark_pipeline.PipelineBlocked) as caught:
        benchmark_pipeline.run_fresh_child_query(
            tmp_path, config, "synthetic fixture query"
        )

    error = caught.value
    assert error.status == "fresh_child_dense_trace"
    assert error.diagnostics["return_code"] == 0
    assert error.diagnostics["stderr_empty"] is True
    assert error.diagnostics["status"] == "current"
    assert error.diagnostics["generation_id"] == _GENERATION_ID
    assert error.diagnostics["degraded_reasons"] == ["semantic_dense_unavailable"]
    assert error.diagnostics["trace_count"] == 1
    assert error.diagnostics["dense_trace_count"] == 0
    assert isinstance(error.diagnostics["stdout_sha256"], str)
    assert isinstance(error.diagnostics["wall_clock_ms"], int)


def test_fresh_child_diagnostics_bounds_and_redacts_untrusted_child_values() -> None:
    """A malicious child JSON document cannot become unbounded diagnostic output."""
    from codex_memory.retrieval.benchmark_pipeline import _fresh_child_diagnostics

    secret = "SENSITIVE_CHILD_VALUE"
    diagnostics = _fresh_child_diagnostics(
        return_code=7,
        stdout=secret * 2_000,
        stderr=secret * 2_000,
        wall_clock_ms=1,
        document={
            "status": secret,
            "generation": {"generation_id": secret * 20},
            "degraded_reasons": [secret] * 1_000,
            "trace": [{"source_channels": ["dense"]}] * 1_000,
        },
    )

    assert diagnostics["status"] == "unknown"
    assert diagnostics["generation_id"] is None
    assert len(diagnostics["degraded_reasons"]) <= 16
    assert diagnostics["trace_count"] == 256
    assert diagnostics["trace_truncated"] is True
    assert isinstance(diagnostics["stderr_sha256"], str)
    assert secret not in repr(diagnostics)


@pytest.mark.parametrize("failure", ("cache_corrupt", "runtime_path"))
def test_configured_script_preserves_pipeline_block_when_secondary_audits_fail(
    tmp_path: Path,
    monkeypatch,
    capsys,
    failure: str,
) -> None:
    """A failed evidence audit cannot replace the primary configured-run stage."""
    from codex_memory.paths import RuntimePathError
    from codex_memory.retrieval.benchmark_pipeline import PipelineBlocked
    from codex_memory.retrieval.cache_database import CacheCorruptError

    module = _benchmark_script_module()
    audit_error = CacheCorruptError("cache corrupt") if failure == "cache_corrupt" else RuntimePathError("unsafe path")
    prepared = SimpleNamespace(
        paths=tmp_path / "retrieval-paths",
        runtime=SimpleNamespace(db_path=tmp_path / "memory.sqlite3"),
        repo=tmp_path,
    )

    monkeypatch.setattr(module, "_semantic_from_fixture", lambda _path: object())
    monkeypatch.setattr(module, "provider_from_config", lambda _config: object())
    monkeypatch.setattr(module, "prepare_benchmark_pipeline", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(module, "_sha256", lambda _path: "0" * 64)

    def primary_failure(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise PipelineBlocked("fresh_child_output", diagnostics={"primary": "retained"})

    def failed_audit(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise audit_error

    monkeypatch.setattr(module, "run_prepared_benchmark_pipeline", primary_failure)
    monkeypatch.setattr(module, "read_provider_attempt_audit", failed_audit)
    monkeypatch.setattr(module, "_artifact_evidence", failed_audit)

    assert module.main([
        "--provider", "configured", "--fixture", str(_FIXTURE),
        "--config-file", str(tmp_path / "fixture.toml"),
    ]) == 2
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "live": {
            "status": "BLOCKED",
            "stage": "fresh_child_output",
            "diagnostics": {
                "primary": "retained",
                "provider_attempt_audit": "unavailable",
                "artifacts": "unavailable",
            },
        },
    }


def test_evaluator_rejects_selected_candidates_not_backed_by_actual_trace() -> None:
    """Would fail if evaluator metrics could bypass the retrieval trace audit."""
    from codex_memory.retrieval.benchmark import (
        BenchmarkProfileRun,
        BenchmarkValidationError,
        evaluate_profiles,
        load_benchmark_fixture,
    )

    case = load_benchmark_fixture(_FIXTURE)[0]
    run = BenchmarkProfileRun(
        "p2_2",
        {case.case_id: ("mem_exact_record",)},
        "AVAILABLE",
        {"dense": "DISABLED", "sparse": "DISABLED", "reranker": "DISABLED"},
        query_status={case.case_id: "current"},
        candidate_trace={
            case.case_id: (
                {
                    "candidate_id": "different_record",
                    "source_channels": ["exact"],
                    "contributions": [],
                    "fused_rank": 1,
                    "selected": True,
                    "reason_codes": [],
                },
            )
        },
    )

    with pytest.raises(BenchmarkValidationError, match="selected candidates do not match retrieval trace"):
        evaluate_profiles(
            (case,),
            (run,),
            clock_ms=0,
            evidence_label="A: framework/FakeProvider",
        )


@pytest.mark.parametrize("bad", ["{}", '{"case_id":"x"}\n', '{"case_id":"x","query":"q","project_scope_key":"p","workspace_scope_keys":[],"relevant":{},"mandatory":[],"forbidden":[],"excluded":[],"notes":"n","extra":true}\n'])
def test_fixture_rejects_missing_or_unknown_fields(tmp_path: Path, bad: str) -> None:
    """Would fail if benchmark fixture input had a permissive silently-changing schema."""
    from codex_memory.retrieval.benchmark import BenchmarkValidationError, load_benchmark_fixture

    path = tmp_path / "bad.jsonl"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(BenchmarkValidationError):
        load_benchmark_fixture(path)
