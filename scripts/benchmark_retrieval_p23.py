"""Run the sanitized, deterministic P2.3 framework benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time
import tomllib

from codex_memory.retrieval.benchmark import (
    BenchmarkProfileRun,
    BenchmarkValidationError,
    build_benchmark_envelope,
    evaluate_profiles,
    fixture_sha256,
    load_benchmark_fixture,
    serialize_benchmark_report,
)
from codex_memory.retrieval.benchmark_pipeline import (
    PipelineBlocked,
    prepare_benchmark_pipeline,
    read_provider_attempt_audit,
    run_benchmark_pipeline,
    run_fresh_child_query,
    run_prepared_benchmark_pipeline,
)
from codex_memory.retrieval.config import SemanticConfig, provider_from_config
from codex_memory.retrieval.local_http_provider import audited_profile_metadata
from codex_memory.retrieval.provider import DisabledProvider, ModelComponent
from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths


_MAX_CONFIG_BYTES = 65_536


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--provider", choices=("fake", "configured"), required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--clock-ms", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.fixture is None:
            raise BenchmarkValidationError("fixture is required")
        cases = load_benchmark_fixture(args.fixture)
        canonical_fixture = args.fixture.with_name("retrieval_p23_canonical_fixture.json")
        if args.provider == "fake":
            factory = _fake_provider_factory()
            with TemporaryDirectory(prefix="codex-memory-benchmark-") as temporary:
                pipeline = run_benchmark_pipeline(
                    cases, canonical_fixture, Path(temporary),
                    provider_factory=factory, config_factory=_fake_config,
                )
            framework = evaluate_profiles(cases, pipeline.runs, clock_ms=args.clock_ms, evidence_label="A: framework/FakeProvider")
            framework_document = framework["framework"]
            assert isinstance(framework_document, dict)
            framework_document["pipeline_provider_call_count"] = pipeline.provider_call_count
            framework_document["pipeline_provider_input_audit"] = pipeline.provider_input_audit
            framework_document["mandatory_slot_audit"] = pipeline.mandatory_slot_audit
            envelope = build_benchmark_envelope(framework=framework, live_status="BLOCKED")
        else:
            if args.config_file is None:
                raise PipelineBlocked()
            config = _semantic_from_fixture(args.config_file)
            provider = provider_from_config(config)
            if isinstance(provider, DisabledProvider):
                raise PipelineBlocked()
            total_started = time.monotonic_ns()
            with TemporaryDirectory(prefix="codex-memory-benchmark-") as temporary:
                prepared = None
                canonical_before_query = None
                prepare_started = time.monotonic_ns()
                try:
                    prepared = prepare_benchmark_pipeline(
                        cases, canonical_fixture, Path(temporary),
                        provider_factory=lambda: provider, config_factory=lambda _provider: config,
                    )
                    prepare_ms = _elapsed_ms(prepare_started)
                    canonical_before_query = _sha256(prepared.runtime.db_path)
                    profiles_started = time.monotonic_ns()
                    pipeline = run_prepared_benchmark_pipeline(prepared, cases)
                    profiles_ms = _elapsed_ms(profiles_started)
                    child = run_fresh_child_query(prepared.repo, config, cases[0].query)
                    attempt_audit = read_provider_attempt_audit(prepared.paths)
                    artifact_evidence = _artifact_evidence(
                        prepared, canonical_before_query
                    )
                    total_ms = _elapsed_ms(total_started)
                except PipelineBlocked as error:
                    # Preserve bounded audit data before TemporaryDirectory removes
                    # the isolated fixture.  This is diagnostic evidence only; it
                    # never converts an incomplete child run into a pass.
                    if prepared is not None:
                        try:
                            error.diagnostics["provider_attempt_audit"] = dict(
                                read_provider_attempt_audit(prepared.paths)
                            )
                        except Exception:
                            # The primary PipelineBlocked is authoritative;
                            # best-effort diagnostics must never mask it.
                            error.diagnostics.setdefault(
                                "provider_attempt_audit", "unavailable"
                            )
                        if canonical_before_query is not None:
                            try:
                                error.diagnostics["artifacts"] = _artifact_evidence(
                                    prepared, canonical_before_query
                                )
                            except Exception:
                                error.diagnostics.setdefault("artifacts", "unavailable")
                    raise
            live = evaluate_profiles(cases, pipeline.runs, clock_ms=args.clock_ms, evidence_label="B: target-machine configured provider")
            live_document = live["framework"]
            assert isinstance(live_document, dict)
            profiles = live_document["profiles"]
            assert isinstance(profiles, list)
            component_status = _live_component_status(profiles, config)
            enabled_statuses = [
                value for component, value in component_status.items()
                if component == "dense" and config.dense
                or component == "sparse" and config.sparse
                or component == "reranker" and config.reranker
            ]
            live_status = (
                "AVAILABLE"
                if enabled_statuses and all(value == "AVAILABLE" for value in enabled_statuses)
                else "DEGRADED"
            )
            capability_audit = attempt_audit["capabilities"]
            if (
                child["status"] != "current"
                or child["dense_trace_count"] < 1
                or capability_audit["success_count"] < 1
                or capability_audit["failure_count"] != 0
                or capability_audit["unresolved_count"] != 0
            ):
                raise PipelineBlocked("UNAVAILABLE")
            envelope = {
                "schema_version": 1,
                "framework": {
                    "evidence_label": "NOT_RUN",
                    "pipeline_provider_call_count": sum(
                        value["admission_count"] for value in attempt_audit.values()
                    ),
                    "pipeline_provider_input_audit": pipeline.provider_input_audit,
                    "profiles": [],
                },
                "live": {
                    "status": live_status,
                    "profiles": profiles,
                    "components": component_status,
                    "provider_attempt_audit": attempt_audit,
                    "fresh_child": child,
                    "identity": {
                        "validation": "controller_capabilities_success",
                        **audited_profile_metadata(),
                    },
                    "stage_timings_ms": {
                        "prepare_build_activate_catch_up": prepare_ms,
                        "profile_queries": profiles_ms,
                        "fresh_child_reopen_query": child["wall_clock_ms"],
                        "total": total_ms,
                    },
                    "artifacts": artifact_evidence,
                },
            }
        envelope["fixture_sha256"] = fixture_sha256(args.fixture)
        print(serialize_benchmark_report(envelope), end="")
    except PipelineBlocked as error:
        blocked = {"live": {"status": "BLOCKED"}}
        if error.status != "BLOCKED":
            blocked["live"]["stage"] = error.status
        if error.diagnostics:
            blocked["live"]["diagnostics"] = error.diagnostics
        print(json.dumps(blocked, ensure_ascii=False, separators=(",", ":")))
        return 2
    except (BenchmarkValidationError, OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError):
        if args.provider == "configured":
            print('{"live":{"status":"BLOCKED"}}')
        else:
            print('{"benchmark":{"status":"INVALID"}}')
        return 2
    return 0


def _live_component_status(profiles: list[object], config: SemanticConfig) -> dict[str, str]:
    """Summarize enabled components from actual query-time profile traces."""
    result: dict[str, str] = {}
    for component, configured in (
        ("dense", config.dense),
        ("sparse", config.sparse),
        ("reranker", config.reranker),
    ):
        if not configured:
            result[component] = "NOT_AVAILABLE" if component == "dense" else "UNSUPPORTED"
            continue
        observed = [
            profile["component_status"][component]
            for profile in profiles
            if isinstance(profile, dict)
            and isinstance(profile.get("component_status"), dict)
            and profile["component_status"].get(component) != "DISABLED"
        ]
        if observed and all(value == "AVAILABLE" for value in observed):
            result[component] = "AVAILABLE"
        elif any(value == "DEGRADED" for value in observed):
            result[component] = "DEGRADED"
        elif observed and all(value == "UNSUPPORTED" for value in observed):
            result[component] = "UNSUPPORTED"
        else:
            result[component] = "NOT_AVAILABLE"
    return result


def _fake_provider_factory():
    """Load the test-only provider only at the developer script boundary."""
    tests = Path(__file__).resolve().parents[1] / "tests"
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    from retrieval_provider_fakes import FakeProvider

    return FakeProvider


def _fake_config(provider) -> SemanticConfig:
    try:
        provider_fingerprint = provider.expected_provider_fingerprint
        dense_fingerprint = provider.component_fingerprint(ModelComponent.DENSE)
        sparse_fingerprint = provider.component_fingerprint(ModelComponent.SPARSE)
        reranker_fingerprint = provider.component_fingerprint(ModelComponent.RERANKER)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PipelineBlocked() from exc
    return SemanticConfig(
        1, "local_http", "http://127.0.0.1:1", "audited-local-v1",
        1_000, 10_000, 100, 100, True, True, True, 16, 16, 8, 16, 1, 0, 2, 30_000,
        provider_fingerprint, dense_fingerprint, sparse_fingerprint, reranker_fingerprint,
    )


def _semantic_from_fixture(path: Path) -> SemanticConfig:
    raw = path.read_bytes()
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("fixture configuration is too large")
    document = tomllib.loads(raw.decode("utf-8"))
    retrieval = document.get("retrieval") if isinstance(document, dict) else None
    if not isinstance(retrieval, dict) or retrieval.get("mode") != "shadow":
        raise ValueError("fixture configuration is invalid")
    return SemanticConfig.from_dict(retrieval.get("semantic"))


def _artifact_evidence(prepared, canonical_before_query: str) -> dict[str, object]:
    canonical_after_query = _sha256(prepared.runtime.db_path)
    connection = sqlite3.connect(prepared.paths.database_path)
    try:
        row = connection.execute(
            "SELECT active_semantic_generation_id FROM retrieval_semantic_state "
            "WHERE singleton_id = 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None or not isinstance(row[0], str):
        raise PipelineBlocked("UNAVAILABLE")
    generation = SemanticGenerationPaths.for_generation(
        prepared.paths.generations_dir, row[0]
    )
    return {
        "canonical_before_query_sha256": canonical_before_query,
        "canonical_after_query_sha256": canonical_after_query,
        "canonical_unchanged_by_queries": canonical_before_query == canonical_after_query,
        "retrieval_database_sha256": _sha256(prepared.paths.database_path),
        "semantic_database_sha256": _sha256(generation.database_path),
        "semantic_manifest_sha256": _sha256(generation.manifest_path),
        "active_semantic_generation_id": row[0],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.monotonic_ns() - started_ns) // 1_000_000)


if __name__ == "__main__":
    raise SystemExit(main())
