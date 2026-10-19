"""Fixed P2.5 offline observations through the installed, lawful runtime.

Fixture setup is reusable by controlled Hook replays. Only ``query_case`` runs
retrieval; it never injects ranks or substitutes a semantic provider. Artifacts
are private runtime evidence, and failed cases remain in every denominator.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from codex_memory import database, structured_ledger
from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import capture_candidate
from codex_memory.event_store import EventRecord, EventStore
from codex_memory.hook_input import parse_hook_payload
from codex_memory.injection_manifest import InjectionAuditError
from codex_memory.injection_store import InjectionPaths, InjectionStore
from codex_memory.ledger_common import canonical_json
from codex_memory.memory_context import build_memory_packet
from codex_memory.memory_evidence import event_uri
from codex_memory.memory_review import review_candidate
from codex_memory.paths import RuntimePaths

from .assembler import PacketBudgetError, build_phase2_packet
from .benchmark import (BenchmarkCase, BenchmarkValidationError, _aggregate,
                        _case_metrics, _percentile, fixture_sha256, load_benchmark_fixture)
from .benchmark_pipeline import (_enabled_semantic_components_ready,
                                 _initialize_fixture_repository, read_provider_attempt_audit)
from .cache_database import connect_cache_database, verify_cache_database
from .config import SemanticConfig, load_retrieval_config, provider_from_config
from .contracts import RetrievalMode, RetrievalRequest, RetrievalScope
from .deadline import DeadlineBudget
from .injection_audit import implementation_fingerprint, prepare_shadow_manifest
from .paths import RetrievalPaths
from .semantic_paths import SemanticGenerationPaths
from .semantic_store import open_verified_semantic_snapshot
from .service import RetrievalService
from .shadow import _bounded_call, _shadow_query


@dataclass(frozen=True, slots=True)
class Workload:
    cases: tuple[BenchmarkCase, ...]
    records: tuple[dict[str, object], ...]
    scenarios: Mapping[str, dict[str, object]]
    query_fixture_sha256: str
    canonical_fixture_sha256: str
    session_id: str
    turn_id: str
    context_id: str


@dataclass(frozen=True, slots=True)
class PreparedCase:
    case: BenchmarkCase
    scenario: Mapping[str, object]
    repo: Path
    workspace: Path
    runtime: RuntimePaths
    paths: RetrievalPaths
    source: EventRecord
    request: RetrievalRequest
    label_to_record_id: Mapping[str, str]
    hook_payload: Mapping[str, object]


def load_workload(fixture: Path) -> Workload:
    """Read the committed labels and companion scenarios without evaluating."""
    cases = load_benchmark_fixture(fixture)
    canonical_path = fixture.with_name("retrieval_p25_canonical_fixture.json")
    document = json.loads(canonical_path.read_text(encoding="utf-8"))
    if document.get("format") != "codex-memory.p25-canonical-fixture.v1":
        raise BenchmarkValidationError("unsupported P2.5 canonical fixture")
    scenarios = document["scenarios"]
    records = document["records"]
    if len(cases) != 24 or set(scenarios) != {case.case_id for case in cases}:
        raise BenchmarkValidationError("P2.5 requires all 24 fixed cases")
    labels = {record["label"] for record in records}
    if len(labels) != len(records):
        raise BenchmarkValidationError("canonical labels must be unique")
    groups = Counter()
    for case in cases:
        scenario = scenarios[case.case_id]
        if not re.fullmatch(r"[a-z]+_[0-9]{2}", case.case_id):
            raise BenchmarkValidationError("case ID cannot name an artifact directory")
        if set(scenario["record_labels"]) - labels or len(set(scenario["record_labels"])) != len(scenario["record_labels"]):
            raise BenchmarkValidationError("scenario record labels are invalid")
        if set(case.relevant) - set(scenario["record_labels"]):
            raise BenchmarkValidationError("relevant label is absent from scenario")
        if type(scenario["max_chars"]) is not int or not 1 <= scenario["max_chars"] <= 4000:
            raise BenchmarkValidationError("scenario character budget is invalid")
        if scenario["expected_outcome"] not in {"complete", "safe_budget_fallback"}:
            raise BenchmarkValidationError("scenario outcome is unsupported")
        groups[scenario["group"], scenario["split"]] += 1
    expected = {(group, split): 2 for group in ("exact", "symbol", "semantic", "state", "scope", "provenance")
                for split in ("development", "held-out")}
    if groups != expected:
        raise BenchmarkValidationError("fixed group and split counts changed")
    return Workload(cases, tuple(records), scenarios, fixture_sha256(fixture),
                    fixture_sha256(canonical_path), document["session_id"],
                    document["turn_id"], document["context_id"])


def _prompt_payload(repo: Path, *, session_id: str, turn_id: str, prompt: str) -> dict[str, object]:
    return {"hook_event_name": "UserPromptSubmit", "session_id": session_id,
            "turn_id": turn_id, "cwd": str(repo), "prompt": prompt}


def prepare_case(case: BenchmarkCase, scenario: Mapping[str, object],
                 records: Sequence[Mapping[str, object]], repo: Path, *,
                 session_id: str = "p25_session", turn_id: str = "p25_turn",
                 context_id: str = "startup") -> PreparedCase:
    """Materialize only a fictional isolated repository; no query/model calls.

The source event is appended before projection and returned for idempotent
installed-Hook replay. Certification and long provenance use actual prompt
events, candidate capture, review and promotion under the canonical policies.
"""
    repo.mkdir(parents=True, exist_ok=False)
    repo = _initialize_fixture_repository(repo)
    workspace = repo / "workspace"
    workspace.mkdir()
    runtime = RuntimePaths.for_repo(repo)
    runtime.ensure()
    paths = RetrievalPaths.for_repo(repo)
    blobs = BlobStore(runtime.blobs_dir)
    connection = database.connect_database(runtime.db_path)
    database.migrate_database(connection)
    known = {item["label"]: item for item in records}
    selected_records = [known[label] for label in scenario["record_labels"]]
    identifiers: dict[str, str] = {}
    try:
        events = EventStore(connection, blobs)
        for item in selected_records:
            label = str(item["label"])
            scope_kind, scope_key = (("project", str(repo)) if item["scope"] == "project" else
                                     ("workspace", str(workspace)) if item["scope"] == "workspace" else
                                     ("project", "p25-foreign-project"))
            provenance = item["provenance"]
            if provenance in {"temporary", "promoted", "certified"}:
                origin_session = str(item.get("session_id", session_id))
                evidence = []
                for index in range(int(item.get("evidence_count", 1))):
                    evidence_event = events.append(parse_hook_payload(_prompt_payload(
                        repo, session_id=origin_session, turn_id=f"origin_{label}_{index}",
                        prompt=f"Explicit fictional instruction {index}: {item['statement']}")))
                    evidence.append(event_uri(origin_session, evidence_event.event_id))
                captured = capture_candidate(
                    connection, blobs, project_scope_key=str(repo), record_type=str(item["record_type"]),
                    scope_kind=scope_kind, scope_key=scope_key, topic_key=str(item["topic_key"]),
                    statement=str(item["statement"]), session_id=origin_session, turn_id=f"capture_{label}",
                    context_id=context_id, evidence_uris=tuple(evidence), idempotency_key=f"p25-capture-{label}")
                if provenance == "temporary":
                    identifiers[label] = captured.candidate_id
                else:
                    reviewed = review_candidate(connection, blobs, candidate_id=captured.candidate_id,
                                                outcome="promote", reason="Explicit fictional prompt evidence reviewed",
                                                idempotency_key=f"p25-promote-{label}")
                    if reviewed.action != "promote" or not reviewed.permanent_record_id:
                        raise BenchmarkValidationError("fixture lawful promotion did not complete")
                    identifiers[label] = reviewed.permanent_record_id
            elif provenance == "direct":
                proposed = structured_ledger.propose_record(
                    connection, record_type=str(item["record_type"]), scope_kind=scope_kind,
                    scope_key=scope_key, topic_key=str(item["topic_key"]), statement=str(item["statement"]),
                    idempotency_key=f"p25-propose-{label}")
                structured_ledger.apply_action(connection, record_id=proposed.record_id,
                                              action="activate", idempotency_key=f"p25-activate-{label}")
                identifiers[label] = proposed.record_id
            else:
                raise BenchmarkValidationError("unsupported fixture provenance")
        for item in selected_records:
            state = item["lifecycle"]
            if state == "active":
                continue
            action = {"invalidated": "invalidate", "closed": "close", "superseded": "supersede"}.get(state)
            if action is None:
                raise BenchmarkValidationError("unsupported fixture lifecycle")
            replacement = identifiers[str(item["superseded_by"])] if action == "supersede" else None
            structured_ledger.apply_action(connection, record_id=identifiers[str(item["label"])],
                                          action=action, replacement_record_id=replacement,
                                          idempotency_key=f"p25-lifecycle-{item['label']}")
        payload = _prompt_payload(repo, session_id=session_id, turn_id=turn_id, prompt=case.query)
        source = events.append(parse_hook_payload(payload))
    finally:
        connection.close()
    projection = RetrievalService(paths).rebuild()
    if projection.status.value != "current":
        raise BenchmarkValidationError("fixture projection failed")
    workspace_keys = tuple(str(workspace) if key == "BENCH_WORKSPACE" else key for key in case.workspace_scope_keys)
    if case.project_scope_key != "BENCH_PROJECT":
        raise BenchmarkValidationError("fixture project placeholder is unsupported")
    request = RetrievalRequest(query=case.query, session_id=session_id, turn_id=turn_id,
                               context_id=context_id, prompt_event_uri=event_uri(session_id, source.event_id),
                               scope=RetrievalScope(str(repo), workspace_keys), mode=RetrievalMode.SHADOW,
                               max_chars=int(scenario["max_chars"]))
    return PreparedCase(case, dict(scenario), repo, workspace, runtime, paths, source,
                        request, identifiers, payload)


def installed_entry() -> Path:
    """Require the current environment's actual installed console entry."""
    path = Path(sys.executable).parent / ("codex-memory-hook.exe" if os.name == "nt" else "codex-memory-hook")
    if not path.is_file() or path.is_symlink():
        raise BenchmarkValidationError("installed codex-memory-hook entry is unavailable")
    return path


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _elapsed(start: int) -> int:
    return max(0, (time.monotonic_ns() - start) // 1_000_000)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _check_budget(deadline: DeadlineBudget) -> None:
    if time.monotonic_ns() >= deadline.deadline_ns:
        raise InjectionAuditError("deadline_exhausted")


def parse_legacy_record_ids(packet: str) -> tuple[str, ...]:
    """Read only actual complete JSON record lines emitted by the v1 renderer."""
    identifiers = []
    for line in packet.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise BenchmarkValidationError("legacy packet record is invalid")
        identifiers.append(record["id"])
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkValidationError("legacy packet duplicates a record")
    return tuple(identifiers)


def evaluate_case(case: BenchmarkCase, selected: Sequence[str], *, latency_ms: int | None) -> dict[str, object]:
    """Small adapter reusing the existing rank evaluator with explicit misses."""
    metrics = _case_metrics(case, tuple(selected), 0 if latency_ms is None else latency_ms)
    metrics["latency_ms"] = latency_ms
    mandatory_count = len(case.mandatory)
    metrics.update(mandatory_count=mandatory_count,
                   mandatory_recall=None if not mandatory_count else
                   (mandatory_count - metrics["mandatory_miss_count"]) / mandatory_count)
    return metrics


def _verified_semantic_identity(paths, config, projection_generation_id):
    """Capture actual model identities from a verified active semantic artifact."""
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        connection.execute("BEGIN")
        verify_cache_database(connection)
        state = connection.execute(
            "SELECT active_semantic_generation_id, validation_state FROM retrieval_semantic_state WHERE singleton_id = 1"
        ).fetchone()
        if state is None or state["validation_state"] != "validated":
            raise BenchmarkValidationError("semantic identity is not validated")
        semantic_paths = SemanticGenerationPaths.for_generation(paths.generations_dir, state["active_semantic_generation_id"])
        with open_verified_semantic_snapshot(semantic_paths) as (manifest, _snapshot):
            if manifest.projection_generation_id != projection_generation_id or manifest.provider_fingerprint != config.expected_provider_fingerprint:
                raise BenchmarkValidationError("semantic identity does not bind configured projection/provider")
            for name in ("dense", "sparse"):
                if getattr(config, name) and getattr(manifest, f"{name}_model_fingerprint") != getattr(config, f"expected_{name}_fingerprint"):
                    raise BenchmarkValidationError("semantic model identity differs from configuration")
            return manifest.to_dict()
    finally:
        connection.close()


def _metric_aggregate(metrics):
    """Keep every rank/mandatory case, but never turn unobserved latency into 0."""
    result = _aggregate([{**item, "latency_ms": item["latency_ms"] or 0} for item in metrics],
                        {name: "DISABLED" for name in ("dense", "sparse", "reranker")})
    latencies = sorted(item["latency_ms"] for item in metrics if item["latency_ms"] is not None)
    result.update(latency_p50_ms=_percentile(latencies, 0.50) if latencies else None,
                  latency_p95_ms=_percentile(latencies, 0.95) if latencies else None,
                  latency_observed_count=len(latencies), latency_unavailable_count=len(metrics) - len(latencies))
    return result


def explain_in_child(repo: Path, injection_id: str, entry: Path) -> dict[str, object]:
    """Launch a fresh installed process, preserving exact wire output evidence."""
    if not entry.is_file() or entry.is_symlink():
        raise BenchmarkValidationError("installed entry is unavailable")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    started_at, started = _utc(), time.monotonic_ns()
    try:
        child = subprocess.run([str(entry), "retrieval", "explain", injection_id], cwd=repo,
                               env=environment, capture_output=True, check=False, timeout=10)
        stdout, stderr, return_code = child.stdout, child.stderr, child.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, return_code = exc.stdout or b"", exc.stderr or b"", None
        timed_out = True
    document = None
    try:
        document = json.loads(stdout)
    except (ValueError, UnicodeError):
        pass
    return {"started_at": started_at, "ended_at": _utc(), "wall_clock_ms": _elapsed(started),
            "entry": str(entry), "entry_sha256": fixture_sha256(entry), "return_code": return_code,
            "timeout": timed_out, "stdout_hex": stdout.hex(), "stderr_hex": stderr.hex(),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(), "document": document}


def query_case(prepared: PreparedCase, *, installed_entry: Path, budget_ms: int = 1000,
               profile: str = "deterministic", provider: object | None = None,
               semantic_config: SemanticConfig | None = None) -> dict[str, object]:
    """Observe legacy and one real query/assembly/commit/explain; retain failures."""
    from codex_memory.cli import _serialize_response
    started_at, started = _utc(), time.monotonic_ns()
    calls = {name: 0 for name in ("query", "assembly", "prepare", "commit", "explain")}
    request = prepared.request
    observation: dict[str, object] = {
        "case_id": prepared.case.case_id, "group": prepared.scenario["group"],
        "split": prepared.scenario["split"], "expected_outcome": prepared.scenario["expected_outcome"],
        "input_record_count": len(prepared.label_to_record_id), "max_chars": request.max_chars,
        "query_sha256": _digest(request.query), "source_event_id": prepared.source.event_id,
        "source_event_blob_sha256": prepared.source.payload_blob_sha256,
        "calls": calls, "manifest_id": None, "legacy_selected_labels": [], "phase2_selected_labels": [],
        "legacy_packet_chars": 0, "phase2_packet_chars": 0, "status": "failed", "started_at": started_at,
        "profile": profile, "provider_call_count": None if profile == "configured" else 0,
        "provider_http_request_count": None if profile == "configured" else 0,
        "provider_admission_attempt_count": 0, "provider_trace_count": 0,
        "provider_trace": [], "trace": [],
        "phase2_selected_ids": [], "legacy_selected_ids": [],
    }
    reverse = {identifier: label for label, identifier in prepared.label_to_record_id.items()}
    stage = "implementation_fingerprint"
    legacy_started = phase2_started = None
    attempts_before = None
    segments: dict[str, int] = {}
    effective_budget_ms = budget_ms
    try:
        before = implementation_fingerprint(deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
        observation["implementation_sha256_before"] = before
        stage = "legacy"
        legacy_started = time.monotonic_ns()
        connection = database.connect_database(prepared.runtime.db_path)
        try:
            legacy = build_memory_packet(connection, BlobStore(prepared.runtime.blobs_dir),
                                         session_id=request.session_id, turn_id=request.turn_id,
                                         context_id=request.context_id, scope_key=request.scope.project_scope_key,
                                         workspace_scope_keys=request.scope.workspace_scope_keys,
                                         prompt_event_uri=request.prompt_event_uri, query=request.query,
                                         max_chars=request.max_chars)
        finally:
            connection.close()
        legacy_ids = parse_legacy_record_ids(legacy)
        observation.update(legacy_selected_ids=list(legacy_ids), legacy_packet=legacy,
                           legacy_packet_chars=len(legacy), legacy_selected_labels=[reverse[item] for item in legacy_ids])
        serialized = _serialize_response({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": legacy}})
        observation["legacy_duration_ms"] = _elapsed(legacy_started)
        if profile not in {"deterministic", "configured"}:
            raise BenchmarkValidationError("unsupported query profile")
        if profile == "configured":
            if provider is None or not isinstance(semantic_config, SemanticConfig):
                raise BenchmarkValidationError("configured semantic provider is required")
            stage = "semantic_build"
            attempts_before = read_provider_attempt_audit(prepared.paths)
            build_started = time.monotonic_ns()
            segments["semantic_build_duration_ms"] = build_started
            build = RetrievalService(prepared.paths).rebuild_semantic(provider, semantic_config)
            observation.update(semantic_build_duration_ms=_elapsed(build_started),
                               semantic_build_status=build.status.value)
            stage = "semantic_validation"
            if build.status.value != "current" or not _enabled_semantic_components_ready(prepared.paths, semantic_config):
                raise BenchmarkValidationError("configured semantic components are not validated and ready")
            semantic_identity = _verified_semantic_identity(prepared.paths, semantic_config, build.generation.generation_id)
            observation["semantic_identity_evidence"] = semantic_identity
            effective_budget_ms = min(budget_ms, semantic_config.query_deadline_ms)
        observation["budget_ms"] = effective_budget_ms
        deadline = DeadlineBudget.start(effective_budget_ms, now_ns=time.monotonic_ns)
        phase2_started = time.monotonic_ns()
        query_started_at = _utc()
        stage = "query"
        calls[stage] += 1
        query_started = time.monotonic_ns()
        segments["query_duration_ms"] = query_started
        provider_trace: list[dict[str, str]] = []
        if profile == "configured":
            shadow = _shadow_query(paths=prepared.paths, request=request, provider=provider,
                                   semantic_config=semantic_config, disable_dense=False,
                                   disable_sparse=False, disable_reranker=False, deadline=deadline)
            provider_trace = [dict(item) for item in shadow.provider_trace]
            result = shadow.reranked_result or shadow.result
        else:
            result = RetrievalService(prepared.paths).query(request)
        query_ms = _elapsed(query_started)
        trace = [item.to_dict() for item in result.trace]
        observation.update(query_duration_ms=query_ms, trace=trace, trace_sha256=_digest(trace),
                           provider_trace=provider_trace, provider_trace_count=len(provider_trace),
                           query_status=result.status.value, index_stale=result.index_stale,
                           degraded_reasons=[item.value for item in result.degraded_reasons],
                           query_result=result.to_dict())
        if profile == "configured":
            stage = "semantic_identity_binding"
            current_identity = _bounded_call(lambda: _verified_semantic_identity(
                prepared.paths, semantic_config, result.generation.generation_id), deadline)
            if current_identity != semantic_identity:
                raise BenchmarkValidationError("semantic artifact changed during query")
        stage = "assembly"
        _check_budget(deadline)
        calls[stage] += 1
        assembly_started = time.monotonic_ns()
        segments["assembly_duration_ms"] = assembly_started
        assembled = build_phase2_packet(prepared.paths, request=request, result=result)
        assembly_ms = _elapsed(assembly_started)
        selected = [item.candidate_id for item in assembled.selected]
        observation.update(assembly_duration_ms=assembly_ms, phase2_selected_ids=selected,
                           phase2_selected_labels=[reverse[item] for item in selected],
                           phase2_packet_chars=assembled.packet_chars, phase2_packet=assembled.packet,
                           phase2_packet_sha256=assembled.packet_sha256,
                           omission_counts=dict(assembled.omission_counts),
                           budget_omitted_ids=list(assembled.budget_omitted_ids))
        stage = "prepare"
        _check_budget(deadline)
        preparation_started = time.monotonic_ns()
        segments["prepare_call_duration_ms"] = preparation_started
        disabled = {"configured": "disabled", "observed": "disabled", "validated": "disabled"}
        if profile == "configured":
            assert semantic_config is not None
            configured_models = {
                name: getattr(semantic_config, f"expected_{name}_fingerprint")
                for name in ("dense", "sparse", "reranker") if getattr(semantic_config, name)
            }
            observed_models = {name: semantic_identity[f"{name}_model_fingerprint"]
                               for name in configured_models if semantic_identity[f"{name}_model_fingerprint"] is not None}
            observed_model_fingerprint = _digest(observed_models) if observed_models else "unknown"
            # Reranker identity is not in the build artifact in P2.3. Its query
            # trace is retained, but this boundary cannot certify a model ID.
            model_validation = observed_model_fingerprint if set(observed_models) == set(configured_models) else "unknown"
            identities = {
                "provider": {"configured": semantic_config.expected_provider_fingerprint,
                             "observed": semantic_identity["provider_fingerprint"],
                             "validated": semantic_identity["provider_fingerprint"]},
                "model": {"configured": _digest(configured_models),
                          "observed": observed_model_fingerprint, "validated": model_validation},
                "preprocessing": {"configured": "not_configured",
                                  "observed": semantic_identity["preprocessing_fingerprint"],
                                  "validated": semantic_identity["preprocessing_fingerprint"]},
            }
            configuration = {"profile": profile, "mode": "shadow", "semantic": semantic_config.to_dict(),
                             "budget_ms": effective_budget_ms}
        else:
            identities = {"provider": dict(disabled), "model": dict(disabled), "preprocessing": dict(disabled)}
            configuration = {"profile": profile, "mode": "shadow", "semantic": "disabled", "budget_ms": effective_budget_ms}
        environment = {**identities, "config_sha256": _digest(configuration),
                       "implementation_sha256": before, "legacy_delivery": "unknown",
                       "host_acknowledgement": "unknown"}
        source = {"event_id": prepared.source.event_id, "event_uri": request.prompt_event_uri,
                  "event_blob_sha256": prepared.source.payload_blob_sha256, "session_id": request.session_id,
                  "turn_id": request.turn_id, "context_id": request.context_id, "origin": "offline"}
        calls[stage] += 1
        manifest = prepare_shadow_manifest(
            request=request, result=result, assembled=assembled, source=source, legacy_packet=legacy,
            legacy_response_bytes=serialized.encode("utf-8"), environment=environment,
            timing={"started_at": query_started_at, "prepared_at": _utc(), "query_duration_ms": query_ms,
                    "assembly_duration_ms": assembly_ms, "prepare_duration_ms": _elapsed(preparation_started),
                    "budget_ms": effective_budget_ms}, blobs=BlobStore(prepared.runtime.blobs_dir), deadline=deadline)
        observation.update(prepare_call_duration_ms=_elapsed(preparation_started), config_sha256=_digest(configuration))
        stage = "commit"
        _check_budget(deadline)
        calls[stage] += 1
        commit_started = time.monotonic_ns()
        segments["commit_duration_ms"] = commit_started
        committed = InjectionStore(InjectionPaths.for_repo(prepared.repo), BlobStore(prepared.runtime.blobs_dir)).commit(manifest, deadline=deadline)
        observation.update(manifest_id=committed.injection_id, manifest_sha256=committed.manifest_sha256,
                           commit_duration_ms=_elapsed(commit_started), audit_duration_ms=_elapsed(deadline.started_ns))
        observation["phase2_pipeline_duration_ms"] = _elapsed(phase2_started)
        observation["environment"] = environment
        stage = "explain"
        calls[stage] += 1
        replay = explain_in_child(prepared.repo, committed.injection_id, installed_entry)
        observation["explain"] = replay
        document = replay["document"]
        if replay["return_code"] != 0 or replay["stderr_hex"] or not isinstance(document, dict):
            raise BenchmarkValidationError("installed explain failed")
        if document.get("injection_id") != committed.injection_id or document.get("selected") != [item.to_dict() for item in assembled.selected]:
            raise BenchmarkValidationError("installed explain selection mismatch")
        if document.get("source") != source or document.get("request", {}).get("query_sha256") != observation["query_sha256"]:
            raise BenchmarkValidationError("installed explain source mismatch")
        stage = "implementation_fingerprint_after"
        after = implementation_fingerprint(deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns))
        observation["implementation_sha256_after"] = after
        if after != before:
            raise BenchmarkValidationError("implementation changed during case")
        observation["status"] = "complete"
    except PacketBudgetError as exc:
        observation.update(status="safe_budget_fallback" if prepared.scenario["expected_outcome"] == "safe_budget_fallback" else "failed",
                           failure_stage=stage, error_type=type(exc).__name__, error_reason="packet_budget_exceeded")
    except Exception as exc:
        observation.update(failure_stage=stage, error_type=type(exc).__name__,
                           error_reason=str(getattr(exc, "reason", type(exc).__name__)), error_detail=str(exc))
    observation.update(ended_at=_utc(), wall_clock_ms=_elapsed(started))
    observation.setdefault("legacy_duration_ms", None if legacy_started is None else _elapsed(legacy_started))
    observation.setdefault("phase2_pipeline_duration_ms", None if phase2_started is None else _elapsed(phase2_started))
    for name, segment_started in segments.items():
        observation.setdefault(name, _elapsed(segment_started))
    if profile == "configured":
        try:
            attempts_after = read_provider_attempt_audit(prepared.paths)
            if attempts_before is None:
                raise BenchmarkValidationError("provider attempt baseline unavailable")
            attempts = {operation: {name: value - attempts_before[operation][name]
                                    for name, value in values.items()} for operation, values in attempts_after.items()}
            if any(value < 0 for values in attempts.values() for value in values.values()):
                raise BenchmarkValidationError("provider attempt history changed during case")
            observation.update(provider_attempt_audit=attempts,
                               provider_admission_attempt_count=sum(value["admission_count"] for value in attempts.values()),
                               provider_attempt_observation="durable admission deltas after pipeline return; not HTTP requests")
        except Exception as exc:
            observation.update(provider_admission_attempt_count=None, provider_attempt_audit_error=type(exc).__name__)
            if observation["status"] == "complete":
                observation.update(status="failed", failure_stage="provider_attempt_audit")
    observation["legacy_metrics"] = evaluate_case(
        prepared.case, observation["legacy_selected_labels"], latency_ms=observation["legacy_duration_ms"])
    observation["phase2_metrics"] = evaluate_case(
        prepared.case, observation["phase2_selected_labels"], latency_ms=observation["phase2_pipeline_duration_ms"])
    union = set(observation["legacy_selected_labels"]) | set(observation["phase2_selected_labels"])
    observation["legacy_overlap"] = (len(set(observation["legacy_selected_labels"]) & set(observation["phase2_selected_labels"])) / len(union)) if union else 1.0
    observation["character_utilization"] = int(observation["phase2_packet_chars"]) / request.max_chars
    observation["whole_record_count"] = len(observation["phase2_selected_ids"])
    observation["per_channel_yield"] = dict(Counter(channel for item in observation["trace"]
                                                     if item["candidate_id"] in observation["phase2_selected_ids"]
                                                     for channel in item["source_channels"]))
    return observation


def _create_output(output_dir: Path) -> None:
    if ".." in output_dir.parts:
        raise ValueError("p25-evidence output cannot contain parent traversal")
    absolute = output_dir.absolute()
    if not any(parent.name == "p25-evidence" and parent.parent.name == ".codex-memory" for parent in (absolute, *absolute.parents)):
        raise ValueError("outputs must stay under .codex-memory/p25-evidence")
    for parent in (absolute, *absolute.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("p25-evidence output cannot traverse a redirect")
    output_dir.mkdir(parents=True, exist_ok=False)


def _write_evidence(path: Path, value: object) -> None:
    # Exclusive creation keeps earlier measurements, including failures.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")


def _aggregate_observations(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for prefix in ("legacy", "phase2"):
        metrics = [case[f"{prefix}_metrics"] for case in cases]
        aggregate = _metric_aggregate(metrics)
        mandatory_count = sum(item["mandatory_count"] for item in metrics)
        aggregate["mandatory_count"] = mandatory_count
        aggregate["mandatory_recall"] = ((mandatory_count - aggregate["mandatory_miss_count"]) / mandatory_count) if mandatory_count else None
        feasible = [case[f"{prefix}_metrics"] for case in cases if case["expected_outcome"] == "complete"]
        feasible_count = sum(item["mandatory_count"] for item in feasible)
        aggregate["feasible_mandatory_count"] = feasible_count
        aggregate["feasible_mandatory_recall"] = ((feasible_count - sum(item["mandatory_miss_count"] for item in feasible)) / feasible_count) if feasible_count else None
        symbol = [case[f"{prefix}_metrics"]["recall_at_5"] for case in cases if case["group"] == "symbol"]
        aggregate["exact_symbol_recall_at_5"] = sum(symbol) / len(symbol) if symbol else None
        result[prefix] = aggregate
    total_ms = sum(int(case["wall_clock_ms"]) for case in cases)
    result.update(case_count=len(cases), prepared_complete_count=sum(case["status"] == "complete" for case in cases),
                  safe_fallback_count=sum(case["status"] == "safe_budget_fallback" for case in cases),
                  preparation_success_rate=sum(case["status"] == "complete" for case in cases) / len(cases) if cases else 0,
                  throughput_cases_per_second=len(cases) * 1000 / total_ms if total_ms else None,
                  raw_wall_clock_ms=[case["wall_clock_ms"] for case in cases],
                  normal_case_count=sum(case["expected_outcome"] == "complete" for case in cases),
                  normal_complete_count=sum(case["expected_outcome"] == "complete" and case["status"] == "complete" for case in cases))
    result["mean_character_utilization"] = sum(float(case.get("character_utilization", 0)) for case in cases) / len(cases) if cases else 0
    result["mean_legacy_overlap"] = sum(float(case.get("legacy_overlap", 0)) for case in cases) / len(cases) if cases else 0
    result["whole_record_counts"] = [case.get("whole_record_count", 0) for case in cases]
    result["by_group_and_split"] = {
        f"{group}/{split}": {prefix: _metric_aggregate([case[f"{prefix}_metrics"] for case in cases
                                                       if case["group"] == group and case["split"] == split])
                               for prefix in ("legacy", "phase2")}
        for group, split in sorted({(case["group"], case["split"]) for case in cases})
    }
    return result


def run_benchmark(*, profile: str, fixture: Path, output_dir: Path) -> dict[str, object]:
    """Run the full frozen sequence once; never overwrite a previous run."""
    if profile not in {"deterministic", "configured"}:
        raise ValueError("unsupported benchmark profile")
    _create_output(output_dir)
    started_at, started = _utc(), time.monotonic_ns()
    try:
        workload = load_workload(fixture)
    except Exception as exc:
        _write_evidence(output_dir / "setup-failure.json", {
            "status": "FAIL", "failure_stage": "fixture_load", "error_type": type(exc).__name__,
            "error_detail": str(exc), "started_at": started_at, "ended_at": _utc(), "wall_clock_ms": _elapsed(started)})
        raise
    report: dict[str, object] = {
        "format": "codex-memory.p25-offline-benchmark.v1", "profile": profile,
        "status": "RUNNING", "started_at": started_at, "cases": [], "provider_call_count": 0,
        "provider_http_request_count": 0, "provider_admission_attempt_count": 0, "provider_trace_count": 0,
        "fixture_sha256": workload.query_fixture_sha256, "canonical_fixture_sha256": workload.canonical_fixture_sha256,
        "labels_unchanged": True, "host_acknowledgement": "unknown", "model_answer_improvement": "not_evaluated",
        "token_usage": "unknown", "percentiles": "descriptive_small_sample", "budget_ms": 1000,
    }
    semantic_config: SemanticConfig | None = None
    provider: object | None = None
    if profile == "configured":
        loaded = load_retrieval_config(Path.cwd())
        if not loaded.semantic_enabled or loaded.semantic is None or loaded.config_sha256 is None:
            report.update(status="BLOCKED", reason="explicit_valid_configured_provider_missing",
                          ended_at=_utc(), wall_clock_ms=_elapsed(started))
            _write_evidence(output_dir / "report.json", report)
            return report
        semantic_config = loaded.semantic
        provider = provider_from_config(semantic_config)
        if provider is None:
            report.update(status="BLOCKED", reason="configured_provider_unavailable",
                          ended_at=_utc(), wall_clock_ms=_elapsed(started))
            _write_evidence(output_dir / "report.json", report)
            return report
        report.update(configured_file_sha256=loaded.config_sha256,
                      configured_provider_fingerprint=semantic_config.expected_provider_fingerprint,
                      provider_call_count=None, provider_http_request_count=None)
    entry = installed_entry()
    report.update(installed_entry=str(entry), installed_entry_sha256=fixture_sha256(entry),
                  implementation_sha256_before=implementation_fingerprint(deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns)))
    _write_evidence(output_dir / "run-start.json", report)
    for case in workload.cases:
        case_dir = output_dir / case.case_id
        case_dir.mkdir()
        case_started_at, case_started = _utc(), time.monotonic_ns()
        try:
            prepared = prepare_case(case, workload.scenarios[case.case_id], workload.records, case_dir / "repository",
                                    session_id=workload.session_id, turn_id=workload.turn_id, context_id=workload.context_id)
            observation = query_case(prepared, installed_entry=entry, profile=profile,
                                     provider=provider, semantic_config=semantic_config)
        except Exception as exc:
            scenario = workload.scenarios[case.case_id]
            observation = {"case_id": case.case_id, "group": scenario["group"], "split": scenario["split"],
                           "expected_outcome": scenario["expected_outcome"], "status": "failed", "failure_stage": "fixture",
                           "error_type": type(exc).__name__, "error_detail": str(exc), "wall_clock_ms": _elapsed(case_started),
                           "started_at": case_started_at, "ended_at": _utc(),
                           "legacy_metrics": evaluate_case(case, (), latency_ms=None),
                           "phase2_metrics": evaluate_case(case, (), latency_ms=None)}
        observation["fixture_and_pipeline_wall_clock_ms"] = _elapsed(case_started)
        report["cases"].append(observation)
        attempts = observation.get("provider_admission_attempt_count", 0)
        if attempts is None:
            report["provider_admission_attempt_count"] = None
        elif report["provider_admission_attempt_count"] is not None:
            report["provider_admission_attempt_count"] += attempts
        report["provider_trace_count"] += int(observation.get("provider_trace_count", 0))
        _write_evidence(case_dir / "observation.json", observation)
    report.update(aggregate=_aggregate_observations(report["cases"]), ended_at=_utc(), wall_clock_ms=_elapsed(started),
                  implementation_sha256_after=implementation_fingerprint(deadline=DeadlineBudget.start(1000, now_ns=time.monotonic_ns)),
                  installed_entry_sha256_after=fixture_sha256(entry))
    report["aggregate"]["throughput_including_fixture_cases_per_second"] = len(workload.cases) * 1000 / report["wall_clock_ms"] if report["wall_clock_ms"] else None
    feasible_failures = [case["case_id"] for case in report["cases"]
                         if case["status"] != case["expected_outcome"] or
                         case["phase2_metrics"]["forbidden_cross_scope_count"] or
                         case["phase2_metrics"]["inactive_or_superseded_leak_count"] or
                         (case["expected_outcome"] == "complete" and case["phase2_metrics"]["mandatory_miss_count"])]
    provider_validation_failures = [
        case["case_id"] for case in report["cases"]
        if any(item.get("status") != "ok" for item in case.get("provider_trace", ()))
    ] if profile == "configured" else []
    feasible_failures = list(dict.fromkeys((*feasible_failures, *provider_validation_failures)))
    stable = (report["implementation_sha256_before"] == report["implementation_sha256_after"] and
              report["installed_entry_sha256"] == report["installed_entry_sha256_after"] and
              fixture_sha256(fixture) == workload.query_fixture_sha256 and
              fixture_sha256(fixture.with_name("retrieval_p25_canonical_fixture.json")) == workload.canonical_fixture_sha256)
    report.update(status="PASS" if not feasible_failures and stable else "FAIL", failed_case_ids=feasible_failures,
                  provider_validation_failure_case_ids=provider_validation_failures,
                  implementation_and_fixture_stable=stable)
    _write_evidence(output_dir / "report.json", report)
    return report
