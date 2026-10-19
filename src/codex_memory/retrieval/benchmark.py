"""Deterministic, sanitized P2.3 benchmark metric reporting.

This module evaluates already selected candidate identifiers together with
optional sanitized query-status and candidate-trace evidence.  It deliberately
does not construct a provider, read canonical memory, or emit fixture prompts,
scope paths, notes, or document text.  Those boundaries make framework metric
evidence useful without allowing it to masquerade as live-provider evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from codex_memory.ledger_common import canonical_json

from .contracts import CandidateTrace, RetrievalValidationError


_FIELDS = frozenset({
    "case_id", "query", "project_scope_key", "workspace_scope_keys", "relevant",
    "mandatory", "forbidden", "excluded", "notes",
})
_PROFILE_ORDER = (
    "p2_2", "p2_2_dense", "p2_2_dense_reranker", "p2_2_dense_sparse_reranker",
)
_COMPONENTS = ("dense", "sparse", "reranker")


class BenchmarkValidationError(ValueError):
    """Raised for a fixture or benchmark run that cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Private fixture material; fields with raw inputs never enter output."""

    case_id: str
    query: str
    project_scope_key: str
    workspace_scope_keys: tuple[str, ...]
    relevant: Mapping[str, int]
    mandatory: tuple[str, ...]
    forbidden: tuple[str, ...]
    excluded: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class BenchmarkProfileRun:
    """One profile's selected identifiers and sanitized retrieval audit evidence."""

    name: str
    selected_ids: Mapping[str, Sequence[str]]
    status: str
    component_status: Mapping[str, str]
    latency_ms: Mapping[str, int] | None = None
    query_status: Mapping[str, str] | None = None
    candidate_trace: Mapping[str, Sequence[Mapping[str, object]]] | None = None


def load_benchmark_fixture(path: Path) -> tuple[BenchmarkCase, ...]:
    """Parse the deliberately strict line-oriented fixed fixture."""
    if not isinstance(path, Path):
        raise BenchmarkValidationError("fixture path must be a Path")
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkValidationError("fixture is unavailable") from exc
    if not raw_lines:
        raise BenchmarkValidationError("fixture must not be empty")
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for line in raw_lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkValidationError("fixture line must be JSON") from exc
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise BenchmarkValidationError("fixture fields do not match the fixed contract")
        case_id = _string(raw["case_id"], "case_id")
        if case_id in seen:
            raise BenchmarkValidationError("fixture case IDs must be unique")
        seen.add(case_id)
        query = _string(raw["query"], "query")
        project_scope_key = _string(raw["project_scope_key"], "project_scope_key")
        notes = _string(raw["notes"], "notes")
        workspace = _identifier_list(raw["workspace_scope_keys"], "workspace_scope_keys")
        mandatory = _identifier_list(raw["mandatory"], "mandatory")
        forbidden = _identifier_list(raw["forbidden"], "forbidden")
        excluded = _identifier_list(raw["excluded"], "excluded")
        relevant_raw = raw["relevant"]
        if not isinstance(relevant_raw, dict) or not relevant_raw:
            raise BenchmarkValidationError("relevant must be a non-empty object")
        relevant: dict[str, int] = {}
        for identifier, relevance in relevant_raw.items():
            key = _identifier(identifier, "relevant identifier")
            if isinstance(relevance, bool) or not isinstance(relevance, int) or not 1 <= relevance <= 3:
                raise BenchmarkValidationError("relevance must be an integer from 1 to 3")
            relevant[key] = relevance
        if set(mandatory) - set(relevant):
            raise BenchmarkValidationError("mandatory IDs must also be relevant")
        cases.append(BenchmarkCase(
            case_id, query, project_scope_key, workspace, dict(sorted(relevant.items())),
            mandatory, forbidden, excluded, notes,
        ))
    return tuple(cases)


def evaluate_profiles(
    cases: Sequence[BenchmarkCase],
    runs: Sequence[BenchmarkProfileRun],
    *,
    clock_ms: int,
    evidence_label: str,
) -> dict[str, object]:
    """Compute deterministic public metrics without serializing private inputs."""
    if isinstance(clock_ms, bool) or not isinstance(clock_ms, int) or clock_ms < 0:
        raise BenchmarkValidationError("clock_ms must be a non-negative integer")
    if evidence_label not in {"A: framework/FakeProvider", "B: target-machine configured provider"}:
        raise BenchmarkValidationError("benchmark evidence label is unsupported")
    known_cases = {case.case_id: case for case in cases}
    profiles = sorted(runs, key=lambda run: _profile_index(run.name))
    if len({run.name for run in profiles}) != len(profiles):
        raise BenchmarkValidationError("profile names must be unique")
    documents: list[dict[str, object]] = []
    baseline_hits: dict[str, tuple[str, ...]] | None = None
    for run in profiles:
        component_status = _component_status(run.component_status)
        if set(run.selected_ids) - set(known_cases):
            raise BenchmarkValidationError("profile references an unknown case")
        _validate_query_evidence_keys(run, known_cases)
        cases_document: list[dict[str, object]] = []
        hit_map: dict[str, tuple[str, ...]] = {}
        for case in cases:
            selected = _selected(run.selected_ids.get(case.case_id, ()))
            result = _case_metrics(case, selected, _latency(run, case.case_id, clock_ms))
            audit = _query_audit(run, case.case_id, selected)
            if audit is not None:
                result.update(audit)
            cases_document.append(result)
            hit_map[case.case_id] = tuple(result["hit_ids"])
        aggregate = _aggregate(cases_document, component_status)
        no_gain: list[str] = []
        if baseline_hits is None:
            baseline_hits = hit_map
        else:
            for case_id, hits in hit_map.items():
                baseline = baseline_hits[case_id]
                if hits == baseline or not set(baseline).issubset(hits):
                    no_gain.append(case_id)
        documents.append({
            "name": run.name,
            "status": _status(run.status),
            "component_status": component_status,
            "cases": cases_document,
            "aggregate": aggregate,
            "no_gain_or_regression_case_ids": no_gain,
        })
    return {
        "schema_version": 1,
        "framework": {"evidence_label": evidence_label, "profiles": documents},
    }


def build_benchmark_envelope(*, framework: Mapping[str, object], live_status: str) -> dict[str, object]:
    """Keep unavailable target-machine evidence structurally separate from fake."""
    status = _status(live_status)
    if "framework" not in framework:
        raise BenchmarkValidationError("framework report is required")
    return {
        "schema_version": 1,
        "framework": framework["framework"],
        "live": {
            "status": status,
            "profiles": [],
            "components": {
                "dense": "BLOCKED" if status == "BLOCKED" else "NOT_AVAILABLE",
                "sparse": "UNSUPPORTED",
                "reranker": "UNSUPPORTED",
            },
        },
    }


def serialize_benchmark_report(report: Mapping[str, object]) -> str:
    """Serialize only the public report shape as one canonical JSON line."""
    _reject_sensitive_output(report)
    return canonical_json(report) + "\n"


def fixture_sha256(path: Path) -> str:
    """Return the fixture bytes fingerprint without retaining their contents."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BenchmarkValidationError("fixture is unavailable") from exc


def _case_metrics(case: BenchmarkCase, selected: tuple[str, ...], latency_ms: int) -> dict[str, object]:
    relevant = set(case.relevant)
    hit_ids = [identifier for identifier in selected if identifier in relevant]
    false_ids = [identifier for identifier in selected if identifier not in relevant]
    forbidden = set(case.forbidden)
    excluded = set(case.excluded)
    mandatory = set(case.mandatory)
    ranks = {identifier: index + 1 for index, identifier in enumerate(selected[:10])}
    first_rank = min((ranks[identifier] for identifier in relevant if identifier in ranks), default=None)
    dcg = sum(
        (2 ** case.relevant[identifier] - 1) / math.log2(index + 2)
        for index, identifier in enumerate(selected[:10]) if identifier in case.relevant
    )
    ideal = sorted(case.relevant.values(), reverse=True)[:10]
    ideal_dcg = sum((2 ** gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return {
        "case_id": case.case_id,
        "hit_ids": hit_ids,
        "false_recall_ids": false_ids,
        "recall_at_5": _ratio(len(set(selected[:5]) & relevant), len(relevant)),
        "recall_at_10": _ratio(len(set(selected[:10]) & relevant), len(relevant)),
        "mrr_at_10": 0.0 if first_rank is None else 1.0 / first_rank,
        "ndcg_at_10": 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg,
        "mandatory_miss_count": len(mandatory - set(selected)),
        "forbidden_cross_scope_count": len(forbidden & set(selected)),
        "inactive_or_superseded_leak_count": len(excluded & set(selected)),
        "latency_ms": latency_ms,
    }


def _aggregate(cases: Sequence[Mapping[str, object]], component_status: Mapping[str, str]) -> dict[str, object]:
    values = lambda field: [float(case[field]) for case in cases]
    latency = sorted(int(case["latency_ms"]) for case in cases)
    fallback = {
        component: len(cases)
        for component, status in component_status.items()
        if status not in {"AVAILABLE", "DISABLED"}
    }
    return {
        "recall_at_5": _mean(values("recall_at_5")),
        "recall_at_10": _mean(values("recall_at_10")),
        "mrr_at_10": _mean(values("mrr_at_10")),
        "ndcg_at_10": _mean(values("ndcg_at_10")),
        "mandatory_miss_count": sum(int(case["mandatory_miss_count"]) for case in cases),
        "forbidden_cross_scope_count": sum(int(case["forbidden_cross_scope_count"]) for case in cases),
        "inactive_or_superseded_leak_count": sum(int(case["inactive_or_superseded_leak_count"]) for case in cases),
        "latency_p50_ms": _percentile(latency, 0.50),
        "latency_p95_ms": _percentile(latency, 0.95),
        "component_fallback_counts": fallback,
    }


def _profile_index(name: str) -> int:
    if name not in _PROFILE_ORDER:
        raise BenchmarkValidationError("unknown benchmark profile")
    return _PROFILE_ORDER.index(name)


def _component_status(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != set(_COMPONENTS):
        raise BenchmarkValidationError("component status fields do not match")
    return {component: _status(value[component]) for component in _COMPONENTS}


def _status(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 64 or any(char.isspace() for char in value):
        raise BenchmarkValidationError("status must be a bounded identifier")
    return value


def _latency(run: BenchmarkProfileRun, case_id: str, clock_ms: int) -> int:
    value = clock_ms if run.latency_ms is None else run.latency_ms.get(case_id, clock_ms)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkValidationError("latency must be a non-negative integer")
    return value


def _validate_query_evidence_keys(
    run: BenchmarkProfileRun,
    known_cases: Mapping[str, BenchmarkCase],
) -> None:
    if run.query_status is None and run.candidate_trace is None:
        return
    if run.query_status is None or run.candidate_trace is None:
        raise BenchmarkValidationError("query status and candidate trace must be supplied together")
    expected = set(known_cases)
    if set(run.query_status) != expected or set(run.candidate_trace) != expected:
        raise BenchmarkValidationError("query evidence must cover every benchmark case exactly")


def _query_audit(
    run: BenchmarkProfileRun,
    case_id: str,
    selected: tuple[str, ...],
) -> dict[str, object] | None:
    if run.query_status is None or run.candidate_trace is None:
        return None
    status = _status(run.query_status[case_id])
    value = run.candidate_trace[case_id]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkValidationError("candidate trace must be a sequence")
    normalized: list[dict[str, object]] = []
    try:
        for item in value:
            if not isinstance(item, Mapping):
                raise BenchmarkValidationError("candidate trace entry must be an object")
            normalized.append(CandidateTrace.from_dict(dict(item)).to_dict())
    except RetrievalValidationError as exc:
        raise BenchmarkValidationError("candidate trace entry is invalid") from exc
    identifiers = tuple(str(item["candidate_id"]) for item in normalized)
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkValidationError("candidate trace identifiers must not repeat")
    traced_selected = tuple(
        str(item["candidate_id"])
        for item in normalized
        if item["selected"] is True
    )
    if traced_selected != selected:
        raise BenchmarkValidationError("selected candidates do not match retrieval trace")
    trace_json = canonical_json(normalized)
    return {
        "retrieval_status": status,
        "candidate_trace_audit": {
            "entry_count": len(normalized),
            "selected_ids": list(traced_selected),
            "sha256": hashlib.sha256(trace_json.encode("utf-8")).hexdigest(),
        },
    }


def _selected(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkValidationError("selected IDs must be a sequence")
    selected = tuple(_identifier(item, "selected ID") for item in value)
    if len(selected) != len(set(selected)):
        raise BenchmarkValidationError("selected IDs must not repeat")
    return selected


def _identifier_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BenchmarkValidationError(f"{name} must be a list")
    identifiers = tuple(_identifier(item, name) for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkValidationError(f"{name} must not repeat values")
    return identifiers


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or any(char.isspace() for char in value):
        raise BenchmarkValidationError(f"{name} must be a bounded identifier")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise BenchmarkValidationError(f"{name} must be bounded text")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _reject_sensitive_output(value: object) -> None:
    """Defence-in-depth: public reports cannot contain paths or raw input fields."""
    prohibited = {"query", "project_scope_key", "workspace_scope_keys", "notes"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in prohibited:
                raise BenchmarkValidationError("public benchmark output contains private fixture material")
            _reject_sensitive_output(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_sensitive_output(item)
    elif isinstance(value, str) and ("\\\\" in value or ":\\" in value):
        raise BenchmarkValidationError("public benchmark output contains a local path")
