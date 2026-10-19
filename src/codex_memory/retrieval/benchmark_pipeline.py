"""Isolated execution of the fixed P2.3 benchmark through real retrieval APIs.

The public benchmark evaluator stays pure.  This module owns only test/dev
fixture materialization: it creates a temporary repository, appends records
through the normal ledger, runs the normal deterministic/semantic builders,
and converts real query candidates, status, and trace back to fixture labels
for the evaluator.
It never opens, copies, or mutates a caller's canonical memory database.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from codex_memory import database, structured_ledger
from codex_memory.blob_store import BlobStore
from codex_memory.candidate_memory import (
    capture_candidate,
)
from codex_memory.hook_input import parse_hook_payload
from codex_memory.memory_evidence import event_uri
from codex_memory.event_store import EventStore
from codex_memory.memory_review import review_candidate
from codex_memory.paths import RuntimePathError, RuntimePaths, require_safe_directory

from .benchmark import BenchmarkCase, BenchmarkProfileRun, BenchmarkValidationError
from .cache_database import connect_cache_database, verify_provider_attempt_history
from .config import SemanticConfig
from .contracts import DegradedReason, RetrievalMode, RetrievalRequest, RetrievalScope, RetrievalStatus
from .paths import RetrievalPaths
from .provider import ProviderOperation, RetrievalModelProvider
from .service import RetrievalService
from .shadow import query_shadow_observation


_CANONICAL_FIELDS = frozenset({"records"})
_RECORD_FIELDS = frozenset({"label", "record_type", "scope", "topic_key", "statement"})
_RECORD_AFTER_SEMANTIC_FIELDS = _RECORD_FIELDS | {"after_semantic"}
_PROFILES = (
    ("p2_2", True, True, True),
    ("p2_2_dense", False, True, True),
    ("p2_2_dense_reranker", False, True, False),
    ("p2_2_dense_sparse_reranker", False, False, False),
)
_CHILD_GENERATION_ID = re.compile(r"\Agen_[0-9a-f]{16,64}\Z")
_MAX_CHILD_DEGRADED_REASONS = 16
_MAX_CHILD_TRACE_ITEMS = 256


class PipelineBlocked(RuntimeError):
    """A configured benchmark cannot run without an explicit valid provider."""

    def __init__(
        self,
        status: str = "BLOCKED",
        *,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(status)
        self.status = status
        self.diagnostics: dict[str, object] = dict(diagnostics or {})


@dataclass(frozen=True, slots=True)
class BenchmarkPipelineResult:
    """Private execution result.  The caller supplies runs to the pure evaluator."""

    runs: tuple[BenchmarkProfileRun, ...]
    label_to_record_id: Mapping[str, str]
    provider_call_count: int
    provider_input_audit: Mapping[str, Mapping[str, int]]
    provider_attempt_audit: Mapping[str, Mapping[str, int]]
    mandatory_slot_audit: Mapping[str, Mapping[str, Mapping[str, object]]]


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkPipeline:
    """One isolated, live fixture retained for repeated pipeline evidence.

    This is intentionally a test/dev boundary: it exposes the same repository,
    deterministic projection, verified semantic generation, and provider that a
    normal benchmark run uses.  It does not introduce another retrieval path.
    """

    repo: Path
    workspace: Path
    runtime: RuntimePaths
    paths: RetrievalPaths
    service: RetrievalService
    provider: RetrievalModelProvider
    config: SemanticConfig
    label_to_record_id: Mapping[str, str]


def prepare_benchmark_pipeline(
    cases: Sequence[BenchmarkCase],
    canonical_fixture: Path,
    temporary_root: Path,
    *,
    provider_factory: Callable[[], RetrievalModelProvider] | None,
    config_factory: Callable[[RetrievalModelProvider], SemanticConfig] | None,
) -> PreparedBenchmarkPipeline:
    """Build one retained fixture through the official ledger/index boundaries."""
    if provider_factory is None or config_factory is None:
        raise PipelineBlocked()
    if not isinstance(temporary_root, Path):
        raise BenchmarkValidationError("temporary root must be a Path")
    if not cases:
        raise BenchmarkValidationError("benchmark cases must not be empty")
    fixture = _load_canonical_fixture(canonical_fixture)
    repo = temporary_root / "benchmark-repository"
    repo.mkdir(parents=True, exist_ok=False)
    repo = _initialize_fixture_repository(repo)
    workspace = repo / "benchmark-workspace"
    workspace.mkdir()
    runtime = RuntimePaths.for_repo(repo)
    runtime.ensure()
    paths = RetrievalPaths.for_repo(repo)
    label_to_record_id = _materialize_fixture(runtime, fixture, repo, workspace, include_after_semantic=False)

    provider = provider_factory()
    config = config_factory(provider)
    if not isinstance(config, SemanticConfig):
        raise PipelineBlocked()
    service = RetrievalService(paths)
    build = service.rebuild_semantic(provider, config)
    if build.status.value != "current" or not _enabled_semantic_components_ready(paths, config):
        raise PipelineBlocked("UNAVAILABLE")

    # This record is deliberately added after semantic materialization and only
    # caught up through the deterministic index.  It remains reachable through
    # its direct/exact deterministic path, never by a synthetic benchmark rank.
    label_to_record_id.update(_materialize_fixture(runtime, fixture, repo, workspace, include_after_semantic=True))
    if service.catch_up().status.value != "current":
        raise PipelineBlocked("UNAVAILABLE")

    return PreparedBenchmarkPipeline(
        repo=repo,
        workspace=workspace,
        runtime=runtime,
        paths=paths,
        service=service,
        provider=provider,
        config=config,
        label_to_record_id=dict(label_to_record_id),
    )


def _initialize_fixture_repository(repo: Path) -> Path:
    """Use Git's long root spelling for scope keys shared with child processes."""
    initialized = subprocess.run(
        ["git", "-C", str(repo), "init", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    if initialized.returncode != 0:
        raise PipelineBlocked("fixture_git_init")
    root = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    if root.returncode != 0 or not root.stdout.strip():
        raise PipelineBlocked("fixture_git_root")
    try:
        requested = require_safe_directory(repo, repo.parent)
        reported = require_safe_directory(Path(root.stdout.strip()))
        if not requested.samefile(reported):
            raise PipelineBlocked("fixture_git_root")
        return reported
    except (OSError, RuntimePathError) as exc:
        raise PipelineBlocked("fixture_git_root") from exc


def run_prepared_benchmark_pipeline(
    prepared: PreparedBenchmarkPipeline,
    cases: Sequence[BenchmarkCase],
    *,
    query_overrides: Mapping[str, str] | None = None,
) -> BenchmarkPipelineResult:
    """Query one established fixture; callers may repeat it without rebuilding."""
    if not isinstance(prepared, PreparedBenchmarkPipeline):
        raise BenchmarkValidationError("prepared benchmark pipeline is invalid")
    if not cases:
        raise BenchmarkValidationError("benchmark cases must not be empty")
    overrides = dict(query_overrides or {})
    if set(overrides) - {case.case_id for case in cases}:
        raise BenchmarkValidationError("query override references an unknown case")
    if any(not isinstance(value, str) or not value for value in overrides.values()):
        raise BenchmarkValidationError("query override must be non-empty text")
    reverse = {record_id: label for label, record_id in prepared.label_to_record_id.items()}
    profile_results = tuple(
        _run_profile(
            name, disable_dense, disable_sparse, disable_reranker,
            cases, overrides, prepared.paths, prepared.service, prepared.provider,
            prepared.config, prepared.repo, prepared.workspace, reverse,
        )
        for name, disable_dense, disable_sparse, disable_reranker in _PROFILES
    )
    runs = tuple(result[0] for result in profile_results)
    mandatory_slot_audit = {run.name: audit for run, audit in profile_results}
    attempt_audit = read_provider_attempt_audit(prepared.paths)
    count = sum(value["admission_count"] for value in attempt_audit.values())
    input_audit = _provider_input_audit(prepared.provider)
    return BenchmarkPipelineResult(
        runs,
        dict(prepared.label_to_record_id),
        count,
        input_audit,
        attempt_audit,
        mandatory_slot_audit,
    )


def read_provider_attempt_audit(
    paths: RetrievalPaths,
) -> Mapping[str, Mapping[str, int]]:
    """Summarize durable provider admissions without exposing request content."""
    audit: dict[str, dict[str, int]] = {
        operation.value: {
            "admission_count": 0,
            "result_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "unresolved_count": 0,
        }
        for operation in ProviderOperation
    }
    connection = connect_cache_database(paths.database_path, readonly=True)
    try:
        verify_provider_attempt_history(connection)
        rows = connection.execute(
            """SELECT admission.operation,
                      COUNT(*) AS admission_count,
                      COUNT(result.result_id) AS result_count,
                      COALESCE(SUM(CASE WHEN result.outcome = 'success' THEN 1 ELSE 0 END), 0)
                        AS success_count,
                      COALESCE(SUM(CASE WHEN result.result_id IS NOT NULL
                                             AND result.outcome <> 'success'
                                        THEN 1 ELSE 0 END), 0)
                        AS failure_count,
                      COALESCE(SUM(CASE WHEN result.result_id IS NULL THEN 1 ELSE 0 END), 0)
                        AS unresolved_count
                 FROM retrieval_provider_attempt_admissions AS admission
                 LEFT JOIN retrieval_provider_attempt_results AS result
                   ON result.attempt_id = admission.attempt_id
                GROUP BY admission.operation
                ORDER BY admission.operation"""
        ).fetchall()
    finally:
        connection.close()
    for row in rows:
        operation = row["operation"]
        if operation not in audit:
            raise BenchmarkValidationError("provider attempt operation is invalid")
        audit[operation] = {
            "admission_count": row["admission_count"],
            "result_count": row["result_count"],
            "success_count": row["success_count"],
            "failure_count": row["failure_count"],
            "unresolved_count": row["unresolved_count"],
        }
    return audit


def run_fresh_child_query(
    repo: Path,
    config: SemanticConfig,
    query: str,
) -> Mapping[str, object]:
    """Reopen one retained fixture through the installed production CLI.

    The isolated repository receives a reconstructed, schema-validated config;
    comments or unknown text from the caller's fixture are never copied.  The
    child inherits no synthetic ``PYTHONPATH`` and therefore imports the same
    installed runtime used by the benchmark command.
    """
    if not isinstance(repo, Path) or not isinstance(config, SemanticConfig):
        raise BenchmarkValidationError("fresh child fixture is invalid")
    if not isinstance(query, str) or not query:
        raise BenchmarkValidationError("fresh child query is invalid")
    repo = repo.resolve()
    codex_dir = repo / ".codex"
    codex_dir.mkdir(exist_ok=True)
    (codex_dir / "memory.toml").write_text(
        _semantic_fixture_toml(config), encoding="utf-8", newline="\n"
    )
    initialized = subprocess.run(
        ["git", "-C", str(repo), "init", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    if initialized.returncode != 0:
        raise PipelineBlocked("fresh_child_git_init")
    command = [
        sys.executable,
        "-m",
        "codex_memory.cli",
        "retrieval",
        "query",
        "--session-id",
        "benchmark-fresh-child-session",
        "--turn-id",
        "benchmark-fresh-child-turn",
        "--context-id",
        "benchmark-fresh-child-context",
        "--query",
        query,
        "--disable-sparse",
        "--disable-reranker",
        "--explain",
    ]
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=(config.query_deadline_ms / 1000) + 10,
        )
    except subprocess.TimeoutExpired as exc:
        wall_clock_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        raise PipelineBlocked(
            "fresh_child_process_timeout",
            diagnostics=_fresh_child_diagnostics(
                return_code=None,
                stdout=getattr(exc, "stdout", None),
                stderr=getattr(exc, "stderr", None),
                wall_clock_ms=wall_clock_ms,
            ),
        ) from exc
    wall_clock_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
    if completed.returncode != 0 or completed.stderr:
        raise PipelineBlocked(
            "fresh_child_process_exit",
            diagnostics=_fresh_child_diagnostics(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                wall_clock_ms=wall_clock_ms,
            ),
        )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineBlocked(
            "fresh_child_output",
            diagnostics=_fresh_child_diagnostics(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                wall_clock_ms=wall_clock_ms,
            ),
        ) from exc
    child_evidence = _fresh_child_diagnostics(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        wall_clock_ms=wall_clock_ms,
        document=document,
    )
    generation = document.get("generation") if isinstance(document, dict) else None
    trace = document.get("trace") if isinstance(document, dict) else None
    degraded = document.get("degraded_reasons") if isinstance(document, dict) else None
    if (
        not isinstance(generation, dict)
        or not isinstance(child_evidence.get("generation_id"), str)
        or not isinstance(trace, list)
        or not isinstance(degraded, list)
        or document.get("status") != "current"
    ):
        raise PipelineBlocked("fresh_child_status", diagnostics=child_evidence)
    dense_trace_count = child_evidence["dense_trace_count"]
    assert isinstance(dense_trace_count, int)
    if dense_trace_count < 1:
        raise PipelineBlocked(
            "fresh_child_dense_trace", diagnostics=child_evidence
        )
    return child_evidence


def _fresh_child_diagnostics(
    *,
    return_code: int | None,
    stdout: object,
    stderr: object,
    wall_clock_ms: int,
    document: object | None = None,
) -> dict[str, object]:
    """Keep bounded, non-content evidence when a child query is blocked."""
    def digest(value: object) -> str | None:
        if isinstance(value, bytes):
            raw = value
        elif isinstance(value, str):
            raw = value.encode("utf-8")
        else:
            return None
        return hashlib.sha256(raw).hexdigest()

    status: str | None = None
    generation_id: str | None = None
    degraded_reasons: list[str] = []
    trace_count: int | None = None
    dense_trace_count = 0
    degraded_truncated = False
    trace_truncated = False
    if isinstance(document, dict):
        candidate_status = document.get("status")
        if isinstance(candidate_status, str):
            status = (
                candidate_status
                if candidate_status in {item.value for item in RetrievalStatus}
                else "unknown"
            )
        generation = document.get("generation")
        candidate_generation_id = (
            generation.get("generation_id") if isinstance(generation, dict) else None
        )
        if (
            isinstance(candidate_generation_id, str)
            and _CHILD_GENERATION_ID.fullmatch(candidate_generation_id)
        ):
            generation_id = candidate_generation_id
        degraded = document.get("degraded_reasons")
        if isinstance(degraded, list):
            degraded_truncated = len(degraded) > _MAX_CHILD_DEGRADED_REASONS
            for item in degraded[:_MAX_CHILD_DEGRADED_REASONS]:
                degraded_reasons.append(
                    item
                    if isinstance(item, str)
                    and item in {reason.value for reason in DegradedReason}
                    else "unknown"
                )
        trace = document.get("trace")
        if isinstance(trace, list):
            trace_truncated = len(trace) > _MAX_CHILD_TRACE_ITEMS
            trace_count = min(len(trace), _MAX_CHILD_TRACE_ITEMS)
            dense_trace_count = sum(
                1
                for item in trace[:_MAX_CHILD_TRACE_ITEMS]
                if isinstance(item, dict)
                and isinstance(item.get("source_channels"), list)
                and "dense" in item["source_channels"]
            )
    return {
        "entry": "python -m codex_memory.cli",
        "return_code": return_code,
        "stderr_empty": stderr in ("", b""),
        "status": status,
        "generation_id": generation_id,
        "degraded_reasons": degraded_reasons,
        "degraded_reasons_truncated": degraded_truncated,
        "trace_count": trace_count,
        "trace_truncated": trace_truncated,
        "dense_trace_count": dense_trace_count,
        "stdout_sha256": digest(stdout),
        "stderr_sha256": digest(stderr),
        "wall_clock_ms": wall_clock_ms,
    }


def _semantic_fixture_toml(config: SemanticConfig) -> str:
    """Serialize only fields accepted by ``SemanticConfig`` into a fixture."""
    values = config.to_dict()
    lines = ["[retrieval]", 'mode = "shadow"', "", "[retrieval.semantic]"]
    for name, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(value)
        elif isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            raise BenchmarkValidationError("semantic fixture field is invalid")
        lines.append(f"{name} = {rendered}")
    return "\n".join(lines) + "\n"


def run_benchmark_pipeline(
    cases: Sequence[BenchmarkCase],
    canonical_fixture: Path,
    temporary_root: Path,
    *,
    provider_factory: Callable[[], RetrievalModelProvider] | None,
    config_factory: Callable[[RetrievalModelProvider], SemanticConfig] | None,
    query_overrides: Mapping[str, str] | None = None,
) -> BenchmarkPipelineResult:
    """Materialize and execute the benchmark using only the official boundaries."""
    prepared = prepare_benchmark_pipeline(
        cases, canonical_fixture, temporary_root,
        provider_factory=provider_factory, config_factory=config_factory,
    )
    return run_prepared_benchmark_pipeline(
        prepared, cases, query_overrides=query_overrides,
    )


def _provider_input_audit(
    provider: RetrievalModelProvider,
) -> Mapping[str, Mapping[str, int]]:
    """Return opaque FakeProvider input evidence without fixture content.

    Production providers are not required to retain request material, so this
    optional test/dev seam deliberately accepts only the test provider's
    already-hashed batches.  It never serializes raw query or document text.
    """
    observed = getattr(provider, "call_inputs", None)
    if not isinstance(observed, Mapping):
        return {}
    audit: dict[str, Mapping[str, int]] = {}
    for operation in ProviderOperation:
        batches = observed.get(operation)
        if not isinstance(batches, list):
            continue
        hashes: list[str] = []
        for batch in batches:
            if not isinstance(batch, tuple) or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in batch
            ):
                continue
            hashes.extend(batch)
        audit[operation.value] = {
            "call_count": len(batches),
            "distinct_input_sha256_count": len(set(hashes)),
        }
    return audit


def _enabled_semantic_components_ready(paths: RetrievalPaths, config: SemanticConfig) -> bool:
    """Require real provider-backed rows before a configured benchmark is live."""
    connection = sqlite3.connect(paths.database_path)
    try:
        state = connection.execute(
            "SELECT active_semantic_generation_id, validation_state "
            "FROM retrieval_semantic_state WHERE singleton_id = 1"
        ).fetchone()
        if state is None or state[1] != "validated" or not isinstance(state[0], str):
            return False
        rows = {
            row[0]: row[1:]
            for row in connection.execute(
                "SELECT component, status, eligible_document_count, indexed_document_count "
                "FROM retrieval_semantic_components WHERE semantic_generation_id = ?",
                (state[0],),
            )
        }
    except (OSError, sqlite3.Error):
        return False
    finally:
        connection.close()
    # Reranking is a bounded query-time observation in P2.3 and intentionally
    # has no build-time generation rows (``task3_no_rerank_build``).  Dense and
    # sparse, however, must have complete provider-backed document coverage so
    # a configured run cannot silently fall back to the deterministic floor.
    for component, enabled in (("dense", config.dense), ("sparse", config.sparse)):
        if enabled:
            row = rows.get(component)
            if row is None or row[0] != "ready" or row[1] != row[2]:
                return False
    return True


def _run_profile(
    name: str,
    disable_dense: bool,
    disable_sparse: bool,
    disable_reranker: bool,
    cases: Sequence[BenchmarkCase],
    overrides: Mapping[str, str],
    paths: RetrievalPaths,
    service: RetrievalService,
    provider: RetrievalModelProvider,
    config: SemanticConfig,
    repo: Path,
    workspace: Path,
    reverse: Mapping[str, str],
) -> tuple[BenchmarkProfileRun, dict[str, dict[str, object]]]:
    selected: dict[str, tuple[str, ...]] = {}
    latencies: dict[str, int] = {}
    query_status: dict[str, str] = {}
    candidate_trace: dict[str, tuple[dict[str, object], ...]] = {}
    observed_statuses: dict[str, list[str]] = {
        component: [] for component in ("dense", "sparse", "reranker")
    }
    mandatory_slot_audit: dict[str, dict[str, object]] = {}
    for case in cases:
        query = overrides.get(case.case_id, case.query)
        scope = RetrievalScope(str(repo), (str(workspace),) if case.workspace_scope_keys else ())
        started = time.monotonic_ns()
        if name == "p2_2":
            request = _request(query, case.case_id, scope, RetrievalMode.OFF)
            result = service.query(request)
            raw_result = result
        else:
            request = _request(query, case.case_id, scope, RetrievalMode.SHADOW)
            observation = query_shadow_observation(
                paths=paths,
                request=request,
                provider=provider,
                semantic_config=config,
                disable_dense=disable_dense,
                disable_sparse=disable_sparse,
                disable_reranker=disable_reranker,
            )
            raw_result = observation.result
            result = observation.reranked_result or raw_result
            for item in observation.provider_trace:
                component = item.get("component")
                status = item.get("status")
                if component in observed_statuses and isinstance(status, str):
                    observed_statuses[component].append(status)
        if case.mandatory:
            identifier = case.mandatory[0]
            raw_slot, raw_candidate = next(
                (
                    (index, candidate)
                    for index, candidate in enumerate(raw_result.candidates)
                    if reverse.get(candidate.candidate_id) == identifier
                ),
                (-1, None),
            )
            reranked_slot = next(
                (
                    index
                    for index, candidate in enumerate(result.candidates)
                    if reverse.get(candidate.candidate_id) == identifier
                ),
                -1,
            )
            mandatory_slot_audit[case.case_id] = {
                "certification_action": "promote",
                "certification_policy_code": "linked_duplicate",
                "mandatory": bool(raw_candidate is not None and raw_candidate.mandatory),
                "raw_slot": raw_slot,
                "reranked_slot": reranked_slot,
            }
        latencies[case.case_id] = max(0, (time.monotonic_ns() - started) // 1_000_000)
        query_status[case.case_id] = result.status.value
        selected[case.case_id] = tuple(
            reverse.get(candidate.candidate_id, "unmapped")
            for candidate in result.candidates
            if candidate.candidate_kind == "record" and candidate.candidate_id in reverse
        )
        trace_document: list[dict[str, object]] = []
        for item in result.trace:
            label = reverse.get(item.candidate_id)
            if label is None:
                continue
            document = item.to_dict()
            document["candidate_id"] = label
            trace_document.append(document)
        candidate_trace[case.case_id] = tuple(trace_document)
    requested = {
        "dense": not disable_dense,
        "sparse": not disable_sparse,
        "reranker": not disable_reranker,
    }
    component_status: dict[str, str] = {}
    for component in ("dense", "sparse", "reranker"):
        if not requested[component]:
            component_status[component] = "DISABLED"
            continue
        statuses = observed_statuses[component]
        if not statuses:
            component_status[component] = "NOT_AVAILABLE"
        elif all(status == "ok" for status in statuses):
            component_status[component] = "AVAILABLE"
        elif all(status == "unsupported" for status in statuses):
            component_status[component] = "UNSUPPORTED"
        else:
            component_status[component] = "DEGRADED"
    available = all(
        not requested[component] or component_status[component] == "AVAILABLE"
        for component in requested
    )
    return (
        BenchmarkProfileRun(
            name,
            selected,
            "AVAILABLE" if available else "DEGRADED",
            component_status,
            latencies,
            query_status,
            candidate_trace,
        ),
        mandatory_slot_audit,
    )


def _request(query: str, case_id: str, scope: RetrievalScope, mode: RetrievalMode) -> RetrievalRequest:
    return RetrievalRequest(
        query, f"benchmark-session-{case_id}", f"benchmark-turn-{case_id}",
        f"benchmark-context-{case_id}", None, scope, mode=mode,
    )


def _load_canonical_fixture(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkValidationError("canonical benchmark fixture is invalid") from exc
    if not isinstance(value, dict) or set(value) != _CANONICAL_FIELDS or not isinstance(value["records"], list):
        raise BenchmarkValidationError("canonical benchmark fixture fields are invalid")
    records: list[dict[str, Any]] = []
    labels: set[str] = set()
    for item in value["records"]:
        if not isinstance(item, dict) or set(item) not in {_RECORD_FIELDS, _RECORD_AFTER_SEMANTIC_FIELDS}:
            raise BenchmarkValidationError("canonical benchmark record fields are invalid")
        label = item.get("label")
        if not isinstance(label, str) or not label or label in labels:
            raise BenchmarkValidationError("canonical benchmark record label is invalid")
        labels.add(label)
        if item.get("record_type") not in {"constitution", "decision"} or item.get("scope") not in {"project", "workspace", "foreign"}:
            raise BenchmarkValidationError("canonical benchmark record identity is invalid")
        if any(not isinstance(item.get(field), str) or not item[field] for field in ("topic_key", "statement")):
            raise BenchmarkValidationError("canonical benchmark record text is invalid")
        if "after_semantic" in item and item["after_semantic"] is not True:
            raise BenchmarkValidationError("after_semantic must be true when present")
        records.append(dict(item))
    return tuple(records)


def _materialize_fixture(
    runtime: RuntimePaths,
    records: Sequence[Mapping[str, Any]],
    repo: Path,
    workspace: Path,
    *,
    include_after_semantic: bool,
) -> dict[str, str]:
    connection = database.connect_database(runtime.db_path)
    database.migrate_database(connection)
    created: dict[str, str] = {}
    try:
        for item in records:
            after = item.get("after_semantic") is True
            if after != include_after_semantic:
                continue
            scope = item["scope"]
            scope_kind, scope_key = (
                ("project", str(repo)) if scope == "project" else
                ("workspace", str(workspace)) if scope == "workspace" else
                ("project", "benchmark-foreign-scope")
            )
            proposal = structured_ledger.propose_record(
                connection,
                record_type=item["record_type"],
                scope_kind=scope_kind,
                scope_key=scope_key,
                topic_key=item["topic_key"],
                statement=item["statement"],
                idempotency_key=f"benchmark-propose-{item['label']}",
            )
            structured_ledger.apply_action(
                connection,
                record_id=proposal.record_id,
                action="activate",
                idempotency_key=f"benchmark-activate-{item['label']}",
            )
            if item["label"] == "mem_constitution":
                # The fixture deliberately uses the same direct-user evidence
                # and candidate-promotion chain as production certification.
                # A bare active Constitution must remain unverified.
                direct_user_event = EventStore(connection, BlobStore(runtime.blobs_dir)).append(parse_hook_payload({
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "benchmark-certification-session",
                    "turn_id": "benchmark-certification-turn",
                    "cwd": str(repo),
                    "prompt": "certify benchmark constitution",
                }))
                candidate = capture_candidate(
                    connection,
                    BlobStore(runtime.blobs_dir),
                    project_scope_key=str(repo),
                    record_type="constitution",
                    scope_kind=scope_kind,
                    scope_key=scope_key,
                    topic_key=item["topic_key"],
                    statement=item["statement"],
                    session_id="benchmark-certification-session",
                    turn_id="benchmark-certification-turn",
                    context_id="benchmark-certification-context",
                    idempotency_key="benchmark-constitution-candidate",
                    evidence_uris=(event_uri("benchmark-certification-session", direct_user_event.event_id),),
                )
                reviewed = review_candidate(
                    connection,
                    BlobStore(runtime.blobs_dir),
                    candidate_id=candidate.candidate_id,
                    outcome="promote",
                    reason="benchmark direct user certification",
                    idempotency_key="benchmark-constitution-promote",
                )
                if (
                    reviewed.action != "promote"
                    or reviewed.policy_code != "linked_duplicate"
                    or reviewed.permanent_record_id != proposal.record_id
                ):
                    raise BenchmarkValidationError("benchmark Constitution certification failed")
            created[item["label"]] = proposal.record_id
        if not include_after_semantic and "mem_superseded" in created and "mem_current" in created:
            structured_ledger.apply_action(
                connection,
                record_id=created["mem_superseded"],
                action="supersede",
                replacement_record_id=created["mem_current"],
                idempotency_key="benchmark-supersede-old",
            )
        return created
    finally:
        connection.close()
