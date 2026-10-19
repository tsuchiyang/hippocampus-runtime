from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Callable, Sequence, TextIO

from codex_memory.ledger_common import canonical_json
from codex_memory.blob_store import BlobStore
from codex_memory.memory_context import build_memory_packet
from codex_memory.project_memory_config import load_project_memory_config
from codex_memory.paths import RuntimePathError

from .cache_database import SCHEMA_VERSION
from .contracts import DegradedReason, RetrievalMode, RetrievalRequest, RetrievalResult, RetrievalScope, RetrievalStatus
from .config import RetrievalConfigStatus, load_retrieval_config, provider_from_config
from .paths import RetrievalPaths
from .canonical_snapshot import (
    CanonicalSnapshotInvalidError,
    CanonicalSnapshotUnavailableError,
)
from .service import RetrievalService
from .query import query_result_document
from .canonical_snapshot import open_verified_canonical_snapshot


_COMMANDS = frozenset({"status", "init", "verify", "rebuild", "catch-up"})
_SEMANTIC_MAINTENANCE_COMMANDS = frozenset({"rebuild", "catch-up"})


def run_retrieval_command(
    arguments: Sequence[str],
    stdout: TextIO,
    stderr: TextIO,
    resolve_repo_root: Callable[[Path], Path],
) -> int:
    """Run one isolated retrieval-substrate command."""
    tokens = list(arguments)
    if len(tokens) >= 2 and tokens[:2] == ["retrieval", "recall"]:
        # Explicit laboratory terminal emission only; never a Hook mode switch.
        from .isolated_cli import run_isolated_recall

        return run_isolated_recall(tokens[2:], stdout, stderr, resolve_repo_root)
    if len(tokens) >= 2 and tokens[:2] == ["retrieval", "explain"]:
        return _run_historical_explain(tokens[2:], stdout, stderr, resolve_repo_root)
    if len(tokens) >= 2 and tokens[:2] == ["retrieval", "query"]:
        return _run_query(tokens[2:], stdout, stderr, resolve_repo_root)
    if any(type(token) is not str for token in tokens) or not _valid_maintenance_tokens(tokens):
        stderr.write("codex-memory-hook: invalid retrieval input\n")
        return 2

    try:
        repo_root = resolve_repo_root(Path.cwd())
        service = RetrievalService(RetrievalPaths.for_repo(repo_root))
    except Exception:
        stderr.write("codex-memory-hook: runtime unavailable\n")
        return 1

    command = tokens[1]
    semantic = len(tokens) == 3
    try:
        if semantic:
            config = load_retrieval_config(repo_root)
            # The semantic maintenance switch is deliberately neither an
            # implicit configuration migration nor a way to activate an
            # arbitrary provider.  A non-shadow or malformed setting remains
            # a bounded read-only failure and creates no provider.
            if not config.semantic_enabled:
                stdout.write(_serialize_result(service.status()))
                return 1
            provider = provider_from_config(config.semantic)
            result = (
                service.catch_up_semantic(provider, config.semantic)
                if command == "catch-up"
                else service.rebuild_semantic(provider, config.semantic)
            )
        else:
            result = service.catch_up() if command == "catch-up" else getattr(service, command)()
        stdout.write(_serialize_result(result))
    except Exception as error:
        if isinstance(error, CanonicalSnapshotUnavailableError):
            reason = DegradedReason.CANONICAL_UNAVAILABLE
        elif isinstance(error, CanonicalSnapshotInvalidError):
            reason = DegradedReason.CANONICAL_INVALID
        else:
            reason = (
                DegradedReason.PATH_UNSAFE
                if isinstance(error, RuntimePathError)
                else DegradedReason.PATH_UNAVAILABLE
            )
        stdout.write(
            _serialize_result(
                RetrievalResult(
                    status=RetrievalStatus.CORRUPT,
                    mode=RetrievalMode.OFF,
                    generation=None,
                    canonical_highwater=None,
                    indexed_highwater=None,
                    degraded_reason=reason,
                )
            )
        )
        return 1

    if command == "status":
        return 0
    if command == "init":
        return 0 if result.status in {RetrievalStatus.CURRENT, RetrievalStatus.STALE} else 1
    return 0 if result.status is RetrievalStatus.CURRENT else 1


def _load_historical_injection(repo_root: Path, injection_id: str) -> dict[str, object]:
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    from .injection_audit import explain_injection

    retrieval_paths = RetrievalPaths.for_repo(repo_root)
    store = InjectionStore(InjectionPaths.for_repo(repo_root), BlobStore(retrieval_paths.runtime_root / "blobs"))
    return explain_injection(store.load(injection_id))


def _run_historical_explain(tokens: list[str], stdout: TextIO, stderr: TextIO, resolve_repo_root: Callable[[Path], Path]) -> int:
    if len(tokens) != 1 or not isinstance(tokens[0], str) or re.fullmatch(r"inj_[0-9a-f]{64}", tokens[0]) is None:
        stderr.write("codex-memory-hook: invalid retrieval input\n")
        return 2
    try:
        repo_root = resolve_repo_root(Path.cwd())
        document = _load_historical_injection(repo_root, tokens[0])
        stdout.write(canonical_json(document) + "\n")
        return 0
    except Exception as error:
        from codex_memory.injection_manifest import InjectionAuditError

        if isinstance(error, InjectionAuditError) and error.reason.value == "invalid_input":
            stderr.write("codex-memory-hook: invalid retrieval input\n")
            return 2
        reason = error.reason.value if isinstance(error, InjectionAuditError) else "unavailable"
        stdout.write(canonical_json({"status": "error", "reason": reason}) + "\n")
        return 1


def _valid_maintenance_tokens(tokens: list[str]) -> bool:
    if len(tokens) == 2:
        return tokens[0] == "retrieval" and tokens[1] in _COMMANDS
    return (
        len(tokens) == 3
        and tokens[0] == "retrieval"
        and tokens[1] in _SEMANTIC_MAINTENANCE_COMMANDS
        and tokens[2] == "--semantic"
    )


def _run_query(tokens: list[str], stdout: TextIO, stderr: TextIO, resolve_repo_root: Callable[[Path], Path]) -> int:
    try:
        values = _parse_query_arguments(tokens)
        repo_root = resolve_repo_root(Path.cwd())
        config = load_project_memory_config(repo_root)
        retrieval_config = load_retrieval_config(repo_root)
        if retrieval_config.status is RetrievalConfigStatus.DISABLED:
            raise ValueError("active retrieval is unavailable")
        request = RetrievalRequest(
            values["query"], values["session_id"], values["turn_id"], values["context_id"],
            values["prompt_event_uri"], RetrievalScope(config.project_key_for(repo_root), config.workspace_scope_keys),
            mode=RetrievalMode.SHADOW if retrieval_config.enabled else RetrievalMode.OFF,
            max_chars=values["max_chars"],
        )
        paths = RetrievalPaths.for_repo(repo_root)
        service = RetrievalService(paths)
        if retrieval_config.semantic_enabled:
            from .shadow import query_shadow

            result = query_shadow(
                paths=paths,
                request=request,
                provider=provider_from_config(retrieval_config.semantic),
                semantic_config=retrieval_config.semantic,
                disable_dense=bool(values["disable_dense"]),
                disable_sparse=bool(values["disable_sparse"]),
                disable_reranker=bool(values["disable_reranker"]),
            )
        else:
            result = service.query(request)
        document = query_result_document(result, explain=bool(values["explain"]))
        if values["legacy"]:
            packet = _legacy_packet(paths, request)
            document["legacy"] = {"packet": packet, "sha256": hashlib.sha256(packet.encode("utf-8")).hexdigest(), "length": len(packet)}
        stdout.write(canonical_json(document) + "\n")
        return 0 if result.status in {RetrievalStatus.CURRENT, RetrievalStatus.STALE, RetrievalStatus.ABSENT} else 1
    except (ValueError, TypeError, KeyError):
        stderr.write("codex-memory-hook: invalid retrieval input\n")
        return 2
    except Exception:
        stderr.write("codex-memory-hook: runtime unavailable\n")
        return 1


def _parse_query_arguments(tokens: list[str]) -> dict[str, object]:
    values: dict[str, object] = {
        "prompt_event_uri": None, "max_chars": 4000, "explain": False,
        "legacy": False, "disable_dense": False, "disable_sparse": False,
        "disable_reranker": False,
    }
    seen: set[str] = set()
    names = {"--session-id": "session_id", "--turn-id": "turn_id", "--context-id": "context_id", "--query": "query", "--prompt-event-uri": "prompt_event_uri", "--max-chars": "max_chars"}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--explain", "--legacy", "--disable-dense", "--disable-sparse", "--disable-reranker"}:
            key = token[2:].replace("-", "_")
            if values[key]:
                raise ValueError("duplicate option")
            values[key] = True; index += 1; continue
        name = names.get(token)
        if name is None or index + 1 >= len(tokens) or name in seen:
            raise ValueError("invalid option")
        value = tokens[index + 1]
        if not isinstance(value, str) or not value:
            raise ValueError("invalid value")
        values[name] = value; seen.add(name); index += 2
    if any(name not in values for name in ("session_id", "turn_id", "context_id", "query")):
        raise ValueError("missing option")
    try:
        values["max_chars"] = int(values["max_chars"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid max chars") from exc
    if not 1 <= values["max_chars"] <= 4000:
        raise ValueError("invalid max chars")
    return values


def _legacy_packet(paths: RetrievalPaths, request: RetrievalRequest) -> str:
    with open_verified_canonical_snapshot(paths) as (connection, _snapshot):
        return build_memory_packet(
            connection, BlobStore(paths.runtime_root / "blobs"), session_id=request.session_id,
            turn_id=request.turn_id, context_id=request.context_id,
            scope_key=request.scope.project_scope_key, prompt_event_uri=request.prompt_event_uri,
            query=request.query, max_chars=request.max_chars,
            workspace_scope_keys=request.scope.workspace_scope_keys,
        )


def _serialize_result(result: RetrievalResult) -> str:
    generation = result.generation
    document = {
        "status": result.status.value,
        "mode": result.mode.value,
        "schema_version": SCHEMA_VERSION if generation is not None else None,
        "active_generation": None if generation is None else generation.generation_id,
        "canonical_highwater": (
            None
            if result.canonical_highwater is None
            else result.canonical_highwater.to_dict()
        ),
        "indexed_highwater": (
            None
            if result.indexed_highwater is None
            else result.indexed_highwater.to_dict()
        ),
        "canonical_content_fingerprint": (
            None if generation is None else generation.canonical_content_fingerprint
        ),
        "provider_fingerprint": (
            None if generation is None else generation.provider_fingerprint
        ),
        "model_fingerprint": (
            None if generation is None else generation.model_fingerprint
        ),
        "validation_state": (
            None if generation is None else generation.validation_state
        ),
        "degraded_reason": (
            None if result.degraded_reason is None else result.degraded_reason.value
        ),
        "capabilities": [],
    }
    return canonical_json(document) + "\n"
