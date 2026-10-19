"""Explicit synthetic UserPromptSubmit command, never a host Hook rollout."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, TextIO

from codex_memory.handlers import handle_user_prompt_submit
from codex_memory.hook_input import HookEnvelope
from codex_memory.injection_manifest import InjectionAuditError
from codex_memory.injection_store import _directories, _read
from codex_memory.ledger_common import canonical_json, parse_json_value
from codex_memory.memory_context import current_context_id
from codex_memory.memory_evidence import event_uri
from codex_memory.project_memory_config import load_project_memory_config

from .canonical_snapshot import (CanonicalSnapshotInvalidError, CanonicalSnapshotUnavailableError,
    open_verified_canonical_snapshot)
from .contracts import RetrievalMode, RetrievalRequest, RetrievalScope
from .deadline import DeadlineBudget
from .injection_audit import _deadline_check
from .isolated_cli import _legacy_and_source, _metadata, _utc, _write_wire
from .isolated_packet import IsolatedPacketFailure, prepare_isolated_packet
from .paths import RetrievalPaths


PROMPT_BUDGET_MS = 1000


def _binding(paths: RetrievalPaths, session_id: str, holds: ExitStack, deadline: DeadlineBudget,
             *, release_deadline: DeadlineBudget) -> str:
    expected = {"format": "codex-memory.p26b-prompt-lab.v1", "project_root": str(paths.repo_root),
        "data_origin": "synthetic", "provider_policy": "disabled", "hook_policy": "isolated_command",
        "session_id": session_id}
    try:
        held = holds.enter_context(_directories(paths.runtime_root, create=False,
            deadline=deadline, release_deadline=release_deadline))
        raw = _read(held, paths.runtime_root / "p26b-lab.json", limit=4096, deadline=deadline)
        value = parse_json_value(raw.decode("utf-8"))
        if value != expected or raw != canonical_json(expected).encode("utf-8"):
            raise ValueError("unapproved laboratory")
        try:
            (paths.repo_root / ".codex" / "hooks.json").lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("registered Hook is not a command-only laboratory")
        return hashlib.sha256(raw).hexdigest()
    except InjectionAuditError as exc:
        if exc.reason.value == "deadline_exhausted":
            raise
        raise IsolatedPacketFailure("lab_not_authorized", 2) from exc
    except Exception as exc:
        raise IsolatedPacketFailure("lab_not_authorized", 2) from exc


def _wire(packet: str, _ids=None, _reasons=None) -> bytes:
    from codex_memory.cli import _serialize_response
    from codex_memory.recovery import user_prompt_submit_output
    return _serialize_response(user_prompt_submit_output(packet)).encode("utf-8")


def run_isolated_user_prompt(
    envelope: HookEnvelope, stdout: TextIO, stderr: TextIO, *,
    process_deadline_ns: int | None, resolve_repo_root: Callable[[Path, int], Path],
    runtime_factory: Callable[[Path, DeadlineBudget], Any],
) -> int:
    return _run_user_prompt(envelope, stdout, stderr,
        process_deadline_ns=process_deadline_ns, resolve_repo_root=resolve_repo_root,
        runtime_factory=runtime_factory, binding_validator=_binding,
        source_origin="offline", output_mode="isolated_prompt")


def _run_user_prompt(
    envelope: HookEnvelope, stdout: TextIO, stderr: TextIO, *,
    process_deadline_ns: int | None, resolve_repo_root: Callable[[Path, int], Path],
    runtime_factory: Callable[[Path, DeadlineBudget], Any], binding_validator: Callable,
    source_origin: str, output_mode: str,
) -> int:
    """Shared output/cleanup contract; each explicit ingress owns its binding."""
    committed = None
    wire_started = wire_emitted = False
    runtime = None
    reason = None
    code = 0
    fallback_deadline = None
    cleanup_status = "not_started"
    try:
        if process_deadline_ns is None or process_deadline_ns < 1_500_000_000:
            raise InjectionAuditError("deadline_exhausted")
        started_ns = process_deadline_ns - 1_500_000_000
        deadline = DeadlineBudget(started_ns, started_ns + PROMPT_BUDGET_MS * 1_000_000)
        fallback_deadline = DeadlineBudget(started_ns, process_deadline_ns - 50_000_000)
        started_at = _utc()
        _deadline_check(deadline)
        root = resolve_repo_root(envelope.cwd, deadline.deadline_ns)
        paths = RetrievalPaths.for_repo(root)
        _deadline_check(deadline)
        with ExitStack() as holds:
            marker_sha = binding_validator(paths, envelope.session_id, holds, deadline,
                release_deadline=fallback_deadline)
            with open_verified_canonical_snapshot(paths, deadline=deadline):
                _deadline_check(deadline)
            runtime = runtime_factory(root, deadline)
            _deadline_check(deadline)
            # The required append is a real canonical write, not an optional
            # read. Let a transient writer finish only within the originating
            # absolute budget; all later validation reads retain zero lock wait.
            remaining_ms = max(0, min(5000, (deadline.deadline_ns - time.monotonic_ns()) // 1_000_000))
            runtime.connection.execute(f"PRAGMA busy_timeout={remaining_ms}")
            try:
                record = handle_user_prompt_submit(envelope, runtime.events)
            finally:
                runtime.connection.execute("PRAGMA busy_timeout=0")
            _deadline_check(deadline)
            config = load_project_memory_config(root)
            request = RetrievalRequest(envelope.prompt, envelope.session_id, envelope.turn_id or "",
                current_context_id(runtime.connection, session_id=envelope.session_id),
                event_uri(envelope.session_id, record.event_id),
                RetrievalScope(config.project_key_for(root), config.workspace_scope_keys),
                mode=RetrievalMode.SHADOW, max_chars=4000)
            config_sha = hashlib.sha256(canonical_json({"lab_marker_sha256": marker_sha,
                "scope": request.scope.to_dict(), "provider_policy": "disabled",
                "output_mode": output_mode}).encode("utf-8")).hexdigest()
            legacy, source, _initial = _legacy_and_source(paths, request, deadline=deadline)
            source = {**source, "origin": source_origin}
            _deadline_check(deadline)
            try:
                raw, committed = prepare_isolated_packet(paths, request=request,
                    legacy_packet=legacy, source=source, config_sha256=config_sha,
                    deadline=deadline, started_at=started_at, prepared_at=_utc,
                    budget_ms=PROMPT_BUDGET_MS, encode_wire=_wire, bound_reads=True)
            except (CanonicalSnapshotInvalidError, CanonicalSnapshotUnavailableError):
                raise
            except Exception as error:
                if isinstance(error, IsolatedPacketFailure) and error.reason in {"canonical_invalid", "canonical_unavailable"}:
                    raise
                _deadline_check(fallback_deadline)
                legacy, _source, _snapshot = _legacy_and_source(paths, request, deadline=fallback_deadline)
                _deadline_check(fallback_deadline)
                raw = _wire(legacy)
                reason = (error.reason if isinstance(error, IsolatedPacketFailure)
                    else error.reason.value if isinstance(error, InjectionAuditError) else "manifest_unavailable")
                committed = None
            _deadline_check(fallback_deadline)
            wire_started = True
            try:
                _write_wire(stdout, raw)
                wire_emitted = True
            except (OSError, UnicodeError) as exc:
                raise IsolatedPacketFailure("output_unavailable") from exc
    except CanonicalSnapshotInvalidError:
        reason, code = "canonical_invalid", 1
    except CanonicalSnapshotUnavailableError:
        reason, code = "canonical_unavailable", 1
    except IsolatedPacketFailure as error:
        reason, code = error.reason, error.code
    except InjectionAuditError as error:
        reason, code = error.reason.value, 1
    except Exception:
        reason, code = "runtime_unavailable", 1
    finally:
        if runtime is not None:
            try:
                runtime.connection.close()
                cleanup_status = "complete"
            except Exception:
                cleanup_status = "unavailable"
                if code == 0:
                    reason = "runtime_close_unavailable"
                code = 1
    if code == 0 and fallback_deadline is not None:
        try:
            _deadline_check(fallback_deadline)
        except InjectionAuditError:
            reason, code = "deadline_exhausted", 1
    delivery = ("emitted" if committed is not None else "legacy_fallback") if wire_emitted else (
        "delivery_unknown" if wire_started else "not_emitted")
    values = {"delivery": delivery, "output_mode": output_mode, "source_origin": source_origin,
        "injection_id": committed.injection_id if wire_started and committed is not None else None,
        "manifest_sha256": committed.manifest_sha256 if wire_started and committed is not None else None,
        "cleanup_status": cleanup_status}
    if reason is not None:
        values["reason"] = reason
    if not _metadata(stderr, **values):
        return 1
    return code
