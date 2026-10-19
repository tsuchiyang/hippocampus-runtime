"""Synchronous stdin/stdout entry point for Codex lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import ntpath
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable, Sequence, TextIO

_MODULE_ENTRY_NS = time.monotonic_ns()

from codex_memory.blob_store import BlobStore
from codex_memory.checkpoint import CheckpointService
from codex_memory.database import connect_database, migrate_database
from codex_memory.event_store import EventStore
from codex_memory.handlers import (
    handle_post_compact,
    handle_post_tool_use,
    handle_pre_compact,
    handle_user_prompt_submit,
    prepare_session_start,
)
from codex_memory.hook_input import (
    HookEnvelope, HookEvent, HookInputError, MAX_HOOK_INPUT_BYTES, parse_hook_payload,
)
from codex_memory.memory_context import build_memory_packet, current_context_id
from codex_memory.memory_evidence import event_uri
from codex_memory.project_memory_config import load_project_memory_config
from codex_memory.paths import RuntimePaths
from codex_memory.recovery import (
    SESSION_START_CONTEXT_LIMIT,
    session_start_output,
    user_prompt_submit_output,
)


_COMMAND_EVENTS = {
    "user-prompt-submit": HookEvent.USER_PROMPT_SUBMIT,
    "post-tool-use": HookEvent.POST_TOOL_USE,
    "pre-compact": HookEvent.PRE_COMPACT,
    "post-compact": HookEvent.POST_COMPACT,
    "session-start": HookEvent.SESSION_START,
}
_GIT_TIMEOUT_SECONDS = 10
_FAILURE_TEXT = (
    '{"continue":false,"stopReason":"Durable checkpoint unavailable",'
    '"systemMessage":"Codex Memory Runtime could not verify durable state."}\n'
)
_STRUCTURED_MEMORY_UNAVAILABLE_OUTPUT = (
    '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
    '"additionalContext":"structured memory unavailable"}}\n'
)
_USER_PROMPT_MEMORY_LIMIT = 4_000
_HOOK_PROCESS_BUDGET_MS = 1_500
_OPTIONAL_SHADOW_RETURN_RESERVE_MS = 200
_ACCEPTANCE_OBSERVER_RETURN_RESERVE_MS = 50
_NS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class _Runtime:
    connection: sqlite3.Connection
    events: EventStore
    checkpoints: CheckpointService
    paths: RuntimePaths
    blobs: BlobStore


def run_ledger_command(
    arguments: Sequence[str],
    stdout: TextIO,
    stderr: TextIO,
    resolve_repo_root: Callable[[Path], Path],
) -> int:
    """Load the ledger command handler only for its command routes."""
    from codex_memory.ledger_cli import run_ledger_command as run_command

    return run_command(arguments, stdout, stderr, resolve_repo_root)


def run_memory_command(
    arguments: Sequence[str],
    stdout: TextIO,
    stderr: TextIO,
    resolve_repo_root: Callable[[Path], Path],
) -> int:
    """Keep command-only initialization outside lifecycle Hook startup."""
    from codex_memory.memory_cli import run_memory_command as run_command

    return run_command(arguments, stdout, stderr, resolve_repo_root)


def main(
    argv: Sequence[str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    *,
    _process_started_ns: int | None = None,
) -> int:
    """Run exactly one hook command without leaking untrusted diagnostics."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    if arguments and arguments[0] == "retrieval":
        from codex_memory.retrieval.cli import run_retrieval_command

        return run_retrieval_command(
            arguments, output_stream, error_stream, _resolve_repo_root
        )
    if arguments and arguments[0] in {"record", "ledger"}:
        return run_ledger_command(
            arguments, output_stream, error_stream, _resolve_repo_root
        )
    if arguments and arguments[0] == "memory":
        return run_memory_command(
            arguments, output_stream, error_stream, _resolve_repo_root
        )

    isolated_prompt = arguments == ["user-prompt-submit", "--isolated-active"]
    selected_host = arguments == ["user-prompt-submit", "--selected-host"]
    selected_compact = (len(arguments) == 2 and arguments[1] == "--selected-compact"
        and arguments[0] in {"user-prompt-submit", "pre-compact", "post-compact", "session-start"})
    if (len(arguments) != 1 and not isolated_prompt and not selected_host and not selected_compact) or arguments[0] not in _COMMAND_EVENTS:
        error_stream.write("codex-memory-hook: unsupported command\n")
        return 2
    command = arguments[0]
    expected_event = _COMMAND_EVENTS[command]
    process_deadline_ns = None
    if command in {"user-prompt-submit", "session-start"} or selected_compact:
        if _process_started_ns is None:
            if (
                argv is None
                and stdin is None
                and stdout is None
                and stderr is None
            ):
                process_started_ns = (
                    _acceptance_process_started_ns()
                    if "CODEX_MEMORY_P25_OBSERVER" in os.environ
                    else _installed_process_started_ns()
                )
            else:
                process_started_ns = time.monotonic_ns()
        else:
            if (
                isinstance(_process_started_ns, bool)
                or not isinstance(_process_started_ns, int)
                or _process_started_ns < 0
            ):
                raise ValueError("_process_started_ns must be a non-negative integer")
            process_started_ns = _process_started_ns
        process_deadline_ns = (
            0
            if process_started_ns is None
            else process_started_ns + _HOOK_PROCESS_BUDGET_MS * _NS_PER_MS
        )

    try:
        payload = _parse_json_object(_read_hook_input(input_stream))
        envelope = parse_hook_payload(payload)
    except Exception:
        error_stream.write("codex-memory-hook: invalid hook input\n")
        return 2
    if envelope.event is not expected_event:
        error_stream.write("codex-memory-hook: hook event mismatch\n")
        return 2

    if isolated_prompt:
        from codex_memory.retrieval.isolated_hook import run_isolated_user_prompt

        return run_isolated_user_prompt(envelope, output_stream, error_stream,
            process_deadline_ns=process_deadline_ns,
            resolve_repo_root=lambda cwd, cutoff: _resolve_repo_root(cwd, deadline_ns=cutoff),
            runtime_factory=lambda root, budget: _construct_runtime(root, deadline=budget))

    if selected_host:
        from codex_memory.retrieval.selected_host import run_selected_host_user_prompt

        return run_selected_host_user_prompt(envelope, output_stream, error_stream,
            process_deadline_ns=process_deadline_ns,
            resolve_repo_root=lambda cwd, cutoff: _resolve_repo_root(cwd, deadline_ns=cutoff),
            runtime_factory=lambda root, budget: _construct_runtime(root, deadline=budget))

    if selected_compact:
        from codex_memory.retrieval.selected_host import run_selected_compact

        return run_selected_compact(command, envelope, output_stream, error_stream,
            process_deadline_ns=process_deadline_ns,
            resolve_repo_root=lambda cwd, cutoff: _resolve_repo_root(cwd, deadline_ns=cutoff),
            runtime_factory=lambda root, budget: _construct_runtime(root, deadline=budget),
            dispatch=_dispatch)

    try:
        repo_root = _resolve_repo_root(envelope.cwd)
        runtime = _construct_runtime(repo_root)
    except Exception:
        error_stream.write("codex-memory-hook: runtime unavailable\n")
        return 1

    try:
        return _dispatch(
            command,
            envelope,
            runtime,
            output_stream,
            error_stream,
            process_deadline_ns=process_deadline_ns,
        )
    finally:
        try:
            runtime.connection.close()
        except Exception:
            pass


def _dispatch(
    command: str,
    envelope: HookEnvelope,
    runtime: _Runtime,
    stdout: TextIO,
    stderr: TextIO,
    *,
    process_deadline_ns: int | None = None,
) -> int:
    if command == "pre-compact":
        try:
            response = handle_pre_compact(envelope, runtime.checkpoints)
            serialized = _serialize_response(response)
        except Exception:
            stdout.write(_FAILURE_TEXT)
            return 0
        stdout.write(serialized)
        return 0

    if command == "session-start":
        try:
            recovery = prepare_session_start(
                envelope, runtime.events, runtime.checkpoints
            )
            if recovery is None:
                return 0
            base_serialized = _serialize_response(recovery.response)
        except Exception:
            stdout.write(_FAILURE_TEXT)
            return 0

        serialized = base_serialized
        shadow_context_id: str | None = None
        try:
            base_packet = _additional_context(recovery.response)
            remaining = SESSION_START_CONTEXT_LIMIT - len(base_packet) - 1
            if remaining > 0:
                memory_limit = min(_USER_PROMPT_MEMORY_LIMIT, remaining)
                # This verified recovery defines the new window even though
                # delivery is not settled until the complete output is flushed.
                shadow_context_id = recovery.checkpoint_id
                packet = build_memory_packet(
                    runtime.connection,
                    runtime.blobs,
                    session_id=recovery.session_id,
                    turn_id=recovery.turn_id,
                    context_id=shadow_context_id,
                    prompt_event_uri=None,
                    query=None,
                    max_chars=memory_limit,
                    **_memory_scope_kwargs(
                        runtime.paths.repo_root, diagnostic_stream=stderr
                    ),
                )
                combined = f"{base_packet}\n{packet}"
                if (
                    packet
                    and len(packet) <= memory_limit
                    and len(combined) <= SESSION_START_CONTEXT_LIMIT
                ):
                    serialized = _serialize_response(session_start_output(combined))
        except Exception:
            serialized = base_serialized
        try:
            written = stdout.write(serialized)
            if written != len(serialized):
                raise OSError("incomplete recovery output")
            stdout.flush()
        except Exception:
            # Partial or unflushed output is not a delivery. Never append a
            # second JSON frame to a stream whose state is now unknown.
            stderr.write("codex-memory-hook: recovery output failed\n")
            return 1
        try:
            # Local transport completion, not an acknowledgement from the host
            # or model. A failed settlement leaves recovery retryable.
            runtime.checkpoints.mark_recovery_delivered(
                recovery.checkpoint_id, recovery.session_id
            )
        except Exception:
            stderr.write("codex-memory-hook: recovery settlement failed\n")
            return 1
        if shadow_context_id is not None:
            _attempt_shadow_after_legacy(
                runtime=runtime,
                session_id=recovery.session_id,
                turn_id=recovery.turn_id,
                context_id=shadow_context_id,
                prompt_event_uri=None,
                query=None,
                hook_event="SessionStart",
                process_deadline_ns=process_deadline_ns,
            )
        return 0

    if command == "user-prompt-submit":
        try:
            record = handle_user_prompt_submit(envelope, runtime.events)
        except Exception:
            stderr.write("codex-memory-hook: hook execution failed\n")
            return 1
        try:
            shadow_context_id = current_context_id(
                runtime.connection, session_id=envelope.session_id
            )
            packet = build_memory_packet(
                runtime.connection,
                runtime.blobs,
                session_id=envelope.session_id,
                turn_id=envelope.turn_id or "",
                context_id=shadow_context_id,
                prompt_event_uri=event_uri(envelope.session_id, record.event_id),
                query=envelope.prompt,
                max_chars=_USER_PROMPT_MEMORY_LIMIT,
                **_memory_scope_kwargs(
                    runtime.paths.repo_root, diagnostic_stream=stderr
                ),
            )
            serialized = _serialize_response(user_prompt_submit_output(packet))
        except Exception:
            stdout.write(_STRUCTURED_MEMORY_UNAVAILABLE_OUTPUT)
            return 0
        write_count = stdout.write(serialized)
        _attempt_shadow_after_legacy(
            runtime=runtime,
            session_id=envelope.session_id,
            turn_id=envelope.turn_id or "",
            context_id=shadow_context_id,
            prompt_event_uri=event_uri(envelope.session_id, record.event_id),
            query=envelope.prompt,
            hook_event="UserPromptSubmit",
            source_record=record,
            source_payload=envelope.raw,
            legacy_packet=packet,
            legacy_response=serialized,
            write_count=write_count,
            process_deadline_ns=process_deadline_ns,
        )
        return 0

    try:
        if command == "post-tool-use":
            handle_post_tool_use(envelope, runtime.events)
            return 0
        if command == "post-compact":
            response = handle_post_compact(envelope, runtime.checkpoints)
            stdout.write(_serialize_response(response))
            return 0
        raise AssertionError("validated command was not dispatched")
    except Exception:
        stderr.write("codex-memory-hook: hook execution failed\n")
        return 1


def _read_hook_input(stream: TextIO) -> str:
    """Bound wire bytes before TextIO newline/encoding translation when available."""
    binary = getattr(stream, "buffer", None)
    if binary is not None:
        raw = binary.read(MAX_HOOK_INPUT_BYTES + 1)
        if len(raw) > MAX_HOOK_INPUT_BYTES:
            raise HookInputError("hook input exceeds byte limit")
        return raw.decode("utf-8")
    # In-memory TextIO injection has no original wire representation. Its
    # decoded UTF-8 representation is still bounded and rechecked by parsing.
    text = stream.read(MAX_HOOK_INPUT_BYTES + 1)
    if len(text.encode("utf-8")) > MAX_HOOK_INPUT_BYTES:
        raise HookInputError("hook input exceeds byte limit")
    return text


def _parse_json_object(raw: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError("non-standard JSON constant")

    def parse_finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite JSON number")
        return number

    start = len(raw) - len(raw.lstrip())
    if start == len(raw):
        raise ValueError("empty input")
    decoder = json.JSONDecoder(
        parse_constant=reject_constant,
        parse_float=parse_finite_float,
    )
    value, end = decoder.raw_decode(raw, start)
    if not isinstance(value, dict) or raw[end:].strip():
        raise ValueError("input must contain exactly one JSON object")
    return value


def _resolve_repo_root(cwd: Path, *, deadline_ns: int | None = None) -> Path:
    command = ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"]
    timeout = _GIT_TIMEOUT_SECONDS
    if deadline_ns is not None:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            from codex_memory.injection_manifest import InjectionAuditError
            raise InjectionAuditError("deadline_exhausted")
        timeout = min(timeout, remaining_ns / 1_000_000_000)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            shell=False,
            text=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("repository root lookup failed") from exc
    if result.returncode != 0:
        raise RuntimeError("repository root lookup failed")
    encoded_root = result.stdout.rstrip(b"\r\n")
    if not encoded_root or b"\0" in encoded_root:
        raise RuntimeError("repository root lookup failed")
    root = Path(encoded_root.decode("utf-8", errors="surrogateescape"))
    return root.resolve()


def _construct_runtime(repo_root: Path, *, deadline: Any = None) -> _Runtime:
    if deadline is not None:
        from codex_memory.retrieval.injection_audit import _deadline_check
        _deadline_check(deadline)
    paths = RuntimePaths.for_repo(repo_root)
    paths.ensure()
    if deadline is None:
        connection = connect_database(paths.db_path)
    else:
        _deadline_check(deadline)
        remaining_ms = max(0, min(5000, (deadline.deadline_ns - time.monotonic_ns()) // 1_000_000))
        connection = connect_database(paths.db_path, busy_timeout_ms=remaining_ms)
    try:
        if deadline is not None:
            connection.set_progress_handler(lambda: int(time.monotonic_ns() >= deadline.deadline_ns), 100)
            _deadline_check(deadline)
            # A current-schema Hook still acquires a write transaction. A
            # competing legal writer may finish within THIS remaining budget;
            # zero lock wait would reject it irrespective of available time.
            remaining_ms = max(0, min(5000, (deadline.deadline_ns - time.monotonic_ns()) // 1_000_000))
            connection.execute(f"PRAGMA busy_timeout={remaining_ms}")
        migrate_database(connection)
        if deadline is not None:
            _deadline_check(deadline)
            connection.execute("PRAGMA busy_timeout=0")
        blobs = BlobStore(paths.blobs_dir)
        events = EventStore(connection, blobs)
        checkpoints = CheckpointService(paths, events, blobs)
        return _Runtime(connection, events, checkpoints, paths, blobs)
    except BaseException:
        connection.close()
        raise
    finally:
        if deadline is not None:
            try:
                connection.set_progress_handler(None, 0)
            except sqlite3.ProgrammingError:
                pass


def _memory_scope_kwargs(
    repo_root: Path, *, diagnostic_stream: TextIO | None = None
) -> dict[str, object]:
    """Return optional scope configuration without affecting base recovery."""
    config = load_project_memory_config(repo_root)
    if config.diagnostic is not None and diagnostic_stream is not None:
        diagnostic_stream.write(config.diagnostic)
    result = {"scope_key": config.project_key_for(repo_root)}
    if config.workspace_scope_keys:
        result["workspace_scope_keys"] = config.workspace_scope_keys
    return result


def _run_shadow_after_legacy(
    *,
    runtime: _Runtime,
    session_id: str,
    turn_id: str,
    context_id: str,
    prompt_event_uri: str | None,
    query: str | None,
    hook_event: str,
    audit_context: dict[str, object] | None = None,
    observer_legacy: dict[str, object] | None = None,
    process_deadline_ns: int | None = None,
) -> None:
    """Best-effort P2.2 audit that cannot alter a completed Hook response."""
    observer = None
    try:
        observer_requested = "CODEX_MEMORY_P25_OBSERVER" in os.environ
        optional_deadline_ns = _optional_shadow_deadline_ns(
            process_deadline_ns,
            return_reserve_ms=(
                _ACCEPTANCE_OBSERVER_RETURN_RESERVE_MS
                if observer_requested
                else _OPTIONAL_SHADOW_RETURN_RESERVE_MS
            ),
        )
        if (
            optional_deadline_ns is not None
            and time.monotonic_ns() >= optional_deadline_ns
        ):
            return
        from codex_memory.retrieval.config import (
            load_retrieval_config,
            provider_from_config,
        )

        config = load_retrieval_config(runtime.paths.repo_root)
        if not config.enabled:
            return
        memory_config = load_project_memory_config(runtime.paths.repo_root)
        from codex_memory.retrieval.contracts import (
            RetrievalMode,
            RetrievalRequest,
            RetrievalScope,
        )
        from codex_memory.retrieval.paths import RetrievalPaths
        from codex_memory.retrieval.shadow import compare

        if not turn_id:
            return
        request = RetrievalRequest(
            query=query,
            session_id=session_id,
            turn_id=turn_id,
            context_id=context_id,
            prompt_event_uri=prompt_event_uri,
            scope=RetrievalScope(
                memory_config.project_key_for(runtime.paths.repo_root),
                memory_config.workspace_scope_keys,
            ),
            mode=RetrievalMode.SHADOW,
            max_chars=_USER_PROMPT_MEMORY_LIMIT,
        )
        audit = config.audit_for_session(session_id) if hook_event == "UserPromptSubmit" else None
        deadline_ms = config.semantic.query_deadline_ms if config.semantic_enabled and config.semantic is not None else None
        capture_context = None
        if audit is not None and audit_context is not None:
            deadline_ms = min(deadline_ms, audit.total_deadline_ms) if deadline_ms is not None else audit.total_deadline_ms
            capture_context = {**audit_context, "config_sha256": config.config_sha256}
        paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
        if (
            optional_deadline_ns is not None
            and time.monotonic_ns() >= optional_deadline_ns
        ):
            return
        observer_kwargs = {}
        if observer_requested:
            # Activation is acceptance-only and requires the existing 1000ms
            # audit configuration and a completed UserPromptSubmit source.
            if capture_context is None or deadline_ms != 1000 or observer_legacy is None:
                return
            from codex_memory.retrieval.deadline import DeadlineBudget
            observer_now_ns = time.monotonic_ns()
            if optional_deadline_ns is None:
                observer_started_ns = observer_now_ns
            else:
                if observer_now_ns >= optional_deadline_ns:
                    return
                # Preallocate the unchanged 1000ms acceptance window against
                # the installed-process deadline.  If startup/Legacy enters
                # this window late, that elapsed time is charged rather than
                # silently abandoning the explicitly requested Observer.
                latest_start_ns = (
                    optional_deadline_ns - deadline_ms * _NS_PER_MS
                )
                if latest_start_ns < 0:
                    return
                observer_started_ns = min(observer_now_ns, latest_start_ns)
            budget = DeadlineBudget(
                observer_started_ns,
                observer_started_ns + deadline_ms * _NS_PER_MS,
            )
            from codex_memory.acceptance_observer import Observer
            observer = Observer.from_environment(paths=paths, request=request, source=capture_context["source"], deadline=budget)
            observer.emit("legacy", observer_legacy)
            observer_kwargs = {"observer": observer, "deadline": budget}
        compare(
            paths=paths,
            request=request,
            hook_event=hook_event,
            semantic_provider_factory=(
                (lambda: provider_from_config(config.semantic))
                if config.semantic_enabled
                else None
            ),
            semantic_config=config.semantic if config.semantic_enabled else None,
            deadline_ms=deadline_ms,
            **({"audit_context": capture_context} if capture_context is not None else {}),
            **(
                {"optional_deadline_ns": optional_deadline_ns}
                if optional_deadline_ns is not None and observer is None
                else {}
            ),
            **observer_kwargs,
        )
    except Exception:
        # The legacy response is already complete; shadow is deliberately inert.
        return
    finally:
        if observer is not None:
            observer.close()


def _attempt_shadow_after_legacy(
    *,
    runtime: _Runtime,
    session_id: str,
    turn_id: str,
    context_id: str,
    prompt_event_uri: str | None,
    query: str | None,
    hook_event: str,
    source_record: object | None = None,
    source_payload: dict[str, object] | None = None,
    legacy_packet: str | None = None,
    legacy_response: str | None = None,
    write_count: object = None,
    process_deadline_ns: int | None = None,
) -> None:
    """Contain an unexpected shadow-boundary failure after legacy stdout."""
    try:
        context = None
        if source_record is not None and hook_event == "UserPromptSubmit" and legacy_response is not None:
            if source_record.event is not HookEvent.USER_PROMPT_SUBMIT:
                return
            delivery = "unknown"
            if type(write_count) is int:
                if write_count == len(legacy_response):
                    delivery = "emitted"
                elif 0 <= write_count < len(legacy_response):
                    delivery = "partial"
            context = {
                "source": {
                    "event_id": source_record.event_id,
                    "event_uri": event_uri(source_record.session_id, source_record.event_id),
                    "event_blob_sha256": source_record.payload_blob_sha256,
                    "session_id": source_record.session_id,
                    "turn_id": source_record.turn_id,
                    "context_id": context_id,
                    "origin": "hook",
                },
                "source_payload": source_payload,
                "legacy_packet": legacy_packet,
                "legacy_response_bytes": legacy_response.encode("utf-8"),
                "legacy_delivery": delivery,
            }
        observer_kwargs = {}
        if "CODEX_MEMORY_P25_OBSERVER" in os.environ and context is not None:
            import hashlib
            observer_kwargs["observer_legacy"] = {
                "delivery": context["legacy_delivery"], "write_count": write_count,
                "serialized_sha256": hashlib.sha256(context["legacy_response_bytes"]).hexdigest(),
                "core_in_transaction": runtime.connection.in_transaction,
            }
        _run_shadow_after_legacy(
            runtime=runtime,
            session_id=session_id,
            turn_id=turn_id,
            context_id=context_id,
            prompt_event_uri=prompt_event_uri,
            query=query,
            hook_event=hook_event,
            process_deadline_ns=process_deadline_ns,
            **({"audit_context": context} if context is not None else {}),
            **observer_kwargs,
        )
    except Exception:
        return


def _optional_shadow_deadline_ns(
    process_deadline_ns: int | None,
    *,
    return_reserve_ms: int,
) -> int | None:
    """Reserve bounded time for health persistence, connection close, and exit."""
    if process_deadline_ns is None:
        return None
    if (
        isinstance(process_deadline_ns, bool)
        or not isinstance(process_deadline_ns, int)
        or process_deadline_ns < 0
    ):
        return 0
    return max(
        0,
        process_deadline_ns - return_reserve_ms * _NS_PER_MS,
    )


def _installed_process_started_ns() -> int | None:
    """Bind the Hook budget to its trusted installed entry when observable."""
    if os.name == "nt":
        observed = _windows_process_started_ns()
        if observed is None or observed > _MODULE_ENTRY_NS:
            return None
        return observed
    return _MODULE_ENTRY_NS


def _acceptance_process_started_ns() -> int | None:
    """Read only the conservative parent floor from an Observer activation.

    The complete activation, ancestry, capability, and source identity remain
    the Observer's responsibility.  A valid parent floor precedes this
    module's entry and therefore only shortens optional work. Arbitrary data
    cannot certify a run because the complete activation later fails closed.
    """
    raw = os.environ.get("CODEX_MEMORY_P25_OBSERVER")
    if raw is None:
        return None
    try:
        value = _parse_json_object(raw)
        started_ns = value.get("process_started_ns")
    except Exception:
        return None
    if (
        isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or started_ns < 0
        or started_ns > _MODULE_ENTRY_NS
    ):
        return None
    return started_ns


def _windows_process_started_ns() -> int | None:
    """Map the trusted installed-entry process start onto the monotonic clock."""
    try:
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [
                ("low", ctypes.c_uint32),
                ("high", ctypes.c_uint32),
            ]

        class ProcessEntry(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("usage", wintypes.DWORD),
                ("pid", wintypes.DWORD),
                ("heap", ctypes.c_size_t),
                ("module", wintypes.DWORD),
                ("threads", wintypes.DWORD),
                ("parent", wintypes.DWORD),
                ("priority", wintypes.LONG),
                ("flags", wintypes.DWORD),
                ("exe", wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        filetime_pointer = ctypes.POINTER(FileTime)
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            filetime_pointer,
            filetime_pointer,
            filetime_pointer,
            filetime_pointer,
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessEntry),
        ]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessEntry),
        ]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        current_process = kernel32.GetCurrentProcess()
        creation = FileTime()
        exit_time = FileTime()
        kernel_time = FileTime()
        user_time = FileTime()
        if not kernel32.GetProcessTimes(
            current_process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None

        creation_ticks = (creation.high << 32) | creation.low
        parent_executable = None
        parent_creation_ticks = None
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if snapshot not in (None, 0, invalid_handle):
            try:
                entry = ProcessEntry()
                entry.size = ctypes.sizeof(entry)
                found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
                parent_pid = None
                current_pid = os.getpid()
                while found:
                    if entry.pid == current_pid:
                        parent_pid = int(entry.parent)
                        break
                    found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
                if parent_pid is not None and parent_pid > 0:
                    parent_process = kernel32.OpenProcess(0x1000, False, parent_pid)
                    if parent_process not in (None, 0, invalid_handle):
                        try:
                            image_capacity = wintypes.DWORD(32_768)
                            image = ctypes.create_unicode_buffer(image_capacity.value)
                            parent_creation = FileTime()
                            parent_exit = FileTime()
                            parent_kernel = FileTime()
                            parent_user = FileTime()
                            if kernel32.QueryFullProcessImageNameW(
                                parent_process,
                                0,
                                image,
                                ctypes.byref(image_capacity),
                            ) and kernel32.GetProcessTimes(
                                parent_process,
                                ctypes.byref(parent_creation),
                                ctypes.byref(parent_exit),
                                ctypes.byref(parent_kernel),
                                ctypes.byref(parent_user),
                            ):
                                parent_executable = image.value
                                parent_creation_ticks = (
                                    parent_creation.high << 32
                                ) | parent_creation.low
                        finally:
                            kernel32.CloseHandle(parent_process)
            finally:
                kernel32.CloseHandle(snapshot)

        try:
            get_system_time = kernel32.GetSystemTimePreciseAsFileTime
        except AttributeError:
            return None
        get_system_time.argtypes = [filetime_pointer]
        get_system_time.restype = None
        system_time = FileTime()
        # Bracket only the UTC sample. Process/ancestry lookup has already
        # elapsed and is included in FILETIME age; including it in this
        # uncertainty window would charge the same time a second time.
        monotonic_before_ns = time.monotonic_ns()
        get_system_time(ctypes.byref(system_time))
        monotonic_after_ns = time.monotonic_ns()

        creation_ticks = _select_installed_entry_creation_100ns(
            child_creation_100ns=creation_ticks,
            python_executable=sys.executable,
            parent_executable=parent_executable,
            parent_creation_100ns=parent_creation_ticks,
        )
        system_ticks = (system_time.high << 32) | system_time.low
        return _map_process_creation_to_monotonic(
            creation_100ns=creation_ticks,
            current_100ns=system_ticks,
            monotonic_before_ns=monotonic_before_ns,
            monotonic_after_ns=monotonic_after_ns,
            module_entry_ns=_MODULE_ENTRY_NS,
        )
    except Exception:
        # Timing discovery is optional.  Any ABI/platform failure disables
        # shadow work rather than escaping into the Phase 0/1B lifecycle.
        return None


def _select_installed_entry_creation_100ns(
    *,
    child_creation_100ns: int,
    python_executable: str,
    parent_executable: str | None,
    parent_creation_100ns: int | None,
) -> int:
    """Adopt only the earlier direct sibling console launcher's FILETIME."""
    if (
        type(parent_creation_100ns) is not int
        or parent_creation_100ns <= 0
        or parent_creation_100ns >= child_creation_100ns
        or not isinstance(parent_executable, str)
        or not parent_executable
        or not isinstance(python_executable, str)
        or not ntpath.isabs(parent_executable)
        or not ntpath.isabs(python_executable)
    ):
        return child_creation_100ns
    expected_launcher = ntpath.join(
        ntpath.dirname(python_executable),
        "codex-memory-hook.exe",
    )
    if ntpath.normcase(ntpath.normpath(parent_executable)) != ntpath.normcase(
        ntpath.normpath(expected_launcher)
    ):
        return child_creation_100ns
    return parent_creation_100ns


def _map_process_creation_to_monotonic(
    *,
    creation_100ns: int,
    current_100ns: int,
    monotonic_before_ns: int,
    monotonic_after_ns: int,
    module_entry_ns: int,
) -> int | None:
    """Conservatively map equal-epoch FILETIMEs onto a monotonic clock."""
    values = (
        creation_100ns,
        current_100ns,
        monotonic_before_ns,
        monotonic_after_ns,
        module_entry_ns,
    )
    if any(type(value) is not int or value < 0 for value in values):
        return None
    if (
        creation_100ns == 0
        or current_100ns < creation_100ns
        or monotonic_after_ns < monotonic_before_ns
    ):
        return None
    age_ns = (current_100ns - creation_100ns) * 100
    if age_ns > monotonic_before_ns:
        return None
    candidate_ns = monotonic_before_ns - age_ns
    if candidate_ns > module_entry_ns:
        return None
    return candidate_ns


def _serialize_response(response: object) -> str:
    return json.dumps(
        response, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) + "\n"


def _additional_context(response: dict[str, object]) -> str:
    hook_output = response.get("hookSpecificOutput")
    if not isinstance(hook_output, dict):
        raise ValueError("invalid hook response")
    packet = hook_output.get("additionalContext")
    if not isinstance(packet, str):
        raise ValueError("invalid hook response")
    return packet


if __name__ == "__main__":
    raise SystemExit(main())
