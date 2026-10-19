"""Explicit deterministic terminal recall; no host activation or provider calls."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import time
from typing import Callable, TextIO

from codex_memory.blob_store import BlobStore
from codex_memory.injection_manifest import InjectionAuditError
from codex_memory.injection_store import InjectionPaths, InjectionStore, _directories, _read
from codex_memory.ledger_common import canonical_json, parse_json_value
from codex_memory.memory_context import build_memory_packet
from codex_memory.memory_evidence import parse_event_uri, resolve_event_evidence
from codex_memory.project_memory_config import load_project_memory_config

from .canonical_snapshot import (
    CanonicalSnapshotInvalidError, CanonicalSnapshotUnavailableError,
    open_verified_canonical_snapshot,
)
from .contracts import RetrievalMode, RetrievalRequest, RetrievalScope
from .deadline import DeadlineBudget
from .injection_audit import _deadline_check
from .isolated_packet import IsolatedPacketFailure as _RecallFailure, prepare_isolated_packet
from .paths import RetrievalPaths


CLI_BUDGET_MS = 5000
LAB_FORMAT = "codex-memory.p26a-cli-lab.v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _metadata(stderr: TextIO, **values: object) -> bool:
    text = canonical_json({"host_acknowledgement": "unknown", **values}) + "\n"
    try:
        if stderr.write(text) != len(text):
            return False
        stderr.flush()
        return True
    except (OSError, UnicodeError):
        return False


def _write_wire(stdout: TextIO, raw: bytes) -> None:
    # Preserve exact UTF-8/LF bytes at the installed Windows entry; StringIO is
    # supported for real in-process command tests without transcoding the wire.
    if hasattr(stdout, "buffer"):
        stream = stdout.buffer
        written = stream.write(raw)
        if written != len(raw):
            raise OSError("incomplete output")
        stream.flush()
    else:
        text = raw.decode("utf-8")
        if stdout.write(text) != len(text):
            raise OSError("incomplete output")
        stdout.flush()


def _lab_binding(paths: RetrievalPaths, stack: ExitStack, deadline: DeadlineBudget) -> str:
    expected = {
        "format": LAB_FORMAT, "project_root": str(paths.repo_root),
        "data_origin": "synthetic", "provider_policy": "disabled", "hook_policy": "disabled",
    }
    try:
        held = stack.enter_context(_directories(paths.runtime_root, create=False, deadline=deadline))
        raw = _read(held, paths.runtime_root / "p26a-lab.json", limit=4096, deadline=deadline)
        value = parse_json_value(raw.decode("utf-8"))
        if value != expected or raw != canonical_json(expected).encode("utf-8"):
            raise ValueError("unapproved laboratory")
        # The marker is a deliberate lab opt-in, not permission to reuse a
        # deployed project's Hook definitions for native execution.
        hook_file = paths.repo_root / ".codex" / "hooks.json"
        try:
            hook_file.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("laboratory contains Hook registration")
        return hashlib.sha256(raw).hexdigest()
    except InjectionAuditError as exc:
        if exc.reason.value == "deadline_exhausted":
            raise
        raise _RecallFailure("lab_not_authorized", 2) from exc
    except Exception as exc:
        raise _RecallFailure("lab_not_authorized", 2) from exc


def _legacy_and_source(paths: RetrievalPaths, request: RetrievalRequest, *, deadline: DeadlineBudget | None = None):
    blobs = BlobStore(paths.runtime_root / "blobs")
    with open_verified_canonical_snapshot(paths, **({"deadline": deadline} if deadline is not None else {})) as (connection, snapshot):
        try:
            evidence = resolve_event_evidence(connection, blobs, request.prompt_event_uri)
            if (evidence.event_name != "UserPromptSubmit"
                    or evidence.session_id != request.session_id or evidence.turn_id != request.turn_id):
                raise ValueError("prompt source differs")
            _session, event_id = parse_event_uri(evidence.uri)
            row = connection.execute("SELECT payload_blob_sha256 FROM events WHERE event_id = ?", (event_id,)).fetchone()
            source = {"event_id": event_id, "event_uri": evidence.uri, "event_blob_sha256": row[0],
                "session_id": request.session_id, "turn_id": request.turn_id,
                "context_id": request.context_id, "origin": "offline"}
        except (sqlite3.Error, OSError):
            # Let the enclosing verified snapshot classify storage-read
            # failures; inability to read is not a source-identity mismatch.
            raise
        except Exception as exc:
            raise _RecallFailure("source_mismatch") from exc
        legacy = build_memory_packet(
            connection, blobs, session_id=request.session_id, turn_id=request.turn_id,
            context_id=request.context_id, scope_key=request.scope.project_scope_key,
            prompt_event_uri=request.prompt_event_uri, query=request.query,
            max_chars=request.max_chars, workspace_scope_keys=request.scope.workspace_scope_keys,
        )
        return legacy, source, snapshot


def _wire_document(packet: str, *, phase2: bool, selected_ids: list[str], reasons: list[str]) -> dict[str, object]:
    return {
        "format": "codex-memory.p26a-cli-packet.v1", "mode": "active_cli" if phase2 else "legacy",
        "status": "phase2" if phase2 else "legacy_fallback", "packet": packet,
        "packet_sha256": hashlib.sha256(packet.encode("utf-8")).hexdigest(),
        "packet_chars": len(packet), "selected_ids": selected_ids,
        "degraded_reasons": reasons, "provider_policy": "disabled",
    }


def run_isolated_recall(tokens: list[str], stdout: TextIO, stderr: TextIO,
                        resolve_repo_root: Callable[[Path], Path]) -> int:
    from .cli import _parse_query_arguments

    deadline = DeadlineBudget.start(CLI_BUDGET_MS, now_ns=time.monotonic_ns)
    started_at = _utc()
    committed = None
    wire_started = False
    wire_emitted = False
    try:
        if tokens.count("--isolated-cli") != 1:
            raise ValueError("explicit isolated flag required")
        values = _parse_query_arguments([item for item in tokens if item != "--isolated-cli"])
        if values["prompt_event_uri"] is None or any(values[key] for key in ("explain", "legacy", "disable_dense", "disable_sparse", "disable_reranker")):
            raise ValueError("unsupported isolated option")
        root = resolve_repo_root(Path.cwd())
        paths = RetrievalPaths.for_repo(root)
        _deadline_check(deadline)
        with ExitStack() as holds:
            marker_sha = _lab_binding(paths, holds, deadline)
            config = load_project_memory_config(paths.repo_root)
            request = RetrievalRequest(
                values["query"], values["session_id"], values["turn_id"], values["context_id"],
                values["prompt_event_uri"], RetrievalScope(config.project_key_for(paths.repo_root), config.workspace_scope_keys),
                mode=RetrievalMode.SHADOW, max_chars=values["max_chars"],
            )
            config_sha = hashlib.sha256(canonical_json({
                "lab_marker_sha256": marker_sha, "scope": request.scope.to_dict(),
                "provider_policy": "disabled", "output_mode": "active_cli",
            }).encode("utf-8")).hexdigest()
            _deadline_check(deadline)
            legacy, source, _initial = _legacy_and_source(paths, request)
            try:
                raw, committed = prepare_isolated_packet(paths, request=request,
                    legacy_packet=legacy, source=source, config_sha256=config_sha,
                    deadline=deadline, started_at=started_at, prepared_at=_utc,
                    budget_ms=CLI_BUDGET_MS, encode_wire=lambda packet, ids, reasons:
                        (canonical_json(_wire_document(packet, phase2=True,
                            selected_ids=ids, reasons=reasons)) + "\n").encode("utf-8"))
            except (CanonicalSnapshotInvalidError, CanonicalSnapshotUnavailableError):
                raise
            except Exception as error:
                if isinstance(error, _RecallFailure) and error.reason in {"canonical_invalid", "canonical_unavailable"}:
                    raise
                _deadline_check(deadline)
                # Do not serve the cached Legacy string after uncertainty:
                # re-establish truth and source for a fresh bounded fallback.
                legacy, _source, _snapshot = _legacy_and_source(paths, request)
                _deadline_check(deadline)
                reason = error.reason if isinstance(error, _RecallFailure) else "manifest_unavailable"
                raw = (canonical_json(_wire_document(legacy, phase2=False, selected_ids=[], reasons=[reason])) + "\n").encode("utf-8")
                committed = None
            try:
                # A write can accept bytes before raising or flushing fails.
                # After starting output, lack of a return cannot prove absence.
                wire_started = True
                _write_wire(stdout, raw)
                wire_emitted = True
            except OSError as exc:
                raise _RecallFailure("output_unavailable") from exc
        # Report once, after the protected scope has completed. Its exit can
        # fail after stdout emission; that must not erase the observed fact.
        if not _metadata(stderr, delivery="emitted" if committed is not None else "legacy_fallback",
            injection_id=None if committed is None else committed.injection_id,
            manifest_sha256=None if committed is None else committed.manifest_sha256):
            return 1
        return 0
    except CanonicalSnapshotInvalidError:
        reason, code = "canonical_invalid", 1
    except CanonicalSnapshotUnavailableError:
        reason, code = "canonical_unavailable", 1
    except _RecallFailure as error:
        reason, code = error.reason, error.code
    except InjectionAuditError as error:
        reason, code = error.reason.value, 1
    except (ValueError, TypeError, KeyError):
        reason, code = "invalid_input", 2
    except Exception:
        reason, code = "unavailable", 1
    if wire_emitted:
        delivery = "emitted" if committed is not None else "legacy_fallback"
    else:
        delivery = "delivery_unknown" if wire_started else "not_emitted"
    _metadata(stderr, delivery=delivery,
        injection_id=committed.injection_id if wire_started and committed is not None else None, reason=reason)
    return code
