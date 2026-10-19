"""Explicit one-session synthetic host ingress, not general active rollout."""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
import hashlib
import os
from pathlib import Path
import re
import sys

from codex_memory.injection_manifest import InjectionAuditError
from codex_memory.injection_store import _directories, _read
from codex_memory.ledger_common import canonical_json, parse_json_value

from .deadline import DeadlineBudget
from .isolated_hook import _run_user_prompt
from .isolated_packet import IsolatedPacketFailure
from .paths import RetrievalPaths


_FIELDS = frozenset({"format", "project_root", "data_origin", "provider_policy",
    "hook_policy", "session_id", "codex_home", "runtime_entry",
    "runtime_entry_sha256", "hooks_sha256"})


def hook_definition(entry: Path, *, compact_cycle: bool = False) -> dict:
    """Exact selected ingress; Windows uses a literal PowerShell invocation."""
    entry_text = str(entry)
    windows_path = entry_text.replace("'", "''")
    flag = "--selected-compact" if compact_cycle else "--selected-host"
    routes = [("UserPromptSubmit", "user-prompt-submit", None)]
    if compact_cycle:
        routes += [("PreCompact", "pre-compact", "manual"),
            ("PostCompact", "post-compact", "manual"), ("SessionStart", "session-start", "compact")]
    hooks = {}
    for event, command, matcher in routes:
        group = {"hooks": [{"type": "command",
            "command": f'"{entry_text}" {command} {flag}',
            "commandWindows": f"& '{windows_path}' {command} {flag}",
            "timeout": 2}]}
        # Compact events cannot emit additionalContext; configuring a spill limit
        # on them is ignored with a host warning, even when the value is zero.
        if event not in {"PreCompact", "PostCompact"}:
            group["hooks"][0]["additionalContextLimit"] = 0
        if matcher is not None:
            group["matcher"] = matcher
        hooks[event] = [group]
    return {"hooks": hooks}


def _binding(paths: RetrievalPaths, session_id: str, holds: ExitStack,
             deadline: DeadlineBudget, *, release_deadline: DeadlineBudget,
             compact_cycle: bool = False) -> str:
    try:
        def hold(directory):
            return holds.enter_context(_directories(directory, create=False,
                deadline=deadline, release_deadline=release_deadline))

        held = hold(paths.runtime_root)
        marker_name = "p26d-compact-lab.json" if compact_cycle else "p26c-host-lab.json"
        raw = _read(held, paths.runtime_root / marker_name, limit=8192, deadline=deadline)
        marker = parse_json_value(raw.decode("utf-8"))
        if not isinstance(marker, dict) or set(marker) != _FIELDS:
            raise ValueError("selected host binding fields differ")
        if any(not isinstance(value, str) or not value for value in marker.values()):
            raise ValueError("selected host binding types differ")
        home_value = os.environ.get("CODEX_HOME", "")
        if not Path(home_value).is_absolute():
            raise ValueError("private HOME required")
        home = Path(os.path.abspath(home_value))
        hold(home)
        home = home.resolve(strict=True)
        # Only the observed installed executable may have a migration alias.
        # Resolve it first, then protect and hash its physical ancestry/leaf.
        # Lab, HOME, marker and Hook paths retain their strict redirect checks.
        entry = (Path(sys.executable).parent / "codex-memory-hook.exe").resolve(strict=True)
        argv_entry = Path(sys.argv[0])
        # The installed uv launcher removes its own ".exe" suffix before main.
        if not argv_entry.suffix:
            argv_entry = argv_entry.with_suffix(".exe")
        if argv_entry.resolve() != entry:
            raise ValueError("selected host must use the installed entry")
        entry_held = hold(entry.parent)
        entry_raw = _read(entry_held, entry, limit=1024 * 1024, deadline=deadline)
        expected = {
            "format": "codex-memory.p26d-selected-compact.v1" if compact_cycle else "codex-memory.p26c-selected-host.v1",
            "project_root": str(paths.repo_root), "data_origin": "synthetic", "provider_policy": "disabled",
            "hook_policy": "selected_compact" if compact_cycle else "selected_host",
            "session_id": session_id, "codex_home": str(home), "runtime_entry": marker["runtime_entry"],
            "runtime_entry_sha256": hashlib.sha256(entry_raw).hexdigest(),
            "hooks_sha256": marker["hooks_sha256"],
        }
        if marker != expected or raw != canonical_json(expected).encode("utf-8"):
            raise ValueError("unapproved selected host laboratory")
        marker_entry = Path(marker["runtime_entry"])
        if not marker_entry.is_absolute() or marker_entry != entry:
            raise ValueError("selected entry physical identity differs")
        if re.fullmatch(r"[0-9a-f]{64}", marker["hooks_sha256"]) is None:
            raise ValueError("hook digest required")
        hook_dir = paths.repo_root / ".codex"
        hook_held = hold(hook_dir)
        forbidden_paths = [hook_dir / "config.toml", home / "hooks.json"]
        if compact_cycle:
            forbidden_paths.append(hook_dir / "memory.toml")
        for forbidden in forbidden_paths:
            try:
                forbidden.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ValueError("unregistered additional hook/config source")
        hook_raw = _read(hook_held, hook_dir / "hooks.json", limit=16384, deadline=deadline)
        hooks = hook_definition(marker_entry, compact_cycle=compact_cycle)
        if (hook_raw != canonical_json(hooks).encode("utf-8")
                or hashlib.sha256(hook_raw).hexdigest() != marker["hooks_sha256"]):
            raise ValueError("selected hook bytes or definition differ")
        return hashlib.sha256(raw).hexdigest()
    except InjectionAuditError as exc:
        if exc.reason.value == "deadline_exhausted":
            raise
        raise IsolatedPacketFailure("selected_host_not_authorized", 2) from exc
    except Exception as exc:
        raise IsolatedPacketFailure("selected_host_not_authorized", 2) from exc


def run_selected_host_user_prompt(envelope, stdout, stderr, **kwargs) -> int:
    return _run_user_prompt(envelope, stdout, stderr, **kwargs,
        binding_validator=_binding, source_origin="hook", output_mode="selected_host")


def run_selected_compact(command, envelope, stdout, stderr, *, process_deadline_ns,
                         resolve_repo_root, runtime_factory, dispatch) -> int:
    """Authorize the compact profile; keep existing checkpoint/recovery behavior."""
    binding = partial(_binding, compact_cycle=True)
    if command == "user-prompt-submit":
        return _run_user_prompt(envelope, stdout, stderr, process_deadline_ns=process_deadline_ns,
            resolve_repo_root=resolve_repo_root, runtime_factory=runtime_factory,
            binding_validator=binding, source_origin="hook", output_mode="selected_compact")
    if ((command in {"pre-compact", "post-compact"} and envelope.trigger != "manual")
            or (command == "session-start" and envelope.source != "compact")):
        stderr.write("codex-memory-hook: selected compact event not authorized\n")
        return 2
    try:
        if type(process_deadline_ns) is not int or process_deadline_ns < 1_500_000_000:
            raise IsolatedPacketFailure("selected_host_not_authorized", 2)
        budget = DeadlineBudget(process_deadline_ns - 1_500_000_000, process_deadline_ns)
        root = resolve_repo_root(envelope.cwd, process_deadline_ns)
        paths = RetrievalPaths.for_repo(root)
        with ExitStack() as holds:
            binding(paths, envelope.session_id, holds, budget, release_deadline=budget)
            runtime = runtime_factory(root, budget)
            try:
                return dispatch(command, envelope, runtime, stdout, stderr,
                    process_deadline_ns=process_deadline_ns)
            finally:
                runtime.connection.close()
    except IsolatedPacketFailure as exc:
        stderr.write("codex-memory-hook: selected compact binding rejected\n")
        return exc.code
    except Exception:
        stderr.write("codex-memory-hook: selected compact runtime unavailable\n")
        return 1
