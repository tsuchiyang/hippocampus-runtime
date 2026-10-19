from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sqlite3
from typing import Callable

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.hook_input import HookEnvelope, parse_hook_payload
from codex_memory.paths import RuntimePaths


@dataclass(frozen=True, slots=True)
class Runtime:
    paths: RuntimePaths
    connection: object
    events: object


def _prompt_envelope(session_id: str, prompt: str) -> HookEnvelope:
    return parse_hook_payload(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": session_id,
            "turn_id": "turn_1",
            "cwd": ".",
            "prompt": prompt,
        }
    )


def _tool_envelope(session_id: str, tool_use_id: str, tool_name: str) -> HookEnvelope:
    return parse_hook_payload(
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "turn_id": "turn_1",
            "cwd": ".",
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "tool_input": {"command": "echo hi"},
            "tool_response": {"output": "hi"},
        }
    )


@pytest.fixture
def runtime(tmp_path: Path):
    database = import_module("codex_memory.database")
    event_store = import_module("codex_memory.event_store")
    paths = RuntimePaths.for_repo(tmp_path)
    connection = database.connect_database(paths.db_path)
    database.migrate_database(connection)
    try:
        yield Runtime(paths, connection, event_store.EventStore(connection, BlobStore(paths.blobs_dir)))
    finally:
        try:
            connection.close()
        except sqlite3.ProgrammingError:
            pass


@pytest.fixture
def prompt_envelope() -> HookEnvelope:
    return _prompt_envelope("session_a", "hello")


@pytest.fixture
def prompt_envelope_factory() -> Callable[[str, str], HookEnvelope]:
    return _prompt_envelope


@pytest.fixture
def tool_envelope_factory() -> Callable[[str, str, str], HookEnvelope]:
    return _tool_envelope


@pytest.fixture
def semantic_functional_clock(monkeypatch: pytest.MonkeyPatch):
    """Healthy logical time for semantic behavior tests, never latency evidence.

    Keep real ledger/projection/index/worker/transaction/response validation.
    Scope clock substitution to the two consumers, not the shared time module.
    Tests of deadlines or real Hook wall-clock must not request this fixture.
    """
    import time
    from types import SimpleNamespace
    from codex_memory.retrieval import service, shadow

    now_ns = lambda: 41_000_000_000
    original_rebuild = service.RetrievalService.rebuild_semantic
    original_catch_up = service.RetrievalService.catch_up_semantic

    def rebuild_with_logical_clock(self, provider, config, **kwargs):
        kwargs.setdefault("clock", now_ns)
        return original_rebuild(self, provider, config, **kwargs)

    def catch_up_with_logical_clock(self, provider, config, **kwargs):
        kwargs.setdefault("clock", now_ns)
        return original_catch_up(self, provider, config, **kwargs)

    monkeypatch.setattr(service.RetrievalService, "rebuild_semantic", rebuild_with_logical_clock)
    monkeypatch.setattr(service.RetrievalService, "catch_up_semantic", catch_up_with_logical_clock)
    monkeypatch.setattr(shadow, "time", SimpleNamespace(monotonic_ns=now_ns))
    monkeypatch.setattr(service, "time", SimpleNamespace(**{
        **vars(time), "monotonic_ns": now_ns,
    }))
    return now_ns


@pytest.fixture
def p26_phase_trace(monkeypatch: pytest.MonkeyPatch):
    """Bounded test-only ingress evidence; no payload/path or clock changes."""
    from functools import wraps
    import time
    from codex_memory import cli
    from codex_memory.retrieval import isolated_hook, selected_host

    records = []

    def measured(stage, operation, deadline=None):
        started = time.perf_counter_ns()
        error = None
        try:
            return operation()
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            if len(records) < 128:
                cutoff = getattr(deadline, "deadline_ns", deadline)
                records.append({
                    "stage": stage,
                    "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000,
                    "remaining_ms": None if type(cutoff) is not int else
                        (cutoff - time.monotonic_ns()) / 1_000_000,
                    "exception_type": error,
                })

    def watch(module, name, stage):
        original = getattr(module, name)

        @wraps(original)
        def observed(*args, **kwargs):
            deadline = kwargs.get("deadline", kwargs.get("deadline_ns"))
            if name == "_binding" and len(args) >= 4:
                deadline = args[3]
            return measured(stage, lambda: original(*args, **kwargs), deadline)

        monkeypatch.setattr(module, name, observed)

    for module, name, stage in (
        (cli, "_resolve_repo_root", "root"),
        (cli, "_construct_runtime", "runtime"),
        (cli, "_dispatch", "dispatch"),
        (isolated_hook, "_binding", "isolated-binding"),
        (selected_host, "_binding", "selected-binding"),
        (isolated_hook, "_legacy_and_source", "legacy"),
        (isolated_hook, "prepare_isolated_packet", "prepare"),
        (isolated_hook, "_write_wire", "wire"),
    ):
        watch(module, name, stage)

    def watch_binding_directories(module, stage):
        original = module._directories

        def directories(*args, **kwargs):
            actual = original(*args, **kwargs)

            class HeldDirectories:
                def __enter__(self):
                    return measured(stage + "-enter", actual.__enter__, kwargs.get("deadline"))

                def __exit__(self, *exc_info):
                    return measured(stage + "-exit", lambda: actual.__exit__(*exc_info),
                        kwargs.get("release_deadline", kwargs.get("deadline")))

            return HeldDirectories()

        monkeypatch.setattr(module, "_directories", directories)

    watch_binding_directories(isolated_hook, "isolated-directories")
    watch_binding_directories(selected_host, "selected-directories")
    return records
