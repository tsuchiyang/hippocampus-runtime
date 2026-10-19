from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from codex_memory import cli
from codex_memory.retrieval import isolated_hook
from test_p26c_selected_host import selected_host_lab  # fixture dependency
from test_p26d_compact import compact_lab, invoke
from test_p26d_promotion_race import _memory_rows


@pytest.mark.parametrize("stage", ["initialization", "event_append"])
def test_short_real_writer_contention_uses_the_original_prompt_budget(compact_lab, monkeypatch, stage):
    """Real selected CLI + SQLite: a writer is held at the target BEGIN.

    The 40ms controlled release starts only after the actual target SQL trace,
    not random sleep/repeated sampling. No Clock, Legacy, provider or safety
    validation is substituted. The lock is on this synthetic fixture only.
    """
    runtime, _marker, _hooks = compact_lab
    before = _memory_rows(runtime.connection)
    ready, attempting = threading.Event(), threading.Event()
    facts, lock_errors, threads = [], [], []
    original_factory, original_connect = cli._construct_runtime, cli.connect_database
    original_append = isolated_hook.handle_user_prompt_submit

    def trace(statement):
        if statement == "BEGIN IMMEDIATE":
            facts.append({"phase": stage, "sql_begin_ns": time.monotonic_ns()})
            attempting.set()

    def start_locker(deadline):
        def locker():
            try:
                connection = sqlite3.connect(runtime.paths.db_path, isolation_level=None, timeout=0)
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    ready.set()
                    left = max(0, (deadline.deadline_ns - time.monotonic_ns()) / 1e9)
                    if not attempting.wait(left):
                        lock_errors.append("target BEGIN not observed")
                    # Deliberate bounded writer-work latency after the observed
                    # SQLite boundary. The original absolute budget is retained.
                    left = max(0, (deadline.deadline_ns - time.monotonic_ns()) / 1e9)
                    threading.Event().wait(min(.040, left))
                    connection.rollback()
                    facts.append({"released_ns": time.monotonic_ns()})
                finally:
                    connection.close()
            except Exception as error:
                lock_errors.append(type(error).__name__)
                ready.set()

        thread = threading.Thread(target=locker)
        threads.append(thread)
        thread.start()
        assert ready.wait(max(0, (deadline.deadline_ns - time.monotonic_ns()) / 1e9))
        assert not lock_errors

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(trace)
        return connection

    def factory(root, *, deadline=None):
        if stage == "initialization":
            start_locker(deadline)
        try:
            value = original_factory(root, deadline=deadline)
        except sqlite3.Error as error:
            facts.append({"sqlite_errorcode": error.sqlite_errorcode, "phase": "initialization"})
            raise
        if stage == "event_append":
            attempting.clear()
            facts.clear()  # initialization is not the target append boundary
            start_locker(deadline)
        return value

    def append(*args):
        try:
            return original_append(*args)
        except sqlite3.Error as error:
            facts.append({"sqlite_errorcode": error.sqlite_errorcode, "phase": "event_append"})
            raise

    monkeypatch.setattr(cli, "connect_database", connect)
    monkeypatch.setattr(cli, "_construct_runtime", factory)
    monkeypatch.setattr(isolated_hook, "handle_user_prompt_submit", append)
    started = time.monotonic_ns()
    try:
        code, stdout, stderr = invoke(compact_lab, "user-prompt-submit", "UserPromptSubmit",
            turn_id="bounded-contention", prompt="What is the HOST-P26C-EXAMPLE verification code?")
    finally:
        attempting.set()
        for thread in threads:
            thread.join(timeout=2)  # failure cleanup only; no extension of Hook budget
    elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
    (runtime.paths.repo_root / "write-contention.json").write_text(json.dumps({
        "stage": stage, "facts": facts, "lock_errors": lock_errors, "exit_status": code,
        "stdout_present": bool(stdout), "stderr": json.loads(stderr), "elapsed_ms": elapsed_ms,
    }), encoding="utf-8")
    assert code == 0, (stderr, facts)
    assert json.loads(stderr)["delivery"] == "emitted"
    assert elapsed_ms < 1500 and not lock_errors and all(not thread.is_alive() for thread in threads)
    assert any("sql_begin_ns" in fact for fact in facts) and any("released_ns" in fact for fact in facts)
    assert not any("sqlite_errorcode" in fact for fact in facts)
    assert "HOST-P26C-EXAMPLE" in stdout
    assert _memory_rows(runtime.connection) == before
    assert [tuple(row) for row in runtime.connection.execute(
        "SELECT session_id, turn_id, event_name FROM events")] == [
        ("selected-session", "bounded-contention", "UserPromptSubmit")]
