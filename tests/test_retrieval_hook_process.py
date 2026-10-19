from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import time

import pytest

from codex_memory import database, structured_ledger
from codex_memory.paths import RuntimePaths
from codex_memory.retrieval.local_http_provider import (
    AUDITED_DENSE_FINGERPRINT,
    AUDITED_PROVIDER_FINGERPRINT,
)
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService


class _DelayedVersionHandler(BaseHTTPRequestHandler):
    requests: list[str] = []
    version_entered = threading.Event()
    version_delay_seconds = 0.25

    def log_message(self, *_args: object) -> None:
        return

    def _write_json(self, payload: object) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except OSError:
            # The child is expected to stop waiting before this delayed
            # response is written; a closed client socket is part of the
            # process-boundary timeout fixture.
            pass

    def do_GET(self) -> None:  # noqa: N802 - http.server protocol name
        type(self).requests.append(self.path)
        if self.path == "/api/version":
            type(self).version_entered.set()
            time.sleep(type(self).version_delay_seconds)
            self._write_json({"version": "0.34.0"})
            return
        if self.path == "/api/tags":
            self._write_json(
                {
                    "models": [
                        {
                            "name": "bge-m3:latest",
                            "digest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab",
                        }
                    ]
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - http.server protocol name
        type(self).requests.append(self.path)
        self.send_error(404)


class _LongDelayedVersionHandler(_DelayedVersionHandler):
    """Provide enough admission time while exceeding the shared budget."""

    version_delay_seconds = 2.0

def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    subprocess.run(
        ["git", "init", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = RuntimePaths.for_repo(repository)
    runtime.ensure()
    connection = database.connect_database(runtime.db_path)
    try:
        database.migrate_database(connection)
        record = structured_ledger.propose_record(
            connection,
            record_type="decision",
            scope_kind="project",
            scope_key=str(repository.resolve()),
            topic_key="subprocess.deadline",
            statement="Subprocess shadow deadline fixture",
            value=None,
            evidence_uris=(),
            idempotency_key="subprocess-shadow-deadline-proposal",
        )
        structured_ledger.apply_action(
            connection,
            record_id=record.record_id,
            action="activate",
            idempotency_key="subprocess-shadow-deadline-activate",
        )
    finally:
        connection.close()
    assert RetrievalService(RetrievalPaths.for_repo(repository)).rebuild().status.value == "current"
    return repository


def _write_shadow_config(
    repository: Path,
    port: int,
    *,
    query_deadline_ms: int = 100,
    read_timeout_ms: int = 250,
) -> None:
    codex = repository / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text(
        f'''[retrieval]
mode = "shadow"

[retrieval.semantic]
version = 1
provider = "local_http"
base_url = "http://127.0.0.1:{port}"
protocol = "audited-local-v1"
query_deadline_ms = {query_deadline_ms}
build_deadline_ms = 1000
connect_timeout_ms = 50
read_timeout_ms = {read_timeout_ms}
dense = true
sparse = false
reranker = false
dense_top_k = 16
sparse_top_k = 16
rerank_top_n = 8
max_batch_items = 16
max_inflight = 1
queue_limit = 0
circuit_failure_threshold = 2
circuit_cooldown_ms = 30000
expected_provider_fingerprint = "{AUDITED_PROVIDER_FINGERPRINT}"
expected_dense_fingerprint = "{AUDITED_DENSE_FINGERPRINT}"
''',
        encoding="utf-8",
    )


def test_real_hook_subprocess_observes_shared_shadow_deadline_at_process_boundary(
    tmp_path: Path,
) -> None:
    """The installed Hook process must return on its budget, not only its Future."""
    executable = shutil.which("codex-memory-hook")
    assert executable is not None, "installed codex-memory-hook entry point is required"
    repository = _repository(tmp_path)
    _LongDelayedVersionHandler.requests = []
    _LongDelayedVersionHandler.version_entered = threading.Event()
    request_events: list[dict[str, object]] = []

    class ObservedVersionHandler(_LongDelayedVersionHandler):
        # Only record arrival. A delayed response attempt is neither a new
        # request nor proof that a response reached the child.
        def do_GET(self) -> None:  # noqa: N802 - http.server protocol name
            request_events.append({
                "event": "request_arrived", "method": "GET", "path": self.path,
                "perf_counter_ns": time.perf_counter_ns(),
            })
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - http.server protocol name
            request_events.append({
                "event": "request_arrived", "method": "POST", "path": self.path,
                "perf_counter_ns": time.perf_counter_ns(),
            })
            super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", 0), ObservedVersionHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _write_shadow_config(
            repository,
            int(server.server_address[1]),
            query_deadline_ms=1000,
            read_timeout_ms=3000,
        )
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "subprocess-shadow-session",
            "turn_id": "subprocess-shadow-turn",
            "cwd": str(repository),
            "prompt": "subprocess deadline",
        }
        started = time.perf_counter()
        result = subprocess.run(
            [executable, "user-prompt-submit"],
            cwd=repository,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        ended = time.perf_counter()
        elapsed_ms = (ended - started) * 1000
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    health_state: dict[str, object]
    retrieval_database = RetrievalPaths.for_repo(repository).database_path
    try:
        health_connection = sqlite3.connect(
            f"{retrieval_database.as_uri()}?mode=ro",
            uri=True,
        )
        health_connection.row_factory = sqlite3.Row
        try:
            admissions = [
                dict(row)
                for row in health_connection.execute(
                    "SELECT attempt_id, operation FROM "
                    "retrieval_provider_attempt_admissions ORDER BY attempt_seq"
                )
            ]
            results = [
                dict(row)
                for row in health_connection.execute(
                    "SELECT attempt_id, outcome, provider_status FROM "
                    "retrieval_provider_attempt_results ORDER BY result_seq"
                )
            ]
            health_events = [
                dict(row)
                for row in health_connection.execute(
                    "SELECT action, failure_kind FROM "
                    "retrieval_provider_health_events ORDER BY event_seq"
                )
            ]
        finally:
            health_connection.close()
        health_state = {
            "read_status": "ok",
            "admissions": admissions,
            "results": results,
            "health_events": health_events,
        }
    except sqlite3.Error as exc:
        health_state = {
            "read_status": "error",
            "error_type": type(exc).__name__,
        }

    # Persist before every original assertion, including the elapsed limit.
    # All filesystem work is outside the original subprocess timing interval.
    (tmp_path / "hook-deadline-evidence.json").write_text(
        json.dumps({
            "command": [executable, "user-prompt-submit"],
            "cwd": str(repository),
            "process_timing": {
                "clock": "time.perf_counter",
                "started_seconds": started,
                "ended_seconds": ended,
                "elapsed_ms": elapsed_ms,
                "scope": "subprocess.run only; excludes server cleanup and sidecar I/O",
            },
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "request_events": list(request_events),
            "request_paths": list(_LongDelayedVersionHandler.requests),
            "request_clock": "time.perf_counter_ns",
            "clock_note": (
                "Request timestamps use the same performance counter as process timing; "
                "process seconds are floating-point readings, not original nanosecond values. "
                "Converting them to nanoseconds introduces floating-point rounding."
            ),
            "request_observation_scope": "handler arrivals through this post-shutdown snapshot",
            "request_observation_cutoff_perf_counter_ns": time.perf_counter_ns(),
            "version_entered": _LongDelayedVersionHandler.version_entered.is_set(),
            "version_request_observed": "/api/version" in _LongDelayedVersionHandler.requests,
            "embed_request_observed": "/api/embed" in _LongDelayedVersionHandler.requests,
            "response_delivery": "not_observed",
            "server_thread_alive": thread.is_alive(),
            "health_state": health_state,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    response = json.loads(result.stdout)
    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "[CODEX MEMORY PROTOCOL v1]" in response["hookSpecificOutput"]["additionalContext"]
    assert elapsed_ms < 1500, f"Hook process exceeded shadow budget: {elapsed_ms:.1f}ms"
    assert _LongDelayedVersionHandler.version_entered.is_set()
    assert "/api/version" in _LongDelayedVersionHandler.requests
    assert "/api/embed" not in _LongDelayedVersionHandler.requests
    assert health_state["read_status"] == "ok"
    assert health_state["health_events"] == [{
        "action": "failure",
        "failure_kind": "timeout",
    }]
    admissions = health_state["admissions"]
    results = health_state["results"]
    assert isinstance(admissions, list) and len(admissions) == 1
    assert isinstance(results, list) and len(results) == 1
    assert admissions[0]["operation"] == "dense_query"
    assert results == [{
        "attempt_id": admissions[0]["attempt_id"],
        "outcome": "failure",
        "provider_status": "timeout",
    }]


def test_real_hook_subprocess_bounds_locked_retrieval_read_at_process_boundary(
    tmp_path: Path,
) -> None:
    """A locked disposable cache must not defeat the Hook's total shadow budget."""
    executable = shutil.which("codex-memory-hook")
    assert executable is not None, "installed codex-memory-hook entry point is required"
    repository = _repository(tmp_path)
    retrieval_path = RetrievalPaths.for_repo(repository).database_path
    lock = sqlite3.connect(retrieval_path, isolation_level=None)
    lock.execute("PRAGMA busy_timeout=0")
    lock.execute("BEGIN EXCLUSIVE")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DelayedVersionHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _write_shadow_config(repository, int(server.server_address[1]))
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "subprocess-locked-cache-session",
            "turn_id": "subprocess-locked-cache-turn",
            "cwd": str(repository),
            "prompt": "locked cache deadline",
        }
        started = time.perf_counter()
        result = subprocess.run(
            [executable, "user-prompt-submit"],
            cwd=repository,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        lock.rollback()
        lock.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.returncode == 0
    assert result.stderr == ""
    response = json.loads(result.stdout)
    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert elapsed_ms < 1500, f"Hook process exceeded shadow budget: {elapsed_ms:.1f}ms"


def test_real_hook_subprocess_does_not_wait_on_health_journal_mode_after_admission(
    tmp_path: Path,
) -> None:
    """A lock acquired after provider admission must not extend the Hook budget."""
    executable = shutil.which("codex-memory-hook")
    assert executable is not None, "installed codex-memory-hook entry point is required"
    repository = _repository(tmp_path)
    _DelayedVersionHandler.requests = []
    _DelayedVersionHandler.version_entered = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DelayedVersionHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "subprocess-health-journal-session",
        "turn_id": "subprocess-health-journal-turn",
        "cwd": str(repository),
        "prompt": "health journal deadline",
    }
    _write_shadow_config(
        repository,
        int(server.server_address[1]),
        query_deadline_ms=1000,
    )
    result_holder: dict[str, object] = {}
    retrieval_path = RetrievalPaths.for_repo(repository).database_path
    lock_holder: dict[str, sqlite3.Connection] = {}
    lock_guard = threading.Lock()
    lock_released = threading.Event()

    def run_hook() -> None:
        result_holder["started_ns"] = time.perf_counter_ns()
        try:
            result_holder["result"] = subprocess.run(
                [executable, "user-prompt-submit"],
                cwd=repository,
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except BaseException as exc:
            result_holder["error"] = exc
        finally:
            result_holder["ended_ns"] = time.perf_counter_ns()

    runner = threading.Thread(target=run_hook, daemon=True)
    runner.start()
    try:
        assert _DelayedVersionHandler.version_entered.wait(5), "provider admission did not reach fixture server"
        lock = None
        for _ in range(100):
            candidate = sqlite3.connect(
                retrieval_path, isolation_level=None, timeout=0, check_same_thread=False
            )
            candidate.execute("PRAGMA busy_timeout=0")
            try:
                candidate.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError:
                candidate.close()
                time.sleep(0.01)
                continue
            lock = candidate
            break
        assert lock is not None, "fixture could not acquire the retrieval database lock"
        lock_holder["connection"] = lock
        bookkeeping_started = time.perf_counter()

        def release_after_budget() -> None:
            if lock_released.wait(2.0):
                return
            with lock_guard:
                current = lock_holder.pop("connection", None)
                if current is not None:
                    current.rollback()
                    current.close()
                lock_released.set()

        unlocker = threading.Thread(target=release_after_budget, daemon=True)
        unlocker.start()
        runner.join(timeout=5)
        process_elapsed_ms = (time.perf_counter() - bookkeeping_started) * 1000
        unlocker.join(timeout=3)
    finally:
        with lock_guard:
            current = lock_holder.pop("connection", None)
            if current is not None:
                current.rollback()
                current.close()
            lock_released.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        result = result_holder.get("result")
        error = result_holder.get("error")
        started_ns = result_holder.get("started_ns")
        ended_ns = result_holder.get("ended_ns")
        evidence = {
            "command": [executable, "user-prompt-submit"],
            "cwd": str(repository),
            "process_clock": "time.perf_counter_ns",
            "started_ns": started_ns,
            "ended_ns": ended_ns,
            "elapsed_ms": (
                (ended_ns - started_ns) / 1_000_000
                if isinstance(started_ns, int) and isinstance(ended_ns, int)
                else None
            ),
            "runner_alive": runner.is_alive(),
            "returncode": result.returncode if isinstance(result, subprocess.CompletedProcess) else None,
            "stdout": result.stdout if isinstance(result, subprocess.CompletedProcess) else None,
            "stderr": result.stderr if isinstance(result, subprocess.CompletedProcess) else None,
            "error_type": type(error).__name__ if error is not None else None,
            "request_paths": list(_DelayedVersionHandler.requests),
            "version_entered": _DelayedVersionHandler.version_entered.is_set(),
            "lock_acquired": "bookkeeping_started" in locals(),
            "lock_released": lock_released.is_set(),
            "server_thread_alive": server_thread.is_alive(),
            "scope": "fixture diagnostic; assertions and installed entry unchanged",
        }
        (tmp_path / "hook-health-journal-evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    assert not runner.is_alive(), "Hook subprocess did not exit within the process timeout"
    if "error" in result_holder:
        raise AssertionError("Hook subprocess failed to return a result") from result_holder["error"]
    result = result_holder["result"]
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert result.stderr == ""
    response = json.loads(result.stdout)
    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert process_elapsed_ms < 1500, f"Hook process exceeded health bookkeeping budget: {process_elapsed_ms:.1f}ms"


@pytest.mark.filterwarnings(
    "ignore:record_property is incompatible with junit_family 'xunit2':pytest.PytestWarning"
)
def test_twelve_short_lived_hook_processes_persist_two_timeout_circuit(
    tmp_path: Path, record_property,
) -> None:
    """Two timed-out Hook processes open one durable circuit for later processes."""
    executable = shutil.which("codex-memory-hook")
    assert executable is not None, "installed codex-memory-hook entry point is required"
    repository = _repository(tmp_path)
    _LongDelayedVersionHandler.requests = []
    _LongDelayedVersionHandler.version_entered = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LongDelayedVersionHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _write_shadow_config(
        repository,
        int(server.server_address[1]),
        query_deadline_ms=500,
        read_timeout_ms=3000,
    )
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "durable-timeout-session",
        "turn_id": "durable-timeout-turn",
        "cwd": str(repository),
        "prompt": "durable timeout circuit",
    }
    results: list[subprocess.CompletedProcess[str]] = []
    elapsed: list[float] = []
    process_log: list[dict[str, object]] = []
    try:
        for index in range(12):
            started = time.perf_counter()
            requests_before = len(_LongDelayedVersionHandler.requests)
            results.append(
                subprocess.run(
                    [executable, "user-prompt-submit"],
                    cwd=repository,
                    input=json.dumps(payload, ensure_ascii=False),
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=3,
                )
            )
            ended = time.perf_counter()
            elapsed_ms = (ended - started) * 1000
            elapsed.append(elapsed_ms)
            result = results[-1]
            process_log.append(
                {
                    "index": index + 1,
                    "started_ms": round(started * 1000, 3),
                    "ended_ms": round(ended * 1000, 3),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "service_requests_delta": len(_LongDelayedVersionHandler.requests) - requests_before,
                    "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
                    "returncode": result.returncode,
                }
            )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    with sqlite3.connect(RetrievalPaths.for_repo(repository).database_path) as connection:
        events = connection.execute(
            """SELECT action, failure_kind
               FROM retrieval_provider_health_events
               ORDER BY event_seq"""
        ).fetchall()
        current = connection.execute(
            """SELECT state, consecutive_failures
               FROM retrieval_provider_health_current
               WHERE provider_fingerprint = ? AND component = 'dense'""",
            (AUDITED_PROVIDER_FINGERPRINT,),
        ).fetchone()
        admissions = connection.execute(
            """SELECT attempt_seq, attempt_id, capability, operation,
                      admitted_at, expires_at
               FROM retrieval_provider_attempt_admissions
               ORDER BY attempt_seq"""
        ).fetchall()
        attempt_results = connection.execute(
            """SELECT result_seq, attempt_id, outcome, provider_status,
                      recorded_at
               FROM retrieval_provider_attempt_results
               ORDER BY result_seq"""
        ).fetchall()

    diagnostic = {
        "runs": process_log,
        "service_requests": list(_LongDelayedVersionHandler.requests),
        "health_events": events,
        "health_current": current,
        "attempt_admissions": admissions,
        "attempt_results": attempt_results,
    }
    diagnostic_json = json.dumps(diagnostic, separators=(",", ":"))
    record_property("p23_hook_processes", diagnostic_json)

    expected_stdout = results[0].stdout
    assert all(
        (result.returncode, result.stdout, result.stderr) == (0, expected_stdout, "")
        for result in results
    ), diagnostic_json
    response = json.loads(expected_stdout)
    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "[CODEX MEMORY PROTOCOL v1]" in response["hookSpecificOutput"]["additionalContext"]
    assert max(elapsed) < 1500, (
        f"Hook process exceeded total budget: {max(elapsed):.1f}ms; {diagnostic_json}"
    )
    assert _LongDelayedVersionHandler.requests.count("/api/version") == 2, diagnostic_json

    assert events == [
        ("failure", "timeout"), ("failure", "timeout"), ("open", None)
    ], diagnostic_json
    assert current == ("open", 2), diagnostic_json
