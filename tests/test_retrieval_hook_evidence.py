from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import test_retrieval_hook_process as hook_process


@pytest.mark.parametrize(
    "requests",
    [[], [("GET", "/api/version")], [("GET", "/api/version"), ("POST", "/api/embed")]],
    ids=["no-requests", "version-only", "version-and-embed"],
)
def test_deadline_evidence_survives_original_elapsed_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requests: list[tuple[str, str]],
    request: pytest.FixtureRequest,
) -> None:
    """Dropping or moving the sidecar after the deadline assertion loses evidence.

    This is a synthetic failure-injection test, not a real Hook observation:
    the child result and performance clock are controlled, the fixture's sleep
    is disabled, and its actual HTTP handler still receives the chosen requests.
    """
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    executable = str(tmp_path / "synthetic-codex-memory-hook.exe")
    stdout = json.dumps(
        {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                "additionalContext": "[CODEX MEMORY PROTOCOL v1]"}}
    )
    sidecar = tmp_path / "hook-deadline-evidence.json"
    clock = {"seconds": 10.0, "reads": 0, "request_index": 0}

    def perf_counter() -> float:
        clock["reads"] += 1
        clock["seconds"] = 10.0 if clock["reads"] == 1 else 11.6
        return clock["seconds"]

    def perf_counter_ns() -> int:
        clock["request_index"] += 1
        return int(clock["seconds"] * 1_000_000_000) + clock["request_index"] * 100_000_000

    def run_child(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # The only replaced external operation is the child execution; the
        # original config writer and server handler remain in the exercised path.
        assert command == [executable, "user-prompt-submit"]
        assert kwargs == {
            "cwd": repository,
            "input": json.dumps({
                "hook_event_name": "UserPromptSubmit",
                "session_id": "subprocess-shadow-session",
                "turn_id": "subprocess-shadow-turn",
                "cwd": str(repository),
                "prompt": "subprocess deadline",
            }, ensure_ascii=False),
            "capture_output": True, "check": False, "text": True, "timeout": 5,
        }
        import tomllib

        config = tomllib.loads((repository / ".codex" / "memory.toml").read_text(encoding="utf-8"))
        port = int(config["retrieval"]["semantic"]["base_url"].rsplit(":", 1)[1])
        for method, path in requests:
            connection = HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                connection.request(method, path)
                response = connection.getresponse()
                assert response.status == (200 if path == "/api/version" else 404)
                response.read()
            finally:
                connection.close()
        assert not sidecar.exists(), "sidecar I/O must remain outside child timing"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(hook_process, "_repository", lambda _tmp_path: repository)
    monkeypatch.setattr(hook_process, "shutil", SimpleNamespace(which=lambda _name: executable))
    monkeypatch.setattr(hook_process, "subprocess", SimpleNamespace(run=run_child))
    monkeypatch.setattr(hook_process, "time", SimpleNamespace(
        perf_counter=perf_counter, perf_counter_ns=perf_counter_ns, sleep=lambda _delay: None,
    ))

    with pytest.raises(AssertionError, match="Hook process exceeded shadow budget: 1600.0ms"):
        hook_process.test_real_hook_subprocess_observes_shared_shadow_deadline_at_process_boundary(tmp_path, request)

    assert sidecar.is_file(), "deadline assertion must leave a readable evidence sidecar"
    evidence = json.loads(sidecar.read_text(encoding="utf-8"))
    assert evidence["command"] == [executable, "user-prompt-submit"]
    assert evidence["cwd"] == str(repository)
    assert evidence["process_timing"]["clock"] == "time.perf_counter"
    assert evidence["process_timing"]["started_seconds"] == 10.0
    assert evidence["process_timing"]["ended_seconds"] == 11.6
    assert evidence["process_timing"]["elapsed_ms"] == pytest.approx(1600.0)
    assert clock["reads"] == 2, "retain the original two process clock readings"
    assert evidence["returncode"] == 0
    assert evidence["stdout"] == stdout
    assert evidence["stderr"] == ""
    assert evidence["request_paths"] == [path for _, path in requests]
    assert evidence["version_entered"] is bool(requests)
    assert evidence["version_request_observed"] is bool(requests)
    assert evidence["embed_request_observed"] is (len(requests) == 2)
    assert evidence["request_events"] == [
        {"event": "request_arrived", "method": method, "path": path,
         "perf_counter_ns": timestamp}
        for (method, path), timestamp in zip(requests, [10_100_000_000, 10_200_000_000])
    ]
    assert evidence["response_delivery"] == "not_observed"
