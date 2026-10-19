"""Two fixed synthetic diagnostics on a private branch, not a full/release gate."""
from datetime import datetime, timezone
import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile

NODES = (
    "tests/test_p26b_prompt.py::test_isolated_prompt_does_not_use_cli_or_new_phase_budget",
    "tests/test_p26d_compact.py::test_compact_profile_real_checkpoint_recovery_and_fresh_window[True]",
    "tests/test_retrieval_benchmark.py::test_pipeline_certifies_mandatory_constitution_and_preserves_its_rerank_slot",
    "tests/test_retrieval_benchmark_pipeline.py::test_pipeline_fixture_remains_available_for_verified_semantic_reopen_in_child_process",
    "tests/test_retrieval_hook_process.py::test_real_hook_subprocess_observes_shared_shadow_deadline_at_process_boundary",
    "tests/test_retrieval_provider_control.py::test_stalled_call_is_discarded_and_keeps_health_ownership[dense]",
    "tests/test_retrieval_provider_control.py::test_stalled_call_is_discarded_and_keeps_health_ownership[sparse]",
    "tests/test_retrieval_provider_control.py::test_unrelated_admission_io_cannot_hold_a_timeout_settlement_past_its_budget[dense]",
    "tests/test_retrieval_provider_control.py::test_unrelated_admission_io_cannot_hold_a_timeout_settlement_past_its_budget[sparse]",
    "tests/test_retrieval_provider_control.py::test_timeout_bookkeeping_does_not_wait_for_post_admission_verification",
    "tests/test_retrieval_provider_control.py::test_timeout_does_not_wait_for_a_blocked_read_only_health_verifier",
    "tests/test_retrieval_provider_control.py::test_repeated_controller_timeouts_open_the_durable_circuit",
    "tests/test_retrieval_provider_durable_admission.py::test_nonfault_release_does_not_wait_past_hook_budget_for_rebuild_lock",
)

if __name__ == "__main__":
    assert sys.platform == "win32" and sys.version_info[:3] == (3, 12, 13)
    assert os.environ.get("GITHUB_REF") == "refs/heads/codex/ci-timing-diagnostics"
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    assert run_id.isdecimal() and attempt == "1", "one diagnostic attempt; no automatic rerun"
    root = Path.cwd().resolve()
    system_temp = Path(tempfile.gettempdir()).resolve(strict=True)
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    assert system_temp.drive.casefold() != runner_temp.drive.casefold(), "no distinct-volume comparison"
    assert all(root != p and root not in p.parents for p in (system_temp, runner_temp))
    output = root / ".artifacts" / ("windows-timing-" + run_id)
    output.mkdir(parents=True, exist_ok=False)
    sys.stdout.reconfigure(encoding="utf-8")  # Supervisor only; no child environment changes.
    git = lambda *args: subprocess.check_output(["git", *args]).decode().strip()
    print(json.dumps({"scope": "diagnostic only, 13 fixed nodes twice; not full/release certification",
        "head": git("rev-parse", "HEAD"), "runtime_tree": git("rev-parse", "HEAD:src"),
        "python": platform.python_version(), "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version, "nodes": NODES}), flush=True)
    results = []
    for label, parent in (("system-temp", system_temp), ("runner-temp", runner_temp)):
        base = parent / ("hippocampus-ci-" + run_id + "-" + label)
        assert base.parent == parent and base.name.startswith("hippocampus-ci-")
        assert not base.exists() and not base.is_symlink()
        command = [sys.executable, "-m", "pytest", "-q", "-rs", "--basetemp", str(base), *NODES]
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(command, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or b"") + (exc.stderr or b"")
            kept = partial[:2 * 1024 * 1024]
            with (output / (label + ".timeout-partial.log")).open("xb") as stream:
                stream.write(kept)
            print(json.dumps({"profile": label, "command": command,
                "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "result": "OUTER_TIMEOUT", "exit_status": 124, "no_further_dispatch": True,
                "observed_bytes": len(partial), "stored_bytes": len(kept),
                "partial_capture_truncated": len(kept) != len(partial),
                "stored_sha256": hashlib.sha256(kept).hexdigest(),
                "partial_console": kept.decode(locale.getencoding(), errors="backslashreplace")}), flush=True)
            raise SystemExit(124)
        raw = result.stdout + result.stderr
        with (output / (label + ".log")).open("xb") as stream:
            stream.write(raw)
        summary = {"profile": label, "command": command, "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(), "exit_status": result.returncode,
            "base_drive": base.drive, "log_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        print(json.dumps(summary), flush=True)
        assert len(raw) <= 2 * 1024 * 1024, "reject oversized diagnostic output"
        print(raw.decode(locale.getencoding()), flush=True)
        results.append(summary)
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump({"scope": "diagnostic only", "profiles": results}, stream, indent=2)
    raise SystemExit(0 if all(r["exit_status"] == 0 for r in results) else 1)
