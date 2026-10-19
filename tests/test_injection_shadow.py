from __future__ import annotations

from io import StringIO
import json
import time

import pytest

from codex_memory.retrieval.config import load_retrieval_config


def _audit_config(root, *, mode="shadow", enabled=True, sessions=None, suffix=""):
    directory = root / ".codex"
    directory.mkdir(exist_ok=True)
    text = (
        f"[retrieval]\nmode = '{mode}'\n[retrieval.audit]\nversion = 1\n"
        f"enabled = {str(enabled).lower()}\nsession_ids = {json.dumps(sessions if sessions is not None else ['audit_session'])}\n"
        f"total_deadline_ms = 1000\n{suffix}"
    )
    (directory / "memory.toml").write_text(text, encoding="utf-8")


def test_audit_config_is_explicit_and_session_selected(tmp_path):
    _audit_config(tmp_path)
    config = load_retrieval_config(tmp_path)
    assert config.enabled
    assert config.audit_for_session("audit_session") is not None
    assert config.audit_for_session("another_session") is None
    assert config.audit.total_deadline_ms == 1000


@pytest.mark.parametrize("mode,enabled", [("off", True), ("shadow", False), ("active", True)])
def test_audit_config_off_false_active_never_selects(tmp_path, mode, enabled):
    _audit_config(tmp_path, mode=mode, enabled=enabled)
    assert load_retrieval_config(tmp_path).audit_for_session("audit_session") is None


@pytest.mark.parametrize("bad", [
    {"version": True}, {"version": 1.0}, {"version": 2}, {"enabled": 1},
    {"session_ids": []}, {"session_ids": ["audit_session", "audit_session"]},
    {"session_ids": ["bad id"]}, {"session_ids": [str(n) for n in range(129)]},
    {"total_deadline_ms": True}, {"total_deadline_ms": 1001}, {"unknown": 0},
])
def test_audit_strict_configuration_rejects_ambiguous_contract(bad):
    from codex_memory.retrieval import config as module
    from codex_memory.retrieval.errors import RetrievalValidationError
    if not hasattr(module, "AuditConfig"):
        pytest.fail("audit config contract missing")
    values = dict(version=1, enabled=True, session_ids=["audit_session"], total_deadline_ms=1000)
    values.update(bad)
    with pytest.raises(RetrievalValidationError):
        module.AuditConfig.from_dict(values)


@pytest.mark.parametrize("observation,expected", [("full", "emitted"), (3, "partial"), (None, "unknown"), (True, "unknown")])
def test_hook_passes_exact_event_and_write_observation_without_extra_output(tmp_path, monkeypatch, observation, expected):
    import test_hook_memory_integration as helper
    from codex_memory import cli
    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = helper._runtime(repository, helper.FakeEvents([]), helper.FakeCheckpoints([]))
    envelope = helper._envelope("UserPromptSubmit", repository)
    monkeypatch.setattr(cli, "current_context_id", lambda *a, **kw: "startup")
    monkeypatch.setattr(cli, "build_memory_packet", lambda *a, **kw: "LEGACY-é")
    observed = []
    monkeypatch.setattr(cli, "_run_shadow_after_legacy", lambda **kw: observed.append(kw))

    class Output(StringIO):
        writes = 0
        def write(self, value):
            self.writes += 1
            result = super().write(value)
            return result if observation == "full" else observation

    stdout, stderr = Output(), StringIO()
    assert cli._dispatch("user-prompt-submit", envelope, runtime, stdout, stderr) == 0
    assert stdout.writes == 1 and stderr.getvalue() == ""
    assert json.loads(stdout.getvalue())["hookSpecificOutput"]["additionalContext"] == "LEGACY-é"
    context = observed[0]["audit_context"]
    assert context["source"]["event_id"] == 17
    assert context["source"]["event_blob_sha256"] == "b" * 64
    assert context["legacy_packet"] == "LEGACY-é"
    assert context["legacy_response_bytes"] == stdout.getvalue().encode("utf-8")
    assert context["legacy_delivery"] == expected


def test_compare_reuses_query_and_absolute_deadline_for_capture(runtime, monkeypatch):
    from codex_memory.retrieval import shadow
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild().status.value == "current"
    request = RetrievalRequest("fixture", "audit_session", "turn", "startup", None, RetrievalScope(str(runtime.paths.repo_root)), mode=RetrievalMode.SHADOW)
    queried = []
    original_query = RetrievalService.query
    def query(self, arg, **kwargs):
        result = original_query(self, arg, **kwargs)
        queried.append(result)
        return result
    captured = []
    def capture(**kwargs):
        captured.append(kwargs)
        return "inj_" + "a" * 64
    monkeypatch.setattr(RetrievalService, "query", query)
    monkeypatch.setattr(shadow, "_capture_injection", capture, raising=False)
    comparison = shadow.compare(paths=paths, request=request, hook_event="UserPromptSubmit", deadline_ms=1000, audit_context={"fixture": True})
    assert len(queried) == len(captured) == 1
    assert captured[0]["result"] is queried[0]
    assert captured[0]["deadline"].deadline_ns - captured[0]["deadline"].started_ns == 1_000_000_000
    assert comparison.injection_id == "inj_" + "a" * 64
    assert comparison.response_content is None


def test_late_capture_cannot_become_success_or_extend_parent_wait(runtime, monkeypatch):
    from codex_memory.retrieval import shadow
    from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    paths = RetrievalPaths.for_repo(runtime.paths.repo_root)
    assert RetrievalService(paths).rebuild().status.value == "current"
    request = RetrievalRequest("fixture", "audit_session", "turn", "startup", None, RetrievalScope(str(runtime.paths.repo_root)), mode=RetrievalMode.SHADOW)
    reached = []
    def slow(**kwargs):
        reached.append(True)
        time.sleep(0.8)
        return "inj_" + "b" * 64
    monkeypatch.setattr(shadow, "_capture_injection", slow, raising=False)
    started = time.monotonic()
    comparison = shadow.compare(paths=paths, request=request, hook_event="UserPromptSubmit", deadline_ms=300, audit_context={"fixture": True})
    assert time.monotonic() - started < 0.6
    assert reached
    assert comparison.injection_id is None
    assert comparison.reason == "deadline_exhausted"


def _real_hook_fixture(runtime):
    from types import SimpleNamespace
    from codex_memory.blob_store import BlobStore
    from codex_memory.hook_input import parse_hook_payload
    from codex_memory.structured_ledger import propose_record, apply_action
    from codex_memory.retrieval.paths import RetrievalPaths
    from codex_memory.retrieval.service import RetrievalService
    record = propose_record(runtime.connection, record_type="decision", scope_kind="project", scope_key=str(runtime.paths.repo_root), topic_key="audit.storage", statement="Use WAL for the audit fixture.", evidence_uris=(), idempotency_key="audit-record")
    apply_action(runtime.connection, record_id=record.record_id, action="activate", idempotency_key="audit-activate")
    assert RetrievalService(RetrievalPaths.for_repo(runtime.paths.repo_root)).rebuild().status.value == "current"
    envelope = parse_hook_payload({"hook_event_name": "UserPromptSubmit", "session_id": "audit_session", "turn_id": "audit_turn", "cwd": str(runtime.paths.repo_root), "prompt": "Which WAL setting?"})
    hook_runtime = SimpleNamespace(paths=runtime.paths, events=runtime.events, connection=runtime.connection, blobs=BlobStore(runtime.paths.blobs_dir))
    return hook_runtime, envelope


def test_real_prompt_shadow_commits_exact_evidence_and_keeps_legacy_bytes(runtime, monkeypatch):
    from codex_memory import cli
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    from codex_memory.retrieval import shadow
    outcomes, failures = [], []
    original_compare, original_capture = shadow.compare, shadow._capture_injection
    def compare(**kwargs):
        result = original_compare(**kwargs)
        outcomes.append(result)
        return result
    def capture(**kwargs):
        try:
            return original_capture(**kwargs)
        except Exception as exc:
            failures.append(repr(exc))
            raise
    monkeypatch.setattr(shadow, "compare", compare)
    monkeypatch.setattr(shadow, "_capture_injection", capture)
    hook_runtime, envelope = _real_hook_fixture(runtime)
    _audit_config(runtime.paths.repo_root, mode="off")
    baseline, errors = StringIO(), StringIO()
    assert cli._dispatch("user-prompt-submit", envelope, hook_runtime, baseline, errors) == 0
    # EventStore's ordinary INSERT OR IGNORE replay advances SQLite's internal
    # AUTOINCREMENT sequence. Compare actual history, not that bookkeeping.
    before = tuple(row for row in runtime.connection.iterdump() if '"sqlite_sequence"' not in row)
    _audit_config(runtime.paths.repo_root)
    stdout, stderr = StringIO(), StringIO()
    assert cli._dispatch("user-prompt-submit", envelope, hook_runtime, stdout, stderr) == 0
    assert (stdout.getvalue(), stderr.getvalue()) == (baseline.getvalue(), errors.getvalue())
    assert tuple(row for row in runtime.connection.iterdump() if '"sqlite_sequence"' not in row) == before
    manifests = list((runtime.paths.runtime_root / "injections").glob("inj_*/manifest.json"))
    assert len(manifests) == 1, (outcomes, failures)
    verified = InjectionStore(InjectionPaths.for_repo(runtime.paths.repo_root), hook_runtime.blobs).load(manifests[0].parent.name)
    body = verified.manifest.body
    assert body["delivery"] == {"v2": "prepared", "legacy": "emitted", "host_acknowledgement": "unknown"}
    assert not body["candidates"]["suppressed_ids"]
    assert body["source"]["session_id"] == envelope.session_id
    assert body["source"]["event_id"] == runtime.events.append(envelope).event_id
    assert verified.serialized_response_bytes == baseline.getvalue().encode("utf-8")


@pytest.mark.parametrize("failure", ["audit_unwritable", "source_mismatch", "snapshot_unavailable"])
def test_real_shadow_audit_failures_preserve_legacy_and_never_commit(runtime, monkeypatch, failure):
    from codex_memory import cli
    from codex_memory.retrieval import assembler
    from codex_memory.injection_store import InjectionStore
    hook_runtime, envelope = _real_hook_fixture(runtime)
    _audit_config(runtime.paths.repo_root, mode="off")
    baseline, base_error = StringIO(), StringIO()
    assert cli._dispatch("user-prompt-submit", envelope, hook_runtime, baseline, base_error) == 0
    _audit_config(runtime.paths.repo_root)
    def unavailable(*args, **kwargs):
        raise PermissionError("private failure path must not reach stderr")
    if failure == "audit_unwritable":
        monkeypatch.setattr(InjectionStore, "commit", unavailable)
    elif failure == "snapshot_unavailable":
        monkeypatch.setattr(assembler, "build_phase2_packet", unavailable)
    else:
        original = cli._run_shadow_after_legacy
        def wrong_source(**kwargs):
            kwargs["audit_context"]["source"]["event_id"] += 1
            return original(**kwargs)
        monkeypatch.setattr(cli, "_run_shadow_after_legacy", wrong_source)
    stdout, stderr = StringIO(), StringIO()
    assert cli._dispatch("user-prompt-submit", envelope, hook_runtime, stdout, stderr) == 0
    assert (stdout.getvalue(), stderr.getvalue()) == (baseline.getvalue(), base_error.getvalue())
    assert not list((runtime.paths.runtime_root / "injections").glob("inj_*/manifest.json"))


def test_installed_audit_hook_obeys_process_budget_and_exact_legacy_protocol(tmp_path):
    import hashlib
    import shutil
    import subprocess
    import test_retrieval_hook_process as helper
    from codex_memory.blob_store import BlobStore
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    executable = shutil.which("codex-memory-hook")
    assert executable, "real installed entry required"
    repository = helper._repository(tmp_path)
    payload = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "audit_session", "turn_id": "audit_turn", "cwd": str(repository), "prompt": "subprocess deadline"}).encode("utf-8")
    observations = []
    for mode in ("off", "shadow"):
        _audit_config(repository, mode=mode)
        started = time.perf_counter()
        child = subprocess.run([executable, "user-prompt-submit"], input=payload, capture_output=True, cwd=repository, timeout=5, check=False)
        elapsed = (time.perf_counter() - started) * 1000
        observations.append({"mode": mode, "wall_clock_ms": elapsed, "returncode": child.returncode, "stdout_sha256": hashlib.sha256(child.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(child.stderr).hexdigest()})
        if mode == "off":
            baseline = child
        else:
            shadow = child
    (tmp_path / "p25-installed-audit-evidence.json").write_text(json.dumps({"command": [executable, "user-prompt-submit"], "samples": observations}), encoding="utf-8")
    assert (shadow.stdout, shadow.stderr, shadow.returncode) == (baseline.stdout, baseline.stderr, baseline.returncode)
    assert shadow.returncode == 0 and shadow.stderr == b""
    assert observations[1]["wall_clock_ms"] < 1500, observations
    manifests = list((repository / ".codex-memory" / "injections").glob("inj_*/manifest.json"))
    assert len(manifests) == 1
    verified = InjectionStore(InjectionPaths.for_repo(repository), BlobStore(repository / ".codex-memory" / "blobs")).load(manifests[0].parent.name)
    # The configured 1000ms remains an upper bound.  The installed process
    # deadline may safely shorten ordinary shadow work after startup/Legacy.
    assert 1 <= verified.manifest.body["timing"]["budget_ms"] <= 1000
    assert verified.manifest.body["delivery"]["v2"] == "prepared"
    assert verified.manifest.body["delivery"]["host_acknowledgement"] == "unknown"
