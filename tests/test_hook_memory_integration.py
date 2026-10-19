from __future__ import annotations

import hashlib
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HOOK_HASHES = {
    ".codex/hooks.json": "F4605FD3349AD4DB99244292DF2741A646AA1BE21837FE90008B21404B4B37DE",
    ".codex/hooks/invoke.ps1": "0280B8B379EBE3F78F39D1C48ABE05BF6BAAB5781D24C6D658046CF2AD1D4938",
}


def _cli():
    return import_module("codex_memory.cli")


def _handlers():
    return import_module("codex_memory.handlers")


def _envelope(
    event_name: str,
    repository: Path,
    *,
    session_id: str = "session-memory",
    turn_id: str = "turn-memory",
    prompt: str = "CURRENT_PROMPT_MUST_NOT_BE_COPIED",
    source: str = "compact",
):
    payload: dict[str, object] = {
        "hook_event_name": event_name,
        "session_id": session_id,
        "cwd": str(repository),
    }
    if event_name in {"UserPromptSubmit", "PreCompact"}:
        payload["turn_id"] = turn_id
    if event_name == "UserPromptSubmit":
        payload["prompt"] = prompt
    elif event_name == "PreCompact":
        payload["trigger"] = "manual"
    elif event_name == "SessionStart":
        payload["source"] = source
    return import_module("codex_memory.hook_input").parse_hook_payload(payload)


class FakeEvents:
    def __init__(self, order: list[str], *, fail: bool = False, event_id: int = 17) -> None:
        self.order = order
        self.fail = fail
        self.event_id = event_id

    def append(self, envelope):
        self.order.append("append:start")
        if self.fail:
            raise RuntimeError("SECRET_APPEND_FAILURE")
        self.order.append("append:complete")
        event_store = import_module("codex_memory.event_store")
        return event_store.EventRecord(
            event_id=self.event_id,
            event_key="sha256:" + "a" * 64,
            session_id=envelope.session_id,
            turn_id=envelope.turn_id,
            event=envelope.event,
            tool_use_id=None,
            tool_name=None,
            created_at="2026-09-01T00:00:00.000000Z",
            payload_blob_sha256="b" * 64,
        )


class FakeCheckpoints:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.delivered: list[tuple[str, str]] = []

    def create_precompact(self, envelope: object) -> None:
        self.order.append("precompact")

    def mark_recovery_delivered(self, checkpoint_id: str, session_id: str) -> None:
        self.order.append("mark-delivered")
        self.delivered.append((checkpoint_id, session_id))


def _runtime(repository: Path, events: object, checkpoints: object):
    return SimpleNamespace(
        connection=object(),
        events=events,
        checkpoints=checkpoints,
        paths=SimpleNamespace(repo_root=repository.resolve()),
        blobs=object(),
    )


def _dispatch(
    command: str, envelope: object, runtime: object,
    *, output_order: list[str] | None = None,
) -> tuple[int, str, str]:
    class ObservedOutput(StringIO):
        def write(self, value: str) -> int:
            if output_order is not None:
                output_order.append("legacy-write")
            return super().write(value)

        def flush(self) -> None:
            if output_order is not None:
                output_order.append("legacy-flush")
            super().flush()

    stdout = ObservedOutput()
    stderr = StringIO()
    code = _cli()._dispatch(command, envelope, runtime, stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _base_recovery(packet: str = "BASE RECOVERY") -> object:
    response = import_module("codex_memory.recovery").session_start_output(packet)
    return SimpleNamespace(
        checkpoint_id="cp-current",
        session_id="session-memory",
        turn_id="turn-before-compact",
        response=response,
    )


def test_prepare_session_start_uses_the_configured_total_context_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    envelope = _envelope("SessionStart", repository)
    checkpoint = SimpleNamespace(
        checkpoint_id="cp-current",
        session_id="session-memory",
        turn_id="turn-before-compact",
        status="COMMITTED",
        recovery_required=True,
    )
    events = FakeEvents([])
    checkpoints = SimpleNamespace(
        latest_committed=lambda session_id: checkpoint,
    )
    observed: dict[str, object] = {}

    def orientation(
        selected: object, source_events: object, *, max_chars: int
    ) -> str:
        observed.update(
            checkpoint=selected,
            events=source_events,
            max_chars=max_chars,
        )
        return "BASE RECOVERY"

    monkeypatch.setattr(_handlers(), "build_orientation_packet", orientation)

    recovery = _handlers().prepare_session_start(envelope, events, checkpoints)

    assert recovery is not None
    assert observed == {
        "checkpoint": checkpoint,
        "events": events,
        "max_chars": 5000,
    }


def test_user_prompt_append_precedes_memory_and_emits_exact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    order: list[str] = []
    events = FakeEvents(order, event_id=37)
    checkpoints = FakeCheckpoints(order)
    runtime = _runtime(repository, events, checkpoints)
    envelope = _envelope("UserPromptSubmit", repository)
    observed: dict[str, object] = {}

    def context_id(connection: object, *, session_id: str) -> str:
        order.append("context")
        observed["context"] = (connection, session_id)
        return "cp-delivered"

    def packet(connection: object, blobs: object, **kwargs: object) -> str:
        order.append("packet")
        observed["packet"] = {"connection": connection, "blobs": blobs, **kwargs}
        return "MEMORY PACKET"

    monkeypatch.setattr(_cli(), "current_context_id", context_id, raising=False)
    monkeypatch.setattr(_cli(), "build_memory_packet", packet, raising=False)

    result = _dispatch("user-prompt-submit", envelope, runtime)

    assert result == (
        0,
        '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
        '"additionalContext":"MEMORY PACKET"}}\n',
        "",
    )
    assert order == ["append:start", "append:complete", "context", "packet"]
    assert observed["context"] == (runtime.connection, "session-memory")
    packet_call = observed["packet"]
    assert packet_call["connection"] is runtime.connection
    assert packet_call["blobs"] is runtime.blobs
    assert packet_call["session_id"] == "session-memory"
    assert packet_call["turn_id"] == "turn-memory"
    assert packet_call["context_id"] == "cp-delivered"
    assert packet_call["scope_key"] == str(repository.resolve())
    assert packet_call["prompt_event_uri"] == "codex-event://session-memory/37"
    assert packet_call["query"] == "CURRENT_PROMPT_MUST_NOT_BE_COPIED"
    assert packet_call["max_chars"] == 4000
    assert "CURRENT_PROMPT_MUST_NOT_BE_COPIED" not in result[1]


def test_shadow_runs_only_after_the_legacy_user_prompt_bytes_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    envelope = _envelope("UserPromptSubmit", repository)
    monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
    monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "LEGACY")

    class OrderedOutput(StringIO):
        def write(self, value: str) -> int:
            order.append("legacy-bytes")
            return super().write(value)

    def shadow(**kwargs: object) -> None:
        assert stdout.getvalue() == (
            '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
            '"additionalContext":"LEGACY"}}\n'
        )
        order.append("shadow")

    monkeypatch.setattr(_cli(), "_run_shadow_after_legacy", shadow, raising=False)
    stdout = OrderedOutput()
    stderr = StringIO()

    code = _cli()._dispatch("user-prompt-submit", envelope, runtime, stdout, stderr)

    assert (code, stdout.getvalue(), stderr.getvalue()) == (
        0,
        '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
        '"additionalContext":"LEGACY"}}\n',
        "",
    )
    assert order == ["append:start", "append:complete", "legacy-bytes", "shadow"]


def test_shadow_semantic_config_is_loaded_only_after_legacy_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live provider can only be constructed inside the post-output shadow seam."""
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    envelope = _envelope("UserPromptSubmit", repository)
    monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
    monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "LEGACY")

    def shadow(**kwargs: object) -> None:
        order.append("shadow")

    monkeypatch.setattr(_cli(), "_run_shadow_after_legacy", shadow, raising=False)
    class OrderedOutput(StringIO):
        def write(self, value: str) -> int:
            order.append("legacy-bytes")
            return super().write(value)
    stdout = OrderedOutput()
    stderr = StringIO()
    code = _cli()._dispatch("user-prompt-submit", envelope, runtime, stdout, stderr)

    assert (code, stderr.getvalue()) == (0, "")
    assert order[-2:] == ["legacy-bytes", "shadow"]


@pytest.mark.parametrize("status,mode", (("absent", "off"), ("invalid", "off"), ("disabled", "off")))
def test_nonsemantic_shadow_configuration_never_constructs_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, mode: str
) -> None:
    """Phase 0/off/invalid settings cannot create an optional model client."""
    from codex_memory.retrieval import config as retrieval_config

    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = _runtime(repository, FakeEvents([]), FakeCheckpoints([]))
    config_status = retrieval_config.RetrievalConfigStatus(status)
    monkeypatch.setattr(
        retrieval_config,
        "load_retrieval_config",
        lambda _root: retrieval_config.RetrievalConfig(config_status, mode),
    )
    monkeypatch.setattr(
        retrieval_config,
        "provider_from_config",
        lambda _config: pytest.fail("provider must not be constructed"),
    )

    _cli()._run_shadow_after_legacy(
        runtime=runtime,
        session_id="session",
        turn_id="turn",
        context_id="context",
        prompt_event_uri=None,
        query="private prompt",
        hook_event="UserPromptSubmit",
    )


def test_process_deadline_caps_semantic_shadow_without_restarting_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_memory.retrieval import config as retrieval_config
    from codex_memory.retrieval import shadow as retrieval_shadow

    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = _runtime(repository, FakeEvents([]), FakeCheckpoints([]))
    semantic = SimpleNamespace(query_deadline_ms=1000)
    config = SimpleNamespace(
        enabled=True,
        semantic_enabled=True,
        semantic=semantic,
        config_sha256="c" * 64,
        audit_for_session=lambda _session: None,
    )
    observed: dict[str, object] = {}
    now_ns = 9_000_000_000
    monkeypatch.setattr(retrieval_config, "load_retrieval_config", lambda _root: config)
    monkeypatch.setattr(retrieval_config, "provider_from_config", lambda _config: object())
    monkeypatch.setattr(_cli().time, "monotonic_ns", lambda: now_ns)
    monkeypatch.setattr(retrieval_shadow, "compare", lambda **kwargs: observed.update(kwargs))

    _cli()._run_shadow_after_legacy(
        runtime=runtime,
        session_id="session",
        turn_id="turn",
        context_id="context",
        prompt_event_uri=None,
        query="private prompt",
        hook_event="UserPromptSubmit",
        process_deadline_ns=now_ns + 700_000_000,
    )

    assert observed["deadline_ms"] == 1000
    assert observed["optional_deadline_ns"] == now_ns + 500_000_000
    assert observed["semantic_config"] is semantic


def test_expired_process_deadline_preserves_legacy_and_starts_no_shadow_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_memory.retrieval import config as retrieval_config

    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = _runtime(repository, FakeEvents([]), FakeCheckpoints([]))
    envelope = _envelope("UserPromptSubmit", repository)
    now_ns = 11_000_000_000
    monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
    monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "LEGACY")
    monkeypatch.setattr(_cli().time, "monotonic_ns", lambda: now_ns)
    monkeypatch.setattr(
        retrieval_config,
        "load_retrieval_config",
        lambda _root: pytest.fail("expired optional work must not read retrieval config"),
    )
    stdout = StringIO()
    stderr = StringIO()

    code = _cli()._dispatch(
        "user-prompt-submit",
        envelope,
        runtime,
        stdout,
        stderr,
        process_deadline_ns=now_ns,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    assert stdout.getvalue() == _cli()._serialize_response(
        import_module("codex_memory.recovery").user_prompt_submit_output("LEGACY")
    )


def test_late_observer_entry_uses_preallocated_absolute_budget_without_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_memory import acceptance_observer
    from codex_memory.retrieval import config as retrieval_config
    from codex_memory.retrieval import shadow as retrieval_shadow

    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = _runtime(repository, FakeEvents([]), FakeCheckpoints([]))
    semantic = SimpleNamespace(query_deadline_ms=1000)
    audit = SimpleNamespace(total_deadline_ms=1000)
    config = SimpleNamespace(
        enabled=True,
        semantic_enabled=True,
        semantic=semantic,
        config_sha256="d" * 64,
        audit_for_session=lambda _session: audit,
    )
    now_ns = 13_000_000_000
    observed: dict[str, object] = {}

    class FakeObserver:
        def emit(self, frame: str, payload: object) -> None:
            observed["frame"] = (frame, payload)

        def close(self) -> None:
            observed["closed"] = True

    fake_observer = FakeObserver()
    monkeypatch.setenv("CODEX_MEMORY_P25_OBSERVER", "1")
    monkeypatch.setattr(retrieval_config, "load_retrieval_config", lambda _root: config)
    monkeypatch.setattr(_cli().time, "monotonic_ns", lambda: now_ns)
    monkeypatch.setattr(
        acceptance_observer.Observer,
        "from_environment",
        lambda **kwargs: observed.update({"observer_init": kwargs}) or fake_observer,
    )
    monkeypatch.setattr(
        retrieval_shadow,
        "compare",
        lambda **kwargs: observed.update({"compare": kwargs}),
    )

    _cli()._run_shadow_after_legacy(
        runtime=runtime,
        session_id="session",
        turn_id="turn",
        context_id="context",
        prompt_event_uri=None,
        query="private prompt",
        hook_event="UserPromptSubmit",
        audit_context={"source": {}},
        observer_legacy={"delivery": "emitted"},
        process_deadline_ns=now_ns + 1_049_000_000,
    )

    budget = observed["observer_init"]["deadline"]
    assert budget.started_ns == now_ns - 1_000_000
    assert budget.deadline_ns == now_ns + 999_000_000
    assert budget.deadline_ns - budget.started_ns == 1_000_000_000
    assert observed["compare"]["deadline"] is budget
    assert "optional_deadline_ns" not in observed["compare"]
    assert observed["frame"] == ("legacy", {"delivery": "emitted"})
    assert observed["closed"] is True


def test_exact_outer_budget_boundary_preserves_full_acceptance_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_memory import acceptance_observer
    from codex_memory.retrieval import config as retrieval_config
    from codex_memory.retrieval import shadow as retrieval_shadow

    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = _runtime(repository, FakeEvents([]), FakeCheckpoints([]))
    semantic = SimpleNamespace(query_deadline_ms=1000)
    audit = SimpleNamespace(total_deadline_ms=1000)
    config = SimpleNamespace(
        enabled=True,
        semantic_enabled=True,
        semantic=semantic,
        config_sha256="e" * 64,
        audit_for_session=lambda _session: audit,
    )
    now_ns = 15_000_000_000
    observed: dict[str, object] = {}

    class FakeObserver:
        def emit(self, frame: str, payload: object) -> None:
            observed["frame"] = (frame, payload)

        def close(self) -> None:
            observed["closed"] = True

    fake_observer = FakeObserver()
    monkeypatch.setenv("CODEX_MEMORY_P25_OBSERVER", "1")
    monkeypatch.setattr(retrieval_config, "load_retrieval_config", lambda _root: config)
    monkeypatch.setattr(_cli().time, "monotonic_ns", lambda: now_ns)
    monkeypatch.setattr(
        acceptance_observer.Observer,
        "from_environment",
        lambda **kwargs: observed.update({"observer_init": kwargs}) or fake_observer,
    )
    monkeypatch.setattr(retrieval_shadow, "compare", lambda **kwargs: observed.update({"compare": kwargs}))

    _cli()._run_shadow_after_legacy(
        runtime=runtime,
        session_id="session",
        turn_id="turn",
        context_id="context",
        prompt_event_uri=None,
        query="private prompt",
        hook_event="UserPromptSubmit",
        audit_context={"source": {}},
        observer_legacy={"delivery": "emitted"},
        process_deadline_ns=now_ns + 1_050_000_000,
    )

    budget = observed["observer_init"]["deadline"]
    assert budget.started_ns == now_ns
    assert budget.deadline_ns == now_ns + 1_000_000_000
    assert observed["compare"]["deadline"] is budget
    assert "optional_deadline_ns" not in observed["compare"]
    assert observed["frame"] == ("legacy", {"delivery": "emitted"})
    assert observed["closed"] is True


def test_shadow_runs_only_after_the_legacy_compact_bytes_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    recovery = _base_recovery("BASE")
    monkeypatch.setattr(_cli(), "prepare_session_start", lambda *args: recovery)
    monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
    monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "MEMORY")

    class OrderedOutput(StringIO):
        def write(self, value: str) -> int:
            order.append("legacy-bytes")
            return super().write(value)

        def flush(self) -> None:
            order.append("legacy-flush")
            super().flush()

    def shadow(**kwargs: object) -> None:
        assert stdout.getvalue() == _cli()._serialize_response(
            import_module("codex_memory.recovery").session_start_output("BASE\nMEMORY")
        )
        order.append("shadow")

    monkeypatch.setattr(_cli(), "_run_shadow_after_legacy", shadow, raising=False)
    stdout = OrderedOutput()
    stderr = StringIO()

    code = _cli()._dispatch(
        "session-start", _envelope("SessionStart", repository), runtime, stdout, stderr
    )

    assert code == 0
    assert order == ["legacy-bytes", "legacy-flush", "mark-delivered", "shadow"]


@pytest.mark.parametrize("command", ("user-prompt-submit", "session-start"))
def test_shadow_boundary_failure_cannot_change_completed_legacy_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    repository = tmp_path / command
    repository.mkdir()
    runtime = _runtime(repository, FakeEvents([]), FakeCheckpoints([]))
    if command == "session-start":
        recovery = _base_recovery("BASE")
        monkeypatch.setattr(_cli(), "prepare_session_start", lambda *args: recovery)
        monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
        monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "MEMORY")
        envelope = _envelope("SessionStart", repository)
        expected = _cli()._serialize_response(
            import_module("codex_memory.recovery").session_start_output("BASE\nMEMORY")
        )
    else:
        monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
        monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "LEGACY")
        envelope = _envelope("UserPromptSubmit", repository)
        expected = (
            '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
            '"additionalContext":"LEGACY"}}\n'
        )
    monkeypatch.setattr(
        _cli(),
        "_run_shadow_after_legacy",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("shadow audit failed")),
        raising=False,
    )

    result = _dispatch(command, envelope, runtime)

    assert result == (0, expected, "")


def test_user_prompt_phase_zero_append_failure_keeps_existing_nonzero_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order, fail=True), FakeCheckpoints(order))
    monkeypatch.setattr(
        _cli(),
        "build_memory_packet",
        lambda *args, **kwargs: pytest.fail("memory must not run after append failure"),
        raising=False,
    )

    result = _dispatch(
        "user-prompt-submit", _envelope("UserPromptSubmit", repository), runtime
    )

    assert result == (1, "", "codex-memory-hook: hook execution failed\n")
    assert order == ["append:start"]
    assert "SECRET_APPEND_FAILURE" not in result[2]


def test_user_prompt_passes_configured_workspace_scope_keys_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    codex = repository / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text(
        '[scope]\nworkspace_keys = ["team-notes", "shared-lab"]\n',
        encoding="utf-8",
    )
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    observed: dict[str, object] = {}

    monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")

    def packet(connection: object, blobs: object, **kwargs: object) -> str:
        observed.update(kwargs)
        return "MEMORY PACKET"

    monkeypatch.setattr(_cli(), "build_memory_packet", packet, raising=False)

    result = _dispatch("user-prompt-submit", _envelope("UserPromptSubmit", repository), runtime)

    assert result[0] == 0
    assert observed["workspace_scope_keys"] == ("team-notes", "shared-lab")


def test_malformed_memory_config_keeps_compact_recovery_and_reports_bounded_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    codex = repository / ".codex"
    codex.mkdir()
    (codex / "memory.toml").write_text("[scope\ninvalid", encoding="utf-8")
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    recovery = _base_recovery()
    monkeypatch.setattr(_cli(), "prepare_session_start", lambda *args: recovery)
    monkeypatch.setattr(_cli(), "current_context_id", lambda *args, **kwargs: "startup")
    monkeypatch.setattr(_cli(), "build_memory_packet", lambda *args, **kwargs: "MEMORY")

    result = _dispatch("session-start", _envelope("SessionStart", repository), runtime)

    assert result[0] == 0
    assert json.loads(result[1])["hookSpecificOutput"]["additionalContext"].startswith(
        "BASE RECOVERY"
    )
    assert result[2] == (
        "codex-memory-hook: invalid memory config; workspace recall disabled\n"
    )


def test_user_prompt_memory_failure_after_append_is_isolated_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    monkeypatch.setattr(
        _cli(),
        "current_context_id",
        lambda *args, **kwargs: "startup",
        raising=False,
    )
    monkeypatch.setattr(
        _cli(),
        "build_memory_packet",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SECRET_RECORD_CONTENT")
        ),
        raising=False,
    )

    result = _dispatch(
        "user-prompt-submit", _envelope("UserPromptSubmit", repository), runtime
    )

    assert result == (
        0,
        '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
        '"additionalContext":"structured memory unavailable"}}\n',
        "",
    )
    assert order == ["append:start", "append:complete"]
    assert "SECRET_RECORD_CONTENT" not in result[1]


def test_precompact_never_calls_phase1b_and_output_is_byte_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    runtime = _runtime(repository, FakeEvents(order), FakeCheckpoints(order))
    for name in ("current_context_id", "build_memory_packet"):
        monkeypatch.setattr(
            _cli(),
            name,
            lambda *args, **kwargs: pytest.fail("PreCompact must not call Phase 1B"),
            raising=False,
        )

    result = _dispatch("pre-compact", _envelope("PreCompact", repository), runtime)

    assert result == (0, '{"continue":true}\n', "")
    assert order == ["precompact"]


def test_compact_session_start_preserves_base_prefix_and_appends_bounded_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    recovery = _base_recovery()
    checkpoints = FakeCheckpoints(order)
    runtime = _runtime(repository, FakeEvents(order), checkpoints)
    observed: dict[str, object] = {}

    def prepare(*args: object) -> object:
        order.append("prepare")
        return recovery

    def context(*args: object, **kwargs: object) -> str:
        pytest.fail("pending recovery must bind its verified checkpoint, not the last delivery")

    def packet(connection: object, blobs: object, **kwargs: object) -> str:
        order.append("memory")
        observed.update(kwargs)
        return "MEMORY AFTER COMPACT"

    monkeypatch.setattr(_cli(), "prepare_session_start", prepare)
    monkeypatch.setattr(_cli(), "current_context_id", context, raising=False)
    monkeypatch.setattr(_cli(), "build_memory_packet", packet, raising=False)

    result = _dispatch(
        "session-start", _envelope("SessionStart", repository), runtime, output_order=order
    )

    assert result[0] == 0 and result[2] == ""
    output = json.loads(result[1])
    additional = output["hookSpecificOutput"]["additionalContext"]
    assert additional == "BASE RECOVERY\nMEMORY AFTER COMPACT"
    assert additional.startswith("BASE RECOVERY") and len(additional) <= 5000
    assert order == ["prepare", "memory", "legacy-write", "legacy-flush", "mark-delivered"]
    assert checkpoints.delivered == [("cp-current", "session-memory")]
    assert observed == {
        "session_id": "session-memory",
        "turn_id": "turn-before-compact",
        "context_id": "cp-current",
        "scope_key": str(repository.resolve()),
        "prompt_event_uri": None,
        "query": None,
        "max_chars": 4000,
    }


@pytest.mark.parametrize(
    "failure", ("exception", "oversize", "empty", "no_budget")
)
def test_compact_memory_failure_or_budget_shortage_returns_exact_base_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    repository = tmp_path / f"repo-{failure}"
    repository.mkdir()
    base = "B" * 4999 if failure == "no_budget" else "BASE RECOVERY"
    recovery = _base_recovery(base)
    order: list[str] = []
    checkpoints = FakeCheckpoints(order)
    runtime = _runtime(repository, FakeEvents(order), checkpoints)
    monkeypatch.setattr(_cli(), "prepare_session_start", lambda *args: recovery)
    monkeypatch.setattr(
        _cli(), "current_context_id", lambda *args, **kwargs: "cp-current", raising=False
    )

    def packet(*args: object, **kwargs: object) -> str:
        order.append("memory")
        if failure == "exception":
            raise RuntimeError("SECRET_OPTIONAL_MEMORY")
        if failure == "no_budget":
            pytest.fail("zero remaining budget must not call memory")
        if failure == "empty":
            return ""
        return "M" * (int(kwargs["max_chars"]) + 1)

    monkeypatch.setattr(_cli(), "build_memory_packet", packet, raising=False)

    result = _dispatch(
        "session-start", _envelope("SessionStart", repository), runtime, output_order=order
    )

    expected = _cli()._serialize_response(recovery.response)
    assert result == (0, expected, "")
    assert checkpoints.delivered == [("cp-current", "session-memory")]
    if failure in {"exception", "oversize", "empty"}:
        assert order == ["memory", "legacy-write", "legacy-flush", "mark-delivered"]
    else:
        assert order == ["legacy-write", "legacy-flush", "mark-delivered"]
    assert "SECRET_OPTIONAL_MEMORY" not in result[1]


def test_compact_delivery_happens_after_optional_memory_write_and_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    order: list[str] = []
    recovery = _base_recovery()
    checkpoints = FakeCheckpoints(order)
    runtime = _runtime(repository, FakeEvents(order), checkpoints)
    real_serialize = _cli()._serialize_response

    def serialize(response: object) -> str:
        order.append("serialize")
        return real_serialize(response)

    def memory(*args: object, **kwargs: object) -> str:
        order.append("memory-fail")
        raise RuntimeError("optional failure")

    monkeypatch.setattr(_cli(), "prepare_session_start", lambda *args: recovery)
    monkeypatch.setattr(_cli(), "_serialize_response", serialize)
    monkeypatch.setattr(
        _cli(), "current_context_id", lambda *args, **kwargs: "cp-current", raising=False
    )
    monkeypatch.setattr(_cli(), "build_memory_packet", memory, raising=False)

    result = _dispatch(
        "session-start", _envelope("SessionStart", repository), runtime, output_order=order
    )

    assert result == (0, real_serialize(recovery.response), "")
    assert order == ["serialize", "memory-fail", "legacy-write", "legacy-flush", "mark-delivered"]
