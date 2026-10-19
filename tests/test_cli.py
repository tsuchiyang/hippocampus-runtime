from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from codex_memory.checkpoint import CheckpointError, CheckpointManifest
from codex_memory.git_capture import GitSnapshot
from codex_memory.transcript import TranscriptCapture


COMMAND_EVENTS = {
    "user-prompt-submit": "UserPromptSubmit",
    "post-tool-use": "PostToolUse",
    "pre-compact": "PreCompact",
    "post-compact": "PostCompact",
    "session-start": "SessionStart",
}


def _cli():
    try:
        return import_module("codex_memory.cli")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Task 9 CLI module is missing: {exc}")


def _payload(event: str, cwd: Path, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "hook_event_name": event,
        "session_id": "session_9",
        "cwd": str(cwd),
    }
    if event in {"UserPromptSubmit", "PostToolUse", "PreCompact", "PostCompact"}:
        value["turn_id"] = "turn_9"
    if event == "UserPromptSubmit":
        value["prompt"] = "保留这个目标"
    elif event == "PostToolUse":
        value.update(
            tool_name="exec_command",
            tool_use_id="tool_9",
            tool_input={"command": "git status"},
            tool_response={"output": "clean"},
        )
    elif event in {"PreCompact", "PostCompact"}:
        value["trigger"] = "manual"
    elif event == "SessionStart":
        value["source"] = "startup"
    value.update(overrides)
    return value


def _invoke(
    arguments: list[str], payload: object | str, *, stdin_prefix: str = ""
) -> tuple[int, str, str]:
    cli = _cli()
    raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    stdout = StringIO()
    stderr = StringIO()
    result = cli.main(arguments, StringIO(stdin_prefix + raw), stdout, stderr)
    return result, stdout.getvalue(), stderr.getvalue()


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEvents:
    def __init__(self, *, fail: bool = False) -> None:
        self.appended: list[object] = []
        self.fail = fail

    def append(self, envelope: object) -> object:
        if self.fail:
            raise RuntimeError("sensitive event failure")
        self.appended.append(envelope)
        return SimpleNamespace(event_id=len(self.appended))

    def latest_user_prompt(self, session_id: str) -> str:
        return "继续进行 Unicode 恢复"

    def recent_tool_names(self, session_id: str, limit: int) -> list[str]:
        return ["pytest"]


class FakeCheckpoints:
    def __init__(self, manifest: CheckpointManifest | None = None) -> None:
        self.manifest = manifest
        self.precompact: list[object] = []
        self.postcompact: list[object] = []
        self.delivered: list[tuple[str, str]] = []
        self.fail_pre = False
        self.fail_post = False
        self.fail_latest = False
        self.fail_delivery = False

    def create_precompact(self, envelope: object) -> object:
        if self.fail_pre:
            raise CheckpointError("secret checkpoint failure")
        self.precompact.append(envelope)
        return object()

    def mark_postcompact(self, envelope: object) -> object | None:
        if self.fail_post:
            raise CheckpointError("secret postcompact failure")
        self.postcompact.append(envelope)
        return self.manifest

    def latest_committed(self, session_id: str) -> CheckpointManifest | None:
        if self.fail_latest:
            raise CheckpointError("secret recovery failure")
        return self.manifest

    def mark_recovery_delivered(self, checkpoint_id: str, session_id: str) -> None:
        if self.fail_delivery:
            raise CheckpointError("secret delivery failure")
        self.delivered.append((checkpoint_id, session_id))


def _manifest(**changes: object) -> CheckpointManifest:
    original = CheckpointManifest(
        schema_version=1,
        checkpoint_id="cp_0123456789abcdef0123456789abcdef",
        status="COMMITTED",
        session_id="session_9",
        turn_id="turn_9",
        trigger="manual",
        created_at="2026-08-30T00:00:00.000000Z",
        raw_event_highwater=9,
        transcript=TranscriptCapture("unavailable", None, None, None),
        git=GitSnapshot(
            repo_root="C:\\repo with spaces",
            branch="master",
            head=None,
            is_unborn=True,
            status_blob_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            working_diff_blob_sha256="a" * 64,
            staged_diff_blob_sha256="b" * 64,
            untracked_paths=[],
        ),
        recovery_required=True,
        manifest_sha256="c" * 64,
    )
    return replace(original, **changes)


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: FakeEvents | None = None,
    checkpoints: FakeCheckpoints | None = None,
) -> tuple[FakeConnection, FakeEvents, FakeCheckpoints]:
    cli = _cli()
    connection = FakeConnection()
    fake_events = events or FakeEvents()
    fake_checkpoints = checkpoints or FakeCheckpoints()
    runtime = SimpleNamespace(
        connection=connection,
        events=fake_events,
        checkpoints=fake_checkpoints,
        paths=SimpleNamespace(repo_root=Path.cwd().resolve()),
        blobs=object(),
    )
    monkeypatch.setattr(cli, "_resolve_repo_root", lambda cwd: Path(cwd))
    monkeypatch.setattr(cli, "_construct_runtime", lambda root: runtime)
    monkeypatch.setattr(
        cli,
        "current_context_id",
        lambda connection, *, session_id: "startup",
        raising=False,
    )

    def fake_memory_packet(*args: object, **kwargs: object) -> str:
        if kwargs.get("prompt_event_uri") is None:
            raise RuntimeError("optional compact memory unavailable")
        return "fake memory packet"

    monkeypatch.setattr(cli, "build_memory_packet", fake_memory_packet, raising=False)
    return connection, fake_events, fake_checkpoints


@pytest.mark.parametrize("command,event", COMMAND_EVENTS.items())
def test_exact_command_mapping_dispatches_expected_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, event: str
) -> None:
    checkpoints = FakeCheckpoints(_manifest())
    _, events, service = _install_fake_runtime(monkeypatch, checkpoints=checkpoints)
    payload = _payload(event, tmp_path)

    code, stdout, stderr = _invoke([command], payload)

    assert code == 0
    assert stderr == ""
    if event in {"UserPromptSubmit", "PostToolUse", "SessionStart"}:
        assert [item.event.value for item in events.appended] == [event]
    else:
        calls = service.precompact if event == "PreCompact" else service.postcompact
        assert [item.event.value for item in calls] == [event]
    if event == "UserPromptSubmit":
        assert stdout == (
            '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
            '"additionalContext":"fake memory packet"}}\n'
        )
    elif event in {"PostToolUse", "SessionStart"}:
        assert stdout == ""
    else:
        assert stdout == '{"continue":true}\n'


def test_lifecycle_entry_propagates_one_absolute_process_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    _install_fake_runtime(monkeypatch)
    observed: dict[str, object] = {}
    original_dispatch = cli._dispatch

    def dispatch(*args: object, **kwargs: object) -> int:
        observed["process_deadline_ns"] = kwargs.get("process_deadline_ns")
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    started_ns = 7_000_000_000
    stdout = StringIO()
    stderr = StringIO()

    result = cli.main(
        ["user-prompt-submit"],
        StringIO(json.dumps(_payload("UserPromptSubmit", tmp_path))),
        stdout,
        stderr,
        _process_started_ns=started_ns,
    )

    assert result == 0
    assert stderr.getvalue() == ""
    assert observed == {
        "process_deadline_ns": started_ns + 1_500_000_000,
    }


def test_installed_entry_fails_optional_deadline_closed_when_start_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    _install_fake_runtime(monkeypatch)
    observed: dict[str, object] = {}
    original_dispatch = cli._dispatch

    def dispatch(*args: object, **kwargs: object) -> int:
        observed["process_deadline_ns"] = kwargs.get("process_deadline_ns")
        return original_dispatch(*args, **kwargs)

    stdin = StringIO(json.dumps(_payload("UserPromptSubmit", tmp_path)))
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(cli, "_installed_process_started_ns", lambda: None)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    monkeypatch.setattr(cli.sys, "argv", ["codex-memory-hook", "user-prompt-submit"])
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    result = cli.main()

    assert result == 0
    assert stderr.getvalue() == ""
    assert observed == {"process_deadline_ns": 0}


def test_installed_observer_entry_uses_parent_conservative_process_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    _install_fake_runtime(monkeypatch)
    observed: dict[str, object] = {}
    started_ns = max(0, cli._MODULE_ENTRY_NS - 500_000_000)
    monkeypatch.setenv(
        "CODEX_MEMORY_P25_OBSERVER",
        json.dumps({"process_started_ns": started_ns}),
    )
    monkeypatch.setattr(
        cli,
        "_installed_process_started_ns",
        lambda: pytest.fail("Observer activation must own the shared floor"),
    )
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda *_args, **kwargs: observed.update(
            process_deadline_ns=kwargs.get("process_deadline_ns")
        ) or 0,
    )
    monkeypatch.setattr(cli.sys, "argv", ["codex-memory-hook", "user-prompt-submit"])
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        StringIO(json.dumps(_payload("UserPromptSubmit", tmp_path))),
    )
    monkeypatch.setattr(cli.sys, "stdout", StringIO())
    monkeypatch.setattr(cli.sys, "stderr", StringIO())

    assert cli.main() == 0
    assert observed == {
        "process_deadline_ns": started_ns + 1_500_000_000,
    }


@pytest.mark.parametrize(
    "activation",
    [
        {},
        {"process_started_ns": True},
        {"process_started_ns": -1},
        {"process_started_ns": "1"},
        {"process_started_ns": 2**63},
    ],
)
def test_installed_observer_entry_rejects_invalid_parent_floor_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation: dict[str, object],
) -> None:
    cli = _cli()
    _install_fake_runtime(monkeypatch)
    observed: dict[str, object] = {}
    monkeypatch.setenv(
        "CODEX_MEMORY_P25_OBSERVER",
        json.dumps(activation),
    )
    monkeypatch.setattr(
        cli,
        "_installed_process_started_ns",
        lambda: pytest.fail("invalid Observer activation must not fall back"),
    )
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda *_args, **kwargs: observed.update(
            process_deadline_ns=kwargs.get("process_deadline_ns")
        ) or 0,
    )
    monkeypatch.setattr(cli.sys, "argv", ["codex-memory-hook", "user-prompt-submit"])
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        StringIO(json.dumps(_payload("UserPromptSubmit", tmp_path))),
    )
    monkeypatch.setattr(cli.sys, "stdout", StringIO())
    monkeypatch.setattr(cli.sys, "stderr", StringIO())

    assert cli.main() == 0
    assert observed == {"process_deadline_ns": 0}


def test_installed_non_shadow_hook_does_not_probe_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    _install_fake_runtime(monkeypatch)
    stdin = StringIO(json.dumps(_payload("PostToolUse", tmp_path)))
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(
        cli,
        "_installed_process_started_ns",
        lambda: pytest.fail("non-shadow hook must not probe process timing"),
    )
    monkeypatch.setattr(cli.sys, "argv", ["codex-memory-hook", "post-tool-use"])
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    result = cli.main()

    assert result == 0
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process FILETIME contract")
def test_windows_process_start_precedes_module_entry() -> None:
    cli = _cli()

    started_ns = cli._windows_process_started_ns()

    assert started_ns is not None
    assert 0 <= started_ns <= cli._MODULE_ENTRY_NS <= cli.time.monotonic_ns()


@pytest.mark.parametrize("lookup_ns", [0, 300_000_000, 750_000_000])
@pytest.mark.parametrize("sample_ns", [0, 5_000_000])
def test_windows_process_clock_bridge_does_not_charge_lookup_time_twice(
    monkeypatch: pytest.MonkeyPatch, lookup_ns: int, sample_ns: int,
) -> None:
    """Only uncertainty in the UTC sample may backdate the real process start."""
    import ctypes

    cli = _cli()
    process_start_ns = 1_000_000_000
    clock = [process_start_ns + 100_000_000]
    epoch_100ns = 116_444_736_000_000_000

    def write_filetime(pointer, ticks):
        pointer._obj.low = ticks & 0xFFFF_FFFF
        pointer._obj.high = ticks >> 32

    def process_times(_handle, creation, *_unused):
        write_filetime(creation, epoch_100ns + process_start_ns // 100)
        # Win32 process lookup/ancestry work takes time. Both clocks advance
        # together, exactly as in production; no deadline is moved or reset.
        clock[0] += lookup_ns
        return 1

    def system_time(pointer):
        clock[0] += sample_ns
        write_filetime(pointer, epoch_100ns + clock[0] // 100)

    kernel = SimpleNamespace(
        GetCurrentProcess=lambda: -1,
        GetProcessTimes=process_times,
        CreateToolhelp32Snapshot=lambda *_args: ctypes.c_void_p(-1).value,
        Process32FirstW=lambda *_args: 0,
        Process32NextW=lambda *_args: 0,
        OpenProcess=lambda *_args: 0,
        QueryFullProcessImageNameW=lambda *_args: 0,
        CloseHandle=lambda *_args: 1,
        GetSystemTimePreciseAsFileTime=system_time,
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel, raising=False)
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    monkeypatch.setattr(cli, "_MODULE_ENTRY_NS", process_start_ns + 50_000_000)

    observed_start = cli._windows_process_started_ns()

    assert observed_start == process_start_ns - sample_ns
    assert observed_start <= process_start_ns


def test_process_creation_filetime_maps_conservatively_to_monotonic_clock() -> None:
    cli = _cli()

    mapped = cli._map_process_creation_to_monotonic(
        creation_100ns=100,
        current_100ns=105,
        monotonic_before_ns=10_000,
        monotonic_after_ns=10_100,
        module_entry_ns=9_600,
    )

    assert mapped == 9_500


def test_installed_entry_uses_earlier_trusted_sibling_launcher_creation() -> None:
    cli = _cli()

    selected = cli._select_installed_entry_creation_100ns(
        child_creation_100ns=200,
        python_executable=r"C:\Lab Env\Scripts\python.exe",
        parent_executable=r"c:\lab env\scripts\CODEX-MEMORY-HOOK.EXE",
        parent_creation_100ns=100,
    )

    assert selected == 100


@pytest.mark.parametrize(
    ("parent_executable", "parent_creation_100ns"),
    [
        (r"C:\Lab Env\Scripts\python.exe", 100),
        (r"C:\Other\codex-memory-hook.exe", 100),
        (r"C:\Lab Env\Scripts\codex-memory-hook.exe", 200),
        (r"C:\Lab Env\Scripts\codex-memory-hook.exe", 201),
        (r"C:\Lab Env\Scripts\codex-memory-hook.exe", None),
        (r"C:\Lab Env\Scripts\codex-memory-hook.exe", True),
        (r"C:\Lab Env\Scripts\codex-memory-hook.exe", 0),
        (r"C:\Lab Env\Scripts\codex-memory-hook.exe", -1),
        (r"codex-memory-hook.exe", 100),
        (r"\\?\C:\Lab Env\Scripts\codex-memory-hook.exe", 100),
        (None, 100),
    ],
)
def test_untrusted_or_non_ancestor_launcher_keeps_child_creation(
    parent_executable: str | None,
    parent_creation_100ns: int | None,
) -> None:
    cli = _cli()

    selected = cli._select_installed_entry_creation_100ns(
        child_creation_100ns=200,
        python_executable=r"C:\Lab Env\Scripts\python.exe",
        parent_executable=parent_executable,
        parent_creation_100ns=parent_creation_100ns,
    )

    assert selected == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"creation_100ns": True},
        {"current_100ns": 99},
        {"monotonic_after_ns": 9_999},
        {"monotonic_before_ns": 400},
        {"module_entry_ns": 9_499},
    ],
)
def test_invalid_process_creation_mapping_fails_closed(
    overrides: dict[str, object],
) -> None:
    cli = _cli()
    values: dict[str, object] = {
        "creation_100ns": 100,
        "current_100ns": 105,
        "monotonic_before_ns": 10_000,
        "monotonic_after_ns": 10_100,
        "module_entry_ns": 9_600,
    }
    values.update(overrides)

    assert cli._map_process_creation_to_monotonic(**values) is None


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["pre-compact", "extra"]])
def test_unknown_or_wrong_arity_command_is_generic_failure(arguments: list[str]) -> None:
    code, stdout, stderr = _invoke(arguments, "{}")

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: unsupported command\n"


def test_memory_prefix_routes_exact_argv_and_streams_without_parsing_hook_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    observed: dict[str, object] = {}

    def run_memory(arguments, stdout, stderr, resolver) -> int:
        observed.update(
            arguments=list(arguments),
            stdout=stdout,
            stderr=stderr,
            resolver=resolver,
        )
        return 73

    monkeypatch.setattr(cli, "run_memory_command", run_memory)
    stdin = StringIO("not hook JSON and must remain unread")
    stdout = StringIO()
    stderr = StringIO()

    code = cli.main(
        ["memory", "pending", "--session-id", "session-a"],
        stdin,
        stdout,
        stderr,
    )

    assert code == 73
    assert observed == {
        "arguments": ["memory", "pending", "--session-id", "session-a"],
        "stdout": stdout,
        "stderr": stderr,
        "resolver": cli._resolve_repo_root,
    }
    assert stdin.tell() == 0


def test_memory_router_does_not_change_legacy_record_or_ledger_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    routed: list[list[str]] = []

    def run_ledger(arguments, stdout, stderr, resolver) -> int:
        routed.append(list(arguments))
        return 41

    monkeypatch.setattr(cli, "run_ledger_command", run_ledger)
    monkeypatch.setattr(
        cli,
        "run_memory_command",
        lambda *args: pytest.fail("legacy prefix must not reach memory routing"),
    )

    assert cli.main(["record", "list"], StringIO(""), StringIO(), StringIO()) == 41
    assert cli.main(["ledger", "verify"], StringIO(""), StringIO(), StringIO()) == 41
    assert routed == [["record", "list"], ["ledger", "verify"]]


def test_cli_module_executes_main_when_invoked_with_dash_m() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "codex_memory.cli", "unknown"],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "codex-memory-hook: unsupported command\n"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \r\n",
        "{",
        "[]",
        '"scalar"',
        "1",
        "null",
        '{"x":NaN}',
        '{"x":Infinity}',
        '{} {}',
        '{} trailing',
    ],
)
def test_strict_json_parser_rejects_malformed_nonobject_and_trailing_data(raw: str) -> None:
    code, stdout, stderr = _invoke(["session-start"], raw)

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: invalid hook input\n"
    assert "Traceback" not in stderr


def test_leading_whitespace_and_unicode_object_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime(monkeypatch)
    payload = _payload("UserPromptSubmit", tmp_path, prompt="目标 保留")

    code, stdout, stderr = _invoke(
        ["user-prompt-submit"], payload, stdin_prefix=" \r\n\t"
    )

    assert code == 0 and stderr == ""
    assert json.loads(stdout) == {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "fake memory packet",
        }
    }


@pytest.mark.parametrize("number", ["1e309", "-1e309"])
def test_overflowing_json_number_is_rejected_before_runtime_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: str
) -> None:
    cli = _cli()
    payload = json.dumps(
        _payload("UserPromptSubmit", tmp_path), ensure_ascii=False
    )
    raw = payload[:-1] + f',"overflow":{number}' + "}"
    monkeypatch.setattr(
        cli,
        "_resolve_repo_root",
        lambda cwd: pytest.fail("non-finite input must not reach repository lookup"),
    )
    monkeypatch.setattr(
        cli,
        "_construct_runtime",
        lambda root: pytest.fail("non-finite input must not construct runtime"),
    )

    code, stdout, stderr = _invoke(["user-prompt-submit"], raw)

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: invalid hook input\n"


@pytest.mark.parametrize("number", ["1e308", "-1e308", "0.125"])
def test_finite_json_float_remains_accepted(number: str) -> None:
    parsed = _cli()._parse_json_object(f'{{"number":{number}}}')

    assert parsed == {"number": float(number)}


def test_command_event_mismatch_fails_before_runtime_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    constructed = False

    def forbidden(root: Path) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError

    monkeypatch.setattr(cli, "_construct_runtime", forbidden)
    code, stdout, stderr = _invoke(
        ["pre-compact"], _payload("PostCompact", tmp_path)
    )

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: hook event mismatch\n"
    assert constructed is False


def test_git_root_resolution_uses_path_safe_raw_byte_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    cwd = tmp_path / "repo with spaces"
    root = tmp_path / "resolved repo"
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, str(root).encode("utf-8") + b"\n", b"")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._resolve_repo_root(cwd) == root.resolve()
    assert observed["command"] == ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"]
    kwargs = observed["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["text"] is False
    assert kwargs["timeout"] == 10


def test_git_root_failure_is_generic_and_does_not_construct_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    monkeypatch.setattr(
        cli,
        "_resolve_repo_root",
        lambda cwd: (_ for _ in ()).throw(RuntimeError("C:\\secret\\repo")),
    )
    monkeypatch.setattr(cli, "_construct_runtime", lambda root: pytest.fail("must not construct"))

    code, stdout, stderr = _invoke(
        ["user-prompt-submit"], _payload("UserPromptSubmit", tmp_path)
    )

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: runtime unavailable\n"
    assert "secret" not in stderr


def test_prompt_and_tool_append_once_and_close_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for command, event in [
        ("user-prompt-submit", "UserPromptSubmit"),
        ("post-tool-use", "PostToolUse"),
    ]:
        connection, events, _ = _install_fake_runtime(monkeypatch)
        code, stdout, stderr = _invoke([command], _payload(event, tmp_path))
        assert code == 0 and stderr == ""
        if command == "user-prompt-submit":
            assert json.loads(stdout)["hookSpecificOutput"]["additionalContext"] == (
                "fake memory packet"
            )
        else:
            assert stdout == ""
        assert [item.event.value for item in events.appended] == [event]
        assert connection.closed is True


def test_runtime_connection_closes_when_handler_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = FakeEvents(fail=True)
    connection, _, _ = _install_fake_runtime(monkeypatch, events=events)

    code, stdout, stderr = _invoke(
        ["user-prompt-submit"], _payload("UserPromptSubmit", tmp_path)
    )

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: hook execution failed\n"
    assert "sensitive" not in stderr
    assert connection.closed is True


def test_precompact_failure_is_structured_block_at_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = FakeCheckpoints()
    checkpoints.fail_pre = True
    _install_fake_runtime(monkeypatch, checkpoints=checkpoints)

    code, stdout, stderr = _invoke(
        ["pre-compact"], _payload("PreCompact", tmp_path)
    )

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "continue": False,
        "stopReason": "Durable checkpoint unavailable",
        "systemMessage": "Codex Memory Runtime could not verify durable state.",
    }
    assert stdout.endswith("\n") and stdout.count("\n") == 1
    assert "secret" not in stdout


def test_postcompact_failure_is_nonzero_with_empty_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = FakeCheckpoints()
    checkpoints.fail_post = True
    _install_fake_runtime(monkeypatch, checkpoints=checkpoints)

    code, stdout, stderr = _invoke(
        ["post-compact"], _payload("PostCompact", tmp_path)
    )

    assert code != 0
    assert stdout == ""
    assert stderr == "codex-memory-hook: hook execution failed\n"


def test_postcompact_without_checkpoint_is_successful_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime(monkeypatch, checkpoints=FakeCheckpoints(None))

    assert _invoke(["post-compact"], _payload("PostCompact", tmp_path)) == (
        0,
        '{"continue":true}\n',
        "",
    )


@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_noncompact_session_start_appends_once_without_lookup_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    class NoLookup(FakeCheckpoints):
        def latest_committed(self, session_id: str) -> CheckpointManifest | None:
            raise AssertionError("noncompact source must not query checkpoints")

    _, events, _ = _install_fake_runtime(monkeypatch, checkpoints=NoLookup())

    result = _invoke(
        ["session-start"], _payload("SessionStart", tmp_path, source=source)
    )

    assert result == (0, "", "")
    assert len(events.appended) == 1


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        _manifest(session_id="another-session"),
        _manifest(status="PREPARING"),  # type: ignore[arg-type]
        _manifest(recovery_required=False),
    ],
)
def test_compact_session_start_fails_closed_for_invalid_recovery_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: CheckpointManifest | None,
) -> None:
    checkpoints = FakeCheckpoints(manifest)
    _, events, _ = _install_fake_runtime(monkeypatch, checkpoints=checkpoints)

    code, stdout, stderr = _invoke(
        ["session-start"], _payload("SessionStart", tmp_path, source="compact")
    )

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["continue"] is False
    assert len(events.appended) == 1
    assert checkpoints.delivered == []


def test_compact_recovery_serializes_writes_and_flushes_before_delivery_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    checkpoints = FakeCheckpoints(_manifest())
    _install_fake_runtime(monkeypatch, checkpoints=checkpoints)
    order: list[str] = []
    original_serialize = cli._serialize_response
    original_prepare = cli.prepare_session_start
    original_mark = checkpoints.mark_recovery_delivered

    def prepare(envelope: object, events: object, service: object) -> object:
        order.append("construct")
        return original_prepare(envelope, events, service)

    def serialize(response: object) -> str:
        order.append("serialize")
        return original_serialize(response)

    def mark(checkpoint_id: str, session_id: str) -> None:
        order.append("mark")
        original_mark(checkpoint_id, session_id)

    class OrderedOutput(StringIO):
        def write(self, text: str) -> int:
            order.append("write")
            return super().write(text)

        def flush(self) -> None:
            order.append("flush")
            return super().flush()

    monkeypatch.setattr(cli, "prepare_session_start", prepare)
    monkeypatch.setattr(cli, "_serialize_response", serialize)
    checkpoints.mark_recovery_delivered = mark  # type: ignore[method-assign]
    stdout = OrderedOutput()
    stderr = StringIO()

    code = cli.main(
        ["session-start"],
        StringIO(json.dumps(_payload("SessionStart", tmp_path, source="compact"))),
        stdout,
        stderr,
    )

    assert code == 0 and stderr.getvalue() == ""
    assert order == ["construct", "serialize", "write", "flush", "mark"]
    response = json.loads(stdout.getvalue())
    assert response["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "继续进行 Unicode 恢复" in response["hookSpecificOutput"]["additionalContext"]
    assert stdout.getvalue().endswith("\n") and "\\u" not in stdout.getvalue()


def test_compact_settlement_failure_never_appends_a_second_frame_after_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = FakeCheckpoints(_manifest())
    checkpoints.fail_delivery = True
    _install_fake_runtime(monkeypatch, checkpoints=checkpoints)

    code, stdout, stderr = _invoke(
        ["session-start"], _payload("SessionStart", tmp_path, source="compact")
    )

    assert code == 1 and stderr == "codex-memory-hook: recovery settlement failed\n"
    assert json.loads(stdout)["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "additionalContext" in stdout
    assert "continue" not in stdout
    assert stdout.count("\n") == 1
    assert "secret" not in stdout


def test_compact_serialization_failure_does_not_consume_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    checkpoints = FakeCheckpoints(_manifest())
    _install_fake_runtime(monkeypatch, checkpoints=checkpoints)
    monkeypatch.setattr(
        cli,
        "_serialize_response",
        lambda response: (_ for _ in ()).throw(ValueError("untrusted serialization")),
    )

    code, stdout, stderr = _invoke(
        ["session-start"], _payload("SessionStart", tmp_path, source="compact")
    )

    assert code == 0 and stderr == ""
    assert json.loads(stdout)["continue"] is False
    assert checkpoints.delivered == []
    assert "untrusted" not in stdout


def test_compact_recovery_is_injected_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()

    class OneShot(FakeCheckpoints):
        def mark_recovery_delivered(self, checkpoint_id: str, session_id: str) -> None:
            super().mark_recovery_delivered(checkpoint_id, session_id)
            self.manifest = replace(manifest, recovery_required=False)

    checkpoints = OneShot(manifest)
    _install_fake_runtime(monkeypatch, checkpoints=checkpoints)
    payload = _payload("SessionStart", tmp_path, source="compact")

    first = _invoke(["session-start"], payload)
    second = _invoke(["session-start"], payload)

    assert first[0] == 0 and "additionalContext" in first[1]
    assert second[0] == 0 and json.loads(second[1])["continue"] is False
    assert len(checkpoints.delivered) == 1


def test_runtime_construction_closes_connection_when_migration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    connection = FakeConnection()
    monkeypatch.setattr(cli, "connect_database", lambda path: connection)
    monkeypatch.setattr(
        cli,
        "migrate_database",
        lambda candidate: (_ for _ in ()).throw(RuntimeError("migration failed")),
    )

    with pytest.raises(RuntimeError, match="migration failed"):
        cli._construct_runtime(tmp_path)

    assert connection.closed is True


def test_real_unborn_repository_lifecycle(tmp_path: Path) -> None:
    repository = tmp_path / "unborn repo with spaces"
    repository.mkdir()
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    session = "integration-session"
    base = {"session_id": session}

    prompt_result = _invoke(
        ["user-prompt-submit"],
        _payload("UserPromptSubmit", repository, **base),
    )
    assert prompt_result[0] == 0 and prompt_result[2] == ""
    assert json.loads(prompt_result[1])["hookSpecificOutput"]["hookEventName"] == (
        "UserPromptSubmit"
    )
    pre = _invoke(
        ["pre-compact"], _payload("PreCompact", repository, **base)
    )
    post = _invoke(
        ["post-compact"], _payload("PostCompact", repository, **base)
    )
    start = _invoke(
        ["session-start"],
        _payload("SessionStart", repository, source="compact", **base),
    )

    assert pre == (0, '{"continue":true}\n', "")
    assert post == (0, '{"continue":true}\n', "")
    assert start[0] == 0 and start[2] == ""
    context = json.loads(start[1])["hookSpecificOutput"]["additionalContext"]
    assert "- Unborn: yes" in context
    assert "Checkpoint" in context
