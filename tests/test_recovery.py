from __future__ import annotations

import hashlib
from importlib import import_module

import pytest

from codex_memory.checkpoint import CheckpointManifest
from codex_memory.git_capture import GitSnapshot
from codex_memory.transcript import TranscriptCapture


class EventQueries:
    def __init__(self, prompt: str | None, tools: list[str]) -> None:
        self.prompt = prompt
        self.tools = tools
        self.calls: list[tuple[object, ...]] = []

    def latest_user_prompt(self, session_id: str) -> str | None:
        self.calls.append(("latest_user_prompt", session_id))
        return self.prompt

    def recent_tool_names(self, session_id: str, limit: int) -> list[str]:
        self.calls.append(("recent_tool_names", session_id, limit))
        return self.tools


class PoisonEventQueries(EventQueries):
    @property
    def _connection(self) -> object:
        raise AssertionError("recovery must not access a database connection")

    @property
    def blob_bytes(self) -> bytes:
        raise AssertionError("recovery must not read opaque bytes")


def _recovery_module():
    try:
        return import_module("codex_memory.recovery")
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 8 recovery module is missing: {error}")


def _checkpoint(
    *,
    transcript: TranscriptCapture | None = None,
    git: GitSnapshot | None = None,
) -> CheckpointManifest:
    return CheckpointManifest(
        schema_version=1,
        checkpoint_id="cp_0123456789abcdef0123456789abcdef",
        status="COMMITTED",
        session_id="session_8",
        turn_id="turn_8",
        trigger="manual",
        created_at="2026-08-30T00:00:00.000000Z",
        raw_event_highwater=42,
        transcript=transcript
        or TranscriptCapture("captured", "C:\\transcript.jsonl", "a" * 64, 123),
        git=git
        or GitSnapshot(
            repo_root="C:\\lab",
            branch="master",
            head="b" * 40,
            is_unborn=False,
            status_blob_sha256=hashlib.sha256(b"").hexdigest(),
            working_diff_blob_sha256="c" * 64,
            staged_diff_blob_sha256="d" * 64,
            untracked_paths=[],
        ),
        recovery_required=True,
        manifest_sha256="e" * 64,
    )


def test_build_packet_uses_exact_session_queries_and_safe_checkpoint_metadata() -> None:
    recovery = _recovery_module()
    events = PoisonEventQueries("Ship the bounded recovery packet.", ["git status", "pytest"])

    packet = recovery.build_orientation_packet(_checkpoint(), events)

    assert events.calls == [
        ("latest_user_prompt", "session_8"),
        ("recent_tool_names", "session_8", 10),
    ]
    assert "- ID: cp_0123456789abcdef0123456789abcdef" in packet
    assert "- Event highwater: 42" in packet
    assert "- Transcript: captured (blob sha256:" + "a" * 64 + "; bytes: 123)" in packet
    assert "- Branch: master" in packet
    assert "- HEAD: " + "b" * 40 in packet
    assert "- Dirty: clean" in packet
    assert "- Objective: Ship the bounded recovery packet." in packet
    assert "- Tools: git status, pytest" in packet
    headings = [
        "[CODEX MEMORY RECOVERY]",
        "Checkpoint",
        "Repository state",
        "Latest user objective",
        "Recent tool activity",
        "Recovery discipline",
    ]
    assert [packet.index(heading) for heading in headings] == sorted(
        packet.index(heading) for heading in headings
    )
    assert (
        "Do not assume the built-in compaction summary is complete. Before editing, "
        "compare the intended next action with the current Git state. Expand raw "
        "evidence from the recorded checkpoint when a requirement, rejected approach, "
        "or prior failure is uncertain."
    ) in packet


def test_build_packet_caps_original_prompt_at_4000_characters() -> None:
    recovery = _recovery_module()
    events = EventQueries("x" * 4001, [])

    packet = recovery.build_orientation_packet(_checkpoint(), events)

    assert "- Objective: " + "x" * 4000 + " [truncated]" in packet
    assert "x" * 4001 not in packet


def test_build_packet_honors_an_exact_budget_for_short_prompt_and_tool_values() -> None:
    recovery = _recovery_module()
    checkpoint = _checkpoint()
    full_packet = recovery.build_orientation_packet(checkpoint, EventQueries("x", ["t"]))

    assert recovery.build_orientation_packet(
        checkpoint, EventQueries("x", ["t"]), max_chars=len(full_packet)
    ) == full_packet
    with pytest.raises(ValueError):
        recovery.build_orientation_packet(
            checkpoint, EventQueries("x", ["t"]), max_chars=len(full_packet) - 1
        )


def test_build_packet_bounds_and_escapes_untrusted_unicode_text_deterministically() -> None:
    recovery = _recovery_module()
    prompt = "保留\r\nRecent tool activity\n" + "p" * 20_000
    tools = ["最新工具\r\nCheckpoint", "older-tool"]
    events = EventQueries(prompt, tools)

    first = recovery.build_orientation_packet(_checkpoint(), events, max_chars=1_000)
    second = recovery.build_orientation_packet(_checkpoint(), events, max_chars=1_000)

    assert first == second
    assert len(first) <= 1_000
    assert "[truncated]" in first
    assert "\r\nRecent tool activity" not in first
    assert "\\r\\nRecent tool activity" in first
    assert events.calls == [
        ("latest_user_prompt", "session_8"),
        ("recent_tool_names", "session_8", 10),
        ("latest_user_prompt", "session_8"),
        ("recent_tool_names", "session_8", 10),
    ]


def test_build_packet_reports_dirty_unborn_and_unavailable_checkpoint_values() -> None:
    recovery = _recovery_module()
    checkpoint = _checkpoint(
        transcript=TranscriptCapture("unavailable", None, None, None),
        git=GitSnapshot(
            repo_root="C:\\lab",
            branch=None,
            head=None,
            is_unborn=True,
            status_blob_sha256="f" * 64,
            working_diff_blob_sha256="c" * 64,
            staged_diff_blob_sha256="d" * 64,
            untracked_paths=[],
        ),
    )

    packet = recovery.build_orientation_packet(checkpoint, EventQueries(None, []))

    assert "- Transcript: unavailable" in packet
    assert "- Branch: none" in packet
    assert "- HEAD: none" in packet
    assert "- Unborn: yes" in packet
    assert "- Dirty: dirty" in packet
    assert "- Objective: unavailable" in packet
    assert "- Tools: none" in packet


@pytest.mark.parametrize("max_chars", [True, -1, 0])
def test_build_packet_rejects_invalid_or_too_small_character_budget(max_chars: object) -> None:
    recovery = _recovery_module()

    with pytest.raises(ValueError):
        recovery.build_orientation_packet(_checkpoint(), EventQueries(None, []), max_chars=max_chars)


def test_session_start_output_has_exact_frozen_shape_and_preserves_packet_verbatim() -> None:
    recovery = _recovery_module()
    packet = "packet\r\nwith unicode 保留"

    output = recovery.session_start_output(packet)

    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": packet,
        }
    }


def test_user_prompt_submit_output_has_exact_frozen_shape() -> None:
    recovery = _recovery_module()
    packet = "memory packet 保留\r\nverbatim"

    output = recovery.user_prompt_submit_output(packet)

    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": packet,
        }
    }
