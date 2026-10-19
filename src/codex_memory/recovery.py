"""Build bounded, evidence-only compaction recovery context."""

from __future__ import annotations

import hashlib
from typing import Protocol

from codex_memory.checkpoint import CheckpointManifest


_PROMPT_LIMIT = 4_000
_TRUNCATION_MARKER = " [truncated]"
_EMPTY_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
SESSION_START_CONTEXT_LIMIT = 5_000
_DISCIPLINE = (
    "Do not assume the built-in compaction summary is complete. Before editing, "
    "compare the intended next action with the current Git state. Expand raw "
    "evidence from the recorded checkpoint when a requirement, rejected approach, "
    "or prior failure is uncertain."
)


class RecoveryEvents(Protocol):
    """The two session-scoped event queries used for recovery context."""

    def latest_user_prompt(self, session_id: str) -> str | None: ...

    def recent_tool_names(self, session_id: str, limit: int) -> list[str]: ...


def build_orientation_packet(
    checkpoint: CheckpointManifest,
    events: RecoveryEvents,
    max_chars: int = 12_000,
) -> str:
    """Render deterministic, bounded recovery context from committed evidence."""
    _validate_max_chars(max_chars)

    prompt = events.latest_user_prompt(checkpoint.session_id)
    tool_names = events.recent_tool_names(checkpoint.session_id, limit=10)

    prompt_candidate, prompt_placeholder = _prompt_values(prompt)
    tools_candidate, tools_placeholder = _tool_values(tool_names)
    minimum = _render_packet(
        checkpoint,
        prompt_placeholder,
        tools_placeholder,
    )
    if len(minimum) > max_chars:
        raise ValueError("max_chars cannot contain the immutable recovery structure")

    remaining = max_chars - len(minimum)
    prompt_value, remaining = _fit_value(
        prompt_candidate, prompt_placeholder, remaining
    )
    tools_value, _ = _fit_value(tools_candidate, tools_placeholder, remaining)
    packet = _render_packet(checkpoint, prompt_value, tools_value)
    if len(packet) > max_chars:
        raise AssertionError("recovery packet exceeded its character budget")
    return packet


def session_start_output(packet: str) -> dict[str, object]:
    """Return the exact frozen SessionStart hook output shape."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": packet,
        }
    }


def user_prompt_submit_output(packet: str) -> dict[str, object]:
    """Return the exact frozen UserPromptSubmit hook output shape."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": packet,
        }
    }


def _validate_max_chars(max_chars: object) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")


def _prompt_values(prompt: str | None) -> tuple[str, str]:
    if prompt is None:
        return "unavailable", "unavailable"
    bounded = prompt[:_PROMPT_LIMIT]
    candidate = _escape_lines(bounded)
    if len(prompt) > _PROMPT_LIMIT:
        candidate += _TRUNCATION_MARKER
    if not candidate:
        candidate = "(empty)"
    return candidate, _variable_placeholder(candidate)


def _tool_values(tool_names: list[str]) -> tuple[str, str]:
    if not tool_names:
        return "none", "none"
    candidate = ", ".join(_escape_lines(name) or "(empty)" for name in tool_names)
    return candidate, _variable_placeholder(candidate)


def _variable_placeholder(candidate: str) -> str:
    if len(candidate) <= len(_TRUNCATION_MARKER):
        return candidate
    return _TRUNCATION_MARKER


def _fit_value(candidate: str, placeholder: str, remaining: int) -> tuple[str, int]:
    difference = len(candidate) - len(placeholder)
    if difference <= remaining:
        return candidate, remaining - difference
    allowed = len(placeholder) + remaining
    if allowed < len(_TRUNCATION_MARKER):
        raise AssertionError("truncation marker was not reserved")
    return candidate[: allowed - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER, 0


def _render_packet(
    checkpoint: CheckpointManifest, prompt: str, tools: str
) -> str:
    transcript = checkpoint.transcript
    if transcript.status == "captured":
        transcript_text = (
            "captured (blob sha256:"
            f"{transcript.blob_sha256 or 'unavailable'}; bytes: "
            f"{transcript.byte_size if transcript.byte_size is not None else 'unavailable'})"
        )
    else:
        transcript_text = "unavailable"

    git = checkpoint.git
    branch = git.branch if git.branch is not None else "none"
    head = git.head if git.head is not None else "none"
    if isinstance(git.status_blob_sha256, str):
        dirty = "clean" if git.status_blob_sha256 == _EMPTY_STATUS_SHA256 else "dirty"
    else:
        dirty = "unavailable"
    unborn = "yes" if git.is_unborn else "no"

    return "\n".join(
        (
            "[CODEX MEMORY RECOVERY]",
            "Checkpoint",
            f"- ID: {checkpoint.checkpoint_id}",
            f"- Event highwater: {checkpoint.raw_event_highwater}",
            f"- Transcript: {transcript_text}",
            "Repository state",
            f"- Branch: {branch}",
            f"- HEAD: {head}",
            f"- Unborn: {unborn}",
            f"- Dirty: {dirty}",
            "Latest user objective",
            f"- Objective: {prompt}",
            "Recent tool activity",
            f"- Tools: {tools}",
            "Recovery discipline",
            _DISCIPLINE,
        )
    )


def _escape_lines(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n")
