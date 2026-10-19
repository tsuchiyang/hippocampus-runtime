from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Mapping


MAX_HOOK_INPUT_BYTES = 1_048_576
MAX_PROMPT_BYTES = 65_536
MAX_TOOL_FIELD_BYTES = 262_144


class HookInputError(ValueError):
    """Raised when Codex hook input violates the supported contract."""


class HookEvent(StrEnum):
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"
    SESSION_START = "SessionStart"


@dataclass(frozen=True, slots=True)
class HookEnvelope:
    event: HookEvent
    session_id: str
    transcript_path: Path | None
    cwd: Path
    model: str | None
    turn_id: str | None
    permission_mode: str | None
    trigger: str | None
    source: str | None
    prompt: str | None
    tool_name: str | None
    tool_use_id: str | None
    raw: dict[str, Any]


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise HookInputError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HookInputError(f"{key} must be a string or null")
    return value


def validate_hook_payload_limits(payload: Mapping[str, Any]) -> None:
    """Reject, never truncate, oversized evidence before any durable write."""
    try:
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise HookInputError("prompt exceeds byte limit")
        for field in ("tool_input", "tool_response"):
            if field in payload:
                raw = json.dumps(payload[field], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                if len(raw.encode("utf-8")) > MAX_TOOL_FIELD_BYTES:
                    raise HookInputError(f"{field} exceeds byte limit")
        raw = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) > MAX_HOOK_INPUT_BYTES:
            raise HookInputError("hook payload exceeds byte limit")
    except HookInputError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise HookInputError("invalid hook payload") from exc


def parse_hook_payload(payload: Mapping[str, Any]) -> HookEnvelope:
    validate_hook_payload_limits(payload)
    event_text = _required_string(payload, "hook_event_name")
    try:
        event = HookEvent(event_text)
    except ValueError as exc:
        raise HookInputError(f"Unsupported hook_event_name: {event_text}") from exc

    session_id = _required_string(payload, "session_id")
    cwd = Path(_required_string(payload, "cwd"))
    transcript_text = _optional_string(payload, "transcript_path")
    turn_id = _optional_string(payload, "turn_id")
    trigger = _optional_string(payload, "trigger")
    source = _optional_string(payload, "source")
    prompt = _optional_string(payload, "prompt")
    tool_name = _optional_string(payload, "tool_name")
    tool_use_id = _optional_string(payload, "tool_use_id")

    if event in {
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.POST_TOOL_USE,
        HookEvent.PRE_COMPACT,
        HookEvent.POST_COMPACT,
    } and not turn_id:
        raise HookInputError(f"turn_id is required for {event.value}")
    if event in {HookEvent.PRE_COMPACT, HookEvent.POST_COMPACT} and trigger not in {"manual", "auto"}:
        raise HookInputError(f"trigger must be manual or auto for {event.value}")
    if event is HookEvent.SESSION_START and source not in {"startup", "resume", "clear", "compact"}:
        raise HookInputError("source must be startup, resume, clear, or compact")
    if event is HookEvent.USER_PROMPT_SUBMIT and prompt is None:
        raise HookInputError("prompt is required for UserPromptSubmit")
    if event is HookEvent.POST_TOOL_USE:
        if not tool_name or not tool_use_id:
            raise HookInputError("tool_name and tool_use_id are required for PostToolUse")
        if "tool_input" not in payload or "tool_response" not in payload:
            raise HookInputError("tool_input and tool_response are required for PostToolUse")

    return HookEnvelope(
        event=event,
        session_id=session_id,
        transcript_path=Path(transcript_text) if transcript_text else None,
        cwd=cwd,
        model=_optional_string(payload, "model"),
        turn_id=turn_id,
        permission_mode=_optional_string(payload, "permission_mode"),
        trigger=trigger,
        source=source,
        prompt=prompt,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        raw=dict(payload),
    )
