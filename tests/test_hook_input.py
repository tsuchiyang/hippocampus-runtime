from pathlib import Path

import pytest

from codex_memory.hook_input import HookEvent, HookInputError, parse_hook_payload


def base_payload(event: str) -> dict[str, object]:
    return {
        "session_id": "thr_123",
        "transcript_path": None,
        "cwd": r"C:\repo with space",
        "hook_event_name": event,
        "model": "test-model",
    }


def test_parse_precompact_requires_turn_and_trigger() -> None:
    payload = base_payload("PreCompact") | {
        "turn_id": "turn_1",
        "trigger": "manual",
    }
    envelope = parse_hook_payload(payload)
    assert envelope.event is HookEvent.PRE_COMPACT
    assert envelope.turn_id == "turn_1"
    assert envelope.trigger == "manual"
    assert envelope.cwd == Path(r"C:\repo with space")


def test_parse_session_start_accepts_compact_source() -> None:
    payload = base_payload("SessionStart") | {"source": "compact"}
    envelope = parse_hook_payload(payload)
    assert envelope.event is HookEvent.SESSION_START
    assert envelope.source == "compact"


def test_parse_post_tool_use_preserves_arbitrary_json() -> None:
    payload = base_payload("PostToolUse") | {
        "turn_id": "turn_1",
        "tool_name": "Bash",
        "tool_use_id": "tool_1",
        "tool_input": {"command": "git status --short"},
        "tool_response": {"output": ""},
    }
    envelope = parse_hook_payload(payload)
    assert envelope.tool_use_id == "tool_1"
    assert envelope.raw["tool_response"] == {"output": ""}


def test_missing_session_id_is_rejected() -> None:
    payload = base_payload("UserPromptSubmit") | {
        "turn_id": "turn_1",
        "prompt": "hello",
    }
    del payload["session_id"]
    with pytest.raises(HookInputError, match="session_id"):
        parse_hook_payload(payload)


def test_unknown_event_is_rejected() -> None:
    with pytest.raises(HookInputError, match="Unsupported hook_event_name"):
        parse_hook_payload(base_payload("UnknownEvent"))


def test_precompact_rejects_invalid_trigger() -> None:
    payload = base_payload("PreCompact") | {"turn_id": "turn_1", "trigger": "later"}
    with pytest.raises(HookInputError, match="trigger"):
        parse_hook_payload(payload)


def test_session_start_rejects_invalid_source() -> None:
    payload = base_payload("SessionStart") | {"source": "unknown"}
    with pytest.raises(HookInputError, match="source"):
        parse_hook_payload(payload)


def test_post_tool_use_requires_tool_fields() -> None:
    payload = base_payload("PostToolUse") | {"turn_id": "turn_1"}
    with pytest.raises(HookInputError, match="tool_name and tool_use_id"):
        parse_hook_payload(payload)


def test_optional_string_rejects_wrong_type() -> None:
    payload = base_payload("SessionStart") | {"source": "startup", "model": 42}
    with pytest.raises(HookInputError, match="model"):
        parse_hook_payload(payload)
