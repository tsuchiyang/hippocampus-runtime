"""Thin event-specific behavior for the Codex hook lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from codex_memory.checkpoint import CheckpointManifest, CheckpointService
from codex_memory.event_store import EventRecord, EventStore
from codex_memory.hook_input import HookEnvelope
from codex_memory.recovery import (
    SESSION_START_CONTEXT_LIMIT,
    build_orientation_packet,
    session_start_output,
)


class RecoveryUnavailableError(RuntimeError):
    """Raised when compact recovery cannot be proven safe to deliver."""


@dataclass(frozen=True, slots=True)
class PendingRecovery:
    """A fully constructed response awaiting its guarded delivery transition."""

    checkpoint_id: str
    session_id: str
    turn_id: str
    response: dict[str, object]


def handle_user_prompt_submit(
    envelope: HookEnvelope, events: EventStore
) -> EventRecord:
    return events.append(envelope)


def handle_post_tool_use(envelope: HookEnvelope, events: EventStore) -> None:
    events.append(envelope)


def handle_pre_compact(
    envelope: HookEnvelope, checkpoints: CheckpointService
) -> dict[str, bool]:
    checkpoints.create_precompact(envelope)
    return {"continue": True}


def handle_post_compact(
    envelope: HookEnvelope, checkpoints: CheckpointService
) -> dict[str, bool]:
    checkpoints.mark_postcompact(envelope)
    return {"continue": True}


def prepare_session_start(
    envelope: HookEnvelope,
    events: EventStore,
    checkpoints: CheckpointService,
) -> PendingRecovery | None:
    """Append SessionStart and prepare compact recovery without consuming it."""
    events.append(envelope)
    if envelope.source != "compact":
        return None

    checkpoint = checkpoints.latest_committed(envelope.session_id)
    _require_recoverable(checkpoint, envelope.session_id)
    assert checkpoint is not None
    packet = build_orientation_packet(
        checkpoint, events, max_chars=SESSION_START_CONTEXT_LIMIT
    )
    return PendingRecovery(
        checkpoint_id=checkpoint.checkpoint_id,
        session_id=envelope.session_id,
        turn_id=checkpoint.turn_id,
        response=session_start_output(packet),
    )


def _require_recoverable(
    checkpoint: CheckpointManifest | None, session_id: str
) -> None:
    if (
        checkpoint is None
        or checkpoint.status != "COMMITTED"
        or checkpoint.session_id != session_id
        or checkpoint.recovery_required is not True
    ):
        raise RecoveryUnavailableError("verified compact recovery is unavailable")
