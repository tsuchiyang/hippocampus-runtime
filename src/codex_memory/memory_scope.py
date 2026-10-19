"""Explicit project/workspace scope matching for memory recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from codex_memory.ledger_common import (
    ValidationError,
    _require_bounded_text,
    scope_keys_equal,
)


MAX_WORKSPACE_SCOPE_KEYS = 32
MAX_SCOPE_KEY_BYTES = 256


def normalize_workspace_scope_keys(
    workspace_scope_keys: Iterable[str],
) -> tuple[str, ...]:
    """Validate, bound, and deterministically deduplicate workspace keys."""
    if isinstance(workspace_scope_keys, (str, bytes)):
        raise ValidationError("workspace scope keys must be a sequence")
    try:
        values = tuple(workspace_scope_keys)
    except TypeError as exc:
        raise ValidationError("workspace scope keys must be a sequence") from exc
    if len(values) > MAX_WORKSPACE_SCOPE_KEYS:
        raise ValidationError("too many workspace scope keys")

    result: list[str] = []
    for value in values:
        _require_bounded_text(
            value,
            "workspace scope key",
            MAX_SCOPE_KEY_BYTES,
            nonblank=True,
        )
        if not any(scope_keys_equal(value, existing) for existing in result):
            result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ScopeContext:
    """The project key and explicitly configured workspace keys for a request."""

    project_scope_key: str
    workspace_scope_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_bounded_text(
            self.project_scope_key,
            "project scope key",
            MAX_SCOPE_KEY_BYTES,
            nonblank=True,
        )
        normalized = normalize_workspace_scope_keys(self.workspace_scope_keys)
        object.__setattr__(self, "workspace_scope_keys", normalized)

    def contains(self, *, scope_kind: str, scope_key: str) -> bool:
        # Unknown kinds are intentionally non-selectable.  This keeps future
        # schema values from accidentally entering a project/workspace packet.
        if scope_kind != "project" and scope_kind != "workspace":
            return False
        _require_bounded_text(scope_key, "scope key", MAX_SCOPE_KEY_BYTES, nonblank=True)
        if scope_kind == "project":
            return scope_keys_equal(scope_key, self.project_scope_key)
        if scope_kind == "workspace":
            return any(
                scope_keys_equal(scope_key, configured)
                for configured in self.workspace_scope_keys
            )
