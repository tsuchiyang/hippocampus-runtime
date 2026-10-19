from __future__ import annotations

import pytest

from codex_memory.ledger_common import ValidationError
from codex_memory.memory_scope import ScopeContext


def test_scope_context_separates_project_and_explicit_workspace_keys() -> None:
    context = ScopeContext(
        project_scope_key=r"C:\Work\Repo",
        workspace_scope_keys=("team-notes", "other-workspace"),
    )

    assert context.contains(scope_kind="project", scope_key=r"C:\Work\Repo")
    assert not context.contains(scope_kind="project", scope_key="team-notes")
    assert context.contains(scope_kind="workspace", scope_key="team-notes")
    assert not context.contains(scope_kind="workspace", scope_key=r"C:\Work\Repo")


def test_scope_context_preserves_first_spelling_and_deduplicates() -> None:
    context = ScopeContext(
        project_scope_key="repo",
        workspace_scope_keys=("Workspace", "Workspace", "Another"),
    )

    assert context.workspace_scope_keys == ("Workspace", "Another")


def test_scope_context_deduplicates_windows_casefold_variants() -> None:
    context = ScopeContext(
        project_scope_key="repo",
        workspace_scope_keys=("Workspace", "workspace"),
    )

    assert context.workspace_scope_keys == ("Workspace",)
    assert context.contains(scope_kind="workspace", scope_key="WORKSPACE")


def test_scope_context_rejects_blank_or_oversized_workspace_keys() -> None:
    with pytest.raises(ValidationError):
        ScopeContext(project_scope_key="repo", workspace_scope_keys=(" ",))
    with pytest.raises(ValidationError):
        ScopeContext(project_scope_key="repo", workspace_scope_keys=("x" * 257,))


def test_scope_context_rejects_too_many_workspace_keys() -> None:
    with pytest.raises(ValidationError):
        ScopeContext(
            project_scope_key="repo",
            workspace_scope_keys=tuple(f"workspace-{i}" for i in range(33)),
        )


def test_scope_context_unknown_kind_is_not_selectable() -> None:
    context = ScopeContext(project_scope_key="repo")

    assert context.contains(scope_kind="future", scope_key="repo") is False


def test_scope_context_rejects_string_as_workspace_sequence() -> None:
    with pytest.raises(ValidationError):
        ScopeContext(project_scope_key="repo", workspace_scope_keys="team")


def test_scope_context_uses_utf8_byte_bound_for_keys() -> None:
    accepted = "é" * 128
    rejected = accepted + "x"

    assert ScopeContext(project_scope_key=accepted).project_scope_key == accepted
    with pytest.raises(ValidationError):
        ScopeContext(project_scope_key=rejected)


def test_scope_context_keeps_opaque_workspace_key_spelling() -> None:
    context = ScopeContext(
        project_scope_key="repo",
        workspace_scope_keys=(r"team/not-a-path", r"TEAM/NOT-A-PATH"),
    )

    assert context.workspace_scope_keys == (r"team/not-a-path",)
