from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sqlite3
from typing import Callable

import pytest

from codex_memory.blob_store import BlobStore
from codex_memory.hook_input import HookEnvelope, parse_hook_payload
from codex_memory.paths import RuntimePaths


@dataclass(frozen=True, slots=True)
class Runtime:
    paths: RuntimePaths
    connection: object
    events: object


def _prompt_envelope(session_id: str, prompt: str) -> HookEnvelope:
    return parse_hook_payload(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": session_id,
            "turn_id": "turn_1",
            "cwd": ".",
            "prompt": prompt,
        }
    )


def _tool_envelope(session_id: str, tool_use_id: str, tool_name: str) -> HookEnvelope:
    return parse_hook_payload(
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "turn_id": "turn_1",
            "cwd": ".",
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "tool_input": {"command": "echo hi"},
            "tool_response": {"output": "hi"},
        }
    )


@pytest.fixture
def runtime(tmp_path: Path):
    database = import_module("codex_memory.database")
    event_store = import_module("codex_memory.event_store")
    paths = RuntimePaths.for_repo(tmp_path)
    connection = database.connect_database(paths.db_path)
    database.migrate_database(connection)
    try:
        yield Runtime(paths, connection, event_store.EventStore(connection, BlobStore(paths.blobs_dir)))
    finally:
        try:
            connection.close()
        except sqlite3.ProgrammingError:
            pass


@pytest.fixture
def prompt_envelope() -> HookEnvelope:
    return _prompt_envelope("session_a", "hello")


@pytest.fixture
def prompt_envelope_factory() -> Callable[[str, str], HookEnvelope]:
    return _prompt_envelope


@pytest.fixture
def tool_envelope_factory() -> Callable[[str, str, str], HookEnvelope]:
    return _tool_envelope
