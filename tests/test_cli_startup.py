"""Command-only initialization must not add latency to lifecycle Hook startup."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


# This is an import-seam regression, not the installed-EXE wall-clock acceptance.
# Delay real command-module loading; leave the actual Hook implementation intact.
_IMPORT_DELAY_PROBE = r'''
import importlib.machinery
from io import StringIO
import json
import sys
import time

delayed = []
command_modules = {"codex_memory.ledger_cli", "codex_memory.memory_cli"}

class DelayedLoader:
    def __init__(self, wrapped, name):
        self.wrapped, self.name = wrapped, name

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        delayed.append(self.name)
        time.sleep(0.05)
        self.wrapped.exec_module(module)

class CommandDelayFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in command_modules:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        spec.loader = DelayedLoader(spec.loader, fullname)
        return spec

sys.meta_path.insert(0, CommandDelayFinder())
from codex_memory.cli import main

prefix = sys.argv[1]
arguments = ["user-prompt-submit"] if prefix == "hook" else [prefix, "unsupported-test-operation"]
out, err = StringIO(), StringIO()
code = main(arguments, StringIO(sys.stdin.read()), out, err)
print(json.dumps({"returncode": code, "stdout": out.getvalue(), "stderr": err.getvalue(),
                  "delayed_command_modules": delayed, "injected_delay_ms": 50 * len(delayed)}))
'''


def _probe(prefix: str, repository: Path, payload: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-B", "-c", _IMPORT_DELAY_PROBE, prefix],
        cwd=repository,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def test_user_prompt_skips_slow_command_only_initialization(tmp_path: Path) -> None:
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)

    result = _probe("hook", repository, {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "startup-delay-session",
        "turn_id": "startup-delay-turn",
        "cwd": str(repository),
        "prompt": "Keep the real legacy response without command-only startup work.",
    })

    assert result["returncode"] == 0
    assert result["stderr"] == ""
    output = json.loads(result["stdout"])["hookSpecificOutput"]
    assert output["hookEventName"] == "UserPromptSubmit"
    assert "[CODEX MEMORY PROTOCOL v1]" in output["additionalContext"]
    assert result["delayed_command_modules"] == []
    assert result["injected_delay_ms"] == 0


def test_shadow_import_defers_retrieval_service_until_the_query_deadline() -> None:
    """Importing the Hook shadow boundary must not eagerly load its query engine.

    The installed Hook imports ``shadow.compare`` before ``compare`` establishes
    the semantic query deadline.  Keeping ``service`` out of that import phase
    makes its initialization part of the already-bounded query work instead of
    unbounded process-start overhead.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import json, sys; "
                "import codex_memory.retrieval.shadow; "
                "deferred = {"
                "'codex_memory.retrieval.service',"
                "'codex_memory.retrieval.cache_database',"
                "'codex_memory.retrieval.provider_health',"
                "'codex_memory.retrieval.provider_control',"
                "'codex_memory.retrieval.semantic_paths',"
                "'codex_memory.retrieval.semantic_store'}; "
                "print(json.dumps({'eager_modules': sorted(deferred & sys.modules.keys())}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stderr == ""
    assert json.loads(probe.stdout) == {"eager_modules": []}


def test_shadow_import_defers_offline_packet_assembly_until_capture() -> None:
    """Hook shadow import must not eagerly initialize the P2.4 packet stack.

    Phase 2 packet assembly is needed only when an explicitly enabled audit
    reaches its bounded capture stage.  Loading that offline stack merely to
    import ``shadow.compare`` adds unbounded process-start work before the
    established semantic deadline.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import json, sys; "
                "import codex_memory.retrieval.shadow; "
                "deferred = {"
                "'codex_memory.retrieval.assembler',"
                "'codex_memory.retrieval.evidence_pack',"
                "'codex_memory.retrieval.render',"
                "'codex_memory.retrieval.canonical_snapshot'}; "
                "print(json.dumps({'eager_modules': sorted(deferred & sys.modules.keys())}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stderr == ""
    assert json.loads(probe.stdout) == {"eager_modules": []}


def test_slow_retrieval_service_import_cannot_outlive_shadow_deadline() -> None:
    """The deferred import is bounded, not merely included in elapsed time."""
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            r'''
import importlib.machinery
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

from codex_memory.retrieval.contracts import RetrievalMode, RetrievalRequest, RetrievalScope
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.shadow import compare

class DelayedLoader:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        time.sleep(0.5)
        self.wrapped.exec_module(module)

class DelayedServiceFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "codex_memory.retrieval.service":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        spec.loader = DelayedLoader(spec.loader)
        return spec

sys.meta_path.insert(0, DelayedServiceFinder())
with TemporaryDirectory() as directory:
    root = Path(directory)
    request = RetrievalRequest(
        "bounded import", "session", "turn", "context", None,
        RetrievalScope(str(root)), mode=RetrievalMode.OFF,
    )
    started = time.perf_counter()
    comparison = compare(
        paths=RetrievalPaths.for_repo(root), request=request,
        hook_event="UserPromptSubmit", deadline_ms=100,
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000
    print(json.dumps({"elapsed_ms": elapsed_ms, "reason": comparison.reason}))
''',
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stderr == ""
    result = json.loads(probe.stdout)
    assert result["elapsed_ms"] < 300
    assert result["reason"] == "deadline_exhausted"


@pytest.mark.parametrize("prefix, expected_module, expected_error", [
    ("memory", "codex_memory.memory_cli", "codex-memory-hook: unsupported command\n"),
    ("record", "codex_memory.ledger_cli", "codex-memory-hook: invalid ledger input\n"),
    ("ledger", "codex_memory.ledger_cli", "codex-memory-hook: invalid ledger input\n"),
])
def test_command_route_initializes_only_its_real_handler(
    tmp_path: Path, prefix: str, expected_module: str, expected_error: str,
) -> None:
    result = _probe(prefix, tmp_path, {})

    assert result["returncode"] == 2
    assert result["stdout"] == ""
    assert result["stderr"] == expected_error
    assert result["delayed_command_modules"] == [expected_module]
    assert result["injected_delay_ms"] == 50
