from io import StringIO
import json

from codex_memory import cli
from codex_memory.retrieval.cli import run_retrieval_command
from codex_memory.retrieval.paths import RetrievalPaths
from codex_memory.retrieval.service import RetrievalService
from codex_memory.retrieval import shadow
from test_project_scope_migration import OLD_KEY, _configure, _fixture


def test_retrieval_cli_uses_same_preserved_scope_as_legacy(tmp_path):
    _fixture(tmp_path)
    _configure(tmp_path, json.dumps(OLD_KEY))
    assert RetrievalService(RetrievalPaths.for_repo(tmp_path)).rebuild().status.value == "current"
    out, err = StringIO(), StringIO()
    code = run_retrieval_command(
        ["retrieval", "query", "--session-id", "migration-session", "--turn-id", "t",
         "--context-id", "startup", "--query", "migration", "--explain", "--legacy"],
        out, err, lambda _: tmp_path,
    )
    assert (code, err.getvalue()) == (0, "")
    value = json.loads(out.getvalue())
    assert len(value["candidates"]) == 1
    assert "KEEP_LEGACY_PROJECT" in value["legacy"]["packet"]
    assert "EXCLUDE_FOREIGN_PROJECT" not in value["legacy"]["packet"]
    assert "EXCLUDE_NEW_PHYSICAL_IDENTITY" not in value["legacy"]["packet"]


def test_explicit_shadow_uses_preserved_scope_without_changing_legacy(tmp_path, monkeypatch):
    _fixture(tmp_path)
    _configure(tmp_path, json.dumps(OLD_KEY))
    monkeypatch.setattr(cli, "_resolve_repo_root", lambda _: tmp_path)
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "migration-session",
               "turn_id": "migration-turn", "cwd": str(tmp_path), "prompt": "migration"}
    observed = []
    compare = shadow.compare

    def observe(**kwargs):
        result = compare(**kwargs)
        observed.append((kwargs["request"].scope, result))
        return result

    monkeypatch.setattr(shadow, "compare", observe)
    path = tmp_path / ".codex" / "memory.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[retrieval]\nmode = "shadow"\n', encoding="utf-8")
    out, err = StringIO(), StringIO()
    assert cli.main(["user-prompt-submit"], StringIO(json.dumps(payload)), out, err) == 0
    assert err.getvalue() == ""
    assert "KEEP_LEGACY_PROJECT" in out.getvalue()
    assert len(observed) == 1
    assert observed[0][0].project_scope_key == OLD_KEY
