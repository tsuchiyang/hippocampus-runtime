"""Run the actual public-tree scanner against small, real Git repositories."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def scanner():
    path = Path(__file__).parents[1] / "scripts/check_release.py"
    if not path.is_file():
        pytest.fail("public release/privacy scanner is missing")
    spec = importlib.util.spec_from_file_location("public_release_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, timeout=10).stdout


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Synthetic Test", "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "synthetic fixture")


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("Synthetic public fixture.\n", encoding="utf-8")
    commit(tmp_path)
    return tmp_path


def test_clean_public_files_and_history_pass(repository):
    result = scanner().scan_repository(repository, history=True)
    assert result["status"] == "PASS" and result["complete"] is True
    assert result["files_scanned"] == 1 and result["history_blobs_scanned"] == 1
    assert result["findings"] == []


@pytest.mark.parametrize("relative", ["auth.json", ".env", ".codex/hooks.json", "memory.sqlite3", "sessions/private.jsonl"])
def test_private_paths_rejected_without_disclosing_content(repository, relative):
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private-content-sentinel", encoding="utf-8")
    commit(repository)
    result = scanner().scan_repository(repository)
    assert result["status"] == "FAIL"
    assert any(row["rule"] == "private_path" for row in result["findings"])
    assert "private-content-sentinel" not in json.dumps(result)


def test_secret_shaped_token_is_rejected_and_redacted(repository):
    token = "sk" + "-" + "synthetic" * 6
    (repository / "sample.txt").write_text(token, encoding="utf-8")
    commit(repository)
    result = scanner().scan_repository(repository)
    assert result["status"] == "FAIL"
    assert any(row["rule"] == "credential_shape" for row in result["findings"])
    assert token not in json.dumps(result)


def test_removed_secret_is_still_found_in_reachable_git_history(repository):
    token = "ghp" + "_" + "SYNTHETIC" * 5
    path = repository / "old.txt"
    path.write_text(token, encoding="utf-8")
    commit(repository)
    path.unlink()
    commit(repository)
    assert scanner().scan_repository(repository)["status"] == "PASS"
    result = scanner().scan_repository(repository, history=True)
    assert result["status"] == "FAIL" and result["history_blobs_scanned"] >= 2
    assert any(row.get("scope") == "history" for row in result["findings"])
    assert token not in json.dumps(result)


def test_deleted_private_alias_is_detected_even_when_git_deduplicates_its_blob(repository):
    private = repository / "auth.json"
    private.write_bytes((repository / "README.md").read_bytes())
    commit(repository)
    private.unlink()
    commit(repository)
    assert scanner().scan_repository(repository)["status"] == "PASS"
    result = scanner().scan_repository(repository, history=True)
    assert result["status"] == "FAIL"
    assert any(row["rule"] == "private_path" and row.get("scope") == "history"
        and row["path"] == "auth.json" for row in result["findings"])


def test_personal_machine_path_is_rejected(repository):
    private = "C" + ":/" + "Users/" + "synthetic-owner/Documents/project"
    (repository / "example.txt").write_text(private, encoding="utf-8")
    commit(repository)
    result = scanner().scan_repository(repository)
    assert any(row["rule"] == "personal_path" for row in result["findings"])
    assert private not in json.dumps(result)


def test_hardlinked_file_is_not_accepted_as_an_independent_snapshot(repository):
    path = repository / "linked.txt"
    path.write_text("synthetic", encoding="utf-8")
    (repository / "other.txt").hardlink_to(path)
    commit(repository)
    result = scanner().scan_repository(repository)
    assert result["status"] == "FAIL"
    assert any(row["rule"] == "unsafe_file" for row in result["findings"])


def test_oversize_file_causes_incomplete_fail_closed_not_partial_pass(repository):
    (repository / "large.txt").write_bytes(b"a" * 129)
    commit(repository)
    result = scanner().scan_repository(repository, max_blob_bytes=128)
    assert result["status"] == "FAIL" and result["complete"] is False
    assert any(row["rule"] == "file_limit" for row in result["findings"])
