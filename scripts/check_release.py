"""Fail-closed, redacted checks for the public Git tree and reachable history.

This is a bounded heuristic audit, not proof that every possible secret is absent.
It never prints matched text. Untracked/ignored developer state is not a release input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

PATTERNS = {
    "credential_shape": re.compile(rb"(?:\bsk-[A-Za-z0-9_-]{24,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{25,}\b|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"),
    "personal_path": re.compile(rb"(?:[A-Za-z]:[\\/](?:Users|codex|UserData)[\\/]|/(?:home|Users)/[^/\s\"']+/)", re.IGNORECASE),
    "credential_url": re.compile(rb"https?://[^\s/\"']+:[^\s/@\"']+@", re.IGNORECASE),
}
PRIVATE_PARTS = {".codex", ".codex-memory", ".superpowers", ".venv", "__pycache__", "sessions", "transcripts", "auth.json", "credentials.json", "models_cache.json"}
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_OBJECTS = 100_000


def _git(root, *args, input=None):
    return subprocess.run(["git", "-C", str(root), *args], input=input, check=True,
        capture_output=True, timeout=120).stdout


def _safe_path(name):
    raw = name.encode("utf-8", errors="replace")
    return "<redacted-path:" + hashlib.sha256(raw).hexdigest()[:16] + ">" if any(p.search(raw) for p in PATTERNS.values()) else name


def _private_name(name):
    parts = PurePosixPath(name).parts
    return (not parts or name.startswith("/") or ".." in parts or "\\" in name
        or any(p.casefold() in PRIVATE_PARTS or p.casefold() == ".env"
            or p.casefold().startswith(".env.") or p.casefold().endswith((".sqlite3", ".sqlite", ".db", ".pem", ".pfx", ".key", "-wal", "-shm")) for p in parts))


def scan_repository(root, *, history=False, max_blob_bytes=16 * 1024 * 1024):
    result = {"status": "FAIL", "complete": True, "files_scanned": 0,
        "history_blobs_scanned": 0, "history_trees_scanned": 0,
        "history_paths_examined": 0, "findings": [], "history_requested": history,
        "scope": "tracked checkout and optionally all reachable Git blobs; heuristic patterns, not legal/security certification"}
    findings = result["findings"]

    def add(rule, name, *, scope="worktree", line=None, oid=None):
        if len(findings) >= 10_000:
            result["complete"] = False
            return
        row = {"rule": rule, "path": _safe_path(name), "scope": scope}
        if line is not None:
            row["line"] = line
        if oid is not None:
            row["blob_oid"] = oid
        findings.append(row)

    def inspect(raw, name, *, scope="worktree", oid=None):
        for rule, expression in PATTERNS.items():
            for match in expression.finditer(raw):
                add(rule, name, scope=scope, line=raw.count(b"\n", 0, match.start())+1, oid=oid)

    try:
        root = Path(root).resolve(strict=True)
        tracked = _git(root, "ls-files", "-z").decode("utf-8").split("\0")
        for name in filter(None, tracked):
            if _private_name(name):
                add("private_path", name)
                continue
            path = root / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                add("unsafe_file", name)
                continue
            if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in [path, *[root / p for p in Path(name).parents if str(p) != "."]]):
                add("unsafe_file", name)
                continue
            if info.st_size > max_blob_bytes:
                add("file_limit", name)
                result["complete"] = False
                continue
            raw = path.read_bytes()
            if len(raw) > max_blob_bytes:
                add("file_limit", name)
                result["complete"] = False
                continue
            inspect(raw, name)
            result["files_scanned"] += 1
        if history:
            # Blob deduplication loses historical aliases. Inspect every distinct
            # reachable root tree's paths separately from its blob contents.
            tree_rows = _git(root, "rev-list", "--all", "--format=%T").decode("ascii").splitlines()
            trees = sorted({row for row in tree_rows if not row.startswith("commit ")})
            if len(trees) > MAX_OBJECTS or any(not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tree) for tree in trees):
                raise ValueError("history tree limit or identity")
            path_bytes = 0
            private_paths = set()
            for tree in trees:
                listing = _git(root, "ls-tree", "-r", "-z", tree)
                path_bytes += len(listing)
                if path_bytes > MAX_TOTAL_BYTES:
                    raise ValueError("history path byte limit")
                for item in filter(None, listing.split(b"\0")):
                    header, name_bytes = item.split(b"\t", 1)
                    _mode, _kind, oid = header.decode("ascii").split()
                    name = name_bytes.decode("utf-8")
                    result["history_paths_examined"] += 1
                    if result["history_paths_examined"] > 1_000_000:
                        raise ValueError("history entry limit")
                    if _private_name(name) and name not in private_paths:
                        private_paths.add(name)
                        add("private_path", name, scope="history", oid=oid)
                result["history_trees_scanned"] += 1
            rows = _git(root, "rev-list", "--objects", "--all").decode("utf-8").splitlines()
            if len(rows) > MAX_OBJECTS:
                raise ValueError("object count limit")
            names = dict(row.split(" ", 1) if " " in row else (row, "<git-object>") for row in rows)
            request = "".join(oid+"\n" for oid in names).encode("ascii")
            headers = _git(root, "cat-file", "--batch-check", input=request).decode("ascii").splitlines()
            blobs = []
            total = 0
            for header in headers:
                oid, kind, size_text = header.split()
                size = int(size_text)
                if kind != "blob":
                    continue
                if size > max_blob_bytes:
                    add("file_limit", names[oid], scope="history", oid=oid)
                    result["complete"] = False
                    continue
                total += size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("history byte limit")
                blobs.append(oid)
            raw = _git(root, "cat-file", "--batch", input="".join(oid+"\n" for oid in blobs).encode("ascii"))
            offset = 0
            for oid in blobs:
                end = raw.index(b"\n", offset)
                actual, kind, size = raw[offset:end].decode("ascii").split()
                if actual != oid or kind != "blob":
                    raise ValueError("unexpected Git object")
                offset = end+1
                size = int(size)
                content = raw[offset:offset+size]
                if len(content) != size or raw[offset+size:offset+size+1] != b"\n":
                    raise ValueError("incomplete Git object")
                name = names[oid]
                inspect(content, name, scope="history", oid=oid)
                result["history_blobs_scanned"] += 1
                offset += size+1
            if offset != len(raw):
                raise ValueError("extra Git object bytes")
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        result["complete"] = False
        findings.append({"rule": "audit_incomplete", "error_type": type(exc).__name__})
    result["status"] = "PASS" if result["complete"] and not findings else "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = scan_repository(args.repository, history=args.history)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n"
    if args.report is not None:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"} | {"finding_count": len(result["findings"])}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
