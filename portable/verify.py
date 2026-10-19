"""Verify the exact inventory of a portable candidate without importing Runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import zipfile

MANIFEST = "PORTABLE-MANIFEST.json"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN = {".git", ".codex", ".codex-memory", ".venv", ".superpowers",
              "__pycache__", "auth.json", "credentials.json", "history.jsonl",
              "sessions", "transcripts", "secrets", "models_cache.json"}
_DEVICES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
            *(f"lpt{i}" for i in range(1, 10))}


def absolute_path(path: Path) -> Path:
    """Normalize lexical parent segments without resolving through filesystem redirects."""
    return Path(os.path.abspath(path))


def safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        raise ValueError("unsafe inventory name")
    try:
        if len(name.encode("utf-8")) > 4096 or any(len(part.encode("utf-8")) > 255 for part in name.split("/")):
            raise ValueError("inventory name length limit")
    except UnicodeError as exc:
        raise ValueError("unsafe inventory name encoding") from exc
    for part in name.split("/"):
        lower = part.casefold()
        if (not part or part in {".", ".."} or part[-1:] in {".", " "}
                or any(ord(c) < 32 or c in ':<>|?*' for c in part)
                or lower in _FORBIDDEN or lower == ".env" or lower.startswith(".env.")
                or lower.endswith((".db", ".sqlite", ".sqlite3", ".pyc", "-wal", "-shm"))
                or lower.split(".")[0] in _DEVICES):
            raise ValueError("unsafe inventory name")
    return name


def _pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def strict_json(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _rows(raw: bytes) -> tuple[dict, dict[str, dict]]:
    manifest = strict_json(raw)
    if (not isinstance(manifest, dict)
            or set(manifest) != {"format", "source_commit", "runtime_sha256", "files"}
            or manifest["format"] != "codex-memory-portable-v1"
            or not isinstance(manifest["source_commit"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"])
            or not isinstance(manifest["runtime_sha256"], str)
            or not _HASH.fullmatch(manifest["runtime_sha256"])
            or not isinstance(manifest["files"], list) or not manifest["files"]):
        raise ValueError("invalid manifest schema")
    rows, seen, total = {}, set(), 0
    for row in manifest["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise ValueError("invalid inventory row")
        name = safe_name(row["path"])
        if name.casefold() in seen or name.casefold() == MANIFEST.casefold():
            raise ValueError("duplicate or case collision")
        seen.add(name.casefold())
        if (not isinstance(row["sha256"], str) or not _HASH.fullmatch(row["sha256"])
                or type(row["size"]) is not int or not 0 <= row["size"] <= MAX_FILE_BYTES):
            raise ValueError("invalid inventory size/hash")
        total += row["size"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("inventory exceeds size limit")
        rows[name] = row
    return manifest, rows


def _check(raw: bytes, row: dict) -> None:
    if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("inventory size/hash mismatch")


def _ordinary(path: Path, *, directory: bool = False) -> None:
    details = path.lstat()
    if (stat.S_ISLNK(details.st_mode)
            or getattr(details, "st_file_attributes", 0) & 0x400
            or (not stat.S_ISDIR(details.st_mode) if directory else not stat.S_ISREG(details.st_mode))):
        raise ValueError("redirect or non-regular inventory object")


def verify_directory(root: Path) -> dict:
    root = absolute_path(root)
    # Refuse redirected extraction roots and their existing ancestors as well.
    for parent in [*reversed(root.parents), root]:
        _ordinary(parent, directory=True)
    marker = root / MANIFEST
    _ordinary(marker)
    if marker.stat().st_size > 1024 * 1024:
        raise ValueError("manifest size limit")
    manifest, rows = _rows(marker.read_bytes())
    found: set[str] = set()
    directories = {parent.as_posix() for name in rows for parent in Path(name).parents
                   if parent != Path(".")}
    for path in root.rglob("*"):
        if path.is_dir():
            _ordinary(path, directory=True)
            if path.relative_to(root).as_posix() not in directories:
                raise ValueError("extra inventory directory")
            continue
        _ordinary(path)
        name = path.relative_to(root).as_posix()
        if name == MANIFEST:
            continue
        safe_name(name)
        if name not in rows:
            raise ValueError("extra inventory file")
        if path.stat().st_size != rows[name]["size"]:
            raise ValueError("inventory size mismatch")
        _check(path.read_bytes(), rows[name])
        found.add(name)
    if found != set(rows):
        raise ValueError("missing inventory file")
    return manifest


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [safe_name(info.filename) for info in infos]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("duplicate archive names")
        if any(info.is_dir() or (info.external_attr >> 16) & 0o170000 not in {0, 0o100000}
               or info.file_size > MAX_FILE_BYTES for info in infos):
            raise ValueError("unsafe archive entry")
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise ValueError("archive size limit")
        manifest, rows = _rows(archive.read(MANIFEST))
        if set(names) != set(rows) | {MANIFEST}:
            raise ValueError("archive inventory mismatch")
        for name, row in rows.items():
            _check(archive.read(name), row)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path)
    group.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        manifest = verify_directory(args.bundle) if args.bundle else verify_archive(args.archive)
        print(json.dumps({"status": "INTEGRITY_VERIFIED", "files": len(manifest["files"]),
                          "source_commit": manifest["source_commit"]}, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "REJECTED", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
