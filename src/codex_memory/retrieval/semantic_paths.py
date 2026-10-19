"""Safe, fixed names for disposable semantic generation leaves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from codex_memory.paths import RuntimePathError, path_is_redirect, require_safe_directory, require_safe_file


_GENERATION_ID = re.compile(r"\Asem_[0-9a-f]{16,64}\Z")


@dataclass(frozen=True, slots=True)
class SemanticGenerationPaths:
    generations_dir: Path
    semantic_generation_id: str
    directory: Path
    database_path: Path
    manifest_path: Path
    ownership_marker_path: Path

    @classmethod
    def for_generation(cls, generations_dir: Path, semantic_generation_id: str) -> "SemanticGenerationPaths":
        if not isinstance(semantic_generation_id, str) or not _GENERATION_ID.fullmatch(semantic_generation_id):
            raise RuntimePathError("invalid semantic generation identifier")
        root = Path(generations_dir)
        directory = root / semantic_generation_id
        return cls(root, semantic_generation_id, directory, directory / "semantic.sqlite3", directory / "generation-manifest.json", root / f".{semantic_generation_id}.owner")

    def validate_directory(self, *, require_exists: bool = True) -> None:
        require_safe_directory(self.generations_dir)
        if require_exists:
            require_safe_directory(self.directory, self.generations_dir)
        elif self.directory.exists() or path_is_redirect(self.directory):
            raise RuntimePathError("semantic generation directory is not absent")

    def validate_leaves(self, *, require_exists: bool = True) -> None:
        self.validate_directory(require_exists=True)
        require_safe_file(self.database_path, self.directory, require_exists=require_exists)
        require_safe_file(self.manifest_path, self.directory, require_exists=require_exists)
