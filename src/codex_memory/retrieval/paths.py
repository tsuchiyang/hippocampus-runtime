from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codex_memory.paths import (
    RuntimePathError,
    ensure_safe_child_directory,
    path_is_redirect,
    require_safe_directory,
    require_safe_file,
)


@dataclass(frozen=True, slots=True)
class RetrievalPaths:
    """Derived retrieval locations kept outside the canonical ledger layout."""

    repo_root: Path
    runtime_root: Path
    canonical_database_path: Path
    retrieval_root: Path
    database_path: Path
    generations_dir: Path
    temp_dir: Path

    @classmethod
    def for_repo(cls, repo_root: Path) -> "RetrievalPaths":
        root = repo_root.resolve()
        runtime = root / ".codex-memory"
        retrieval = runtime / "retrieval"
        return cls(
            repo_root=root,
            runtime_root=runtime,
            canonical_database_path=runtime / "db" / "memory.sqlite3",
            retrieval_root=retrieval,
            database_path=retrieval / "retrieval.sqlite3",
            generations_dir=retrieval / "generations",
            temp_dir=retrieval / "temp",
        )

    def validate(self) -> None:
        """Validate existing paths only; this method never creates a path."""
        self.validate_canonical()
        self._validate_existing_retrieval_state()

    def validate_canonical(self) -> None:
        """Validate canonical truth without trusting or inspecting disposable cache paths."""
        require_safe_directory(self.repo_root)
        require_safe_directory(self.runtime_root, self.repo_root)
        canonical_dir = self.canonical_database_path.parent
        require_safe_directory(canonical_dir, self.runtime_root)
        require_safe_file(
            self.canonical_database_path, canonical_dir, require_exists=True
        )
        for sidecar in (
            Path(f"{self.canonical_database_path}-journal"),
            Path(f"{self.canonical_database_path}-wal"),
            Path(f"{self.canonical_database_path}-shm"),
        ):
            require_safe_file(sidecar, canonical_dir, require_exists=False,
                              allow_disappearance=True)

    def prepare_for_write(self) -> None:
        """Create precisely the disposable retrieval directories after validation."""
        self.validate()
        if not self.retrieval_root.exists():
            ensure_safe_child_directory(self.runtime_root, self.retrieval_root)
        else:
            require_safe_directory(self.retrieval_root, self.runtime_root)
        for child in (self.generations_dir, self.temp_dir):
            if not child.exists():
                ensure_safe_child_directory(self.retrieval_root, child)
            else:
                require_safe_directory(child, self.retrieval_root)
        self._validate_existing_retrieval_state()

    def _validate_existing_retrieval_state(self) -> None:
        if not self.retrieval_root.exists():
            if path_is_redirect(self.retrieval_root):
                raise RuntimePathError("unsafe retrieval directory redirect")
            return
        require_safe_directory(self.retrieval_root, self.runtime_root)
        for child in (self.generations_dir, self.temp_dir):
            if child.exists() or path_is_redirect(child):
                require_safe_directory(child, self.retrieval_root)
        if self.generations_dir.exists():
            for generation in self.generations_dir.iterdir():
                require_safe_directory(generation, self.generations_dir)
                for leaf in generation.iterdir():
                    require_safe_file(leaf, generation, require_exists=True)
        for leaf in (
            self.database_path,
            Path(f"{self.database_path}-journal"),
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
        ):
            require_safe_file(leaf, self.retrieval_root, require_exists=False,
                              allow_disappearance=(leaf != self.database_path))
