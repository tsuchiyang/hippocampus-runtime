"""Optional project-local configuration for automatic memory scope."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import tomllib

from codex_memory.ledger_common import ValidationError
from codex_memory.memory_scope import ScopeContext, normalize_workspace_scope_keys
from codex_memory.paths import (
    RuntimePathError,
    path_is_redirect,
    require_safe_directory,
    require_safe_file,
)


MEMORY_CONFIG_RELATIVE_PATH = Path(".codex") / "memory.toml"
INVALID_MEMORY_CONFIG_DIAGNOSTIC = (
    "codex-memory-hook: invalid memory config; workspace recall disabled\n"
)
MAX_PROJECT_MEMORY_CONFIG_BYTES = 65_536


class ProjectMemoryConfigStatus(StrEnum):
    ABSENT = "absent"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ProjectMemoryConfig:
    status: ProjectMemoryConfigStatus = ProjectMemoryConfigStatus.VALID
    workspace_scope_keys: tuple[str, ...] = ()
    diagnostic_code: str | None = None
    # Kept as a compatibility surface for the existing invalid-config stderr.
    diagnostic: str | None = None
    # Explicit logical identity; never used as a filesystem/storage path.
    project_scope_key: str | None = None

    def project_key_for(self, repo_root: Path) -> str:
        """Preserve an owner-configured identity across physical directory moves."""
        return self.project_scope_key if self.project_scope_key is not None else str(repo_root)

    @property
    def valid(self) -> bool:
        return self.status is ProjectMemoryConfigStatus.VALID

    @property
    def error(self) -> str | None:
        """Compatibility alias for callers that use error terminology."""
        return self.diagnostic

    @property
    def workspace_keys(self) -> tuple[str, ...]:
        """Compatibility alias matching the TOML field name."""
        return self.workspace_scope_keys


def project_memory_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / MEMORY_CONFIG_RELATIVE_PATH


def load_project_memory_config(repo_root: Path) -> ProjectMemoryConfig:
    """Load project memory scope config; malformed config fails closed."""
    try:
        root = require_safe_directory(Path(repo_root))
        codex_dir = root / ".codex"
        if path_is_redirect(codex_dir):
            return _invalid("unsafe_path")
        try:
            codex_dir.lstat()
        except FileNotFoundError:
            return ProjectMemoryConfig(status=ProjectMemoryConfigStatus.ABSENT)
        except OSError:
            return _invalid("unsafe_path")
        safe_codex_dir = require_safe_directory(codex_dir, root)
        path = project_memory_config_path(safe_codex_dir.parent)
        if path_is_redirect(path):
            return _invalid("unsafe_path")
        try:
            path.lstat()
        except FileNotFoundError:
            return ProjectMemoryConfig(status=ProjectMemoryConfigStatus.ABSENT)
        safe_path = require_safe_file(path, safe_codex_dir, require_exists=True)
        if safe_path.stat().st_size > MAX_PROJECT_MEMORY_CONFIG_BYTES:
            return _invalid("too_large")
        with safe_path.open("rb") as stream:
            # Read one byte beyond the cap after opening the validated path.
            # The pre-open stat is only an optimization; a file can grow
            # between that check and the read, and truncating at the cap would
            # otherwise make an oversized TOML document look valid.
            raw = stream.read(MAX_PROJECT_MEMORY_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return ProjectMemoryConfig(status=ProjectMemoryConfigStatus.ABSENT)
    except RuntimePathError:
        return _invalid("unsafe_path")
    except OSError:
        return _invalid("read_error")

    try:
        if len(raw) > MAX_PROJECT_MEMORY_CONFIG_BYTES:
            return _invalid("too_large")
        document = tomllib.loads(raw.decode("utf-8"))
        if not isinstance(document, dict):
            return _invalid("invalid_schema")
        scope = document.get("scope", {})
        if not isinstance(scope, dict):
            return _invalid("invalid_schema")
        if set(scope) - {"workspace_keys", "project_key"}:
            return _invalid("invalid_schema")
        keys = scope.get("workspace_keys", [])
        if not isinstance(keys, list):
            return _invalid("invalid_schema")
        if any(not isinstance(value, str) for value in keys):
            return _invalid("invalid_schema")
        normalized = normalize_workspace_scope_keys(keys)
        project_key = scope.get("project_key")
        if "project_key" in scope:
            ScopeContext(project_key)
    except (UnicodeError, tomllib.TOMLDecodeError):
        return _invalid("invalid_toml")
    except ValidationError:
        return _invalid("invalid_scope")
    except (OSError, TypeError):
        return _invalid("read_error")
    return ProjectMemoryConfig(
        workspace_scope_keys=normalized,
        status=ProjectMemoryConfigStatus.VALID,
        project_scope_key=project_key,
    )


def load_workspace_scope_keys(repo_root: Path) -> tuple[str, ...]:
    """Return configured workspace keys while ignoring invalid config."""
    return load_project_memory_config(repo_root).workspace_scope_keys


def load_memory_config(repo_root: Path) -> ProjectMemoryConfig:
    """Compatibility alias for the project-local memory configuration loader."""
    return load_project_memory_config(repo_root)


def _invalid(code: str) -> ProjectMemoryConfig:
    return ProjectMemoryConfig(
        workspace_scope_keys=(),
        diagnostic=INVALID_MEMORY_CONFIG_DIAGNOSTIC,
        status=ProjectMemoryConfigStatus.INVALID,
        diagnostic_code=code,
    )
