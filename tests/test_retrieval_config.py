from __future__ import annotations

from pathlib import Path

import pytest

from codex_memory.retrieval.config import (
    RetrievalConfigStatus,
    SemanticConfig,
    SemanticConfigStatus,
    load_retrieval_config,
)
from codex_memory.retrieval.errors import RetrievalValidationError
from codex_memory.project_memory_config import load_workspace_scope_keys


def _valid_semantic_values() -> dict[str, object]:
    return {
        "version": 1, "provider": "local_http", "base_url": "http://127.0.0.1:11434",
        "protocol": "audited-local-v1", "query_deadline_ms": 8000, "build_deadline_ms": 300000,
        "connect_timeout_ms": 500, "read_timeout_ms": 4000, "dense": True, "sparse": True,
        "reranker": True, "dense_top_k": 64, "sparse_top_k": 64, "rerank_top_n": 40,
        "max_batch_items": 16, "max_inflight": 1, "queue_limit": 0,
        "circuit_failure_threshold": 2, "circuit_cooldown_ms": 30000,
        "expected_provider_fingerprint": "1" * 64, "expected_dense_fingerprint": "2" * 64,
        "expected_sparse_fingerprint": "3" * 64, "expected_reranker_fingerprint": "4" * 64,
    }


def _semantic_toml(values: dict[str, object]) -> str:
    def render(value: object) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            return f"'{value}'"
        return str(value)
    return "[retrieval]\nmode = 'shadow'\n[retrieval.semantic]\n" + "\n".join(
        f"{key} = {render(value)}" for key, value in values.items()
    ) + "\n"


def test_missing_retrieval_config_defaults_to_off_without_creating_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    config = load_retrieval_config(repository)

    assert config.status is RetrievalConfigStatus.ABSENT
    assert config.enabled is False
    assert config.mode == "off"
    assert not (repository / ".codex").exists()


def test_explicit_shadow_is_the_only_enabled_mode(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    codex = repository / ".codex"
    codex.mkdir(parents=True)
    (codex / "memory.toml").write_text("[retrieval]\nmode = 'shadow'\n", encoding="utf-8")

    config = load_retrieval_config(repository)

    assert config.status is RetrievalConfigStatus.VALID
    assert config.enabled is True
    assert config.mode == "shadow"


def test_active_and_malformed_retrieval_config_disable_only_retrieval(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    codex = repository / ".codex"
    codex.mkdir(parents=True)
    config_path = codex / "memory.toml"

    config_path.write_text("[retrieval]\nmode = 'active'\n", encoding="utf-8")
    active = load_retrieval_config(repository)
    config_path.write_text("[retrieval\nmode = 'shadow'\n", encoding="utf-8")
    malformed = load_retrieval_config(repository)

    assert active.status is RetrievalConfigStatus.DISABLED
    assert active.mode == "off"
    assert active.enabled is False
    assert malformed.status is RetrievalConfigStatus.INVALID
    assert malformed.mode == "off"
    assert malformed.enabled is False


def test_invalid_retrieval_section_does_not_change_existing_workspace_scope_loading(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    codex = repository / ".codex"
    codex.mkdir(parents=True)
    (codex / "memory.toml").write_text(
        "[scope]\nworkspace_keys = ['team']\n\n[retrieval]\nmode = 'unsupported'\n",
        encoding="utf-8",
    )

    retrieval = load_retrieval_config(repository)

    assert retrieval.status is RetrievalConfigStatus.INVALID
    assert load_workspace_scope_keys(repository) == ("team",)


def test_strict_semantic_config_is_additive_to_shadow_only(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    codex = repository / ".codex"
    codex.mkdir(parents=True)
    hashes = "\n".join(
        f"expected_{name}_fingerprint = '{value * 64}'"
        for name, value in (("provider", "1"), ("dense", "2"), ("sparse", "3"), ("reranker", "4"))
    )
    (codex / "memory.toml").write_text(
        """[retrieval]
mode = 'shadow'
[retrieval.semantic]
version = 1
provider = 'local_http'
base_url = 'http://127.0.0.1:11434'
protocol = 'audited-local-v1'
query_deadline_ms = 8000
build_deadline_ms = 300000
connect_timeout_ms = 500
read_timeout_ms = 4000
dense = true
sparse = true
reranker = true
dense_top_k = 64
sparse_top_k = 64
rerank_top_n = 40
max_batch_items = 16
max_inflight = 1
queue_limit = 0
circuit_failure_threshold = 2
circuit_cooldown_ms = 30000
""" + hashes + "\n",
        encoding="utf-8",
    )

    config = load_retrieval_config(repository)

    assert config.status is RetrievalConfigStatus.VALID
    assert config.semantic_status is SemanticConfigStatus.VALID
    assert config.semantic is not None
    assert config.semantic.base_url == "http://127.0.0.1:11434"
    assert SemanticConfig.from_dict(config.semantic.to_dict()) == config.semantic


@pytest.mark.parametrize("base_url", [
    "https://127.0.0.1:11434", "http://localhost:11434", "http://10.0.0.1:11434",
    "http://127.0.0.1:11434/path", "http://user@127.0.0.1:11434", "http://127.0.0.1:11434?q=1",
])
def test_semantic_config_rejects_non_loopback_or_ambiguous_urls(tmp_path: Path, base_url: str) -> None:
    values = _valid_semantic_values()
    values["base_url"] = base_url
    with pytest.raises(RetrievalValidationError):
        SemanticConfig.from_dict(values)
    repository = tmp_path / "repository"
    codex = repository / ".codex"
    codex.mkdir(parents=True)
    (codex / "memory.toml").write_text(_semantic_toml(values), encoding="utf-8")
    config = load_retrieval_config(repository)
    assert config.status is RetrievalConfigStatus.VALID
    assert config.semantic_status is SemanticConfigStatus.INVALID
    assert config.semantic is None


@pytest.mark.parametrize(("field", "bad"), [
    ("version", True), ("version", 1.0), ("max_inflight", True), ("max_inflight", 1.0),
    ("queue_limit", False), ("queue_limit", 0.0), ("circuit_failure_threshold", True),
    ("circuit_failure_threshold", 2.0), ("circuit_cooldown_ms", 30_000.0),
])
def test_semantic_config_fixed_numbers_require_exact_integers(tmp_path: Path, field: str, bad: object) -> None:
    values = _valid_semantic_values()
    values[field] = bad
    with pytest.raises(RetrievalValidationError):
        SemanticConfig.from_dict(values)

    repository = tmp_path / "repository"
    codex = repository / ".codex"
    codex.mkdir(parents=True)
    (codex / "memory.toml").write_text(_semantic_toml(values), encoding="utf-8")
    config = load_retrieval_config(repository)
    assert config.status is RetrievalConfigStatus.VALID
    assert config.semantic_status is SemanticConfigStatus.INVALID
    assert config.semantic is None
