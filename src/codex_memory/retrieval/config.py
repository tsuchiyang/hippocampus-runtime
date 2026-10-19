"""Optional, read-only configuration for P2.2 retrieval shadow mode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import ipaddress
from pathlib import Path
import tomllib
from typing import Any
from urllib.parse import urlsplit

from codex_memory.paths import (
    RuntimePathError,
    path_is_redirect,
    require_safe_directory,
    require_safe_file,
)
from .errors import RetrievalValidationError


RETRIEVAL_CONFIG_RELATIVE_PATH = Path(".codex") / "memory.toml"
MAX_RETRIEVAL_CONFIG_BYTES = 65_536


class RetrievalConfigStatus(StrEnum):
    ABSENT = "absent"
    VALID = "valid"
    DISABLED = "disabled"
    INVALID = "invalid"


class SemanticConfigStatus(StrEnum):
    ABSENT = "absent"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    version: int
    provider: str
    base_url: str
    protocol: str
    query_deadline_ms: int
    build_deadline_ms: int
    connect_timeout_ms: int
    read_timeout_ms: int
    dense: bool
    sparse: bool
    reranker: bool
    dense_top_k: int
    sparse_top_k: int
    rerank_top_n: int
    max_batch_items: int
    max_inflight: int
    queue_limit: int
    circuit_failure_threshold: int
    circuit_cooldown_ms: int
    expected_provider_fingerprint: str
    expected_dense_fingerprint: str | None
    expected_sparse_fingerprint: str | None
    expected_reranker_fingerprint: str | None

    def __post_init__(self) -> None:
        if not _exact_int(self.version, 1) or self.provider != "local_http" or self.protocol != "audited-local-v1":
            raise RetrievalValidationError("unsupported semantic configuration")
        if _loopback_url(self.base_url) != self.base_url:
            raise RetrievalValidationError("base_url must be canonical numeric loopback http")
        if not _bounded_int(self.query_deadline_ms, 100, 14_000) or not _bounded_int(self.build_deadline_ms, 1_000, 3_600_000):
            raise RetrievalValidationError("invalid semantic deadline")
        if not _bounded_int(self.connect_timeout_ms, 1, 3_600_000) or not _bounded_int(self.read_timeout_ms, 1, 3_600_000):
            raise RetrievalValidationError("invalid semantic timeout")
        if not _bounded_int(self.dense_top_k, 1, 256) or not _bounded_int(self.sparse_top_k, 1, 256) or not _bounded_int(self.rerank_top_n, 1, 64) or not _bounded_int(self.max_batch_items, 1, 64):
            raise RetrievalValidationError("invalid semantic limit")
        if not all(isinstance(value, bool) for value in (self.dense, self.sparse, self.reranker)):
            raise RetrievalValidationError("semantic components must be boolean")
        if not (_exact_int(self.max_inflight, 1) and _exact_int(self.queue_limit, 0) and _exact_int(self.circuit_failure_threshold, 2) and _exact_int(self.circuit_cooldown_ms, 30_000)):
            raise RetrievalValidationError("unsupported semantic admission or circuit policy")
        if not _is_sha256(self.expected_provider_fingerprint):
            raise RetrievalValidationError("provider fingerprint must be sha256")
        for enabled, fingerprint in ((self.dense, self.expected_dense_fingerprint), (self.sparse, self.expected_sparse_fingerprint), (self.reranker, self.expected_reranker_fingerprint)):
            if enabled != (fingerprint is not None) or (fingerprint is not None and not _is_sha256(fingerprint)):
                raise RetrievalValidationError("component fingerprint must exactly match enabled state")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "version": self.version, "provider": self.provider, "base_url": self.base_url, "protocol": self.protocol,
            "query_deadline_ms": self.query_deadline_ms, "build_deadline_ms": self.build_deadline_ms,
            "connect_timeout_ms": self.connect_timeout_ms, "read_timeout_ms": self.read_timeout_ms,
            "dense": self.dense, "sparse": self.sparse, "reranker": self.reranker,
            "dense_top_k": self.dense_top_k, "sparse_top_k": self.sparse_top_k,
            "rerank_top_n": self.rerank_top_n, "max_batch_items": self.max_batch_items,
            "max_inflight": self.max_inflight, "queue_limit": self.queue_limit,
            "circuit_failure_threshold": self.circuit_failure_threshold, "circuit_cooldown_ms": self.circuit_cooldown_ms,
            "expected_provider_fingerprint": self.expected_provider_fingerprint,
        }
        for enabled, field, fingerprint in ((self.dense, "expected_dense_fingerprint", self.expected_dense_fingerprint), (self.sparse, "expected_sparse_fingerprint", self.expected_sparse_fingerprint), (self.reranker, "expected_reranker_fingerprint", self.expected_reranker_fingerprint)):
            if enabled:
                data[field] = fingerprint
        return data

    @classmethod
    def from_dict(cls, value: Any) -> "SemanticConfig":
        config = _semantic_config(value)
        if config is None:
            raise RetrievalValidationError("invalid semantic configuration")
        return config


_HASH_LENGTH = 64
_SEMANTIC_BASE_FIELDS = {
    "version", "provider", "base_url", "protocol", "query_deadline_ms", "build_deadline_ms",
    "connect_timeout_ms", "read_timeout_ms", "dense", "sparse", "reranker", "dense_top_k",
    "sparse_top_k", "rerank_top_n", "max_batch_items", "max_inflight", "queue_limit",
    "circuit_failure_threshold", "circuit_cooldown_ms", "expected_provider_fingerprint",
}


def _semantic_config(value: Any) -> SemanticConfig | None:
    if not isinstance(value, dict):
        return None
    enabled = {"dense": "expected_dense_fingerprint", "sparse": "expected_sparse_fingerprint", "reranker": "expected_reranker_fingerprint"}
    expected_fields = set(_SEMANTIC_BASE_FIELDS)
    for component, field in enabled.items():
        flag = value.get(component)
        if not isinstance(flag, bool):
            return None
        if flag:
            expected_fields.add(field)
    if set(value) != expected_fields:
        return None
    if not _exact_int(value["version"], 1) or value["provider"] != "local_http" or value["protocol"] != "audited-local-v1":
        return None
    base_url = _loopback_url(value["base_url"])
    if base_url is None:
        return None
    if not _bounded_int(value["query_deadline_ms"], 100, 14_000) or not _bounded_int(value["build_deadline_ms"], 1_000, 3_600_000):
        return None
    if not _bounded_int(value["connect_timeout_ms"], 1, 3_600_000) or not _bounded_int(value["read_timeout_ms"], 1, 3_600_000):
        return None
    if not _bounded_int(value["dense_top_k"], 1, 256) or not _bounded_int(value["sparse_top_k"], 1, 256) or not _bounded_int(value["rerank_top_n"], 1, 64) or not _bounded_int(value["max_batch_items"], 1, 64):
        return None
    if not (_exact_int(value["max_inflight"], 1) and _exact_int(value["queue_limit"], 0) and _exact_int(value["circuit_failure_threshold"], 2) and _exact_int(value["circuit_cooldown_ms"], 30_000)):
        return None
    fingerprints = {"provider": value["expected_provider_fingerprint"]}
    for component, field in enabled.items():
        fingerprints[component] = value[field] if value[component] else None
    if any(fingerprint is not None and not _is_sha256(fingerprint) for fingerprint in fingerprints.values()):
        return None
    return SemanticConfig(
        value["version"], value["provider"], base_url, value["protocol"], value["query_deadline_ms"],
        value["build_deadline_ms"], value["connect_timeout_ms"], value["read_timeout_ms"], value["dense"],
        value["sparse"], value["reranker"], value["dense_top_k"], value["sparse_top_k"], value["rerank_top_n"],
        value["max_batch_items"], value["max_inflight"], value["queue_limit"], value["circuit_failure_threshold"],
        value["circuit_cooldown_ms"], fingerprints["provider"], fingerprints["dense"], fingerprints["sparse"], fingerprints["reranker"],
    )


def _bounded_int(value: Any, minimum: int, maximum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and minimum <= value <= maximum


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == _HASH_LENGTH and all(character in "0123456789abcdef" for character in value)


def _loopback_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        return None
    try:
        host = parsed.hostname
        if host is None:
            return None
        address = ipaddress.ip_address(host)
        port = parsed.port
    except ValueError:
        return None
    if not ((address.version == 4 and address in ipaddress.ip_network("127.0.0.0/8")) or address == ipaddress.IPv6Address("::1")):
        return None
    if port is None:
        return None
    host_text = str(address) if address.version == 4 else f"[{address}]"
    return f"http://{host_text}:{port}"


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Explicit, bounded P2.5 prompt sessions; never an all-session switch."""

    version: int
    enabled: bool
    session_ids: tuple[str, ...]
    total_deadline_ms: int

    def __post_init__(self) -> None:
        from .contracts import _str

        if not _exact_int(self.version, 1) or type(self.enabled) is not bool:
            raise RetrievalValidationError("invalid audit version or enabled flag")
        if not isinstance(self.session_ids, tuple) or len(self.session_ids) > 128:
            raise RetrievalValidationError("invalid audit session list")
        for session in self.session_ids:
            _str(session, "audit session", identifier=True)
        if len(set(self.session_ids)) != len(self.session_ids) or (self.enabled and not self.session_ids):
            raise RetrievalValidationError("audit sessions must be unique and explicit")
        if not _bounded_int(self.total_deadline_ms, 1, 1000):
            raise RetrievalValidationError("audit deadline must be between 1 and 1000ms")

    @classmethod
    def from_dict(cls, value: Any) -> "AuditConfig":
        if not isinstance(value, dict) or set(value) != {"version", "enabled", "session_ids", "total_deadline_ms"}:
            raise RetrievalValidationError("invalid audit configuration fields")
        if not isinstance(value["session_ids"], list):
            raise RetrievalValidationError("audit sessions must be a list")
        return cls(value["version"], value["enabled"], tuple(value["session_ids"]), value["total_deadline_ms"])


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """A bounded configuration result that never affects legacy memory scope."""

    status: RetrievalConfigStatus
    mode: str = "off"
    semantic_status: SemanticConfigStatus = SemanticConfigStatus.ABSENT
    semantic: SemanticConfig | None = None
    audit: AuditConfig | None = None
    config_sha256: str | None = None

    @property
    def enabled(self) -> bool:
        return self.status is RetrievalConfigStatus.VALID and self.mode == "shadow"

    @property
    def semantic_enabled(self) -> bool:
        return self.enabled and self.semantic_status is SemanticConfigStatus.VALID and self.semantic is not None

    def audit_for_session(self, session_id: str) -> AuditConfig | None:
        if self.enabled and self.audit is not None and self.audit.enabled and session_id in self.audit.session_ids:
            return self.audit
        return None


def load_retrieval_config(repo_root: Path) -> RetrievalConfig:
    """Read the optional retrieval section without creating or changing a path."""
    try:
        root = require_safe_directory(Path(repo_root))
        codex_dir = root / ".codex"
        if path_is_redirect(codex_dir):
            return _invalid()
        try:
            codex_dir.lstat()
        except FileNotFoundError:
            return _absent()
        safe_codex = require_safe_directory(codex_dir, root)
        path = safe_codex / "memory.toml"
        if path_is_redirect(path):
            return _invalid()
        try:
            path.lstat()
        except FileNotFoundError:
            return _absent()
        safe_path = require_safe_file(path, safe_codex, require_exists=True)
        if safe_path.stat().st_size > MAX_RETRIEVAL_CONFIG_BYTES:
            return _invalid()
        with safe_path.open("rb") as stream:
            raw = stream.read(MAX_RETRIEVAL_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return _absent()
    except (OSError, RuntimePathError):
        return _invalid()

    if len(raw) > MAX_RETRIEVAL_CONFIG_BYTES:
        return _invalid()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError):
        return _invalid()
    if not isinstance(document, dict):
        return _invalid()
    section = document.get("retrieval")
    if section is None:
        return _absent()
    if not isinstance(section, dict) or "mode" not in section or set(section) - {"mode", "semantic", "audit"}:
        return _invalid()
    audit = None
    if "audit" in section:
        try:
            audit = AuditConfig.from_dict(section["audit"])
        except RetrievalValidationError:
            return _invalid()
    import hashlib

    config_sha256 = hashlib.sha256(raw).hexdigest()
    mode = section.get("mode")
    if mode == "off":
        if "semantic" in section:
            return RetrievalConfig(RetrievalConfigStatus.VALID, "off", SemanticConfigStatus.INVALID, None)
        return RetrievalConfig(RetrievalConfigStatus.VALID, "off")
    if mode == "shadow":
        if "semantic" not in section:
            return RetrievalConfig(RetrievalConfigStatus.VALID, "shadow", audit=audit, config_sha256=config_sha256)
        semantic = _semantic_config(section["semantic"])
        return RetrievalConfig(RetrievalConfigStatus.VALID, "shadow", SemanticConfigStatus.VALID if semantic else SemanticConfigStatus.INVALID, semantic, audit, config_sha256)
    if mode == "active":
        return RetrievalConfig(RetrievalConfigStatus.DISABLED, "off")
    return _invalid()


def _absent() -> RetrievalConfig:
    return RetrievalConfig(RetrievalConfigStatus.ABSENT, "off")


def _invalid() -> RetrievalConfig:
    return RetrievalConfig(RetrievalConfigStatus.INVALID, "off")


def provider_from_config(config: SemanticConfig | None):
    """Construct only the single audited local profile; all else is no-I/O."""
    from .provider import provider_from_config as _provider_from_config

    return _provider_from_config(config)
