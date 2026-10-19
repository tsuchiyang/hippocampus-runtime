from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import re
from typing import Any, ClassVar

from .errors import RetrievalValidationError
from codex_memory.ledger_common import ValidationError
from codex_memory.memory_scope import ScopeContext


class RetrievalStatus(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    ABSENT = "absent"
    CORRUPT = "corrupt"
    UNSUPPORTED = "unsupported"


class RetrievalMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    ACTIVE = "active"


class DegradedReason(StrEnum):
    INDEX_ABSENT = "index_absent"
    INDEX_STALE = "index_stale"
    INDEX_CORRUPT = "index_corrupt"
    PATH_UNAVAILABLE = "path_unavailable"
    PATH_UNSAFE = "path_unsafe"
    SCHEMA_UNSUPPORTED = "schema_unsupported"
    CANONICAL_UNAVAILABLE = "canonical_unavailable"
    CANONICAL_INVALID = "canonical_invalid"
    FTS5_UNAVAILABLE = "fts5_unavailable"
    STALE_DELTA_USED = "stale_delta_used"
    STALE_DELTA_EXCEEDED = "stale_delta_exceeded"
    SEMANTIC_INDEX_UNAVAILABLE = "semantic_index_unavailable"
    SEMANTIC_INDEX_STALE = "semantic_index_stale"
    SEMANTIC_DENSE_UNAVAILABLE = "semantic_dense_unavailable"
    SEMANTIC_SPARSE_UNAVAILABLE = "semantic_sparse_unavailable"


_HASH = re.compile(r"\A[0-9a-f]{64}\Z")
_ID = re.compile(r"\A[^\s]+\Z")
_TIMESTAMP = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")


def canonical_contract_sha256(value: Any) -> str:
    """Hash JSON-safe contract metadata in its single canonical representation."""
    try:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RetrievalValidationError("contract metadata must be canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dict(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise RetrievalValidationError(f"{name} must be an object")
    if set(value) != fields:
        raise RetrievalValidationError(f"{name} has unknown or missing fields")
    return value


def _str(value: Any, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or (identifier and not _ID.fullmatch(value)):
        raise RetrievalValidationError(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RetrievalValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _enum(value: Any, enum: type[StrEnum], name: str) -> StrEnum:
    if isinstance(value, enum):
        return value
    if not isinstance(value, str):
        raise RetrievalValidationError(f"{name} must be a known value")
    try:
        return enum(value)
    except ValueError as exc:
        raise RetrievalValidationError(f"{name} must be a known value") from exc


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise RetrievalValidationError(f"{name} must be a UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RetrievalValidationError(f"{name} must be a UTC timestamp") from exc
    return value


class _Contract:
    _fields: ClassVar[set[str]]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RetrievalScope(_Contract):
    project_scope_key: str
    workspace_scope_keys: tuple[str, ...] = ()
    _fields: ClassVar[set[str]] = {"project_scope_key", "workspace_scope_keys"}

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_scope_keys, tuple):
            raise RetrievalValidationError("workspace_scope_keys must be a tuple")
        try:
            phase1b_scope = ScopeContext(
                self.project_scope_key,
                self.workspace_scope_keys,
            )
        except ValidationError as exc:
            raise RetrievalValidationError("invalid retrieval scope") from exc
        object.__setattr__(
            self,
            "workspace_scope_keys",
            phase1b_scope.workspace_scope_keys,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"project_scope_key": self.project_scope_key, "workspace_scope_keys": list(self.workspace_scope_keys)}

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalScope:
        d = _dict(value, cls._fields, "RetrievalScope")
        ws = d["workspace_scope_keys"]
        if not isinstance(ws, list):
            raise RetrievalValidationError("workspace_scope_keys must be a list")
        return cls(d["project_scope_key"], tuple(ws))


@dataclass(frozen=True, slots=True)
class EpistemicContext(_Contract):
    actor_id: str = "codex"
    role: str = "operator"
    memory_view_id: str = "ground_truth.default"
    epistemic_policy_id: str = "default"
    truth_domain: str = "ground_truth"
    _fields: ClassVar[set[str]] = {"actor_id", "role", "memory_view_id", "epistemic_policy_id", "truth_domain"}

    def __post_init__(self) -> None:
        for name in self._fields:
            _str(getattr(self, name), name, identifier=True)
        if self.memory_view_id != "ground_truth.default" or self.truth_domain != "ground_truth":
            raise RetrievalValidationError("only ground_truth epistemic context is supported")
        if self.epistemic_policy_id != "default":
            raise RetrievalValidationError("unsupported epistemic policy")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("actor_id", "role", "memory_view_id", "epistemic_policy_id", "truth_domain")}

    @classmethod
    def from_dict(cls, value: Any) -> EpistemicContext:
        d = _dict(value, cls._fields, "EpistemicContext")
        return cls(**d)


@dataclass(frozen=True, slots=True)
class IndexHighwater(_Contract):
    record_seq: int = 0
    memory_event_seq: int = 0
    evidence_seq: int = 0
    relation_seq: int = 0
    candidate_seq: int = 0
    candidate_event_seq: int = 0
    candidate_evidence_seq: int = 0
    _fields: ClassVar[set[str]] = {"record_seq", "memory_event_seq", "evidence_seq", "relation_seq", "candidate_seq", "candidate_event_seq", "candidate_evidence_seq"}

    def __post_init__(self) -> None:
        for name in self._fields: _int(getattr(self, name), name)

    _field_order: ClassVar[tuple[str, ...]] = (
        "record_seq", "memory_event_seq", "evidence_seq", "relation_seq",
        "candidate_seq", "candidate_event_seq", "candidate_evidence_seq",
    )

    def to_dict(self) -> dict[str, Any]: return {k: getattr(self, k) for k in self._field_order}

    @classmethod
    def from_dict(cls, value: Any) -> IndexHighwater:
        d = _dict(value, cls._fields, "IndexHighwater")
        return cls(**d)


@dataclass(frozen=True, slots=True)
class IndexGeneration(_Contract):
    generation_id: str
    state: str
    canonical_highwater: IndexHighwater
    indexed_highwater: IndexHighwater
    canonical_content_fingerprint: str
    provider_fingerprint: str
    model_fingerprint: str
    validation_state: str
    created_at: str
    updated_at: str
    degraded_reason: DegradedReason | None = None
    _fields: ClassVar[set[str]] = {"generation_id", "state", "canonical_highwater", "indexed_highwater", "canonical_content_fingerprint", "provider_fingerprint", "model_fingerprint", "validation_state", "created_at", "updated_at", "degraded_reason"}

    def __post_init__(self) -> None:
        _str(self.generation_id, "generation_id", identifier=True)
        if self.state not in {"building", "ready", "active", "retired", "failed"}: raise RetrievalValidationError("invalid generation state")
        if not isinstance(self.canonical_highwater, IndexHighwater) or not isinstance(self.indexed_highwater, IndexHighwater): raise RetrievalValidationError("highwater must be IndexHighwater")
        for name in ("canonical_content_fingerprint",):
            if not isinstance(getattr(self, name), str) or not _HASH.fullmatch(getattr(self, name)): raise RetrievalValidationError(f"{name} must be sha256")
        _str(self.provider_fingerprint, "provider_fingerprint"); _str(self.model_fingerprint, "model_fingerprint"); _str(self.validation_state, "validation_state")
        _timestamp(self.created_at, "created_at"); _timestamp(self.updated_at, "updated_at")
        if self.degraded_reason is not None and not isinstance(self.degraded_reason, DegradedReason): raise RetrievalValidationError("invalid degraded_reason")

    def to_dict(self) -> dict[str, Any]:
        return {"generation_id": self.generation_id, "state": self.state, "canonical_highwater": self.canonical_highwater.to_dict(), "indexed_highwater": self.indexed_highwater.to_dict(), "canonical_content_fingerprint": self.canonical_content_fingerprint, "provider_fingerprint": self.provider_fingerprint, "model_fingerprint": self.model_fingerprint, "validation_state": self.validation_state, "created_at": self.created_at, "updated_at": self.updated_at, "degraded_reason": self.degraded_reason.value if self.degraded_reason else None}

    @classmethod
    def from_dict(cls, value: Any) -> IndexGeneration:
        d = _dict(value, cls._fields, "IndexGeneration")
        return cls(d["generation_id"], d["state"], IndexHighwater.from_dict(d["canonical_highwater"]), IndexHighwater.from_dict(d["indexed_highwater"]), d["canonical_content_fingerprint"], d["provider_fingerprint"], d["model_fingerprint"], d["validation_state"], d["created_at"], d["updated_at"], None if d["degraded_reason"] is None else _enum(d["degraded_reason"], DegradedReason, "degraded_reason"))


@dataclass(frozen=True, slots=True)
class RetrievalRequest(_Contract):
    query: str | None
    session_id: str
    turn_id: str
    context_id: str
    prompt_event_uri: str | None
    scope: RetrievalScope
    epistemic_context: EpistemicContext = EpistemicContext()
    mode: RetrievalMode = RetrievalMode.OFF
    max_chars: int = 4000
    _fields: ClassVar[set[str]] = {"query", "session_id", "turn_id", "context_id", "prompt_event_uri", "scope", "epistemic_context", "mode", "max_chars"}

    def __post_init__(self) -> None:
        if self.query is not None and not isinstance(self.query, str): raise RetrievalValidationError("query must be text or null")
        for name in ("session_id", "turn_id", "context_id"): _str(getattr(self, name), name, identifier=True)
        if self.prompt_event_uri is not None: _str(self.prompt_event_uri, "prompt_event_uri", identifier=True)
        if not isinstance(self.scope, RetrievalScope) or not isinstance(self.epistemic_context, EpistemicContext): raise RetrievalValidationError("invalid nested contract")
        if not isinstance(self.mode, RetrievalMode): raise RetrievalValidationError("invalid mode")
        if isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int) or not 1 <= self.max_chars <= 4000: raise RetrievalValidationError("max_chars must be between 1 and 4000")

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "session_id": self.session_id, "turn_id": self.turn_id, "context_id": self.context_id, "prompt_event_uri": self.prompt_event_uri, "scope": self.scope.to_dict(), "epistemic_context": self.epistemic_context.to_dict(), "mode": self.mode.value, "max_chars": self.max_chars}

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalRequest:
        d = _dict(value, cls._fields, "RetrievalRequest")
        return cls(d["query"], d["session_id"], d["turn_id"], d["context_id"], d["prompt_event_uri"], RetrievalScope.from_dict(d["scope"]), EpistemicContext.from_dict(d["epistemic_context"]), _enum(d["mode"], RetrievalMode, "mode"), d["max_chars"])


@dataclass(frozen=True, slots=True)
class RetrievalResult(_Contract):
    status: RetrievalStatus
    mode: RetrievalMode
    generation: IndexGeneration | None
    canonical_highwater: IndexHighwater | None
    indexed_highwater: IndexHighwater | None
    degraded_reason: DegradedReason | None
    record_ids: tuple[str, ...] = ()
    _fields: ClassVar[set[str]] = {"status", "mode", "generation", "canonical_highwater", "indexed_highwater", "degraded_reason", "record_ids"}

    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus) or not isinstance(self.mode, RetrievalMode): raise RetrievalValidationError("invalid result enum")
        for x, typ in ((self.generation, IndexGeneration), (self.canonical_highwater, IndexHighwater), (self.indexed_highwater, IndexHighwater)):
            if x is not None and not isinstance(x, typ): raise RetrievalValidationError("invalid result nested contract")
        if self.degraded_reason is not None and not isinstance(self.degraded_reason, DegradedReason): raise RetrievalValidationError("invalid degraded_reason")
        if not isinstance(self.record_ids, tuple): raise RetrievalValidationError("record_ids must be a tuple")
        for x in self.record_ids: _str(x, "record_ids", identifier=True)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "mode": self.mode.value, "generation": self.generation.to_dict() if self.generation else None, "canonical_highwater": self.canonical_highwater.to_dict() if self.canonical_highwater else None, "indexed_highwater": self.indexed_highwater.to_dict() if self.indexed_highwater else None, "degraded_reason": self.degraded_reason.value if self.degraded_reason else None, "record_ids": list(self.record_ids)}

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalResult:
        d = _dict(value, cls._fields, "RetrievalResult")
        ids = d["record_ids"]
        if not isinstance(ids, list): raise RetrievalValidationError("record_ids must be a list")
        return cls(_enum(d["status"], RetrievalStatus, "status"), _enum(d["mode"], RetrievalMode, "mode"), None if d["generation"] is None else IndexGeneration.from_dict(d["generation"]), None if d["canonical_highwater"] is None else IndexHighwater.from_dict(d["canonical_highwater"]), None if d["indexed_highwater"] is None else IndexHighwater.from_dict(d["indexed_highwater"]), None if d["degraded_reason"] is None else _enum(d["degraded_reason"], DegradedReason, "degraded_reason"), tuple(ids))


class CandidateChannel(StrEnum):
    MANDATORY_CONSTITUTION = "mandatory_constitution"
    TEMPORARY = "temporary"
    EXACT = "exact"
    BM25 = "bm25"
    SYMBOL = "symbol"
    RECENT = "recent"
    STALE_DELTA = "stale_delta"
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True, slots=True)
class RetrievalPolicy(_Contract):
    rrf_k: int = 60
    exact_weight: int = 8
    symbol_weight: int = 6
    bm25_weight: int = 4
    recent_weight: int = 1
    stale_delta_weight: int = 8
    dense_weight: int = 4
    sparse_weight: int = 4
    dense_top_k: int = 64
    sparse_top_k: int = 64
    rerank_top_n: int = 40
    per_channel_limit: int = 32
    fused_candidate_limit: int = 64
    stale_delta_limit: int = 128
    _fields: ClassVar[set[str]] = {
        "rrf_k",
        "exact_weight",
        "symbol_weight",
        "bm25_weight",
        "recent_weight",
        "stale_delta_weight",
        "dense_weight",
        "sparse_weight",
        "dense_top_k",
        "sparse_top_k",
        "rerank_top_n",
        "per_channel_limit",
        "fused_candidate_limit",
        "stale_delta_limit",
    }
    _field_order: ClassVar[tuple[str, ...]] = (
        "rrf_k",
        "exact_weight",
        "symbol_weight",
        "bm25_weight",
        "recent_weight",
        "stale_delta_weight",
        "dense_weight",
        "sparse_weight",
        "dense_top_k",
        "sparse_top_k",
        "rerank_top_n",
        "per_channel_limit",
        "fused_candidate_limit",
        "stale_delta_limit",
    )

    def __post_init__(self) -> None:
        for name in self._field_order:
            _int(getattr(self, name), name, minimum=1)
        if self.dense_top_k > 256 or self.sparse_top_k > 256 or self.rerank_top_n > 64:
            raise RetrievalValidationError("semantic retrieval limits exceed their bounded policy")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._field_order}

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalPolicy:
        d = _dict(value, cls._fields, "RetrievalPolicy")
        return cls(**{name: d[name] for name in cls._field_order})


@dataclass(frozen=True, slots=True)
class CandidateContribution(_Contract):
    channel: CandidateChannel
    channel_rank: int
    numerator: int
    denominator: int
    _fields: ClassVar[set[str]] = {"channel", "channel_rank", "numerator", "denominator"}
    _field_order: ClassVar[tuple[str, ...]] = ("channel", "channel_rank", "numerator", "denominator")

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _enum(self.channel, CandidateChannel, "channel"))
        _int(self.channel_rank, "channel_rank", minimum=1)
        _int(self.numerator, "numerator", minimum=1)
        _int(self.denominator, "denominator", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "channel_rank": self.channel_rank,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateContribution:
        d = _dict(value, cls._fields, "CandidateContribution")
        return cls(_enum(d["channel"], CandidateChannel, "channel"), d["channel_rank"], d["numerator"], d["denominator"])


@dataclass(frozen=True, slots=True)
class RetrievedCandidate(_Contract):
    candidate_kind: str
    candidate_id: str
    record_type: str
    scope_kind: str
    scope_key: str
    state: str
    sequence: int
    content_sha256: str
    mandatory: bool
    fused_score_numerator: int
    fused_score_denominator: int
    contributions: tuple[CandidateContribution, ...]
    trust_label: str | None = None
    _fields: ClassVar[set[str]] = {
        "candidate_kind",
        "candidate_id",
        "record_type",
        "scope_kind",
        "scope_key",
        "state",
        "sequence",
        "content_sha256",
        "mandatory",
        "fused_score_numerator",
        "fused_score_denominator",
        "contributions",
        "trust_label",
    }
    _field_order: ClassVar[tuple[str, ...]] = (
        "candidate_kind",
        "candidate_id",
        "record_type",
        "scope_kind",
        "scope_key",
        "state",
        "sequence",
        "content_sha256",
        "mandatory",
        "fused_score_numerator",
        "fused_score_denominator",
        "contributions",
        "trust_label",
    )

    def __post_init__(self) -> None:
        if self.candidate_kind not in {"record", "temporary"}:
            raise RetrievalValidationError("invalid candidate_kind")
        for name in ("candidate_id", "record_type", "scope_kind", "scope_key", "state"):
            _str(getattr(self, name), name)
        _int(self.sequence, "sequence", minimum=0)
        if not isinstance(self.content_sha256, str) or not _HASH.fullmatch(self.content_sha256):
            raise RetrievalValidationError("content_sha256 must be sha256")
        if not isinstance(self.mandatory, bool):
            raise RetrievalValidationError("mandatory must be boolean")
        _int(self.fused_score_numerator, "fused_score_numerator", minimum=0)
        _int(self.fused_score_denominator, "fused_score_denominator", minimum=1)
        if not isinstance(self.contributions, tuple):
            raise RetrievalValidationError("contributions must be a tuple")
        normalized = []
        for contribution in self.contributions:
            if not isinstance(contribution, CandidateContribution):
                raise RetrievalValidationError("invalid contribution")
            normalized.append(contribution)
        object.__setattr__(self, "contributions", tuple(normalized))
        if self.trust_label is not None:
            _str(self.trust_label, "trust_label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_kind": self.candidate_kind,
            "candidate_id": self.candidate_id,
            "record_type": self.record_type,
            "scope_kind": self.scope_kind,
            "scope_key": self.scope_key,
            "state": self.state,
            "sequence": self.sequence,
            "content_sha256": self.content_sha256,
            "mandatory": self.mandatory,
            "fused_score_numerator": self.fused_score_numerator,
            "fused_score_denominator": self.fused_score_denominator,
            "contributions": [contribution.to_dict() for contribution in self.contributions],
            "trust_label": self.trust_label,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RetrievedCandidate:
        d = _dict(value, cls._fields, "RetrievedCandidate")
        contributions = d["contributions"]
        if not isinstance(contributions, list):
            raise RetrievalValidationError("contributions must be a list")
        return cls(
            d["candidate_kind"],
            d["candidate_id"],
            d["record_type"],
            d["scope_kind"],
            d["scope_key"],
            d["state"],
            d["sequence"],
            d["content_sha256"],
            d["mandatory"],
            d["fused_score_numerator"],
            d["fused_score_denominator"],
            tuple(CandidateContribution.from_dict(item) for item in contributions),
            d["trust_label"],
        )


@dataclass(frozen=True, slots=True)
class CandidateTrace(_Contract):
    candidate_id: str
    source_channels: tuple[CandidateChannel, ...]
    contributions: tuple[CandidateContribution, ...]
    fused_rank: int | None
    selected: bool
    reason_codes: tuple[str, ...]
    raw_score_hex: str | None = None
    _fields: ClassVar[set[str]] = {
        "candidate_id",
        "source_channels",
        "contributions",
        "fused_rank",
        "selected",
        "reason_codes",
        "raw_score_hex",
    }
    _field_order: ClassVar[tuple[str, ...]] = (
        "candidate_id",
        "source_channels",
        "contributions",
        "fused_rank",
        "selected",
        "reason_codes",
        "raw_score_hex",
    )

    def __post_init__(self) -> None:
        _str(self.candidate_id, "candidate_id")
        if not isinstance(self.source_channels, tuple):
            raise RetrievalValidationError("source_channels must be a tuple")
        object.__setattr__(
            self,
            "source_channels",
            tuple(_enum(channel, CandidateChannel, "source_channels") for channel in self.source_channels),
        )
        if not isinstance(self.contributions, tuple):
            raise RetrievalValidationError("contributions must be a tuple")
        normalized = []
        for contribution in self.contributions:
            if not isinstance(contribution, CandidateContribution):
                raise RetrievalValidationError("invalid contribution")
            normalized.append(contribution)
        object.__setattr__(self, "contributions", tuple(normalized))
        if self.fused_rank is not None:
            _int(self.fused_rank, "fused_rank", minimum=1)
        if not isinstance(self.selected, bool):
            raise RetrievalValidationError("selected must be boolean")
        if not isinstance(self.reason_codes, tuple):
            raise RetrievalValidationError("reason_codes must be a tuple")
        for reason_code in self.reason_codes:
            _str(reason_code, "reason_codes")
        if self.raw_score_hex is not None:
            if not isinstance(self.raw_score_hex, str):
                raise RetrievalValidationError("raw_score_hex must be float.hex text or null")
            try:
                import math
                score = float.fromhex(self.raw_score_hex)
                if not math.isfinite(score) or score.hex() != self.raw_score_hex:
                    raise ValueError
            except ValueError as exc:
                raise RetrievalValidationError("raw_score_hex must be finite float.hex text or null") from exc

    def to_dict(self) -> dict[str, Any]:
        result = {
            "candidate_id": self.candidate_id,
            "source_channels": [channel.value for channel in self.source_channels],
            "contributions": [contribution.to_dict() for contribution in self.contributions],
            "fused_rank": self.fused_rank,
            "selected": self.selected,
            "reason_codes": list(self.reason_codes),
        }
        # Keep deterministic-only P2.2 public output byte-compatible.
        if self.raw_score_hex is not None:
            result["raw_score_hex"] = self.raw_score_hex
        return result

    @classmethod
    def from_dict(cls, value: Any) -> CandidateTrace:
        if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
            raise RetrievalValidationError("CandidateTrace must be an object")
        fields = cls._fields if "raw_score_hex" in value else cls._fields - {"raw_score_hex"}
        d = _dict(value, fields, "CandidateTrace")
        source_channels = d["source_channels"]
        contributions = d["contributions"]
        reason_codes = d["reason_codes"]
        if not isinstance(source_channels, list):
            raise RetrievalValidationError("source_channels must be a list")
        if not isinstance(contributions, list):
            raise RetrievalValidationError("contributions must be a list")
        if not isinstance(reason_codes, list):
            raise RetrievalValidationError("reason_codes must be a list")
        return cls(
            d["candidate_id"],
            tuple(_enum(channel, CandidateChannel, "source_channels") for channel in source_channels),
            tuple(CandidateContribution.from_dict(item) for item in contributions),
            d["fused_rank"],
            d["selected"],
            tuple(reason_codes),
            d.get("raw_score_hex"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalQueryResult(_Contract):
    status: RetrievalStatus
    mode: RetrievalMode
    generation: IndexGeneration | None
    canonical_highwater: IndexHighwater | None
    indexed_highwater: IndexHighwater | None
    index_stale: bool
    degraded_reasons: tuple[DegradedReason, ...]
    candidates: tuple[RetrievedCandidate, ...]
    trace: tuple[CandidateTrace, ...]
    _fields: ClassVar[set[str]] = {
        "status",
        "mode",
        "generation",
        "canonical_highwater",
        "indexed_highwater",
        "index_stale",
        "degraded_reasons",
        "candidates",
        "trace",
    }
    _field_order: ClassVar[tuple[str, ...]] = (
        "status",
        "mode",
        "generation",
        "canonical_highwater",
        "indexed_highwater",
        "index_stale",
        "degraded_reasons",
        "candidates",
        "trace",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus) or not isinstance(self.mode, RetrievalMode):
            raise RetrievalValidationError("invalid result enum")
        for x, typ in ((self.generation, IndexGeneration), (self.canonical_highwater, IndexHighwater), (self.indexed_highwater, IndexHighwater)):
            if x is not None and not isinstance(x, typ):
                raise RetrievalValidationError("invalid result nested contract")
        if not isinstance(self.index_stale, bool):
            raise RetrievalValidationError("index_stale must be boolean")
        if not isinstance(self.degraded_reasons, tuple):
            raise RetrievalValidationError("degraded_reasons must be a tuple")
        if not isinstance(self.candidates, tuple):
            raise RetrievalValidationError("candidates must be a tuple")
        if not isinstance(self.trace, tuple):
            raise RetrievalValidationError("trace must be a tuple")
        for degraded_reason in self.degraded_reasons:
            if not isinstance(degraded_reason, DegradedReason):
                raise RetrievalValidationError("invalid degraded_reason")
        for candidate in self.candidates:
            if not isinstance(candidate, RetrievedCandidate):
                raise RetrievalValidationError("invalid candidate")
        for trace in self.trace:
            if not isinstance(trace, CandidateTrace):
                raise RetrievalValidationError("invalid trace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "mode": self.mode.value,
            "generation": self.generation.to_dict() if self.generation else None,
            "canonical_highwater": self.canonical_highwater.to_dict() if self.canonical_highwater else None,
            "indexed_highwater": self.indexed_highwater.to_dict() if self.indexed_highwater else None,
            "index_stale": self.index_stale,
            "degraded_reasons": [reason.value for reason in self.degraded_reasons],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "trace": [trace.to_dict() for trace in self.trace],
        }

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalQueryResult:
        d = _dict(value, cls._fields, "RetrievalQueryResult")
        degraded_reasons = d["degraded_reasons"]
        candidates = d["candidates"]
        trace = d["trace"]
        if not isinstance(degraded_reasons, list):
            raise RetrievalValidationError("degraded_reasons must be a list")
        if not isinstance(candidates, list):
            raise RetrievalValidationError("candidates must be a list")
        if not isinstance(trace, list):
            raise RetrievalValidationError("trace must be a list")
        return cls(
            _enum(d["status"], RetrievalStatus, "status"),
            _enum(d["mode"], RetrievalMode, "mode"),
            None if d["generation"] is None else IndexGeneration.from_dict(d["generation"]),
            None if d["canonical_highwater"] is None else IndexHighwater.from_dict(d["canonical_highwater"]),
            None if d["indexed_highwater"] is None else IndexHighwater.from_dict(d["indexed_highwater"]),
            d["index_stale"],
            tuple(_enum(value, DegradedReason, "degraded_reasons") for value in degraded_reasons),
            tuple(RetrievedCandidate.from_dict(item) for item in candidates),
            tuple(CandidateTrace.from_dict(item) for item in trace),
        )
