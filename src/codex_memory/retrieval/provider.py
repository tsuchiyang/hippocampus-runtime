"""Strict, dependency-free boundary for optional local retrieval models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import math
import re
from typing import Any, Protocol

from .contracts import canonical_contract_sha256
from .errors import RetrievalValidationError


_HASH = re.compile(r"\A[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"\A[^\s]+\Z")
_MAX_SPARSE_TERMS = 4_096
_MAX_SPARSE_TERM_BYTES = 256


class ModelComponent(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    RERANKER = "reranker"


class ProviderOperation(StrEnum):
    CAPABILITIES = "capabilities"
    DENSE_DOCUMENTS = "dense_documents"
    DENSE_QUERY = "dense_query"
    SPARSE_DOCUMENTS = "sparse_documents"
    SPARSE_QUERY = "sparse_query"
    RERANK = "rerank"


class CapabilityState(StrEnum):
    DISABLED = "disabled"
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


class ProviderCallStatus(StrEnum):
    OK = "ok"
    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    BUSY = "busy"
    TIMEOUT = "timeout"
    INPUT_TOO_LONG = "input_too_long"
    INVALID_RESPONSE = "invalid_response"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    CIRCUIT_OPEN = "circuit_open"
    DEADLINE_EXHAUSTED = "deadline_exhausted"


_SAFE_REASON_CODES = frozenset(status.value for status in ProviderCallStatus)


def _enum(value: Any, kind: type[StrEnum], name: str) -> StrEnum:
    if isinstance(value, kind):
        return value
    if isinstance(value, str):
        try:
            return kind(value)
        except ValueError:
            pass
    raise RetrievalValidationError(f"{name} must be a known value")


def _string(value: Any, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or (identifier and not _IDENTIFIER.fullmatch(value)):
        raise RetrievalValidationError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise RetrievalValidationError(f"{name} must be a lowercase sha256")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RetrievalValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RetrievalValidationError(f"{name} must be boolean")
    return value


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields or any(not isinstance(key, str) for key in value):
        raise RetrievalValidationError(f"{name} has unknown or missing fields")
    return value


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    component: ModelComponent
    model_id: str
    revision: str
    digest: str
    dimension: int | None
    max_input_tokens: int | None
    max_input_chars: int | None
    truncation: str
    normalization: str
    quantization: str
    adapters: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _enum(self.component, ModelComponent, "component"))
        for name in ("model_id", "revision", "truncation", "normalization", "quantization"):
            _string(getattr(self, name), name, identifier=name in {"model_id", "revision"})
        _sha256(self.digest, "digest")
        if self.dimension is not None:
            _integer(self.dimension, "dimension", minimum=1)
        if self.component is ModelComponent.DENSE and self.dimension is None:
            raise RetrievalValidationError("dense identity requires dimension")
        for name in ("max_input_tokens", "max_input_chars"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, minimum=1)
        if self.max_input_tokens is None and self.max_input_chars is None:
            raise RetrievalValidationError("identity requires an explicit input limit")
        if not isinstance(self.adapters, tuple):
            raise RetrievalValidationError("adapters must be a tuple")
        for adapter in self.adapters:
            _string(adapter, "adapter", identifier=True)

    @property
    def fingerprint(self) -> str:
        return canonical_contract_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"component": self.component.value, "model_id": self.model_id, "revision": self.revision,
                "digest": self.digest, "dimension": self.dimension, "max_input_tokens": self.max_input_tokens,
                "max_input_chars": self.max_input_chars, "truncation": self.truncation,
                "normalization": self.normalization, "quantization": self.quantization, "adapters": list(self.adapters)}

    @classmethod
    def from_dict(cls, value: Any) -> "ModelIdentity":
        data = _object(value, {"component", "model_id", "revision", "digest", "dimension", "max_input_tokens", "max_input_chars", "truncation", "normalization", "quantization", "adapters"}, "ModelIdentity")
        if not isinstance(data["adapters"], list):
            raise RetrievalValidationError("adapters must be a list")
        return cls(_enum(data["component"], ModelComponent, "component"), data["model_id"], data["revision"], data["digest"], data["dimension"], data["max_input_tokens"], data["max_input_chars"], data["truncation"], data["normalization"], data["quantization"], tuple(data["adapters"]))


@dataclass(frozen=True, slots=True)
class ComponentCapability:
    component: ModelComponent
    state: CapabilityState
    identity: ModelIdentity | None
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _enum(self.component, ModelComponent, "component"))
        object.__setattr__(self, "state", _enum(self.state, CapabilityState, "state"))
        if self.state is CapabilityState.AVAILABLE:
            if not isinstance(self.identity, ModelIdentity) or self.identity.component is not self.component or self.reason is not None:
                raise RetrievalValidationError("available component requires matching identity and no reason")
        elif self.identity is not None:
            raise RetrievalValidationError("non-available component cannot carry identity")
        if self.reason is not None:
            _string(self.reason, "reason")

    def to_dict(self) -> dict[str, Any]:
        return {"component": self.component.value, "state": self.state.value,
                "identity": self.identity.to_dict() if self.identity else None, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Any) -> "ComponentCapability":
        data = _object(value, {"component", "state", "identity", "reason"}, "ComponentCapability")
        return cls(_enum(data["component"], ModelComponent, "component"), _enum(data["state"], CapabilityState, "state"), None if data["identity"] is None else ModelIdentity.from_dict(data["identity"]), data["reason"])


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    protocol_version: int
    provider_type: str
    provider_version: str
    provider_digest: str
    dense: ComponentCapability
    sparse: ComponentCapability
    reranker: ComponentCapability
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _integer(self.protocol_version, "protocol_version", minimum=1)
        _string(self.provider_type, "provider_type", identifier=True)
        _string(self.provider_version, "provider_version", identifier=True)
        _sha256(self.provider_digest, "provider_digest")
        for component, expected in ((self.dense, ModelComponent.DENSE), (self.sparse, ModelComponent.SPARSE), (self.reranker, ModelComponent.RERANKER)):
            if not isinstance(component, ComponentCapability) or component.component is not expected:
                raise RetrievalValidationError("capabilities must contain each exact component")
        object.__setattr__(self, "fingerprint", canonical_contract_sha256(self._fingerprint_payload()))

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "provider_type": self.provider_type, "provider_version": self.provider_version, "provider_digest": self.provider_digest, "dense": self.dense.to_dict(), "sparse": self.sparse.to_dict(), "reranker": self.reranker.to_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {**self._fingerprint_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Any) -> "ProviderCapabilities":
        data = _object(value, {"protocol_version", "provider_type", "provider_version", "provider_digest", "dense", "sparse", "reranker", "fingerprint"}, "ProviderCapabilities")
        result = cls(data["protocol_version"], data["provider_type"], data["provider_version"], data["provider_digest"], ComponentCapability.from_dict(data["dense"]), ComponentCapability.from_dict(data["sparse"]), ComponentCapability.from_dict(data["reranker"]))
        if data["fingerprint"] != result.fingerprint:
            raise RetrievalValidationError("provider fingerprint does not match canonical capability metadata")
        return result


@dataclass(frozen=True, slots=True)
class ProviderInput:
    item_id: str
    position: int
    text: str
    text_sha256: str

    def __post_init__(self) -> None:
        _string(self.item_id, "item_id", identifier=True)
        _integer(self.position, "position")
        if not isinstance(self.text, str):
            raise RetrievalValidationError("text must be a string")
        _sha256(self.text_sha256, "text_sha256")
        if self.text_sha256 != self.sha256_text(self.text):
            raise RetrievalValidationError("text_sha256 does not match text")

    @staticmethod
    def sha256_text(text: str) -> str:
        if not isinstance(text, str):
            raise RetrievalValidationError("text must be a string")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def from_text(cls, item_id: str, position: int, text: str) -> "ProviderInput":
        return cls(item_id, position, text, cls.sha256_text(text))

@dataclass(frozen=True, slots=True)
class DenseItem:
    item_id: str
    position: int
    values: tuple[float, ...]
    truncated: bool

    def __post_init__(self) -> None:
        _string(self.item_id, "item_id", identifier=True); _integer(self.position, "position"); _boolean(self.truncated, "truncated")
        if not isinstance(self.values, tuple) or not self.values:
            raise RetrievalValidationError("dense values must be a non-empty tuple")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in self.values):
            raise RetrievalValidationError("dense values must be finite")
        if math.fsum(float(value) * float(value) for value in self.values) <= 0.0:
            raise RetrievalValidationError("dense values must have non-zero norm")

    def to_dict(self) -> dict[str, Any]: return {"item_id": self.item_id, "position": self.position, "values": list(self.values), "truncated": self.truncated}
    @classmethod
    def from_dict(cls, value: Any) -> "DenseItem":
        data = _object(value, {"item_id", "position", "values", "truncated"}, "DenseItem")
        if not isinstance(data["values"], list): raise RetrievalValidationError("dense values must be a list")
        return cls(data["item_id"], data["position"], tuple(data["values"]), data["truncated"])


@dataclass(frozen=True, slots=True)
class SparseItem:
    item_id: str
    position: int
    weights: tuple[tuple[str, float], ...]
    truncated: bool

    def __post_init__(self) -> None:
        _string(self.item_id, "item_id", identifier=True); _integer(self.position, "position"); _boolean(self.truncated, "truncated")
        if not isinstance(self.weights, tuple) or len(self.weights) > _MAX_SPARSE_TERMS:
            raise RetrievalValidationError("sparse weights must be a bounded tuple")
        seen: set[str] = set()
        for entry in self.weights:
            if not isinstance(entry, tuple) or len(entry) != 2: raise RetrievalValidationError("sparse weight must be a pair")
            term, weight = entry
            _string(term, "sparse term")
            if len(term.encode("utf-8")) > _MAX_SPARSE_TERM_BYTES or term in seen: raise RetrievalValidationError("sparse terms must be unique and bounded")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0: raise RetrievalValidationError("sparse weights must be finite and non-negative")
            seen.add(term)

    def to_dict(self) -> dict[str, Any]: return {"item_id": self.item_id, "position": self.position, "weights": [[term, weight] for term, weight in self.weights], "truncated": self.truncated}
    @classmethod
    def from_dict(cls, value: Any) -> "SparseItem":
        data = _object(value, {"item_id", "position", "weights", "truncated"}, "SparseItem")
        if not isinstance(data["weights"], list): raise RetrievalValidationError("sparse weights must be a list")
        pairs = []
        for entry in data["weights"]:
            if not isinstance(entry, list) or len(entry) != 2: raise RetrievalValidationError("sparse weight must be a pair")
            pairs.append((entry[0], entry[1]))
        return cls(data["item_id"], data["position"], tuple(pairs), data["truncated"])


@dataclass(frozen=True, slots=True)
class RerankItem:
    item_id: str
    position: int
    score: float
    truncated: bool

    def __post_init__(self) -> None:
        _string(self.item_id, "item_id", identifier=True); _integer(self.position, "position"); _boolean(self.truncated, "truncated")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise RetrievalValidationError("rerank score must be finite")

    def to_dict(self) -> dict[str, Any]: return {"item_id": self.item_id, "position": self.position, "score": self.score, "truncated": self.truncated}
    @classmethod
    def from_dict(cls, value: Any) -> "RerankItem":
        data = _object(value, {"item_id", "position", "score", "truncated"}, "RerankItem")
        return cls(**data)


_OPERATION_COMPONENT = {ProviderOperation.DENSE_DOCUMENTS: ModelComponent.DENSE, ProviderOperation.DENSE_QUERY: ModelComponent.DENSE, ProviderOperation.SPARSE_DOCUMENTS: ModelComponent.SPARSE, ProviderOperation.SPARSE_QUERY: ModelComponent.SPARSE, ProviderOperation.RERANK: ModelComponent.RERANKER}


@dataclass(frozen=True, slots=True)
class ProviderCall:
    operation: ProviderOperation
    component: ModelComponent | None
    status: ProviderCallStatus
    provider_fingerprint: str | None
    model_fingerprint: str | None
    elapsed_ms: int
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _enum(self.operation, ProviderOperation, "operation")); object.__setattr__(self, "status", _enum(self.status, ProviderCallStatus, "status"))
        if self.component is not None: object.__setattr__(self, "component", _enum(self.component, ModelComponent, "component"))
        expected = _OPERATION_COMPONENT.get(self.operation)
        if (self.operation is ProviderOperation.CAPABILITIES and self.component is not None) or (expected is not None and self.component is not expected):
            raise RetrievalValidationError("operation and component must be an exact pair")
        if self.operation is ProviderOperation.CAPABILITIES and self.model_fingerprint is not None:
            raise RetrievalValidationError("capability calls cannot carry a model fingerprint")
        if self.provider_fingerprint is not None: _sha256(self.provider_fingerprint, "provider_fingerprint")
        if self.model_fingerprint is not None: _sha256(self.model_fingerprint, "model_fingerprint")
        _integer(self.elapsed_ms, "elapsed_ms")
        if self.reason is not None and (not isinstance(self.reason, str) or self.reason not in _SAFE_REASON_CODES):
            raise RetrievalValidationError("reason must be a safe provider status code")

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.value, "component": self.component.value if self.component else None, "status": self.status.value, "provider_fingerprint": self.provider_fingerprint, "model_fingerprint": self.model_fingerprint, "elapsed_ms": self.elapsed_ms, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Any) -> "ProviderCall":
        data = _object(value, {"operation", "component", "status", "provider_fingerprint", "model_fingerprint", "elapsed_ms", "reason"}, "ProviderCall")
        return cls(_enum(data["operation"], ProviderOperation, "operation"), None if data["component"] is None else _enum(data["component"], ModelComponent, "component"), _enum(data["status"], ProviderCallStatus, "status"), data["provider_fingerprint"], data["model_fingerprint"], data["elapsed_ms"], data["reason"])


@dataclass(frozen=True, slots=True)
class CapabilitiesResponse:
    call: ProviderCall
    capabilities: ProviderCapabilities | None
    def __post_init__(self) -> None:
        if not isinstance(self.call, ProviderCall) or self.call.operation is not ProviderOperation.CAPABILITIES: raise RetrievalValidationError("invalid capabilities call")
        if self.capabilities is not None and not isinstance(self.capabilities, ProviderCapabilities): raise RetrievalValidationError("invalid capabilities")
        if self.call.status is ProviderCallStatus.OK and self.capabilities is None: raise RetrievalValidationError("ok capabilities response requires capabilities")


@dataclass(frozen=True, slots=True)
class DenseResponse:
    call: ProviderCall
    items: tuple[DenseItem, ...]
    def __post_init__(self) -> None: _response_shape(self.call, self.items, ProviderOperation.DENSE_DOCUMENTS, ProviderOperation.DENSE_QUERY, DenseItem)


@dataclass(frozen=True, slots=True)
class SparseResponse:
    call: ProviderCall
    items: tuple[SparseItem, ...]
    def __post_init__(self) -> None: _response_shape(self.call, self.items, ProviderOperation.SPARSE_DOCUMENTS, ProviderOperation.SPARSE_QUERY, SparseItem)


@dataclass(frozen=True, slots=True)
class RerankResponse:
    call: ProviderCall
    items: tuple[RerankItem, ...]
    def __post_init__(self) -> None: _response_shape(self.call, self.items, ProviderOperation.RERANK, None, RerankItem)


def _response_shape(call: Any, items: Any, first: ProviderOperation, second: ProviderOperation | None, item_type: type[Any]) -> None:
    if not isinstance(call, ProviderCall) or call.operation not in {first, second}: raise RetrievalValidationError("response operation does not match type")
    if not isinstance(items, tuple) or any(not isinstance(item, item_type) for item in items): raise RetrievalValidationError("response items must be a tuple of matching items")
    if call.status is not ProviderCallStatus.OK and items:
        raise RetrievalValidationError("non-ok responses must not carry model items")


def _failure_call(call: ProviderCall, status: ProviderCallStatus) -> ProviderCall:
    return replace(call, status=status, reason=status.value)


def _is_expired(deadline: Any, now_ns: Callable[[], int] | None) -> bool:
    return deadline is not None and now_ns is not None and deadline.remaining_ms(now_ns=now_ns) == 0


def _checked(response: Any, requested: Sequence[ProviderInput], *, dimension: int | None = None, deadline: Any = None, now_ns: Callable[[], int] | None = None, expected_provider_fingerprint: str | None = None, expected_model_fingerprint: str | None = None) -> Any:
    if not isinstance(requested, Sequence) or any(not isinstance(item, ProviderInput) for item in requested): raise RetrievalValidationError("requested items must be provider inputs")
    if expected_provider_fingerprint is not None: _sha256(expected_provider_fingerprint, "expected_provider_fingerprint")
    if expected_model_fingerprint is not None: _sha256(expected_model_fingerprint, "expected_model_fingerprint")
    if _is_expired(deadline, now_ns): return type(response)(_failure_call(response.call, ProviderCallStatus.TIMEOUT), ())
    if response.call.status is not ProviderCallStatus.OK: return response
    if (
        expected_provider_fingerprint is not None and response.call.provider_fingerprint != expected_provider_fingerprint
    ) or (
        expected_model_fingerprint is not None and response.call.model_fingerprint != expected_model_fingerprint
    ):
        return type(response)(_failure_call(response.call, ProviderCallStatus.FINGERPRINT_MISMATCH), ())
    if tuple(item.position for item in requested) != tuple(range(len(requested))) or len(response.items) != len(requested) or any((result.item_id, result.position) != (source.item_id, source.position) for source, result in zip(requested, response.items)):
        return type(response)(_failure_call(response.call, ProviderCallStatus.INVALID_RESPONSE), ())
    if any(item.truncated for item in response.items): return type(response)(_failure_call(response.call, ProviderCallStatus.INPUT_TOO_LONG), ())
    if dimension is not None and any(len(item.values) != dimension for item in response.items): return type(response)(_failure_call(response.call, ProviderCallStatus.INVALID_RESPONSE), ())
    return response


def checked_dense_response(response: DenseResponse, requested: Sequence[ProviderInput], *, dimension: int, deadline: Any = None, now_ns: Callable[[], int] | None = None, expected_provider_fingerprint: str | None = None, expected_model_fingerprint: str | None = None) -> DenseResponse:
    _integer(dimension, "dimension", minimum=1)
    return _checked(response, requested, dimension=dimension, deadline=deadline, now_ns=now_ns, expected_provider_fingerprint=expected_provider_fingerprint, expected_model_fingerprint=expected_model_fingerprint)


def checked_sparse_response(response: SparseResponse, requested: Sequence[ProviderInput], *, deadline: Any = None, now_ns: Callable[[], int] | None = None, expected_provider_fingerprint: str | None = None, expected_model_fingerprint: str | None = None) -> SparseResponse:
    return _checked(response, requested, deadline=deadline, now_ns=now_ns, expected_provider_fingerprint=expected_provider_fingerprint, expected_model_fingerprint=expected_model_fingerprint)


def checked_rerank_response(response: RerankResponse, requested: Sequence[ProviderInput], *, deadline: Any = None, now_ns: Callable[[], int] | None = None, expected_provider_fingerprint: str | None = None, expected_model_fingerprint: str | None = None) -> RerankResponse:
    return _checked(response, requested, deadline=deadline, now_ns=now_ns, expected_provider_fingerprint=expected_provider_fingerprint, expected_model_fingerprint=expected_model_fingerprint)


class RetrievalModelProvider(Protocol):
    # Providers accepted by the controller must preserve an enriched
    # deadline_ms value's originating absolute cutoff at every I/O boundary.
    absolute_deadline_aware: bool

    def capabilities(self, *, deadline_ms: int) -> CapabilitiesResponse: ...
    def embed_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> DenseResponse: ...
    def embed_query(self, item: ProviderInput, *, deadline_ms: int) -> DenseResponse: ...
    def sparse_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> SparseResponse: ...
    def sparse_query(self, item: ProviderInput, *, deadline_ms: int) -> SparseResponse: ...
    def rerank(self, query: ProviderInput, documents: Sequence[ProviderInput], *, deadline_ms: int) -> RerankResponse: ...


class DisabledProvider:
    """An explicit no-I/O provider used whenever semantic retrieval is absent."""
    call_count = 0
    absolute_deadline_aware = True

    @staticmethod
    def _call(operation: ProviderOperation, component: ModelComponent | None, status: ProviderCallStatus) -> ProviderCall:
        return ProviderCall(operation, component, status, None, None, 0, None)

    def capabilities(self, *, deadline_ms: int) -> CapabilitiesResponse:
        _integer(deadline_ms, "deadline_ms", minimum=1)
        disabled = lambda component: ComponentCapability(component, CapabilityState.DISABLED, None, None)
        capabilities = ProviderCapabilities(1, "disabled", "v1", "0" * 64, disabled(ModelComponent.DENSE), disabled(ModelComponent.SPARSE), disabled(ModelComponent.RERANKER))
        return CapabilitiesResponse(ProviderCall(ProviderOperation.CAPABILITIES, None, ProviderCallStatus.OK, capabilities.fingerprint, None, 0, None), capabilities)

    def embed_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> DenseResponse:
        _integer(deadline_ms, "deadline_ms", minimum=1)
        return DenseResponse(self._call(ProviderOperation.DENSE_DOCUMENTS, ModelComponent.DENSE, ProviderCallStatus.DISABLED), ())
    def embed_query(self, item: ProviderInput, *, deadline_ms: int) -> DenseResponse:
        _integer(deadline_ms, "deadline_ms", minimum=1)
        return DenseResponse(self._call(ProviderOperation.DENSE_QUERY, ModelComponent.DENSE, ProviderCallStatus.DISABLED), ())
    def sparse_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> SparseResponse:
        _integer(deadline_ms, "deadline_ms", minimum=1)
        return SparseResponse(self._call(ProviderOperation.SPARSE_DOCUMENTS, ModelComponent.SPARSE, ProviderCallStatus.DISABLED), ())
    def sparse_query(self, item: ProviderInput, *, deadline_ms: int) -> SparseResponse:
        _integer(deadline_ms, "deadline_ms", minimum=1)
        return SparseResponse(self._call(ProviderOperation.SPARSE_QUERY, ModelComponent.SPARSE, ProviderCallStatus.DISABLED), ())
    def rerank(self, query: ProviderInput, documents: Sequence[ProviderInput], *, deadline_ms: int) -> RerankResponse:
        _integer(deadline_ms, "deadline_ms", minimum=1)
        return RerankResponse(self._call(ProviderOperation.RERANK, ModelComponent.RERANKER, ProviderCallStatus.DISABLED), ())


def provider_from_config(config: object) -> RetrievalModelProvider:
    """Return an I/O provider only for the exact audited local profile."""
    # Imports are intentionally local: configuration owns path parsing, while
    # this protocol boundary owns the rule that unrecognized settings are
    # always no-I/O.
    from .config import SemanticConfig
    from .local_http_provider import LocalHttpProvider, is_exact_audited_config

    if not isinstance(config, SemanticConfig) or not is_exact_audited_config(config):
        return DisabledProvider()
    return LocalHttpProvider.from_config(config)
