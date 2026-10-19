"""Audited, loopback-only Ollama dense embedding adapter.

This module deliberately implements only the protocol evidence captured by the
P2.3 inventory.  It does not discover endpoints, forward credentials, follow
redirects, or infer sparse/reranker support from a model name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from http.client import HTTPConnection, HTTPResponse
import json
import math
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import SemanticConfig, _loopback_url
from .deadline import AbsoluteDeadlineMilliseconds, DeadlineBudget
from .provider import (
    CapabilityState,
    CapabilitiesResponse,
    ComponentCapability,
    DenseItem,
    DenseResponse,
    ModelComponent,
    ModelIdentity,
    ProviderCall,
    ProviderCallStatus,
    ProviderCapabilities,
    ProviderInput,
    ProviderOperation,
    RerankResponse,
    SparseResponse,
)


_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MODEL_ID = "bge-m3"
_MODEL_REVISION = "latest"
_MODEL_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
_OLLAMA_VERSION = "0.34.0"
_PROVIDER_TYPE = "ollama"
_PROVIDER_DIGEST = hashlib.sha256(b"ollama-audited-local-v1:0.34.0").hexdigest()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        return None


class _DeadlineExceeded(Exception):
    """Internal signal that a later HTTP phase has no deadline remaining."""


class _ReadPhaseTimedOut(Exception):
    """Internal signal for a timed-out send after connection completed."""


def _phase_timeout_ms(budget: DeadlineBudget, cap_ms: int) -> int | None:
    remaining_ms = budget.remaining_ms(now_ns=time.monotonic_ns)
    return min(remaining_ms, cap_ms) if remaining_ms >= 1 else None


class _BudgetedHTTPConnection(HTTPConnection):
    """Switch from a connect timeout to a resampled read-phase timeout."""

    def __init__(self, *args: Any, budget: DeadlineBudget, read_timeout_ms: int, **kwargs: Any) -> None:
        self._budget = budget
        self._read_timeout_ms = read_timeout_ms
        super().__init__(*args, **kwargs)
        self._connect_timeout_ms = max(1, int(float(self.timeout) * 1000))
        self.response_class = lambda sock, *response_args, **response_kwargs: _BudgetedHTTPResponse(
            sock, *response_args, budget=budget, read_timeout_ms=read_timeout_ms, **response_kwargs,
        )

    def _set_read_timeout(self) -> None:
        timeout_ms = _phase_timeout_ms(self._budget, self._read_timeout_ms)
        if timeout_ms is None:
            raise _DeadlineExceeded()
        self.timeout = timeout_ms / 1000
        if self.sock is None:
            raise OSError("HTTP connection did not expose a connected socket")
        self.sock.settimeout(self.timeout)

    def connect(self) -> None:
        timeout_ms = _phase_timeout_ms(
            self._budget, self._connect_timeout_ms
        )
        if timeout_ms is None:
            raise _DeadlineExceeded()
        self.timeout = timeout_ms / 1000
        super().connect()
        self._set_read_timeout()

    def send(self, data: Any) -> None:
        if _phase_timeout_ms(self._budget, self._read_timeout_ms) is None:
            raise _DeadlineExceeded()
        was_connected = self.sock is not None
        if was_connected:
            self._set_read_timeout()
        try:
            super().send(data)
        except TimeoutError as exc:
            if was_connected:
                raise _ReadPhaseTimedOut() from exc
            raise
        if self.sock is not None:
            self._set_read_timeout()


class _BudgetedHTTPResponse(HTTPResponse):
    """Resample the read phase without reaching through response internals."""

    def __init__(self, sock: Any, *args: Any, budget: DeadlineBudget, read_timeout_ms: int, **kwargs: Any) -> None:
        self._budget = budget
        self._read_timeout_ms = read_timeout_ms
        self._budget_socket = sock
        super().__init__(sock, *args, **kwargs)

    def _set_read_timeout(self) -> None:
        timeout_ms = _phase_timeout_ms(self._budget, self._read_timeout_ms)
        if timeout_ms is None:
            raise _DeadlineExceeded()
        self._budget_socket.settimeout(timeout_ms / 1000)

    def begin(self) -> None:
        self._set_read_timeout()
        super().begin()

    def read(self, amount: int | None = None) -> bytes:
        self._set_read_timeout()
        return super().read(amount)


class _BudgetedHTTPHandler(HTTPHandler):
    """Construct HTTP connections that keep one absolute deadline across phases."""

    def __init__(self, budget: DeadlineBudget, read_timeout_ms: int) -> None:
        super().__init__()
        self._budget = budget
        self._read_timeout_ms = read_timeout_ms

    def _connection(self, host: str, timeout: float, **kwargs: Any) -> _BudgetedHTTPConnection:
        return _BudgetedHTTPConnection(
            host, timeout=timeout, budget=self._budget,
            read_timeout_ms=self._read_timeout_ms, **kwargs,
        )

    def http_open(self, request: Request):  # type: ignore[no-untyped-def]
        return self.do_open(self._connection, request)


def _identity() -> ModelIdentity:
    return ModelIdentity(
        ModelComponent.DENSE, _MODEL_ID, _MODEL_REVISION, _MODEL_DIGEST, 1024,
        2048, None, "reject", "l2", "F16", (),
    )


def _capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(
        1, _PROVIDER_TYPE, _OLLAMA_VERSION, _PROVIDER_DIGEST,
        ComponentCapability(ModelComponent.DENSE, CapabilityState.AVAILABLE, _identity(), None),
        ComponentCapability(ModelComponent.SPARSE, CapabilityState.UNSUPPORTED, None, "unsupported"),
        ComponentCapability(ModelComponent.RERANKER, CapabilityState.UNSUPPORTED, None, "unsupported"),
    )


AUDITED_DENSE_FINGERPRINT = _identity().fingerprint
AUDITED_PROVIDER_FINGERPRINT = _capabilities().fingerprint


def audited_profile_metadata() -> dict[str, object]:
    """Return the non-secret identity pinned and verified by the handshake."""
    return {
        "provider_type": _PROVIDER_TYPE,
        "provider_version": _OLLAMA_VERSION,
        "provider_fingerprint": AUDITED_PROVIDER_FINGERPRINT,
        "dense": {
            "model_id": _MODEL_ID,
            "revision": _MODEL_REVISION,
            "digest": _MODEL_DIGEST,
            "fingerprint": AUDITED_DENSE_FINGERPRINT,
            "dimension": 1024,
            "max_input_tokens": 2048,
            "truncation": "reject",
            "normalization": "l2",
            "quantization": "F16",
        },
        "sparse": "UNSUPPORTED",
        "reranker": "UNSUPPORTED",
    }


def is_exact_audited_config(config: SemanticConfig) -> bool:
    """Return true only for the dense-only profile verified in Task 0."""
    return (
        isinstance(config, SemanticConfig)
        and config.provider == "local_http"
        and config.protocol == "audited-local-v1"
        and config.dense is True
        and config.sparse is False
        and config.reranker is False
        and config.expected_provider_fingerprint == AUDITED_PROVIDER_FINGERPRINT
        and config.expected_dense_fingerprint == AUDITED_DENSE_FINGERPRINT
        and config.expected_sparse_fingerprint is None
        and config.expected_reranker_fingerprint is None
    )


@dataclass(slots=True)
class LocalHttpProvider:
    """A small no-auth local HTTP provider for the audited dense profile."""

    base_url: str
    connect_timeout_ms: int = 500
    read_timeout_ms: int = 4_000
    expected_provider_fingerprint: str = AUDITED_PROVIDER_FINGERPRINT
    expected_dense_fingerprint: str = AUDITED_DENSE_FINGERPRINT
    absolute_deadline_aware = True

    def __post_init__(self) -> None:
        if _loopback_url(self.base_url) != self.base_url:
            raise ValueError("adapter base URL must be canonical numeric loopback http")
        for value in (self.connect_timeout_ms, self.read_timeout_ms):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("adapter timeouts must be positive integers")
        if self.expected_provider_fingerprint != AUDITED_PROVIDER_FINGERPRINT or self.expected_dense_fingerprint != AUDITED_DENSE_FINGERPRINT:
            raise ValueError("adapter fingerprints must be the audited profile")

    @classmethod
    def from_config(cls, config: SemanticConfig) -> "LocalHttpProvider":
        if not is_exact_audited_config(config):
            raise ValueError("configuration is not the audited dense-only profile")
        return cls(
            config.base_url, config.connect_timeout_ms, config.read_timeout_ms,
            config.expected_provider_fingerprint, config.expected_dense_fingerprint,
        )

    @classmethod
    def audited_for_tests(cls, base_url: str) -> "LocalHttpProvider":
        # This is deliberately test-only construction for in-process loopback fixtures.
        return cls(base_url)

    def capabilities(self, *, deadline_ms: int) -> CapabilitiesResponse:
        started = time.monotonic_ns()
        budget = self._budget(deadline_ms)
        if budget is None:
            return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None, ProviderCallStatus.DEADLINE_EXHAUSTED, started), None)
        return self._handshake(budget, started)

    def _handshake(self, budget: DeadlineBudget, started_ns: int) -> CapabilitiesResponse:
        status, version = self._version(budget)
        if status is not ProviderCallStatus.OK:
            return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None, status, started_ns), None)
        status, digest = self._model_digest(budget)
        if status is not ProviderCallStatus.OK:
            return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None, status, started_ns), None)
        capabilities = _capabilities()
        if version != _OLLAMA_VERSION or digest != _MODEL_DIGEST:
            return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None, ProviderCallStatus.FINGERPRINT_MISMATCH, started_ns), None)
        if capabilities.fingerprint != self.expected_provider_fingerprint or _identity().fingerprint != self.expected_dense_fingerprint:
            return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None, ProviderCallStatus.FINGERPRINT_MISMATCH, started_ns), None)
        return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None, ProviderCallStatus.OK, started_ns, verified=True), capabilities)

    def embed_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> DenseResponse:
        return self._embed(ProviderOperation.DENSE_DOCUMENTS, items, deadline_ms)

    def embed_query(self, item: ProviderInput, *, deadline_ms: int) -> DenseResponse:
        return self._embed(ProviderOperation.DENSE_QUERY, (item,), deadline_ms)

    def sparse_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> SparseResponse:
        return SparseResponse(self._call(ProviderOperation.SPARSE_DOCUMENTS, ModelComponent.SPARSE, ProviderCallStatus.UNSUPPORTED, time.monotonic_ns()), ())

    def sparse_query(self, item: ProviderInput, *, deadline_ms: int) -> SparseResponse:
        return SparseResponse(self._call(ProviderOperation.SPARSE_QUERY, ModelComponent.SPARSE, ProviderCallStatus.UNSUPPORTED, time.monotonic_ns()), ())

    def rerank(self, query: ProviderInput, documents: Sequence[ProviderInput], *, deadline_ms: int) -> RerankResponse:
        return RerankResponse(self._call(ProviderOperation.RERANK, ModelComponent.RERANKER, ProviderCallStatus.UNSUPPORTED, time.monotonic_ns()), ())

    def _embed(self, operation: ProviderOperation, items: Sequence[ProviderInput], deadline_ms: int) -> DenseResponse:
        started = time.monotonic_ns()
        budget = self._budget(deadline_ms)
        if budget is None:
            return DenseResponse(self._call(operation, ModelComponent.DENSE, ProviderCallStatus.DEADLINE_EXHAUSTED, started), ())
        if not self._valid_inputs(items):
            return DenseResponse(self._call(operation, ModelComponent.DENSE, ProviderCallStatus.INVALID_RESPONSE, started), ())
        handshake = self._handshake(budget, started)
        if handshake.call.status is not ProviderCallStatus.OK:
            return DenseResponse(self._call(operation, ModelComponent.DENSE, handshake.call.status, started), ())
        status, payload = self._request("POST", "/api/embed", {"input": [item.text for item in items], "model": _MODEL_ID, "truncate": False}, budget)
        if status is not ProviderCallStatus.OK:
            return DenseResponse(self._call(operation, ModelComponent.DENSE, status, started, verified=True), ())
        vectors = self._parse_embeddings(payload, len(items))
        if vectors is None:
            return DenseResponse(self._call(operation, ModelComponent.DENSE, ProviderCallStatus.INVALID_RESPONSE, started, verified=True), ())
        return DenseResponse(
            self._call(operation, ModelComponent.DENSE, ProviderCallStatus.OK, started, verified=True),
            tuple(DenseItem(item.item_id, item.position, vector, False) for item, vector in zip(items, vectors, strict=True)),
        )

    @staticmethod
    def _valid_inputs(items: Sequence[ProviderInput]) -> bool:
        return isinstance(items, Sequence) and bool(items) and all(isinstance(item, ProviderInput) for item in items) and tuple(item.position for item in items) == tuple(range(len(items)))

    def _version(self, budget: DeadlineBudget) -> tuple[ProviderCallStatus, str | None]:
        status, payload = self._request("GET", "/api/version", None, budget)
        if status is not ProviderCallStatus.OK:
            return status, None
        if not isinstance(payload, dict) or set(payload) != {"version"} or not isinstance(payload["version"], str):
            return ProviderCallStatus.INVALID_RESPONSE, None
        return ProviderCallStatus.OK, payload["version"]

    def _model_digest(self, budget: DeadlineBudget) -> tuple[ProviderCallStatus, str | None]:
        status, payload = self._request("GET", "/api/tags", None, budget)
        if status is not ProviderCallStatus.OK:
            return status, None
        if not isinstance(payload, dict) or set(payload) != {"models"} or not isinstance(payload["models"], list):
            return ProviderCallStatus.INVALID_RESPONSE, None
        matching = [entry for entry in payload["models"] if isinstance(entry, dict) and entry.get("name") == "bge-m3:latest"]
        if len(matching) != 1 or not isinstance(matching[0].get("digest"), str):
            return ProviderCallStatus.INVALID_RESPONSE, None
        return ProviderCallStatus.OK, matching[0]["digest"]

    @staticmethod
    def _parse_embeddings(payload: Any, count: int) -> tuple[tuple[float, ...], ...] | None:
        if not isinstance(payload, dict) or set(payload) != {"model", "embeddings", "total_duration", "load_duration", "prompt_eval_count"}:
            return None
        if payload["model"] not in {_MODEL_ID, f"{_MODEL_ID}:{_MODEL_REVISION}"}:
            return None
        if any(isinstance(payload[name], bool) or not isinstance(payload[name], int) or payload[name] < 0 for name in ("total_duration", "load_duration", "prompt_eval_count")):
            return None
        if payload["prompt_eval_count"] > 2048:
            return None
        raw = payload["embeddings"]
        if not isinstance(raw, list) or len(raw) != count:
            return None
        parsed: list[tuple[float, ...]] = []
        for vector in raw:
            if not isinstance(vector, list) or len(vector) != 1024:
                return None
            values = tuple(float(value) for value in vector if not isinstance(value, bool) and isinstance(value, (int, float)))
            if len(values) != len(vector) or not all(math.isfinite(value) for value in values):
                return None
            if math.fsum(value * value for value in values) <= 0:
                return None
            parsed.append(values)
        return tuple(parsed)

    @staticmethod
    def _budget(deadline_ms: int) -> DeadlineBudget | None:
        if isinstance(deadline_ms, AbsoluteDeadlineMilliseconds):
            return deadline_ms.absolute_deadline
        if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, int) or deadline_ms < 1:
            return None
        return DeadlineBudget.start(deadline_ms, now_ns=time.monotonic_ns)

    def _request(self, method: str, path: str, payload: dict[str, object] | None, budget: DeadlineBudget) -> tuple[ProviderCallStatus, Any | None]:
        connect_timeout_ms = _phase_timeout_ms(budget, self.connect_timeout_ms)
        if connect_timeout_ms is None:
            return ProviderCallStatus.DEADLINE_EXHAUSTED, None
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if body is not None and len(body) > _MAX_RESPONSE_BYTES:
            return ProviderCallStatus.INPUT_TOO_LONG, None
        request = Request(self.base_url + path, data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        opener = build_opener(
            ProxyHandler({}), _NoRedirect(), _BudgetedHTTPHandler(budget, self.read_timeout_ms),
        )
        try:
            with opener.open(request, timeout=connect_timeout_ms / 1000) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    return ProviderCallStatus.INVALID_RESPONSE, None
                declared = response.headers.get("Content-Length")
                if declared is not None and (not declared.isdecimal() or int(declared) > _MAX_RESPONSE_BYTES):
                    return ProviderCallStatus.INVALID_RESPONSE, None
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return ProviderCallStatus.INVALID_RESPONSE, None
                if _phase_timeout_ms(budget, self.read_timeout_ms) is None:
                    return ProviderCallStatus.DEADLINE_EXHAUSTED, None
        except _DeadlineExceeded:
            return ProviderCallStatus.DEADLINE_EXHAUSTED, None
        except _ReadPhaseTimedOut:
            return ProviderCallStatus.TIMEOUT, None
        except HTTPError as exc:
            if exc.code == 404:
                return ProviderCallStatus.UNSUPPORTED, None
            if exc.code in {429, 503}:
                return ProviderCallStatus.BUSY, None
            if exc.code == 400:
                return ProviderCallStatus.INPUT_TOO_LONG, None
            return ProviderCallStatus.UNAVAILABLE, None
        except TimeoutError:
            return ProviderCallStatus.TIMEOUT, None
        except URLError:
            return ProviderCallStatus.UNAVAILABLE, None
        except (OSError, ValueError):
            return ProviderCallStatus.UNAVAILABLE, None
        try:
            return ProviderCallStatus.OK, json.loads(raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return ProviderCallStatus.INVALID_RESPONSE, None

    def _call(self, operation: ProviderOperation, component: ModelComponent | None, status: ProviderCallStatus, started_ns: int, *, verified: bool = False) -> ProviderCall:
        elapsed_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        return ProviderCall(
            operation, component, status, AUDITED_PROVIDER_FINGERPRINT if verified else None,
            AUDITED_DENSE_FINGERPRINT if verified and component is ModelComponent.DENSE else None,
            elapsed_ms, None if status is ProviderCallStatus.OK else status.value,
        )
