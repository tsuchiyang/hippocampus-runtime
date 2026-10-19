"""Test-only deterministic provider fixture; never import from production code."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from codex_memory.retrieval.provider import (
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
    RerankItem,
    RerankResponse,
    SparseItem,
    SparseResponse,
)


class FakeProvider:
    """Deterministic framework fixture, not evidence of a live provider."""

    absolute_deadline_aware = True
    evidence_label = "framework_fake"
    provider_digest = "f" * 64

    def __init__(self, statuses: dict[ModelComponent, ProviderCallStatus] | None = None) -> None:
        self.statuses = dict(statuses or {})
        self.call_counts: dict[ProviderOperation, int] = {operation: 0 for operation in ProviderOperation}
        # Retain only already-hashed inputs.  R3 needs to prove that the
        # benchmark changes what flows through the formal provider boundary
        # without putting fixture text into evidence or reports.
        self.call_inputs: dict[ProviderOperation, list[tuple[str, ...]]] = {
            operation: [] for operation in ProviderOperation
        }
        self._capabilities = ProviderCapabilities(
            1, "fake", "v1", self.provider_digest,
            self._component_capability(ModelComponent.DENSE, 4),
            self._component_capability(ModelComponent.SPARSE, None),
            self._component_capability(ModelComponent.RERANKER, None),
        )

    @staticmethod
    def _identity(component: ModelComponent, dimension: int | None) -> ModelIdentity:
        return ModelIdentity(component, f"fake-{component.value}", "r1", hashlib.sha256(component.value.encode()).hexdigest(), dimension, 128, 4096, "reject", "l2", "none", ())

    @classmethod
    def component_fingerprint(cls, component: ModelComponent) -> str:
        """Return the fixed fixture identity without making a provider call."""
        return cls._identity(
            component,
            4 if component is ModelComponent.DENSE else None,
        ).fingerprint

    @property
    def expected_provider_fingerprint(self) -> str:
        """Return the fixed capabilities identity without a provider request."""
        return self._capabilities.fingerprint

    def _component_capability(self, component: ModelComponent, dimension: int | None) -> ComponentCapability:
        status = self.statuses.get(component, ProviderCallStatus.OK)
        if status is ProviderCallStatus.OK:
            return ComponentCapability(component, CapabilityState.AVAILABLE, self._identity(component, dimension), None)
        state = {
            ProviderCallStatus.DISABLED: CapabilityState.DISABLED,
            ProviderCallStatus.UNSUPPORTED: CapabilityState.UNSUPPORTED,
        }.get(status, CapabilityState.UNAVAILABLE)
        return ComponentCapability(component, state, None, status.value)

    def _call(
        self,
        operation: ProviderOperation,
        component: ModelComponent | None,
        input_hashes: tuple[str, ...] = (),
    ) -> ProviderCall:
        self.call_counts[operation] += 1
        self.call_inputs[operation].append(input_hashes)
        status = ProviderCallStatus.OK if component is None else self.statuses.get(component, ProviderCallStatus.OK)
        model_fingerprint = None
        if component is not None:
            identity = getattr(self._capabilities, component.value).identity
            model_fingerprint = identity.fingerprint if identity is not None else None
        return ProviderCall(operation, component, status, self._capabilities.fingerprint, model_fingerprint, 0, None)

    def capabilities(self, *, deadline_ms: int) -> CapabilitiesResponse:
        return CapabilitiesResponse(self._call(ProviderOperation.CAPABILITIES, None), self._capabilities)

    @staticmethod
    def _vector(item: ProviderInput) -> tuple[float, ...]:
        digest = hashlib.sha256(f"{item.item_id}:{item.text_sha256}".encode()).digest()
        return tuple(float(digest[index] + 1) for index in range(4))

    @staticmethod
    def _weights(item: ProviderInput) -> tuple[tuple[str, float], ...]:
        terms = sorted(set(re.findall(r"[A-Za-z0-9_]+", item.text.casefold())))
        return tuple((term, float(index + 1)) for index, term in enumerate(terms))

    def embed_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> DenseResponse:
        call = self._call(
            ProviderOperation.DENSE_DOCUMENTS,
            ModelComponent.DENSE,
            tuple(item.text_sha256 for item in items),
        )
        return DenseResponse(call, () if call.status is not ProviderCallStatus.OK else tuple(DenseItem(item.item_id, item.position, self._vector(item), False) for item in items))

    def embed_query(self, item: ProviderInput, *, deadline_ms: int) -> DenseResponse:
        call = self._call(
            ProviderOperation.DENSE_QUERY,
            ModelComponent.DENSE,
            (item.text_sha256,),
        )
        return DenseResponse(call, () if call.status is not ProviderCallStatus.OK else (DenseItem(item.item_id, item.position, self._vector(item), False),))

    def sparse_documents(self, items: Sequence[ProviderInput], *, deadline_ms: int) -> SparseResponse:
        call = self._call(
            ProviderOperation.SPARSE_DOCUMENTS,
            ModelComponent.SPARSE,
            tuple(item.text_sha256 for item in items),
        )
        return SparseResponse(call, () if call.status is not ProviderCallStatus.OK else tuple(SparseItem(item.item_id, item.position, self._weights(item), False) for item in items))

    def sparse_query(self, item: ProviderInput, *, deadline_ms: int) -> SparseResponse:
        call = self._call(
            ProviderOperation.SPARSE_QUERY,
            ModelComponent.SPARSE,
            (item.text_sha256,),
        )
        return SparseResponse(call, () if call.status is not ProviderCallStatus.OK else (SparseItem(item.item_id, item.position, self._weights(item), False),))

    def rerank(self, query: ProviderInput, documents: Sequence[ProviderInput], *, deadline_ms: int) -> RerankResponse:
        call = self._call(
            ProviderOperation.RERANK,
            ModelComponent.RERANKER,
            (query.text_sha256, *(item.text_sha256 for item in documents)),
        )
        if call.status is not ProviderCallStatus.OK:
            return RerankResponse(call, ())
        return RerankResponse(call, tuple(
            RerankItem(document.item_id, document.position, int.from_bytes(hashlib.sha256(f"{query.text_sha256}:{document.text_sha256}".encode()).digest()[:8], "big") / (2**64 - 1), False)
            for document in documents
        ))


__all__ = ["FakeProvider"]
