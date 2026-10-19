from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.errors import RetrievalValidationError
from codex_memory.retrieval.provider import (
    CapabilityState,
    ComponentCapability,
    DenseItem,
    DenseResponse,
    DisabledProvider,
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
    checked_dense_response,
    checked_rerank_response,
    checked_sparse_response,
)
from retrieval_provider_fakes import FakeProvider


def _input(item_id: str, position: int) -> ProviderInput:
    text = f"input {item_id}"
    return ProviderInput.from_text(item_id, position, text)


def _identity(component: ModelComponent, dimension: int | None = 4) -> ModelIdentity:
    return ModelIdentity(
        component, "test-model", "r1", "a" * 64, dimension, 128, 4096,
        "reject", "l2", "none", ("adapter-a",),
    )


def _call(operation: ProviderOperation, component: ModelComponent | None, status: ProviderCallStatus = ProviderCallStatus.OK) -> ProviderCall:
    return ProviderCall(operation, component, status, "b" * 64, None if operation is ProviderOperation.CAPABILITIES else "c" * 64, 0, None)


def test_provider_enums_and_canonical_capability_round_trip_are_frozen() -> None:
    assert [value.value for value in ModelComponent] == ["dense", "sparse", "reranker"]
    assert [value.value for value in ProviderOperation] == [
        "capabilities", "dense_documents", "dense_query", "sparse_documents", "sparse_query", "rerank",
    ]
    assert [value.value for value in CapabilityState] == ["disabled", "available", "unsupported", "unavailable"]
    assert [value.value for value in ProviderCallStatus] == [
        "ok", "disabled", "unsupported", "unavailable", "busy", "timeout", "input_too_long",
        "invalid_response", "fingerprint_mismatch", "circuit_open", "deadline_exhausted",
    ]
    dense = ComponentCapability(ModelComponent.DENSE, CapabilityState.AVAILABLE, _identity(ModelComponent.DENSE), None)
    sparse = ComponentCapability(ModelComponent.SPARSE, CapabilityState.UNSUPPORTED, None, "not audited")
    reranker = ComponentCapability(ModelComponent.RERANKER, CapabilityState.DISABLED, None, None)
    capability = ProviderCapabilities(1, "fake", "1", "d" * 64, dense, sparse, reranker)

    payload = capability.to_dict()
    assert ProviderCapabilities.from_dict(payload) == capability
    assert len(capability.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        capability.provider_type = "other"  # type: ignore[misc]
    with pytest.raises(RetrievalValidationError):
        ProviderCapabilities.from_dict({**payload, "extra": True})


def test_call_operation_component_pairs_are_exact() -> None:
    assert _call(ProviderOperation.CAPABILITIES, None).to_dict()["component"] is None
    with pytest.raises(RetrievalValidationError):
        _call(ProviderOperation.CAPABILITIES, ModelComponent.DENSE)
    with pytest.raises(RetrievalValidationError):
        _call(ProviderOperation.DENSE_QUERY, ModelComponent.SPARSE)
    with pytest.raises(RetrievalValidationError):
        _call(ProviderOperation.RERANK, None)


def test_capability_call_cannot_claim_a_model_fingerprint_and_call_reason_is_safe_code() -> None:
    with pytest.raises(RetrievalValidationError):
        ProviderCall(ProviderOperation.CAPABILITIES, None, ProviderCallStatus.OK, "b" * 64, "c" * 64, 0, None)

    payload = _call(ProviderOperation.CAPABILITIES, None).to_dict()
    payload["model_fingerprint"] = "c" * 64
    with pytest.raises(RetrievalValidationError):
        ProviderCall.from_dict(payload)

    safe = ProviderCall(ProviderOperation.DENSE_QUERY, ModelComponent.DENSE, ProviderCallStatus.INVALID_RESPONSE, "b" * 64, "c" * 64, 0, "invalid_response")
    assert safe.to_dict()["reason"] == "invalid_response"
    for unsafe in (('h' + 'ttp://user:secret@127.0.0.1'), "Bearer token", "bad\nreason", "x" * 1025):
        with pytest.raises(RetrievalValidationError):
            ProviderCall(ProviderOperation.DENSE_QUERY, ModelComponent.DENSE, ProviderCallStatus.INVALID_RESPONSE, "b" * 64, "c" * 64, 0, unsafe)


def test_provider_inputs_and_items_reject_invalid_identifiers_hashes_and_numbers() -> None:
    item = _input("a", 0)
    assert item.text_sha256 == ProviderInput.sha256_text(item.text)
    with pytest.raises(RetrievalValidationError):
        ProviderInput("a", 1, "text", "0" * 64)
    for values in ((float("nan"), 1.0), (float("inf"), 1.0), (0.0, 0.0)):
        with pytest.raises(RetrievalValidationError):
            DenseItem("a", 0, values, False)
    with pytest.raises(RetrievalValidationError):
        SparseItem("a", 0, (("term", -1.0),), False)
    with pytest.raises(RetrievalValidationError):
        SparseItem("a", 0, tuple((str(index), 1.0) for index in range(4097)), False)
    with pytest.raises(RetrievalValidationError):
        RerankItem("a", 0, math.nan, False)


def test_checked_responses_reject_wrong_mapping_dimensions_truncation_and_expiry() -> None:
    requested = (_input("a", 0), _input("b", 1))
    valid = DenseResponse(_call(ProviderOperation.DENSE_DOCUMENTS, ModelComponent.DENSE), (
        DenseItem("a", 0, (1.0, 0.0, 0.0, 0.0), False), DenseItem("b", 1, (0.0, 1.0, 0.0, 0.0), False),
    ))
    assert checked_dense_response(valid, requested, dimension=4).call.status is ProviderCallStatus.OK
    wrong_order = DenseResponse(valid.call, tuple(reversed(valid.items)))
    assert checked_dense_response(wrong_order, requested, dimension=4).call.status is ProviderCallStatus.INVALID_RESPONSE
    wrong_id = DenseResponse(valid.call, (DenseItem("wrong", 0, (1.0, 0.0, 0.0, 0.0), False), valid.items[1]))
    assert checked_dense_response(wrong_id, requested, dimension=4).call.status is ProviderCallStatus.INVALID_RESPONSE
    wrong_dimension = DenseResponse(valid.call, (DenseItem("a", 0, (1.0, 0.0), False), DenseItem("b", 1, (0.0, 1.0), False)))
    assert checked_dense_response(wrong_dimension, requested, dimension=4).call.status is ProviderCallStatus.INVALID_RESPONSE
    truncated = DenseResponse(valid.call, (DenseItem("a", 0, (1.0, 0.0, 0.0, 0.0), True), DenseItem("b", 1, (0.0, 1.0, 0.0, 0.0), False)))
    assert checked_dense_response(truncated, requested, dimension=4).call.status is ProviderCallStatus.INPUT_TOO_LONG
    expired = DeadlineBudget.start(1, now_ns=lambda: 0)
    assert checked_dense_response(valid, requested, dimension=4, deadline=expired, now_ns=lambda: 1_000_000).call.status is ProviderCallStatus.TIMEOUT


@pytest.mark.parametrize("requested", [(_input("a", 1),), (_input("a", 0), _input("b", 2))])
def test_checked_responses_require_contiguous_zero_based_request_positions(requested: tuple[ProviderInput, ...]) -> None:
    response = DenseResponse(_call(ProviderOperation.DENSE_DOCUMENTS, ModelComponent.DENSE), tuple(
        DenseItem(item.item_id, item.position, (1.0, 0.0, 0.0, 0.0), False) for item in requested
    ))
    assert checked_dense_response(response, requested, dimension=4).call.status is ProviderCallStatus.INVALID_RESPONSE


def test_checked_sparse_and_rerank_responses_are_one_to_one() -> None:
    requested = (_input("a", 0), _input("b", 1))
    sparse = SparseResponse(_call(ProviderOperation.SPARSE_DOCUMENTS, ModelComponent.SPARSE), (
        SparseItem("a", 0, (("a", 1.0),), False), SparseItem("b", 1, (("b", 1.0),), False),
    ))
    assert checked_sparse_response(sparse, requested).call.status is ProviderCallStatus.OK
    assert checked_sparse_response(SparseResponse(sparse.call, sparse.items[:1]), requested).call.status is ProviderCallStatus.INVALID_RESPONSE
    assert checked_sparse_response(SparseResponse(sparse.call, (SparseItem("wrong", 0, (("a", 1.0),), False), sparse.items[1])), requested).call.status is ProviderCallStatus.INVALID_RESPONSE
    rerank = RerankResponse(_call(ProviderOperation.RERANK, ModelComponent.RERANKER), (
        RerankItem("a", 0, 0.2, False), RerankItem("b", 1, 0.1, False),
    ))
    assert checked_rerank_response(rerank, requested).call.status is ProviderCallStatus.OK
    assert checked_rerank_response(RerankResponse(rerank.call, rerank.items[:1]), requested).call.status is ProviderCallStatus.INVALID_RESPONSE
    assert checked_rerank_response(RerankResponse(rerank.call, tuple(reversed(rerank.items))), requested).call.status is ProviderCallStatus.INVALID_RESPONSE

    non_contiguous = (_input("a", 1),)
    assert checked_sparse_response(SparseResponse(sparse.call, (SparseItem("a", 1, (("a", 1.0),), False),)), non_contiguous).call.status is ProviderCallStatus.INVALID_RESPONSE
    assert checked_rerank_response(RerankResponse(rerank.call, (RerankItem("a", 1, 0.2, False),)), non_contiguous).call.status is ProviderCallStatus.INVALID_RESPONSE


def test_disabled_provider_makes_zero_io_and_fake_is_deterministic_framework_only(monkeypatch) -> None:
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    disabled = DisabledProvider()
    capabilities = disabled.capabilities(deadline_ms=100)
    assert capabilities.call.status is ProviderCallStatus.OK
    assert [component.state for component in (capabilities.capabilities.dense, capabilities.capabilities.sparse, capabilities.capabilities.reranker)] == [CapabilityState.DISABLED] * 3
    assert disabled.embed_documents((_input("a", 0),), deadline_ms=100).items == ()
    assert disabled.call_count == 0

    fake = FakeProvider()
    first = fake.embed_documents((_input("a", 0),), deadline_ms=100)
    second = fake.embed_documents((_input("a", 0),), deadline_ms=100)
    assert first == second
    assert fake.evidence_label == "framework_fake"
