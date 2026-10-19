from .contracts import (
    DegradedReason, EpistemicContext, IndexGeneration, IndexHighwater,
    RetrievalMode, RetrievalRequest, RetrievalResult, RetrievalScope,
    RetrievalStatus,
)
from .errors import ProviderResponseError, RetrievalError, RetrievalValidationError
from .deadline import DeadlineBudget
from .provider import (
    CapabilityState, CapabilitiesResponse, ComponentCapability, DenseItem,
    DenseResponse, DisabledProvider, ModelComponent, ModelIdentity,
    ProviderCall, ProviderCallStatus, ProviderCapabilities, ProviderInput,
    ProviderOperation, RerankItem, RerankResponse, RetrievalModelProvider,
    SparseItem, SparseResponse,
)

_ASSEMBLER_EXPORTS = frozenset({
    "AssembledPacket",
    "PacketAuditSnapshot",
    "PacketBudgetError",
    "PriorContextSelection",
    "SelectionIdentity",
    "build_phase2_packet",
})


def __getattr__(name: str):
    """Preserve packet exports without loading offline assembly on Hook import."""
    if name not in _ASSEMBLER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import assembler

    value = getattr(assembler, name)
    globals()[name] = value
    return value

__all__ = [
    "AssembledPacket", "CapabilityState", "CapabilitiesResponse", "ComponentCapability", "DeadlineBudget",
    "DegradedReason", "DenseItem", "DenseResponse", "DisabledProvider", "EpistemicContext",
    "IndexGeneration", "IndexHighwater", "ModelComponent", "ModelIdentity", "ProviderCall",
    "ProviderCallStatus", "ProviderCapabilities", "ProviderInput", "ProviderOperation", "ProviderResponseError",
    "RerankItem", "RerankResponse", "RetrievalError", "RetrievalMode", "RetrievalModelProvider",
    "RetrievalRequest", "RetrievalResult", "RetrievalScope", "RetrievalStatus",
    "PacketAuditSnapshot", "PacketBudgetError", "PriorContextSelection", "RetrievalValidationError",
    "SelectionIdentity", "SparseItem", "SparseResponse", "build_phase2_packet",
]
