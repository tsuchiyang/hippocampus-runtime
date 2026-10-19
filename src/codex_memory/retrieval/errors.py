from __future__ import annotations


class RetrievalError(RuntimeError):
    """Base class for retrieval-domain failures."""


class RetrievalValidationError(RetrievalError, ValueError):
    """Caller input violates a frozen retrieval contract."""


class ProviderResponseError(RetrievalError):
    """An optional model provider returned an unusable response."""
