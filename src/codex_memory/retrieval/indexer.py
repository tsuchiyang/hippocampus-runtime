from __future__ import annotations

import time
from typing import Callable

from .contracts import RetrievalResult
from .config import SemanticConfig
from .paths import RetrievalPaths
from .projector import changed_record_ids
from .provider import RetrievalModelProvider


def rebuild_index(
    paths: RetrievalPaths,
    *,
    failure_injector: Callable[[str], None] | None = None,
) -> RetrievalResult:
    """Build one disposable retrieval generation from canonical memory."""
    from .service import RetrievalService

    return RetrievalService(paths).rebuild(failure_injector=failure_injector)


def catch_up_index(
    paths: RetrievalPaths,
    *,
    failure_injector: Callable[[str], None] | None = None,
) -> RetrievalResult:
    """Advance a stale disposable index through one immutable generation."""
    from .service import RetrievalService

    return RetrievalService(paths).catch_up(failure_injector=failure_injector)


def rebuild_semantic_index(
    paths: RetrievalPaths,
    provider: RetrievalModelProvider,
    config: SemanticConfig,
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RetrievalResult:
    """Public semantic build facade; provider selection stays outside this layer."""
    from .semantic_index import rebuild_semantic_index as build

    return build(paths, provider, config, clock=clock)


def catch_up_semantic_index(
    paths: RetrievalPaths,
    provider: RetrievalModelProvider,
    config: SemanticConfig,
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RetrievalResult:
    from .semantic_index import catch_up_semantic_index as build

    return build(paths, provider, config, clock=clock)
