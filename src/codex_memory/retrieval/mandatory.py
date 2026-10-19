"""Certified Constitution and same-session direct temporary channels."""
from __future__ import annotations
import sqlite3
from codex_memory.blob_store import BlobStore
from codex_memory.memory_authority import select_authorized_items
from codex_memory.memory_scope import ScopeContext
from .contracts import RetrievalRequest
from .exact import ChannelResult

def _scope(request: RetrievalRequest) -> ScopeContext:
    if not isinstance(request, RetrievalRequest): raise TypeError("request must be RetrievalRequest")
    return ScopeContext(request.scope.project_scope_key, request.scope.workspace_scope_keys)

def mandatory_candidates(connection: sqlite3.Connection, blobs: BlobStore, request: RetrievalRequest, *, limit: int = 32) -> ChannelResult:
    items, _ = select_authorized_items(connection, blobs, session_id=request.session_id, scope_context=_scope(request), query=request.query)
    selected = [item for item in items if item.record_type == "constitution" and item.certified]
    selected.sort(key=lambda item: (-item.sequence, item.identifier))
    return ChannelResult(tuple(item.identifier for item in selected[:limit]), ("certified_constitution",))

def temporary_candidates(connection: sqlite3.Connection, blobs: BlobStore, request: RetrievalRequest, *, limit: int = 32) -> ChannelResult:
    items, _ = select_authorized_items(connection, blobs, session_id=request.session_id, scope_context=_scope(request), query=request.query)
    return ChannelResult(tuple(item.identifier for item in items if item.state == "temporary")[:limit], ("same_session_temporary",))
