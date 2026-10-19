"""Deterministic literal and symbol candidate channels."""
from __future__ import annotations
from dataclasses import dataclass
import re
import sqlite3
import unicodedata

from codex_memory.memory_scope import ScopeContext
from .contracts import RetrievalRequest

_TOKEN = re.compile(r"[\w./:-]+|--[\w-]+|'[^']+'|\"[^\"]+\"", re.UNICODE)
_SYMBOL = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\Z")
MAX_QUERY_BYTES = 4096

@dataclass(frozen=True, slots=True)
class ChannelResult:
    ids: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()

def _scope(request: RetrievalRequest) -> ScopeContext:
    if not isinstance(request, RetrievalRequest): raise TypeError("request must be RetrievalRequest")
    return ScopeContext(request.scope.project_scope_key, request.scope.workspace_scope_keys)

def _visible(row: sqlite3.Row, scope: ScopeContext) -> bool:
    return row["state"] == "active" and scope.contains(scope_kind=row["scope_kind"], scope_key=row["scope_key"])

def _terms(query: str | None) -> tuple[str, ...]:
    if not query or len(query.encode("utf-8")) > MAX_QUERY_BYTES: return ()
    return tuple(dict.fromkeys(unicodedata.normalize("NFKC", value).casefold() for value in _TOKEN.findall(query) if value))

def exact_candidates(connection: sqlite3.Connection, request: RetrievalRequest, *, generation_id: str, limit: int = 32, excluded_ids: frozenset[str] = frozenset()) -> ChannelResult:
    terms = _terms(request.query); scope = _scope(request)
    if not terms: return ChannelResult(())
    rows = connection.execute("SELECT * FROM retrieval_documents WHERE generation_id = ?", (generation_id,)).fetchall()
    scored = []
    for row in rows:
        if not _visible(row, scope) or row["record_id"] in excluded_ids: continue
        values = (row["record_id"], row["content_sha256"], row["topic_key"], row["statement"], row["search_text"])
        haystack = unicodedata.normalize("NFKC", " ".join(values)).casefold()
        score = sum(term in haystack for term in terms)
        if score: scored.append((score, row["record_seq"], row["record_id"]))
    scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
    return ChannelResult(tuple(value[2] for value in scored[:limit]), ("exact_match",))

def symbol_candidates(connection: sqlite3.Connection, request: RetrievalRequest, *, generation_id: str, limit: int = 32, excluded_ids: frozenset[str] = frozenset()) -> ChannelResult:
    terms = tuple(term for term in _terms(request.query) if len(term.encode("utf-8")) <= 256 and _SYMBOL.fullmatch(term))
    if not terms: return ChannelResult(())
    scope = _scope(request); rows = connection.execute("""SELECT d.* FROM retrieval_symbols AS s
      JOIN retrieval_documents AS d ON d.document_id = s.document_id
      WHERE s.generation_id = ? AND s.normalized_symbol IN (%s)""" % ",".join("?" for _ in terms), (generation_id, *terms)).fetchall()
    selected = [row for row in rows if _visible(row, scope) and row["record_id"] not in excluded_ids]
    selected.sort(key=lambda row: (-row["record_seq"], row["record_id"]))
    return ChannelResult(tuple(row["record_id"] for row in selected[:limit]), ("symbol_match",))
