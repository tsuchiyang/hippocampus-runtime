"""Safe FTS5-only BM25 channel."""
from __future__ import annotations
import re
import sqlite3
import unicodedata
from .contracts import RetrievalRequest
from .exact import ChannelResult, _scope, _visible

_WORD = re.compile(r"[\w-]+", re.UNICODE)
MAX_TOKEN_BYTES = 256
MAX_TOKENS = 32
MAX_QUERY_CHARS = 4096

def build_fts_match(query: str | None) -> str:
    if not query: return ""
    # `len` is O(1) for str.  Reject before normalization/regex work so an
    # untrusted request cannot force a whole-input tokenization allocation.
    if len(query) > MAX_QUERY_CHARS: return ""
    tokens = []
    for word in _WORD.findall(unicodedata.normalize("NFKC", query).casefold()):
        value = word.encode("utf-8")[:MAX_TOKEN_BYTES].decode("utf-8", "ignore")
        if value and value not in tokens: tokens.append(value)
        if len(tokens) == MAX_TOKENS: break
    return " ".join('"' + value.replace('"', '""') + '"' for value in tokens)

def bm25_candidates(connection: sqlite3.Connection, request: RetrievalRequest, *, generation_id: str, limit: int = 32, excluded_ids: frozenset[str] = frozenset()) -> ChannelResult:
    if request.query is not None and len(request.query) > MAX_QUERY_CHARS:
        return ChannelResult((), ("query_over_budget",))
    match = build_fts_match(request.query)
    if not match: return ChannelResult(())
    present = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'").fetchone()
    if present is None: return ChannelResult((), ("fts5_unavailable",))
    try:
        rows = connection.execute("""SELECT d.*, bm25(retrieval_documents_fts) AS rank FROM retrieval_documents_fts AS f
            JOIN retrieval_documents AS d ON d.document_id = f.document_id
            WHERE retrieval_documents_fts MATCH ? AND f.generation_id = ?""", (match, generation_id)).fetchall()
    except sqlite3.Error:
        return ChannelResult((), ("fts5_unavailable",))
    scope = _scope(request); visible = [row for row in rows if _visible(row, scope) and row["record_id"] not in excluded_ids]
    visible.sort(key=lambda row: (row["rank"], -row["record_seq"], row["record_id"]))
    return ChannelResult(tuple(row["record_id"] for row in visible[:limit]), ("bm25",))
