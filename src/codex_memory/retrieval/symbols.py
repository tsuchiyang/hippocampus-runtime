from __future__ import annotations

from dataclasses import dataclass
import hashlib

from codex_memory.ledger_common import canonical_json

from .documents import ProjectedDocument


MAX_SYMBOL_BYTES = 256
MAX_SYMBOLS_PER_DOCUMENT = 16


@dataclass(frozen=True, slots=True)
class ProjectedSymbol:
    symbol_id: str
    document_id: str
    ordinal: int
    symbol_kind: str
    normalized_symbol: str
    display_symbol: str
    symbol_sha256: str


def extract_symbols(document: ProjectedDocument) -> tuple[ProjectedSymbol, ...]:
    values = (
        ("record_id", document.record_id),
        ("topic_key", document.topic_key),
        *(("relation_endpoint", endpoint) for endpoint in document.relation_endpoints),
    )
    if len(values) > MAX_SYMBOLS_PER_DOCUMENT:
        raise ValueError("too many symbols")
    symbols: list[ProjectedSymbol] = []
    for ordinal, (kind, display) in enumerate(values):
        normalized = display.casefold()
        if not normalized or len(normalized.encode("utf-8")) > MAX_SYMBOL_BYTES:
            raise ValueError("symbol exceeds bounds")
        symbol_sha256 = hashlib.sha256(
            canonical_json(
                {
                    "document_id": document.document_id,
                    "display_symbol": display,
                    "normalized_symbol": normalized,
                    "ordinal": ordinal,
                    "symbol_kind": kind,
                }
            ).encode("utf-8")
        ).hexdigest()
        symbols.append(
            ProjectedSymbol(
                symbol_id="sym_" + symbol_sha256[:32],
                document_id=document.document_id,
                ordinal=ordinal,
                symbol_kind=kind,
                normalized_symbol=normalized,
                display_symbol=display,
                symbol_sha256=symbol_sha256,
            )
        )
    return tuple(symbols)
