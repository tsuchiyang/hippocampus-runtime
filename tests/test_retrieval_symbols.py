from __future__ import annotations

import importlib


def test_symbols_are_bounded_deterministic_and_include_relation_endpoints() -> None:
    documents = importlib.import_module("codex_memory.retrieval.documents")
    symbols = importlib.import_module("codex_memory.retrieval.symbols")
    document = documents.build_document(
        record_id="mem-1",
        record_seq=1,
        record_type="decision",
        scope_kind="project",
        scope_key="C:/Repo",
        topic_key="memory.retrieval",
        statement="Use deterministic symbols.",
        value_json="null",
        state="active",
        created_at="2026-09-05T00:00:00.000000Z",
        last_memory_event_seq=1,
        content_sha256="a" * 64,
        safe_references=(),
        relation_endpoints=("mem-2",),
    )

    first = symbols.extract_symbols(document)
    second = symbols.extract_symbols(document)

    assert first == second
    assert [(symbol.symbol_kind, symbol.display_symbol) for symbol in first] == [
        ("record_id", "mem-1"),
        ("topic_key", "memory.retrieval"),
        ("relation_endpoint", "mem-2"),
    ]
    assert all(len(symbol.normalized_symbol.encode("utf-8")) <= symbols.MAX_SYMBOL_BYTES for symbol in first)
    assert all(len(symbol.symbol_sha256) == 64 for symbol in first)
