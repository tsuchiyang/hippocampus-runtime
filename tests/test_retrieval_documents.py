from __future__ import annotations

import importlib


def test_document_builder_derives_stable_safe_metadata_and_hashes() -> None:
    documents = importlib.import_module("codex_memory.retrieval.documents")

    document = documents.build_document(
        record_id="mem-1",
        record_seq=7,
        record_type="decision",
        scope_kind="project",
        scope_key="C:/Example",
        topic_key="retrieval.snapshot",
        statement="Keep only verified rows.",
        value_json='{"enabled":true}',
        state="active",
        created_at="2026-09-05T00:00:00.000000Z",
        last_memory_event_seq=11,
        content_sha256="a" * 64,
        safe_references=("https:sha256:" + "b" * 64,),
        relation_endpoints=("mem-2",),
    )

    assert document.document_id == documents.deterministic_document_id("mem-1", "a" * 64)
    assert document.evidence_count == 1
    assert document.safe_references == ("https:sha256:" + "b" * 64,)
    assert document.relation_endpoints == ("mem-2",)
    assert document.search_text == (
        "decision project C:/Example retrieval.snapshot Keep only verified rows. "
        '{"enabled":true} active mem-1 mem-2'
    )
    assert len(document.document_sha256) == 64
    assert document.document_sha256 == documents.document_sha256(document)
