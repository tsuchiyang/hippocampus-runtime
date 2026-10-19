from __future__ import annotations

from dataclasses import dataclass
import hashlib

from codex_memory.ledger_common import canonical_json


MAX_DOCUMENT_TEXT_BYTES = 65_536
MAX_EVIDENCE_PER_DOCUMENT = 256


class UnsafeProjectionContent(ValueError):
    """Canonical content that cannot safely become a derived document."""


@dataclass(frozen=True, slots=True)
class ProjectedDocument:
    document_id: str
    record_id: str
    record_seq: int
    record_type: str
    scope_kind: str
    scope_key: str
    topic_key: str
    statement: str
    value_json: str
    search_text: str
    state: str
    created_at: str
    last_memory_event_seq: int
    evidence_count: int
    content_sha256: str
    document_sha256: str
    safe_references: tuple[str, ...]
    relation_endpoints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectedSource:
    source_id: str
    document_id: str
    ordinal: int
    safe_reference: str
    source_sha256: str


def deterministic_document_id(record_id: str, content_sha256: str) -> str:
    return "doc_" + _sha256({"content_sha256": content_sha256, "record_id": record_id})[:32]


def build_document(
    *,
    record_id: str,
    record_seq: int,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    value_json: str,
    state: str,
    created_at: str,
    last_memory_event_seq: int,
    content_sha256: str,
    safe_references: tuple[str, ...],
    relation_endpoints: tuple[str, ...],
) -> ProjectedDocument:
    if len(safe_references) > MAX_EVIDENCE_PER_DOCUMENT:
        raise UnsafeProjectionContent("too many evidence references")
    if any(not reference or "\x00" in reference for reference in safe_references):
        raise UnsafeProjectionContent("unsafe evidence reference")
    search_text = " ".join(
        (
            record_type,
            scope_kind,
            scope_key,
            topic_key,
            statement,
            value_json,
            state,
            record_id,
            *relation_endpoints,
        )
    )
    if len(search_text.encode("utf-8")) > MAX_DOCUMENT_TEXT_BYTES:
        raise UnsafeProjectionContent("derived document exceeds content limit")
    document_id = deterministic_document_id(record_id, content_sha256)
    value = {
        "content_sha256": content_sha256,
        "created_at": created_at,
        "document_id": document_id,
        "evidence_count": len(safe_references),
        "last_memory_event_seq": last_memory_event_seq,
        "record_id": record_id,
        "record_seq": record_seq,
        "record_type": record_type,
        "relation_endpoints": relation_endpoints,
        "safe_references": safe_references,
        "scope_key": scope_key,
        "scope_kind": scope_kind,
        "search_text": search_text,
        "state": state,
        "statement": statement,
        "topic_key": topic_key,
        "value_json": value_json,
    }
    return ProjectedDocument(
        document_id=document_id,
        record_id=record_id,
        record_seq=record_seq,
        record_type=record_type,
        scope_kind=scope_kind,
        scope_key=scope_key,
        topic_key=topic_key,
        statement=statement,
        value_json=value_json,
        search_text=search_text,
        state=state,
        created_at=created_at,
        last_memory_event_seq=last_memory_event_seq,
        evidence_count=len(safe_references),
        content_sha256=content_sha256,
        document_sha256=_sha256(value),
        safe_references=safe_references,
        relation_endpoints=relation_endpoints,
    )


def document_sha256(document: ProjectedDocument) -> str:
    return _sha256(
        {
            "content_sha256": document.content_sha256,
            "created_at": document.created_at,
            "document_id": document.document_id,
            "evidence_count": document.evidence_count,
            "last_memory_event_seq": document.last_memory_event_seq,
            "record_id": document.record_id,
            "record_seq": document.record_seq,
            "record_type": document.record_type,
            "relation_endpoints": document.relation_endpoints,
            "safe_references": document.safe_references,
            "scope_key": document.scope_key,
            "scope_kind": document.scope_kind,
            "search_text": document.search_text,
            "state": document.state,
            "statement": document.statement,
            "topic_key": document.topic_key,
            "value_json": document.value_json,
        }
    )


def build_source(document_id: str, ordinal: int, safe_reference: str) -> ProjectedSource:
    source_sha256 = _sha256(
        {"document_id": document_id, "ordinal": ordinal, "safe_reference": safe_reference}
    )
    return ProjectedSource(
        source_id="src_" + source_sha256[:32],
        document_id=document_id,
        ordinal=ordinal,
        safe_reference=safe_reference,
        source_sha256=source_sha256,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
