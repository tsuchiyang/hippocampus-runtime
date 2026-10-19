from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3
import tempfile
from typing import Callable

from codex_memory.ledger_common import ValidationError, canonical_json
from codex_memory.memory_evidence import safe_evidence_reference

from .canonical_snapshot import CanonicalSnapshot
from .documents import (
    MAX_EVIDENCE_PER_DOCUMENT,
    ProjectedDocument,
    ProjectedSource,
    UnsafeProjectionContent,
    build_document,
    build_source,
)
from .symbols import MAX_SYMBOLS_PER_DOCUMENT, ProjectedSymbol, extract_symbols


@dataclass(frozen=True, slots=True)
class ProjectionBatch:
    documents: tuple[ProjectedDocument, ...]
    sources: tuple[ProjectedSource, ...]
    symbols: tuple[ProjectedSymbol, ...]
    omitted_unsafe_count: int
    manifest_sha256: str
    index_policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class StaleDelta:
    """A bounded, lifecycle-safe view of canonical changes after an index point."""

    record_ids: tuple[str, ...]
    overflowed: bool


@dataclass(frozen=True, slots=True)
class ProjectionDigest:
    """Fixed-memory counts and exact ordered digests for one full projection."""

    document_count: int
    source_count: int
    symbol_count: int
    fts_count: int
    omitted_unsafe_count: int
    manifest_sha256: str
    index_policy_fingerprint: str
    document_rows_sha256: str
    source_rows_sha256: str
    symbol_rows_sha256: str
    fts_rows_sha256: str


class ProjectionDigestBuilder:
    """Accumulate exact projection semantics without retaining projected rows."""

    def __init__(self, snapshot: CanonicalSnapshot, *, omitted_unsafe_count: int = 0) -> None:
        self.snapshot = snapshot
        self.document_count = 0
        self.source_count = 0
        self.symbol_count = 0
        self.fts_count = 0
        self.omitted_unsafe_count = omitted_unsafe_count
        self._document_manifest = _ManifestListSpool()
        self._source_manifest = _ManifestListSpool()
        self._symbol_manifest = _ManifestListSpool()
        self._document_rows = hashlib.sha256()
        self._source_rows = hashlib.sha256()
        self._symbol_rows = hashlib.sha256()
        self._fts_rows = hashlib.sha256()
        self._closed = False

    def add_document(self, document: ProjectedDocument, *, include_fts: bool = True) -> None:
        self.add_document_row(
            document.document_sha256,
            (
                document.document_id, document.record_id, document.record_seq,
                document.record_type, document.scope_kind, document.scope_key,
                document.topic_key, document.statement, document.value_json,
                document.search_text, document.state, document.created_at,
                document.last_memory_event_seq, document.evidence_count,
                document.content_sha256, document.document_sha256,
            ),
            include_fts=include_fts,
        )

    def add_document_row(
        self, document_sha256: str, values: tuple[object, ...], *, include_fts: bool
    ) -> None:
        self.document_count += 1
        self._document_manifest.add(document_sha256)
        _update_row_digest(self._document_rows, values)
        if include_fts:
            self.add_fts_row((values[0], values[9]))

    def add_source(self, source: ProjectedSource) -> None:
        self.add_source_row(
            source.source_sha256,
            (source.source_id, source.document_id, source.ordinal,
             source.safe_reference, source.source_sha256),
        )

    def add_source_row(self, source_sha256: str, values: tuple[object, ...]) -> None:
        self.source_count += 1
        self._source_manifest.add(source_sha256)
        _update_row_digest(self._source_rows, values)

    def add_symbol(self, symbol: ProjectedSymbol) -> None:
        self.add_symbol_row(
            symbol.symbol_sha256,
            (
                symbol.symbol_id, symbol.document_id, symbol.ordinal,
                symbol.symbol_kind, symbol.normalized_symbol,
                symbol.display_symbol, symbol.symbol_sha256,
            ),
        )

    def add_symbol_row(self, symbol_sha256: str, values: tuple[object, ...]) -> None:
        self.symbol_count += 1
        self._symbol_manifest.add(symbol_sha256)
        _update_row_digest(self._symbol_rows, values)

    def add_fts_row(self, values: tuple[object, ...]) -> None:
        self.fts_count += 1
        _update_row_digest(self._fts_rows, values)

    def omit_unsafe(self) -> None:
        self.omitted_unsafe_count += 1

    def finish(self) -> ProjectionDigest:
        if self._closed:
            raise RuntimeError("projection digest builder is closed")
        try:
            manifest = hashlib.sha256()
            manifest.update(b'{"documents":[')
            self._document_manifest.update_digest(manifest)
            manifest.update(b'],"highwater":')
            manifest.update(canonical_json(self.snapshot.highwater.to_dict()).encode("utf-8"))
            manifest.update(b',"omitted_unsafe_count":')
            manifest.update(str(self.omitted_unsafe_count).encode("ascii"))
            manifest.update(b',"sources":[')
            self._source_manifest.update_digest(manifest)
            manifest.update(b'],"symbols":[')
            self._symbol_manifest.update_digest(manifest)
            manifest.update(b']}')
            return ProjectionDigest(
                document_count=self.document_count,
                source_count=self.source_count,
                symbol_count=self.symbol_count,
                fts_count=self.fts_count,
                omitted_unsafe_count=self.omitted_unsafe_count,
                manifest_sha256=manifest.hexdigest(),
                index_policy_fingerprint=_index_policy_fingerprint(),
                document_rows_sha256=self._document_rows.hexdigest(),
                source_rows_sha256=self._source_rows.hexdigest(),
                symbol_rows_sha256=self._symbol_rows.hexdigest(),
                fts_rows_sha256=self._fts_rows.hexdigest(),
            )
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._document_manifest.close()
        self._source_manifest.close()
        self._symbol_manifest.close()


class _ManifestListSpool:
    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._first = True

    def add(self, value: str) -> None:
        if not self._first:
            self._file.write(b",")
        self._first = False
        self._file.write(canonical_json(value).encode("utf-8"))

    def update_digest(self, digest: hashlib._Hash) -> None:
        self._file.flush()
        self._file.seek(0)
        while chunk := self._file.read(65_536):
            digest.update(chunk)

    def close(self) -> None:
        self._file.close()


def _update_row_digest(digest: object, values: tuple[object, ...]) -> None:
    digest.update(canonical_json(list(values)).encode("utf-8"))
    digest.update(b"\n")


def changed_record_ids(
    connection: sqlite3.Connection,
    since: "IndexHighwater",
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Return every permanent record affected after *since*, in record order.

    Canonical records are immutable, so each append-only cursor names an
    explicit dependency of a projected document.  Relations deliberately add
    both endpoints; candidate activity only matters when a promotion certifies
    an existing Constitution record.
    """
    from .contracts import IndexHighwater

    if not isinstance(since, IndexHighwater):
        raise TypeError("since must be IndexHighwater")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    sql, parameters = _changed_records_query(since, limit=limit)
    return tuple(
        row[0]
        for row in connection.execute(sql, parameters)
    )


def project_stale_delta(
    connection: sqlite3.Connection,
    *,
    since: "IndexHighwater",
    limit: int,
) -> StaleDelta:
    """Return bounded stale candidates, never exposing terminal lifecycle rows."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    changed = changed_record_ids(connection, since, limit=limit + 1)
    if len(changed) > limit:
        return StaleDelta(record_ids=(), overflowed=True)
    if not changed:
        return StaleDelta(record_ids=(), overflowed=False)
    sql, parameters = _changed_records_query(since, limit=limit, active_only=True)
    rows = connection.execute(sql, parameters)
    return StaleDelta(record_ids=tuple(row[0] for row in rows), overflowed=False)


def _changed_records_query(
    since: "IndexHighwater", *, limit: int | None, active_only: bool = False
) -> tuple[str, tuple[int, ...]]:
    """One ordered canonical scan; never expand IDs into bound SQL variables."""
    predicate = "AND c.state IN ('proposed', 'active')" if active_only else ""
    suffix = "" if limit is None else " LIMIT ?"
    parameters: tuple[int, ...] = (
        since.record_seq,
        since.memory_event_seq,
        since.evidence_seq,
        since.relation_seq,
        since.relation_seq,
        since.candidate_event_seq,
    ) + (() if limit is None else (limit,))
    return (
        """WITH affected(record_id) AS (
            SELECT record_id FROM memory_records WHERE record_seq > ?
            UNION SELECT record_id FROM memory_events WHERE event_seq > ?
            UNION SELECT record_id FROM memory_evidence WHERE evidence_seq > ?
            UNION SELECT source_record_id FROM memory_relations WHERE relation_seq > ?
            UNION SELECT target_record_id FROM memory_relations WHERE relation_seq > ?
            UNION SELECT c.permanent_record_id
            FROM memory_candidate_current AS c
            JOIN memory_records AS r ON r.record_id = c.permanent_record_id
            WHERE c.state = 'promoted' AND r.record_type = 'constitution'
              AND c.last_event_seq > ?
        )
        SELECT r.record_id
        FROM memory_records AS r
        JOIN affected AS a ON a.record_id = r.record_id
        LEFT JOIN memory_current AS c ON c.record_id = r.record_id
        WHERE 1 = 1 """
        + predicate
        + " ORDER BY r.record_seq"
        + suffix,
        parameters,
    )


def project_full_snapshot(
    connection: sqlite3.Connection, snapshot: CanonicalSnapshot
) -> ProjectionBatch:
    evidence_by_record: dict[str, tuple[str, ...]] = {}
    for row in connection.execute(
        "SELECT record_id, ordinal, uri FROM memory_evidence ORDER BY record_id, ordinal"
    ):
        evidence_by_record.setdefault(row["record_id"], tuple())
        evidence_by_record[row["record_id"]] += (row["uri"],)
    endpoints: dict[str, list[str]] = {}
    for row in connection.execute(
        "SELECT source_record_id, target_record_id FROM memory_relations ORDER BY relation_seq"
    ):
        endpoints.setdefault(row["source_record_id"], []).append(row["target_record_id"])
        endpoints.setdefault(row["target_record_id"], []).append(row["source_record_id"])

    documents: list[ProjectedDocument] = []
    sources: list[ProjectedSource] = []
    symbols: list[ProjectedSymbol] = []
    omitted = 0
    rows = connection.execute(
        """SELECT r.*, c.state, c.last_event_seq
        FROM memory_records AS r
        JOIN memory_current AS c ON c.record_id = r.record_id
        ORDER BY r.record_seq"""
    )
    for row in rows:
        try:
            safe_references = tuple(
                safe_evidence_reference(uri)
                for uri in evidence_by_record.get(row["record_id"], ())
            )
            document = build_document(
                record_id=row["record_id"],
                record_seq=row["record_seq"],
                record_type=row["record_type"],
                scope_kind=row["scope_kind"],
                scope_key=row["scope_key"],
                topic_key=row["topic_key"],
                statement=row["statement"],
                value_json=row["value_json"],
                state=row["state"],
                created_at=row["created_at"],
                last_memory_event_seq=row["last_event_seq"],
                content_sha256=row["content_sha256"],
                safe_references=safe_references,
                relation_endpoints=tuple(endpoints.get(row["record_id"], ())),
            )
            document_symbols = extract_symbols(document)
        except (UnsafeProjectionContent, ValidationError, ValueError):
            omitted += 1
            continue
        documents.append(document)
        sources.extend(
            build_source(document.document_id, ordinal, reference)
            for ordinal, reference in enumerate(safe_references)
        )
        symbols.extend(document_symbols)
    manifest = {
        "documents": [document.document_sha256 for document in documents],
        "highwater": snapshot.highwater.to_dict(),
        "omitted_unsafe_count": omitted,
        "sources": [source.source_sha256 for source in sources],
        "symbols": [symbol.symbol_sha256 for symbol in symbols],
    }
    return ProjectionBatch(
        documents=tuple(documents),
        sources=tuple(sources),
        symbols=tuple(symbols),
        omitted_unsafe_count=omitted,
        manifest_sha256=hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest(),
        index_policy_fingerprint=_index_policy_fingerprint(),
    )


def stream_projection(
    connection: sqlite3.Connection,
    snapshot: CanonicalSnapshot,
    *,
    record_ids: frozenset[str] | None = None,
):
    """Yield canonical projection rows in stable order without global maps/lists."""
    del snapshot  # The verified snapshot owns consistency; rows carry its projection.
    for row in connection.execute(
        """SELECT r.*, c.state, c.last_event_seq FROM memory_records AS r
        JOIN memory_current AS c ON c.record_id = r.record_id ORDER BY r.record_seq"""
    ):
        if record_ids is not None and row["record_id"] not in record_ids:
            continue
        try:
            references = tuple(
                safe_evidence_reference(item[0]) for item in connection.execute(
                    """SELECT uri FROM memory_evidence WHERE record_id = ?
                    ORDER BY ordinal LIMIT ?""",
                    (row["record_id"], MAX_EVIDENCE_PER_DOCUMENT + 1),
                )
            )
            endpoints = tuple(item[0] for item in connection.execute(
                """SELECT CASE WHEN source_record_id = ? THEN target_record_id
                ELSE source_record_id END FROM memory_relations
                WHERE source_record_id = ? OR target_record_id = ?
                ORDER BY relation_seq LIMIT ?""",
                (
                    row["record_id"], row["record_id"], row["record_id"],
                    MAX_SYMBOLS_PER_DOCUMENT - 1,
                ),
            ))
            document = build_document(
                record_id=row["record_id"], record_seq=row["record_seq"], record_type=row["record_type"],
                scope_kind=row["scope_kind"], scope_key=row["scope_key"], topic_key=row["topic_key"],
                statement=row["statement"], value_json=row["value_json"], state=row["state"],
                created_at=row["created_at"], last_memory_event_seq=row["last_event_seq"],
                content_sha256=row["content_sha256"], safe_references=references, relation_endpoints=endpoints,
            )
            symbols = extract_symbols(document)
        except (UnsafeProjectionContent, ValidationError, ValueError):
            yield "omitted", None
            continue
        yield "document", document
        for ordinal, reference in enumerate(references):
            yield "source", build_source(document.document_id, ordinal, reference)
        for symbol in symbols:
            yield "symbol", symbol


def streaming_projection_digest(
    connection: sqlite3.Connection,
    snapshot: CanonicalSnapshot,
    *,
    row_consumer: Callable[[str, object | None], None] | None = None,
) -> ProjectionDigest:
    """Digest a verified projection and optionally consume each row as it streams."""
    builder = ProjectionDigestBuilder(snapshot)
    try:
        for kind, value in stream_projection(connection, snapshot):
            if kind == "omitted":
                builder.omit_unsafe()
            elif kind == "document":
                builder.add_document(value)
            elif kind == "source":
                builder.add_source(value)
            else:
                builder.add_symbol(value)
            if row_consumer is not None:
                row_consumer(kind, value)
        return builder.finish()
    except BaseException:
        builder.close()
        raise


def _index_policy_fingerprint() -> str:
    return hashlib.sha256(canonical_json({
        "format": "codex-memory.retrieval-projection.v1", "max_document_text_bytes": 65_536,
        "max_symbols_per_document": 16, "max_symbol_bytes": 256,
    }).encode("utf-8")).hexdigest()
