"""Read-only reconstruction of bounded, provenance-rich memory evidence packs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import sqlite3
from typing import Any
from urllib.parse import urlsplit

from codex_memory.blob_store import BlobIntegrityError, BlobStore
from codex_memory.ledger_common import (
    RecordNotFound,
    ValidationError,
    VerificationError,
    canonical_json,
    parse_json_value,
)
from codex_memory.memory_authority import constitution_certification
from codex_memory.memory_evidence import (
    parse_event_uri,
    resolve_event_evidence,
    safe_evidence_reference,
)
from codex_memory.memory_scope import ScopeContext
from codex_memory.memory_security import SecretRejected, reject_secrets
from .canonical_snapshot import (
    CanonicalSnapshot,
    require_matching_canonical_snapshot,
)
from .contracts import RetrievalQueryResult, RetrievalRequest, RetrievedCandidate
from .errors import RetrievalValidationError


_REVIEW_PAYLOAD_KEYS = frozenset(
    {
        "evidence_uris",
        "permanent_record_created",
        "permanent_record_id",
        "policy_code",
        "policy_version",
        "reason",
        "requested_outcome",
        "superseded_record_id",
    }
)


def reject_semantic_secrets(value: object) -> None:
    """Scan decoded fields and their combined content without JSON-escape artifacts."""
    strings: list[str] = []
    _collect_and_scan_semantic_strings(value, strings)
    reject_secrets(statement="\n".join(strings), value=None)


def _collect_and_scan_semantic_strings(value: object, strings: list[str]) -> None:
    if isinstance(value, str):
        strings.append(value)
        reject_secrets(statement=value, value=None)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                strings.append(key)
                reject_secrets(statement=key, value=None)
                lowered = key.lower()
                if any(
                    fragment in lowered
                    for fragment in ("password", "token", "api_key", "secret")
                ):
                    reject_secrets(
                        statement=f"{key}={canonical_json(item)}", value=None
                    )
                if lowered == "authorization":
                    reject_secrets(
                        statement=f"authorization: {item}", value=None
                    )
            _collect_and_scan_semantic_strings(item, strings)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_and_scan_semantic_strings(item, strings)


@dataclass(frozen=True, slots=True)
class SafeEvidence:
    reference: str
    reference_sha256: str
    event_session_id: str | None = None
    event_turn_id: str | None = None
    event_name: str | None = None
    event_created_at: str | None = None
    payload_blob_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_created_at": self.event_created_at,
            "event_name": self.event_name,
            "event_session_id": self.event_session_id,
            "event_turn_id": self.event_turn_id,
            "payload_blob_sha256": self.payload_blob_sha256,
            "reference": self.reference,
            "reference_sha256": self.reference_sha256,
        }

    def to_packet_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "reference": self.reference,
            "reference_sha256": self.reference_sha256,
        }
        if self.event_turn_id is not None:
            result["event_turn_id"] = self.event_turn_id
        if self.event_name is not None:
            result["event_name"] = self.event_name
        if self.payload_blob_sha256 is not None:
            result["payload_blob_sha256"] = self.payload_blob_sha256
        return result


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    event_id: str
    event_sequence: int
    action: str
    event_sha256: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "created_at": self.created_at,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "event_sha256": self.event_sha256,
        }

    def to_packet_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "event_sha256": self.event_sha256,
        }


@dataclass(frozen=True, slots=True)
class MemoryOrigin:
    session_id: str
    turn_id: str
    context_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
        }


@dataclass(frozen=True, slots=True)
class CandidateReviewProvenance:
    event: LedgerEvent
    requested_outcome: str
    reason: str
    policy_code: str
    policy_version: str
    permanent_record_created: bool | None
    permanent_record_id: str | None
    superseded_record_id: str | None
    evidence: tuple[SafeEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "evidence_refs": [item.reference_sha256 for item in self.evidence],
            "permanent_record_created": self.permanent_record_created,
            "permanent_record_id": self.permanent_record_id,
            "policy_code": self.policy_code,
            "policy_version": self.policy_version,
            "reason": self.reason,
            "requested_outcome": self.requested_outcome,
            "superseded_record_id": self.superseded_record_id,
        }

    def to_packet_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "event": self.event.to_packet_dict(),
            "evidence_refs": [item.reference_sha256 for item in self.evidence],
            "policy_code": self.policy_code,
            "reason": self.reason,
        }
        if self.requested_outcome != self.event.action:
            result["requested_outcome"] = self.requested_outcome
        if self.permanent_record_id is not None:
            result["permanent_record_created"] = self.permanent_record_created
        if self.superseded_record_id is not None:
            result["superseded_record_id"] = self.superseded_record_id
        return result


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    candidate_id: str
    candidate_content_sha256: str
    origin: MemoryOrigin
    capture_event: LedgerEvent
    capture_evidence: tuple[SafeEvidence, ...]
    reviews: tuple[CandidateReviewProvenance, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_content_sha256": self.candidate_content_sha256,
            "candidate_id": self.candidate_id,
            "capture_event": self.capture_event.to_dict(),
            "capture_evidence_refs": [
                item.reference_sha256 for item in self.capture_evidence
            ],
            "origin": self.origin.to_dict(),
            "reviews": [item.to_dict() for item in self.reviews],
        }

    def to_packet_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "capture_event": self.capture_event.to_packet_dict(),
            "capture_evidence_refs": [
                item.reference_sha256 for item in self.capture_evidence
            ],
            "origin": self.origin.to_dict(),
            "reviews": [item.to_packet_dict() for item in self.reviews],
        }


@dataclass(frozen=True, slots=True)
class PromotionProvenance(CandidateProvenance):
    @property
    def review_event(self) -> LedgerEvent:
        return self.reviews[-1].event

    @property
    def review_reason(self) -> str:
        return self.reviews[-1].reason

    @property
    def policy_code(self) -> str:
        return self.reviews[-1].policy_code

    @property
    def permanent_record_created(self) -> bool:
        value = self.reviews[-1].permanent_record_created
        if type(value) is not bool:
            raise VerificationError("promotion provenance creation flag is missing")
        return value

    @property
    def superseded_record_id(self) -> str | None:
        return self.reviews[-1].superseded_record_id

    @property
    def review_evidence(self) -> tuple[SafeEvidence, ...]:
        return self.reviews[-1].evidence


@dataclass(frozen=True, slots=True)
class SupersessionProvenance:
    relation_id: str
    relation_sequence: int
    relation_type: str
    source_record_id: str
    target_record_id: str
    relation_sha256: str
    lifecycle_event_id: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "lifecycle_event_id": self.lifecycle_event_id,
            "relation_id": self.relation_id,
            "relation_sequence": self.relation_sequence,
            "relation_sha256": self.relation_sha256,
            "relation_type": self.relation_type,
            "source_record_id": self.source_record_id,
            "target_record_id": self.target_record_id,
        }

    def to_packet_dict(self) -> dict[str, object]:
        result = self.to_dict()
        del result["created_at"]
        return result


@dataclass(frozen=True, slots=True, init=False)
class MemoryCard:
    identifier: str
    candidate_kind: str
    record_type: str
    scope_kind: str
    scope_key: str
    topic_key: str
    statement: str
    _value_json: str
    state: str
    sequence: int
    content_sha256: str
    trust_label: str
    certified: bool
    last_event: LedgerEvent

    def __init__(
        self,
        identifier: str,
        candidate_kind: str,
        record_type: str,
        scope_kind: str,
        scope_key: str,
        topic_key: str,
        statement: str,
        value: Any,
        state: str,
        sequence: int,
        content_sha256: str,
        trust_label: str,
        certified: bool,
        last_event: LedgerEvent,
    ) -> None:
        for name, item in (
            ("identifier", identifier),
            ("candidate_kind", candidate_kind),
            ("record_type", record_type),
            ("scope_kind", scope_kind),
            ("scope_key", scope_key),
            ("topic_key", topic_key),
            ("statement", statement),
            ("state", state),
            ("sequence", sequence),
            ("content_sha256", content_sha256),
            ("trust_label", trust_label),
            ("certified", certified),
            ("last_event", last_event),
        ):
            object.__setattr__(self, name, item)
        object.__setattr__(self, "_value_json", canonical_json(value))

    @property
    def value(self) -> Any:
        """Return a detached JSON value; callers cannot mutate the card."""
        return parse_json_value(self._value_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_kind": self.candidate_kind,
            "certified": self.certified,
            "content_sha256": self.content_sha256,
            "id": self.identifier,
            "kind": self.record_type,
            "last_event": self.last_event.to_dict(),
            "scope": {"key": self.scope_key, "kind": self.scope_kind},
            "sequence": self.sequence,
            "state": self.state,
            "statement": self.statement,
            "topic": self.topic_key,
            "trust_label": self.trust_label,
            "value": self.value,
        }

    def to_packet_dict(self) -> dict[str, object]:
        result = self.to_dict()
        result["last_event"] = self.last_event.to_packet_dict()
        return result


@dataclass(frozen=True, slots=True)
class EvidencePack:
    card: MemoryCard
    evidence: tuple[SafeEvidence, ...]
    promotions: tuple[PromotionProvenance, ...] = ()
    candidate_provenance: CandidateProvenance | None = None
    supersession_relations: tuple[SupersessionProvenance, ...] = ()
    provenance_status: str = "complete"
    provenance_omissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        registry = {item.reference_sha256: item for item in self.evidence}
        if len(registry) != len(self.evidence):
            raise RetrievalValidationError("evidence registry contains duplicates")
        provenance = (*self.promotions,)
        if self.candidate_provenance is not None:
            provenance = (*provenance, self.candidate_provenance)
        for source in provenance:
            if source.candidate_content_sha256 != self.card.content_sha256:
                raise RetrievalValidationError("candidate content provenance mismatch")
            referenced = (
                *source.capture_evidence,
                *(evidence for review in source.reviews for evidence in review.evidence),
            )
            if any(
                registry.get(item.reference_sha256) != item for item in referenced
            ):
                raise RetrievalValidationError("candidate evidence reference is missing")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "card": self.card.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "pack_version": 1,
            "provenance_status": self.provenance_status,
        }
        if self.promotions:
            result["promotions"] = [item.to_dict() for item in self.promotions]
        if self.candidate_provenance is not None:
            result["candidate_provenance"] = self.candidate_provenance.to_dict()
        if self.provenance_omissions:
            result["provenance_omissions"] = list(self.provenance_omissions)
        if self.supersession_relations:
            result["supersession_relations"] = [
                item.to_dict() for item in self.supersession_relations
            ]
        return result

    def to_packet_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "card": self.card.to_packet_dict(),
            "evidence": [item.to_packet_dict() for item in self.evidence],
            "pack_version": 1,
            "provenance_status": self.provenance_status,
        }
        if self.promotions:
            result["promotions"] = [
                item.to_packet_dict() for item in self.promotions
            ]
        if self.candidate_provenance is not None:
            result["candidate_provenance"] = (
                self.candidate_provenance.to_packet_dict()
            )
        if self.provenance_omissions:
            result["provenance_omissions"] = list(self.provenance_omissions)
        if self.supersession_relations:
            result["supersession_relations"] = [
                item.to_packet_dict() for item in self.supersession_relations
            ]
        return result

    @property
    def candidate_reviews(self) -> tuple[CandidateReviewProvenance, ...]:
        if self.candidate_provenance is None:
            return ()
        return self.candidate_provenance.reviews

    @property
    def pack_sha256(self) -> str:
        return evidence_pack_sha256(self)


@dataclass(frozen=True, slots=True)
class PackOmission:
    candidate_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"candidate_id": self.candidate_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class EvidencePackBuildResult:
    packs: tuple[EvidencePack, ...]
    omissions: tuple[PackOmission, ...]
    unsafe_records_omitted: int
    degraded_reasons: tuple[str, ...]
    prompt_evidence: SafeEvidence | None = None


def evidence_pack_sha256(pack: EvidencePack) -> str:
    if not isinstance(pack, EvidencePack):
        raise TypeError("pack must be EvidencePack")
    return hashlib.sha256(canonical_json(pack.to_dict()).encode("utf-8")).hexdigest()


def build_evidence_packs(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    snapshot: CanonicalSnapshot,
) -> EvidencePackBuildResult:
    """Reconstruct selected candidates from one verified canonical snapshot.

    The builder reads event blobs only to verify their hashes. Blob payload bytes
    are never retained in or serialized from an evidence pack.
    """
    if not isinstance(request, RetrievalRequest):
        raise RetrievalValidationError("request must be RetrievalRequest")
    if not isinstance(result, RetrievalQueryResult):
        raise RetrievalValidationError("result must be RetrievalQueryResult")
    if not isinstance(snapshot, CanonicalSnapshot):
        raise RetrievalValidationError("snapshot must be CanonicalSnapshot")
    if result.canonical_highwater != snapshot.highwater:
        raise RetrievalValidationError("retrieval result does not match canonical snapshot")

    scope = ScopeContext(
        request.scope.project_scope_key, request.scope.workspace_scope_keys
    )
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN")
    packs: list[EvidencePack] = []
    omissions: list[PackOmission] = []
    degraded: list[str] = []
    unsafe_count = 0
    try:
        require_matching_canonical_snapshot(connection, snapshot)
        prompt_evidence = _verified_prompt_evidence(
            connection, blobs, request=request
        )
        for candidate in result.candidates:
            try:
                if candidate.candidate_kind == "record":
                    pack = _permanent_pack(
                        connection, blobs, scope=scope, candidate=candidate
                    )
                elif candidate.candidate_kind == "temporary":
                    pack = _temporary_pack(
                        connection,
                        blobs,
                        scope=scope,
                        session_id=request.session_id,
                        candidate=candidate,
                    )
                else:
                    raise _PackOmitted("unsupported_candidate_kind")
                reject_semantic_secrets(pack.to_dict())
                reject_semantic_secrets(
                    parse_json_value(canonical_json(pack.to_packet_dict()))
                )
            except SecretRejected:
                unsafe_count += 1
                omissions.append(PackOmission(candidate.candidate_id, "unsafe_content"))
                degraded.append("unsafe_content_omitted")
            except _PackOmitted as exc:
                omissions.append(PackOmission(candidate.candidate_id, exc.reason))
                degraded.append(exc.reason)
            except (BlobIntegrityError, OSError):
                omissions.append(
                    PackOmission(candidate.candidate_id, "evidence_unavailable")
                )
                degraded.append("evidence_unavailable")
            except RecordNotFound:
                omissions.append(PackOmission(candidate.candidate_id, "evidence_missing"))
                degraded.append("evidence_missing")
            except (ValidationError, VerificationError, ValueError):
                omissions.append(PackOmission(candidate.candidate_id, "provenance_invalid"))
                degraded.append("provenance_invalid")
            else:
                packs.append(pack)
        return EvidencePackBuildResult(
            packs=tuple(packs),
            omissions=tuple(omissions),
            unsafe_records_omitted=unsafe_count,
            degraded_reasons=tuple(dict.fromkeys(degraded)),
            prompt_evidence=prompt_evidence,
        )
    finally:
        if owns_transaction and connection.in_transaction:
            connection.rollback()


class _PackOmitted(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _verified_prompt_evidence(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    request: RetrievalRequest,
) -> SafeEvidence | None:
    uri = request.prompt_event_uri
    if uri is None:
        return None
    try:
        session_id, _event_id = parse_event_uri(uri)
    except ValueError as exc:
        raise RetrievalValidationError("prompt evidence URI is invalid") from exc
    if session_id != request.session_id:
        raise RetrievalValidationError(
            "prompt evidence belongs to another session"
        )
    try:
        evidence = _safe_evidence(
            connection, blobs, uri, expected_session=request.session_id
        )
    except RecordNotFound as exc:
        raise RetrievalValidationError("prompt evidence event was not found") from exc
    except (BlobIntegrityError, OSError) as exc:
        raise VerificationError("prompt evidence blob verification failed") from exc
    if evidence.event_name != "UserPromptSubmit":
        raise RetrievalValidationError(
            "prompt evidence is not the current session user prompt"
        )
    return evidence


def _permanent_pack(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    scope: ScopeContext,
    candidate: RetrievedCandidate,
) -> EvidencePack:
    row = connection.execute(
        """SELECT r.*, c.state, c.last_event_seq, c.last_event_id, c.updated_at
        FROM memory_records AS r JOIN memory_current AS c ON c.record_id = r.record_id
        WHERE r.record_id = ?""",
        (candidate.candidate_id,),
    ).fetchone()
    if row is None:
        raise _PackOmitted("candidate_missing")
    if row["state"] != "active":
        raise _PackOmitted("candidate_not_current")
    if not scope.contains(scope_kind=row["scope_kind"], scope_key=row["scope_key"]):
        raise _PackOmitted("candidate_out_of_scope")
    if (
        candidate.record_type,
        candidate.scope_kind,
        candidate.scope_key,
        candidate.state,
        candidate.sequence,
        candidate.content_sha256,
    ) != (
        row["record_type"],
        row["scope_kind"],
        row["scope_key"],
        row["state"],
        row["record_seq"],
        row["content_sha256"],
    ):
        raise _PackOmitted("candidate_identity_mismatch")

    value = parse_json_value(row["value_json"])
    reject_semantic_secrets({"statement": row["statement"], "value": value})
    lifecycle_row = connection.execute(
        "SELECT * FROM memory_events WHERE event_seq = ? AND event_id = ?",
        (row["last_event_seq"], row["last_event_id"]),
    ).fetchone()
    if lifecycle_row is None or lifecycle_row["record_id"] != row["record_id"]:
        raise VerificationError("record lifecycle provenance is missing")
    certified = False
    if row["record_type"] == "constitution":
        certified, _references = constitution_certification(
            connection, blobs, record=row
        )
    if candidate.mandatory != certified:
        raise _PackOmitted("candidate_authority_mismatch")

    permanent_evidence = tuple(
        _safe_evidence(connection, blobs, item["uri"])
        for item in connection.execute(
            "SELECT uri FROM memory_evidence WHERE record_id = ? ORDER BY ordinal",
            (row["record_id"],),
        )
    )
    promotions = _promotion_provenance(
        connection, blobs, permanent_record_id=row["record_id"]
    )
    relations = tuple(
        SupersessionProvenance(
            relation_id=item["relation_id"],
            relation_sequence=item["relation_seq"],
            relation_type=item["relation_type"],
            source_record_id=item["source_record_id"],
            target_record_id=item["target_record_id"],
            relation_sha256=item["relation_sha256"],
            lifecycle_event_id=item["event_id"],
            created_at=item["created_at"],
        )
        for item in connection.execute(
            """SELECT * FROM memory_relations
            WHERE source_record_id = ? OR target_record_id = ? ORDER BY relation_seq""",
            (row["record_id"], row["record_id"]),
        )
    )
    evidence_groups = [permanent_evidence]
    for promotion in promotions:
        evidence_groups.append(promotion.capture_evidence)
        evidence_groups.extend(review.evidence for review in promotion.reviews)
    evidence = _merge_safe_evidence(*evidence_groups)
    return EvidencePack(
        card=MemoryCard(
            identifier=row["record_id"],
            candidate_kind="record",
            record_type=row["record_type"],
            scope_kind=row["scope_kind"],
            scope_key=row["scope_key"],
            topic_key=row["topic_key"],
            statement=row["statement"],
            value=value,
            state=row["state"],
            sequence=row["record_seq"],
            content_sha256=row["content_sha256"],
            trust_label=(
                "CERTIFIED CONSTITUTION"
                if certified
                else "UNVERIFIED CONSTITUTION"
                if row["record_type"] == "constitution"
                else "RETRIEVED MEMORY"
            ),
            certified=certified,
            last_event=_ledger_event(lifecycle_row),
        ),
        evidence=evidence,
        promotions=promotions,
        supersession_relations=relations,
    )


def _temporary_pack(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    scope: ScopeContext,
    session_id: str,
    candidate: RetrievedCandidate,
) -> EvidencePack:
    row = connection.execute(
        """SELECT c.*, current.state, current.last_event_seq,
                  current.last_event_id, current.updated_at
        FROM memory_candidates AS c
        JOIN memory_candidate_current AS current
          ON current.candidate_id = c.candidate_id
        WHERE c.candidate_id = ?""",
        (candidate.candidate_id,),
    ).fetchone()
    if row is None:
        raise _PackOmitted("candidate_missing")
    if row["state"] != "temporary" or row["session_id"] != session_id:
        raise _PackOmitted("temporary_not_visible")
    if not scope.contains(scope_kind=row["scope_kind"], scope_key=row["scope_key"]):
        raise _PackOmitted("candidate_out_of_scope")
    if (
        candidate.record_type,
        candidate.scope_kind,
        candidate.scope_key,
        candidate.state,
        candidate.sequence,
        candidate.content_sha256,
        candidate.mandatory,
        candidate.trust_label,
    ) != (
        row["record_type"],
        row["scope_kind"],
        row["scope_key"],
        row["state"],
        row["candidate_seq"],
        row["content_sha256"],
        False,
        "UNVERIFIED TEMPORARY",
    ):
        raise _PackOmitted("candidate_identity_mismatch")
    value = parse_json_value(row["value_json"])
    reject_semantic_secrets({"statement": row["statement"], "value": value})
    lifecycle = connection.execute(
        "SELECT * FROM memory_candidate_events WHERE event_seq = ? AND event_id = ?",
        (row["last_event_seq"], row["last_event_id"]),
    ).fetchone()
    if lifecycle is None or lifecycle["candidate_id"] != row["candidate_id"]:
        raise VerificationError("temporary lifecycle provenance is missing")
    capture_evidence = tuple(
        _safe_evidence(
            connection, blobs, item["uri"], expected_session=row["session_id"]
        )
        for item in connection.execute(
            "SELECT uri FROM memory_candidate_evidence "
            "WHERE candidate_id = ? ORDER BY ordinal",
            (row["candidate_id"],),
        )
    )
    reviews = _candidate_reviews(
        connection,
        blobs,
        candidate_id=row["candidate_id"],
        expected_session=row["session_id"],
    )
    evidence = _merge_safe_evidence(
        capture_evidence, *(review.evidence for review in reviews)
    )
    capture = connection.execute(
        "SELECT * FROM memory_candidate_events "
        "WHERE candidate_id = ? AND action = 'capture' ORDER BY event_seq LIMIT 1",
        (row["candidate_id"],),
    ).fetchone()
    if capture is None:
        raise VerificationError("candidate capture provenance is missing")
    candidate_provenance = CandidateProvenance(
        candidate_id=row["candidate_id"],
        candidate_content_sha256=row["content_sha256"],
        origin=MemoryOrigin(
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            context_id=row["context_id"],
        ),
        capture_event=_ledger_event(capture),
        capture_evidence=capture_evidence,
        reviews=reviews,
    )
    return EvidencePack(
        card=MemoryCard(
            identifier=row["candidate_id"],
            candidate_kind="temporary",
            record_type=row["record_type"],
            scope_kind=row["scope_kind"],
            scope_key=row["scope_key"],
            topic_key=row["topic_key"],
            statement=row["statement"],
            value=value,
            state=row["state"],
            sequence=row["candidate_seq"],
            content_sha256=row["content_sha256"],
            trust_label="UNVERIFIED TEMPORARY",
            certified=False,
            last_event=_ledger_event(lifecycle),
        ),
        evidence=evidence,
        candidate_provenance=candidate_provenance,
    )


def _promotion_provenance(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    permanent_record_id: str,
) -> tuple[PromotionProvenance, ...]:
    rows = connection.execute(
        """SELECT c.*
        FROM memory_candidates AS c
        JOIN memory_candidate_current AS current
          ON current.candidate_id = c.candidate_id
        WHERE current.state = 'promoted'
          AND current.permanent_record_id = ?
        ORDER BY c.candidate_seq""",
        (permanent_record_id,),
    ).fetchall()
    result: list[PromotionProvenance] = []
    for row in rows:
        capture = connection.execute(
            "SELECT * FROM memory_candidate_events "
            "WHERE candidate_id = ? AND action = 'capture' ORDER BY event_seq LIMIT 1",
            (row["candidate_id"],),
        ).fetchone()
        if capture is None:
            raise VerificationError("candidate capture provenance is missing")
        capture_uris = tuple(
            item["uri"]
            for item in connection.execute(
                "SELECT uri FROM memory_candidate_evidence "
                "WHERE candidate_id = ? ORDER BY ordinal",
                (row["candidate_id"],),
            )
        )
        reviews = _candidate_reviews(
            connection,
            blobs,
            candidate_id=row["candidate_id"],
            expected_session=row["session_id"],
        )
        if (
            not reviews
            or reviews[-1].event.action != "promote"
            or reviews[-1].permanent_record_id != permanent_record_id
            or type(reviews[-1].permanent_record_created) is not bool
        ):
            raise VerificationError("candidate promotion provenance is invalid")
        result.append(
            PromotionProvenance(
                candidate_id=row["candidate_id"],
                candidate_content_sha256=row["content_sha256"],
                origin=MemoryOrigin(
                    session_id=row["session_id"],
                    turn_id=row["turn_id"],
                    context_id=row["context_id"],
                ),
                capture_event=_ledger_event(capture),
                capture_evidence=tuple(
                    _safe_evidence(
                        connection, blobs, uri, expected_session=row["session_id"]
                    )
                    for uri in capture_uris
                ),
                reviews=reviews,
            )
        )
    return tuple(result)


def _candidate_reviews(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    *,
    candidate_id: str,
    expected_session: str,
) -> tuple[CandidateReviewProvenance, ...]:
    return tuple(
        _candidate_review_provenance(
            connection,
            blobs,
            row,
            expected_session=expected_session,
        )
        for row in connection.execute(
            "SELECT * FROM memory_candidate_events "
            "WHERE candidate_id = ? AND action <> 'capture' ORDER BY event_seq",
            (candidate_id,),
        )
    )


def _candidate_review_provenance(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    row: sqlite3.Row,
    *,
    expected_session: str,
) -> CandidateReviewProvenance:
    payload = parse_json_value(row["payload_json"])
    if not isinstance(payload, dict) or set(payload) != _REVIEW_PAYLOAD_KEYS:
        raise VerificationError("candidate review provenance is invalid")
    evidence_uris = payload["evidence_uris"]
    reason = payload["reason"]
    policy_code = payload["policy_code"]
    policy_version = payload["policy_version"]
    requested_outcome = payload["requested_outcome"]
    permanent_created = payload["permanent_record_created"]
    permanent_id = payload["permanent_record_id"]
    superseded_id = payload["superseded_record_id"]
    if (
        not isinstance(evidence_uris, list)
        or any(not isinstance(uri, str) for uri in evidence_uris)
        or not isinstance(reason, str)
        or not isinstance(policy_code, str)
        or not isinstance(policy_version, str)
        or not isinstance(requested_outcome, str)
        or (permanent_created is not None and type(permanent_created) is not bool)
        or (permanent_id is not None and not isinstance(permanent_id, str))
        or (superseded_id is not None and not isinstance(superseded_id, str))
    ):
        raise VerificationError("candidate review provenance is invalid")
    reject_secrets(statement=reason, value=None)
    return CandidateReviewProvenance(
        event=_ledger_event(row),
        requested_outcome=requested_outcome,
        reason=reason,
        policy_code=policy_code,
        policy_version=policy_version,
        permanent_record_created=permanent_created,
        permanent_record_id=permanent_id,
        superseded_record_id=superseded_id,
        evidence=tuple(
            _safe_evidence(
                connection,
                blobs,
                uri,
                expected_session=expected_session,
            )
            for uri in evidence_uris
        ),
    )


def _merge_safe_evidence(
    *groups: tuple[SafeEvidence, ...],
) -> tuple[SafeEvidence, ...]:
    result: list[SafeEvidence] = []
    by_hash: dict[str, SafeEvidence] = {}
    for group in groups:
        for item in group:
            prior = by_hash.get(item.reference_sha256)
            if prior is not None:
                if prior != item:
                    raise VerificationError("evidence reference hash collision")
                continue
            by_hash[item.reference_sha256] = item
            result.append(item)
    return tuple(result)


def _ledger_event(row: sqlite3.Row) -> LedgerEvent:
    return LedgerEvent(
        event_id=row["event_id"],
        event_sequence=row["event_seq"],
        action=row["action"],
        event_sha256=row["event_sha256"],
        created_at=row["created_at"],
    )


def _safe_evidence(
    connection: sqlite3.Connection,
    blobs: BlobStore,
    uri: str,
    *,
    expected_session: str | None = None,
) -> SafeEvidence:
    if urlsplit(uri).scheme.lower() != "codex-event":
        reference = safe_evidence_reference(uri)
        return SafeEvidence(reference, _text_sha256(reference))
    event = resolve_event_evidence(connection, blobs, uri)
    if expected_session is not None and event.session_id != expected_session:
        raise VerificationError("candidate evidence belongs to another session")
    row = connection.execute(
        "SELECT payload_blob_sha256 FROM events WHERE event_id = ? AND session_id = ?",
        (event.event_id, event.session_id),
    ).fetchone()
    if row is None:
        raise VerificationError("event evidence metadata is missing")
    return SafeEvidence(
        reference=event.uri,
        reference_sha256=_text_sha256(event.uri),
        event_session_id=event.session_id,
        event_turn_id=event.turn_id,
        event_name=event.event_name,
        event_created_at=event.created_at,
        payload_blob_sha256=row["payload_blob_sha256"],
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
