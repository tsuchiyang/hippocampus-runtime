"""Deterministic, read-only selection for bounded Phase 2 memory packets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Any

from codex_memory.blob_store import BlobStore

from .canonical_snapshot import CanonicalSnapshot, open_verified_canonical_snapshot
from .deadline import DeadlineBudget
from .contracts import (
    CandidateChannel,
    EpistemicContext,
    IndexHighwater,
    RetrievalQueryResult,
    RetrievalRequest,
    RetrievalScope,
)
from .errors import RetrievalValidationError
from .evidence_pack import (
    EvidencePack,
    EvidencePackBuildResult,
    PackOmission,
    SafeEvidence,
    build_evidence_packs,
)
from .paths import RetrievalPaths
from .render import (
    render_evidence_line,
    render_phase2_packet,
    render_prompt_evidence,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class SelectionIdentity:
    candidate_id: str
    content_sha256: str
    last_event_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise RetrievalValidationError("selection candidate ID is invalid")
        if not isinstance(self.content_sha256, str) or not _SHA256.fullmatch(
            self.content_sha256
        ):
            raise RetrievalValidationError("selection content hash is invalid")
        if (
            isinstance(self.last_event_sequence, bool)
            or not isinstance(self.last_event_sequence, int)
            or self.last_event_sequence < 0
        ):
            raise RetrievalValidationError("selection lifecycle sequence is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "content_sha256": self.content_sha256,
            "last_event_sequence": self.last_event_sequence,
        }


@dataclass(frozen=True, slots=True)
class PriorContextSelection:
    """A future P2.5 committed-manifest view consumed as pure input in P2.4."""

    manifest_id: str
    committed: bool
    session_id: str
    context_id: str
    scope: RetrievalScope
    epistemic_context: EpistemicContext
    canonical_highwater: IndexHighwater
    canonical_content_fingerprint: str
    index_stale: bool
    degraded_reasons: tuple[str, ...]
    records: tuple[SelectionIdentity, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.manifest_id, "manifest ID"),
            (self.session_id, "session ID"),
            (self.context_id, "context ID"),
        ):
            if not isinstance(value, str) or not value:
                raise RetrievalValidationError(f"prior {label} is invalid")
        if not isinstance(self.committed, bool) or not isinstance(self.index_stale, bool):
            raise RetrievalValidationError("prior selection flags are invalid")
        if not isinstance(self.scope, RetrievalScope):
            raise RetrievalValidationError("prior retrieval scope is invalid")
        if not isinstance(self.epistemic_context, EpistemicContext):
            raise RetrievalValidationError("prior epistemic context is invalid")
        if not isinstance(self.canonical_highwater, IndexHighwater):
            raise RetrievalValidationError("prior canonical highwater is invalid")
        if (
            not isinstance(self.canonical_content_fingerprint, str)
            or not _SHA256.fullmatch(self.canonical_content_fingerprint)
        ):
            raise RetrievalValidationError(
                "prior canonical content fingerprint is invalid"
            )
        if not isinstance(self.degraded_reasons, tuple) or any(
            not isinstance(reason, str) or not reason for reason in self.degraded_reasons
        ):
            raise RetrievalValidationError("prior degraded reasons are invalid")
        if not isinstance(self.records, tuple) or any(
            not isinstance(item, SelectionIdentity) for item in self.records
        ):
            raise RetrievalValidationError("prior selection records are invalid")
        identifiers = [item.candidate_id for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise RetrievalValidationError("prior selection contains duplicate records")


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    packs: tuple[EvidencePack, ...]
    pack_groups: tuple[tuple[EvidencePack, ...], ...]
    duplicate_ids: tuple[str, ...]
    suppressed_ids: tuple[str, ...]
    build_omissions: tuple[PackOmission, ...]
    unsafe_records_omitted: int
    degraded_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PacketAuditSnapshot:
    """Deep-frozen facts captured inside the packet's canonical transaction."""

    canonical_snapshot: CanonicalSnapshot
    query_candidates: tuple[object, ...]
    query_trace: tuple[object, ...]
    prompt_evidence: object | None
    evidence_packs: tuple[object, ...]
    final_selected: tuple[object, ...]
    budget_omitted_ids: tuple[str, ...]
    build_omissions: tuple[object, ...]
    duplicate_ids: tuple[str, ...]
    suppressed_ids: tuple[str, ...]
    omission_counts: tuple[tuple[str, int], ...]
    degraded_reasons: tuple[str, ...]
    request_binding: object | None = None
    result_metadata: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_snapshot, CanonicalSnapshot):
            raise RetrievalValidationError("audit canonical snapshot is invalid")
        for name in ("query_candidates", "query_trace", "evidence_packs", "final_selected", "build_omissions"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise RetrievalValidationError(f"audit {name} must be a tuple")
            object.__setattr__(self, name, tuple(_freeze_audit(item) for item in value))
        for name in ("budget_omitted_ids", "duplicate_ids", "suppressed_ids", "degraded_reasons"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise RetrievalValidationError(f"audit {name} is invalid")
        if self.prompt_evidence is not None:
            object.__setattr__(self, "prompt_evidence", _freeze_audit(self.prompt_evidence))
        for name in ("request_binding", "result_metadata"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_audit(value))
        if not isinstance(self.omission_counts, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
            or isinstance(item[1], bool) or not isinstance(item[1], int) or item[1] < 0
            for item in self.omission_counts
        ):
            raise RetrievalValidationError("audit omission counts are invalid")


@dataclass(frozen=True, slots=True)
class AssembledPacket:
    packet: str
    manifest_id: str
    packet_sha256: str
    packet_chars: int
    selected: tuple[SelectionIdentity, ...]
    budget_omitted_ids: tuple[str, ...]
    build_omissions: tuple[PackOmission, ...]
    duplicate_ids: tuple[str, ...]
    suppressed_ids: tuple[str, ...]
    omission_counts: tuple[tuple[str, int], ...]
    degraded_reasons: tuple[str, ...]
    audit_snapshot: PacketAuditSnapshot | None = None


class PacketBudgetError(RetrievalValidationError):
    """The immutable protocol or mandatory data cannot fit the hard cap."""


def build_phase2_packet(
    paths: RetrievalPaths,
    *,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    prior: PriorContextSelection | None = None,
    deadline: DeadlineBudget | None = None,
) -> AssembledPacket:
    """Reopen canonical truth read-only and assemble an offline Phase 2 packet."""
    if not isinstance(paths, RetrievalPaths):
        raise RetrievalValidationError("paths must be RetrievalPaths")
    if not isinstance(request, RetrievalRequest):
        raise RetrievalValidationError("request must be RetrievalRequest")
    if not isinstance(result, RetrievalQueryResult):
        raise RetrievalValidationError("result must be RetrievalQueryResult")
    if request.mode.value == "active":
        raise RetrievalValidationError("active retrieval mode is unavailable in P2.4")
    with open_verified_canonical_snapshot(paths, **({"deadline": deadline} if deadline is not None else {})) as (connection, snapshot):
        build = build_evidence_packs(
            connection,
            BlobStore(paths.runtime_root / "blobs"),
            request=request,
            result=result,
            snapshot=snapshot,
        )
        return assemble_evidence_packet(
            request=request,
            result=result,
            snapshot=snapshot,
            build=build,
            prior=prior,
        )


def select_evidence_packs(
    *,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    snapshot: CanonicalSnapshot,
    build: EvidencePackBuildResult,
    prior: PriorContextSelection | None = None,
) -> EvidenceSelection:
    """Order, deduplicate, and conservatively suppress complete evidence packs."""
    if not isinstance(request, RetrievalRequest):
        raise RetrievalValidationError("request must be RetrievalRequest")
    if not isinstance(result, RetrievalQueryResult):
        raise RetrievalValidationError("result must be RetrievalQueryResult")
    if not isinstance(snapshot, CanonicalSnapshot):
        raise RetrievalValidationError("snapshot must be CanonicalSnapshot")
    if not isinstance(build, EvidencePackBuildResult):
        raise RetrievalValidationError("build must be EvidencePackBuildResult")
    if prior is not None and not isinstance(prior, PriorContextSelection):
        raise RetrievalValidationError("prior must be PriorContextSelection or null")
    if result.canonical_highwater != snapshot.highwater:
        raise RetrievalValidationError("retrieval result does not match canonical snapshot")

    certified = [pack for pack in build.packs if pack.card.certified]
    temporary = [
        pack
        for pack in build.packs
        if not pack.card.certified and pack.card.candidate_kind == "temporary"
    ]
    permanent = [
        pack
        for pack in build.packs
        if not pack.card.certified and pack.card.candidate_kind == "record"
    ]
    permanent = list(_diversify_permanent(tuple(permanent), result))
    ordered = (*certified, *temporary, *permanent)

    reasons = [reason.value for reason in result.degraded_reasons]
    reasons.extend(build.degraded_reasons)
    current_degraded_reasons = tuple(dict.fromkeys(reasons))
    prior_by_id = {} if prior is None else {
        item.candidate_id: item for item in prior.records
    }
    suppressible = _prior_is_usable(
        prior,
        request,
        result,
        snapshot,
        degraded_reasons=current_degraded_reasons,
    )
    suppressed_ids: list[str] = []
    eligible_ordered: list[EvidencePack] = []
    for pack in ordered:
        if suppressible and _matches_prior_selection(pack, prior_by_id):
            suppressed_ids.append(pack.card.identifier)
            continue
        eligible_ordered.append(pack)
    representatives, duplicate_ids, pack_groups = _deduplicate_packs(
        tuple(eligible_ordered),
        request=request,
        prompt_evidence=build.prompt_evidence,
    )

    return EvidenceSelection(
        packs=representatives,
        pack_groups=pack_groups,
        duplicate_ids=tuple(duplicate_ids),
        suppressed_ids=tuple(suppressed_ids),
        build_omissions=build.omissions,
        unsafe_records_omitted=build.unsafe_records_omitted,
        degraded_reasons=current_degraded_reasons,
    )


def _deduplicate_packs(
    ordered: tuple[EvidencePack, ...],
    *,
    request: RetrievalRequest,
    prompt_evidence: SafeEvidence | None,
) -> tuple[
    tuple[EvidencePack, ...],
    tuple[str, ...],
    tuple[tuple[EvidencePack, ...], ...],
]:
    groups: dict[str, list[tuple[int, EvidencePack]]] = {}
    group_order: list[str] = []
    for index, pack in enumerate(ordered):
        digest = pack.card.content_sha256
        if digest not in groups:
            groups[digest] = []
            group_order.append(digest)
        groups[digest].append((index, pack))

    chosen_indices: set[int] = set()
    representatives: list[EvidencePack] = []
    pack_groups: list[tuple[EvidencePack, ...]] = []
    for digest in group_order:
        options = groups[digest]
        best_partition = min(_partition_rank(pack) for _index, pack in options)
        eligible = [
            (index, pack)
            for index, pack in options
            if _partition_rank(pack) == best_partition
        ]
        chosen_index, chosen = min(
            eligible,
            key=lambda item: (
                len(render_evidence_line(item[1]))
                + len(
                    render_prompt_evidence(
                        request,
                        prompt_evidence,
                        packs=(item[1],),
                    )
                ),
                item[0],
            ),
        )
        chosen_indices.add(chosen_index)
        representatives.append(chosen)
        fallback = [
            (index, pack)
            for index, pack in options
            if index != chosen_index
            and (not chosen.card.certified or pack.card.certified)
        ]
        fallback.sort(
            key=lambda item: (
                _partition_rank(item[1]),
                len(render_evidence_line(item[1]))
                + len(
                    render_prompt_evidence(
                        request,
                        prompt_evidence,
                        packs=(item[1],),
                    )
                ),
                item[0],
            )
        )
        pack_groups.append((chosen, *(pack for _index, pack in fallback)))

    duplicates = tuple(
        pack.card.identifier
        for index, pack in enumerate(ordered)
        if index not in chosen_indices
    )
    return tuple(representatives), duplicates, tuple(pack_groups)


def _partition_rank(pack: EvidencePack) -> int:
    if pack.card.certified:
        return 0
    if pack.card.candidate_kind == "temporary":
        return 1
    return 2


def _matches_prior_selection(
    pack: EvidencePack, prior_by_id: dict[str, SelectionIdentity]
) -> bool:
    prior = prior_by_id.get(pack.card.identifier)
    return (
        pack.card.candidate_kind == "record"
        and not pack.card.certified
        and prior is not None
        and prior.content_sha256 == pack.card.content_sha256
        and prior.last_event_sequence == pack.card.last_event.event_sequence
    )


def assemble_evidence_packet(
    *,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    snapshot: CanonicalSnapshot,
    build: EvidencePackBuildResult,
    prior: PriorContextSelection | None = None,
) -> AssembledPacket:
    """Allocate complete pack lines under the request's exact character cap."""
    if request.mode.value == "active":
        raise RetrievalValidationError("active retrieval mode is unavailable in P2.4")
    selection = select_evidence_packs(
        request=request, result=result, snapshot=snapshot, build=build, prior=prior
    )
    # The manifest value is fixed-width, so a placeholder gives exact budget
    # accounting before the final selected identities are known.
    manifest_id = "asm_" + "0" * 32
    chosen_certified: list[EvidencePack] = []
    chosen_temporary: list[EvidencePack] = []
    chosen_permanent: list[EvidencePack] = []
    budget_omitted_ids: list[str] = []
    chosen_pairs: list[tuple[EvidencePack, EvidencePack]] = []

    def counts() -> dict[str, int]:
        return {
            "budget": len(selection.packs)
            - len(chosen_certified)
            - len(chosen_temporary)
            - len(chosen_permanent),
            "duplicate": len(selection.duplicate_ids),
            "provenance": sum(
                omission.reason != "unsafe_content"
                for omission in selection.build_omissions
            ),
            "suppressed": len(selection.suppressed_ids),
            "unsafe": selection.unsafe_records_omitted,
        }

    def render() -> str:
        return render_phase2_packet(
            request=request,
            result=result,
            snapshot=snapshot,
            manifest_id=manifest_id,
            certified_packs=tuple(chosen_certified),
            temporary_packs=tuple(chosen_temporary),
            permanent_packs=tuple(chosen_permanent),
            omission_counts=counts(),
            degraded_reasons=selection.degraded_reasons,
            prompt_evidence=build.prompt_evidence,
        )

    skeleton = render()
    if len(skeleton) > request.max_chars:
        raise PacketBudgetError("immutable Phase 2 protocol exceeds packet limit")
    def choose_fitting_option(
        options: tuple[EvidencePack, ...],
    ) -> EvidencePack | None:
        for option in options:
            target = (
                chosen_certified
                if option.card.certified
                else chosen_temporary
                if option.card.candidate_kind == "temporary"
                else chosen_permanent
            )
            target.append(option)
            if len(render()) <= request.max_chars:
                return option
            target.pop()
        return None

    grouped = tuple(zip(selection.packs, selection.pack_groups, strict=True))
    deferred_permanent: list[tuple[EvidencePack, tuple[EvidencePack, ...]]] = []
    for partition_rank in (0, 1):
        for pack, group in grouped:
            if _partition_rank(pack) != partition_rank:
                continue
            same_partition = tuple(
                option
                for option in group
                if _partition_rank(option) == partition_rank
            )
            chosen = choose_fitting_option(same_partition)
            if chosen is not None:
                chosen_pairs.append((pack, chosen))
                continue
            lower_partition = tuple(
                option
                for option in group
                if _partition_rank(option) > partition_rank
            )
            if lower_partition:
                deferred_permanent.append((pack, lower_partition))
                continue
            if pack.card.certified:
                raise PacketBudgetError(
                    "certified Constitution does not fit packet limit"
                )

    permanent_groups = tuple(
        (pack, group)
        for pack, group in grouped
        if _partition_rank(pack) == 2
    )
    permanent_priority = {
        pack.card.identifier: index
        for index, pack in enumerate(
            _diversify_permanent(
                tuple(
                    pack
                    for pack in build.packs
                    if not pack.card.certified
                    and pack.card.candidate_kind == "record"
                ),
                result,
            )
        )
    }
    pending_permanent = sorted(
        (*deferred_permanent, *permanent_groups),
        key=lambda item: min(
            permanent_priority.get(
                option.card.identifier, len(permanent_priority)
            )
            for option in item[1]
        ),
    )
    for pack, group in pending_permanent:
        chosen = choose_fitting_option(group)
        if chosen is not None:
            chosen_pairs.append((pack, chosen))

    represented_ids = {
        preferred.card.identifier for preferred, _chosen in chosen_pairs
    }
    budget_omitted_ids = [
        pack.card.identifier
        for pack in selection.packs
        if pack.card.identifier not in represented_ids
    ]

    duplicate_ids = list(selection.duplicate_ids)
    for preferred, chosen in chosen_pairs:
        if preferred is chosen:
            continue
        replacement_index = duplicate_ids.index(chosen.card.identifier)
        duplicate_ids[replacement_index] = preferred.card.identifier
    final_packs = tuple(
        (*chosen_certified, *chosen_temporary, *chosen_permanent)
    )
    final_selection = EvidenceSelection(
        packs=final_packs,
        pack_groups=tuple((pack,) for pack in final_packs),
        duplicate_ids=tuple(duplicate_ids),
        suppressed_ids=selection.suppressed_ids,
        build_omissions=selection.build_omissions,
        unsafe_records_omitted=selection.unsafe_records_omitted,
        degraded_reasons=selection.degraded_reasons,
    )
    manifest_id = _assembly_manifest_id(
        request=request,
        result=result,
        snapshot=snapshot,
        selection=final_selection,
        prior=prior,
        prompt_evidence=build.prompt_evidence,
        budget_omitted_ids=tuple(budget_omitted_ids),
    )
    packet = render()
    if len(packet) > request.max_chars:
        raise PacketBudgetError("Phase 2 packet budget reservation failed")
    selected = tuple(
        SelectionIdentity(
            pack.card.identifier,
            pack.card.content_sha256,
            pack.card.last_event.event_sequence,
        )
        for pack in final_packs
    )
    audit_snapshot = PacketAuditSnapshot(
        canonical_snapshot=snapshot,
        query_candidates=tuple(_freeze_audit(item.to_dict()) for item in result.candidates),
        query_trace=tuple(_freeze_audit(item.to_dict()) for item in result.trace),
        prompt_evidence=(
            None
            if build.prompt_evidence is None
            else _freeze_audit(build.prompt_evidence.to_dict())
        ),
        evidence_packs=tuple(_freeze_audit(pack.to_dict()) for pack in final_packs),
        final_selected=tuple(_freeze_audit(item.to_dict()) for item in selected),
        budget_omitted_ids=tuple(budget_omitted_ids),
        build_omissions=tuple(_freeze_audit(item.to_dict()) for item in selection.build_omissions),
        duplicate_ids=tuple(duplicate_ids),
        suppressed_ids=selection.suppressed_ids,
        omission_counts=tuple(sorted(counts().items())),
        degraded_reasons=selection.degraded_reasons,
        request_binding=_request_audit_binding(request),
        result_metadata=_result_audit_metadata(result),
    )
    return AssembledPacket(
        packet=packet,
        manifest_id=manifest_id,
        packet_sha256=hashlib.sha256(packet.encode("utf-8")).hexdigest(),
        packet_chars=len(packet),
        selected=selected,
        budget_omitted_ids=tuple(budget_omitted_ids),
        build_omissions=selection.build_omissions,
        duplicate_ids=tuple(duplicate_ids),
        suppressed_ids=selection.suppressed_ids,
        omission_counts=tuple(sorted(counts().items())),
        degraded_reasons=selection.degraded_reasons,
        audit_snapshot=audit_snapshot,
    )


def _freeze_audit(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_audit(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_audit(item) for item in value)
    return value


def _request_audit_binding(request: RetrievalRequest) -> dict[str, object]:
    from codex_memory.ledger_common import canonical_json

    return {
        "query_sha256": hashlib.sha256(canonical_json(request.query).encode("utf-8")).hexdigest(),
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "context_id": request.context_id,
        "prompt_event_uri": request.prompt_event_uri,
        "scope": request.scope.to_dict(),
        "epistemic_context": request.epistemic_context.to_dict(),
        "max_chars": request.max_chars,
        "mode": request.mode.value,
    }


def _result_audit_metadata(result: RetrievalQueryResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "mode": result.mode.value,
        "generation": None if result.generation is None else result.generation.to_dict(),
        "canonical_highwater": None if result.canonical_highwater is None else result.canonical_highwater.to_dict(),
        "indexed_highwater": None if result.indexed_highwater is None else result.indexed_highwater.to_dict(),
        "index_stale": result.index_stale,
        "degraded_reasons": [reason.value for reason in result.degraded_reasons],
    }


def _assembly_manifest_id(
    *,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    snapshot: CanonicalSnapshot,
    selection: EvidenceSelection,
    prior: PriorContextSelection | None,
    prompt_evidence: SafeEvidence | None,
    budget_omitted_ids: tuple[str, ...],
) -> str:
    from codex_memory.ledger_common import canonical_json

    payload = {
        "budget_omitted_ids": list(budget_omitted_ids),
        "canonical_content_fingerprint": snapshot.content_fingerprint,
        "canonical_highwater": snapshot.highwater.to_dict(),
        "context_id": request.context_id,
        "degraded_reasons": list(selection.degraded_reasons),
        "duplicate_ids": list(selection.duplicate_ids),
        "epistemic_context": request.epistemic_context.to_dict(),
        "format": "codex-memory.phase2-assembly.v1",
        "generation_id": (
            None if result.generation is None else result.generation.generation_id
        ),
        "index_stale": result.index_stale,
        "indexed_highwater": (
            None
            if result.indexed_highwater is None
            else result.indexed_highwater.to_dict()
        ),
        "max_chars": request.max_chars,
        "mode": request.mode.value,
        "packs": [
            {
                "candidate_id": pack.card.identifier,
                "pack_sha256": pack.pack_sha256,
            }
            for pack in selection.packs
        ],
        "provenance_omissions": [
            omission.to_dict() for omission in selection.build_omissions
        ],
        "prompt_evidence": (
            None if prompt_evidence is None else prompt_evidence.to_dict()
        ),
        "prior_manifest_id": None if prior is None else prior.manifest_id,
        "query_sha256": hashlib.sha256(
            (request.query or "").encode("utf-8")
        ).hexdigest(),
        "scope": request.scope.to_dict(),
        "session_id": request.session_id,
        "status": result.status.value,
        "suppressed_ids": list(selection.suppressed_ids),
        "turn_id": request.turn_id,
        "unsafe_records_omitted": selection.unsafe_records_omitted,
    }
    return "asm_" + hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()[:32]


def _prior_is_usable(
    prior: PriorContextSelection | None,
    request: RetrievalRequest,
    result: RetrievalQueryResult,
    snapshot: CanonicalSnapshot,
    *,
    degraded_reasons: tuple[str, ...],
) -> bool:
    if prior is None or not prior.committed:
        return False
    if prior.session_id != request.session_id or prior.context_id != request.context_id:
        return False
    if prior.scope != request.scope or prior.epistemic_context != request.epistemic_context:
        return False
    if prior.canonical_content_fingerprint != snapshot.content_fingerprint:
        return False
    if (
        prior.index_stale != result.index_stale
        or prior.degraded_reasons != degraded_reasons
    ):
        return False
    current = snapshot.highwater
    previous = prior.canonical_highwater
    return (
        previous.memory_event_seq == current.memory_event_seq
        and previous.relation_seq == current.relation_seq
        and previous.candidate_event_seq == current.candidate_event_seq
    )


def _diversify_permanent(
    packs: tuple[EvidencePack, ...], result: RetrievalQueryResult
) -> tuple[EvidencePack, ...]:
    """Keep exact priority, then round-robin ordinary candidates by topic."""
    exact_ids = {
        trace.candidate_id
        for trace in result.trace
        if CandidateChannel.EXACT in trace.source_channels
    }
    exact = tuple(pack for pack in packs if pack.card.identifier in exact_ids)
    ordinary = tuple(pack for pack in packs if pack.card.identifier not in exact_ids)
    buckets: dict[str, list[EvidencePack]] = {}
    for pack in ordinary:
        buckets.setdefault(pack.card.topic_key, []).append(pack)
    diversified: list[EvidencePack] = []
    while buckets:
        exhausted: list[str] = []
        for topic, topic_packs in buckets.items():
            diversified.append(topic_packs.pop(0))
            if not topic_packs:
                exhausted.append(topic)
        for topic in exhausted:
            del buckets[topic]
    return (*exact, *diversified)
