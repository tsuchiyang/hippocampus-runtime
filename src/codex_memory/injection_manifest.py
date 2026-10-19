"""Strict, immutable codec for content-addressed injection manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from codex_memory.ledger_common import ValidationError, canonical_json
from codex_memory.memory_evidence import parse_event_uri
from codex_memory.retrieval.contracts import RetrievalQueryResult
from codex_memory.retrieval.errors import RetrievalValidationError


_MAX_MANIFEST_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_BODY_FIELDS = frozenset({"format", "source", "request", "snapshot", "identities", "candidates", "evidence", "packets", "timing", "delivery"})
_GROUP_FIELDS = {
    "source": frozenset({"event_id", "event_uri", "event_blob_sha256", "session_id", "turn_id", "context_id", "origin"}),
    "request": frozenset({"project_scope_key", "workspace_scope_keys", "actor_id", "role", "memory_view_id", "epistemic_policy_id", "truth_domain", "query_sha256", "max_chars", "mode"}),
    "snapshot": frozenset({"canonical_highwater", "indexed_highwater", "canonical_content_fingerprint", "generation", "index_stale", "degraded_reasons", "requirement_version"}),
    "identities": frozenset({"provider", "model", "preprocessing", "config_sha256", "implementation_sha256"}),
    "candidates": frozenset({"query_result", "final_selected", "budget_omitted_ids", "build_omissions", "duplicate_ids", "suppressed_ids", "omission_counts"}),
    "evidence": frozenset({"packs"}),
    "packets": frozenset({"prepared_packet_ref", "packet_chars", "assembly_id", "legacy_additional_context_ref", "serialized_response_ref"}),
    "timing": frozenset({"started_at", "prepared_at", "query_duration_ms", "assembly_duration_ms", "prepare_duration_ms", "budget_ms"}),
    "delivery": frozenset({"v2", "legacy", "host_acknowledgement"}),
}
_HIGHWATER_FIELDS = frozenset({"record_seq", "memory_event_seq", "evidence_seq", "relation_seq", "candidate_seq", "candidate_event_seq", "candidate_evidence_seq"})
_IDENTITY_FIELDS = frozenset({"configured", "observed", "validated"})
_BLOB_REF_FIELDS = frozenset({"sha256", "byte_size", "relative_path"})
_SELECTION_FIELDS = frozenset({"candidate_id", "content_sha256", "last_event_sequence"})
_OMISSION_FIELDS = frozenset({"candidate_id", "reason"})
_OMISSION_COUNT_FIELDS = frozenset({"budget", "duplicate", "provenance", "suppressed", "unsafe"})
_PACK_REQUIRED_FIELDS = frozenset({"card", "evidence", "pack_version", "provenance_status"})
_PACK_OPTIONAL_FIELDS = frozenset({"promotions", "candidate_provenance", "provenance_omissions", "supersession_relations"})
_CARD_FIELDS = frozenset({"candidate_kind", "certified", "content_sha256", "id", "kind", "last_event", "scope", "sequence", "state", "statement", "topic", "trust_label", "value"})
_EVENT_FIELDS = frozenset({"action", "created_at", "event_id", "event_sequence", "event_sha256"})
_SAFE_EVIDENCE_FIELDS = frozenset({"event_created_at", "event_name", "event_session_id", "event_turn_id", "payload_blob_sha256", "reference", "reference_sha256"})
_PROVENANCE_FIELDS = frozenset({"candidate_content_sha256", "candidate_id", "capture_event", "capture_evidence_refs", "origin", "reviews"})
_ORIGIN_FIELDS = frozenset({"context_id", "session_id", "turn_id"})
_REVIEW_FIELDS = frozenset({"event", "evidence_refs", "permanent_record_created", "permanent_record_id", "policy_code", "policy_version", "reason", "requested_outcome", "superseded_record_id"})
_RELATION_FIELDS = frozenset({"created_at", "lifecycle_event_id", "relation_id", "relation_sequence", "relation_sha256", "relation_type", "source_record_id", "target_record_id"})


class InjectionAuditReason(StrEnum):
    INVALID_INPUT = "invalid_input"
    MISSING = "missing"
    TAMPERED = "tampered"
    UNSUPPORTED = "unsupported"
    PATH_UNSAFE = "path_unsafe"
    UNAVAILABLE = "unavailable"
    DEADLINE_EXHAUSTED = "deadline_exhausted"
    SOURCE_MISMATCH = "source_mismatch"
    SIZE_LIMIT = "size_limit"


class InjectionAuditError(RuntimeError):
    """A bounded audit failure with a stable, non-sensitive public reason."""

    def __init__(self, reason: InjectionAuditReason | str, message: str | None = None, *, cause: BaseException | None = None) -> None:
        self.reason = InjectionAuditReason(reason)
        self.cause = cause
        super().__init__(message or self.reason.value)
        if cause is not None:
            self.__cause__ = cause


def _strict_decode(raw: bytes) -> Any:
    if not isinstance(raw, bytes):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest bytes are invalid")
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise InjectionAuditError(InjectionAuditReason.SIZE_LIMIT, "manifest exceeds size limit")

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite number")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite number")
        return result

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        decoder = json.JSONDecoder(parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=unique_object)
        start = 0
        while start < len(text) and text[start] in " \t\r\n":
            start += 1
        value, end = decoder.raw_decode(text, start)
        if any(char not in " \t\r\n" for char in text[end:]):
            raise ValueError("trailing input")
        canonical_json(value)
        return value
    except (UnicodeError, ValueError, TypeError, RecursionError, json.JSONDecodeError, ValidationError) as exc:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest is not strict JSON", cause=exc) from exc


def _require_object(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} must be an object")
    if set(value) != fields:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} has unknown or missing fields")
    return value


def _validate_body(value: object) -> dict[str, Any]:
    body = _require_object(value, _BODY_FIELDS, "manifest body")
    if body["format"] != "codex-memory.injection.v1":
        raise InjectionAuditError(InjectionAuditReason.UNSUPPORTED, "manifest format is unsupported")
    for name, fields in _GROUP_FIELDS.items():
        _require_object(body[name], fields, name)
    request = body["request"]
    if request["truth_domain"] != "ground_truth" or request["memory_view_id"] != "ground_truth.default":
        raise InjectionAuditError(InjectionAuditReason.UNSUPPORTED, "manifest epistemic domain is unsupported")
    if not isinstance(request["mode"], str) or request["mode"] not in {"off", "shadow"}:
        raise InjectionAuditError(InjectionAuditReason.UNSUPPORTED, "manifest retrieval mode is unsupported")
    if not isinstance(body["source"]["origin"], str) or body["source"]["origin"] not in {"hook", "offline"}:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "source origin is invalid")
    if body["delivery"]["v2"] != "prepared":
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "v2 delivery state is invalid")
    snapshot = body["snapshot"]
    _require_object(snapshot["canonical_highwater"], _HIGHWATER_FIELDS, "canonical_highwater")
    if snapshot["indexed_highwater"] is not None:
        _require_object(snapshot["indexed_highwater"], _HIGHWATER_FIELDS, "indexed_highwater")
    identities = body["identities"]
    for name in ("provider", "model", "preprocessing"):
        _require_object(identities[name], _IDENTITY_FIELDS, name)
    packets = body["packets"]
    for name in ("prepared_packet_ref", "legacy_additional_context_ref", "serialized_response_ref"):
        _require_object(packets[name], _BLOB_REF_FIELDS, name)
    candidates = body["candidates"]
    try:
        RetrievalQueryResult.from_dict(candidates["query_result"])
    except (RetrievalValidationError, TypeError, ValueError) as exc:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "query_result is invalid", cause=exc) from exc
    if not isinstance(candidates["final_selected"], list):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "final_selected must be a list")
    for item in candidates["final_selected"]:
        _require_object(item, _SELECTION_FIELDS, "final selection")
        _require_text(item["candidate_id"], "final selection candidate_id")
        _require_hash(item["content_sha256"], "final selection content_sha256")
        _require_integer(item["last_event_sequence"], "final selection last_event_sequence")
    if not isinstance(candidates["build_omissions"], list):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "build_omissions must be a list")
    for item in candidates["build_omissions"]:
        _require_object(item, _OMISSION_FIELDS, "build omission")
        _require_text(item["candidate_id"], "build omission candidate_id")
        _require_text(item["reason"], "build omission reason")
    omission_counts = _require_object(candidates["omission_counts"], _OMISSION_COUNT_FIELDS, "omission_counts")
    for name, count in omission_counts.items():
        _require_integer(count, f"omission_counts.{name}")
    packs = body["evidence"]["packs"]
    if not isinstance(packs, list):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence packs must be a list")
    for pack in packs:
        _validate_evidence_pack(pack)
    _validate_scalar_contract(body)
    try:
        canonical_json(body)
    except ValidationError as exc:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest body is not canonical JSON", cause=exc) from exc
    return body


def _require_text(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")


def _require_hash(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")


def _require_integer(value: object, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")


def _require_string_list(value: object, name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")


def _require_timestamp(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.endswith("Z"):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid", cause=exc) from exc


def _validate_event(value: object, name: str) -> None:
    event = _require_object(value, _EVENT_FIELDS, name)
    for field in ("action", "event_id"):
        _require_text(event[field], f"{name}.{field}")
    _require_integer(event["event_sequence"], f"{name}.event_sequence")
    _require_hash(event["event_sha256"], f"{name}.event_sha256")
    _require_timestamp(event["created_at"], f"{name}.created_at")


def _validate_provenance(value: object, name: str) -> None:
    provenance = _require_object(value, _PROVENANCE_FIELDS, name)
    _require_text(provenance["candidate_id"], f"{name}.candidate_id")
    _require_hash(provenance["candidate_content_sha256"], f"{name}.candidate_content_sha256")
    _validate_event(provenance["capture_event"], f"{name}.capture_event")
    _require_string_list(provenance["capture_evidence_refs"], f"{name}.capture_evidence_refs")
    for digest in provenance["capture_evidence_refs"]:
        _require_hash(digest, f"{name}.capture_evidence_refs")
    origin = _require_object(provenance["origin"], _ORIGIN_FIELDS, f"{name}.origin")
    for field in _ORIGIN_FIELDS:
        _require_text(origin[field], f"{name}.origin.{field}")
    if not isinstance(provenance["reviews"], list):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name}.reviews is invalid")
    for raw_review in provenance["reviews"]:
        review = _require_object(raw_review, _REVIEW_FIELDS, f"{name}.review")
        _validate_event(review["event"], f"{name}.review.event")
        _require_string_list(review["evidence_refs"], f"{name}.review.evidence_refs")
        for digest in review["evidence_refs"]:
            _require_hash(digest, f"{name}.review.evidence_refs")
        for field in ("policy_code", "policy_version", "reason", "requested_outcome"):
            _require_text(review[field], f"{name}.review.{field}")
        if review["permanent_record_created"] is not None and not isinstance(review["permanent_record_created"], bool):
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name}.review.permanent_record_created is invalid")
        for field in ("permanent_record_id", "superseded_record_id"):
            _require_text(review[field], f"{name}.review.{field}", nullable=True)


def _validate_evidence_pack(value: object) -> None:
    if not isinstance(value, dict) or not _PACK_REQUIRED_FIELDS.issubset(value) or not set(value).issubset(_PACK_REQUIRED_FIELDS | _PACK_OPTIONAL_FIELDS):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence pack has unknown or missing fields")
    if type(value["pack_version"]) is not int or value["pack_version"] != 1:
        raise InjectionAuditError(InjectionAuditReason.UNSUPPORTED, "evidence pack version is unsupported")
    _require_text(value["provenance_status"], "evidence pack provenance_status")
    card = _require_object(value["card"], _CARD_FIELDS, "evidence card")
    for field in ("candidate_kind", "id", "kind", "state", "statement", "topic", "trust_label"):
        _require_text(card[field], f"evidence card.{field}")
    if not isinstance(card["certified"], bool):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence card.certified is invalid")
    _require_hash(card["content_sha256"], "evidence card.content_sha256")
    _require_integer(card["sequence"], "evidence card.sequence")
    scope = _require_object(card["scope"], frozenset({"key", "kind"}), "evidence card.scope")
    _require_text(scope["key"], "evidence card.scope.key")
    _require_text(scope["kind"], "evidence card.scope.kind")
    _validate_event(card["last_event"], "evidence card.last_event")
    if not isinstance(value["evidence"], list):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence registry is invalid")
    for raw_evidence in value["evidence"]:
        evidence = _require_object(raw_evidence, _SAFE_EVIDENCE_FIELDS, "safe evidence")
        _require_text(evidence["reference"], "safe evidence.reference")
        _require_hash(evidence["reference_sha256"], "safe evidence.reference_sha256")
        _require_hash(evidence["payload_blob_sha256"], "safe evidence.payload_blob_sha256", nullable=True)
        for field in ("event_session_id", "event_turn_id", "event_name"):
            _require_text(evidence[field], f"safe evidence.{field}", nullable=True)
        _require_timestamp(evidence["event_created_at"], "safe evidence.event_created_at", nullable=True)
    for field in ("promotions",):
        if field in value:
            if not isinstance(value[field], list):
                raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"evidence pack.{field} is invalid")
            for item in value[field]:
                _validate_provenance(item, f"evidence pack.{field}")
    if "candidate_provenance" in value:
        _validate_provenance(value["candidate_provenance"], "evidence pack.candidate_provenance")
    if "provenance_omissions" in value:
        _require_string_list(value["provenance_omissions"], "evidence pack.provenance_omissions")
    if "supersession_relations" in value:
        if not isinstance(value["supersession_relations"], list):
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence pack.supersession_relations is invalid")
        for raw_relation in value["supersession_relations"]:
            relation = _require_object(raw_relation, _RELATION_FIELDS, "supersession relation")
            for field in ("lifecycle_event_id", "relation_id", "relation_type", "source_record_id", "target_record_id"):
                _require_text(relation[field], f"supersession relation.{field}")
            _require_integer(relation["relation_sequence"], "supersession relation.relation_sequence")
            _require_hash(relation["relation_sha256"], "supersession relation.relation_sha256")
            _require_timestamp(relation["created_at"], "supersession relation.created_at")


def _validate_scalar_contract(body: dict[str, Any]) -> None:
    source = body["source"]
    if isinstance(source["event_id"], bool) or not isinstance(source["event_id"], int) or source["event_id"] < 1:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "source event ID is invalid")
    try:
        uri_session, uri_event_id = parse_event_uri(source["event_uri"])
    except ValueError as exc:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "source event URI is invalid", cause=exc) from exc
    if uri_session != source["session_id"] or uri_event_id != source["event_id"]:
        raise InjectionAuditError(InjectionAuditReason.SOURCE_MISMATCH, "source event URI does not match source identity")
    for name in ("session_id", "turn_id", "context_id"):
        if not isinstance(source[name], str) or not source[name]:
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"source {name} is invalid")
    request = body["request"]
    for name in ("project_scope_key", "actor_id", "role", "memory_view_id", "epistemic_policy_id", "truth_domain"):
        _require_text(request[name], f"request.{name}")
    snapshot = body["snapshot"]
    _require_text(snapshot["requirement_version"], "snapshot.requirement_version")
    _require_string_list(snapshot["degraded_reasons"], "snapshot.degraded_reasons")
    for name in ("provider", "model", "preprocessing"):
        for state in _IDENTITY_FIELDS:
            _require_text(body["identities"][name][state], f"identities.{name}.{state}")
    _require_text(body["packets"]["assembly_id"], "packets.assembly_id")
    _require_timestamp(body["timing"]["started_at"], "timing.started_at")
    _require_timestamp(body["timing"]["prepared_at"], "timing.prepared_at")
    hash_fields = (
        body["source"]["event_blob_sha256"],
        body["request"]["query_sha256"],
        body["snapshot"]["canonical_content_fingerprint"],
        body["identities"]["config_sha256"],
        body["identities"]["implementation_sha256"],
        *(body["packets"][name]["sha256"] for name in ("prepared_packet_ref", "legacy_additional_context_ref", "serialized_response_ref")),
    )
    if any(not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value) for value in hash_fields):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest sha256 field is invalid")
    integer_fields = (
        body["request"]["max_chars"], body["packets"]["packet_chars"],
        *(body["timing"][name] for name in ("query_duration_ms", "assembly_duration_ms", "prepare_duration_ms", "budget_ms")),
        *(body["packets"][name]["byte_size"] for name in ("prepared_packet_ref", "legacy_additional_context_ref", "serialized_response_ref")),
        *(value for highwater_name in ("canonical_highwater", "indexed_highwater") if (highwater := body["snapshot"][highwater_name]) is not None for value in highwater.values()),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integer_fields):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest integer field is invalid")
    if not 1 <= body["request"]["max_chars"] <= 4000:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "max_chars is invalid")
    for name in ("prepared_packet_ref", "legacy_additional_context_ref", "serialized_response_ref"):
        ref = body["packets"][name]
        expected = f"sha256/{ref['sha256'][:2]}/{ref['sha256']}.bin"
        if ref["relative_path"] != expected:
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} path is invalid")
    legacy = body["delivery"]["legacy"]
    if not isinstance(legacy, str) or legacy not in {"emitted", "partial", "unknown"} or body["delivery"]["host_acknowledgement"] != "unknown":
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "delivery state is invalid")
    if not isinstance(body["snapshot"]["index_stale"], bool):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "index_stale is invalid")
    for name in ("workspace_scope_keys",):
        if not isinstance(body["request"][name], list) or any(not isinstance(item, str) or not item for item in body["request"][name]):
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")
    for name in ("budget_omitted_ids", "duplicate_ids", "suppressed_ids"):
        value = body["candidates"][name]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, f"{name} is invalid")
    query_result = RetrievalQueryResult.from_dict(body["candidates"]["query_result"])
    if query_result.mode.value == "active":
        raise InjectionAuditError(InjectionAuditReason.UNSUPPORTED, "query result retrieval mode is unsupported")
    if (
        query_result.canonical_highwater is None
        or query_result.canonical_highwater.to_dict() != snapshot["canonical_highwater"]
        or (None if query_result.indexed_highwater is None else query_result.indexed_highwater.to_dict()) != snapshot["indexed_highwater"]
        or query_result.index_stale is not snapshot["index_stale"]
        or (None if query_result.generation is None else query_result.generation.to_dict()) != snapshot["generation"]
        or [reason.value for reason in query_result.degraded_reasons] != snapshot["degraded_reasons"]
    ):
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "snapshot does not match query_result")
    selected = body["candidates"]["final_selected"]
    packs = body["evidence"]["packs"]
    if [pack["card"]["id"] for pack in packs] != [item["candidate_id"] for item in selected]:
        raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence packs do not match final selection")
    for pack, selection in zip(packs, selected, strict=True):
        if pack["card"]["content_sha256"] != selection["content_sha256"] or pack["card"]["last_event"]["event_sequence"] != selection["last_event_sequence"]:
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "evidence pack identity does not match final selection")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, init=False)
class InjectionManifest:
    body: Mapping[str, object]
    manifest_sha256: str
    injection_id: str

    def __init__(
        self,
        body: dict[str, object],
        manifest_sha256: str | None = None,
        injection_id: str | None = None,
    ) -> None:
        validated = _validate_body(body)
        try:
            copied = json.loads(canonical_json(validated))
            body_bytes = canonical_json(copied).encode("utf-8")
        except (ValidationError, UnicodeError, ValueError, TypeError) as exc:
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest body is invalid", cause=exc) from exc
        digest = hashlib.sha256(body_bytes).hexdigest()
        expected_id = "inj_" + digest
        if manifest_sha256 is not None and manifest_sha256 != digest:
            raise InjectionAuditError(InjectionAuditReason.TAMPERED, "manifest body is tampered")
        if injection_id is not None and injection_id != expected_id:
            raise InjectionAuditError(InjectionAuditReason.TAMPERED, "manifest ID is tampered")
        object.__setattr__(self, "body", _freeze(copied))
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "injection_id", expected_id)
        if len(self.to_bytes()) > _MAX_MANIFEST_BYTES:
            raise InjectionAuditError(InjectionAuditReason.SIZE_LIMIT, "manifest exceeds size limit")

    @classmethod
    def from_body(cls, body: dict[str, object]) -> "InjectionManifest":
        return cls(body)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "InjectionManifest":
        document = _require_object(_strict_decode(raw), frozenset({"body", "manifest_sha256", "injection_id"}), "manifest")
        expected_hash = document["manifest_sha256"]
        expected_id = document["injection_id"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(char not in _HEX for char in expected_hash):
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest digest is invalid")
        if expected_id != "inj_" + expected_hash:
            raise InjectionAuditError(InjectionAuditReason.TAMPERED, "manifest ID is tampered")
        return cls(document["body"], expected_hash, expected_id)

    def to_bytes(self) -> bytes:
        document = {"body": _thaw(self.body), "manifest_sha256": self.manifest_sha256, "injection_id": self.injection_id}
        try:
            return canonical_json(document).encode("utf-8")
        except (ValidationError, UnicodeError) as exc:  # defensive against forged instances
            raise InjectionAuditError(InjectionAuditReason.INVALID_INPUT, "manifest is invalid", cause=exc) from exc
