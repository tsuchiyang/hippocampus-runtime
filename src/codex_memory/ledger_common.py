from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from typing import Any, Sequence
import unicodedata
from urllib.parse import urlsplit


_RECORD_TYPES = frozenset(
    {"constitution", "decision", "assumption", "rejected_approach", "failure"}
)
_SCOPE_KINDS = frozenset({"project", "workspace"})
_TOPIC_PATTERN = re.compile(r"[a-z0-9_-]+(?:[./][a-z0-9_-]+)*\Z", re.ASCII)
_URI_SCHEME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_MALFORMED_PERCENT_PATTERN = re.compile(r"%(?![0-9A-Fa-f]{2})")
_AUTHORITY_REQUIRED_SCHEMES = frozenset({"http", "https", "ssh", "git+ssh"})
_EXECUTABLE_EVIDENCE_SCHEMES = frozenset({"javascript", "vbscript", "data"})
_MAX_JSON_DEPTH = 100


class LedgerError(RuntimeError):
    """Base class for bounded structured-ledger domain failures."""


class ValidationError(LedgerError):
    """Raised when caller-authored content violates the frozen format."""


class IdempotencyConflict(LedgerError):
    """Raised when an idempotency key is reused for different input."""


class RecordNotFound(LedgerError):
    """Raised when a requested structured record does not exist."""


class InvalidTransition(LedgerError):
    """Raised when a lifecycle action is invalid for the current state."""


class VerificationError(LedgerError):
    """Raised when immutable ledger data or its projection is inconsistent."""


@dataclass(frozen=True, slots=True)
class ValidatedRecordContent:
    value_json: str
    content_sha256: str


def canonical_json(value: Any) -> str:
    """Return the ledger's exact UTF-8 canonical JSON representation."""
    _require_string_object_keys(value)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        serialized.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("value is not canonical JSON") from exc
    return serialized


def parse_json_value(raw: str) -> Any:
    """Parse exactly one strict JSON value for the CLI's value boundary."""
    if not isinstance(raw, str):
        raise ValidationError("value JSON must be text")

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite number")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        return number

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
        return result

    start = len(raw) - len(raw.lstrip())
    if start == len(raw):
        raise ValidationError("empty value JSON")
    decoder = json.JSONDecoder(
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )
    try:
        value, end = decoder.raw_decode(raw, start)
        if raw[end:].strip():
            raise ValueError("trailing input")
        canonical_json(value)
    except (ValueError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError("invalid value JSON") from exc
    return value


def validate_record_content(
    *,
    record_type: str,
    scope_kind: str,
    scope_key: str,
    topic_key: str,
    statement: str,
    value: Any,
) -> ValidatedRecordContent:
    if not isinstance(record_type, str) or record_type not in _RECORD_TYPES:
        raise ValidationError("invalid record type")
    if not isinstance(scope_kind, str) or scope_kind not in _SCOPE_KINDS:
        raise ValidationError("invalid scope kind")
    _require_bounded_text(scope_key, "scope key", 256, nonblank=True)
    if not isinstance(topic_key, str):
        raise ValidationError("invalid topic key")
    try:
        topic_bytes = topic_key.encode("ascii")
    except (UnicodeError, AttributeError) as exc:
        raise ValidationError("invalid topic key") from exc
    if len(topic_bytes) > 128 or _TOPIC_PATTERN.fullmatch(topic_key) is None:
        raise ValidationError("invalid topic key")
    _require_bounded_text(statement, "statement", 8192, nonblank=True)
    value_json = canonical_json(value)
    if len(value_json.encode("utf-8")) > 32768:
        raise ValidationError("value JSON is too large")
    content_sha256 = _hash_preimage(
        {
            "record_type": record_type,
            "scope": {"kind": scope_kind, "key": scope_key},
            "statement": statement,
            "topic_key": topic_key,
            "value": value,
        }
    )
    return ValidatedRecordContent(
        value_json=value_json,
        content_sha256=content_sha256,
    )


def validate_evidence_uris(
    evidence_uris: Sequence[str], *, maximum: int | None = None
) -> tuple[str, ...]:
    if isinstance(evidence_uris, (str, bytes)):
        raise ValidationError("evidence must be a URI sequence")
    try:
        evidence = tuple(evidence_uris)
    except TypeError as exc:
        raise ValidationError("evidence must be a URI sequence") from exc
    if maximum is not None and (type(maximum) is not int or len(evidence) > maximum):
        raise ValidationError("too many evidence URIs")
    for uri in evidence:
        _validate_uri(uri)
    return evidence


def validate_new_evidence_uris(
    evidence_uris: Sequence[str], *, maximum: int | None = None,
) -> tuple[str, ...]:
    """Apply ingress safety without making immutable historical evidence invalid."""
    evidence = validate_evidence_uris(evidence_uris, maximum=maximum)
    if any(urlsplit(uri).scheme.lower() in _EXECUTABLE_EVIDENCE_SCHEMES for uri in evidence):
        raise ValidationError("executable evidence URI is not allowed")
    return evidence


def scope_keys_equal(left: str, right: str, *, windows: bool | None = None) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValidationError("scope keys must be text")
    if windows is None:
        windows = os.name == "nt"
    if not isinstance(windows, bool):
        raise ValidationError("windows flag must be boolean")
    return left.casefold() == right.casefold() if windows else left == right


def _validate_idempotency_key(value: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValidationError("invalid idempotency key")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValidationError("invalid idempotency key")


def _validate_uri(uri: str) -> None:
    if not isinstance(uri, str):
        raise ValidationError("invalid evidence URI")
    try:
        encoded = uri.encode("utf-8")
    except UnicodeError as exc:
        raise ValidationError("invalid evidence URI") from exc
    if not encoded or len(encoded) > 2048:
        raise ValidationError("invalid evidence URI")
    match = _URI_SCHEME_PATTERN.match(uri)
    if match is None or match.end() == len(uri):
        raise ValidationError("invalid evidence URI")
    if "\\" in uri or any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in uri
    ):
        raise ValidationError("invalid evidence URI")
    if _MALFORMED_PERCENT_PATTERN.search(uri):
        raise ValidationError("invalid evidence URI")
    scheme = uri[: match.end() - 1].lower()
    scheme_specific = uri[match.end() :]
    main_component = scheme_specific.split("#", 1)[0].split("?", 1)[0]
    if not main_component:
        raise ValidationError("invalid evidence URI")
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise ValidationError("invalid evidence URI") from exc
    if scheme_specific.startswith("//"):
        if scheme == "file":
            if not parsed.path.startswith("/") or parsed.path == "/":
                raise ValidationError("invalid evidence URI")
        elif not parsed.netloc:
            raise ValidationError("invalid evidence URI")
    elif scheme in _AUTHORITY_REQUIRED_SCHEMES:
        raise ValidationError("invalid evidence URI")


def _require_bounded_text(value: str, name: str, maximum: int, *, nonblank: bool) -> None:
    if not isinstance(value, str) or (nonblank and not value.strip()):
        raise ValidationError(f"invalid {name}")
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ValidationError(f"invalid {name}") from exc
    if length > maximum:
        raise ValidationError(f"invalid {name}")


def _require_string_object_keys(
    value: Any, active: set[int] | None = None, depth: int = 0
) -> None:
    if not isinstance(value, (dict, list, tuple)):
        return
    if depth > _MAX_JSON_DEPTH:
        raise ValidationError("JSON value is too deeply nested")
    ancestors = set() if active is None else active
    identity = id(value)
    if identity in ancestors:
        raise ValidationError("circular JSON value")
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValidationError("JSON object keys must be strings")
                _require_string_object_keys(child, ancestors, depth + 1)
        else:
            for child in value:
                _require_string_object_keys(child, ancestors, depth + 1)
    finally:
        ancestors.remove(identity)


def _hash_preimage(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
