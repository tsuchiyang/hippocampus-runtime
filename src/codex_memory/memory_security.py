from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from codex_memory.ledger_common import canonical_json


_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_BARE_VALUE = (
    r"(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}[^,};\r\n'\"]*|[^,};\r\n'\"]+)"
)
_BEARER = re.compile(
    r"\bauthorization\s*:\s*bearer\s*(?:"
    r'"(?P<double>(?:\\.|[^"\\])*)"|'
    r"'(?P<single>(?:\\.|[^'\\])*)'|"
    rf"(?P<bare>{_BARE_VALUE}))",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"\b(?:[a-z0-9_]*?(?:password|token|api_key|secret)[a-z0-9_]*)"
    r"\s*['\"]?\s*[:=]\s*(?:"
    r'"(?P<double>(?:\\.|[^"\\])*)"|'
    r"'(?P<single>(?:\\.|[^'\\])*)'|"
    rf"(?P<bare>{_BARE_VALUE}))",
    re.IGNORECASE,
)
_PREFIXES = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-|ghp_|github_pat_|xoxb-)[A-Za-z0-9_-]+",
    re.IGNORECASE,
)
_AWS_ACCESS_KEY_ID = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
_PLACEHOLDER_VALUES = frozenset(
    {"<redacted>", "[redacted]", "${token}", "***", "example", "dummy", "test-only"}
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    category: str
    location: str


class SecretRejected(ValueError):
    """Raised when candidate content contains a deterministic secret pattern."""


def find_secrets(*, statement: str, value: Any) -> tuple[SecretFinding, ...]:
    """Return non-sensitive findings for statement and canonical value content."""
    findings = list(_find_in_text(statement, "statement"))
    findings.extend(_find_in_text(canonical_json(value), "value"))
    return tuple(findings)


def reject_secrets(*, statement: str, value: Any) -> None:
    """Raise a bounded error without exposing the matched input."""
    findings = find_secrets(statement=statement, value=value)
    if findings:
        finding = findings[0]
        raise SecretRejected(
            f"memory content rejected: {finding.category} in {finding.location}"
        )


def _find_in_text(text: str, location: str) -> tuple[SecretFinding, ...]:
    if not isinstance(text, str):
        return (SecretFinding("invalid_content", location),)
    findings: list[SecretFinding] = []
    if _PEM_PRIVATE_KEY.search(text):
        findings.append(SecretFinding("private_key", location))
    for bearer in _BEARER.finditer(text):
        if not _is_placeholder(_matched_value(bearer)):
            findings.append(SecretFinding("bearer_token", location))
    for assignment in _ASSIGNMENT.finditer(text):
        if not _is_placeholder(_matched_value(assignment)):
            findings.append(SecretFinding("credential_assignment", location))
    if _PREFIXES.search(text):
        findings.append(SecretFinding("credential_prefix", location))
    if _AWS_ACCESS_KEY_ID.search(text):
        findings.append(SecretFinding("aws_access_key_id", location))
    return tuple(findings)


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDER_VALUES


def _matched_value(match: re.Match[str]) -> str:
    for name in ("double", "single", "bare"):
        value = match.group(name)
        if value is not None:
            return value
    return ""
