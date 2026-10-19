from __future__ import annotations

import importlib.util
from importlib import import_module

import pytest


def _security():
    spec = importlib.util.find_spec("codex_memory.memory_security")
    assert spec is not None, "memory security module is required"
    module = import_module("codex_memory.memory_security")
    for name in ("SecretFinding", "SecretRejected", "find_secrets", "reject_secrets"):
        assert hasattr(module, name), f"memory security API is missing {name}"
    return module


@pytest.mark.parametrize(
    ("statement", "value", "location"),
    [
        ("-----BEGIN SYNTHETIC PRIVATE KEY-----\nabc\n-----END SYNTHETIC PRIVATE KEY-----", None, "statement"),
        ("Authorization: Bearer synthetic-token-value", None, "statement"),
        ("password = synthetic-token-value", None, "statement"),
        ("token = synthetic-token-value", None, "statement"),
        ("api_key = synthetic-token-value", None, "statement"),
        ("secret = synthetic-token-value", None, "statement"),
        ("sk-synthetic-token-value", None, "statement"),
        ("github_pat_syntheticvalue", None, "statement"),
        ("xoxb-syntheticvalue", None, "statement"),
        ("ordinary text", {"nested": {"credential": "ghp_syntheticvalue"}}, "value"),
        ("ordinary text", {"nested": {"credential": ('A' + 'KIAABCDEFGHIJKLMNOP')}}, "value"),
    ],
    ids=(
        "pem",
        "bearer",
        "password-assignment",
        "token-assignment",
        "api-key-assignment",
        "secret-assignment",
        "sk-prefix",
        "github-prefix",
        "slack-prefix",
        "nested-prefix",
        "nested-aws",
    ),
)
def test_synthetic_secrets_are_rejected_without_echoing_source(
    statement: str, value: object, location: str, capsys
) -> None:
    security = _security()

    findings = security.find_secrets(statement=statement, value=value)

    assert findings
    assert all(tuple(finding.__dataclass_fields__) == ("category", "location") for finding in findings)
    assert findings[0].location == location
    with pytest.raises(security.SecretRejected) as raised:
        security.reject_secrets(statement=statement, value=value)
    message = str(raised.value)
    assert message.startswith("memory content rejected: ")
    assert f" in {location}" in message
    assert "synthetic-token-value" not in message
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("statement", "value"),
    [
        ("<redacted>", None),
        ("[REDACTED]", None),
        ("${TOKEN}", None),
        ("***", None),
        ("example credential documentation", {"value": "dummy test-only"}),
        ("token = ${TOKEN}", {"nested": "[REDACTED]"}),
        ("ordinary text", {"note": "Authorization: Bearer ${TOKEN}"}),
    ],
    ids=("angle", "brackets", "environment", "stars", "words", "assignment", "nested-bearer"),
)
def test_redacted_and_synthetic_placeholders_are_allowed(statement: str, value: object) -> None:
    security = _security()

    assert security.find_secrets(statement=statement, value=value) == ()
    assert security.reject_secrets(statement=statement, value=value) is None


def test_placeholder_exemptions_are_exact_values_not_substrings() -> None:
    security = _security()
    rejected_values = (
        "real-example-credential",
        "dummy-real-secret",
        "test-only-real-secret",
    )

    for rejected in rejected_values:
        findings = security.find_secrets(statement=f"token = {rejected}", value=None)
        assert findings
        with pytest.raises(security.SecretRejected) as raised:
            security.reject_secrets(statement=f"token = {rejected}", value=None)
        assert str(raised.value) == "memory content rejected: credential_assignment in statement"


@pytest.mark.parametrize(
    ("statement", "value"),
    [
        ("token = ${TOKEN}synthetic-real-secret", None),
        ("token = dummy synthetic-real-secret", None),
        ('password = "dummy synthetic-real-secret"', None),
        ("Authorization: Bearer ${TOKEN}synthetic-real-secret", None),
        ("ordinary text", {"api_key": "${TOKEN}synthetic-real-secret"}),
        ("ordinary text", {"password": "dummy synthetic-real-secret"}),
    ],
    ids=(
        "assignment-environment-prefix",
        "assignment-unquoted-placeholder-prefix",
        "assignment-quoted-placeholder-prefix",
        "bearer-environment-prefix",
        "json-environment-prefix",
        "json-quoted-placeholder-prefix",
    ),
)
def test_placeholder_prefix_cannot_hide_a_larger_credential(
    statement: str, value: object
) -> None:
    security = _security()

    findings = security.find_secrets(statement=statement, value=value)

    assert findings
    with pytest.raises(security.SecretRejected) as raised:
        security.reject_secrets(statement=statement, value=value)
    assert "synthetic-real-secret" not in str(raised.value)


def test_placeholder_before_real_credential_does_not_mask_later_matches() -> None:
    security = _security()
    real = "real-credential-value"
    statement = f"Authorization: Bearer ${{TOKEN}}\nAuthorization: Bearer {real}"
    nested_value = {
        "first_token": "${TOKEN}",
        "second_token": real,
    }

    statement_findings = security.find_secrets(statement=statement, value=None)
    value_findings = security.find_secrets(statement="ordinary text", value=nested_value)

    assert any(finding.location == "statement" for finding in statement_findings)
    assert any(finding.location == "value" for finding in value_findings)
    with pytest.raises(security.SecretRejected) as raised:
        security.reject_secrets(statement=statement, value=nested_value)
    assert str(raised.value).startswith("memory content rejected: ")
    assert real not in str(raised.value)
