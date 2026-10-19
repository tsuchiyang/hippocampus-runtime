from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterator, Sequence

from codex_memory.ledger_common import canonical_json

from .contracts import IndexHighwater


LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION_V2 = 2
SCHEMA_VERSION_V3 = 3
SCHEMA_VERSION_V4 = 4
SCHEMA_VERSION = SCHEMA_VERSION_V4
GENERATION_STATES = frozenset({"building", "ready", "active", "retired", "failed"})
EVENT_ACTIONS = frozenset({"create", "validate", "activate", "retire", "fail"})
_PROVIDER = "unconfigured:p2.1"
_MODEL = "unconfigured:p2.1"
_HASH = re.compile(r"\A[0-9a-f]{64}\Z")
_TIMESTAMP = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_SQLITE_TOKENIZER_WHITESPACE = frozenset({"\u0009", "\u000a", "\u000c", "\u000d", "\u0020"})
_SQLITE_TOKENIZER_WHITESPACE_CLASS = r"[\u0009\u000a\u000c\u000d\u0020]"
_SEMANTIC_COMPONENTS = frozenset({"dense", "sparse", "reranker"})
_PROVIDER_HEALTH_COMPONENTS = frozenset(
    {"capabilities", "dense", "sparse", "reranker"}
)
_PROVIDER_OPERATIONS = frozenset(
    {
        "capabilities", "dense_documents", "dense_query",
        "sparse_documents", "sparse_query", "rerank",
    }
)
_PROVIDER_STATUSES = frozenset(
    {
        "ok", "disabled", "unsupported", "unavailable", "busy", "timeout",
        "input_too_long", "invalid_response", "fingerprint_mismatch",
        "circuit_open", "deadline_exhausted",
    }
)
_SEMANTIC_COMPONENT_STATUSES = frozenset(
    {"disabled", "ready", "partial", "unsupported", "unavailable", "failed"}
)
_SEMANTIC_COMPONENT_REASONS = frozenset(
    {
        "disabled",
        "unsupported",
        "unavailable",
        "busy",
        "timeout",
        "input_too_long",
        "invalid_response",
        "fingerprint_mismatch",
        "circuit_open",
        "deadline_exhausted",
        "provider_fingerprint_mismatch",
        "task3_no_rerank_build",
    }
)
_ABSENT_SEMANTIC_COVERAGE_JSON = canonical_json(IndexHighwater().to_dict())


def _is_sqlite_tokenizer_whitespace(character: str) -> bool:
    return character in _SQLITE_TOKENIZER_WHITESPACE


class CacheCorruptError(RuntimeError):
    """The disposable cache cannot be trusted or replayed."""


class CacheUnsupportedError(CacheCorruptError):
    """The cache advertises a schema other than the frozen schema one."""


_LEGACY_TABLE_COLUMNS = {
    "retrieval_schema_meta": ("key", "value"),
    "retrieval_generations": (
        "generation_seq", "generation_id", "canonical_highwater_json",
        "indexed_highwater_json", "canonical_content_fingerprint",
        "provider_fingerprint", "model_fingerprint", "created_at",
    ),
    "retrieval_generation_events": (
        "event_seq", "event_id", "generation_id", "action", "payload_json",
        "event_sha256", "created_at",
    ),
    "retrieval_generation_current": (
        "generation_id", "state", "last_event_seq", "last_event_id",
        "validation_state", "degraded_reason", "failure_metadata_json", "updated_at",
    ),
    "retrieval_state": (
        "singleton_id", "active_generation_id", "validation_state",
        "degraded_reason", "failure_metadata_json", "created_at", "updated_at",
    ),
}

_V2_TABLE_COLUMNS = {
    **_LEGACY_TABLE_COLUMNS,
    "retrieval_generation_capabilities": (
        "generation_id", "fts5_state", "sqlite_version",
        "index_policy_fingerprint", "projection_manifest_sha256",
        "projected_document_count", "omitted_unsafe_count", "created_at",
    ),
    "retrieval_documents": (
        "document_seq", "document_id", "generation_id", "record_id",
        "record_seq", "record_type", "scope_kind", "scope_key", "topic_key",
        "statement", "value_json", "search_text", "state", "created_at",
        "last_memory_event_seq", "evidence_count", "content_sha256",
        "document_sha256",
    ),
    "retrieval_document_sources": (
        "source_seq", "source_id", "generation_id", "document_id", "ordinal",
        "safe_reference", "source_sha256",
    ),
    "retrieval_symbols": (
        "symbol_seq", "symbol_id", "generation_id", "document_id", "ordinal",
        "symbol_kind", "normalized_symbol", "display_symbol", "symbol_sha256",
    ),
    "retrieval_runs": (
        "run_seq", "run_id", "mode", "generation_id", "query_sha256",
        "canonical_highwater_json", "indexed_highwater_json", "status",
        "degraded_reasons_json", "trace_sha256", "created_at",
    ),
    "retrieval_candidate_traces": (
        "trace_seq", "trace_id", "run_id", "candidate_kind", "candidate_id",
        "channel", "channel_rank", "contribution_numerator",
        "contribution_denominator", "fused_rank", "selected",
        "reason_codes_json", "trace_sha256",
    ),
}

_V3_TABLE_COLUMNS = {
    **_V2_TABLE_COLUMNS,
    "retrieval_generations": (
        "generation_seq", "generation_id", "canonical_highwater_json",
        "indexed_highwater_json", "canonical_content_fingerprint",
        "provider_fingerprint", "model_fingerprint", "retrieval_schema_version",
        "created_at",
    ),
    "retrieval_runs": (
        "run_seq", "run_id", "mode", "generation_id", "query_sha256",
        "canonical_highwater_json", "indexed_highwater_json", "status",
        "wall_clock_ms", "provider_trace_json", "provider_trace_sha256",
        "degraded_reasons_json", "trace_sha256", "created_at",
    ),
    "retrieval_candidate_traces": (
        "trace_seq", "trace_id", "run_id", "candidate_kind", "candidate_id",
        "channel", "channel_rank", "contribution_numerator",
        "contribution_denominator", "fused_rank", "selected", "reason_codes_json",
        "raw_score_hex", "rerank_score_hex", "trace_sha256",
    ),
    "retrieval_semantic_generations": (
        "semantic_generation_seq", "semantic_generation_id", "projection_generation_id",
        "provider_fingerprint", "dense_model_fingerprint", "sparse_model_fingerprint",
        "reranker_model_fingerprint", "preprocessing_fingerprint", "build_config_sha256",
        "manifest_relative_path", "manifest_sha256", "semantic_database_sha256", "created_at",
    ),
    "retrieval_semantic_generation_events": (
        "event_seq", "event_id", "semantic_generation_id", "action", "payload_json",
        "event_sha256", "created_at",
    ),
    "retrieval_semantic_generation_current": (
        "semantic_generation_id", "state", "last_event_seq", "last_event_id",
        "validation_state", "degraded_reasons_json", "updated_at",
    ),
    "retrieval_semantic_state": (
        "singleton_id", "active_semantic_generation_id", "validation_state",
        "degraded_reasons_json", "created_at", "updated_at",
    ),
    "retrieval_semantic_components": (
        "semantic_generation_id", "component", "status", "model_fingerprint",
        "preprocessing_fingerprint", "vector_dimension", "eligible_document_count",
        "indexed_document_count", "coverage_highwater_json", "component_manifest_sha256", "created_at",
    ),
    "retrieval_provider_health_events": (
        "event_seq", "event_id", "provider_fingerprint", "component", "action",
        "failure_kind", "open_until", "lease_until", "payload_sha256", "created_at",
    ),
    "retrieval_provider_health_current": (
        "provider_fingerprint", "component", "state", "consecutive_failures", "open_until",
        "probe_lease_until", "last_event_seq", "last_event_id", "updated_at",
    ),
}

_V4_TABLE_COLUMNS = {
    **_V3_TABLE_COLUMNS,
    "retrieval_provider_attempt_admissions": (
        "attempt_seq", "attempt_id", "provider_fingerprint", "capability", "operation",
        "circuit_key", "history_epoch", "fencing_version", "admitted_at", "expires_at",
        "payload_sha256",
    ),
    "retrieval_provider_attempt_results": (
        "result_seq", "result_id", "attempt_id", "outcome", "provider_status",
        "recorded_at", "payload_sha256",
    ),
}


def _append_only_trigger_statement(table: str, operation: str) -> str:
    return f"""
        CREATE TRIGGER {table}_reject_{operation}
        BEFORE {operation.upper()} ON {table}
        BEGIN
            SELECT RAISE(ABORT, 'append-only table');
        END
    """


def _append_only_guard_statement(table: str, unique_keys: Sequence[Sequence[str]]) -> str:
    conflicts = " OR ".join(
        "(" + " AND ".join(f"{column} = NEW.{column}" for column in columns) + ")"
        for columns in unique_keys
    )
    return f"""
        CREATE TRIGGER {table}_guard_insert_conflict
        BEFORE INSERT ON {table}
        WHEN EXISTS (SELECT 1 FROM {table} WHERE {conflicts})
        BEGIN
            SELECT RAISE(ABORT, 'append-only table');
        END
    """


_LEGACY_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE retrieval_schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE retrieval_generations (
        generation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        generation_id TEXT NOT NULL UNIQUE,
        canonical_highwater_json TEXT NOT NULL,
        indexed_highwater_json TEXT NOT NULL,
        canonical_content_fingerprint TEXT NOT NULL,
        provider_fingerprint TEXT NOT NULL,
        model_fingerprint TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE retrieval_generation_events (
        event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
        action TEXT NOT NULL CHECK (action IN ('create', 'validate', 'activate', 'retire', 'fail')),
        payload_json TEXT NOT NULL,
        event_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE retrieval_generation_current (
        generation_id TEXT PRIMARY KEY REFERENCES retrieval_generations(generation_id),
        state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')),
        last_event_seq INTEGER NOT NULL,
        last_event_id TEXT NOT NULL REFERENCES retrieval_generation_events(event_id),
        validation_state TEXT NOT NULL,
        degraded_reason TEXT,
        failure_metadata_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE retrieval_state (
        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
        active_generation_id TEXT REFERENCES retrieval_generations(generation_id),
        validation_state TEXT NOT NULL,
        degraded_reason TEXT,
        failure_metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX retrieval_generation_events_generation_seq ON retrieval_generation_events(generation_id, event_seq)",
    "CREATE INDEX retrieval_generation_current_state_generation ON retrieval_generation_current(state, generation_id)",
    _append_only_trigger_statement("retrieval_generations", "update"),
    _append_only_trigger_statement("retrieval_generations", "delete"),
    _append_only_guard_statement("retrieval_generations", (("generation_seq",), ("generation_id",))),
    _append_only_trigger_statement("retrieval_generation_events", "update"),
    _append_only_trigger_statement("retrieval_generation_events", "delete"),
    _append_only_guard_statement("retrieval_generation_events", (("event_seq",), ("event_id",))),
)


def _v2_schema_statements(fts5_available: bool) -> tuple[str, ...]:
    statements = [
        """
        CREATE TABLE retrieval_schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_generations (
            generation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id TEXT NOT NULL UNIQUE,
            canonical_highwater_json TEXT NOT NULL,
            indexed_highwater_json TEXT NOT NULL,
            canonical_content_fingerprint TEXT NOT NULL,
            provider_fingerprint TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_generation_events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            action TEXT NOT NULL CHECK (action IN ('create', 'validate', 'activate', 'retire', 'fail')),
            payload_json TEXT NOT NULL,
            event_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_generation_current (
            generation_id TEXT PRIMARY KEY REFERENCES retrieval_generations(generation_id),
            state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')),
            last_event_seq INTEGER NOT NULL,
            last_event_id TEXT NOT NULL REFERENCES retrieval_generation_events(event_id),
            validation_state TEXT NOT NULL,
            degraded_reason TEXT,
            failure_metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_generation_id TEXT REFERENCES retrieval_generations(generation_id),
            validation_state TEXT NOT NULL,
            degraded_reason TEXT,
            failure_metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_generation_capabilities (
            generation_id TEXT PRIMARY KEY REFERENCES retrieval_generations(generation_id),
            fts5_state TEXT NOT NULL CHECK (fts5_state IN ('available', 'unavailable')),
            sqlite_version TEXT NOT NULL,
            index_policy_fingerprint TEXT NOT NULL,
            projection_manifest_sha256 TEXT NOT NULL,
            projected_document_count INTEGER NOT NULL,
            omitted_unsafe_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_documents (
            document_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL UNIQUE,
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            record_id TEXT NOT NULL,
            record_seq INTEGER NOT NULL,
            record_type TEXT NOT NULL,
            scope_kind TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            topic_key TEXT NOT NULL,
            statement TEXT NOT NULL,
            value_json TEXT NOT NULL,
            search_text TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_memory_event_seq INTEGER NOT NULL,
            evidence_count INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL,
            document_sha256 TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_document_sources (
            source_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL UNIQUE,
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            document_id TEXT NOT NULL REFERENCES retrieval_documents(document_id),
            ordinal INTEGER NOT NULL,
            safe_reference TEXT NOT NULL,
            source_sha256 TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_symbols (
            symbol_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id TEXT NOT NULL UNIQUE,
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            document_id TEXT NOT NULL REFERENCES retrieval_documents(document_id),
            ordinal INTEGER NOT NULL,
            symbol_kind TEXT NOT NULL,
            normalized_symbol TEXT NOT NULL,
            display_symbol TEXT NOT NULL,
            symbol_sha256 TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_runs (
            run_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            mode TEXT NOT NULL CHECK (mode IN ('off', 'shadow', 'active')),
            generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            query_sha256 TEXT NOT NULL,
            canonical_highwater_json TEXT NOT NULL,
            indexed_highwater_json TEXT NOT NULL,
            status TEXT NOT NULL,
            degraded_reasons_json TEXT NOT NULL,
            trace_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE retrieval_candidate_traces (
            trace_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL REFERENCES retrieval_runs(run_id),
            candidate_kind TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            channel_rank INTEGER NOT NULL,
            contribution_numerator INTEGER NOT NULL,
            contribution_denominator INTEGER NOT NULL,
            fused_rank INTEGER,
            selected INTEGER NOT NULL,
            reason_codes_json TEXT NOT NULL,
            trace_sha256 TEXT NOT NULL
        )
        """,
    ]
    if fts5_available:
        statements.insert(
            6,
            "CREATE VIRTUAL TABLE retrieval_documents_fts USING fts5(document_id UNINDEXED, generation_id UNINDEXED, search_text)",
        )
    statements.extend(
        [
            "CREATE INDEX retrieval_generation_events_generation_seq ON retrieval_generation_events(generation_id, event_seq)",
            "CREATE INDEX retrieval_generation_current_state_generation ON retrieval_generation_current(state, generation_id)",
            "CREATE INDEX retrieval_documents_generation_seq ON retrieval_documents(generation_id, document_seq)",
            "CREATE INDEX retrieval_document_sources_document_seq ON retrieval_document_sources(generation_id, document_id, ordinal)",
            "CREATE INDEX retrieval_symbols_document_seq ON retrieval_symbols(generation_id, document_id, ordinal)",
            "CREATE INDEX retrieval_runs_generation_seq ON retrieval_runs(generation_id, run_seq)",
            "CREATE INDEX retrieval_candidate_traces_run_seq ON retrieval_candidate_traces(run_id, trace_seq)",
            _append_only_trigger_statement("retrieval_generations", "update"),
            _append_only_trigger_statement("retrieval_generations", "delete"),
            _append_only_guard_statement("retrieval_generations", (("generation_seq",), ("generation_id",))),
            _append_only_trigger_statement("retrieval_generation_events", "update"),
            _append_only_trigger_statement("retrieval_generation_events", "delete"),
            _append_only_guard_statement("retrieval_generation_events", (("event_seq",), ("event_id",))),
            _append_only_trigger_statement("retrieval_generation_capabilities", "update"),
            _append_only_trigger_statement("retrieval_generation_capabilities", "delete"),
            _append_only_guard_statement("retrieval_generation_capabilities", (("generation_id",),)),
            _append_only_trigger_statement("retrieval_documents", "update"),
            _append_only_trigger_statement("retrieval_documents", "delete"),
            _append_only_guard_statement("retrieval_documents", (("document_seq",), ("document_id",))),
            _append_only_trigger_statement("retrieval_document_sources", "update"),
            _append_only_trigger_statement("retrieval_document_sources", "delete"),
            _append_only_guard_statement("retrieval_document_sources", (("source_seq",), ("source_id",))),
            _append_only_trigger_statement("retrieval_symbols", "update"),
            _append_only_trigger_statement("retrieval_symbols", "delete"),
            _append_only_guard_statement("retrieval_symbols", (("symbol_seq",), ("symbol_id",))),
            _append_only_trigger_statement("retrieval_runs", "update"),
            _append_only_trigger_statement("retrieval_runs", "delete"),
            _append_only_guard_statement("retrieval_runs", (("run_seq",), ("run_id",))),
            _append_only_trigger_statement("retrieval_candidate_traces", "update"),
            _append_only_trigger_statement("retrieval_candidate_traces", "delete"),
            _append_only_guard_statement("retrieval_candidate_traces", (("trace_seq",), ("trace_id",))),
        ]
    )
    return tuple(statements)


def _v3_schema_statements(fts5_available: bool) -> tuple[str, ...]:
    """Frozen v3 DDL.  v1/v2 remain independently recognisable."""
    statements: list[str] = []
    for statement in _v2_schema_statements(fts5_available):
        if "CREATE TABLE retrieval_generations" in statement:
            statement = statement.replace(
                "model_fingerprint TEXT NOT NULL,\n            created_at",
                "model_fingerprint TEXT NOT NULL,\n            retrieval_schema_version INTEGER NOT NULL,\n            created_at",
            )
        elif "CREATE TABLE retrieval_runs" in statement:
            statement = statement.replace(
                "status TEXT NOT NULL,\n            degraded_reasons_json",
                "status TEXT NOT NULL,\n            wall_clock_ms INTEGER NOT NULL,\n            provider_trace_json TEXT NOT NULL,\n            provider_trace_sha256 TEXT NOT NULL,\n            degraded_reasons_json",
            )
        elif "CREATE TABLE retrieval_candidate_traces" in statement:
            statement = statement.replace(
                "reason_codes_json TEXT NOT NULL,\n            trace_sha256",
                "reason_codes_json TEXT NOT NULL,\n            raw_score_hex TEXT,\n            rerank_score_hex TEXT,\n            trace_sha256",
            )
        statements.append(statement)
    statements.extend((
        """CREATE TABLE retrieval_semantic_generations (
            semantic_generation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            semantic_generation_id TEXT NOT NULL UNIQUE,
            projection_generation_id TEXT NOT NULL REFERENCES retrieval_generations(generation_id),
            provider_fingerprint TEXT NOT NULL,
            dense_model_fingerprint TEXT,
            sparse_model_fingerprint TEXT,
            reranker_model_fingerprint TEXT,
            preprocessing_fingerprint TEXT NOT NULL,
            build_config_sha256 TEXT NOT NULL,
            manifest_relative_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            semantic_database_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE retrieval_semantic_generation_events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            semantic_generation_id TEXT NOT NULL REFERENCES retrieval_semantic_generations(semantic_generation_id),
            action TEXT NOT NULL CHECK (action IN ('create', 'validate', 'activate', 'retire', 'fail')),
            payload_json TEXT NOT NULL,
            event_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE retrieval_semantic_generation_current (
            semantic_generation_id TEXT PRIMARY KEY REFERENCES retrieval_semantic_generations(semantic_generation_id),
            state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')),
            last_event_seq INTEGER NOT NULL,
            last_event_id TEXT NOT NULL REFERENCES retrieval_semantic_generation_events(event_id),
            validation_state TEXT NOT NULL,
            degraded_reasons_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE retrieval_semantic_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_semantic_generation_id TEXT REFERENCES retrieval_semantic_generations(semantic_generation_id),
            validation_state TEXT NOT NULL,
            degraded_reasons_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE retrieval_semantic_components (
            semantic_generation_id TEXT NOT NULL REFERENCES retrieval_semantic_generations(semantic_generation_id),
            component TEXT NOT NULL CHECK (component IN ('dense', 'sparse', 'reranker')),
            status TEXT NOT NULL CHECK (status IN ('disabled', 'ready', 'partial', 'unsupported', 'unavailable', 'failed')),
            model_fingerprint TEXT,
            preprocessing_fingerprint TEXT NOT NULL,
            vector_dimension INTEGER,
            eligible_document_count INTEGER NOT NULL,
            indexed_document_count INTEGER NOT NULL,
            coverage_highwater_json TEXT NOT NULL,
            component_manifest_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (semantic_generation_id, component)
        )""",
        """CREATE TABLE retrieval_provider_health_events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            provider_fingerprint TEXT NOT NULL,
            component TEXT NOT NULL CHECK (component IN ('dense', 'sparse', 'reranker')),
            action TEXT NOT NULL CHECK (action IN ('success', 'failure', 'open', 'half_open', 'probe_acquired', 'probe_released')),
            failure_kind TEXT,
            open_until TEXT,
            lease_until TEXT,
            payload_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE retrieval_provider_health_current (
            provider_fingerprint TEXT NOT NULL,
            component TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('closed', 'open', 'half_open')),
            consecutive_failures INTEGER NOT NULL,
            open_until TEXT,
            probe_lease_until TEXT,
            last_event_seq INTEGER NOT NULL,
            last_event_id TEXT NOT NULL REFERENCES retrieval_provider_health_events(event_id),
            updated_at TEXT NOT NULL,
            PRIMARY KEY (provider_fingerprint, component)
        )""",
        "CREATE INDEX retrieval_semantic_generation_events_generation_seq ON retrieval_semantic_generation_events(semantic_generation_id, event_seq)",
        "CREATE INDEX retrieval_semantic_components_generation_component ON retrieval_semantic_components(semantic_generation_id, component)",
        "CREATE INDEX retrieval_provider_health_events_provider_component_seq ON retrieval_provider_health_events(provider_fingerprint, component, event_seq)",
    ))
    for table, keys in (
        ("retrieval_semantic_generations", (("semantic_generation_seq",), ("semantic_generation_id",))),
        ("retrieval_semantic_generation_events", (("event_seq",), ("event_id",))),
        ("retrieval_semantic_components", (("semantic_generation_id", "component"),)),
        ("retrieval_provider_health_events", (("event_seq",), ("event_id",))),
    ):
        statements.extend((_append_only_trigger_statement(table, "update"), _append_only_trigger_statement(table, "delete"), _append_only_guard_statement(table, keys)))
    return tuple(statements)


def _v4_schema_statements(fts5_available: bool) -> tuple[str, ...]:
    """Provider-control extension; the independently verified v3 DDL stays frozen."""
    statements: list[str] = []
    for statement in _v3_schema_statements(fts5_available):
        if "CREATE TABLE retrieval_provider_health_events" in statement:
            statement = statement.replace(
                "component IN ('dense', 'sparse', 'reranker')",
                "component IN ('capabilities', 'dense', 'sparse', 'reranker')",
            )
        elif "CREATE TABLE retrieval_provider_health_current" in statement:
            statement = statement.replace(
                "component TEXT NOT NULL,",
                "component TEXT NOT NULL CHECK (component IN ('capabilities', 'dense', 'sparse', 'reranker')),",
            )
        statements.append(statement)
    statements.extend((
        """CREATE TABLE retrieval_provider_attempt_admissions (
            attempt_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL UNIQUE,
            provider_fingerprint TEXT NOT NULL,
            capability TEXT NOT NULL CHECK (capability IN ('capabilities', 'dense', 'sparse', 'reranker')),
            operation TEXT NOT NULL CHECK (operation IN ('capabilities', 'dense_documents', 'dense_query', 'sparse_documents', 'sparse_query', 'rerank')),
            circuit_key TEXT NOT NULL,
            history_epoch INTEGER NOT NULL CHECK (history_epoch >= 1),
            fencing_version INTEGER NOT NULL CHECK (fencing_version >= 1),
            admitted_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            UNIQUE (circuit_key, history_epoch, fencing_version)
        )""",
        """CREATE TABLE retrieval_provider_attempt_results (
            result_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id TEXT NOT NULL UNIQUE,
            attempt_id TEXT NOT NULL UNIQUE REFERENCES retrieval_provider_attempt_admissions(attempt_id),
            outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'released', 'expired_unknown')),
            provider_status TEXT,
            recorded_at TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        )""",
        "CREATE INDEX retrieval_provider_attempt_admissions_provider_seq ON retrieval_provider_attempt_admissions(provider_fingerprint, attempt_seq)",
    ))
    for table, keys in (
        ("retrieval_provider_attempt_admissions", (("attempt_seq",), ("attempt_id",), ("circuit_key", "history_epoch", "fencing_version"))),
        ("retrieval_provider_attempt_results", (("result_seq",), ("result_id",), ("attempt_id",))),
    ):
        statements.extend((
            _append_only_trigger_statement(table, "update"),
            _append_only_trigger_statement(table, "delete"),
            _append_only_guard_statement(table, keys),
        ))
    return tuple(statements)


def _normal_sql(value: str) -> str:
    """Normalize SQL with SQLite's ASCII-only case folding outside quotes."""
    tokens: list[tuple[str, str]] = []
    quote: str | None = None
    word: list[str] = []
    index = 0

    def flush_word() -> None:
        if word:
            tokens.append(("word", "".join(word)))
            word.clear()

    while index < len(value):
        character = value[index]
        if quote is not None:
            word.append(character)
            if character == quote:
                if quote != "]" and index + 1 < len(value) and value[index + 1] == quote:
                    word.append(value[index + 1])
                    index += 1
                else:
                    quote = None
                    flush_word()
            index += 1
            continue
        if _is_sqlite_tokenizer_whitespace(character):
            flush_word()
            index += 1
            continue
        if character in "'\"`[":
            flush_word()
            quote = "]" if character == "[" else character
            word.append(character)
            index += 1
            continue
        if character.isalnum() or character in "_$":
            word.append(
                chr(ord(character) + 32)
                if "A" <= character <= "Z"
                else character
            )
            index += 1
            continue
        flush_word()
        operator = next(
            (candidate for candidate in ("->>", "!=", ">=", "<=", "<>", "==", "||", "<<", ">>", "->", ":=") if value.startswith(candidate, index)),
            None,
        )
        if operator is not None:
            tokens.append(("operator", operator))
            index += len(operator)
            continue
        if character in "(),.":
            tokens.append(("punctuation", character))
        elif character in "=<>!+-*/%|&^~:":
            tokens.append(("operator", character))
        else:
            tokens.append(("punctuation", character))
        index += 1
    flush_word()
    result: list[str] = []
    previous: str | None = None
    for kind, token in tokens:
        if result and ((kind == previous == "word") or (kind == previous == "operator")):
            result.append(" ")
        result.append(token)
        previous = kind
    normalized = "".join(result)
    return normalized[:-1] if normalized.endswith(";") else normalized


def _schema_sql_from_statements(statements: Sequence[str]) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}
    for statement in statements:
        match = re.search(
            rf"CREATE{_SQLITE_TOKENIZER_WHITESPACE_CLASS}+(?:(VIRTUAL{_SQLITE_TOKENIZER_WHITESPACE_CLASS}+)?TABLE|INDEX|TRIGGER)"
            rf"{_SQLITE_TOKENIZER_WHITESPACE_CLASS}+([A-Za-z_][A-Za-z0-9_]*)",
            statement,
            re.I,
        )
        if match is None:
            raise AssertionError("invalid frozen schema statement")
        kind = "table" if match.group(0).lower().startswith("create virtual table") or match.group(0).lower().startswith("create table") else match.group(0).split()[1].lower()
        expected[(kind, match.group(2))] = _normal_sql(statement)
    return expected


_LEGACY_EXPECTED_SCHEMA_SQL = _schema_sql_from_statements(_LEGACY_SCHEMA_STATEMENTS)


def connect_cache_database(
    path: Path, *, readonly: bool = False, busy_timeout_ms: int = 5000
) -> sqlite3.Connection:
    """Open one caller-owned cache connection with the frozen SQLite settings.

    ``busy_timeout_ms`` is applied at connection creation, before any
    lock-sensitive pragma.  Deadline-bound health bookkeeping passes zero so
    ``journal_mode`` cannot wait on a concurrent cache lock before the caller
    gets a chance to enforce its request budget.
    """
    if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or busy_timeout_ms < 0:
        raise ValueError("busy_timeout_ms must be a non-negative integer")
    connection: sqlite3.Connection | None = None
    try:
        timeout_seconds = busy_timeout_ms / 1000
        if readonly:
            connection = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro",
                uri=True,
                timeout=timeout_seconds,
                isolation_level=None,
            )
            connection.execute("PRAGMA query_only=ON")
        else:
            connection = sqlite3.connect(
                path, timeout=timeout_seconds, isolation_level=None
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA recursive_triggers=ON")
        connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        if not readonly:
            if busy_timeout_ms > 0:
                connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
        return connection
    except BaseException:
        if connection is not None:
            connection.close()
        raise


@contextmanager
def immediate_transaction(
    connection: sqlite3.Connection,
    *,
    before_commit: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Commit a cache transaction only after its optional final acceptance check."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        if before_commit is not None:
            before_commit()
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def initialize_schema(connection: sqlite3.Connection, *, in_transaction: bool = False) -> None:
    """Create the immutable live schema once; never migrate an existing cache."""
    if in_transaction:
        _initialize_schema_v4(connection)
        return
    with immediate_transaction(connection):
        _initialize_schema_v4(connection)


def _initialize_schema(connection: sqlite3.Connection) -> None:
    raise AssertionError("legacy schema initializer is intentionally unavailable")


def initialize_schema_v3(connection: sqlite3.Connection, *, in_transaction: bool = False) -> None:
    if in_transaction:
        _initialize_schema_v3(connection)
        return
    with immediate_transaction(connection):
        _initialize_schema_v3(connection)


def _initialize_schema_v4(connection: sqlite3.Connection) -> None:
    version = inspect_cache_schema_version(connection)
    if version is not None:
        if version == SCHEMA_VERSION_V4:
            verify_cache_database(connection)
            return
        raise CacheUnsupportedError("unsupported retrieval schema")
    other_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    if other_table:
        raise CacheCorruptError("retrieval schema is corrupt")
    fts5_available = _supports_fts5(connection)
    for statement in _v4_schema_statements(fts5_available):
        connection.execute(statement)
    connection.execute(
        f"INSERT INTO retrieval_schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION_V4}')"
    )
    connection.execute(
        """INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', NULL, '{}',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z')"""
    )
    connection.execute(
        """INSERT INTO retrieval_semantic_state(
            singleton_id, active_semantic_generation_id, validation_state,
            degraded_reasons_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', '[]',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z')"""
    )
def _initialize_schema_v3(connection: sqlite3.Connection) -> None:
    version = inspect_cache_schema_version(connection)
    if version is not None:
        if version == SCHEMA_VERSION_V3:
            verify_cache_database(connection)
            return
        raise CacheUnsupportedError("unsupported retrieval schema")
    other_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    if other_table:
        raise CacheCorruptError("retrieval schema is corrupt")
    fts5_available = _supports_fts5(connection)
    for statement in _v3_schema_statements(fts5_available):
        connection.execute(statement)
    connection.execute(
        f"INSERT INTO retrieval_schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION_V3}')"
    )
    connection.execute(
        """INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', NULL, '{}',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z')"""
    )
    connection.execute(
        """INSERT INTO retrieval_semantic_state(
            singleton_id, active_semantic_generation_id, validation_state,
            degraded_reasons_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', '[]',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z')"""
    )


def _initialize_schema_v2(connection: sqlite3.Connection) -> None:
    meta_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_schema_meta'"
    ).fetchone()
    if meta_exists:
        version = inspect_cache_schema_version(connection)
        if version == SCHEMA_VERSION_V2:
            verify_cache_database(connection)
            return
        raise CacheUnsupportedError("unsupported retrieval schema")
    other_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    if other_table:
        raise CacheCorruptError("retrieval schema is corrupt")
    fts5_available = _supports_fts5(connection)
    for statement in _v2_schema_statements(fts5_available):
        connection.execute(statement)
    connection.execute(
        f"INSERT INTO retrieval_schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION_V2}')"
    )
    connection.execute(
        """INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', NULL, '{}',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z')"""
    )


def inspect_cache_schema_version(connection: sqlite3.Connection) -> int | None:
    try:
        meta_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_schema_meta'"
        ).fetchone()
        if not meta_exists:
            return None
        meta = list(connection.execute("SELECT key, value FROM retrieval_schema_meta"))
        if len(meta) != 1 or meta[0]["key"] != "schema_version":
            raise CacheCorruptError("retrieval schema metadata is corrupt")
        try:
            return int(meta[0]["value"])
        except (TypeError, ValueError) as exc:
            raise CacheCorruptError("retrieval schema metadata is corrupt") from exc
    except sqlite3.Error as exc:
        raise CacheCorruptError("retrieval schema metadata is corrupt") from exc


def _supports_fts5(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SAVEPOINT retrieval_fts5_probe")
        connection.execute(
            "CREATE VIRTUAL TABLE temp.retrieval_fts5_probe USING fts5(search_text)"
        )
    except sqlite3.Error:
        try:
            connection.execute("ROLLBACK TO retrieval_fts5_probe")
        except sqlite3.Error:
            pass
        try:
            connection.execute("RELEASE retrieval_fts5_probe")
        except sqlite3.Error:
            pass
        return False
    else:
        connection.execute("ROLLBACK TO retrieval_fts5_probe")
        connection.execute("RELEASE retrieval_fts5_probe")
        return True


def initialize_schema_v2(connection: sqlite3.Connection, *, in_transaction: bool = False) -> None:
    if in_transaction:
        _initialize_schema_v2(connection)
        return
    with immediate_transaction(connection):
        _initialize_schema_v2(connection)


def _initialize_schema_v2(connection: sqlite3.Connection) -> None:
    version = inspect_cache_schema_version(connection)
    if version is not None:
        if version == SCHEMA_VERSION_V2:
            verify_cache_database(connection)
            return
        raise CacheUnsupportedError("unsupported retrieval schema")
    other_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    if other_table:
        raise CacheCorruptError("retrieval schema is corrupt")
    fts5_available = _supports_fts5(connection)
    for statement in _v2_schema_statements(fts5_available):
        connection.execute(statement)
    connection.execute(
        f"INSERT INTO retrieval_schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION_V2}')"
    )
    connection.execute(
        """INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, NULL, 'none', NULL, '{}',
            '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z')"""
    )


def deterministic_generation_id(
    canonical_highwater: IndexHighwater,
    canonical_content_fingerprint: str,
    *,
    schema_version: int = SCHEMA_VERSION_V2,
) -> str:
    preimage = {
        "canonical_content_fingerprint": canonical_content_fingerprint,
        "canonical_highwater": canonical_highwater.to_dict(),
        "model_fingerprint": _MODEL,
        "provider_fingerprint": _PROVIDER,
        "retrieval_schema_version": schema_version,
    }
    return "gen_" + hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()[:32]


def append_generation(
    connection: sqlite3.Connection,
    *,
    generation_id: str,
    canonical_highwater: IndexHighwater,
    indexed_highwater: IndexHighwater,
    canonical_content_fingerprint: str,
    provider_fingerprint: str,
    model_fingerprint: str,
    created_at: str,
) -> None:
    values = (
        generation_id, canonical_json(canonical_highwater.to_dict()),
        canonical_json(indexed_highwater.to_dict()), canonical_content_fingerprint,
        provider_fingerprint, model_fingerprint, created_at,
    )
    if inspect_cache_schema_version(connection) in {SCHEMA_VERSION_V3, SCHEMA_VERSION_V4}:
        generation_schema_version = (
            SCHEMA_VERSION_V3 if generation_id == deterministic_generation_id(
                canonical_highwater, canonical_content_fingerprint, schema_version=SCHEMA_VERSION_V3
            ) else SCHEMA_VERSION_V2
        )
        connection.execute(
            """INSERT INTO retrieval_generations(
                generation_id, canonical_highwater_json, indexed_highwater_json,
                canonical_content_fingerprint, provider_fingerprint, model_fingerprint,
                retrieval_schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (*values[:-1], generation_schema_version, values[-1]),
        )
    else:
        connection.execute(
            """INSERT INTO retrieval_generations(
                generation_id, canonical_highwater_json, indexed_highwater_json,
                canonical_content_fingerprint, provider_fingerprint, model_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""", values,
        )


def _event_values(generation_id: str, action: str, payload_json: str, created_at: str) -> tuple[str, str]:
    preimage = {
        "action": action,
        "created_at": created_at,
        "generation_id": generation_id,
        "payload": _json_object(payload_json),
    }
    event_sha256 = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
    return f"evt_{event_sha256[:32]}", event_sha256


def append_generation_event(
    connection: sqlite3.Connection,
    *,
    generation_id: str,
    action: str,
    created_at: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if action not in EVENT_ACTIONS:
        raise CacheCorruptError("invalid lifecycle action")
    payload_json = canonical_json({} if payload is None else payload)
    event_id, event_sha256 = _event_values(generation_id, action, payload_json, created_at)
    connection.execute(
        """INSERT INTO retrieval_generation_events(
            event_id, generation_id, action, payload_json, event_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        (event_id, generation_id, action, payload_json, event_sha256, created_at),
    )


def copy_history(connection: sqlite3.Connection, generations: Sequence[sqlite3.Row], events: Sequence[sqlite3.Row], *, source_schema_version: int = SCHEMA_VERSION_V2) -> None:
    for row in generations:
        values = tuple(row)
        if inspect_cache_schema_version(connection) in {SCHEMA_VERSION_V3, SCHEMA_VERSION_V4} and len(values) == 8:
            connection.execute(
                "INSERT INTO retrieval_generations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*values[:-1], source_schema_version, values[-1]),
            )
        else:
            connection.execute(
                "INSERT INTO retrieval_generations VALUES ({})".format(", ".join("?" for _ in values)), values
            )
    for row in events:
        connection.execute(
            "INSERT INTO retrieval_generation_events VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(row)
        )


def read_v2_immutable_rows(connection: sqlite3.Connection) -> dict[str, tuple[sqlite3.Row, ...]]:
    """Read verified v2 derived history for explicit v3 rebuild compatibility."""
    if inspect_cache_schema_version(connection) != SCHEMA_VERSION_V2:
        return {}
    verify_cache_database(connection)
    ordered = {
        "retrieval_generation_capabilities": "generation_id",
        "retrieval_documents": "document_seq",
        "retrieval_document_sources": "source_seq",
        "retrieval_symbols": "symbol_seq",
        "retrieval_runs": "run_seq",
        "retrieval_candidate_traces": "trace_seq",
    }
    return {table: tuple(connection.execute(f"SELECT * FROM {table} ORDER BY {order}")) for table, order in ordered.items()}


def copy_v2_immutable_rows(connection: sqlite3.Connection, rows: dict[str, Sequence[sqlite3.Row]]) -> None:
    """Copy v2 immutable rows, writing v3-only compatibility fields explicitly."""
    for table in ("retrieval_generation_capabilities", "retrieval_documents", "retrieval_document_sources", "retrieval_symbols"):
        for row in rows.get(table, ()):
            values = tuple(row)
            connection.execute(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in values)})", values)
    for row in rows.get("retrieval_runs", ()):
        values = tuple(row)
        connection.execute("INSERT INTO retrieval_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*values[:8], 0, "[]", hashlib.sha256(b"[]").hexdigest(), *values[8:]))
    for row in rows.get("retrieval_candidate_traces", ()):
        connection.execute(
            """INSERT INTO retrieval_candidate_traces(
                trace_seq, trace_id, run_id, candidate_kind, candidate_id, channel,
                channel_rank, contribution_numerator, contribution_denominator,
                fused_rank, selected, reason_codes_json, raw_score_hex,
                rerank_score_hex, trace_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)""",
            tuple(row),
        )


def read_history(connection: sqlite3.Connection) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    connection.execute("BEGIN")
    try:
        verify_cache_database(connection)
        return (
            list(connection.execute("SELECT * FROM retrieval_generations ORDER BY generation_seq")),
            list(connection.execute("SELECT * FROM retrieval_generation_events ORDER BY event_seq")),
        )
    finally:
        if connection.in_transaction:
            connection.rollback()


def read_history_for_rebuild(connection: sqlite3.Connection) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    connection.execute("BEGIN")
    try:
        version = inspect_cache_schema_version(connection)
        if version == LEGACY_SCHEMA_VERSION:
            verify_legacy_schema_v1(connection)
        else:
            verify_cache_database(connection)
        return (
            list(connection.execute("SELECT * FROM retrieval_generations ORDER BY generation_seq")),
            list(connection.execute("SELECT * FROM retrieval_generation_events ORDER BY event_seq")),
        )
    finally:
        if connection.in_transaction:
            connection.rollback()


def read_semantic_history_for_rebuild(connection: sqlite3.Connection) -> dict[str, tuple[sqlite3.Row, ...]]:
    """Read immutable v3 semantic and provider-health history for rebuild."""
    connection.execute("BEGIN")
    try:
        if inspect_cache_schema_version(connection) not in {SCHEMA_VERSION_V3, SCHEMA_VERSION_V4}:
            return {}
        verify_cache_database(connection)
        return {
            "generations": tuple(connection.execute("SELECT * FROM retrieval_semantic_generations ORDER BY semantic_generation_seq")),
            "events": tuple(connection.execute("SELECT * FROM retrieval_semantic_generation_events ORDER BY event_seq")),
            "components": tuple(connection.execute("SELECT * FROM retrieval_semantic_components ORDER BY semantic_generation_id, component")),
            "provider_health_events": tuple(connection.execute("SELECT * FROM retrieval_provider_health_events ORDER BY event_seq")),
            "provider_attempt_admissions": tuple(connection.execute("SELECT * FROM retrieval_provider_attempt_admissions ORDER BY attempt_seq")) if inspect_cache_schema_version(connection) == SCHEMA_VERSION_V4 else (),
            "provider_attempt_results": tuple(connection.execute("SELECT * FROM retrieval_provider_attempt_results ORDER BY result_seq")) if inspect_cache_schema_version(connection) == SCHEMA_VERSION_V4 else (),
        }
    finally:
        if connection.in_transaction:
            connection.rollback()


def copy_semantic_history(connection: sqlite3.Connection, history: dict[str, Sequence[sqlite3.Row]]) -> None:
    """Copy immutable semantic/health metadata; mutable projections replay later."""
    for table, key in (
        ("retrieval_semantic_generations", "generations"),
        ("retrieval_semantic_generation_events", "events"),
        ("retrieval_semantic_components", "components"),
        ("retrieval_provider_health_events", "provider_health_events"),
        ("retrieval_provider_attempt_admissions", "provider_attempt_admissions"),
        ("retrieval_provider_attempt_results", "provider_attempt_results"),
    ):
        for row in history.get(key, ()):
            values = tuple(row)
            connection.execute(
                f"INSERT INTO {table} VALUES ({', '.join('?' for _ in values)})", values
            )


def rebuild_projections(connection: sqlite3.Connection) -> None:
    states, active = _replay(connection)
    connection.execute("DELETE FROM retrieval_generation_current")
    connection.execute("DELETE FROM retrieval_state")
    for generation_id, value in states.items():
        connection.execute(
            """INSERT INTO retrieval_generation_current(
                generation_id, state, last_event_seq, last_event_id, validation_state,
                degraded_reason, failure_metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                generation_id, value["state"], value["event_seq"], value["event_id"],
                value["validation_state"], value["degraded_reason"], "{}", value["updated_at"],
            ),
        )
    timestamps = [value["updated_at"] for value in states.values()]
    timestamp = max(timestamps, default="1970-01-01T00:00:00.000000Z")
    validation_state = "validated" if active is not None else "none"
    connection.execute(
        """INSERT INTO retrieval_state(
            singleton_id, active_generation_id, validation_state, degraded_reason,
            failure_metadata_json, created_at, updated_at
        ) VALUES (1, ?, ?, NULL, '{}', ?, ?)""",
        (active, validation_state, timestamp, timestamp),
    )
    if inspect_cache_schema_version(connection) in {SCHEMA_VERSION_V3, SCHEMA_VERSION_V4}:
        rebuild_semantic_projections(connection)
        rebuild_provider_health_projections(connection)


def append_semantic_generation(
    connection: sqlite3.Connection, *, semantic_generation_id: str, projection_generation_id: str,
    provider_fingerprint: str, dense_model_fingerprint: str | None,
    sparse_model_fingerprint: str | None, reranker_model_fingerprint: str | None,
    preprocessing_fingerprint: str, build_config_sha256: str,
    manifest_relative_path: str, manifest_sha256: str, semantic_database_sha256: str,
    created_at: str,
) -> None:
    if not isinstance(semantic_generation_id, str) or not semantic_generation_id.startswith("sem_"):
        raise CacheCorruptError("invalid semantic generation")
    if not manifest_relative_path.startswith("generations/") or ".." in manifest_relative_path.replace("\\", "/").split("/"):
        raise CacheCorruptError("invalid semantic generation path")
    if any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in (provider_fingerprint, preprocessing_fingerprint, build_config_sha256, manifest_sha256, semantic_database_sha256)):
        raise CacheCorruptError("invalid semantic generation hash")
    connection.execute(
        """INSERT INTO retrieval_semantic_generations(
            semantic_generation_id, projection_generation_id, provider_fingerprint,
            dense_model_fingerprint, sparse_model_fingerprint, reranker_model_fingerprint,
            preprocessing_fingerprint, build_config_sha256, manifest_relative_path,
            manifest_sha256, semantic_database_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (semantic_generation_id, projection_generation_id, provider_fingerprint,
         dense_model_fingerprint, sparse_model_fingerprint, reranker_model_fingerprint,
         preprocessing_fingerprint, build_config_sha256, manifest_relative_path,
         manifest_sha256, semantic_database_sha256, created_at),
    )


def _semantic_create_payload(raw: str) -> tuple[dict[str, str | None], list[str]]:
    payload = _json_object(raw)
    if set(payload) != {"component_degraded_reasons", "degraded_reasons"}:
        raise CacheCorruptError("semantic create payload is invalid")
    mapping = payload["component_degraded_reasons"]
    reasons = payload["degraded_reasons"]
    if (
        not isinstance(mapping, dict)
        or set(mapping) != _SEMANTIC_COMPONENTS
        or any(
            reason is not None
            and (not isinstance(reason, str) or reason not in _SEMANTIC_COMPONENT_REASONS)
            for reason in mapping.values()
        )
        or not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
    ):
        raise CacheCorruptError("semantic create payload is invalid")
    expected = sorted(
        {
            reason
            for reason in mapping.values()
            if reason is not None and reason != "disabled"
        }
    )
    if reasons != expected:
        raise CacheCorruptError("semantic create degraded reasons are invalid")
    return mapping, reasons


def _semantic_component_manifest_sha256(row: sqlite3.Row, reason: str | None) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "component": row["component"],
                "indexed_document_count": row["indexed_document_count"],
                "model_fingerprint": row["model_fingerprint"],
                "reason": reason,
                "semantic_generation_id": row["semantic_generation_id"],
                "status": row["status"],
            }
        ).encode("utf-8")
    ).hexdigest()


def _semantic_event_values(semantic_generation_id: str, action: str, payload_json: str, created_at: str) -> tuple[str, str]:
    preimage = {"action": action, "created_at": created_at, "semantic_generation_id": semantic_generation_id, "payload": _json_object(payload_json)}
    digest = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
    return f"sem_evt_{digest[:32]}", digest


def append_semantic_generation_event(connection: sqlite3.Connection, *, semantic_generation_id: str, action: str, created_at: str, payload: dict[str, Any] | None = None) -> None:
    if action not in EVENT_ACTIONS:
        raise CacheCorruptError("invalid semantic lifecycle action")
    payload_json = canonical_json({} if payload is None else payload)
    if action == "create":
        _semantic_create_payload(payload_json)
    event_id, event_sha256 = _semantic_event_values(semantic_generation_id, action, payload_json, created_at)
    connection.execute("INSERT INTO retrieval_semantic_generation_events(event_id, semantic_generation_id, action, payload_json, event_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?)", (event_id, semantic_generation_id, action, payload_json, event_sha256, created_at))


def append_semantic_component(
    connection: sqlite3.Connection, *, semantic_generation_id: str, component: str,
    status: str, model_fingerprint: str | None, preprocessing_fingerprint: str,
    vector_dimension: int | None, eligible_document_count: int,
    indexed_document_count: int, coverage_highwater: dict[str, Any],
    component_manifest_sha256: str, created_at: str,
) -> None:
    """Append one immutable component coverage row for a verified semantic build."""
    if (
        not isinstance(semantic_generation_id, str) or not semantic_generation_id.startswith("sem_")
        or component not in {"dense", "sparse", "reranker"}
        or status not in {"disabled", "ready", "partial", "unsupported", "unavailable", "failed"}
        or model_fingerprint is not None and not _HASH.fullmatch(model_fingerprint)
        or not _HASH.fullmatch(preprocessing_fingerprint)
        or vector_dimension is not None and (type(vector_dimension) is not int or vector_dimension < 1)
        or type(eligible_document_count) is not int or eligible_document_count < 0
        or type(indexed_document_count) is not int or not 0 <= indexed_document_count <= eligible_document_count
        or not _HASH.fullmatch(component_manifest_sha256)
        or not isinstance(created_at, str) or not _TIMESTAMP.fullmatch(created_at)
    ):
        raise CacheCorruptError("invalid semantic component")
    try:
        coverage = IndexHighwater.from_dict(coverage_highwater)
    except (TypeError, ValueError) as exc:
        raise CacheCorruptError("invalid semantic component coverage") from exc
    if coverage.to_dict() != coverage_highwater:
        raise CacheCorruptError("invalid semantic component coverage")
    coverage_json = canonical_json(coverage.to_dict())
    _verify_component_coverage_boundary(
        connection,
        semantic_generation_id=semantic_generation_id,
        status=status,
        coverage_json=coverage_json,
    )
    connection.execute(
        """INSERT INTO retrieval_semantic_components(
            semantic_generation_id, component, status, model_fingerprint,
            preprocessing_fingerprint, vector_dimension, eligible_document_count,
            indexed_document_count, coverage_highwater_json,
            component_manifest_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            semantic_generation_id, component, status, model_fingerprint,
            preprocessing_fingerprint, vector_dimension, eligible_document_count,
            indexed_document_count, coverage_json, component_manifest_sha256,
            created_at,
        ),
    )


def _verify_component_coverage_boundary(
    connection: sqlite3.Connection,
    *,
    semantic_generation_id: str,
    status: str,
    coverage_json: str,
) -> None:
    row = connection.execute(
        """SELECT projection.indexed_highwater_json
        FROM retrieval_semantic_generations AS semantic
        JOIN retrieval_generations AS projection
          ON projection.generation_id = semantic.projection_generation_id
        WHERE semantic.semantic_generation_id = ?""",
        (semantic_generation_id,),
    ).fetchone()
    if row is None:
        raise CacheCorruptError("semantic component projection is missing")
    expected = row[0] if status == "ready" else _ABSENT_SEMANTIC_COVERAGE_JSON
    if coverage_json != expected:
        raise CacheCorruptError("semantic component coverage boundary is invalid")


def rebuild_semantic_projections(connection: sqlite3.Connection) -> None:
    """Derive the only mutable semantic pointers from immutable events."""
    rows = list(connection.execute("SELECT * FROM retrieval_semantic_generation_events ORDER BY event_seq"))
    known = {row[0] for row in connection.execute("SELECT semantic_generation_id FROM retrieval_semantic_generations")}
    states: dict[str, tuple[str, sqlite3.Row]] = {}
    degraded_by_generation: dict[str, str] = {}
    active: str | None = None
    for row in rows:
        generation_id = row["semantic_generation_id"]
        if generation_id not in known or row["action"] not in EVENT_ACTIONS or _semantic_event_values(generation_id, row["action"], row["payload_json"], row["created_at"]) != (row["event_id"], row["event_sha256"]):
            raise CacheCorruptError("semantic lifecycle is invalid")
        prior = states.get(generation_id)
        action = row["action"]
        if action == "create" and prior is None:
            _, reasons = _semantic_create_payload(row["payload_json"])
            degraded_by_generation[generation_id] = canonical_json(reasons)
            state = "building"
        elif action == "validate" and prior is not None and prior[0] == "building": state = "ready"
        elif action == "activate" and prior is not None and prior[0] == "ready" and active is None: state, active = "active", generation_id
        elif action == "retire" and prior is not None and prior[0] == "active" and active == generation_id: state, active = "retired", None
        elif action == "fail" and prior is not None and prior[0] in {"building", "ready"}: state = "failed"
        else: raise CacheCorruptError("semantic lifecycle is invalid")
        states[generation_id] = (state, row)
    if any(state in {"building", "ready"} for state, _ in states.values()):
        raise CacheCorruptError("semantic lifecycle is incomplete")
    if set(states) != known:
        raise CacheCorruptError("semantic generation has no complete lifecycle")
    connection.execute("DELETE FROM retrieval_semantic_generation_current")
    connection.execute("DELETE FROM retrieval_semantic_state")
    for generation_id, (state, row) in states.items():
        connection.execute(
            "INSERT INTO retrieval_semantic_generation_current VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                generation_id,
                state,
                row["event_seq"],
                row["event_id"],
                "validated" if state in {"active", "retired"} else "failed",
                degraded_by_generation[generation_id],
                row["created_at"],
            ),
        )
    timestamp = max((row["created_at"] for _, row in states.values()), default="1970-01-01T00:00:00.000000Z")
    active_reasons = "[]" if active is None else degraded_by_generation[active]
    connection.execute(
        "INSERT INTO retrieval_semantic_state VALUES (1, ?, ?, ?, ?, ?)",
        (active, "validated" if active else "none", active_reasons, timestamp, timestamp),
    )


_HEALTH_ACTIONS = frozenset({"success", "failure", "open", "half_open", "probe_acquired", "probe_released"})


def _provider_health_event_values(
    provider_fingerprint: str, component: str, action: str, failure_kind: str | None,
    open_until: str | None, lease_until: str | None, payload_sha256: str, created_at: str,
) -> tuple[str, str]:
    preimage = {
        "action": action, "component": component, "created_at": created_at,
        "failure_kind": failure_kind, "lease_until": lease_until, "open_until": open_until,
        "payload_sha256": payload_sha256, "provider_fingerprint": provider_fingerprint,
    }
    digest = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
    return f"health_evt_{digest[:32]}", digest


def append_provider_health_event(
    connection: sqlite3.Connection, *, provider_fingerprint: str, component: str,
    action: str, failure_kind: str | None = None, open_until: str | None = None,
    lease_until: str | None = None, payload_sha256: str | None = None,
    created_at: str,
) -> None:
    if not _HASH.fullmatch(provider_fingerprint) or component not in _PROVIDER_HEALTH_COMPONENTS or action not in _HEALTH_ACTIONS:
        raise CacheCorruptError("invalid provider health event")
    if failure_kind is not None and (not isinstance(failure_kind, str) or not failure_kind):
        raise CacheCorruptError("invalid provider health event")
    for timestamp in (open_until, lease_until):
        if timestamp is not None and (not isinstance(timestamp, str) or not _TIMESTAMP.fullmatch(timestamp)):
            raise CacheCorruptError("invalid provider health event")
    if not isinstance(created_at, str) or not _TIMESTAMP.fullmatch(created_at):
        raise CacheCorruptError("invalid provider health event")
    payload_sha256 = payload_sha256 or hashlib.sha256(b"").hexdigest()
    if not _HASH.fullmatch(payload_sha256):
        raise CacheCorruptError("invalid provider health event")
    event_id, event_sha256 = _provider_health_event_values(provider_fingerprint, component, action, failure_kind, open_until, lease_until, payload_sha256, created_at)
    connection.execute("""INSERT INTO retrieval_provider_health_events(
        event_id, provider_fingerprint, component, action, failure_kind, open_until,
        lease_until, payload_sha256, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (event_id, provider_fingerprint, component, action, failure_kind, open_until, lease_until, payload_sha256, created_at))


def _provider_health_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CacheCorruptError("provider health timestamp is invalid") from exc


def _provider_health_projection_rows(connection: sqlite3.Connection) -> dict[tuple[str, str], tuple[str, int, str | None, str | None, int, str, str]]:
    state: dict[tuple[str, str], tuple[str, int, str | None, str | None, int, str, str]] = {}
    last_actions: dict[tuple[str, str], str | None] = {}
    pending_probe_failures: set[tuple[str, str]] = set()
    for row in connection.execute("SELECT * FROM retrieval_provider_health_events ORDER BY event_seq"):
        values = (row["provider_fingerprint"], row["component"], row["action"], row["failure_kind"], row["open_until"], row["lease_until"], row["payload_sha256"], row["created_at"])
        if (
            not isinstance(row["provider_fingerprint"], str) or not _HASH.fullmatch(row["provider_fingerprint"])
            or row["component"] not in _PROVIDER_HEALTH_COMPONENTS
            or row["action"] not in _HEALTH_ACTIONS
            or row["failure_kind"] is not None and (not isinstance(row["failure_kind"], str) or not row["failure_kind"])
            or any(value is not None and (not isinstance(value, str) or not _TIMESTAMP.fullmatch(value)) for value in (row["open_until"], row["lease_until"]))
            or not isinstance(row["created_at"], str) or not _TIMESTAMP.fullmatch(row["created_at"])
            or not isinstance(row["payload_sha256"], str) or not _HASH.fullmatch(row["payload_sha256"])
            or _provider_health_event_values(*values)[0] != row["event_id"]
        ):
            raise CacheCorruptError("provider health event is invalid")
        key = (row["provider_fingerprint"], row["component"])
        prior = state.get(key, ("closed", 0, None, None, 0, "", ""))
        prior_action = last_actions.get(key)
        if prior[6] and (
            _provider_health_timestamp(row["created_at"])
            < _provider_health_timestamp(prior[6])
        ):
            raise CacheCorruptError("provider health lifecycle is invalid")
        if row["action"] in {"success", "failure"}:
            # A terminal call may start from the closed state, or complete a
            # specifically admitted half-open probe.  An append-only event
            # history must not be able to close/reopen an open circuit without
            # the corresponding admission and probe evidence.
            if prior[0] == "open" or (
                prior[0] == "half_open" and prior_action != "probe_acquired"
            ):
                raise CacheCorruptError("provider health lifecycle is invalid")
            if key in pending_probe_failures:
                raise CacheCorruptError("provider health lifecycle is invalid")
            if prior[0] == "half_open" and _provider_health_timestamp(row["created_at"]) < _provider_health_timestamp(prior[6]):
                raise CacheCorruptError("provider health lifecycle is invalid")
            if row["action"] == "success":
                next_state = ("closed", 0, None, None)
            else:
                next_state = (prior[0], prior[1] + 1, prior[2], prior[3])
                if prior[0] == "half_open":
                    pending_probe_failures.add(key)
        elif row["action"] == "open":
            if (
                row["open_until"] is None
                or prior[0] not in {"closed", "half_open"}
                or prior[1] < 1
                or prior_action != "failure"
                or prior[0] == "half_open" and key not in pending_probe_failures
                or _provider_health_timestamp(row["open_until"]) <= _provider_health_timestamp(row["created_at"])
            ):
                raise CacheCorruptError("provider health lifecycle is invalid")
            next_state = ("open", prior[1], row["open_until"], None)
            pending_probe_failures.discard(key)
        elif row["action"] == "half_open":
            if (
                prior[0] != "open"
                or prior[2] is None
                or _provider_health_timestamp(row["created_at"])
                < _provider_health_timestamp(prior[2])
            ):
                raise CacheCorruptError("provider health lifecycle is invalid")
            next_state = ("half_open", prior[1], prior[2], None)
        elif row["action"] == "probe_acquired":
            if (
                prior[0] != "half_open"
                or prior[3] is not None
                or row["lease_until"] is None
                or _provider_health_timestamp(row["lease_until"])
                <= _provider_health_timestamp(row["created_at"])
            ):
                raise CacheCorruptError("provider health probe event is invalid")
            next_state = ("half_open", prior[1], prior[2], row["lease_until"])
        else:
            if (
                prior[0] != "half_open"
                or prior[3] is None
                or prior_action != "probe_acquired"
                or key in pending_probe_failures
                or _provider_health_timestamp(row["created_at"])
                < _provider_health_timestamp(prior[3])
            ):
                raise CacheCorruptError("provider health lifecycle is invalid")
            next_state = ("half_open", prior[1], prior[2], None)
        state[key] = (*next_state, row["event_seq"], row["event_id"], row["created_at"])
        last_actions[key] = row["action"]
    if pending_probe_failures:
        raise CacheCorruptError("provider health lifecycle is invalid")
    return state


def rebuild_provider_health_projections(connection: sqlite3.Connection) -> None:
    expected = _provider_health_projection_rows(connection)
    connection.execute("DELETE FROM retrieval_provider_health_current")
    for (provider_fingerprint, component), row in expected.items():
        connection.execute("INSERT INTO retrieval_provider_health_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (provider_fingerprint, component, *row))


def _replay(connection: sqlite3.Connection) -> tuple[dict[str, dict[str, Any]], str | None]:
    generation_ids = {row[0] for row in connection.execute("SELECT generation_id FROM retrieval_generations")}
    states: dict[str, dict[str, Any]] = {}
    active: str | None = None
    for event in connection.execute("SELECT * FROM retrieval_generation_events ORDER BY event_seq"):
        generation_id = event["generation_id"]
        if generation_id not in generation_ids:
            raise CacheCorruptError("lifecycle references missing generation")
        if event["action"] not in EVENT_ACTIONS:
            raise CacheCorruptError("invalid lifecycle action")
        try:
            payload = _json_object(event["payload_json"])
        except CacheCorruptError:
            raise CacheCorruptError("invalid lifecycle payload") from None
        expected_id, expected_sha = _event_values(generation_id, event["action"], event["payload_json"], event["created_at"])
        if event["event_id"] != expected_id or event["event_sha256"] != expected_sha:
            raise CacheCorruptError("lifecycle event hash mismatch")
        if payload:
            raise CacheCorruptError("lifecycle payload must be empty")
        prior = states.get(generation_id)
        action = event["action"]
        if action == "create":
            if prior is not None:
                raise CacheCorruptError("invalid lifecycle create")
            state, validation = "building", "pending"
        elif action == "validate":
            if prior is None or prior["state"] != "building":
                raise CacheCorruptError("invalid lifecycle validation")
            state, validation = "ready", "validated"
        elif action == "activate":
            if prior is None or prior["state"] != "ready" or active is not None:
                raise CacheCorruptError("invalid lifecycle activation")
            state, validation, active = "active", "validated", generation_id
        elif action == "retire":
            if prior is None or prior["state"] != "active" or active != generation_id:
                raise CacheCorruptError("invalid lifecycle retirement")
            state, validation, active = "retired", "validated", None
        else:
            if prior is None or prior["state"] not in {"building", "ready"}:
                raise CacheCorruptError("invalid lifecycle failure")
            state, validation = "failed", "failed"
        states[generation_id] = {
            "state": state,
            "event_seq": event["event_seq"],
            "event_id": event["event_id"],
            "validation_state": validation,
            "degraded_reason": None,
            "updated_at": event["created_at"],
        }
    if set(states) != generation_ids:
        raise CacheCorruptError("generation has no complete lifecycle")
    if any(value["state"] in {"building", "ready"} for value in states.values()):
        raise CacheCorruptError("generation lifecycle is incomplete")
    return states, active


def verify_legacy_schema_v1(connection: sqlite3.Connection) -> None:
    """Validate schema, immutable history, projections, and SQLite integrity."""
    try:
        meta_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_schema_meta'"
        ).fetchone()
        if not meta_exists:
            raise CacheCorruptError("retrieval schema metadata is missing")
        meta = list(connection.execute("SELECT key, value FROM retrieval_schema_meta"))
        if len(meta) != 1 or meta[0]["key"] != "schema_version":
            raise CacheCorruptError("retrieval schema metadata is corrupt")
        if meta[0]["value"] != str(LEGACY_SCHEMA_VERSION):
            raise CacheUnsupportedError("unsupported retrieval schema")
        actual_schema = {
            (row["type"], row["name"]): _normal_sql(row["sql"])
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name != 'sqlite_sequence'"
            )
            if row["name"] == "retrieval_documents_fts"
            or not row["name"].startswith("retrieval_documents_fts_")
        }
        if actual_schema != _LEGACY_EXPECTED_SCHEMA_SQL:
            raise CacheCorruptError("retrieval schema definition is invalid")
        for table, columns in _LEGACY_TABLE_COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != columns:
                raise CacheCorruptError("retrieval schema columns are invalid")
        _verify_schema_pragmas(connection)
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise CacheCorruptError("retrieval cache integrity check failed")
        if list(connection.execute("PRAGMA foreign_key_check")):
            raise CacheCorruptError("retrieval cache foreign key check failed")
        _verify_generation_metadata(connection, schema_version=LEGACY_SCHEMA_VERSION)
        states, active = _replay(connection)
        actual_current = {
            row["generation_id"]: dict(row)
            for row in connection.execute("SELECT * FROM retrieval_generation_current")
        }
        if set(actual_current) != set(states):
            raise CacheCorruptError("retrieval projection is incomplete")
        for generation_id, expected in states.items():
            current = actual_current[generation_id]
            field_map = {
                "state": "state", "last_event_seq": "event_seq", "last_event_id": "event_id",
                "validation_state": "validation_state", "degraded_reason": "degraded_reason", "updated_at": "updated_at",
            }
            for name, expected_name in field_map.items():
                if current[name] != expected[expected_name]:
                    raise CacheCorruptError("retrieval projection does not replay")
            if current["failure_metadata_json"] != "{}":
                raise CacheCorruptError("retrieval failure metadata is invalid")
        rows = list(connection.execute("SELECT * FROM retrieval_state"))
        if len(rows) != 1 or rows[0]["singleton_id"] != 1 or rows[0]["active_generation_id"] != active:
            raise CacheCorruptError("retrieval active state is invalid")
        timestamp = max((value["updated_at"] for value in states.values()), default="1970-01-01T00:00:00.000000Z")
        if (
            rows[0]["validation_state"] != ("validated" if active else "none")
            or rows[0]["degraded_reason"] is not None
            or rows[0]["failure_metadata_json"] != "{}"
            or rows[0]["created_at"] != timestamp
            or rows[0]["updated_at"] != timestamp
        ):
            raise CacheCorruptError("retrieval state is invalid")
    except CacheCorruptError:
        raise
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise CacheCorruptError("retrieval cache is corrupt") from exc


def verify_cache_database(connection: sqlite3.Connection) -> None:
    version = inspect_cache_schema_version(connection)
    if version is None:
        raise CacheCorruptError("retrieval schema metadata is missing")
    if version == SCHEMA_VERSION_V2:
        _verify_cache_database_v2(connection)
        return
    if version == SCHEMA_VERSION_V3:
        _verify_cache_database_v3(connection)
        return
    if version == SCHEMA_VERSION_V4:
        _verify_cache_database_v4(connection)
        return
    if version == LEGACY_SCHEMA_VERSION:
        raise CacheUnsupportedError("unsupported retrieval schema")
    raise CacheUnsupportedError("unsupported retrieval schema")


def _verify_schema_pragmas(connection: sqlite3.Connection) -> None:
    expected_indexes = {
        "retrieval_generation_events": ("retrieval_generation_events_generation_seq", ("generation_id", "event_seq"), 0),
        "retrieval_generation_current": ("retrieval_generation_current_state_generation", ("state", "generation_id"), 0),
    }
    for table, (name, columns, unique) in expected_indexes.items():
        matches = [row for row in connection.execute(f"PRAGMA index_list({table})") if row[1] == name]
        if len(matches) != 1 or matches[0][2] != unique:
            raise CacheCorruptError("retrieval schema indexes are invalid")
        if tuple(row[2] for row in connection.execute(f"PRAGMA index_info({name})")) != columns:
            raise CacheCorruptError("retrieval schema index columns are invalid")
    expected_foreign_keys = {
        "retrieval_generation_events": {("generation_id", "retrieval_generations", "generation_id")},
        "retrieval_generation_current": {
            ("generation_id", "retrieval_generations", "generation_id"),
            ("last_event_id", "retrieval_generation_events", "event_id"),
        },
        "retrieval_state": {("active_generation_id", "retrieval_generations", "generation_id")},
    }
    for table, expected in expected_foreign_keys.items():
        actual = {(row[3], row[2], row[4]) for row in connection.execute(f"PRAGMA foreign_key_list({table})")}
        if actual != expected:
            raise CacheCorruptError("retrieval schema foreign keys are invalid")


def _verify_generation_metadata(connection: sqlite3.Connection, *, schema_version: int = SCHEMA_VERSION_V2) -> None:
    for row in connection.execute("SELECT * FROM retrieval_generations ORDER BY generation_seq"):
        try:
            canonical = IndexHighwater.from_dict(json.loads(row["canonical_highwater_json"]))
            indexed = IndexHighwater.from_dict(json.loads(row["indexed_highwater_json"]))
        except (TypeError, ValueError, CacheCorruptError) as exc:
            raise CacheCorruptError("generation highwater is invalid") from exc
        if (
            canonical_json(canonical.to_dict()) != row["canonical_highwater_json"]
            or canonical_json(indexed.to_dict()) != row["indexed_highwater_json"]
            or canonical != indexed
        ):
            raise CacheCorruptError("generation highwater is invalid")
        if not isinstance(row["canonical_content_fingerprint"], str) or not _HASH.fullmatch(row["canonical_content_fingerprint"]):
            raise CacheCorruptError("generation fingerprint is invalid")
        if row["provider_fingerprint"] != _PROVIDER or row["model_fingerprint"] != _MODEL:
            raise CacheCorruptError("generation provider metadata is invalid")
        if not isinstance(row["created_at"], str) or not _TIMESTAMP.fullmatch(row["created_at"]):
            raise CacheCorruptError("generation timestamp is invalid")
        row_schema_version = row["retrieval_schema_version"] if "retrieval_schema_version" in row.keys() else schema_version
        if row_schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION_V2, SCHEMA_VERSION_V3}:
            raise CacheCorruptError("generation schema version is invalid")
        if row["generation_id"] != deterministic_generation_id(
            canonical,
            row["canonical_content_fingerprint"],
            schema_version=row_schema_version,
        ):
            raise CacheCorruptError("generation id is not deterministic")


def _verify_generation_projections(connection: sqlite3.Connection) -> None:
    """Verify mutable generation pointers are a pure replay of immutable events."""
    states, active = _replay(connection)
    actual_current = {
        row["generation_id"]: dict(row)
        for row in connection.execute("SELECT * FROM retrieval_generation_current")
    }
    if set(actual_current) != set(states):
        raise CacheCorruptError("retrieval projection is incomplete")
    for generation_id, expected in states.items():
        current = actual_current[generation_id]
        field_map = {
            "state": "state", "last_event_seq": "event_seq", "last_event_id": "event_id",
            "validation_state": "validation_state", "degraded_reason": "degraded_reason",
            "updated_at": "updated_at",
        }
        for name, expected_name in field_map.items():
            if current[name] != expected[expected_name]:
                raise CacheCorruptError("retrieval projection does not replay")
        if current["failure_metadata_json"] != "{}":
            raise CacheCorruptError("retrieval failure metadata is invalid")
    rows = list(connection.execute("SELECT * FROM retrieval_state"))
    if len(rows) != 1 or rows[0]["singleton_id"] != 1 or rows[0]["active_generation_id"] != active:
        raise CacheCorruptError("retrieval active state is invalid")
    timestamp = max((value["updated_at"] for value in states.values()), default="1970-01-01T00:00:00.000000Z")
    if (
        rows[0]["validation_state"] != ("validated" if active else "none")
        or rows[0]["degraded_reason"] is not None
        or rows[0]["failure_metadata_json"] != "{}"
        or rows[0]["created_at"] != timestamp
        or rows[0]["updated_at"] != timestamp
    ):
        raise CacheCorruptError("retrieval state is invalid")


def _verify_cache_database_v2(connection: sqlite3.Connection) -> None:
    try:
        fts5_present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
        ).fetchone() is not None
        actual_schema = {
            (row["type"], row["name"]): _normal_sql(row["sql"])
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name != 'sqlite_sequence'"
            )
            if row["name"] == "retrieval_documents_fts"
            or not row["name"].startswith("retrieval_documents_fts_")
        }
        expected_schema = _schema_sql_from_statements(_v2_schema_statements(fts5_present))
        if actual_schema != expected_schema:
            raise CacheCorruptError("retrieval schema definition is invalid")
        expected_columns = dict(_V2_TABLE_COLUMNS)
        if fts5_present:
            expected_columns["retrieval_documents_fts"] = (
                "document_id",
                "generation_id",
                "search_text",
            )
        for table, columns in expected_columns.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != columns:
                raise CacheCorruptError("retrieval schema columns are invalid")
        _verify_schema_pragmas_v2(connection, fts5_present)
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise CacheCorruptError("retrieval cache integrity check failed")
        if list(connection.execute("PRAGMA foreign_key_check")):
            raise CacheCorruptError("retrieval cache foreign key check failed")
        _verify_generation_metadata(connection)
        _verify_generation_capabilities(connection, fts5_present)
        _verify_documents(connection)
        _verify_document_sources(connection)
        _verify_symbols(connection)
        _verify_runs(connection)
        _verify_candidate_traces(connection)
    except CacheCorruptError:
        raise
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise CacheCorruptError("retrieval cache is corrupt") from exc


def _verify_cache_database_v3(connection: sqlite3.Connection) -> None:
    _verify_cache_database_v3_or_v4(
        connection, _v3_schema_statements, _V3_TABLE_COLUMNS,
        schema_version=SCHEMA_VERSION_V3,
    )


def _verify_cache_database_v4(connection: sqlite3.Connection) -> None:
    _verify_cache_database_v3_or_v4(
        connection, _v4_schema_statements, _V4_TABLE_COLUMNS,
        schema_version=SCHEMA_VERSION_V3,
    )
    verify_provider_attempt_history(connection)


def verify_provider_attempt_history(connection: sqlite3.Connection) -> None:
    """Verify every immutable provider-control row before use or rebuild copy."""
    try:
        _verify_provider_attempt_history(connection)
    except CacheCorruptError:
        raise
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise CacheCorruptError("provider attempt history is corrupt") from exc


def _verify_provider_attempt_history(connection: sqlite3.Connection) -> None:
    known_attempts: set[str] = set()
    latest_fence: dict[tuple[str, int], int] = {}
    for row in connection.execute(
        "SELECT * FROM retrieval_provider_attempt_admissions ORDER BY attempt_seq"
    ):
        capability = row["capability"]
        operation = row["operation"]
        preimage = {
            "attempt_id": row["attempt_id"],
            "capability": capability,
            "circuit_key": row["circuit_key"],
            "expires_at": row["expires_at"],
            "fencing_version": row["fencing_version"],
            "history_epoch": row["history_epoch"],
            "operation": operation,
            "provider_fingerprint": row["provider_fingerprint"],
            "admitted_at": row["admitted_at"],
        }
        expected_circuit = hashlib.sha256(
            f"provider:{row['provider_fingerprint']}".encode("ascii")
        ).hexdigest()
        key = (row["circuit_key"], row["history_epoch"])
        if (
            not isinstance(row["attempt_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", row["attempt_id"])
            or not isinstance(row["provider_fingerprint"], str)
            or not _HASH.fullmatch(row["provider_fingerprint"])
            or capability not in _PROVIDER_HEALTH_COMPONENTS
            or operation not in _PROVIDER_OPERATIONS
            or (capability == "capabilities") != (operation == "capabilities")
            or row["circuit_key"] != expected_circuit
            or row["history_epoch"] != 1
            or row["fencing_version"] != latest_fence.get(key, 0) + 1
            or not isinstance(row["admitted_at"], str)
            or not _TIMESTAMP.fullmatch(row["admitted_at"])
            or not isinstance(row["expires_at"], str)
            or not _TIMESTAMP.fullmatch(row["expires_at"])
            or _provider_health_timestamp(row["expires_at"])
            <= _provider_health_timestamp(row["admitted_at"])
            or hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
            != row["payload_sha256"]
        ):
            raise CacheCorruptError("provider attempt history is corrupt")
        latest_fence[key] = row["fencing_version"]
        known_attempts.add(row["attempt_id"])

    settled: set[str] = set()
    for row in connection.execute(
        "SELECT * FROM retrieval_provider_attempt_results ORDER BY result_seq"
    ):
        preimage = {
            "attempt_id": row["attempt_id"],
            "outcome": row["outcome"],
            "provider_status": row["provider_status"],
            "recorded_at": row["recorded_at"],
        }
        digest = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
        outcome = row["outcome"]
        status = row["provider_status"]
        if (
            row["attempt_id"] not in known_attempts
            or row["attempt_id"] in settled
            or outcome not in {"success", "failure", "released", "expired_unknown"}
            or status is not None and status not in _PROVIDER_STATUSES
            or outcome == "success" and status != "ok"
            or outcome == "failure" and status not in {
                "unavailable", "busy", "timeout", "invalid_response"
            }
            or outcome in {"released", "expired_unknown"} and status is not None
            or not isinstance(row["recorded_at"], str)
            or not _TIMESTAMP.fullmatch(row["recorded_at"])
            or row["payload_sha256"] != digest
            or row["result_id"] != f"attempt_result_{digest[:32]}"
        ):
            raise CacheCorruptError("provider attempt history is corrupt")
        settled.add(row["attempt_id"])


def _verify_cache_database_v3_or_v4(
    connection: sqlite3.Connection,
    schema_statements,
    table_columns,
    *,
    schema_version: int,
) -> None:
    try:
        fts5_present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_documents_fts'"
        ).fetchone() is not None
        actual_schema = {
            (row["type"], row["name"]): _normal_sql(row["sql"])
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name != 'sqlite_sequence'"
            )
            if row["name"] == "retrieval_documents_fts" or not row["name"].startswith("retrieval_documents_fts_")
        }
        if actual_schema != _schema_sql_from_statements(schema_statements(fts5_present)):
            raise CacheCorruptError("retrieval schema definition is invalid")
        columns = dict(table_columns)
        if fts5_present:
            columns["retrieval_documents_fts"] = ("document_id", "generation_id", "search_text")
        for table, expected in columns.items():
            if tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})")) != expected:
                raise CacheCorruptError("retrieval schema columns are invalid")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or list(connection.execute("PRAGMA foreign_key_check")):
            raise CacheCorruptError("retrieval cache integrity check failed")
        _verify_generation_metadata(connection, schema_version=schema_version)
        _verify_generation_projections(connection)
        _verify_generation_capabilities(connection, fts5_present)
        _verify_documents(connection); _verify_document_sources(connection); _verify_symbols(connection)
        _verify_runs(connection); _verify_candidate_traces(connection)
        _verify_semantic_metadata(connection)
    except CacheCorruptError:
        raise
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise CacheCorruptError("retrieval cache is corrupt") from exc


def _verify_semantic_metadata(connection: sqlite3.Connection) -> None:
    for row in connection.execute("SELECT * FROM retrieval_semantic_generations ORDER BY semantic_generation_seq"):
        for field in ("provider_fingerprint", "preprocessing_fingerprint", "build_config_sha256", "manifest_sha256", "semantic_database_sha256"):
            if not isinstance(row[field], str) or not _HASH.fullmatch(row[field]):
                raise CacheCorruptError("semantic generation hash is invalid")
        if not isinstance(row["manifest_relative_path"], str) or not row["manifest_relative_path"].startswith("generations/"):
            raise CacheCorruptError("semantic generation path is invalid")
        if not isinstance(row["created_at"], str) or not _TIMESTAMP.fullmatch(row["created_at"]):
            raise CacheCorruptError("semantic generation timestamp is invalid")
    component_rows: dict[str, dict[str, sqlite3.Row]] = {}
    for row in connection.execute("SELECT * FROM retrieval_semantic_components ORDER BY semantic_generation_id, component"):
        if (
            row["component"] not in _SEMANTIC_COMPONENTS
            or row["status"] not in _SEMANTIC_COMPONENT_STATUSES
            or type(row["eligible_document_count"]) is not int
            or type(row["indexed_document_count"]) is not int
            or row["indexed_document_count"] > row["eligible_document_count"]
            or row["indexed_document_count"] < 0
        ):
            raise CacheCorruptError("semantic component counts are invalid")
        try:
            coverage = IndexHighwater.from_dict(_json_object(row["coverage_highwater_json"]))
        except (TypeError, ValueError) as exc:
            raise CacheCorruptError("semantic component coverage is invalid") from exc
        if canonical_json(coverage.to_dict()) != row["coverage_highwater_json"]:
            raise CacheCorruptError("semantic component coverage is invalid")
        _verify_component_coverage_boundary(
            connection,
            semantic_generation_id=row["semantic_generation_id"],
            status=row["status"],
            coverage_json=row["coverage_highwater_json"],
        )
        if (
            not isinstance(row["component_manifest_sha256"], str)
            or not _HASH.fullmatch(row["component_manifest_sha256"])
            or row["model_fingerprint"] is not None
            and not _HASH.fullmatch(row["model_fingerprint"])
            or row["component"] != "dense" and row["vector_dimension"] is not None
            or row["vector_dimension"] is not None
            and (type(row["vector_dimension"]) is not int or row["vector_dimension"] < 1)
        ):
            raise CacheCorruptError("semantic component hash is invalid")
        component_rows.setdefault(row["semantic_generation_id"], {})[row["component"]] = row
    known = {row[0] for row in connection.execute("SELECT semantic_generation_id FROM retrieval_semantic_generations")}
    states: dict[str, tuple[str, sqlite3.Row]] = {}
    degraded_by_generation: dict[str, str] = {}
    reasons_by_generation: dict[str, dict[str, str | None]] = {}
    active: str | None = None
    for row in connection.execute("SELECT * FROM retrieval_semantic_generation_events ORDER BY event_seq"):
        if row["semantic_generation_id"] not in known or row["action"] not in EVENT_ACTIONS or _semantic_event_values(row["semantic_generation_id"], row["action"], row["payload_json"], row["created_at"]) != (row["event_id"], row["event_sha256"]):
            raise CacheCorruptError("semantic lifecycle event is invalid")
        generation_id, action = row["semantic_generation_id"], row["action"]
        prior = states.get(generation_id)
        if action == "create" and prior is None:
            reason_mapping, degraded_reasons = _semantic_create_payload(row["payload_json"])
            reasons_by_generation[generation_id] = reason_mapping
            degraded_by_generation[generation_id] = canonical_json(degraded_reasons)
            state = "building"
        elif action == "validate" and prior is not None and prior[0] == "building": state = "ready"
        elif action == "activate" and prior is not None and prior[0] == "ready" and active is None: state, active = "active", generation_id
        elif action == "retire" and prior is not None and prior[0] == "active" and active == generation_id: state, active = "retired", None
        elif action == "fail" and prior is not None and prior[0] in {"building", "ready"}: state = "failed"
        else: raise CacheCorruptError("semantic lifecycle projection is invalid")
        states[generation_id] = (state, row)
    if any(state in {"building", "ready"} for state, _ in states.values()):
        raise CacheCorruptError("semantic lifecycle is incomplete")
    if set(states) != known:
        raise CacheCorruptError("semantic generation has no complete lifecycle")
    if set(component_rows) != known or any(
        set(rows) != _SEMANTIC_COMPONENTS for rows in component_rows.values()
    ):
        raise CacheCorruptError("semantic components are incomplete")
    for generation_id, rows in component_rows.items():
        mapping = reasons_by_generation.get(generation_id)
        if mapping is None:
            raise CacheCorruptError("semantic component reasons are missing")
        for component, row in rows.items():
            reason = mapping[component]
            if row["component_manifest_sha256"] != _semantic_component_manifest_sha256(row, reason):
                raise CacheCorruptError("semantic component reason binding is invalid")
            if row["status"] == "ready" and (
                row["indexed_document_count"] != row["eligible_document_count"]
                or reason is not None
            ):
                raise CacheCorruptError("ready semantic component is invalid")
            if row["status"] == "partial" and (
                not 0 < row["indexed_document_count"] < row["eligible_document_count"]
                or reason is None
            ):
                raise CacheCorruptError("partial semantic component is invalid")
            if row["status"] == "disabled" and (
                row["indexed_document_count"] != 0
                or row["model_fingerprint"] is not None
                or reason != "disabled"
            ):
                raise CacheCorruptError("disabled semantic component is invalid")
            if row["status"] == "unsupported" and (
                row["indexed_document_count"] != 0
                or row["model_fingerprint"] is not None
                or reason != "unsupported"
            ):
                raise CacheCorruptError("unsupported semantic component is invalid")
            if row["status"] == "unavailable" and (
                row["indexed_document_count"] != 0
                or row["model_fingerprint"] is not None
                or reason
                not in {"unavailable", "busy", "timeout", "circuit_open", "deadline_exhausted"}
            ):
                raise CacheCorruptError("unavailable semantic component is invalid")
            if row["status"] == "failed" and reason is None:
                raise CacheCorruptError("failed semantic component is invalid")
    actual_current = {row["semantic_generation_id"]: row for row in connection.execute("SELECT * FROM retrieval_semantic_generation_current")}
    if set(actual_current) != set(states):
        raise CacheCorruptError("semantic lifecycle projection does not replay")
    for generation_id, (state, event) in states.items():
        row = actual_current[generation_id]
        if (row["state"], row["last_event_seq"], row["last_event_id"], row["validation_state"], row["degraded_reasons_json"], row["updated_at"]) != (state, event["event_seq"], event["event_id"], "validated" if state in {"active", "retired"} else "failed", degraded_by_generation[generation_id], event["created_at"]):
            raise CacheCorruptError("semantic lifecycle projection does not replay")
    state_row = connection.execute("SELECT * FROM retrieval_semantic_state").fetchone()
    timestamp = max((event["created_at"] for _, event in states.values()), default="1970-01-01T00:00:00.000000Z")
    active_reasons = "[]" if active is None else degraded_by_generation[active]
    if state_row is None or tuple(state_row) != (1, active, "validated" if active is not None else "none", active_reasons, timestamp, timestamp):
        raise CacheCorruptError("semantic active state does not replay")
    actual_health = {
        (row["provider_fingerprint"], row["component"]): tuple(row)[2:]
        for row in connection.execute("SELECT * FROM retrieval_provider_health_current ORDER BY provider_fingerprint, component")
    }
    expected_health = _provider_health_projection_rows(connection)
    if actual_health != expected_health:
        raise CacheCorruptError("provider health projection does not replay")


def _verify_schema_pragmas_v2(connection: sqlite3.Connection, fts5_present: bool) -> None:
    _verify_schema_pragmas(connection)
    expected_indexes = {
        "retrieval_documents": ("retrieval_documents_generation_seq", ("generation_id", "document_seq"), 0),
        "retrieval_document_sources": (
            "retrieval_document_sources_document_seq",
            ("generation_id", "document_id", "ordinal"),
            0,
        ),
        "retrieval_symbols": ("retrieval_symbols_document_seq", ("generation_id", "document_id", "ordinal"), 0),
        "retrieval_runs": ("retrieval_runs_generation_seq", ("generation_id", "run_seq"), 0),
        "retrieval_candidate_traces": (
            "retrieval_candidate_traces_run_seq",
            ("run_id", "trace_seq"),
            0,
        ),
    }
    for table, (name, columns, unique) in expected_indexes.items():
        matches = [row for row in connection.execute(f"PRAGMA index_list({table})") if row[1] == name]
        if len(matches) != 1 or matches[0][2] != unique:
            raise CacheCorruptError("retrieval schema indexes are invalid")
        if tuple(row[2] for row in connection.execute(f"PRAGMA index_info({name})")) != columns:
            raise CacheCorruptError("retrieval schema index columns are invalid")
    if fts5_present:
        # The virtual table has no foreign key metadata, but its presence is part of the frozen profile.
        pass
    expected_foreign_keys = {
        "retrieval_generation_capabilities": {("generation_id", "retrieval_generations", "generation_id")},
        "retrieval_documents": {("generation_id", "retrieval_generations", "generation_id")},
        "retrieval_document_sources": {
            ("generation_id", "retrieval_generations", "generation_id"),
            ("document_id", "retrieval_documents", "document_id"),
        },
        "retrieval_symbols": {
            ("generation_id", "retrieval_generations", "generation_id"),
            ("document_id", "retrieval_documents", "document_id"),
        },
        "retrieval_runs": {("generation_id", "retrieval_generations", "generation_id")},
        "retrieval_candidate_traces": {("run_id", "retrieval_runs", "run_id")},
    }
    for table, expected in expected_foreign_keys.items():
        actual = {(row[3], row[2], row[4]) for row in connection.execute(f"PRAGMA foreign_key_list({table})")}
        if actual != expected:
            raise CacheCorruptError("retrieval schema foreign keys are invalid")


def _verify_generation_capabilities(connection: sqlite3.Connection, fts5_present: bool) -> None:
    for row in connection.execute(
        "SELECT * FROM retrieval_generation_capabilities ORDER BY generation_id"
    ):
        if row["fts5_state"] != ("available" if fts5_present else "unavailable"):
            raise CacheCorruptError("retrieval capabilities profile is invalid")
        if row["sqlite_version"] != sqlite3.sqlite_version:
            raise CacheCorruptError("retrieval capabilities sqlite version is invalid")
        for field in ("index_policy_fingerprint", "projection_manifest_sha256"):
            if not isinstance(row[field], str) or not _HASH.fullmatch(row[field]):
                raise CacheCorruptError("retrieval capabilities hash is invalid")
        for field in ("projected_document_count", "omitted_unsafe_count"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
                raise CacheCorruptError("retrieval capabilities count is invalid")
        if not isinstance(row["created_at"], str) or not _TIMESTAMP.fullmatch(row["created_at"]):
            raise CacheCorruptError("retrieval capabilities timestamp is invalid")


def _verify_documents(connection: sqlite3.Connection) -> None:
    documents = list(connection.execute("SELECT * FROM retrieval_documents ORDER BY document_seq"))
    source_counts = {
        row["document_id"]: row["count"]
        for row in connection.execute(
            "SELECT document_id, COUNT(*) AS count FROM retrieval_document_sources GROUP BY document_id"
        )
    }
    for row in documents:
        if not isinstance(row["document_id"], str) or not row["document_id"]:
            raise CacheCorruptError("retrieval document identifier is invalid")
        if not isinstance(row["generation_id"], str) or not row["generation_id"]:
            raise CacheCorruptError("retrieval document generation is invalid")
        if row["evidence_count"] != source_counts.get(row["document_id"], 0):
            raise CacheCorruptError("retrieval document evidence count is invalid")
        for field in ("content_sha256", "document_sha256"):
            if not isinstance(row[field], str) or not _HASH.fullmatch(row[field]):
                raise CacheCorruptError("retrieval document hash is invalid")


def _verify_document_sources(connection: sqlite3.Connection) -> None:
    for row in connection.execute(
        "SELECT s.*, d.generation_id AS document_generation_id FROM retrieval_document_sources AS s JOIN retrieval_documents AS d ON d.document_id = s.document_id ORDER BY s.source_seq"
    ):
        if row["generation_id"] != row["document_generation_id"]:
            raise CacheCorruptError("retrieval document source generation mismatch")
        if not isinstance(row["source_sha256"], str) or not _HASH.fullmatch(row["source_sha256"]):
            raise CacheCorruptError("retrieval document source hash is invalid")


def _verify_symbols(connection: sqlite3.Connection) -> None:
    for row in connection.execute(
        "SELECT * FROM retrieval_symbols ORDER BY symbol_seq"
    ):
        if not isinstance(row["symbol_sha256"], str) or not _HASH.fullmatch(row["symbol_sha256"]):
            raise CacheCorruptError("retrieval symbol hash is invalid")


def _verify_runs(connection: sqlite3.Connection) -> None:
    for row in connection.execute("SELECT * FROM retrieval_runs ORDER BY run_seq"):
        if not isinstance(row["query_sha256"], str) or not _HASH.fullmatch(row["query_sha256"]):
            raise CacheCorruptError("retrieval run hash is invalid")
        if not isinstance(row["trace_sha256"], str) or not _HASH.fullmatch(row["trace_sha256"]):
            raise CacheCorruptError("retrieval run hash is invalid")
        reasons = _json_object_or_array(row["degraded_reasons_json"], expect_array=True)
        if any(not isinstance(value, str) for value in reasons):
            raise CacheCorruptError("retrieval run degraded reasons are invalid")


def _verify_candidate_traces(connection: sqlite3.Connection) -> None:
    allowed_channels = {
        "mandatory_constitution",
        "temporary",
        "exact",
        "bm25",
        "symbol",
        "recent",
        "stale_delta",
    }
    for row in connection.execute("SELECT * FROM retrieval_candidate_traces ORDER BY trace_seq"):
        if row["channel"] not in allowed_channels:
            raise CacheCorruptError("retrieval candidate trace channel is invalid")
        if isinstance(row["selected"], bool) or row["selected"] not in (0, 1):
            raise CacheCorruptError("retrieval candidate trace selection is invalid")
        if row["fused_rank"] is not None and (isinstance(row["fused_rank"], bool) or row["fused_rank"] < 1):
            raise CacheCorruptError("retrieval candidate trace rank is invalid")
        for field in ("trace_sha256",):
            if not isinstance(row[field], str) or not _HASH.fullmatch(row[field]):
                raise CacheCorruptError("retrieval candidate trace hash is invalid")
        reasons = _json_object_or_array(row["reason_codes_json"], expect_array=True)
        if any(not isinstance(value, str) for value in reasons):
            raise CacheCorruptError("retrieval candidate trace reason codes are invalid")


def _json_object_or_array(raw: str, *, expect_array: bool = False) -> Any:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise CacheCorruptError("invalid JSON") from exc
    if expect_array and not isinstance(value, list):
        raise CacheCorruptError("invalid JSON")
    if not expect_array and not isinstance(value, dict):
        raise CacheCorruptError("invalid JSON")
    if canonical_json(value) != raw:
        raise CacheCorruptError("noncanonical JSON")
    return value


def _json_object(raw: str) -> dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise CacheCorruptError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise CacheCorruptError("invalid JSON")
    if canonical_json(value) != raw:
        raise CacheCorruptError("noncanonical JSON")
    return value
