"""Safe serialization and immutable query helpers for the retrieval service."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from codex_memory.ledger_common import canonical_json

from .contracts import RetrievalQueryResult


def serialize_query_result(result: RetrievalQueryResult, *, explain: bool = False) -> str:
    """Serialize the public query shape; traces are opt-in and no request text is present."""
    return canonical_json(query_result_document(result, explain=explain)) + "\n"


def query_result_document(result: RetrievalQueryResult, *, explain: bool = False) -> dict[str, object]:
    """Produce a public-only result document without paths or model input text.

    Optional semantic raw scores live only inside the opt-in trace; they are
    finite ``float.hex`` diagnostics validated by the contract, never candidate
    authority or an ordinary non-explain result field.
    """
    document: dict[str, object] = result.to_dict()
    for candidate in document["candidates"]:
        if _absolute_path(candidate["scope_key"]):
            candidate["scope_key"] = "<local-scope>"
    if not explain:
        document.pop("trace")
    return document


def _absolute_path(value: object) -> bool:
    return isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
