"""Audited, fixture-only tests for the local dense HTTP adapter.

The fixture server deliberately never exposes a real endpoint, credential, or
model request outside the process.  It records only the protocol properties
needed to prove the adapter contract.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from codex_memory.retrieval import config as config_module
from codex_memory.retrieval import provider as provider_module
from codex_memory.retrieval.config import SemanticConfig
from codex_memory.retrieval.deadline import (
    AbsoluteDeadlineMilliseconds,
    DeadlineBudget,
)
from codex_memory.retrieval.provider import (
    DisabledProvider,
    ProviderCallStatus,
    ProviderInput,
)


def _config(*, base_url: str, dense: bool = True, sparse: bool = False, reranker: bool = False) -> SemanticConfig:
    return SemanticConfig(
        1, "local_http", base_url, "audited-local-v1", 8_000, 300_000,
        500, 4_000, dense, sparse, reranker, 64, 64, 40, 16, 1, 0, 2, 30_000,
        "1" * 64, "2" * 64 if dense else None, "3" * 64 if sparse else None,
        "4" * 64 if reranker else None,
    )


def _audited_routes(embed: tuple[object, ...]) -> dict[str, tuple[object, ...]]:
    return {
        "/api/version": (200, "application/json", {"version": "0.34.0"}),
        "/api/tags": (200, "application/json", {"models": [{"name": "bge-m3:latest", "digest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"}]}),
        "/api/embed": embed,
    }


def test_factory_refuses_non_audited_or_non_dense_only_configurations() -> None:
    factory = getattr(config_module, "provider_from_config", None)
    boundary_factory = getattr(provider_module, "provider_from_config", None)

    assert callable(factory), "Task 6 must expose provider_from_config"
    assert callable(boundary_factory), "provider boundary must own provider_from_config"
    assert isinstance(factory(_config(base_url="http://127.0.0.1:11333", sparse=True)), DisabledProvider)
    assert isinstance(factory(_config(base_url="http://127.0.0.1:11333", reranker=True)), DisabledProvider)
    assert isinstance(factory(_config(base_url="http://127.0.0.1:11333", dense=False)), DisabledProvider)


def test_audited_profile_metadata_is_static_safe_and_capability_specific() -> None:
    from codex_memory.retrieval.local_http_provider import audited_profile_metadata

    metadata = audited_profile_metadata()

    assert metadata == {
        "provider_type": "ollama",
        "provider_version": "0.34.0",
        "provider_fingerprint": metadata["provider_fingerprint"],
        "dense": {
            "model_id": "bge-m3",
            "revision": "latest",
            "digest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab",
            "fingerprint": metadata["dense"]["fingerprint"],
            "dimension": 1024,
            "max_input_tokens": 2048,
            "truncation": "reject",
            "normalization": "l2",
            "quantization": "F16",
        },
        "sparse": "UNSUPPORTED",
        "reranker": "UNSUPPORTED",
    }
    assert len(metadata["provider_fingerprint"]) == 64
    assert len(metadata["dense"]["fingerprint"]) == 64


@pytest.mark.parametrize("base_url", [
    "http://localhost:11333", "https://127.0.0.1:11333", "http://10.0.0.1:11333",
    "http://127.0.0.1:11333/path", "http://user@127.0.0.1:11333",
])
def test_adapter_constructor_refuses_noncanonical_or_nonloopback_urls(base_url: str) -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    with pytest.raises(ValueError):
        LocalHttpProvider.audited_for_tests(base_url)


class _FixtureServer:
    def __init__(self, routes: dict[str, tuple[object, ...]]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.paths: list[str] = []
        self.incomplete_requests: list[str] = []
        self.request_recorded = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _handler(self):  # type: ignore[no-untyped-def]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                outer.paths.append(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    outer.incomplete_requests.append(self.path)
                    outer.request_recorded.set()
                    return
                outer.requests.append((self.path, payload))
                outer.request_recorded.set()
                route = outer.routes.get(self.path, (404, "application/json", {}))
                status, mime, body, *rest = route
                delay = next((item for item in rest if isinstance(item, (int, float)) and not isinstance(item, bool)), 0)
                headers = next((item for item in rest if isinstance(item, dict)), {})
                chunks = next((item for item in rest if isinstance(item, list)), None)
                time.sleep(delay)
                encoded = b"".join(chunk for _, chunk in chunks) if chunks is not None else (body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode("utf-8"))
                self.send_response(status)
                self.send_header("Content-Type", mime)
                if "Content-Length" not in headers:
                    self.send_header("Content-Length", str(len(encoded)))
                for name, value in headers.items():
                    if value is not None:
                        self.send_header(name, str(value))
                self.end_headers()
                try:
                    if chunks is None:
                        self.wfile.write(encoded)
                    else:
                        for chunk_delay, chunk in chunks:
                            time.sleep(chunk_delay)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except OSError:
                    pass

            def do_GET(self) -> None:  # noqa: N802
                outer.paths.append(self.path)
                route = outer.routes.get(self.path, (404, "application/json", {}))
                status, mime, body, *rest = route
                delay = next((item for item in rest if isinstance(item, (int, float)) and not isinstance(item, bool)), 0)
                headers = next((item for item in rest if isinstance(item, dict)), {})
                chunks = next((item for item in rest if isinstance(item, list)), None)
                time.sleep(delay)
                encoded = b"".join(chunk for _, chunk in chunks) if chunks is not None else (body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode("utf-8"))
                self.send_response(status)
                self.send_header("Content-Type", mime)
                if "Content-Length" not in headers:
                    self.send_header("Content-Length", str(len(encoded)))
                for name, value in headers.items():
                    if value is not None:
                        self.send_header(name, str(value))
                self.end_headers()
                try:
                    if chunks is None:
                        self.wfile.write(encoded)
                    else:
                        for chunk_delay, chunk in chunks:
                            time.sleep(chunk_delay)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except OSError:
                    pass

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@contextmanager
def _server(routes: dict[str, tuple[object, ...]]) -> Iterator[_FixtureServer]:
    fixture = _FixtureServer(routes)
    fixture.start()
    try:
        yield fixture
    finally:
        fixture.close()


def test_dense_mapping_uses_exact_fields_preserves_request_order_and_marks_sparse_unsupported() -> None:
    from codex_memory.retrieval.local_http_provider import (
        AUDITED_DENSE_FINGERPRINT,
        AUDITED_PROVIDER_FINGERPRINT,
    )

    routes = {
        "/api/version": (200, "application/json", {"version": "0.34.0"}),
        "/api/tags": (200, "application/json", {"models": [{"name": "bge-m3:latest", "digest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"}]}),
        "/api/embed": (200, "application/json", {"model": "bge-m3:latest", "embeddings": [[1.0] * 1024, [2.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1}),
    }
    with _server(routes) as fixture:
        config = _config(base_url=fixture.base_url)
        object.__setattr__(config, "expected_provider_fingerprint", AUDITED_PROVIDER_FINGERPRINT)
        object.__setattr__(config, "expected_dense_fingerprint", AUDITED_DENSE_FINGERPRINT)
        provider = config_module.provider_from_config(config)
        inputs = (ProviderInput.from_text("alpha", 0, "first"), ProviderInput.from_text("beta", 1, "second"))

        capabilities = provider.capabilities(deadline_ms=500)
        response = provider.embed_documents(inputs, deadline_ms=500)
        sparse = provider.sparse_documents(inputs, deadline_ms=500)

    assert capabilities.call.status is ProviderCallStatus.OK
    assert capabilities.capabilities is not None
    assert capabilities.capabilities.dense.identity is not None
    assert capabilities.capabilities.dense.identity.fingerprint == AUDITED_DENSE_FINGERPRINT
    assert response.call.status is ProviderCallStatus.OK
    assert [(item.item_id, item.position) for item in response.items] == [("alpha", 0), ("beta", 1)]
    assert sparse.call.status is ProviderCallStatus.UNSUPPORTED
    assert fixture.requests[-1][1] == {"input": ["first", "second"], "model": "bge-m3", "truncate": False}


def test_audited_034_profile_accepts_observed_034_service() -> None:
    """The audited local service version must be bound to the live response."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {
        "model": "bge-m3",
        "embeddings": [[1.0] * 1024],
        "total_duration": 1,
        "load_duration": 1,
        "prompt_eval_count": 1,
    }
    routes = _audited_routes((200, "application/json", body))
    routes["/api/version"] = (200, "application/json", {"version": "0.34.0"})
    with _server(routes) as fixture:
        provider = LocalHttpProvider.audited_for_tests(fixture.base_url)
        response = provider.embed_query(
            ProviderInput.from_text("query", 0, "controlled fixture"),
            deadline_ms=500,
        )

    assert response.call.status is ProviderCallStatus.OK
    assert len(response.items) == 1
    assert len(response.items[0].values) == 1024


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, ProviderCallStatus.UNSUPPORTED), (429, ProviderCallStatus.BUSY), (503, ProviderCallStatus.BUSY), (400, ProviderCallStatus.INPUT_TOO_LONG)],
)
def test_dense_http_failures_have_stable_safe_status(status: int, expected: ProviderCallStatus) -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    with _server(_audited_routes((status, "application/json", {"error": "private detail"}))) as fixture:
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    assert response.call.status is expected
    assert response.items == ()
    assert response.call.reason == expected.value


@pytest.mark.parametrize("mime, body", [
    ("text/plain", {"embeddings": [[1.0]]}),
    ("application/json", {"embeddings": ["wrong"]}),
    ("application/json", {"embeddings": [[float("nan")]]}),
    ("application/json", {"model": "bge-m3", "embeddings": [[1.0]], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1}),
])
def test_dense_rejects_untrusted_response_shapes(mime: str, body: object) -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    with _server(_audited_routes((200, mime, body))) as fixture:
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    assert response.call.status is ProviderCallStatus.INVALID_RESPONSE
    assert response.items == ()


def test_dense_rejects_response_beyond_the_audited_no_truncation_limit() -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {"model": "bge-m3", "embeddings": [[1.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 2_049}
    with _server(_audited_routes((200, "application/json", body))) as fixture:
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    assert response.call.status is ProviderCallStatus.INVALID_RESPONSE
    assert response.items == ()


def test_dense_requires_a_fresh_successful_capability_handshake_before_embedding() -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {"model": "bge-m3", "embeddings": [[1.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1}
    routes = {
        "/api/version": (200, "application/json", {"version": "0.34.0"}),
        "/api/tags": (200, "application/json", {"models": [{"name": "bge-m3:latest", "digest": "0" * 64}]}),
        "/api/embed": (200, "application/json", body),
    }
    with _server(routes) as fixture:
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    assert response.call.status is ProviderCallStatus.FINGERPRINT_MISMATCH
    assert response.call.provider_fingerprint is None
    assert response.call.model_fingerprint is None
    assert all(path != "/api/embed" for path, _ in fixture.requests)


def test_capability_handshake_uses_one_absolute_deadline_for_its_two_requests() -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    routes = {
        "/api/version": (200, "application/json", {"version": "0.34.0"}, 0.11),
        "/api/tags": (200, "application/json", {"models": [{"name": "bge-m3:latest", "digest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"}]}, 0.11),
    }
    with _server(routes) as fixture:
        started = time.monotonic()
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).capabilities(deadline_ms=150)
        elapsed = time.monotonic() - started

    assert response.call.status in {ProviderCallStatus.TIMEOUT, ProviderCallStatus.DEADLINE_EXHAUSTED}
    assert elapsed < 0.20


def test_connected_response_can_use_read_cap_beyond_connect_cap() -> None:
    """Would fail if the connect cap is reused to read an established response."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {
        "model": "bge-m3", "embeddings": [[1.0] * 1024],
        "total_duration": 1, "load_duration": 0, "prompt_eval_count": 1,
    }
    routes = _audited_routes((200, "application/json", body, 0.8))
    with _server(routes) as fixture:
        response = LocalHttpProvider.audited_for_tests(
            fixture.base_url
        ).embed_query(ProviderInput.from_text("q", 0, "fixture"), deadline_ms=3000)

    assert response.call.status is ProviderCallStatus.OK
    assert len(response.items) == 1


def test_connected_response_times_out_when_the_total_budget_is_under_its_delay() -> None:
    """Would fail if an established response reset the absolute request budget."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {
        "model": "bge-m3", "embeddings": [[1.0] * 1024],
        "total_duration": 1, "load_duration": 0, "prompt_eval_count": 1,
    }
    with _server(_audited_routes((200, "application/json", body, 0.8))) as fixture:
        started = time.monotonic()
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("q", 0, "fixture"), deadline_ms=700
        )
        elapsed = time.monotonic() - started

    assert response.call.status is ProviderCallStatus.TIMEOUT
    assert elapsed < 0.78


def test_slow_response_header_uses_the_smaller_read_cap_after_connect() -> None:
    """Would fail if the configured read cap were not applied after connect."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {
        "model": "bge-m3", "embeddings": [[1.0] * 1024],
        "total_duration": 1, "load_duration": 0, "prompt_eval_count": 1,
    }
    with _server(_audited_routes((200, "application/json", body, 0.2))) as fixture:
        provider = LocalHttpProvider(fixture.base_url, connect_timeout_ms=500, read_timeout_ms=100)
        response = provider.embed_query(ProviderInput.from_text("q", 0, "fixture"), deadline_ms=2000)

    assert response.call.status is ProviderCallStatus.TIMEOUT


def test_request_body_resamples_deadline_at_send_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Would fail if an expired budget still allowed the real body send."""

    from codex_memory.retrieval import local_http_provider as module
    from codex_memory.retrieval.deadline import DeadlineBudget

    body = {
        "model": "bge-m3", "embeddings": [[1.0] * 1024],
        "total_duration": 1, "load_duration": 0, "prompt_eval_count": 1,
    }
    current_ns = [0]

    class AdvanceDeadlineBeforeBody(module._BudgetedHTTPConnection):
        def send(self, data: object) -> None:
            if isinstance(data, bytes) and data.startswith(b'{"input":'):
                current_ns[0] = 100_000_000
            super().send(data)

    monkeypatch.setattr(module, "_BudgetedHTTPConnection", AdvanceDeadlineBeforeBody)
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: current_ns[0])
    with _server(_audited_routes((200, "application/json", body))) as fixture:
        status, _ = module.LocalHttpProvider.audited_for_tests(fixture.base_url)._request(
            "POST", "/api/embed", {"input": ["q"], "model": "bge-m3", "truncate": False},
            DeadlineBudget(0, 50_000_000),
        )
        assert fixture.request_recorded.wait(timeout=1)

    assert status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert fixture.requests == []
    assert fixture.incomplete_requests == ["/api/embed"]


def test_controller_cancels_local_http_before_delayed_method_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent can cancel a claimed worker before any loopback request."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider
    from codex_memory.retrieval.provider_control import ProviderCallController

    lookup_started = threading.Event()
    release_lookup = threading.Event()

    class DelayedLookupLocalProvider(LocalHttpProvider):
        armed = False

        def __getattribute__(self, name):  # type: ignore[no-untyped-def]
            if name == "embed_query" and object.__getattribute__(self, "armed"):
                lookup_started.set()
                release_lookup.wait(2)
            return super().__getattribute__(name)

    with _server(_audited_routes((404, "application/json", {}))) as fixture:
        provider = DelayedLookupLocalProvider.audited_for_tests(fixture.base_url)
        provider.armed = True
        controller = ProviderCallController(
            provider,
            DeadlineBudget.start(150, now_ns=time.monotonic_ns),
            None,
            provider.expected_provider_fingerprint,
            now_ns=time.monotonic_ns,
        )

        response = controller.dense_query(
            ProviderInput.from_text("query", 0, "fixture"), dimension=1024
        )
        try:
            assert lookup_started.is_set()
            assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
            assert fixture.paths == []
        finally:
            release_lookup.set()
        assert controller.last_completion is not None
        assert controller.last_completion.wait(1)
        assert fixture.paths == []


def test_local_http_keeps_absolute_deadline_after_final_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-gate delayed adapter entry cannot emit a late request."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider
    from codex_memory.retrieval.provider_control import ProviderCallController

    method_entered = threading.Event()
    release_method = threading.Event()

    class DelayedEntryLocalProvider(LocalHttpProvider):
        def embed_query(self, item, *, deadline_ms):  # type: ignore[no-untyped-def]
            method_entered.set()
            release_method.wait(2)
            return super().embed_query(item, deadline_ms=deadline_ms)

    with _server(_audited_routes((404, "application/json", {}))) as fixture:
        provider = DelayedEntryLocalProvider.audited_for_tests(fixture.base_url)
        controller = ProviderCallController(
            provider,
            DeadlineBudget.start(150, now_ns=time.monotonic_ns),
            None,
            provider.expected_provider_fingerprint,
            now_ns=time.monotonic_ns,
        )

        response = controller.dense_query(
            ProviderInput.from_text("query", 0, "fixture"), dimension=1024
        )
        try:
            assert method_entered.is_set()
            assert response.call.status is ProviderCallStatus.TIMEOUT
            assert fixture.paths == []
        finally:
            release_method.set()
        assert controller.last_completion is not None
        assert controller.last_completion.wait(1)
        assert fixture.paths == []


def test_absolute_deadline_is_not_restarted_at_local_adapter_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import local_http_provider as module

    current_ns = [50_000_000]
    opened = []
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: current_ns[0])
    monkeypatch.setattr(
        module,
        "build_opener",
        lambda *_: opened.append(True),
    )
    provider = module.LocalHttpProvider.audited_for_tests(
        "http://127.0.0.1:11333"
    )

    response = provider.capabilities(
        deadline_ms=AbsoluteDeadlineMilliseconds(
            50, DeadlineBudget(0, 50_000_000)
        )
    )

    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED
    assert opened == []


def test_absolute_deadline_is_rechecked_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import local_http_provider as module

    current_ns = [50_000_000]
    base_connect_calls = []
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: current_ns[0])
    monkeypatch.setattr(
        module.HTTPConnection,
        "connect",
        lambda _self: base_connect_calls.append(True),
    )
    connection = module._BudgetedHTTPConnection(
        "127.0.0.1:11333",
        timeout=0.05,
        budget=DeadlineBudget(0, 50_000_000),
        read_timeout_ms=50,
    )

    with pytest.raises(module._DeadlineExceeded):
        connection.connect()

    assert base_connect_calls == []


def test_connect_uses_resampled_positive_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import local_http_provider as module

    current_ns = [49_000_000]
    observed_timeouts = []

    class Socket:
        def settimeout(self, _timeout):  # type: ignore[no-untyped-def]
            return None

    def connect(connection):  # type: ignore[no-untyped-def]
        observed_timeouts.append(connection.timeout)
        connection.sock = Socket()

    monkeypatch.setattr(module.time, "monotonic_ns", lambda: current_ns[0])
    monkeypatch.setattr(module.HTTPConnection, "connect", connect)
    connection = module._BudgetedHTTPConnection(
        "127.0.0.1:11333",
        timeout=0.5,
        budget=DeadlineBudget(0, 50_000_000),
        read_timeout_ms=50,
    )

    connection.connect()

    assert observed_timeouts == [0.001]


def test_absolute_deadline_is_rechecked_before_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_memory.retrieval import local_http_provider as module

    current_ns = [50_000_000]
    base_send_calls = []
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: current_ns[0])
    monkeypatch.setattr(
        module.HTTPConnection,
        "send",
        lambda _self, _data: base_send_calls.append(True),
    )
    connection = module._BudgetedHTTPConnection(
        "127.0.0.1:11333",
        timeout=0.05,
        budget=DeadlineBudget(0, 50_000_000),
        read_timeout_ms=50,
    )

    with pytest.raises(module._DeadlineExceeded):
        connection.send(b"GET / HTTP/1.1\r\n\r\n")

    assert base_send_calls == []


def test_trickled_response_that_finishes_after_deadline_is_rejected() -> None:
    """Would fail if body reads did not recheck the original absolute deadline."""
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    body = {
        "model": "bge-m3", "embeddings": [[1.0] * 1024],
        "total_duration": 1, "load_duration": 0, "prompt_eval_count": 1,
    }
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    chunks = [(0.0, encoded[:1000])] + [(0.06, encoded[index:index + 1000]) for index in range(1000, len(encoded), 1000)]
    with _server(_audited_routes((200, "application/json", body, chunks))) as fixture:
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("q", 0, "fixture"), deadline_ms=200
        )

    assert response.call.status is ProviderCallStatus.DEADLINE_EXHAUSTED


@pytest.mark.parametrize("version,digest", [
    ("0.0.0", "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"),
    ("0.33.3", "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"),
])
def test_each_dense_call_revalidates_the_audited_version_and_digest(version: str, digest: str) -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    routes = {
        "/api/version": (200, "application/json", {"version": version}),
        "/api/tags": (200, "application/json", {"models": [{"name": "bge-m3:latest", "digest": digest}]}),
        "/api/embed": (200, "application/json", {"model": "bge-m3", "embeddings": [[1.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1}),
    }
    with _server(routes) as fixture:
        provider = LocalHttpProvider.audited_for_tests(fixture.base_url)
        first = provider.capabilities(deadline_ms=500)
        response = provider.embed_query(ProviderInput.from_text("query", 0, "query"), deadline_ms=500)

    assert first.call.status is ProviderCallStatus.FINGERPRINT_MISMATCH
    assert response.call.status is ProviderCallStatus.FINGERPRINT_MISMATCH
    assert all(path != "/api/embed" for path, _ in fixture.requests)


def test_dense_revalidates_after_a_previous_successful_handshake_detects_model_change() -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    routes = _audited_routes((200, "application/json", {"model": "bge-m3", "embeddings": [[1.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1}))
    with _server(routes) as fixture:
        provider = LocalHttpProvider.audited_for_tests(fixture.base_url)
        initial = provider.capabilities(deadline_ms=500)
        fixture.routes["/api/tags"] = (200, "application/json", {"models": [{"name": "bge-m3:latest", "digest": "e" * 64}]})
        response = provider.embed_query(ProviderInput.from_text("query", 0, "query"), deadline_ms=500)

    assert initial.call.status is ProviderCallStatus.OK
    assert response.call.status is ProviderCallStatus.FINGERPRINT_MISMATCH
    assert [path for path, _ in fixture.requests].count("/api/embed") == 0


def test_adapter_refuses_redirect_instead_of_following_it() -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    routes = _audited_routes((302, "application/json", {}, {"Location": "/api/redirected"}))
    routes["/api/redirected"] = (200, "application/json", {"model": "bge-m3", "embeddings": [[1.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1})
    with _server(routes) as fixture:
        response = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    assert response.call.status is ProviderCallStatus.UNAVAILABLE
    assert [path for path, _ in fixture.requests].count("/api/redirected") == 0


def test_adapter_installs_empty_proxy_handler_and_no_redirect_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    from codex_memory.retrieval import local_http_provider as module

    captured: list[object] = []
    original = module.build_opener

    def recording_opener(*handlers: object):  # type: ignore[no-untyped-def]
        captured.extend(handlers)
        return original(*handlers)

    monkeypatch.setattr(module, "build_opener", recording_opener)
    with _server(_audited_routes((404, "application/json", {}))) as fixture:
        module.LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    proxy = next(handler for handler in captured if isinstance(handler, module.ProxyHandler))
    assert proxy.proxies == {}
    assert any(isinstance(handler, module._NoRedirect) for handler in captured)


def test_adapter_rejects_declared_and_streamed_oversize_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    from codex_memory.retrieval import local_http_provider as module

    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 128)
    declared = _audited_routes((200, "application/json", {}, {"Content-Length": "129"}))
    with _server(declared) as fixture:
        first = module.LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )
    streamed = _audited_routes((200, "application/json", b"x" * 129, {"Content-Length": None}))
    with _server(streamed) as fixture:
        second = module.LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=500
        )

    assert first.call.status is ProviderCallStatus.INVALID_RESPONSE
    assert second.call.status is ProviderCallStatus.INVALID_RESPONSE


def test_adapter_reports_connection_unavailable_timeout_and_count_mismatch() -> None:
    from codex_memory.retrieval.local_http_provider import LocalHttpProvider

    unavailable = LocalHttpProvider.audited_for_tests("http://127.0.0.1:1").embed_query(
        ProviderInput.from_text("query", 0, "query"), deadline_ms=100
    )
    # Leave enough startup margin for the full suite while retaining a
    # deterministic read-phase timeout.  The server deliberately waits longer
    # than the adapter's total budget before sending the handshake response.
    timeout_routes = {
        "/api/version": (200, "application/json", {"version": "0.34.0"}, 1.0),
    }
    with _server(timeout_routes) as fixture:
        timeout = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_query(
            ProviderInput.from_text("query", 0, "query"), deadline_ms=250
        )
    with _server(_audited_routes((200, "application/json", {"model": "bge-m3", "embeddings": [[1.0] * 1024], "total_duration": 1, "load_duration": 1, "prompt_eval_count": 1}))) as fixture:
        count = LocalHttpProvider.audited_for_tests(fixture.base_url).embed_documents(
            (ProviderInput.from_text("a", 0, "one"), ProviderInput.from_text("b", 1, "two")), deadline_ms=500
        )

    assert unavailable.call.status is ProviderCallStatus.UNAVAILABLE
    assert timeout.call.status is ProviderCallStatus.TIMEOUT
    assert count.call.status is ProviderCallStatus.INVALID_RESPONSE
    assert count.items == ()


def test_capability_response_streams_are_closed_after_each_handshake_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from codex_memory.retrieval import local_http_provider as module

    class Response:
        def __init__(self, payload: object) -> None:
            self.headers = Message()
            self.headers["Content-Type"] = "application/json"
            self.headers["Content-Length"] = str(len(json.dumps(payload).encode("utf-8")))
            self._payload = json.dumps(payload).encode("utf-8")
            self.closed = False

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def read(self, amount: int) -> bytes:
            return self._payload

    responses = [
        Response({"version": "0.34.0"}),
        Response({"models": [{"name": "bge-m3:latest", "digest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"}]}),
    ]
    all_responses = list(responses)

    class Opener:
        def open(self, request: object, timeout: float):  # type: ignore[no-untyped-def]
            return responses.pop(0)

    monkeypatch.setattr(module, "build_opener", lambda *handlers: Opener())
    result = module.LocalHttpProvider.audited_for_tests("http://127.0.0.1:11333").capabilities(deadline_ms=500)

    assert result.call.status is ProviderCallStatus.OK
    assert responses == []
    assert all(response.closed for response in all_responses)
