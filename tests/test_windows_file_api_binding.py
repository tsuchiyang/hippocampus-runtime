"""Avoid repeated API setup without reusing file handles or verified contents."""
from __future__ import annotations

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from codex_memory.retrieval import semantic_store


pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows file-handle API")


def test_file_api_binding_is_reused_without_reloading_kernel32(monkeypatch):
    """Regression: recreating WinDLL for every held object repeats typed setup."""
    factory = ctypes.WinDLL
    constructions = []

    def observed_factory(*args, **kwargs):
        constructions.append((args, kwargs))
        return factory(*args, **kwargs)

    monkeypatch.setattr(ctypes, "WinDLL", observed_factory)
    first = semantic_store._windows_kernel32()
    second = semantic_store._windows_kernel32()
    assert first is second
    assert len(constructions) <= 1  # A previous legitimate caller may have warmed it.
    if constructions:
        assert constructions == [(("kernel32",), {"use_last_error": True})]
    assert first.CreateFileW.restype is ctypes.c_void_p
    assert first.GetFileInformationByHandle.restype is ctypes.c_int
    assert first.ReadFile.restype is ctypes.c_int


def test_failed_signature_setup_does_not_publish_partial_binding(monkeypatch):
    semantic_store._windows_kernel32.cache_clear()
    factory = ctypes.WinDLL
    constructions = []

    class IncompleteBinding:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            if name == "ReadFile":
                raise OSError("injected final signature setup failure")
            return getattr(self.actual, name)

    def fail_once(*args, **kwargs):
        constructions.append((args, kwargs))
        actual = factory(*args, **kwargs)
        return IncompleteBinding(actual) if len(constructions) == 1 else actual

    try:
        monkeypatch.setattr(ctypes, "WinDLL", fail_once)
        with pytest.raises(OSError, match="final signature setup failure"):
            semantic_store._windows_kernel32()
        api = semantic_store._windows_kernel32()
        assert api is semantic_store._windows_kernel32()
        assert len(constructions) == 2
        assert api.ReadFile.restype is ctypes.c_int
    finally:
        semantic_store._windows_kernel32.cache_clear()


def test_reused_api_still_holds_each_file_and_reads_fresh_contents(tmp_path):
    target = tmp_path / "held.bin"
    for content in (b"first", b"second-and-different"):
        target.write_bytes(content)
        handle, state = semantic_store._open_windows_acceptance_handle(target, directory=False)
        try:
            assert state[3] == len(content)
            with pytest.raises(PermissionError):
                target.write_bytes(b"must remain blocked")
            assert semantic_store._read_windows_handle(handle, max_bytes=1024,
                deadline_ns=time.monotonic_ns() + 1_000_000_000) == content
        finally:
            assert semantic_store._windows_kernel32().CloseHandle(handle)
    # The successful second write proves that the first handle was released.
    assert target.read_bytes() == b"second-and-different"


def test_shared_binding_preserves_per_thread_native_error(tmp_path):
    target = tmp_path / "held.bin"
    target.write_bytes(b"held")
    held, _ = semantic_store._open_windows_acceptance_handle(target, directory=False)
    ready = threading.Barrier(2)

    def fail_open(path, desired_access):
        api = semantic_store._windows_kernel32()
        handle = api.CreateFileW(str(path), desired_access, 1, None, 3, 0x00200000, None)
        assert handle == ctypes.c_void_p(-1).value
        # Both native calls complete before either reads its ctypes error slot.
        ready.wait(timeout=2)
        return ctypes.get_last_error()

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            missing = workers.submit(fail_open, tmp_path / "missing.bin", 0x80000000)
            write_denied = workers.submit(fail_open, target, 0x40000000)
            assert missing.result(timeout=3) == 2  # ERROR_FILE_NOT_FOUND
            assert write_denied.result(timeout=3) == 32  # ERROR_SHARING_VIOLATION
    finally:
        assert semantic_store._windows_kernel32().CloseHandle(held)
