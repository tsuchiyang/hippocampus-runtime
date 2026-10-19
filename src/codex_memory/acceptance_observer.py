"""Private P2.5 acceptance pipe protocol. Never imported by an ordinary Hook.

No files, callbacks, SQL, exception messages or replacement results travel on
this channel. The parent owns evidence. A broken channel invalidates acceptance
and must never become a canonical database failure.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

from codex_memory.ledger_common import canonical_json, parse_json_value

ACTIVATION = "CODEX_MEMORY_P25_OBSERVER"
_PROCESS_BUDGET_NS = 1_500_000_000
_RETURN_RESERVE_NS = 50_000_000
_VALIDATION_BUDGET_NS = 1_000_000_000
VERSION = "p25-observer-v2"
MAX_FRAME = 8192
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_FIELDS = {"run_id", "request_id", "session_id", "turn_id", "source_sha256", "fixture_identity", "canonical_identity"}
_DATA_FIELDS = {
    "hello": {"installed"}, "bound": {"entry_pid"},
    "legacy": {"delivery", "write_count", "serialized_sha256", "core_in_transaction"},
    "canonical_opened": {"in_transaction"}, "release": set(),
    "validation_read_started": {"stage", "operation", "in_transaction", "budget_ms", "started_ns", "deadline_ns", "observed_ns"},
    "validation_no_result_before_deadline": {"stage", "operation", "in_transaction", "budget_ms", "started_ns", "deadline_ns", "observed_ns"},
    "canonical_failure": {"kind", "stage", "causes"},
    "query": {"status", "reasons", "canonical_highwater", "generation"},
    "audit_attempt": set(), "audit_result": {"recorded"},
    "terminal": {"reason", "recorded", "injection_id"},
}
_NEXT = {
    None: {"hello"}, "hello": {"legacy"}, "legacy": {"canonical_opened", "canonical_failure", "terminal"},
    "canonical_opened": {"validation_read_started", "canonical_failure", "terminal"},
    "validation_read_started": {"canonical_failure", "query", "terminal", "validation_no_result_before_deadline"},
    "validation_no_result_before_deadline": set(),
    "canonical_failure": {"query"}, "query": {"audit_attempt", "terminal"},
    "audit_attempt": {"audit_result"}, "audit_result": {"terminal"}, "terminal": set(),
}


class AcceptanceObserverError(RuntimeError):
    """Acceptance failed; no implication about canonical validity."""


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity(path: Path) -> list[int]:
    value = path.lstat()
    if path.is_symlink() or value.st_file_attributes & 0x400:
        raise AcceptanceObserverError("reparse_object")
    if stat.S_ISREG(value.st_mode) and value.st_nlink != 1:
        raise AcceptanceObserverError("hardlinked_object")
    return [value.st_dev, value.st_ino]


def validate_binding(binding):
    if type(binding) is not dict or set(binding) != _BINDING_FIELDS:
        raise AcceptanceObserverError("binding_fields")
    for name in ("run_id", "session_id", "turn_id"):
        if type(binding[name]) is not str or not 1 <= len(binding[name]) <= 128:
            raise AcceptanceObserverError("binding_identifier")
    for name in ("request_id", "source_sha256"):
        if type(binding[name]) is not str or not _HASH.fullmatch(binding[name]):
            raise AcceptanceObserverError("binding_digest")
    for name in ("fixture_identity", "canonical_identity"):
        if type(binding[name]) is not list or len(binding[name]) != 2 or any(type(v) is not int or v < 0 for v in binding[name]):
            raise AcceptanceObserverError("binding_identity")


def check_budget(deadline):
    # The observer and shadow controller share this absolute nanosecond
    # cutoff.  DeadlineBudget.remaining_ms() intentionally floors for APIs
    # that accept integer milliseconds; using it as a boolean would expire
    # the observer during the final fractional millisecond while the outer
    # bounded call still considers the same budget live.
    if deadline is None or time.monotonic_ns() >= deadline.deadline_ns:
        raise AcceptanceObserverError("observer_deadline")


def _delivery_time(deadline):
    """Linearize a validated parent receipt inside its existing budget."""
    observed_ns = time.monotonic_ns()
    if deadline is None or observed_ns >= deadline.deadline_ns:
        raise AcceptanceObserverError("observer_deadline")
    return observed_ns


class _ValidationCompletionGate:
    """Linearize first-read completion against the one-shot deadline claim."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "pending"

    def complete(self) -> bool:
        with self._lock:
            if self._state == "deadline_claimed":
                return False
            # Idempotent because the snapshot wrapper may report completion
            # again after the one observed first read.
            self._state = "completed"
            return True

    def claim_if_unset(self) -> bool:
        with self._lock:
            if self._state != "pending":
                return False
            self._state = "deadline_claimed"
            return True


def installed_identity(deadline):
    """Observe this interpreter's resolved package; never echo activation data."""
    check_budget(deadline)
    from codex_memory.retrieval.injection_audit import implementation_fingerprint
    import codex_memory.cli
    root = Path(codex_memory.cli.__file__).resolve().parent
    value = {"source_root": str(root), "implementation_sha256": implementation_fingerprint(deadline=deadline)}
    check_budget(deadline)
    return value


def validate_installed_identity(value):
    if (type(value) is not dict or set(value) != {"source_root", "implementation_sha256"}
            or type(value["source_root"]) is not str or not Path(value["source_root"]).is_absolute()
            or ".." in Path(value["source_root"]).parts
            or type(value["implementation_sha256"]) is not str or not _HASH.fullmatch(value["implementation_sha256"])):
        raise AcceptanceObserverError("installed_identity")


def _kernel():
    if os.name != "nt":
        raise AcceptanceObserverError("windows_pipe_required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        # This API is exported by KernelBase (not kernel32 on this platform).
        kernel.CompareObjectHandles = ctypes.WinDLL("KernelBase", use_last_error=True).CompareObjectHandles
    except (AttributeError, OSError):
        raise AcceptanceObserverError("handle_comparison_unavailable") from None
    signatures = {
        "GetFileType": ([wintypes.HANDLE], wintypes.DWORD),
        "GetStdHandle": ([wintypes.DWORD], wintypes.HANDLE),
        "CompareObjectHandles": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "SetNamedPipeHandleState": ([wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, ctypes.c_void_p], wintypes.BOOL),
        "ReadFile": ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
        "WriteFile": ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
        "PeekNamedPipe": ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
        "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        "Process32FirstW": ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        "Process32NextW": ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        try:
            function = getattr(kernel, name)
        except AttributeError:
            raise AcceptanceObserverError("handle_comparison_unavailable") from None
        function.argtypes, function.restype = arguments, result
    return kernel


class Pipe:
    """One inherited anonymous pipe endpoint, nonblocking and bounded.

    GetFileType and FileAccessInformation reject files and the wrong direction.
    PIPE_NOWAIT prevents a full pipe or disconnected parent from stalling Hook.
    """
    def __init__(self, handle: int, *, write: bool):
        if type(handle) is not int or handle <= 0:
            raise AcceptanceObserverError("invalid_handle")
        self.handle = handle
        self.write = write
        self.kernel = _kernel()
        # Console/CRT handles and DuplicateHandle aliases must never become
        # nonce-bearing evidence pipes. Compare kernel objects, not integers.
        import msvcrt
        standards = [self.kernel.GetStdHandle(code & 0xffffffff) for code in (-10, -11, -12)]
        for fd in (0, 1, 2):
            try:
                standards.append(msvcrt.get_osfhandle(fd))
            except OSError:
                pass
        for standard in standards:
            if standard not in (None, 0, -1, ctypes.c_void_p(-1).value):
                if handle == standard or self.kernel.CompareObjectHandles(wintypes.HANDLE(handle), wintypes.HANDLE(standard)):
                    raise AcceptanceObserverError("standard_stream_handle")
        if self.kernel.GetFileType(wintypes.HANDLE(handle)) != 3:
            raise AcceptanceObserverError("not_pipe")
        class IOStatus(ctypes.Structure):
            _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_size_t)]
        status, access = IOStatus(), wintypes.DWORD()
        query = ctypes.WinDLL("ntdll").NtQueryInformationFile
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(IOStatus), ctypes.c_void_p, wintypes.ULONG, ctypes.c_int]
        query.restype = wintypes.LONG
        code = query(wintypes.HANDLE(handle), ctypes.byref(status), ctypes.byref(access), ctypes.sizeof(access), 8)
        required, forbidden = (2, 1) if write else (1, 2)
        if code < 0 or not access.value & required or access.value & forbidden:
            raise AcceptanceObserverError("pipe_direction")
        mode = wintypes.DWORD(1)  # PIPE_NOWAIT, byte mode
        if not self.kernel.SetNamedPipeHandleState(wintypes.HANDLE(handle), ctypes.byref(mode), None, None):
            raise AcceptanceObserverError("pipe_nonblocking_unavailable")
        self.buffer = bytearray()

    def send(self, raw: bytes, deadline):
        check_budget(deadline)
        if not self.write or not raw.endswith(b"\n") or len(raw) > MAX_FRAME:
            raise AcceptanceObserverError("invalid_frame")
        count = wintypes.DWORD()
        if not self.kernel.WriteFile(wintypes.HANDLE(self.handle), raw, len(raw), ctypes.byref(count), None) or count.value != len(raw):
            raise AcceptanceObserverError("pipe_write_failed")
        check_budget(deadline)

    def receive(self, deadline) -> bytes:
        if self.write:
            raise AcceptanceObserverError("pipe_direction")
        while True:
            check_budget(deadline)
            if b"\n" in self.buffer:
                raw, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                return bytes(raw) + b"\n"
            available = wintypes.DWORD()
            if not self.kernel.PeekNamedPipe(wintypes.HANDLE(self.handle), None, 0, None, ctypes.byref(available), None):
                raise AcceptanceObserverError("pipe_disconnected")
            if available.value:
                size = min(available.value, MAX_FRAME + 1 - len(self.buffer))
                chunk, count = ctypes.create_string_buffer(size), wintypes.DWORD()
                if not self.kernel.ReadFile(wintypes.HANDLE(self.handle), chunk, size, ctypes.byref(count), None) or not count.value:
                    raise AcceptanceObserverError("pipe_read_failed")
                self.buffer.extend(chunk.raw[:count.value])
                if len(self.buffer) > MAX_FRAME:
                    raise AcceptanceObserverError("oversize_frame")
            else:
                time.sleep(min(0.001, deadline.remaining_ms(now_ns=time.monotonic_ns) / 1000))

    def receive_decoded(self, deadline, *, nonce, binding, parent_pid, child_pid, sequence, previous, kind=None):
        """Receive and decode one complete bound frame inside one budget."""
        raw = self.receive(deadline)
        result = decode(raw, nonce=nonce, binding=binding, parent_pid=parent_pid,
            child_pid=child_pid, sequence=sequence, kind=kind)
        if result["kind"] not in _NEXT[previous]:
            raise AcceptanceObserverError("event_order")
        return result, _delivery_time(deadline)

    def _send_validation_deadline(self, raw: bytes):
        """One fixed diagnostic WriteFile on this already PIPE_NOWAIT endpoint.

        Private to Observer.terminal; never a general unbudgeted event send.
        A failed/partial write consumes the one shot and fails acceptance.
        """
        if getattr(self, "_deadline_diagnostic_sent", False):
            raise AcceptanceObserverError("deadline_diagnostic_replayed")
        self._deadline_diagnostic_sent = True
        value = parse_json_value(raw.decode("utf-8"))
        if (not self.write or len(raw) > MAX_FRAME or not raw.endswith(b"\n")
                or value["kind"] != "validation_no_result_before_deadline"
                or set(value["data"]) != _DATA_FIELDS[value["kind"]]):
            raise AcceptanceObserverError("invalid_deadline_diagnostic")
        validate_data(value["kind"], value["data"])
        count = wintypes.DWORD()
        if not self.kernel.WriteFile(wintypes.HANDLE(self.handle), raw, len(raw), ctypes.byref(count), None) or count.value != len(raw):
            raise AcceptanceObserverError("pipe_write_failed")

    def require_empty(self, *, allow_closed=False):
        available = wintypes.DWORD()
        ok = self.kernel.PeekNamedPipe(wintypes.HANDLE(self.handle), None, 0, None, ctypes.byref(available), None)
        closed = not ok and allow_closed and ctypes.get_last_error() == 109
        if self.buffer or not (ok or closed) or available.value:
            raise AcceptanceObserverError("unexpected_control")


def frame(*, nonce, binding, parent_pid, child_pid, sequence, kind, data):
    return (canonical_json({"version": VERSION, "nonce": nonce, "binding": digest(binding),
        "parent_pid": parent_pid, "child_pid": child_pid, "sequence": sequence,
        "kind": kind, "data": data}) + "\n").encode("utf-8")


def decode(raw, *, nonce, binding, parent_pid, child_pid, sequence, kind=None):
    try:
        if type(raw) is not bytes or len(raw) > MAX_FRAME or not raw.endswith(b"\n"):
            raise ValueError()
        value = parse_json_value(raw.decode("utf-8"))
        expected = {"version", "nonce", "binding", "parent_pid", "child_pid", "sequence", "kind", "data"}
        if type(value) is not dict or set(value) != expected or raw != (canonical_json(value) + "\n").encode("utf-8"):
            raise ValueError()
        if value["version"] != VERSION or value["nonce"] != nonce or value["binding"] != digest(binding):
            raise ValueError()
        for key, actual in (("parent_pid", parent_pid), ("child_pid", child_pid), ("sequence", sequence)):
            if type(value[key]) is not int or value[key] != actual:
                raise ValueError()
        if value["kind"] not in _DATA_FIELDS or (kind is not None and value["kind"] != kind):
            raise ValueError()
        if type(value["data"]) is not dict or set(value["data"]) != _DATA_FIELDS[value["kind"]]:
            raise ValueError()
        validate_data(value["kind"], value["data"])
        return value
    except Exception as exc:
        raise AcceptanceObserverError("invalid_protocol") from None


def validate_data(kind, data):
    """Fixed typed facts only; unknown values cannot become verifier success."""
    valid = True
    if kind == "hello":
        validate_installed_identity(data["installed"])
    elif kind == "bound":
        valid = type(data["entry_pid"]) is int and data["entry_pid"] > 0
    elif kind == "legacy":
        valid = (data["delivery"] in {"emitted", "partial", "unknown"}
            and (data["write_count"] is None or type(data["write_count"]) is int and data["write_count"] >= 0)
            and type(data["core_in_transaction"]) is bool
            and type(data["serialized_sha256"]) is str and _HASH.fullmatch(data["serialized_sha256"]))
    elif kind == "canonical_opened":
        valid = data["in_transaction"] is False
    elif kind in {"validation_read_started", "validation_no_result_before_deadline"}:
        valid = (data["stage"] == "invariant_validation" and data["operation"] == "PRAGMA integrity_check"
            and data["in_transaction"] is True and type(data["budget_ms"]) is int and data["budget_ms"] == 1000
            and all(type(data[k]) is int and data[k] >= 0 for k in ("started_ns", "deadline_ns", "observed_ns")))
        if valid:
            valid = (data["deadline_ns"] - data["started_ns"] == 1_000_000_000
                and (data["started_ns"] <= data["observed_ns"] < data["deadline_ns"] if kind == "validation_read_started"
                    else data["observed_ns"] >= data["deadline_ns"]))
    elif kind == "canonical_failure":
        valid = (data["kind"] in {"invalid", "unavailable"}
            and data["stage"] in {"open", "schema_query", "ddl_read", "highwater_read", "invariant_validation"}
            and type(data["causes"]) is list and 1 <= len(data["causes"]) <= 16)
        if valid:
            for cause in data["causes"]:
                valid = valid and (type(cause) is dict and set(cause) == {"type", "sqlite_errorcode", "sqlite_errorname"}
                    and type(cause["type"]) is str and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", cause["type"])
                    and (cause["sqlite_errorcode"] is None or type(cause["sqlite_errorcode"]) is int and cause["sqlite_errorcode"] >= 0)
                    and (cause["sqlite_errorname"] is None or type(cause["sqlite_errorname"]) is str and re.fullmatch(r"SQLITE_[A-Z_0-9]{1,64}", cause["sqlite_errorname"])))
    elif kind == "query":
        from codex_memory.retrieval.contracts import RetrievalStatus, DegradedReason, IndexHighwater
        valid = (data["status"] in {v.value for v in RetrievalStatus}
            and type(data["reasons"]) is list and len(data["reasons"]) <= 32
            and all(type(v) is str and v in {r.value for r in DegradedReason} for v in data["reasons"])
            and (data["generation"] is None or type(data["generation"]) is str and len(data["generation"]) <= 128))
        if data["canonical_highwater"] is not None:
            IndexHighwater.from_dict(data["canonical_highwater"])
    elif kind == "audit_result":
        valid = type(data["recorded"]) is bool
    elif kind == "terminal":
        valid = (type(data["recorded"]) is bool
            and (data["reason"] is None or type(data["reason"]) is str and re.fullmatch(r"[a-z_]{1,80}", data["reason"]))
            and (data["injection_id"] is None or type(data["injection_id"]) is str and re.fullmatch(r"inj_[0-9a-f]{64}", data["injection_id"])))
    if not valid:
        raise AcceptanceObserverError("event_data")


def parent_pid(pid: int) -> int:
    """Read one OS process-ancestry edge."""
    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
            ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD),
            ("exe", wintypes.WCHAR * 260)]
    kernel = _kernel()
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise AcceptanceObserverError("process_identity_unavailable")
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        ok = kernel.Process32FirstW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        while ok:
            if entry.pid == pid:
                return entry.parent
            ok = kernel.Process32NextW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        raise AcceptanceObserverError("process_identity_unavailable")
    finally:
        kernel.CloseHandle(wintypes.HANDLE(snapshot))


def _ancestor_distance(descendant: int, ancestor: int, *, max_hops: int) -> int | None:
    """Return a bounded live ancestry distance without accepting extra launchers."""
    if (type(descendant) is not int or descendant <= 0
            or type(ancestor) is not int or ancestor <= 0
            or type(max_hops) is not int or max_hops < 0):
        raise AcceptanceObserverError("process_identity_unavailable")
    current = descendant
    seen: set[int] = set()
    for distance in range(max_hops + 1):
        if current == ancestor:
            return distance
        if current in seen or current <= 0:
            return None
        seen.add(current)
        if distance != max_hops:
            current = parent_pid(current)
    return None


def _entry_chain_matches(descendant: int, entry: int, acceptance_parent: int) -> bool:
    """Bind CPython to the exact entry child through at most two trampolines."""
    if (type(entry) is not int or entry <= 0
            or type(acceptance_parent) is not int or acceptance_parent <= 0):
        return False
    return (_ancestor_distance(descendant, entry, max_hops=2) is not None
            and parent_pid(entry) == acceptance_parent)


def _validate_activation_process_window(
    process_started_ns: int,
    deadline,
    *,
    observed_ns: int,
) -> None:
    """Require the child's validation window inside the parent's process cap."""
    if (
        isinstance(process_started_ns, bool)
        or not isinstance(process_started_ns, int)
        or process_started_ns < 0
        or isinstance(observed_ns, bool)
        or not isinstance(observed_ns, int)
        or observed_ns < process_started_ns
        or not hasattr(deadline, "started_ns")
        or not hasattr(deadline, "deadline_ns")
        or deadline.started_ns < process_started_ns
        or observed_ns < deadline.started_ns
        or deadline.deadline_ns - deadline.started_ns !=
            _VALIDATION_BUDGET_NS
        or deadline.deadline_ns >
            process_started_ns + _PROCESS_BUDGET_NS - _RETURN_RESERVE_NS
    ):
        raise AcceptanceObserverError("activation_deadline")


class Observer:
    def __init__(self, reader, writer, *, nonce, binding, parent, child, deadline):
        self.reader, self.writer = reader, writer
        self.nonce, self.binding = nonce, binding
        self.parent, self.child, self.deadline = parent, child, deadline
        self.sequence, self.state = 0, None
        self.opened = False
        self._lock = threading.RLock()
        self._validation_pending = False
        self._validation_completed = _ValidationCompletionGate()

    @classmethod
    def from_environment(cls, *, paths, request, source, deadline):
        raw = os.environ.get(ACTIVATION)
        if raw is None:
            return None
        try:
            value = parse_json_value(raw)
            if len(raw) > MAX_FRAME or type(value) is not dict or set(value) != {"version", "read_handle", "write_handle", "nonce", "parent_pid", "process_started_ns", "binding"} or value["version"] != VERSION:
                raise AcceptanceObserverError("activation_fields")
            if type(value["nonce"]) is not str or not _HASH.fullmatch(value["nonce"]):
                raise AcceptanceObserverError("activation_nonce")
            if type(value["parent_pid"]) is not int or value["parent_pid"] <= 0 or value["parent_pid"] == os.getpid():
                raise AcceptanceObserverError("activation_pid")
            _validate_activation_process_window(
                value["process_started_ns"], deadline,
                observed_ns=time.monotonic_ns(),
            )
            # The bound console entry and uv-managed venv interpreter are two
            # separate Windows trampolines.  The real CPython child therefore
            # sees the acceptance parent exactly one to three edges above it.
            if _ancestor_distance(os.getpid(), value["parent_pid"], max_hops=3) is None:
                raise AcceptanceObserverError("activation_parent")
            validate_binding(value["binding"])
            from codex_memory.retrieval.shadow import shadow_run_id
            actual = {"run_id": value["binding"]["run_id"], "request_id": shadow_run_id(session_id=request.session_id, turn_id=request.turn_id, hook_event="UserPromptSubmit", query=request.query),
                "session_id": request.session_id, "turn_id": request.turn_id, "source_sha256": digest(source),
                "fixture_identity": identity(paths.repo_root), "canonical_identity": identity(paths.canonical_database_path)}
            if value["binding"] != actual or value["read_handle"] == value["write_handle"]:
                raise AcceptanceObserverError("activation_binding")
            reader = Pipe(value["read_handle"], write=False)
            writer = Pipe(value["write_handle"], write=True)
            observer = cls(reader, writer, nonce=value["nonce"], binding=actual, parent=value["parent_pid"], child=os.getpid(), deadline=deadline)
            observer.emit("hello", {"installed": installed_identity(deadline)})
            bound = observer.control("bound", sequence=0)
            entry = bound["entry_pid"]
            if not _entry_chain_matches(os.getpid(), entry, observer.parent):
                raise AcceptanceObserverError("parent_binding")
            return observer
        except AcceptanceObserverError:
            raise
        except Exception:
            raise AcceptanceObserverError("activation_invalid") from None

    def emit(self, kind, data):
        with self._lock:
            if kind == "validation_no_result_before_deadline":
                raise AcceptanceObserverError("child_deadline_proof_required")
            self._emit(kind, data)

    def _emit(self, kind, data):
        if kind not in _NEXT[self.state]:
            raise AcceptanceObserverError("event_order")
        raw = frame(nonce=self.nonce, binding=self.binding, parent_pid=self.parent, child_pid=self.child, sequence=self.sequence, kind=kind, data=data)
        decode(raw, nonce=self.nonce, binding=self.binding, parent_pid=self.parent, child_pid=self.child, sequence=self.sequence, kind=kind)
        if kind in {"terminal", "validation_read_started"}:
            # The child deadline governs whether this event may be sealed.
            # Once sealed, a full write can be accepted by the parent's
            # separate process-delivery budget even if send()'s post-write
            # diagnostic observes the child deadline expiring.  Validation
            # start also seals its pending flag here so a delivered frame can
            # still produce the one-shot Contract B terminal.
            check_budget(self.deadline)
            self.sequence += 1
            self.state = kind
            if kind == "validation_read_started":
                self._validation_pending = True
            self.writer.send(raw, self.deadline)
            return
        self.writer.send(raw, self.deadline)
        self.sequence += 1
        self.state = kind

    def control(self, kind, *, sequence):
        check_budget(self.deadline)
        raw = self.reader.receive(self.deadline)
        result = decode(raw, nonce=self.nonce, binding=self.binding, parent_pid=self.parent, child_pid=self.child, sequence=sequence, kind=kind)
        check_budget(self.deadline)
        return result["data"]

    def canonical_opened(self, paths, deadline, *, in_transaction=False):
        if deadline is not self.deadline:
            raise AcceptanceObserverError("budget_identity")
        check_budget(deadline)
        if identity(paths.repo_root) != self.binding["fixture_identity"] or identity(paths.canonical_database_path) != self.binding["canonical_identity"]:
            raise AcceptanceObserverError("fixture_identity")
        if self.opened:
            return  # Semantic second query cannot replay this one-shot barrier.
        self.reader.require_empty()
        self.opened = True
        self.emit("canonical_opened", {"in_transaction": in_transaction})
        self.control("release", sequence=1)

    def validation_read_started(self, deadline, *, stage, in_transaction):
        with self._lock:
            if deadline is not self.deadline:
                raise AcceptanceObserverError("budget_identity")
            self.emit("validation_read_started", {"stage": stage, "operation": "PRAGMA integrity_check",
                "in_transaction": in_transaction, "budget_ms": 1000, "started_ns": deadline.started_ns,
                "deadline_ns": deadline.deadline_ns, "observed_ns": time.monotonic_ns()})

    def validation_finished(self):
        # Publish the first-read return before contending with terminal's state
        # lock. A delayed state update must never hide an already-returned read.
        # One-shot: this observer observes only the first validation read.
        completed = self._validation_completed.complete()
        with self._lock:
            if not completed or self.state == "validation_no_result_before_deadline":
                raise AcceptanceObserverError("late_validation_result")
            self._validation_pending = False

    def canonical_failure(self, error):
        from codex_memory.retrieval.canonical_snapshot import _exception_chain
        causes = [{"type": type(node).__name__, "sqlite_errorcode": getattr(node, "sqlite_errorcode", None), "sqlite_errorname": getattr(node, "sqlite_errorname", None)} for node in _exception_chain(error)]
        self.emit("canonical_failure", {"kind": error.kind.value, "stage": error.stage.value, "causes": causes})

    def query_result(self, result):
        from dataclasses import asdict
        self.emit("query", {"status": result.status.value, "reasons": [v.value for v in result.degraded_reasons],
            "canonical_highwater": None if result.canonical_highwater is None else asdict(result.canonical_highwater),
            "generation": None if result.generation is None else result.generation.generation_id})

    def terminal(self, result):
        with self._lock:
            self.reader.require_empty()
            if (result.reason == "deadline_exhausted" and self.state == "validation_read_started"
                    and self._validation_pending and time.monotonic_ns() >= self.deadline.deadline_ns
                    and self._validation_completed.claim_if_unset()):
                data = {"stage": "invariant_validation", "operation": "PRAGMA integrity_check", "in_transaction": True,
                    "budget_ms": 1000, "started_ns": self.deadline.started_ns, "deadline_ns": self.deadline.deadline_ns,
                    "observed_ns": time.monotonic_ns()}
                raw = frame(nonce=self.nonce, binding=self.binding, parent_pid=self.parent, child_pid=self.child,
                    sequence=self.sequence, kind="validation_no_result_before_deadline", data=data)
                decode(raw, nonce=self.nonce, binding=self.binding, parent_pid=self.parent, child_pid=self.child,
                    sequence=self.sequence, kind="validation_no_result_before_deadline")
                # Seal BEFORE the nonblocking attempt; neither a late worker
                # nor a failed diagnostic can reopen this request.
                self.state = "validation_no_result_before_deadline"
                self.sequence += 1
                self._validation_pending = False
                Pipe._send_validation_deadline(self.writer, raw)
                return
            if result.reason == "deadline_exhausted" and self.state == "validation_read_started":
                raise AcceptanceObserverError("child_deadline_not_proven")
            self.emit("terminal", {"reason": result.reason, "recorded": result.recorded, "injection_id": result.injection_id})

    def close(self):
        for pipe in (self.reader, self.writer):
            if pipe.handle is not None:
                pipe.kernel.CloseHandle(wintypes.HANDLE(pipe.handle))
                pipe.handle = None
