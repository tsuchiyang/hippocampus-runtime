"""Bounded Windows audit storage. Returned bytes are held-object snapshots.

No host acknowledgement is inferred. A late install may leave a prepared artifact
but never returns a successful commit. Callers isolate uninterruptible OS I/O.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from .blob_store import BlobRef, BlobStore, BlobIntegrityError
from .injection_manifest import (InjectionManifest, InjectionAuditError, _freeze,
                                 _strict_decode, _require_timestamp)
from .ledger_common import canonical_json, ValidationError
from .memory_security import SecretRejected
from .paths import RuntimePathError, ensure_safe_child_directory
from .retrieval.deadline import DeadlineBudget

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PACKET_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024
MAX_RECEIPTS = 128
MAX_RECEIPT_AGGREGATE_BYTES = 1024 * 1024
_REF_KEYS = ('prepared_packet_ref', 'legacy_additional_context_ref', 'serialized_response_ref')
_WIRE_ENCODINGS = {
    'utf-8': 'utf-8', 'utf8': 'utf-8',
    'utf-16-le': 'utf-16-le', 'utf16le': 'utf-16-le',
    'utf-16-be': 'utf-16-be', 'utf16be': 'utf-16-be',
    'gbk': 'gbk', 'cp936': 'gbk',
    'cp1252': 'cp1252', 'windows-1252': 'cp1252',
}


def _wire_encoding(value):
    # Finite aliases only; never dispatch caller strings into codec lookup.
    if not isinstance(value, str) or len(value) > 32:
        raise InjectionAuditError('unsupported')
    codec = _WIRE_ENCODINGS.get(value.lower().replace('_', '-'))
    if codec is None:
        raise InjectionAuditError('unsupported')
    return codec


def _check(deadline):
    if time.monotonic_ns() >= deadline.deadline_ns:
        raise InjectionAuditError('deadline_exhausted')
    if os.name != 'nt':
        raise InjectionAuditError('unsupported')


@contextmanager
def _errors():
    # Imports occur only inside optional audit work, never Phase 0 startup.
    from .retrieval.semantic_store import SemanticStoreError
    try:
        yield
    except InjectionAuditError:
        raise
    except FileNotFoundError as exc:
        raise InjectionAuditError('missing', cause=exc) from exc
    except (RuntimePathError, SemanticStoreError) as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise InjectionAuditError('missing', cause=exc) from exc
        raise InjectionAuditError('path_unsafe', cause=exc) from exc
    except BlobIntegrityError as exc:
        raise InjectionAuditError('tampered', cause=exc) from exc
    except SecretRejected as exc:
        raise InjectionAuditError('invalid_input', cause=exc) from exc
    except (ValidationError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise InjectionAuditError('invalid_input', cause=exc) from exc
    except OSError as exc:
        raise InjectionAuditError('unavailable', cause=exc) from exc


def _hold(stack, path, *, directory, deadline, install=False):
    from .retrieval.semantic_store import _open_windows_acceptance_handle, _windows_kernel32, _windows_handle_state
    _check(deadline)
    if install and directory:
        import ctypes
        kernel = _windows_kernel32()
        handle = kernel.CreateFileW(str(path), 0x80000000, 0x3, None, 3, 0x02200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        stack.callback(kernel.CloseHandle, handle)
        state = _windows_handle_state(handle)
        if not state[0] & 0x10 or state[0] & 0x400:
            raise InjectionAuditError('path_unsafe')
    else:
        handle, state = _open_windows_acceptance_handle(path, directory=directory)
        stack.callback(_windows_kernel32().CloseHandle, handle)
    _check(deadline)
    return handle, state


@contextmanager
def _directories(path, *, create, deadline, install=None, release_deadline=None):
    """Hold ancestors; an optional read-only lease retains the original start.

    Acquisition still uses ``deadline``. A caller with an existing fallback
    window may hold the same handles through that window, without reopening
    paths or extending any creation/installation work.
    """
    if release_deadline is not None and (
        create or install is not None or not isinstance(release_deadline, DeadlineBudget)
        or release_deadline.started_ns != deadline.started_ns
        or release_deadline.deadline_ns < deadline.deadline_ns
    ):
        raise InjectionAuditError('invalid_input')
    path = Path(os.path.abspath(path))  # lexical normalization, never resolve redirects
    with _errors(), ExitStack() as stack:
        for index, item in enumerate((*reversed(path.parents), path)):
            _check(deadline)
            if create and index:
                ensure_safe_child_directory(item.parent, item)
            _hold(stack, item, directory=True, deadline=deadline, install=item == install)
        yield stack
        _check(deadline if release_deadline is None else release_deadline)


def _read(stack, path, *, limit, deadline):
    from .retrieval.semantic_store import _read_windows_handle
    handle, state = _hold(stack, path, directory=False, deadline=deadline)
    if state[3] > limit:
        raise InjectionAuditError('size_limit')
    try:
        raw = bytes(_read_windows_handle(handle, max_bytes=limit, deadline_ns=deadline.deadline_ns))
    except Exception:
        _check(deadline)
        raise
    _check(deadline)
    return raw


def _scan(raw, *, encoding='utf-8'):
    from .retrieval.evidence_pack import reject_semantic_secrets
    codec = _wire_encoding(encoding)
    try:
        text = raw.decode(codec, errors='strict')
    except UnicodeError as exc:
        raise InjectionAuditError('tampered', cause=exc) from exc
    reject_semantic_secrets(text)
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        pass
    else:
        reject_semantic_secrets(parsed)
    return text


def _blob(blobs, ref, *, deadline, encoding='utf-8'):
    if ref.byte_size > MAX_PACKET_BYTES:
        raise InjectionAuditError('size_limit')
    path = blobs.path_for(ref)
    with _directories(path.parent, create=False, deadline=deadline) as stack:
        raw = _read(stack, path, limit=MAX_PACKET_BYTES, deadline=deadline)
        if len(raw) != ref.byte_size or hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise InjectionAuditError('tampered')
        _scan(raw, encoding=encoding)
        return raw


def put_audit_blob(blobs: BlobStore, data: bytes, *, deadline: DeadlineBudget,
                   encoding: str = 'utf-8') -> BlobRef:
    """Guard and scan exact bytes; optional encoding is for observed wire only.

    No transcoding or newline changes. Manifest packet refs still require UTF-8.
    """
    with _errors():
        _check(deadline)
        if not isinstance(data, bytes):
            raise InjectionAuditError('invalid_input')
        if len(data) > MAX_PACKET_BYTES:
            raise InjectionAuditError('size_limit')
        _scan(data, encoding=encoding)
        digest = hashlib.sha256(data).hexdigest()
        ref = BlobRef(digest, len(data), f'sha256/{digest[:2]}/{digest}.bin')
        path = blobs.path_for(ref)
        with _directories(path.parent, create=True, deadline=deadline, install=path.parent):
            blobs.put_bytes(data)
            if _blob(blobs, ref, deadline=deadline, encoding=encoding) != data:
                raise InjectionAuditError('tampered')
        return ref


@dataclass(frozen=True, slots=True)
class InjectionPaths:
    repo_root: Path
    injections_dir: Path
    receipts_dir: Path
    temp_dir: Path

    @classmethod
    def for_repo(cls, repo_root: Path) -> InjectionPaths:
        root = Path(os.path.abspath(repo_root))
        base = root / '.codex-memory' / 'injections'
        return cls(root, base, base / 'receipts', base / 'temp')

    def ensure_for_write(self) -> None:
        self._ensure(DeadlineBudget.start(5000, now_ns=time.monotonic_ns))

    def _ensure(self, deadline):
        with _directories(self.repo_root, create=False, deadline=deadline):
            for path in (self.injections_dir, self.receipts_dir, self.temp_dir):
                with _directories(path, create=True, deadline=deadline):
                    pass


@dataclass(frozen=True, slots=True)
class CommittedInjection:
    injection_id: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedInjection:
    manifest: InjectionManifest
    prepared_packet_bytes: bytes
    legacy_additional_context_bytes: bytes
    serialized_response_bytes: bytes
    receipts: tuple = ()


def _id(value, prefix):
    if not isinstance(value, str) or re.fullmatch(prefix + r'[0-9a-f]{64}', value) is None:
        raise InjectionAuditError('invalid_input')


@contextmanager
def _serialized(paths, deadline):
    """Cooperating processes serialize directory handles using the same budget.

    READ-sharing validation prevents concurrent directory installation on Windows.
    The mutex coordinates our operations; it is not a security boundary.
    """
    import ctypes
    _check(deadline)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.ReleaseMutex.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    key = hashlib.sha256(str(paths.injections_dir).casefold().encode('utf-8')).hexdigest()
    handle = kernel.CreateMutexW(None, False, 'Local\\codex-memory-injection-' + key)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        wait_ms = max(0, min(60000, (deadline.deadline_ns - time.monotonic_ns()) // 1_000_000))
        status = kernel.WaitForSingleObject(handle, wait_ms)
        if status == 258:
            raise InjectionAuditError('deadline_exhausted')
        if status not in (0, 128):  # acquired or abandoned; all disk bytes still verified
            raise InjectionAuditError('unavailable')
        acquired = True
        _check(deadline)
        yield
        _check(deadline)
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


class InjectionStore:
    def __init__(self, paths: InjectionPaths, blobs: BlobStore):
        self.paths, self.blobs = paths, blobs

    def _packets(self, manifest, deadline):
        packets = manifest.body['packets']
        refs = [BlobRef(**packets[key]) for key in _REF_KEYS]
        if sum(ref.byte_size for ref in refs) > MAX_PACKET_BYTES:
            raise InjectionAuditError('size_limit')
        values = tuple(_blob(self.blobs, ref, deadline=deadline) for ref in refs)
        chars = len(values[0].decode('utf-8'))
        if chars != packets['packet_chars'] or chars > manifest.body['request']['max_chars']:
            raise InjectionAuditError('tampered')
        return values

    def _manifest(self, path, injection_id, deadline):
        with _directories(path.parent, create=False, deadline=deadline) as stack:
            raw = _read(stack, path, limit=MAX_MANIFEST_BYTES, deadline=deadline)
            try:
                manifest = InjectionManifest.from_bytes(raw)
            except InjectionAuditError as exc:
                raise InjectionAuditError('tampered', cause=exc) from exc
            if manifest.injection_id != injection_id or manifest.to_bytes() != raw:
                raise InjectionAuditError('tampered')
            _scan(raw)
            values = self._packets(manifest, deadline)
            return VerifiedInjection(manifest, *values)

    def _install(self, raw, destination, leaf, validate, deadline):
        """Private exclusive staging, fsync, no-replace install, final held validation.

        Staging directory and leaf handles must close for Windows MoveFileW.
        Replacement in that narrow window cannot yield success without final exact
        validation. Orphans are retained; no pathname-based recursive cleanup.
        """
        from .retrieval.semantic_store import _atomic_directory_rename_noreplace, SemanticStoreError
        with _directories(self.paths.temp_dir, create=False, deadline=deadline, install=destination.parent):
            with _directories(destination.parent, create=False, deadline=deadline, install=destination.parent):
                stage = self.paths.temp_dir / uuid4().hex
                stage.mkdir()
                with ExitStack() as leaf_stack:
                    with _directories(stage, create=False, deadline=deadline):
                        with (stage / leaf).open('xb') as stream:
                            stream.write(raw)
                            stream.flush()
                            os.fsync(stream.fileno())
                        observed = _read(leaf_stack, stage / leaf, limit=len(raw), deadline=deadline)
                        if observed != raw:
                            raise InjectionAuditError('tampered')
                        validate(stage / leaf)
                    leaf_stack.close()
                    _check(deadline)
                    try:
                        _atomic_directory_rename_noreplace(stage, destination)
                    except SemanticStoreError:
                        import ctypes
                        failure = ctypes.WinError(ctypes.windll.kernel32.GetLastError())
                        # Only a fully validated identical winner is idempotent.
                        try:
                            validate(destination / leaf)
                        except InjectionAuditError as exc:
                            raise exc from failure
                    _check(deadline)
                    validate(destination / leaf)
                    _check(deadline)

    def commit(self, manifest: InjectionManifest, *, deadline: DeadlineBudget) -> CommittedInjection:
        with _errors(), _serialized(self.paths, deadline):
            _check(deadline)
            manifest = InjectionManifest.from_bytes(manifest.to_bytes())
            raw = manifest.to_bytes()
            _scan(raw)
            self._packets(manifest, deadline)  # Blob-before-manifest, even staging.
            self.paths._ensure(deadline)
            def validate(path):
                found = self._manifest(path, manifest.injection_id, deadline)
                if found.manifest.to_bytes() != raw:
                    raise InjectionAuditError('tampered')
            self._install(raw, self.paths.injections_dir / manifest.injection_id,
                          'manifest.json', validate, deadline)
            _check(deadline)
            return CommittedInjection(manifest.injection_id, manifest.manifest_sha256)

    def load(self, injection_id: str, *, deadline: DeadlineBudget | None = None) -> VerifiedInjection:
        """Read only; inherit a caller budget, otherwise use the existing 5s cap."""
        if deadline is None:
            deadline = DeadlineBudget.start(5000, now_ns=time.monotonic_ns)
        elif not isinstance(deadline, DeadlineBudget):
            raise InjectionAuditError('invalid_input')
        else:
            # A caller may narrow the old read cap, never extend it. Preserve
            # its original cutoff rather than handing reopen a fresh budget.
            deadline = DeadlineBudget(deadline.started_ns,
                min(deadline.deadline_ns, time.monotonic_ns() + 5_000_000_000))
        with _errors(), _serialized(self.paths, deadline):
            _id(injection_id, 'inj_')
            found = self._manifest(self.paths.injections_dir / injection_id / 'manifest.json', injection_id, deadline)
            receipts = self._receipts(found.manifest, deadline)
            return VerifiedInjection(found.manifest, found.prepared_packet_bytes,
                                     found.legacy_additional_context_bytes,
                                     found.serialized_response_bytes, receipts)

    def _receipt(self, raw, *, expected_id=None):
        if len(raw) > MAX_RECEIPT_BYTES:
            raise InjectionAuditError('size_limit')
        document = _strict_decode(raw)
        if not isinstance(document, dict) or set(document) != {'body', 'receipt_sha256', 'receipt_id'}:
            raise InjectionAuditError('tampered')
        body = document['body']
        if not isinstance(body, dict) or set(body) != {'format', 'injection_id', 'packet_sha256', 'observed_at', 'kind', 'observations'}:
            raise InjectionAuditError('invalid_input')
        if body['format'] != 'codex-memory.injection-receipt.v1':
            raise InjectionAuditError('unsupported')
        _id(body['injection_id'], 'inj_')
        _id(body['packet_sha256'], '')
        _require_timestamp(body['observed_at'], 'observed_at')
        observations = body['observations']
        fields = {'runtime_completion': {'audit_duration_ms'},
                  'parent_process_exit': {'exit_code', 'wall_duration_ms'},
                  'wire_observation': {'observed_wire_response_ref', 'observation_layer', 'encoding', 'newline'}}
        kind = body['kind']
        if not isinstance(kind, str) or kind not in fields or not isinstance(observations, dict) or set(observations) != fields[kind]:
            raise InjectionAuditError('invalid_input')
        for key, value in observations.items():
            if key in {'audit_duration_ms', 'wall_duration_ms', 'exit_code'}:
                if isinstance(value, bool) or not isinstance(value, int) or (key != 'exit_code' and value < 0):
                    raise InjectionAuditError('invalid_input')
            elif key != 'observed_wire_response_ref' and (not isinstance(value, str) or not value):
                raise InjectionAuditError('invalid_input')
        if kind == 'wire_observation':
            _wire_encoding(observations['encoding'])
            ref = observations['observed_wire_response_ref']
            if not isinstance(ref, dict) or set(ref) != {'sha256', 'byte_size', 'relative_path'}:
                raise InjectionAuditError('invalid_input')
            self.blobs.path_for(BlobRef(**ref))
        digest = hashlib.sha256(canonical_json(body).encode('utf-8')).hexdigest()
        if document['receipt_sha256'] != digest or document['receipt_id'] != 'rcpt_' + digest or (expected_id is not None and expected_id != document['receipt_id']):
            raise InjectionAuditError('tampered')
        if canonical_json(document).encode('utf-8') != raw:
            raise InjectionAuditError('tampered')
        _scan(raw)
        return document

    def _receipt_link(self, document, manifest, deadline):
        body = document['body']
        if body['injection_id'] != manifest.injection_id or body['packet_sha256'] != manifest.body['packets']['prepared_packet_ref']['sha256']:
            raise InjectionAuditError('tampered')
        if body['kind'] == 'wire_observation':
            _blob(self.blobs, BlobRef(**body['observations']['observed_wire_response_ref']),
                  deadline=deadline, encoding=body['observations']['encoding'])

    def append_receipt(self, receipt: dict[str, object], *, deadline: DeadlineBudget) -> str:
        with _errors(), _serialized(self.paths, deadline):
            _check(deadline)
            digest = hashlib.sha256(canonical_json(receipt).encode('utf-8')).hexdigest()
            document = {'body': receipt, 'receipt_sha256': digest, 'receipt_id': 'rcpt_' + digest}
            raw = canonical_json(document).encode('utf-8')
            document = self._receipt(raw)
            injection_id = document['body']['injection_id']
            manifest = self._manifest(self.paths.injections_dir / injection_id / 'manifest.json', injection_id, deadline).manifest
            self._receipt_link(document, manifest, deadline)
            self.paths._ensure(deadline)
            self._receipt_capacity(raw, document['receipt_id'], deadline)
            def validate(path):
                with _directories(path.parent, create=False, deadline=deadline) as stack:
                    found = _read(stack, path, limit=MAX_RECEIPT_BYTES, deadline=deadline)
                    if found != raw:
                        raise InjectionAuditError('tampered')
                    self._receipt_link(self._receipt(found, expected_id=document['receipt_id']), manifest, deadline)
            self._install(raw, self.paths.receipts_dir / document['receipt_id'], 'receipt.json', validate, deadline)
            return document['receipt_id']

    def _receipt_capacity(self, raw, receipt_id, deadline):
        count, total, exists = 0, 0, False
        with _directories(self.paths.receipts_dir, create=False, deadline=deadline):
            with os.scandir(self.paths.receipts_dir) as entries:
                for entry in entries:
                    _check(deadline)
                    count += 1
                    if count > MAX_RECEIPTS:
                        raise InjectionAuditError('size_limit')
                    with _directories(Path(entry.path), create=False, deadline=deadline) as stack:
                        data = _read(stack, Path(entry.path) / 'receipt.json', limit=MAX_RECEIPT_BYTES, deadline=deadline)
                        self._receipt(data, expected_id=entry.name)
                        total += len(data)
                        exists = exists or entry.name == receipt_id
                    if total > MAX_RECEIPT_AGGREGATE_BYTES:
                        raise InjectionAuditError('size_limit')
        if not exists and (count >= MAX_RECEIPTS or total + len(raw) > MAX_RECEIPT_AGGREGATE_BYTES):
            raise InjectionAuditError('size_limit')

    def _receipts(self, manifest, deadline):
        results, total = [], 0
        with ExitStack() as root:
            try:
                root.enter_context(_directories(self.paths.receipts_dir, create=False, deadline=deadline))
            except InjectionAuditError as exc:
                if exc.reason == 'missing':
                    return ()
                raise
            with os.scandir(self.paths.receipts_dir) as entries:
                for count, entry in enumerate(entries, 1):
                    _check(deadline)
                    if count > MAX_RECEIPTS:
                        raise InjectionAuditError('size_limit')
                    try:
                        _id(entry.name, 'rcpt_')
                    except InjectionAuditError as exc:
                        raise InjectionAuditError('tampered', cause=exc) from exc
                    with _directories(Path(entry.path), create=False, deadline=deadline) as stack:
                        raw = _read(stack, Path(entry.path) / 'receipt.json', limit=MAX_RECEIPT_BYTES, deadline=deadline)
                        total += len(raw)
                        if total > MAX_RECEIPT_AGGREGATE_BYTES:
                            raise InjectionAuditError('size_limit')
                        try:
                            document = self._receipt(raw, expected_id=entry.name)
                        except InjectionAuditError as exc:
                            raise InjectionAuditError('tampered', cause=exc) from exc
                        if document['body']['injection_id'] == manifest.injection_id:
                            self._receipt_link(document, manifest, deadline)
                            results.append(_freeze(document))
        return tuple(sorted(results, key=lambda item: item['receipt_id']))
