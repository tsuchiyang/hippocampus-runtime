from dataclasses import asdict
import hashlib
import time
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_injection_manifest import manifest_body
from codex_memory.blob_store import BlobRef, BlobStore
from codex_memory.injection_manifest import InjectionManifest, InjectionAuditError
from codex_memory.retrieval.deadline import DeadlineBudget


def budget():
    return DeadlineBudget.start(10000, now_ns=time.monotonic_ns)


@pytest.fixture
def setup(tmp_path, manifest_body):
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    paths = InjectionPaths.for_repo(tmp_path)
    blobs = BlobStore(tmp_path / '.codex-memory' / 'blobs')
    refs = [blobs.put_bytes(raw) for raw in (b'packet', b'legacy', b'{"ok":true}\n')]
    for key, ref in zip(('prepared_packet_ref', 'legacy_additional_context_ref', 'serialized_response_ref'), refs):
        manifest_body['packets'][key] = asdict(ref)
    manifest_body['packets']['packet_chars'] = 6
    manifest_body['candidates']['omission_counts'] = dict.fromkeys(('budget', 'duplicate', 'provenance', 'suppressed', 'unsafe'), 0)
    return InjectionStore(paths, blobs), InjectionManifest.from_body(manifest_body), refs


def test_roundtrip_and_same_body_concurrency(setup):
    store, manifest, refs = setup
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: store.commit(manifest, deadline=budget()), range(2)))
    assert results[0] == results[1]
    loaded = store.load(manifest.injection_id)
    assert loaded.manifest.to_bytes() == manifest.to_bytes()
    assert loaded.prepared_packet_bytes == b'packet'
    assert loaded.serialized_response_bytes == b'{"ok":true}\n'


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'directory', 'hardlink'])
def test_bad_packet_never_commits(setup, tmp_path, damage):
    store, manifest, refs = setup
    path = store.blobs.path_for(refs[0])
    path.unlink()
    if damage == 'corrupt':
        path.write_bytes(b'broken')
    elif damage == 'directory':
        path.mkdir()
    elif damage == 'hardlink':
        sentinel = tmp_path / 'outside'
        sentinel.write_bytes(b'packet')
        path.hardlink_to(sentinel)
    with pytest.raises(InjectionAuditError):
        store.commit(manifest, deadline=budget())
    assert not (store.paths.injections_dir / manifest.injection_id).exists()
    if damage == 'hardlink':
        assert sentinel.read_bytes() == b'packet'


def test_corrupt_final_never_overwritten(setup):
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    leaf = store.paths.injections_dir / manifest.injection_id / 'manifest.json'
    leaf.write_bytes(b'foreign')
    with pytest.raises(InjectionAuditError):
        store.commit(manifest, deadline=budget())
    assert leaf.read_bytes() == b'foreign'


def test_expired_budget_no_commit(setup):
    store, manifest, _ = setup
    with pytest.raises(InjectionAuditError) as error:
        store.commit(manifest, deadline=DeadlineBudget(0, 0))
    assert error.value.reason == 'deadline_exhausted'


def test_load_does_not_create(tmp_path):
    from codex_memory.injection_store import InjectionPaths, InjectionStore
    store = InjectionStore(InjectionPaths.for_repo(tmp_path), BlobStore(tmp_path / '.codex-memory/blobs'))
    with pytest.raises(InjectionAuditError):
        store.load('inj_' + '0' * 64)
    assert not (tmp_path / '.codex-memory').exists()


def receipt(manifest):
    return {'format': 'codex-memory.injection-receipt.v1', 'injection_id': manifest.injection_id,
            'packet_sha256': manifest.body['packets']['prepared_packet_ref']['sha256'],
            'observed_at': '2026-09-14T00:00:02Z', 'kind': 'runtime_completion',
            'observations': {'audit_duration_ms': 5}}


def test_receipt_immutable_selfhashed_and_linked(setup):
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    before = manifest.to_bytes()
    first = store.append_receipt(receipt(manifest), deadline=budget())
    assert store.append_receipt(receipt(manifest), deadline=budget()) == first
    loaded = store.load(manifest.injection_id)
    assert loaded.manifest.to_bytes() == before
    assert loaded.receipts[0]['body']['observations']['audit_duration_ms'] == 5
    with pytest.raises(TypeError):
        loaded.receipts[0]['body']['observations']['audit_duration_ms'] = 99
    leaf = store.paths.receipts_dir / first / 'receipt.json'
    leaf.write_bytes(b'{}')
    with pytest.raises(InjectionAuditError) as error:
        store.load(manifest.injection_id)
    assert error.value.reason == 'tampered'


@pytest.mark.parametrize('change', ['host_ack', 'wrong_hash', 'boolean_duration'])
def test_receipt_rejects_unproven_observation(setup, change):
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    body = receipt(manifest)
    if change == 'host_ack':
        body['observations']['host_acknowledgement'] = 'received'
    elif change == 'wrong_hash':
        body['packet_sha256'] = '0' * 64
    else:
        body['observations']['audit_duration_ms'] = True
    with pytest.raises(InjectionAuditError):
        store.append_receipt(body, deadline=budget())


def test_replay_after_retrieval_deleted_preserves_canonical(setup):
    store, manifest, _ = setup
    runtime = store.paths.injections_dir.parent
    db = runtime / 'db'
    db.mkdir()
    canonical = db / 'memory.sqlite3'
    canonical.write_bytes(b'canonical fixture sentinel')
    cache = runtime / 'retrieval'
    cache.mkdir()
    (cache / 'disposable').write_bytes(b'health history physically lost with cache')
    store.commit(manifest, deadline=budget())
    before = store.load(manifest.injection_id)
    shutil.rmtree(cache)
    assert store.load(manifest.injection_id) == before
    assert canonical.read_bytes() == b'canonical fixture sentinel'


@pytest.mark.parametrize('target', ['manifest', 'packet', 'directory'])
def test_held_read_denies_real_replacement(setup, monkeypatch, target):
    from codex_memory.retrieval import semantic_store
    store, manifest, refs = setup
    store.commit(manifest, deadline=budget())
    leaf = store.paths.injections_dir / manifest.injection_id / 'manifest.json'
    path = {'manifest': leaf, 'packet': store.blobs.path_for(refs[0]), 'directory': leaf.parent}[target]
    original = semantic_store._read_windows_handle
    attacked = []
    def read(handle, **kwargs):
        if not attacked:
            # During manifest read, manifest leaf/directory held. Packet target
            # attack must run during its own held leaf read, below.
            if target == 'packet':
                state = semantic_store._windows_handle_state(handle)
                if state[3] != 6:
                    return original(handle, **kwargs)
            with pytest.raises(OSError) as error:
                path.rename(path.with_name(path.name + '.moved'))
            attacked.append(error.value.winerror)
        return original(handle, **kwargs)
    monkeypatch.setattr(semantic_store, '_read_windows_handle', read)
    assert store.load(manifest.injection_id).prepared_packet_bytes == b'packet'
    assert attacked == [32]


def test_stage_replacement_window_never_returns_committed(setup, monkeypatch):
    from codex_memory.retrieval import semantic_store
    store, manifest, _ = setup
    original = semantic_store._atomic_directory_rename_noreplace
    attacked = []
    def rename(source, destination):
        moved = destination.parent / (source.name + '.original')
        source.rename(moved)
        source.mkdir()
        (source / 'manifest.json').write_bytes(b'foreign replacement')
        attacked.append(moved)
        return original(source, destination)
    monkeypatch.setattr(semantic_store, '_atomic_directory_rename_noreplace', rename)
    with pytest.raises(InjectionAuditError):
        store.commit(manifest, deadline=budget())
    assert len(attacked) == 1
    assert (attacked[0] / 'manifest.json').read_bytes() == manifest.to_bytes()


def test_install_parent_rename_denied_and_no_replace_still_works(setup, monkeypatch):
    from codex_memory.retrieval import semantic_store
    store, manifest, _ = setup
    original = semantic_store._atomic_directory_rename_noreplace
    attacked = []
    def rename(source, destination):
        with pytest.raises(OSError) as error:
            destination.parent.rename(destination.parent.with_name('redirected'))
        attacked.append(error.value.winerror)
        return original(source, destination)
    monkeypatch.setattr(semantic_store, '_atomic_directory_rename_noreplace', rename)
    assert store.commit(manifest, deadline=budget()).injection_id == manifest.injection_id
    assert attacked == [32]


def test_initial_reparse_directory_rejected_outside_unchanged(setup, tmp_path):
    import subprocess
    store, manifest, _ = setup
    outside = tmp_path / 'outside-dir'
    outside.mkdir()
    sentinel = outside / 'sentinel'
    sentinel.write_bytes(b'untouched')
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(store.paths.injections_dir), str(outside)], capture_output=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(InjectionAuditError):
        store.commit(manifest, deadline=budget())
    assert sentinel.read_bytes() == b'untouched'
    assert list(outside.iterdir()) == [sentinel]


def test_late_install_leaves_prepared_but_returns_deadline_error(setup, monkeypatch):
    from codex_memory.retrieval import semantic_store
    from codex_memory import injection_store
    store, manifest, _ = setup
    original = semantic_store._atomic_directory_rename_noreplace
    deadline = budget()
    def rename(source, destination):
        original(source, destination)
        monkeypatch.setattr(injection_store.time, 'monotonic_ns', lambda: deadline.deadline_ns + 1)
    monkeypatch.setattr(semantic_store, '_atomic_directory_rename_noreplace', rename)
    with pytest.raises(InjectionAuditError) as error:
        store.commit(manifest, deadline=deadline)
    assert error.value.reason == 'deadline_exhausted'
    assert (store.paths.injections_dir / manifest.injection_id / 'manifest.json').read_bytes() == manifest.to_bytes()


def test_put_audit_blob_exact_and_guarded(setup):
    from codex_memory.injection_store import put_audit_blob
    store, _, _ = setup
    ref = put_audit_blob(store.blobs, '精确\n'.encode(), deadline=budget())
    assert store.blobs.read(ref) == '精确\n'.encode()


def test_two_processes_same_body_commit(setup):
    import subprocess
    import sys
    store, manifest, _ = setup
    script = '''import sys,time
from pathlib import Path
from codex_memory.injection_store import InjectionPaths,InjectionStore
from codex_memory.injection_manifest import InjectionManifest
from codex_memory.blob_store import BlobStore
from codex_memory.retrieval.deadline import DeadlineBudget
root=Path(sys.argv[1])
store=InjectionStore(InjectionPaths.for_repo(root),BlobStore(root/'.codex-memory/blobs'))
manifest=InjectionManifest.from_bytes(sys.stdin.buffer.read())
print(store.commit(manifest,deadline=DeadlineBudget.start(10000,now_ns=time.monotonic_ns)).injection_id)
'''
    children = [subprocess.Popen([sys.executable, '-c', script, str(store.paths.repo_root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda child: child.communicate(manifest.to_bytes(), timeout=15), children))
    for child, (stdout, stderr) in zip(children, results):
        assert child.returncode == 0, stderr
        assert stdout.decode().strip() == manifest.injection_id
    assert store.load(manifest.injection_id).manifest == manifest


def test_mutex_wait_keeps_caller_deadline(setup):
    from codex_memory.injection_store import _serialized
    store, manifest, _ = setup
    with _serialized(store.paths, budget()):
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(store.commit, manifest, deadline=DeadlineBudget.start(50, now_ns=time.monotonic_ns))
            with pytest.raises(InjectionAuditError) as error:
                future.result(timeout=2)
            assert error.value.reason == 'deadline_exhausted'


def test_audit_file_directory_rejected_without_outside_writes(setup):
    store, manifest, _ = setup
    store.paths.injections_dir.write_bytes(b'foreign sentinel')
    with pytest.raises(InjectionAuditError):
        store.commit(manifest, deadline=budget())
    assert store.paths.injections_dir.read_bytes() == b'foreign sentinel'


def test_fsync_failure_leaves_no_committed_manifest(setup, monkeypatch):
    from codex_memory import injection_store
    store, manifest, _ = setup
    def denied(_fd):
        raise PermissionError('fixture denies flush')
    monkeypatch.setattr(injection_store.os, 'fsync', denied)
    with pytest.raises(InjectionAuditError) as error:
        store.commit(manifest, deadline=budget())
    assert error.value.reason == 'unavailable'
    assert not (store.paths.injections_dir / manifest.injection_id).exists()


def test_packet_size_limit_before_staging(setup, monkeypatch):
    from codex_memory import injection_store
    store, manifest, _ = setup
    monkeypatch.setattr(injection_store, 'MAX_PACKET_BYTES', 1)
    with pytest.raises(InjectionAuditError) as error:
        store.commit(manifest, deadline=budget())
    assert error.value.reason == 'size_limit'
    assert not store.paths.injections_dir.exists()


def test_character_count_mismatch_refused(setup):
    store, manifest, _ = setup
    body = json.loads(manifest.to_bytes())['body']
    body['packets']['packet_chars'] = 5
    manifest = InjectionManifest.from_body(body)
    with pytest.raises(InjectionAuditError) as error:
        store.commit(manifest, deadline=budget())
    assert error.value.reason == 'tampered'


def test_missing_receipts_is_unknown(setup):
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    store.paths.receipts_dir.rmdir()
    assert store.load(manifest.injection_id).receipts == ()
    assert not store.paths.receipts_dir.exists()


def test_receipt_capacity_rejected_before_install(setup, monkeypatch):
    from codex_memory import injection_store
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    store.append_receipt(receipt(manifest), deadline=budget())
    monkeypatch.setattr(injection_store, 'MAX_RECEIPTS', 1)
    other = receipt(manifest)
    other['observations']['audit_duration_ms'] = 6
    with pytest.raises(InjectionAuditError) as error:
        store.append_receipt(other, deadline=budget())
    assert error.value.reason == 'size_limit'
    assert len(list(store.paths.receipts_dir.iterdir())) == 1


def test_manifest_initial_hardlink_rejected(setup, tmp_path):
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    leaf = store.paths.injections_dir / manifest.injection_id / 'manifest.json'
    sentinel = tmp_path / 'outside-manifest'
    sentinel.hardlink_to(leaf)
    with pytest.raises(InjectionAuditError):
        store.load(manifest.injection_id)
    assert sentinel.read_bytes() == manifest.to_bytes()


def test_stage_junction_swap_rejected_outside_sentinel_preserved(setup, tmp_path, monkeypatch):
    import subprocess
    from codex_memory.retrieval import semantic_store
    store, manifest, _ = setup
    outside = tmp_path / 'outside-audit'
    outside.mkdir()
    sentinel = outside / 'manifest.json'
    sentinel.write_bytes(manifest.to_bytes())
    original = semantic_store._atomic_directory_rename_noreplace
    attacked = []
    def rename(source, destination):
        source.rename(destination.parent / (source.name + '.original'))
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(source), str(outside)], capture_output=True)
        assert result.returncode == 0, result.stderr
        attacked.append(True)
        original(source, destination)
    monkeypatch.setattr(semantic_store, '_atomic_directory_rename_noreplace', rename)
    with pytest.raises(InjectionAuditError):
        store.commit(manifest, deadline=budget())
    assert attacked == [True]
    assert sentinel.read_bytes() == manifest.to_bytes()
    assert list(outside.iterdir()) == [sentinel]


def wire_receipt(manifest, ref, encoding):
    body = receipt(manifest)
    body['kind'] = 'wire_observation'
    body['observations'] = {'observed_wire_response_ref': asdict(ref),
                            'observation_layer': 'parent_process_pipe',
                            'encoding': encoding, 'newline': 'CRLF'}
    return body


@pytest.mark.parametrize(('declared', 'codec', 'text'), [
    ('gbk', 'gbk', '中文\r\n'), ('cp936', 'gbk', '中文\r\n'),
    ('UTF-8', 'utf-8', '中文\r\n'), ('utf-16-le', 'utf-16-le', '中文\r\n'),
    ('utf-16-be', 'utf-16-be', '中文\r\n'), ('cp1252', 'cp1252', 'café\r\n'),
])
def test_wire_receipt_declared_encoding_preserves_exact_bytes(setup, declared, codec, text):
    from codex_memory.injection_store import put_audit_blob
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    raw = text.encode(codec)
    ref = put_audit_blob(store.blobs, raw, deadline=budget(), encoding=declared)
    receipt_id = store.append_receipt(wire_receipt(manifest, ref, declared), deadline=budget())
    loaded = store.load(manifest.injection_id)
    observations = loaded.receipts[0]['body']['observations']
    assert loaded.receipts[0]['receipt_id'] == receipt_id
    assert observations['encoding'] == declared
    assert store.blobs.read(BlobRef(**observations['observed_wire_response_ref'])) == raw
    assert loaded.serialized_response_bytes == b'{"ok":true}\n'


@pytest.mark.parametrize(('encoding', 'raw', 'reason'), [
    ('unicode_escape', b'plain', 'unsupported'), ('utf-7', b'plain', 'unsupported'),
    ('gbk', b'\x81', 'tampered'), ('utf-16-le', b'\x01', 'tampered'),
    ('gbk', 'password=真实凭据\r\n'.encode('gbk'), 'invalid_input'),
    ('utf-16-be', 'password=真实凭据\r\n'.encode('utf-16-be'), 'invalid_input'),
])
def test_wire_receipt_encoding_invalid_bytes_and_secret_negatives(setup, encoding, raw, reason):
    store, manifest, _ = setup
    store.commit(manifest, deadline=budget())
    # An untrusted/external wire observer can supply existing BlobRefs.
    ref = store.blobs.put_bytes(raw)
    with pytest.raises(InjectionAuditError) as error:
        store.append_receipt(wire_receipt(manifest, ref, encoding), deadline=budget())
    assert error.value.reason == reason
    assert store.load(manifest.injection_id).receipts == ()


def test_prepared_packet_still_requires_utf8(setup):
    store, manifest, _ = setup
    ref = store.blobs.put_bytes('中文'.encode('gbk'))
    body = json.loads(manifest.to_bytes())['body']
    body['packets']['prepared_packet_ref'] = asdict(ref)
    body['packets']['packet_chars'] = 2
    with pytest.raises(InjectionAuditError) as error:
        store.commit(InjectionManifest.from_body(body), deadline=budget())
    assert error.value.reason == 'tampered'
