"""Authenticated, relocatable TRAIN-only engineering packet; no ROOT on Tigris."""
import hashlib
from pathlib import Path

import numpy as np

from . import generation_benchmark_engine as engine
from .bridge import Particles
from .contracts import load_json, validate, safe_relative, sha256_file
from .readers import Pair
from .storage import publish_bytes, GIB

ATOL, RTOL = 1e-12, 1e-10


def arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        result = {k: handle[k] for k in handle.files}
    if set(result) != {*engine.FIELDS, 'offsets', 'jet_identity'}:
        raise ValueError('Portable arrays contain missing/forbidden fields')
    n = len(result['jet_identity'])
    if (result['jet_identity'].dtype != np.dtype('u1') or result['jet_identity'].shape != (n, 32)
            or result['offsets'].dtype != np.dtype('<i8') or result['offsets'].shape != (n+1,)):
        raise ValueError('Portable identities/offsets schema differs')
    for name, dtype in zip(engine.FIELDS, engine.DTYPES):
        if result[name].dtype != np.dtype(dtype):
            raise ValueError('Portable dtype differs: '+name)
    offsets = result['offsets']
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(result['p4']):
        raise ValueError('Portable offsets are invalid')
    # Particles' invariants are per particle, not per jet. Validate the entire
    # block once, rather than decoding again and constructing 1,000 redundant
    # per-jet objects just to check the same finite/charge/tracking/p4 rules.
    # Generation's own output readback/digests remain unchanged.
    Particles(*(result[k] for k in engine.FIELDS), tuple(str(i) for i in range(len(result['p4']))))
    return result


def checked_block(root, row):
    path = safe_relative(Path(root), row['relative'])
    if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
        raise ValueError('Portable block bytes changed: '+row['relative'])
    return path


def read_packet(path, expected_sha256):
    path = Path(path)
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError('Trusted portable manifest fingerprint differs')
    row = load_json(path)
    validate(row, 'PORT_PACKET')
    validate(row['bundle'], 'GEN_BUNDLE')
    validate(row['membership'], 'GEN_MEMBERSHIP')
    validate(row['source'], 'SOURCE')
    validate(row['environment'], 'NUMERICAL_ENVIRONMENT')
    if (row['role'] != 'train' or row['jets'] != 10000 or row['candidate'] != 'JOINT'
            or row['replica'] != 0 or row['native_hlt_access'] is not False
            or row['production_qualified'] is not False or row['membership']['role'] != 'train'
            or row['membership']['jets'] != row['jets']
            or row['reference']['jets'] != row['jets']
            or row['reference']['ordered_identities'] != row['membership']['ordered_identities']):
        raise PermissionError('Portable packet scope differs')
    relatives = []
    for blocks in (row['inputs'], row['reference']['blocks']):
        if len(blocks) != 10 or any(b['jets'] != 1000 for b in blocks):
            raise ValueError('Incomplete portable 10k block registry')
        for block in blocks:
            relatives.append(block['relative'])
            checked_block(path.parent, block)
    if len(set(relatives)) != len(relatives):
        raise ValueError('Portable blocks alias')
    gate = row['gate_reference']
    if (gate['jets'] != 64 or len(gate['blocks']) != 1 or gate['blocks'][0]['jets'] != 64
            or gate['ordered_identities'] != row['membership']['gate_ordered_identities']):
        raise ValueError('Portable gate reference differs')
    checked_block(path.parent, gate['blocks'][0])
    if gate['blocks'][0]['relative'] in relatives:
        raise ValueError('Portable gate aliases full blocks')
    return row


def export_stream(stream, root, blocks):
    rows = []
    def flush():
        value = engine.pack(rows)
        path = publish_bytes(Path(root), f'packet/inputs/block_{len(blocks):05d}.npz',
                             engine.encode(value, 'stored'), remaining_bytes=2*GIB)
        engine.readback(path, value)
        blocks.append(dict(relative=path.relative_to(Path(root)/'packet').as_posix(),
                           bytes=path.stat().st_size, sha256=sha256_file(path), jets=len(rows)))
        rows.clear()
    for pair in stream:
        if pair.hlt is not None or pair.diagnostic_class is not None:
            raise PermissionError('Portable input cannot include HLT or labels')
        if pair.offline.keys != tuple(f'part:{i}' for i in range(len(pair.offline))):
            raise ValueError('Unexpected offline random-key construction')
        rows.append((pair.identity, pair.offline, ''))
        if len(rows) == 1000:
            flush()
        yield pair
    if rows:
        flush()


def pairs(packet, root, *, limit=None):
    if limit not in (None, 64):
        raise PermissionError('Unregistered portable population')
    target = packet['jets'] if limit is None else 64
    count, identities, seen = 0, hashlib.sha256(), set()
    for block in packet['inputs']:
        path = checked_block(root, block)
        values = arrays(path)
        for index, (lo, hi) in enumerate(zip(values['offsets'][:-1], values['offsets'][1:])):
            if count == target:
                break
            identity = bytes(values['jet_identity'][index]).hex()
            if identity in seen:
                raise ValueError('Duplicate portable identity')
            seen.add(identity); identities.update(bytes.fromhex(identity)); count += 1
            p = Particles(*(values[k][lo:hi] for k in engine.FIELDS), tuple(f'part:{j}' for j in range(hi-lo)))
            yield Pair(identity, 'portable-train', p, None)
        checked_block(root, block)
        if count == target:
            break
    key = 'gate_ordered_identities' if limit else 'ordered_identities'
    if count != target or identities.hexdigest() != packet['membership'][key]:
        raise ValueError('Portable stream escaped frozen membership')


def compare_arrays(actual, reference):
    if set(actual) != set(reference):
        raise ValueError('Replay fields differ')
    exact, differences = True, {}
    for name in actual:
        a, b = actual[name], reference[name]
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Replay shape/dtype/nonfinite mismatch: '+name)
        identical = a.tobytes() == b.tobytes()
        exact = exact and identical
        if name in ('p4', 'tracking'):
            delta = np.abs(a-b)
            differences[name] = float(delta.max()) if delta.size else 0.
            if not np.all(delta <= ATOL+RTOL*np.abs(b)):
                raise ValueError('Cross-site numerical drift exceeds frozen tolerance: '+name)
        elif not identical:
            raise ValueError('Cross-site discrete drift: '+name)
    return dict(exact_bytes=exact, max_absolute_difference=differences, atol=ATOL, rtol=RTOL)


def compare_run(row, root, packet, packet_root, *, gate=False):
    count = 64 if gate else packet['jets']
    if row['jets'] != count:
        raise ValueError('Replay population differs')
    reference = packet['gate_reference'] if gate else packet['reference']
    for name in ('jets', 'particles', 'ordered_identities', 'generation_key_digest', 'flags'):
        if row[name] != reference[name]:
            raise ValueError('Cross-site discrete/key/flag drift: '+name)
    if len(row['blocks']) != len(reference['blocks']):
        raise ValueError('Missing replay blocks')
    reports = []
    for a, b in zip(row['blocks'], reference['blocks']):
        if a['jets'] != b['jets']:
            raise ValueError('Replay block boundary differs')
        reports.append(compare_arrays(arrays(checked_block(root, a)), arrays(checked_block(packet_root, b))))
    return dict(compatible=True, exact_bytes=all(r['exact_bytes'] for r in reports),
        max_absolute_difference={k: max(r['max_absolute_difference'][k] for r in reports)
                                 for k in ('p4', 'tracking')}, atol=ATOL, rtol=RTOL)
