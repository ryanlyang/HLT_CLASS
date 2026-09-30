"""Physical-bank validation, complete commit marker, train/validation reader."""
from pathlib import Path
import hashlib

import numpy as np

from .contracts import artifact, validate, safe, load_json, sha256_file, write, digest_ids
from . import population as pop, storage
from hlt_classification.cms2jc2_response import generation_benchmark_engine as engine
from hlt_classification.cms2jc2_response.bridge import Particles


def arrays(path):
    with np.load(path, allow_pickle=False) as bank:
        expected = {*engine.FIELDS, 'offsets', 'jet_identity'}
        if set(bank.files) != expected:
            raise ValueError('Physical bank fields differ')
        values = {k: bank[k] for k in expected}
    offsets, identity = values['offsets'], values['jet_identity']
    if (offsets.dtype != np.dtype('<i8') or offsets.ndim != 1 or len(offsets) < 2
            or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or identity.dtype != np.dtype('u1') or identity.shape != (len(offsets)-1, 32)):
        raise ValueError('Physical bank offsets/identities differ')
    for name, dtype in zip(engine.FIELDS, engine.DTYPES):
        expected_shape = (int(offsets[-1]), 4) if name in ('p4', 'tracking', 'valid') else (int(offsets[-1]),)
        if values[name].dtype != np.dtype(dtype) or values[name].shape != expected_shape:
            raise ValueError('Physical bank shape/dtype differs: '+name)
    return values


def particles(values):
    for i, (lo, hi) in enumerate(zip(values['offsets'][:-1], values['offsets'][1:])):
        p = Particles(*(values[k][lo:hi] for k in engine.FIELDS), tuple(str(j) for j in range(hi-lo)))
        yield bytes(values['jet_identity'][i]).hex(), p


def verify_shard(study, receipt, *, physical=True):
    shard = next(s for s in study['shards'] if s['shard_id'] == receipt['shard_id'])
    validate(receipt, 'SHARD', parents={'study': study['content_hash'],
        'population': study['population']['content_hash']}, test=shard['role'] == 'final_test')
    if (receipt['role'] != shard['role'] or receipt['jets'] != shard['jets']
            or receipt['ordered_identities'] != shard['ordered_identities']
            or receipt.get('physical_schema_verified') is not True
            or not receipt['blocks'] or receipt['source'] != study['source']['content_hash']
            or receipt['environment'] != study['numerical_environment']['content_hash']):
        raise ValueError('Shard identity/lineage differs')
    if shard['role'] == 'final_test':
        from .campaign import test_lock
        if receipt['test_lock'] != test_lock(study)['content_hash']:
            raise PermissionError('Sealed shard lacks exact build lock')
    elif receipt['test_lock'] is not None:
        raise ValueError('Non-test shard has test lock')
    row, entries = pop.entries_for(study['population'], shard)
    expected_ids = list(pop.ids(study['population']['parents']['inventory'], row, entries))
    physical_hash, actual_ids, count, total = hashlib.sha256(), [], 0, 0
    prefix = f"attempts/{receipt['attempt']}/shards/{shard['shard_id']}/"
    for index, block in enumerate(receipt['blocks']):
        if block['relative'] != prefix+f'block_{index:05d}.npz':
            raise ValueError('Shard block path/order differs')
        path = safe(study['root'], block['relative'])
        if path.stat().st_size != block['bytes'] or sha256_file(path) != block['sha256']:
            raise ValueError('Shard block checksum differs')
        values = arrays(path)
        if len(values['jet_identity']) != block['jets'] or not 1 <= block['jets'] <= 1000:
            raise ValueError('Block row count differs')
        if physical:
            for identity, p in particles(values):
                actual_ids.append(identity)
                physical_hash.update(bytes.fromhex(engine.physical_digest(identity, p)))
                count += len(p)
        else:
            # The committed worker already validated every physical object.
            # Recheck bytes/shape/membership without serially redoing millions
            # of numpy validation calls in a one-CPU final aggregation job.
            actual_ids.extend(bytes(i).hex() for i in values['jet_identity'])
            count += int(values['offsets'][-1])
        total += block['bytes']
    if (actual_ids != expected_ids or count != receipt['particles']
            or (physical and physical_hash.hexdigest() != receipt['physical_digest'])
            or total != receipt['output_bytes']):
        raise ValueError('Shard physical data/ordering differs')
    return receipt


def completed(study, *, physical=False):
    rows = {}
    for path in sorted(safe(study['root'], 'shards').glob('*.json')):
        row = verify_shard(study, load_json(path), physical=physical)
        if path.name != row['shard_id']+'.json' or row['shard_id'] in rows:
            raise ValueError('Duplicate/renamed shard receipt')
        rows[row['shard_id']] = row
    return rows


def finalize(study):
    from .campaign import require_preflight, test_lock, study_kind
    require_preflight(study)
    lock = test_lock(study)
    rows = completed(study)
    missing = [s['shard_id'] for s in study['shards'] if s['shard_id'] not in rows]
    if missing:
        raise ValueError(f'Incomplete dataset: {len(missing)} shards missing; no complete manifest published')
    counts = {role: sum(r['jets'] for r in rows.values() if r['role'] == role) for role in pop.COUNTS}
    if counts != pop.COUNTS or storage.usage(study['root']) > study['storage']['budget_bytes']:
        raise ValueError('Dataset counts/storage budget differ')
    refs = []
    for shard in study['shards']:
        relative = f"shards/{shard['shard_id']}.json"
        path = safe(study['root'], relative)
        refs.append(dict(relative=relative, sha256=sha256_file(path), content_hash=rows[shard['shard_id']]['content_hash']))
    reduced = study_kind(study) == 'STUDY_REDUCED_CONFIRMATION'
    extra = dict(confirmation_scope=study['confirmation_scope']) if reduced else {}
    result = artifact('DATASET_REDUCED_CONFIRMATION' if reduced else 'DATASET', **extra,
        parents={'study': study['content_hash'],
        'population': study['population']['content_hash'], 'test_lock': lock['content_hash']}, test=True,
        counts=counts, shards=refs, candidate='JOINT', replica=0,
        dataset_kind=study['dataset_kind'], physics_production_qualified=False,
        confirmation_status=study['confirmation_status'],
        reviewed_confirmation_hash=study['reviewed_confirmation_hash'],
        source=study['source'], numerical_environment=study['numerical_environment'],
        model_inputs=list(engine.FIELDS), identity_is_metadata=True, native_hlt_access=False,
        output_bytes=sum(r['output_bytes'] for r in rows.values()))
    write(safe(study['root'], 'dataset_manifest.json'), result)
    return result


def read_role(root, role):
    """Training/validation access only; does not expose mapping construction keys."""
    if role not in ('train', 'validation'):
        raise PermissionError('Final-test inference is sealed; this reader has no unlock mode')
    root = Path(root).resolve()
    study = load_json(root/'study_spec.json')
    from .campaign import validate_study, study_kind
    validate_study(study)
    manifest = load_json(root/'dataset_manifest.json')
    reduced = study_kind(study) == 'STUDY_REDUCED_CONFIRMATION'
    if reduced and manifest.get('confirmation_scope') != study['confirmation_scope']:
        raise ValueError('Dataset reduced evidence differs')
    validate(manifest, 'DATASET_REDUCED_CONFIRMATION' if reduced else 'DATASET',
        parents={'study': study['content_hash'],
        'population': study['population']['content_hash'],
        'test_lock': load_json(root/'test_build_lock.json')['content_hash']}, test=True)
    # Reader verifies only the authorized role's physical blocks, not sealed test particles.
    wanted = {s['shard_id'] for s in study['shards'] if s['role'] == role}
    found = set()
    for reference in manifest['shards']:
        path = safe(root, reference['relative'])
        if sha256_file(path) != reference['sha256']:
            raise ValueError('Manifest shard receipt changed')
        row = load_json(path)
        if row['content_hash'] != reference['content_hash']:
            raise ValueError('Manifest shard lineage differs')
        if row['shard_id'] not in wanted:
            continue
        if row['shard_id'] in found:
            raise ValueError('Duplicate manifest shard')
        found.add(row['shard_id'])
        verify_shard(study, row)
        for block in row['blocks']:
            values = arrays(safe(root, block['relative']))
            for identity, p in particles(values):
                yield identity, {k: getattr(p, k) for k in engine.FIELDS}
    if found != wanted:
        raise ValueError('Manifest omitted authorized population')
