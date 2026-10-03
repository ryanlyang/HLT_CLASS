"""Authenticated physical banks and independent train/validation releases."""
from pathlib import Path
import hashlib
import numpy as np
from .contracts import artifact, validate, safe, load_json, sha256_file, write, canonical_sha256
from . import population as pop, storage, codec as engine
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
    from .campaign import validate_attempt, require_preflight
    shard = next(s for s in study['shards'] if s['shard_id'] == receipt['shard_id'])
    attempt = load_json(safe(study['root'], f"attempts/{receipt['attempt']}/attempt_spec.json"))
    validate_attempt(attempt, study)
    require_preflight(study, attempt)
    validate(receipt, 'SHARD', parents=dict(study=study['content_hash'],
        attempt=attempt['content_hash'], shard=canonical_sha256(shard)), test=shard['role'] == 'final_test')
    if shard['shard_id'] not in attempt['shards']:
        raise ValueError('Shard not in producing attempt')
    if (receipt['role'] != shard['role'] or receipt['jets'] != shard['jets']
            or receipt['ordered_identities'] != shard['ordered_identities']
            or receipt.get('physical_schema_verified') is not True
            or not receipt['blocks']):
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
        if sha256_file(path) != block['sha256']:
            raise ValueError('Shard block changed during read')
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


def completed(study, *, physical=False, role=None):
    rows = {}
    wanted = {s['shard_id'] for s in study['shards'] if role is None or s['role'] == role}
    for path in sorted(safe(study['root'], 'shards').glob('*.json')):
        if role is not None and path.stem not in wanted:
            continue  # Never open other roles' particle banks.
        row = verify_shard(study, load_json(path), physical=physical)
        if path.name != row['shard_id']+'.json' or row['shard_id'] in rows:
            raise ValueError('Duplicate/renamed shard receipt')
        rows[row['shard_id']] = row
    return rows


def manifest(study, role=None):
    from .campaign import test_lock
    if role is not None and role not in ('train', 'validation'):
        raise PermissionError('No final-test diagnostic/release mode')
    rows = completed(study, role=role)
    wanted = [s for s in study['shards'] if role is None or s['role'] == role]
    missing = [s['shard_id'] for s in wanted if s['shard_id'] not in rows]
    if missing:
        raise ValueError(f'Incomplete dataset: {len(missing)} shards missing; no manifest published')
    counts = {r: sum(v['jets'] for v in rows.values() if v['role'] == r)
              for r in ([role] if role is not None else pop.COUNTS)}
    if counts != {r: study['counts'][r] for r in counts} or storage.usage(study['root']) > study['storage']['budget_bytes']:
        raise ValueError('Dataset counts/storage budget differ')
    refs = []
    for shard in wanted:
        relative = f"shards/{shard['shard_id']}.json"
        refs.append(dict(relative=relative, sha256=sha256_file(safe(study['root'], relative)),
                         content_hash=rows[shard['shard_id']]['content_hash']))
    result = artifact('MANIFEST' if role is None else 'ROLE_MANIFEST',
        parents=dict(study=study['content_hash'], population=study['population']['content_hash']),
        test=role is None, counts=counts, shards=refs, role=role,
        test_lock=test_lock(study)['content_hash'] if role is None else None,
        dataset_kind=study['kind'], physics_production_qualified=False, recipe='NOISE_V3',
        model_inputs=list(engine.FIELDS), identity_is_metadata=True, native_hlt_accessed=False,
        output_bytes=sum(v['output_bytes'] for v in rows.values()))
    name = 'dataset_manifest.json' if role is None else f'releases/{role}.json'
    write(safe(study['root'], name), result)
    return result


def read_role(root, role):
    """Yield (canonical jet identity, physical arrays); no final-test unlock."""
    if role not in ('train', 'validation'):
        raise PermissionError('Final-test inference remains sealed')
    from .campaign import validate_study
    root = Path(root).resolve()
    study = load_json(root/'study_spec.json')
    validate_study(study)
    if Path(study['root']).resolve() != root:
        raise ValueError('Dataset root differs')
    path = root/f'releases/{role}.json'
    value = load_json(path)
    validate(value, 'ROLE_MANIFEST', parents=dict(study=study['content_hash'],
        population=study['population']['content_hash']), test=False)
    expected = [s['shard_id'] for s in study['shards'] if s['role'] == role]
    if (value['role'] != role or value['counts'] != {role: study['counts'][role]}
            or [Path(r['relative']).stem for r in value['shards']] != expected):
        raise ValueError('Released population differs')
    for reference in value['shards']:
        path = safe(root, reference['relative'])
        if sha256_file(path) != reference['sha256']:
            raise ValueError('Manifest receipt checksum differs')
        receipt = load_json(path)
        if receipt['content_hash'] != reference['content_hash'] or receipt['role'] != role:
            raise ValueError('Manifest receipt lineage differs')
        verify_shard(study, receipt, physical=False)
        for block in receipt['blocks']:
            path = safe(root, block['relative'])
            values = arrays(path)
            if sha256_file(path) != block['sha256']:
                raise ValueError('Consumer bank checksum differs')
            for identity, p in particles(values):
                yield identity, {k: getattr(p, k) for k in engine.FIELDS}


def status(study):
    """Metadata-only progress; not physical integrity certification."""
    counts = {r: 0 for r in pop.COUNTS}
    for shard in study['shards']:
        path = safe(study['root'], f'shards/{shard["shard_id"]}.json')
        if not path.is_file():
            continue
        row = load_json(path)
        validate(row, 'SHARD', test=shard['role'] == 'final_test')
        if (row['parents']['study'] != study['content_hash'] or row['parents']['shard'] != canonical_sha256(shard)
                or row['shard_id'] != shard['shard_id'] or row['jets'] != shard['jets']):
            raise ValueError('Progress receipt lineage differs')
        counts[shard['role']] += row['jets']
    return dict(committed_jets=counts, required=study['counts'], physical_blocks_reverified=False,
                complete_manifest=(Path(study['root'])/'dataset_manifest.json').is_file())

