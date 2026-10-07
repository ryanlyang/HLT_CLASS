"""Re-encode authenticated V1 banks; exact membership, immutable publication."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import shutil
import time
import numpy as np

from hlt_classification.data.cache_contracts import atomic_publish_bytes
from hlt_classification.literature_context_consumer import RelocatedDataset, _check_file, _role
from hlt_classification.literature_context_production import output as old_output, population, codec
from hlt_classification.literature_context_production.contracts import validate as old_validate, artifact as old_artifact
from .contracts import artifact, validate, require, safe, write, checked, ref, load_json, sha256_file, canonical_sha256, GIB
from .kernel import recipe, reencode

COUNTS = dict(train=1000000, validation=250000, final_test=1000000)


def frozen_inverse(parent):
    from hlt_classification.literature_context import transform
    name = 'src/hlt_classification/literature_context/transform.py'
    require(parent.bundle['source']['file_sha256'].get(name) == sha256_file(Path(transform.__file__)),
            'V1 inverse implementation differs from the original production source')


def source(spec):
    r = spec['source_request']
    return RelocatedDataset(proxy_root=r['study_root'], offline_root=r['offline_root'],
        provenance_root=r['provenance_root'], expected_manifest_sha256=r['manifest_hash'])


def source_receipt(parent, shard, *, materialize_test=False):
    if shard['role'] != 'final_test':
        return parent._receipt(shard)
    if not materialize_test:
        raise PermissionError('Test input requires explicit materialization authority')
    reference = parent._references[shard['shard_id']]
    path = _check_file(safe(parent.proxy_root, reference['relative']), reference)
    row = load_json(path)
    require(row['content_hash'] == reference['content_hash'], 'Test receipt hash differs')
    attempt = load_json(safe(parent.proxy_root, f"attempts/{row['attempt']}/attempt_spec.json"))
    old_validate(attempt, 'ATTEMPT', parents=dict(study=parent.study['content_hash']), test=False)
    _check_file(parent.proxy_root/'study_spec.json', attempt['study'])
    require(attempt['name'] == row['attempt'] and shard['shard_id'] in attempt['shards'], 'Test attempt differs')
    preflight = load_json(safe(parent.proxy_root, f"attempts/{row['attempt']}/preflight.json"))
    old_validate(preflight, 'PREFLIGHT', parents=dict(study=parent.study['content_hash'],
        attempt=attempt['content_hash']), test=False)
    require(preflight['exact_replay'] and preflight['resource_envelope_ok'], 'V1 preflight differs')
    lock = load_json(safe(parent.proxy_root, 'test_build_lock.json'))
    require(lock == old_artifact('TEST_BUILD_LOCK', parents=dict(study=parent.study['content_hash'],
        population=parent.study['population']['content_hash']), materialize=True, evaluate=False,
        purpose='sealed_deterministic_materialization_only')
        and row['test_lock'] == lock['content_hash'], 'V1 test-build lock differs')
    old_validate(row, 'SHARD', parents=dict(study=parent.study['content_hash'],
        attempt=attempt['content_hash'], shard=canonical_sha256(shard)), test=True)
    require(all(row[k] == shard[k] for k in ('shard_id', 'role', 'jets', 'ordered_identities'))
        and row['physical_schema_verified'] is True and 0 <= row['inverse_max_scaled_error'] <= 1e-10,
        'V1 test receipt scope differs')
    prefix = f"attempts/{row['attempt']}/shards/{shard['shard_id']}/"
    require(row['blocks'] and all(b['relative'] == prefix+f'block_{i:05d}.npz'
        and 1 <= b['jets'] <= 1000 for i, b in enumerate(row['blocks']))
        and sum(b['jets'] for b in row['blocks']) == shard['jets']
        and sum(b['bytes'] for b in row['blocks']) == row['output_bytes'], 'V1 test block coverage differs')
    return row


def validate_spec(spec, *, check_source=False):
    from .workflow import policy, source_lock
    validate(spec, 'WORKFLOW', parents=dict(source=spec['source']['content_hash'],
        parent_manifest=spec['parent_manifest_hash'], selection=spec['selection_hash']))
    parent = source(spec)
    require(spec['recipe'] == recipe() and spec['policy'] == policy()
        and spec['counts'] == COUNTS == parent.study['counts']
        and spec['population'] == parent.study['population'] and spec['shards'] == parent.study['shards']
        and spec['parent_manifest_hash'] == parent.manifest['content_hash']
        and spec['materialize_final_test'] is True and spec['automatic_science'] is True,
        'V2 workflow scientific scope differs')
    from hlt_classification.cms_proxy_ladder import release
    original = load_json(checked(spec['original_release']))
    release.validate_release(original, root=Path(spec['original_release']['path']).parent)
    require(original['schema_version'] == 3 and original['content_hash'] == spec['selection_hash']
        and original['request'] == spec['source_request']
        and original['identity_sha256'] == spec['selection_identities']
        and original['counts'] == spec['policy']['counts'], 'Original V1 selection differs')
    checked(spec['original_gate'])
    gate = load_json(spec['original_gate']['path'])
    require(gate['request'] == spec['source_request'] and gate['source_commit'] == spec['parent_commit'],
            'Original gate/source differs')
    validate(spec['source'], 'SOURCE')
    require(spec['source_commit'] == spec['source']['commit'], 'V2 source commit differs')
    if check_source:
        require(source_lock(Path(spec['project_dir']), spec['source_commit']) == spec['source'], 'V2 source drift')
        frozen_inverse(parent)
    roots = [Path(spec[k]).resolve() for k in ('root', 'dataset_root', 'project_dir')]
    protected = [parent.proxy_root, parent.offline_root, parent.provenance_root]
    for i, a in enumerate(roots):
        for b in roots[i+1:]+protected:
            require(not (a.is_relative_to(b) or b.is_relative_to(a)), 'Output overlaps protected paths')
    expected = {s['shard_id'] for s in spec['shards']}
    require(set(spec['allowances']) == expected and all(type(v) is int and v > 0 for v in spec['allowances'].values())
        and sum(spec['allowances'].values()) + 2*GIB <= spec['storage']['dataset_budget_bytes']
        and spec['storage']['persistent'] is True
        and spec['storage']['available_quota_bytes'] >= spec['storage']['dataset_budget_bytes'] + 10*GIB,
        'Storage reservation/quota contract differs')
    return parent


def create(*, original_gate, root, dataset_root, project_dir, commit, available_quota_gib, persistent):
    from hlt_classification.cms_proxy_ladder import context, release
    from .workflow import policy, source_lock
    if persistent is not True or type(available_quota_gib) not in (int, float) or not np.isfinite(available_quota_gib):
        raise PermissionError('Explicit persistent storage and available quota required')
    root, dest, project = (Path(p).resolve() for p in (root, dataset_root, project_dir))
    if root.exists() or dest.exists():
        raise FileExistsError('Fresh separate workflow and dataset roots required; preserve existing outputs')
    gate_path = Path(original_gate).resolve(strict=True)
    gate = load_json(gate_path)
    context.validate_gate(gate)
    require(gate['schema_version'] == 8, 'Expected original CONTEXT_V1 Oscar gate')
    old_root = gate_path.parent/'release'
    original = load_json(old_root/'release.json')
    release.validate_release(original, root=old_root)
    require(original['request'] == gate['request'] and original['counts'] == policy()['counts'], 'V1 population differs')
    parent = source(dict(source_request=gate['request']))
    allowances = {s['shard_id']: 2*source_receipt(parent, s, materialize_test=True)['output_bytes']+4*2**20
                  for s in parent.study['shards']}
    lock = source_lock(project, commit)
    spec = artifact('WORKFLOW', parents=dict(source=lock['content_hash'],
        parent_manifest=parent.manifest['content_hash'], selection=original['content_hash']),
        root=str(root), dataset_root=str(dest), project_dir=str(project), source_commit=commit,
        source=lock, source_request=gate['request'], parent_commit=gate['source_commit'],
        parent_manifest_hash=parent.manifest['content_hash'], original_gate=ref(gate_path),
        original_release=ref(old_root/'release.json'), selection_hash=original['content_hash'],
        selection_identities=original['identity_sha256'], recipe=recipe(), policy=policy(),
        counts=parent.study['counts'], population=parent.study['population'], shards=parent.study['shards'],
        allowances=allowances, storage=dict(dataset_budget_bytes=20*GIB,
            available_quota_bytes=int(available_quota_gib*GIB), persistent=True),
        materialize_final_test=True, automatic_science=True)
    validate_spec(spec, check_source=True)
    for path in (root.parent, dest.parent):
        require(path.is_dir() and shutil.disk_usage(path).free >= 30*GIB, 'Insufficient filesystem headroom')
    root.mkdir()
    dest.mkdir()
    write(root/'workflow_spec.json', spec)
    write(dest/'study_spec.json', artifact('STUDY', parents=dict(workflow=spec['content_hash']),
        workflow=ref(root/'workflow_spec.json'), root=str(dest), recipe=recipe(),
        source_request=spec['source_request'], population=spec['population'], shards=spec['shards'],
        counts=spec['counts'], input_contract=recipe()['input_contract'],
        inventory=parent.study['inventory'], kind='controlled_context_v2_synthetic_proxy'))
    return spec


@contextmanager
def claim(root, name):
    path = safe(root, f'claims/{name}.active')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise PermissionError('Task active/interrupted; preserve claim and inspect before recovery') from exc
    try:
        yield
    except BaseException:
        raise  # Keep claim on failure, including after publication; never blindly replay.
    else:
        path.rmdir()


def _block_task(args):
    path, record, target = args
    path = _check_file(Path(path), record)
    values = old_output.arrays(path)
    _check_file(path, record)
    before = list(old_output.particles(values))
    rows = list(map(reencode, before))
    result = codec.pack(rows)
    for key in set(values)-{'tracking'}:
        require(result[key].dtype == values[key].dtype and result[key].tobytes() == values[key].tobytes(),
                'Frozen bank structure differs')
    blob = codec.encode(result, 'deflate')
    return target, blob, result, rows


def generate(spec, index, *, workers=6):
    parent = validate_spec(spec)
    frozen_inverse(parent)
    if type(index) is not int or not 0 <= index < len(spec['shards']) or type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError('Unregistered shard/worker count')
    shard = spec['shards'][index]
    root, name = Path(spec['dataset_root']), shard['shard_id']
    receipt_path = safe(root, f'shards/{name}.json')
    if receipt_path.exists():
        return verify_receipt(spec, shard, physical=True)
    original = source_receipt(parent, shard, materialize_test=spec['materialize_final_test'])
    row, entries = population.entries_for(spec['population'], shard)
    expected = list(population.ids(spec['population']['parents']['inventory'], row, entries))
    tasks = [(str(safe(parent.proxy_root, b['relative'])), b, b['relative']) for b in original['blocks']]
    started = time.monotonic()
    with claim(root, name):
        require(shutil.disk_usage(root).free >= spec['allowances'][name]+2*GIB, 'Insufficient free disk space')
        serial = _block_task(tasks[0])
        blocks, seen, size, particles, inverse, frontend = [], [], 0, 0, 0., 0.
        # At most six 1000-jet blocks in flight; no whole-dataset process map.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for offset in range(0, len(tasks), workers):
                futures = [pool.submit(_block_task, t) for t in tasks[offset:offset+workers]]
                for number, future in enumerate(futures, start=offset):
                    relative, blob, values, rows = future.result()
                    if number == 0:
                        require(blob == serial[1], 'Serial/process replay differs')
                        del serial
                    size += len(blob)
                    require(size <= spec['allowances'][name], 'Shard reservation exceeded')
                    require(shutil.disk_usage(root).free >= len(blob)+2*GIB, 'Insufficient output headroom')
                    path = safe(root, relative)
                    atomic_publish_bytes(path, blob)
                    codec.readback(path, values)
                    identities = [r[0] for r in rows]
                    seen.extend(identities)
                    particles += int(values['offsets'][-1])
                    inverse = max(inverse, max(r[2]['inverse_max_scaled_error'] for r in rows))
                    frontend = max(frontend, max(r[2]['frontend_max_scaled_error'] for r in rows))
                    blocks.append(dict(relative=relative, bytes=len(blob), sha256=sha256_file(path), jets=len(rows)))
        require(seen == expected and particles == original['particles'], 'V2 jet/particle membership differs')
        receipt = artifact('SHARD', parents=dict(workflow=spec['content_hash'], shard=canonical_sha256(shard),
            source=original['content_hash']), test=shard['role']=='final_test',
            shard_id=name, role=shard['role'], jets=shard['jets'], ordered_identities=shard['ordered_identities'],
            blocks=blocks, particles=particles, output_bytes=size, exact_process_replay=True,
            inverse_max_scaled_error=inverse, frontend_max_scaled_error=frontend,
            seconds=time.monotonic()-started, workers=workers, physical_schema_verified=True)
        write(receipt_path, receipt)
    return receipt


def verify_receipt(spec, shard, *, physical=False, parent=None):
    parent = source(spec) if parent is None else parent
    original = source_receipt(parent, shard, materialize_test=spec['materialize_final_test'])
    root = Path(spec['dataset_root'])
    receipt = load_json(safe(root, f"shards/{shard['shard_id']}.json"))
    validate(receipt, 'SHARD', parents=dict(workflow=spec['content_hash'], shard=canonical_sha256(shard),
        source=original['content_hash']), test=shard['role']=='final_test')
    require(all(receipt[k] == shard[k] for k in ('shard_id', 'role', 'jets', 'ordered_identities'))
        and receipt['exact_process_replay'] is True and receipt['physical_schema_verified'] is True
        and np.isfinite(receipt['inverse_max_scaled_error']) and 0 <= receipt['inverse_max_scaled_error'] <= 1e-10
        and np.isfinite(receipt['frontend_max_scaled_error']) and receipt['frontend_max_scaled_error'] >= 0
        and [b['relative'] for b in receipt['blocks']] == [b['relative'] for b in original['blocks']]
        and [b['jets'] for b in receipt['blocks']] == [b['jets'] for b in original['blocks']]
        and receipt['particles'] == original['particles'] and receipt['output_bytes'] <= spec['allowances'][shard['shard_id']]
        and sum(b['bytes'] for b in receipt['blocks']) == receipt['output_bytes'], 'V2 receipt structure differs')
    if physical:
        row, entries = population.entries_for(spec['population'], shard)
        expected = list(population.ids(spec['population']['parents']['inventory'], row, entries))
        seen, count = [], 0
        for b in receipt['blocks']:
            path = _check_file(safe(root, b['relative']), b)
            values = old_output.arrays(path)
            _check_file(path, b)
            require(len(values['jet_identity']) == b['jets'], 'V2 bank count differs')
            seen.extend(bytes(i).hex() for i in values['jet_identity'])
            count += int(values['offsets'][-1])
        require(seen == expected and count == receipt['particles'], 'V2 physical membership differs')
    return receipt


def finalize(spec):
    parent = validate_spec(spec)
    root = Path(spec['dataset_root'])
    receipts = [verify_receipt(spec, s, physical=True, parent=parent) for s in spec['shards']]
    counts = {role: sum(r['jets'] for r in receipts if r['role'] == role) for role in COUNTS}
    require(counts == spec['counts'], 'Full dataset is incomplete')
    refs = [dict(relative=f"shards/{r['shard_id']}.json", sha256=sha256_file(root/'shards'/f"{r['shard_id']}.json"),
                 content_hash=r['content_hash']) for r in receipts]
    value = artifact('MANIFEST', parents=dict(workflow=spec['content_hash'],
        study=load_json(root/'study_spec.json')['content_hash']), test=True,
        counts=counts, recipe=recipe(), shards=refs, output_bytes=sum(r['output_bytes'] for r in receipts))
    require(value['output_bytes']+2*GIB <= spec['storage']['dataset_budget_bytes'], 'Dataset budget exceeded')
    write(root/'dataset_manifest.json', value)
    return value


class Dataset(RelocatedDataset):
    """Reuse the exact paired reader, but authenticate an explicit V2 namespace."""
    def __init__(self, root):
        self.proxy_root = Path(root).resolve(strict=True)
        self.study = load_json(self.proxy_root/'study_spec.json')
        validate(self.study, 'STUDY')
        self.spec = load_json(checked(self.study['workflow']))
        parent = validate_spec(self.spec)
        self._parent = parent
        require(self.study['parents'] == dict(workflow=self.spec['content_hash'])
            and Path(self.spec['dataset_root']) == self.proxy_root
            and self.study['population'] == self.spec['population']
            and self.study['shards'] == self.spec['shards'] and self.study['recipe'] == recipe()
            and self.study['counts'] == self.spec['counts'] and self.study['source_request'] == self.spec['source_request']
            and self.study['input_contract'] == recipe()['input_contract']
            and self.study['inventory'] == parent.study['inventory']
            and self.study['root'] == str(self.proxy_root)
            and self.study['kind'] == 'controlled_context_v2_synthetic_proxy', 'V2 study differs')
        self.manifest = load_json(self.proxy_root/'dataset_manifest.json')
        validate(self.manifest, 'MANIFEST', parents=dict(workflow=self.spec['content_hash'],
            study=self.study['content_hash']), test=True)
        require(self.manifest['counts'] == self.spec['counts'] and self.manifest['recipe'] == recipe()
            and [r['relative'] for r in self.manifest['shards']] ==
            [f"shards/{s['shard_id']}.json" for s in self.spec['shards']], 'V2 manifest coverage differs')
        self.offline_root, self.provenance_root = parent.offline_root, parent.provenance_root
        self.inventory, self.local_references = parent.inventory, parent.local_references
        self._references = dict(zip((s['shard_id'] for s in self.spec['shards']), self.manifest['shards']))

    def _shards(self, role, shard_ids):
        _role(role)
        rows = [s for s in self.study['shards'] if s['role'] == role]
        if shard_ids is not None:
            ids = tuple(shard_ids)
            require(ids and len(ids)==len(set(ids)) and set(ids)<={s['shard_id'] for s in rows}, 'Invalid ordinary shard subset')
            rows = [s for s in rows if s['shard_id'] in ids]
        return rows

    def _receipt(self, shard):
        _role(shard['role'])
        reference = self._references[shard['shard_id']]
        path = _check_file(safe(self.proxy_root, reference['relative']), reference)
        require(load_json(path)['content_hash'] == reference['content_hash'], 'V2 committed receipt differs')
        return verify_receipt(self.spec, shard, parent=self._parent)

    def describe(self):
        return dict(root=str(self.proxy_root), counts=self.study['counts'], recipe='CONTEXT_V2',
                    manifest_hash=self.manifest['content_hash'], final_test_evaluated=False)
