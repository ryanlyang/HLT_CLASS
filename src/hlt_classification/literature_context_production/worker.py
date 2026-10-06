"""Scheduled exact-replay gate, bounded generation, and publication."""
import importlib.metadata
from pathlib import Path
import platform
import time

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms2jc2_response.measurement import Measurement
from hlt_classification.literature_proxy.campaign import exclusive
from hlt_classification.literature_proxy_v3.inputs import file_key
from hlt_classification.literature_proxy_v3.contracts import read_reference
from . import campaign as c, codec, engine, output, population as pop, storage, submission
from .contracts import artifact, checked, load_json, safe, write, atomic_publish_bytes, sha256_file, GIB


def environment():
    return dict(python=platform.python_version(), machine=platform.machine(),
                numpy=importlib.metadata.version('numpy'))


def require_environment(frozen):
    old = frozen['environment']
    expected = dict(python=old['python'], machine=old['machine'], numpy=old['packages']['numpy'])
    if environment() != expected:
        raise PermissionError('Python/architecture/NumPy differs from frozen pilot; explicit portability study required')


def pilot_sample(study, spec, *, count=64):
    """Read only selected training rows through production ROOT path; saved CONTEXT target."""
    candidates, expected = [], {}
    for record in spec['inputs']:
        selected = record['v1']['selected']
        pairs = list(zip(selected['entries'], selected['identities']))[:count-len(candidates)]
        if not pairs:
            break
        candidates.extend((selected['path'], int(e), i) for e, i in pairs)
        path = Path(spec['root'])/'blocks'/f'{file_key(record)}.npz'
        receipt = load_json(Path(spec['root'])/'receipt.json')
        if sha256_file(path) != receipt['outputs'][path.relative_to(spec['root']).as_posix()]:
            raise ValueError('Saved pilot block changed')
        with np.load(path, allow_pickle=False) as data:
            identities, offsets = data['identity'].tolist(), data['CONTEXT_offsets']
            for _, identity in pairs:
                index = identities.index(identity)
                lo, hi = map(int, offsets[index:index+2])
                p = Particles(*(data['CONTEXT_'+k][lo:hi] for k in codec.FIELDS),
                              tuple(str(j) for j in range(hi-lo)))
                expected[identity] = codec.physical_digest(identity, p)
        if len(candidates) == count:
            break
    raw = []
    for shard in study['shards']:
        if shard['role'] != 'train':
            continue
        chosen = [(e, i) for path, e, i in candidates if path == shard['path']]
        if not chosen:
            continue
        _, allowed = pop.entries_for(study['population'], shard)
        entries = [e for e, _ in chosen if e in allowed]
        if entries:
            raw.extend(pop.iterate(study['data_root'], study['population'], shard, selected=entries))
    if not raw or {i for i, _ in raw} != set(expected) or len(raw) != len(expected):
        raise ValueError('Pilot replay rows escaped production membership')
    return raw, expected


def replay(study, attempt, spec, rates, workers):
    """Same production reader, process engine and bank writer as bulk generation."""
    start = time.monotonic()
    raw, expected = pilot_sample(study, spec)
    serial_start = time.monotonic()
    serial = [r for rows, _ in engine.parallel(iter(raw), rates, 1) for r in rows]
    serial_seconds = time.monotonic()-serial_start
    if {i: codec.physical_digest(i, p) for i, p, _ in serial} != expected:
        raise ValueError('Production raw offline -> CONTEXT differs from saved pilot')
    directory = safe(study['root'], f'attempts/{attempt["name"]}/miniature')

    def publish(name, blob):
        if storage.usage(directory)+len(blob) > 128*2**20:
            raise OSError('Miniature exceeded fixed metadata allowance')
        storage.check_space(study['root'], len(blob))
        path = directory/name
        atomic_publish_bytes(path, blob)
        return path

    result = engine.process(iter(raw), rates, workers, publish, chunk=1)
    actual = {}
    for block in result['blocks']:
        for identity, p in output.particles(output.arrays(block['path'])):
            actual[identity] = codec.physical_digest(identity, p)
    if actual != expected or result['jets'] != len(expected):
        raise ValueError('Production process/writer replay differs')
    return dict(jets=len(raw), exact_replay=True, serial_seconds=serial_seconds,
                inverse_max_scaled_error=result['inverse_max_scaled_error'],
                process_write_seconds=result['seconds'], bytes=result['output_bytes'],
                elapsed_seconds=time.monotonic()-start)


def preflight(study, attempt):
    frozen = c.bundle(study)
    require_environment(frozen)
    directory = safe(study['root'], f'attempts/{attempt["name"]}')
    with exclusive(directory/'preflight.lock'), Measurement() as measure:
        started = time.monotonic()
        checked(frozen['pilot_receipt']); checked(frozen['pilot_report'])
        spec, report, original = c.authenticate_pilot(checked(frozen['pilot']))
        if spec['content_hash'] != study['pilot_hash'] or spec['calibration'] != frozen['calibration']:
            raise ValueError('Pilot evidence changed')
        population = pop.build(read_reference(study['inventory']), load_json(checked(study['profile'])),
                               read_reference(study['donor_profile']))
        if (population != study['population'] or original['data_root'] != study['data_root']
                or pop.shards(population) != study['shards']):
            raise ValueError('Production population/input location differs')
        for reference in attempt['retained_shards'].values():
            output.verify_shard(study, load_json(checked(reference)), physical=False)
        authentication_seconds = time.monotonic()-started
        storage.check_space(study['root'], max(0, study['storage']['budget_bytes']-storage.usage(study['root'])))
        sample = replay(study, attempt, spec, frozen['calibration'], 36)
    measurements = measure.report()
    largest = max(s['jets'] for s in study['shards'])
    seconds = authentication_seconds+sample['process_write_seconds']+4*sample['serial_seconds']/sample['jets']*largest/36
    memory = 2*measurements['sampled_peak_tree_rss_bytes']+2*GIB
    projected_bytes = 2*sample['bytes']/sample['jets']*sum(study['counts'].values())+storage.METADATA_ALLOWANCE
    ok = seconds < 2*3600 and memory < 64*GIB and projected_bytes <= study['storage']['budget_bytes']
    result = artifact('PREFLIGHT', parents=dict(study=study['content_hash'], attempt=attempt['content_hash']),
        **sample, authentication_seconds=authentication_seconds, measurement=measurements,
        projected_shard_seconds=seconds, projected_memory_bytes=memory, projected_dataset_bytes=projected_bytes,
        resource_envelope_ok=bool(ok), projections_are_estimates=True, environment=environment())
    if not ok:
        write(directory/'preflight_rejected.json', result)
        raise OSError('Preflight resource envelope exceeded; array not admitted, evidence preserved')
    write(directory/'preflight.json', result)
    return result


def run(attempt_path, task, index=None):
    study, attempt, _ = submission.get_plan(attempt_path)
    identity = submission.worker_identity(study, attempt, task, index)
    frozen = c.bundle(study)
    require_environment(frozen)
    print(f'JC2-CTX-PROD task={task} index={index} job={identity["job_id"]} start=true', flush=True)
    if task == 'preflight':
        return preflight(study, attempt)
    c.require_preflight(study, attempt)
    if task == 'generate':
        shard_id = attempt['shards'][index]
        shard = next(s for s in study['shards'] if s['shard_id'] == shard_id)
        return engine.generate_shard(study, attempt, shard, frozen['calibration'])
    if task == 'finalize':
        # Each role's completeness is independent; test is never exposed for training.
        output.manifest(study, 'train')
        output.manifest(study, 'validation')
        return output.manifest(study)
    raise ValueError('Unknown production task')
