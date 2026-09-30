"""Tigris-only direct ROOT replay, measured resources and finite speed report."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import platform
import statistics
import time

from . import generation_direct_campaign as c, generation_benchmark_data as data
from . import generation_benchmark_engine as engine, generation_portable_data as portable
from . import dev_campaign as dev
from .bridge import Particles
from .contracts import artifact, load_json, validate, safe_relative
from .dev_data import checked_file
from .generation_benchmark_worker import storage_info
from .measurement import Measurement
from .provenance import numerical_environment


@contextmanager
def stream(spec, study, task_id, *, limit=None):
    task = next(t for t in spec['tasks'] if t['task_id'] == task_id)
    if (spec.get('contract') != c.CONTRACT or task['action'] not in ('jt_gate', 'jt_run')
            or limit not in (None, 64) or (limit == 64) != (task['action'] == 'jt_gate')
            or spec['study']['path'] != str(Path(study['root'])/'study_spec.json')):
        raise PermissionError('No direct Tigris TRAIN reader capability')
    claim = load_json(dev.stage_dir(spec)/'claims'/task_id/'claim.json')
    validate(claim, 'DEV_CLAIM', parents={'stage': spec['content_hash']})
    if claim['task_id'] != task_id:
        raise PermissionError('Wrong direct Tigris read claim')
    original, _, _ = c.inputs(study)
    iterator = data._iterate(original, limit=limit)
    try:
        yield iterator
    finally:
        iterator.close()


def compare_reference(row, root, original, gate):
    # The existing tolerant comparator checks all exact discrete fields, flags,
    # identities/order and generation keys. Only 64 cross-site rows exist here.
    return portable.compare_run(row, root, {'gate_reference': gate['serial']}, original['root'], gate=True)


def prefix_digest(row, root, count=64):
    """Physical digest independent of full-run output block boundaries."""
    digest, identities, consumed = hashlib.sha256(), hashlib.sha256(), 0
    for block in row['blocks']:
        values = portable.arrays(portable.checked_block(root, block))
        for i, (lo, hi) in enumerate(zip(values['offsets'][:-1], values['offsets'][1:])):
            identity = bytes(values['jet_identity'][i]).hex()
            particle = Particles(*(values[k][lo:hi] for k in engine.FIELDS), tuple(str(j) for j in range(hi-lo)))
            digest.update(bytes.fromhex(engine.physical_digest(identity, particle)))
            identities.update(bytes.fromhex(identity))
            consumed += 1
            if consumed == count:
                return dict(physical_digest=digest.hexdigest(), ordered_identities=identities.hexdigest())
    raise ValueError('Incomplete Tigris gate prefix in full run')


def validate_run(row, root, membership):
    if (row['jets'] != 10000 or row['ordered_identities'] != membership['ordered_identities']
            or len(row['blocks']) != 10 or any(b['jets'] != 1000 for b in row['blocks'])
            or len({b['relative'] for b in row['blocks']}) != 10
            or sum(b['bytes'] for b in row['blocks']) != row['output_bytes']
            or not math.isfinite(row['processing_seconds']) or row['processing_seconds'] <= 0):
        raise ValueError('Direct Tigris full-run population/blocks/timing differs')
    for block in row['blocks']:
        portable.checked_block(root, block)


def gate_for_screen(spec, study):
    parent = load_json(checked_file(spec['parent_spec']))
    return c.accepted(parent, study)


def run(spec, task_id):
    from .dev_worker import allocation, publish_result, publish_outputs
    from .dev_submission import scheduler_identity
    started = time.perf_counter()
    print(f'CMS2JC2-TG phase=authenticate task={task_id}', flush=True)
    study = c.validate_stage(spec)
    task = next(t for t in spec['tasks'] if t['task_id'] == task_id)
    allocated = allocation(study, task)
    allocated['scheduler_evidence'] = scheduler_identity(spec, study, task_id, allocated['job_id'])
    env = numerical_environment()
    if env != study['numerical_environment']:
        raise ValueError('Direct Tigris allocated-node numerical environment differs')
    for parent in task['depends_on']:
        dev.verified_outputs(spec, parent)
    if task['action'] == 'jt_run':
        c.storage_check(study['root'], gate_for_screen(spec, study)['projection'])
    claim = dev.stage_dir(spec)/'claims'/task_id
    claim.mkdir(parents=True, exist_ok=False)
    dev.write(spec['root'], f'stages/{spec["name"]}/claims/{task_id}/claim.json',
        artifact('DEV_CLAIM', parents={'stage': spec['content_hash']}, task_id=task_id, allocation=allocated), 'DEV_CLAIM')
    original, reference, bundle = c.inputs(study)
    authentication_seconds = time.perf_counter()-started
    with Measurement() as measured:
        prefix = f'generated/{spec["name"]}/{task_id}'
        if task['action'] == 'jt_gate':
            reading = time.perf_counter()
            with stream(spec, study, task_id, limit=64) as source:
                pairs = list(source)
            input_seconds = time.perf_counter()-reading
            rows, checks = [], []
            for variant, workers, trace in (('serial', 1, False), ('trace', 1, True), ('parallel', 4, False)):
                print(f'CMS2JC2-TG phase=gate variant={variant}', flush=True)
                row = engine.process(iter(pairs), bundle, root=Path(spec['root']), relative=f'{prefix}/{variant}',
                    workers=workers, chunk=8, trace=trace, compression='stored', expected_count=64)
                checks.append(compare_reference(row, spec['root'], original, reference))
                rows.append(row)
            engine.parity(rows)
            result = dict(jets=64, serial=rows[0], replay_runs=rows[1:], comparisons=checks,
                compatible=True, within_site_exact=True, cross_site_jets=64, input_read_seconds=input_seconds)
        elif task['action'] == 'jt_run':
            with stream(spec, study, task_id) as source:
                row = engine.process(source, bundle, root=Path(spec['root']), relative=prefix,
                    workers=task['params']['workers'], chunk=32, compression='stored', expected_count=10000)
            validate_run(row, spec['root'], original['membership'])
            prefix_check = prefix_digest(row, spec['root'])
            gate = gate_for_screen(spec, study)['serial']
            if any(prefix_check[k] != gate[k] for k in prefix_check):
                raise ValueError('Full run changed Tigris gate prefix')
            result = dict(run=row, task_id=task_id, params=task['params'], gate_prefix=prefix_check,
                          cross_site_jets=64, full_run_cross_site_compared=False)
        else:
            result = report(spec, study)
    print(f'CMS2JC2-TG phase=final_authenticate task={task_id}', flush=True)
    c.validate_stage(spec)
    if numerical_environment() != env:
        raise ValueError('Direct Tigris numerical environment changed')
    outputs = {}
    if task['action'] != 'jt_report':
        result.update(environment=env, architecture=platform.machine(), allocation=allocated,
            measurement=measured.report(), filesystem=storage_info(spec['root']),
            initial_authentication_seconds=authentication_seconds, total_worker_seconds=time.perf_counter()-started)
        is_gate = task['action'] == 'jt_gate'
        if is_gate:
            result['projection'] = c.projection(result)
        result = artifact('TG_GATE' if is_gate else 'TG_RUN', parents=c.parents(spec), **result)
        rows = [result['serial'], *result['replay_runs']] if is_gate else [result['run']]
        for i, row in enumerate(rows):
            for j, block in enumerate(row['blocks']):
                outputs[f'block_{i}_{j}'] = safe_relative(Path(spec['root']), block['relative'])
    kind = result['contract'].removeprefix('CMS2JC2_RESPONSE_').split('/')[0]
    outputs['result'] = publish_result(spec, task_id, result, kind)
    return publish_outputs(spec, task_id, outputs)


def report(spec, study):
    original, _, _ = c.inputs(study)
    gate = gate_for_screen(spec, study)['serial']
    rows = []
    for task in spec['tasks']:
        if task['action'] != 'jt_run':
            continue
        row = dev.product(spec, task['task_id'], 'result')
        validate(row, 'TG_RUN', parents=c.parents(spec))
        if (row['task_id'] != task['task_id'] or row['params'] != task['params']
                or row['environment'] != study['numerical_environment'] or row['cross_site_jets'] != 64
                or row['full_run_cross_site_compared'] is not False):
            raise ValueError('Direct Tigris run registration differs')
        validate_run(row['run'], spec['root'], original['membership'])
        prefix = prefix_digest(row['run'], spec['root'])
        if prefix != row['gate_prefix'] or any(prefix[k] != gate[k] for k in prefix):
            raise ValueError('Saved Tigris prefix changed')
        rows.append(row)
    engine.parity([r['run'] for r in rows])
    summaries = []
    for workers in c.WORKERS:
        runs = [r for r in rows if r['params']['workers'] == workers]
        if len(runs) != 2 or sorted(r['params']['repeat'] for r in runs) != [0, 1]:
            raise ValueError('Missing direct Tigris repeat; no survivor selection')
        if any(not math.isfinite(r['total_worker_seconds'])
               or r['total_worker_seconds'] < r['run']['processing_seconds'] for r in runs):
            raise ValueError('Invalid direct Tigris worker timing')
        process = statistics.mean(r['run']['processing_seconds'] for r in runs)
        total = statistics.mean(r['total_worker_seconds'] for r in runs)
        summaries.append(dict(workers=workers, processing_jets_per_second=10000/process,
            worker_jets_per_second=10000/total, mean_processing_seconds=process, mean_worker_seconds=total,
            repeat_seconds=[r['total_worker_seconds'] for r in runs],
            allocated_cpu_hours_per_million=total*workers/10000*1e6/3600,
            bytes_per_jet=statistics.mean(r['run']['output_bytes'] for r in runs)/10000))
    return artifact('TG_REPORT', parents=c.parents(spec), rows=[r['content_hash'] for r in rows],
        settings=summaries, fastest_workers=min(summaries, key=lambda r: r['mean_worker_seconds'])['workers'],
        efficient_workers=min(summaries, key=lambda r: r['allocated_cpu_hours_per_million'])['workers'],
        raw_root_ingestion_measured=True, cross_site_jets=64, full_run_cross_site_compared=False,
        within_site_exact=True, queue_time_included=False, production_qualified=False)


def read(spec):
    study = c.validate_stage(spec, source=False)
    if spec['stage'] == 'tigris_direct_gate':
        return c.accepted(spec, study)
    row = dev.product(spec, 'jt_report', 'result')
    validate(row, 'TG_REPORT', parents=c.parents(spec))
    if row != report(spec, study):
        raise ValueError('Direct Tigris report changed')
    return row


def render(spec):
    row = read(spec)
    if spec['stage'] == 'tigris_direct_gate':
        return 'Tigris 64-jet replay passed. Separately review the 16/36/72-CPU screen.\n'+str(row['projection'])
    lines = ['Tigris direct ROOT + JOINT generation + output/readback.',
             'workers    processing jets/s    worker jets/s    CPU hours/million']
    lines += [f'{r["workers"]:7} {r["processing_jets_per_second"]:20.2f} {r["worker_jets_per_second"]:16.2f} '
              f'{r["allocated_cpu_hours_per_million"]:20.2f}' for r in row['settings']]
    return '\n'.join(lines+[f'Fastest: {row["fastest_workers"]}; CPU-efficient: {row["efficient_workers"]}',
        'Cross-site comparison: 64 jets only. Within-Tigris exact parity: all 10k, all six runs.',
        'TRAIN-only engineering evidence. No production or final-test authorization.'])
