"""Claimed export/replay tasks; cross-site agreement never implies qualification."""
from copy import deepcopy
from pathlib import Path
import platform
import statistics
import time

from . import generation_portable_campaign as c, generation_portable_data as d
from . import generation_benchmark_campaign as gen, generation_benchmark_data as gd
from . import generation_benchmark_engine as engine, dev_campaign as dev
from .generation_benchmark_worker import storage_info
from .contracts import artifact, load_json, safe_relative, sha256_file, validate
from .dev_data import checked_file, file_ref
from .measurement import Measurement
from .provenance import numerical_environment
from .storage import publish_bytes, GIB


def export(spec, study, allocated):
    parent = load_json(checked_file(study['donor']))
    original = gen.validate_stage(parent, source=False)
    gate = gen.accepted(parent)
    bundle = load_json(checked_file(original['bundle']))
    blocks, started = [], time.perf_counter()
    # This new export capability is explicitly claimed by run() before entry.
    # It reuses the exact membership validator and bounded offline-only reader;
    # no broad read capability is added to GEN_STAGE or the historical studies.
    gd.validate_membership(original)
    with _stream(original) as source:
        reference = engine.process(d.export_stream(source, spec['root'], blocks), bundle,
            root=Path(spec['root']), relative='packet/reference', workers=16, chunk=32,
            compression='stored', expected_count=10000)
    if reference['ordered_identities'] != original['membership']['ordered_identities']:
        raise ValueError('Export population differs')
    reference = deepcopy(reference)
    for b in reference['blocks']:
        b['relative'] = str(Path(b['relative']).relative_to('packet').as_posix())
    gate_reference = deepcopy(gate['serial'])
    for i, b in enumerate(gate_reference['blocks']):
        old = d.checked_block(parent['root'], b)
        new = publish_bytes(Path(spec['root']), f'packet/reference_gate/block_{i:05d}.npz',
                            old.read_bytes(), remaining_bytes=2*GIB)
        b['relative'] = new.relative_to(Path(spec['root'])/'packet').as_posix()
    return artifact('PORT_PACKET', parents={'export_stage': spec['content_hash'],
        'gate': gate['content_hash'], 'bundle': bundle['content_hash'],
        'membership': original['membership']['content_hash']},
        jets=10000, role='train', candidate='JOINT', replica=0, native_hlt_access=False,
        production_qualified=False, source=study['source'], environment=study['numerical_environment'],
        review=original['review'], membership=original['membership'], bundle=bundle,
        inputs=blocks, reference=reference, gate_reference=gate_reference, allocation=allocated,
        root_read_and_reference_seconds=time.perf_counter()-started)


def _stream(original):
    from contextlib import closing
    return closing(gd._iterate(original))


def run(spec, task_id):
    from .dev_worker import allocation, publish_result, publish_outputs
    from .dev_submission import scheduler_identity
    start = time.perf_counter()
    print(f'CMS2JC2-PORT phase=authenticate task={task_id}', flush=True)
    study = c.validate_stage(spec)
    task = next(t for t in spec['tasks'] if t['task_id'] == task_id)
    allocated = allocation(study, task)
    allocated['scheduler_evidence'] = scheduler_identity(spec, study, task_id, allocated['job_id'])
    env = numerical_environment()
    if env != study['numerical_environment']:
        raise ValueError('Portable allocated-node numerical environment differs')
    for parent in task['depends_on']:
        dev.verified_outputs(spec, parent)
    claim = dev.stage_dir(spec)/'claims'/task_id
    claim.mkdir(parents=True, exist_ok=False)
    dev.write(spec['root'], f'stages/{spec["name"]}/claims/{task_id}/claim.json',
        artifact('DEV_CLAIM', parents={'stage': spec['content_hash']}, task_id=task_id, allocation=allocated), 'DEV_CLAIM')
    auth = time.perf_counter()-start
    outputs = {}
    with Measurement() as measured:
        if task['action'] == 'jp_export':
            packet = export(spec, study, allocated)
            result = None
        elif task['action'] == 'jp_report':
            result = report(spec, study)
        else:
            ref = study['packet']
            packet = d.read_packet(ref['path'], ref['sha256'])
            packet_root = Path(ref['path']).parent
            prefix = f'generated/{spec["name"]}/{task_id}'
            if task['action'] == 'jp_gate':
                rows, checks = [], []
                for variant, workers, trace in (('serial', 1, False), ('trace', 1, True), ('parallel', 4, False)):
                    print(f'CMS2JC2-PORT phase=gate variant={variant}', flush=True)
                    row = engine.process(d.pairs(packet, packet_root, limit=64), packet['bundle'],
                        root=Path(spec['root']), relative=f'{prefix}/{variant}', workers=workers, chunk=8,
                        trace=trace, compression='stored', expected_count=64)
                    checks.append(d.compare_run(row, spec['root'], packet, packet_root, gate=True))
                    rows.append(row)
                engine.parity(rows)
                result = dict(jets=64, serial=rows[0], replay_runs=rows[1:], comparisons=checks,
                              compatible=True, within_site_exact=True)
            else:
                row = engine.process(d.pairs(packet, packet_root), packet['bundle'], root=Path(spec['root']),
                    relative=prefix, workers=task['params']['workers'], chunk=32,
                    compression='stored', expected_count=10000)
                result = dict(run=row, comparison=d.compare_run(row, spec['root'], packet, packet_root),
                              task_id=task_id, params=task['params'])
    print(f'CMS2JC2-PORT phase=final_authenticate task={task_id}', flush=True)
    c.validate_stage(spec)
    if numerical_environment() != env:
        raise ValueError('Portable environment changed during execution')
    common = dict(environment=env, architecture=platform.machine(), allocation=allocated,
        measurement=measured.report(), filesystem=storage_info(spec['root']),
        initial_authentication_seconds=auth, total_worker_seconds=time.perf_counter()-start)
    if task['action'] == 'jp_export':
        path = dev.write(spec['root'], 'packet/manifest.json', packet, 'PORT_PACKET')
        d.read_packet(path, sha256_file(path))
        outputs['packet'] = path
        for i, b in enumerate([*packet['inputs'], *packet['reference']['blocks'], *packet['gate_reference']['blocks']]):
            outputs[f'block_{i}'] = d.checked_block(path.parent, b)
        result = artifact('PORT_EXPORT', parents=c.parents(spec), packet=file_ref(path), **common)
    elif task['action'] != 'jp_report':
        result.update(common)
        gate = task['action'] == 'jp_gate'
        if gate:
            result['projection'] = c.projection(result, study['max_workers'])
        result = artifact('PORT_GATE' if gate else 'PORT_RUN', parents=c.parents(spec), **result)
        rows = [result['serial'], *result['replay_runs']] if gate else [result['run']]
        for i, row in enumerate(rows):
            for j, b in enumerate(row['blocks']):
                outputs[f'block_{i}_{j}'] = safe_relative(Path(spec['root']), b['relative'])
    kind = result['contract'].removeprefix('CMS2JC2_RESPONSE_').split('/')[0]
    outputs['result'] = publish_result(spec, task_id, result, kind)
    return publish_outputs(spec, task_id, outputs)


def report(spec, study):
    packet = d.read_packet(study['packet']['path'], study['packet']['sha256'])
    rows = []
    for task in spec['tasks']:
        if task['action'] != 'jp_run':
            continue
        row = dev.product(spec, task['task_id'], 'result')
        validate(row, 'PORT_RUN', parents=c.parents(spec))
        if (row['task_id'] != task['task_id'] or row['params'] != task['params']
                or row['environment'] != study['numerical_environment']):
            raise ValueError('Portable run registration differs')
        check = d.compare_run(row['run'], study['root'], packet, Path(study['packet']['path']).parent)
        if row['comparison'] != check:
            raise ValueError('Saved replay comparison differs')
        rows.append(row)
    engine.parity([r['run'] for r in rows])
    summaries = []
    for n in (v for v in (16, 36, 72, 144) if v <= study['max_workers']):
        runs = [r for r in rows if r['params']['workers'] == n]
        if len(runs) != 2 or sorted(r['params']['repeat'] for r in runs) != [0, 1]:
            raise ValueError('Missing portable repeat; no survivor selection')
        process = statistics.mean(r['run']['processing_seconds'] for r in runs)
        total = statistics.mean(r['total_worker_seconds'] for r in runs)
        if not (0 < process <= total):
            raise ValueError('Invalid portable timing')
        summaries.append(dict(workers=n, jets_per_second=10000/process, worker_jets_per_second=10000/total,
            mean_processing_seconds=process, mean_worker_seconds=total,
            repeat_seconds=[r['total_worker_seconds'] for r in runs],
            allocated_cpu_hours_per_million=total*n/10000*1e6/3600,
            bytes_per_jet=statistics.mean(r['run']['output_bytes'] for r in runs)/10000,
            exact_sporc_bytes=all(r['comparison']['exact_bytes'] for r in runs)))
    return artifact('PORT_REPORT', parents=c.parents(spec), rows=[r['content_hash'] for r in rows],
        settings=summaries, fastest_workers=min(summaries, key=lambda r: r['mean_worker_seconds'])['workers'],
        efficient_workers=min(summaries, key=lambda r: r['allocated_cpu_hours_per_million'])['workers'],
        raw_root_ingestion_measured=False, queue_time_included=False, production_qualified=False)


def read(spec):
    study = c.validate_stage(spec, source=False)
    if spec['stage'] == 'port_gate':
        return c.accepted(spec, study)
    task, kind = ('jp_export', 'PORT_EXPORT') if spec['stage'] == 'port_export' else ('jp_report', 'PORT_REPORT')
    result = dev.product(spec, task, 'result')
    validate(result, kind, parents=c.parents(spec))
    if spec['stage'] == 'port_screen' and result != report(spec, study):
        raise ValueError('Portable report changed')
    return result


def render(spec):
    row = read(spec)
    if spec['stage'] == 'port_export':
        return ('Copy the packet directory only after this successful receipt.\n'
                f'Manifest: {row["packet"]["path"]}\nTrusted SHA256: {row["packet"]["sha256"]}')
    if spec['stage'] == 'port_gate':
        return 'Tigris replay gate passed; screen is a separate submission.\n'+str(row['projection'])
    lines = ['Tigris cached-input generation + output/readback; NOT raw ROOT throughput.',
             'workers    processing jets/s    worker jets/s    exact SPORC bytes']
    lines += [f'{r["workers"]:7} {r["jets_per_second"]:20.2f} {r["worker_jets_per_second"]:16.2f} '
              f'{str(r["exact_sporc_bytes"]):>20}' for r in row['settings']]
    return '\n'.join(lines+[f'Fastest: {row["fastest_workers"]}; CPU-efficient: {row["efficient_workers"]}',
                             'TRAIN only. No production or final-test authorization.'])
