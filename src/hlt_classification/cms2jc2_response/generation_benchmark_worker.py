"""Claimed generation-only workers, measurements, and a finite speed report."""
import math
import os
from pathlib import Path
import shutil
import statistics
import time

from . import generation_benchmark_campaign as campaign, generation_benchmark_data as data
from . import generation_benchmark_engine as engine, dev_campaign as dev
from .contracts import artifact, load_json, validate, safe_relative, sha256_file
from .dev_data import checked_file
from .measurement import Measurement
from .provenance import numerical_environment


def storage_info(root):
    path = Path(root).resolve(strict=True)
    result = dict(root=str(path), device=os.stat(path).st_dev, free_bytes=shutil.disk_usage(path).free,
                  filesystem_type=None, mount_point=None)
    mounts = Path('/proc/mounts')
    if mounts.is_file():
        for line in mounts.read_text().splitlines():
            parts = line.split()
            mount = Path(parts[1].replace('\\040', ' '))
            if (path == mount or path.is_relative_to(mount)) and len(str(mount)) > len(result['mount_point'] or ''):
                result.update(filesystem_type=parts[2], mount_point=str(mount))
    return result


def run(spec, task_id):
    from .dev_worker import allocation, publish_result, publish_outputs
    from .dev_submission import scheduler_identity
    started = time.perf_counter()
    print(f'CMS2JC2-GEN phase=authenticate task={task_id}', flush=True)
    study = campaign.validate_stage(spec)
    task = next(t for t in spec['tasks'] if t['task_id'] == task_id)
    allocated = allocation(study, task)
    allocated['scheduler_evidence'] = scheduler_identity(spec, study, task_id, allocated['job_id'])
    if numerical_environment() != study['numerical_environment']:
        raise ValueError('Frozen numerical environment differs')
    for dependency in task['depends_on']:
        dev.verified_outputs(spec, dependency)
    claim = dev.stage_dir(spec)/'claims'/task_id
    claim.mkdir(parents=True, exist_ok=False)
    dev.write(spec['root'], f'stages/{spec["name"]}/claims/{task_id}/claim.json',
        artifact('DEV_CLAIM', parents={'stage': spec['content_hash']}, task_id=task_id, allocation=allocated), 'DEV_CLAIM')
    auth_seconds = time.perf_counter()-started
    filesystem = storage_info(spec['root'])
    outputs = {}
    with Measurement() as measured:
        if task['action'] == 'jg_report':
            result = report(spec)
        else:
            begin = time.perf_counter()
            bundle = load_json(checked_file(study['bundle']))
            load_seconds = time.perf_counter()-begin
            prefix = f'generated/{spec["name"]}/{task_id}'
            if task['action'] == 'jg_gate':
                # Retain only 64 jets in RAM; compare trace on/off as well as
                # process order and both physical encodings. No larger role opens.
                begin = time.perf_counter()
                with data.stream(spec, study, task_id, limit=64) as source:
                    pairs = list(source)
                input_seconds = time.perf_counter()-begin
                results = []
                for name, workers, trace, compression in (
                        ('serial', 1, False, 'stored'), ('trace', 1, True, 'stored'),
                        ('parallel', 4, False, 'stored'), ('compressed', 4, False, 'deflate')):
                    print(f'CMS2JC2-GEN phase=gate variant={name}', flush=True)
                    result = engine.process(iter(pairs), bundle, root=Path(spec['root']),
                        relative=f'{prefix}/{name}', workers=workers, chunk=8, compression=compression,
                        expected_count=64, trace=trace)
                    results.append(result)
                engine.parity(results)
                result = artifact('GEN_GATE', parents=campaign.parents(spec), jets=64, parity=True,
                    serial=results[0], replay_runs=results[1:], bundle_load_seconds=load_seconds,
                    input_read_seconds=input_seconds)
            else:
                setting = task['params']['setting']
                with data.stream(spec, study, task_id) as source:
                    result = engine.process(source, bundle, root=Path(spec['root']), relative=prefix,
                        workers=setting['cpus'], chunk=setting['chunk'], compression=setting['compression'],
                        expected_count=data.COUNT)
                if result['ordered_identities'] != study['membership']['ordered_identities']:
                    raise ValueError('Generated population differs from frozen TRAIN sample')
                result = artifact('GEN_RUN', parents=campaign.parents(spec), **result,
                    task_id=task_id, setting=setting, repeat=task['params']['repeat'], bundle_load_seconds=load_seconds)
    # Revalidate sources and lineage after computation, before the success receipt.
    print(f'CMS2JC2-GEN phase=final_authenticate task={task_id}', flush=True)
    campaign.validate_stage(spec)
    if numerical_environment() != study['numerical_environment']:
        raise ValueError('Numerical environment changed during generation')
    if task['action'] != 'jg_report':
        fields = {k: v for k, v in result.items() if k not in ('contract', 'schema_version', 'content_hash', 'parents', 'final_test_accessed')}
        fields.update(measurement=measured.report(), allocation=allocated,
                      filesystem=filesystem, initial_authentication_seconds=auth_seconds,
                      total_worker_seconds=time.perf_counter()-started)
        if task['action'] == 'jg_gate':
            fields['projection'] = campaign.projection(fields)
        result = artifact('GEN_GATE' if task['action'] == 'jg_gate' else 'GEN_RUN', parents=campaign.parents(spec), **fields)
        rows = [result['serial'], *result['replay_runs']] if task['action'] == 'jg_gate' else [result]
        for i, row in enumerate(rows):
            for j, block in enumerate(row['blocks']):
                outputs[f'block_{i}_{j}'] = safe_relative(Path(spec['root']), block['relative'])
    kind = result['contract'].removeprefix('CMS2JC2_RESPONSE_').split('/')[0]
    outputs['result'] = publish_result(spec, task_id, result, kind)
    outputs['measurement'] = publish_result(spec, task_id+'_measurement', artifact('DEV_MEASUREMENT',
        allocation=allocated, measurement=measured.report(), numerical_environment=study['numerical_environment']['content_hash']), 'DEV_MEASUREMENT')
    return publish_outputs(spec, task_id, outputs)


def validate_run(spec, task, row):
    validate(row, 'GEN_RUN', parents=campaign.parents(spec))
    if (row['task_id'] != task['task_id'] or row['setting'] != task['params']['setting']
            or row['repeat'] != task['params']['repeat'] or row['jets'] != data.COUNT):
        raise ValueError('Run registration differs')
    times = (row['processing_seconds'], row['total_worker_seconds'])
    if not all(math.isfinite(x) and x > 0 for x in times) or times[1] < times[0]:
        raise ValueError('Invalid measured run times')
    if (len(row['blocks']) != math.ceil(data.COUNT/1000)
            or sum(b['jets'] for b in row['blocks']) != row['jets']
            or sum(b['bytes'] for b in row['blocks']) != row['output_bytes']):
        raise ValueError('Missing or truncated output blocks')
    for block in row['blocks']:
        path = safe_relative(Path(spec['root']), block['relative'])
        if path.stat().st_size != block['bytes'] or sha256_file(path) != block['sha256']:
            raise ValueError('Output block corrupt')


def summarize(rows):
    engine.parity(rows)
    summaries = []
    for setting in campaign.SETTINGS:
        runs = [r for r in rows if r['setting'] == setting]
        if len(runs) != 2 or sorted(r['repeat'] for r in runs) != [0, 1]:
            raise ValueError('Need both repeats of every setting; no survivor selection')
        mean = lambda name: statistics.mean(r[name] for r in runs)
        processing, total = mean('processing_seconds'), mean('total_worker_seconds')
        summaries.append(dict(**setting, processing_seconds=processing, total_worker_seconds=total,
            processing_jets_per_second=data.COUNT/processing, end_to_end_jets_per_second=data.COUNT/total,
            seconds_by_repeat=[r['total_worker_seconds'] for r in runs], bytes_per_jet=mean('output_bytes')/data.COUNT,
            allocated_cpu_hours_per_million=total*setting['cpus']*1_000_000/data.COUNT/3600))
    fastest = min(summaries, key=lambda r: (r['total_worker_seconds'], r['id']))['id']
    efficient = min(summaries, key=lambda r: (r['allocated_cpu_hours_per_million'], r['id']))['id']
    projections = []
    for cpus in (32, 64, 128):
        best = min(summaries, key=lambda r: (r['total_worker_seconds']/(cpus//r['cpus']), r['id']))
        jobs = cpus//best['cpus']
        waves = math.ceil(2_250_000/data.COUNT/jobs)
        projections.append(dict(available_cpus=cpus, setting=best['id'], jobs=jobs,
            processing_hours=waves*best['processing_seconds']/3600,
            end_to_end_hours=waves*best['total_worker_seconds']/3600,
            storage_gib=2_250_000*best['bytes_per_jet']/2**30))
    return dict(settings=summaries, fastest_elapsed=fastest, best_cpu_efficiency=efficient, projections=projections)


def report(spec):
    study = load_json(checked_file(spec['study']))
    rows = []
    for task in spec['tasks']:
        if task['action'] != 'jg_run':
            continue
        row = dev.product(spec, task['task_id'], 'result')
        validate_run(spec, task, row)
        if row['ordered_identities'] != study['membership']['ordered_identities']:
            raise ValueError('Run identity differs')
        rows.append(row)
    return artifact('GEN_REPORT', parents=campaign.parents(spec), run_hashes=[r['content_hash'] for r in rows],
        **summarize(rows), mapping='JOINT', replica=0, training_jets=data.COUNT,
        production_qualified=False, queue_time_included=False, independent_scaling_measured=False)
