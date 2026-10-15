"""One fresh CUDA process per probe, all stress checks before timed passes.

No model/backend change. A new process is the resource-lifetime boundary used
by scientific workers too. Small/instrumented parity is never full admission.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import torch

from hlt_classification.cms2jc2_response.measurement import Measurement
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from . import alljets_fusion_campaign as c, alljets_fusion_preflight as pre
from . import alljets_fusion_worker as w, alljets_fusion_sync as sync, regular_cache

ORDER = ('FUSION_D000_D000_probe', 'FUSION_U050_probe', 'FINAL_DIRECT_D000_probe')
POLICY = dict(version='isolated_cuda_probes_v1', fresh_process_per_probe=True,
              stress_before_any_population_pass=True, order=list(ORDER),
              phases=['parity', 'stress', 'measure'], parity_process_separate=True, automatic_retry=False,
              traced_dispatch=False, cuda_launch_blocking=False,
              model_factory='alljets_fusion_sync.new_model', scientific_recipe_changed=False)
RUNTIME_KEYS = ('job_id', 'environment', 'gpu', 'cublas_workspace_config', 'cudnn_v8_disabled')


def policy(gate=None):
    if gate is not None and gate.get('schema_version') == 4:
        from .alljets_batch128 import POLICY as batch_policy
        return batch_policy
    return POLICY


def sequence(gate=None):
    order = policy(gate)['order']
    return ([(kind, name) for name in order for kind in ('parity', 'stress')]
            + [('measure', name) for name in order])


class SafePhaseLog(pre.PhaseLog):
    """Do not mask the real exception by querying a poisoned CUDA context."""
    def __init__(self, output, sample):
        self.failed, self.last = False, {}
        def safe_sample():
            if not self.failed:
                self.last = sample()
            return dict(self.last)
        super().__init__(output, safe_sample)

    @contextmanager
    def phase(self, name):
        with super().phase(name) as row:
            try:
                yield row
            except BaseException:
                self.failed = True
                raise


def _node(name, gate=None):
    if name not in policy(gate)['order']:
        raise ValueError('Unregistered isolated probe')
    return next(n for n in pre.probe_nodes(gate) if n['node_id'] == name)


def _scope(gate):
    if (gate.get('schema_version') not in (3, 4) or gate.get('preflight_execution') != policy(gate)
            or gate.get('pair_storage') != sync.PAIR_STORAGE):
        raise PermissionError('Isolated preflight requires its exact versioned policy')


def _inputs(gate, node):
    parent = load_json(gate['parent_import']['spec']['path'])
    auth = regular_cache.metadata(parent['admission']['prepared_root'])
    p = auth[1]
    if p['content_hash'] != gate['prepared_sha256']:
        raise ValueError('Isolated prepared cache changed')
    bound = sum(2*p['summaries'][role][co]['resident_bytes'] for role in ('train', 'validation')
                for co in {node['coordinate'], node['context_coordinate']}-{None}) + 256*2**20
    if bound > .30*c.MEMORY_MB*1024**2:
        raise MemoryError('Paired cached inputs exceed reserved RAM envelope')
    train = w.cache(gate, p, node, 'train', authenticated=auth)
    val = w.cache(gate, p, node, 'validation', authenticated=auth)
    return parent, auth, p, train, val


def run_child(gate, kind, name):
    _scope(gate)
    if (kind, name) not in sequence(gate):
        raise ValueError('Unregistered probe phase')
    job = w.execution(gate)
    c.validate_gate(gate)  # Original pinned sources authenticated, read-only.
    node = _node(name, gate)
    batch = gate['training_recipe']['batch_size']
    batch128 = gate.get('schema_version') == 4
    output = Path(gate['root'])/'preflight/children'/(kind+'_'+name)
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(job_id=job, environment=pre.installed_environment(), gpu=pre.gpu_identity(),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset'))
    original = load_json(gate['recovery']['original_runtime']['path'])
    if any(runtime[k] != original[k] for k in ('environment', 'gpu', 'cudnn_v8_disabled')):
        raise PermissionError('Isolated recovery software/GPU/backend differs')
    write_immutable_json(output/'runtime.json', runtime)
    log = SafePhaseLog(output, lambda: pre.memory_snapshot(torch))
    parities, fields = {}, {}
    try:
        with Measurement() as measured:
            torch.cuda.reset_peak_memory_stats()
            with log.phase(name+'_load_cached_inputs') as load_phase:
                parent, auth, p, train, val = _inputs(gate, node)
            maxima = pre.support(p)
            paired = isinstance(train, pre.PairedCache)
            if kind == 'parity':
                if name == ORDER[0]:
                    native = train.primary
                    with log.phase('installed_and_native_mask_parity'):
                        parities['installed'] = pre.installed_parity(native, device='cuda')
                        parities['native_mask'] = pre.native_mask_parity(native.batch(np.arange(4)), device='cuda')
                        pre.clear()
                if paired:
                    for bf16 in (False, True):
                        key = name+('_bf16' if bf16 else '_fp32')
                        with log.phase(key+'_offload_parity'):
                            parities[key] = sync.native_offload_parity(train.primary.batch(np.arange(4)),
                                train.context.batch(np.arange(4)), device='cuda', bf16=bf16)
                            pre.clear()
                fields['parities'] = parities
            elif kind == 'stress':
                with log.phase(name+f'_train_stress{batch}_steps3'):
                    model = sync.new_model(node)
                    if batch128:
                        from .alljets_batch128 import stress
                        stats = stress(model, train)
                    else:
                        stats = pre.stress(model, train)
                    if paired:
                        pre.validate_offload_stats(stats, calls=3)
                    del model
                    pre.clear()
                    pre.check_headroom(pre.memory_snapshot(torch), memory_mb=c.MEMORY_MB,
                        future_cache_bytes=0, gpu_bytes=runtime['gpu']['total_memory_bytes'])
                with log.phase(name+f'_validation_inference_stress{batch}_steps3'):
                    model = sync.new_model(node)
                    pre.validation_stress(model, val, **(dict(batch_size=128) if batch128 else {}))
                    del model
                    pre.clear()
                fields.update(stress_offload=stats, stress_batch=batch, stress_steps=3,
                    validation_stress_batch=batch, validation_stress_steps=3, validation_stress_inference_only=True)
            else:
                targets = (None if node['teacher'] is None else
                           np.full((len(train), 11), 1/11, np.float32) if batch128 else
                           w.teacher(gate, 'U000', train.identities))
                with log.phase(name+'_full_population_technical_pass'):
                    model = sync.new_model(node)
                    training, state = pre.train_kernel(model, train, val, node=node, device='cuda',
                        acceptance_passes=1, teacher_probabilities=targets,
                        teacher_identities=None if targets is None else train.identities,
                        **(dict(training_recipe=gate['training_recipe']) if batch128 else {}))
                with log.phase(name+'_checkpoint_train_bank_readback') as bank_phase:
                    pre.save_state(output/(name+'.pt'), state)
                    restored = torch.load(output/(name+'.pt'), map_location='cpu', weights_only=True)
                    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
                        raise ValueError('Technical selected checkpoint roundtrip differs')
                    model.load_state_dict(restored, strict=True)
                    probabilities = pre.predict(model, train, device='cuda', temperature=2., batch_size=batch)
                    args = dict(foundation_sha256=p['content_hash'], teacher_report_sha256=training['content_hash'],
                                teacher_node=name, role='train')
                    pre.publish_bank(output/(name+'_bank'), identities=train.identities, probabilities=probabilities, **args)
                    if not np.array_equal(probabilities, pre.load_bank(output/(name+'_bank'), expected_identities=train.identities, **args)):
                        raise ValueError('Technical TRAIN bank roundtrip differs')
                fields.update(training=training, cache_seconds=load_phase['seconds'], inference_seconds=bank_phase['seconds'])
            snap = pre.memory_snapshot(torch)
            pre.check_headroom(snap, memory_mb=c.MEMORY_MB, future_cache_bytes=0,
                               gpu_bytes=runtime['gpu']['total_memory_bytes'])
            if regular_cache.metadata(parent['admission']['prepared_root'])[:2] != auth[:2]:
                raise ValueError('Prepared cache changed during isolated probe')
            w.execution(gate, runtime)
        peak = max(snap['peak_rss_bytes'], measured.report()['sampled_peak_tree_rss_bytes'])
        result = c.bound_artifact(gate, 'PROBE', parents=dict(gate=gate['content_hash'], prepared=p['content_hash']),
            phase=kind, node=node, process_id=os.getpid(), runtime=runtime, source_snapshot=gate['source_snapshot'],
            policy=policy(gate), pair_storage=sync.PAIR_STORAGE, counts=dict(train=len(train), validation=len(val)),
            support_maxima=maxima, peak_rss_bytes=peak, gpu_peak_bytes=snap['gpu_peak_bytes'],
            completed=True, science_admissible=False, **fields,
            outputs=[file_ref(path, root=output) for path in sorted(output.rglob('*')) if path.is_file()])
        write_immutable_json(output/'result.json', result)
        return result
    except BaseException as exc:
        # Do not clear/synchronize/sample CUDA after failure. The process exits.
        write_immutable_json(output/'failure.json', dict(phase=kind, node=name, completed=False,
            science_admissible=False, error=repr(exc), process_id=os.getpid()))
        raise


def _launch(gate, kind, name, logfile):
    args = [sys.executable, '-u', '-s', str(Path(gate['project_dir'])/'scripts/luka_fullsim_alljets_fusion.py'),
            'preflight-child', '--spec', str(Path(gate['root'])/'gate_spec.json'), '--kind', kind, '--node', name]
    env = dict(os.environ, PYTHONPATH=str(Path(gate['project_dir'])/'src'), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_LAUNCH_BLOCKING='0')
    with logfile.open('x', encoding='utf-8') as stream:
        with subprocess.Popen(args, cwd=gate['project_dir'], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = child.wait()
            return code, child.pid


def check_child(row, gate, kind, name, runtime=None):
    c.validate_bound(row, gate, 'PROBE', dict(gate=gate['content_hash'], prepared=gate['prepared_sha256']))
    expected = dict(phase=kind, node=_node(name, gate), source_snapshot=gate['source_snapshot'], policy=policy(gate),
        pair_storage=sync.PAIR_STORAGE, counts=c.COUNTS, completed=True, science_admissible=False)
    if (any(row.get(k) != v for k,v in expected.items()) or type(row.get('process_id')) is not int
            or row['process_id'] <= 0 or (runtime is not None and row['runtime'] != runtime)):
        raise ValueError('Isolated probe source/runtime/coverage differs')
    directory = Path(gate['root'])/'preflight/children'/(kind+'_'+name)
    paths = [ref['path'] for ref in row['outputs']]
    required = {'runtime.json'}
    if kind == 'measure':
        required |= {name+'.pt', name+'_bank/manifest.json'}
    if len(paths) != len(set(paths)) or not required <= set(paths):
        raise ValueError('Missing/duplicate isolated outputs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=directory)
    if load_json(directory/'runtime.json') != row['runtime']:
        raise ValueError('Isolated runtime file differs')
    return row


def validate_execution(report, gate):
    _scope(gate)
    if report.get('preflight_execution') != policy(gate):
        raise ValueError('Wrong process isolation policy')
    children = report.get('isolated_children', [])
    if [(v['phase'], v['node']) for v in children] != sequence(gate):
        raise ValueError('All isolated stresses must precede all timed passes')
    seen, stress, measurements, parities, peaks, gpu_peaks = set(), {}, {}, {}, [], []
    output = Path(gate['root'])/'preflight'
    runtime = {k:report[k] for k in RUNTIME_KEYS}
    for child in children:
        kind, name = child['phase'], child['node']
        prefix = 'children/'+kind+'_'+name
        if (child['result']['path'] != prefix+'/result.json' or child['log']['path'] != prefix+'.out'
                or child['result'] not in report['outputs'] or child['log'] not in report['outputs']):
            raise ValueError('Unbound isolated process result/log')
        for ref in (child['result'], child['log']):
            validate_file_ref(ref, root=output)
        row = check_child(load_json(output/child['result']['path']), gate, kind, name, runtime)
        if child.get('pid') != row['process_id'] or row['process_id'] in seen or child.get('exit_code') != 0:
            raise ValueError('Fresh successful isolated processes required')
        seen.add(row['process_id'])
        if row['support_maxima'] != report['support_maxima']:
            raise ValueError('Isolated support population differs')
        peaks.append(row['peak_rss_bytes'])
        gpu_peaks.append(row['gpu_peak_bytes'])
        if kind == 'parity':
            parities.update(row['parities'])
        elif kind == 'stress':
            stress[name] = {k:row[k] for k in ('stress_offload', 'stress_batch', 'stress_steps',
                'validation_stress_batch', 'validation_stress_steps', 'validation_stress_inference_only')}
        else:
            measurements[name] = dict(node=row['node'], counts=row['counts'],
                training=row['training'], cache_seconds=row['cache_seconds'], inference_seconds=row['inference_seconds'],
                **stress[name])
    if (report['parities'] != parities or report['measurements'] != [measurements[n['node_id']] for n in pre.probe_nodes(gate)]
            or report['peak_rss_bytes'] < max(peaks) or report['gpu_peak_bytes'] != max(gpu_peaks)):
        raise ValueError('Isolated measurement aggregation differs')


def run(gate):
    _scope(gate)
    job = w.execution(gate)  # Coordinator never creates a CUDA context.
    parent = c.validate_gate(gate)
    p = regular_cache.metadata(parent['admission']['prepared_root'])[1]
    maxima = pre.support(p)
    output = Path(gate['root'])/'preflight'
    output.mkdir(exist_ok=False)
    (output/'children').mkdir()
    rows, children, runtime = {}, [], None
    with Measurement() as measured:
        for kind, name in sequence(gate):
            prefix = 'children/'+kind+'_'+name
            print(f'LUKA isolated {kind}: {name}; fresh process, no automatic retry', flush=True)
            code, pid = _launch(gate, kind, name, output/(prefix+'.out'))
            if code:
                write_immutable_json(output/'failure.json', dict(phase=kind, node=name, exit_code=code,
                    process_id=pid, science_admissible=False, log=prefix+'.out'))
                raise RuntimeError(f'Isolated {kind}/{name} failed (exit {code}); stopped before later probes. See {prefix}.out')
            row = check_child(load_json(output/prefix/'result.json'), gate, kind, name, runtime)
            if row['process_id'] != pid or row['runtime']['job_id'] != job:
                raise ValueError('Probe PID/job differs from launched child')
            if runtime is None:
                runtime = row['runtime']
                write_immutable_json(output/'runtime.json', runtime)
            rows[kind, name] = row
            children.append(dict(phase=kind, node=name, pid=pid, exit_code=code,
                result=file_ref(output/prefix/'result.json', root=output), log=file_ref(output/(prefix+'.out'), root=output)))
    parities, measurements = {}, []
    for node in pre.probe_nodes(gate):
        name = node['node_id']
        stress, timing = rows['stress', name], rows['measure', name]
        parities.update(rows['parity', name]['parities'])
        measurements.append(dict(node=node, counts=c.COUNTS, training=timing['training'],
            cache_seconds=timing['cache_seconds'], inference_seconds=timing['inference_seconds'],
            **{k:stress[k] for k in ('stress_offload', 'stress_batch', 'stress_steps',
                'validation_stress_batch', 'validation_stress_steps', 'validation_stress_inference_only')}))
    peak = max(measured.report()['sampled_peak_tree_rss_bytes'], *(r['peak_rss_bytes'] for r in rows.values()))
    gpu_peak = max(r['gpu_peak_bytes'] for r in rows.values())
    train_minutes, reduce_minutes = pre.projections(measurements)
    c.validate_gate(gate)
    result = c.bound_artifact(gate, 'PREFLIGHT', parents=dict(gate=gate['content_hash'], prepared=p['content_hash']),
        **runtime, cpus=6, memory_mb=c.MEMORY_MB, counts=c.COUNTS, site=gate['execution_site'],
        source_snapshot=gate['source_snapshot'], preparation_source_snapshot=p['source_snapshot'],
        recipe=gate['training_recipe'], parities=parities, support_maxima=maxima, measurements=measurements,
        technical_teacher=pre.technical_teacher(gate), peak_rss_bytes=peak, gpu_peak_bytes=gpu_peak,
        projected_train_minutes=train_minutes, projected_reduce_minutes=reduce_minutes,
        resource_envelope_ok=bool(peak < .85*c.MEMORY_MB*1024**2 and gpu_peak < .9*runtime['gpu']['total_memory_bytes']),
        fits_debug_24h=max(train_minutes,reduce_minutes) <= 1440, checkpoint_probability_roundtrip=True,
        scientific_fit=False, scientific_metrics_for_selection=False, technical_checks_passed=True,
        pair_storage=sync.PAIR_STORAGE, preflight_execution=policy(gate), isolated_children=children,
        outputs=[file_ref(path, root=output) for path in sorted(output.rglob('*')) if path.is_file()])
    write_immutable_json(output/'measurement.json', result)
    print(f'LUKA fusion measured requests: train={train_minutes}min reduce={reduce_minutes}min; debug ceiling=1440min', flush=True)
    pre.validate_evidence(result, gate, p)
    write_immutable_json(output/'preflight.json', result)
    return result
