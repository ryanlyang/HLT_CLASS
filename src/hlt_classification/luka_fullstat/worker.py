"""Gate tasks and authenticated fresh CE/KD fits; no test evaluator exists."""
from pathlib import Path
import os
import shutil

import numpy as np

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref, safe
from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays, write_immutable_json,
)
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from hlt_classification.jetclass2_delphes.reporting import recovery, evaluate_probabilities
from .contracts import ORDINARY, artifact, validate
from . import campaign as c, cache, data


def gate_task(gate, task, file_index=None):
    c.validate_gate(gate)
    root = Path(gate['root'])
    if task == 'foundation':
        result = data.build(gate['container'], root/'foundation', gate_hash=gate['content_hash'])
        c.validate_foundation(gate, result)
        bound = cache.storage_bound(result)
        print(f'FullStat cache required_free_GiB={bound/2**30:.2f}; declared quota is checked separately', flush=True)
        if bound > gate['declared_available_cache_bytes'] or bound > shutil.disk_usage(gate['cache_root']).free:
            raise OSError('Insufficient declared quota/free cache storage; no data deleted')
        write_immutable_json(root/'storage.json', artifact('STORAGE', parents=dict(foundation=result['content_hash']),
            required_free_bytes=bound, declared_available_cache_bytes=gate['declared_available_cache_bytes']))
    elif task == 'prepare':
        foundation, _ = data.load(root/'foundation', expected_gate=gate['content_hash'])
        c.validate_foundation(gate, foundation)
        storage = load_json(root/'storage.json')
        validate(storage, 'STORAGE', dict(foundation=foundation['content_hash']))
        if storage['required_free_bytes'] != cache.storage_bound(foundation):
            raise ValueError('Storage reservation differs')
        result = cache.prepare_file(gate['container'], root/'foundation', gate['cache_root'], file_index,
                                    workers=gate['preparation_workers'])
    elif task == 'assemble':
        c.validate_foundation(gate, data.load(root/'foundation', expected_gate=gate['content_hash'])[0])
        result = cache.assemble(root/'foundation', gate['cache_root'], gate_hash=gate['content_hash'])
    elif task == 'preflight':
        from .preflight import run
        result = run(gate)
    else:
        raise ValueError('Unknown gate task; no test operation')
    c.validate_gate(gate)
    return result


def validate_training(spec, node, report):
    kernel_validate(report, 'KERNEL_TRAINING_REPORT')
    if (report['node'] != node or report['recipe_sha256'] != spec['training_recipe']['content_hash']
            or report['foundation_sha256'] != spec['prepared_sha256']
            or report['scientific_fit'] is not True or report['acceptance_only'] is not False
            or report['final_test_accessed'] is not False or report['selected_weights_restored'] is not True
            or report['passes'] != 20 or len(report['validation_history']) != 20
            or report['validation']['rows'] != spec['counts']['validation']
            or not 1 <= report['selected_pass'] <= 20):
        raise ValueError('Scientific twenty-epoch fit binding differs')
    for number, row in enumerate(report['validation_history'], 1):
        from .recipe import learning_rate
        if row['pass_number'] != number or row['learning_rate'] != learning_rate(number):
            raise ValueError('Scientific epoch/schedule history differs')


def completed(spec, task_id, seen=None):
    seen = {} if seen is None else seen
    if task_id in seen:
        return seen[task_id]
    task = next((t for t in c.tasks() if t['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unregistered science task')
    root = Path(spec['root'])
    path = safe(root, task_id+'/receipt.json')
    if not path.is_file():
        return None
    row = load_json(path)
    validate(row, 'TASK', dict(spec=spec['content_hash']))
    if (row['task_id'] != task_id or row['source_commit'] != spec['source_commit']
            or row['final_test_accessed'] is not False):
        raise ValueError('Task identity/source differs')
    paths = [ref['path'] for ref in row['outputs']]
    if len(paths) != len(set(paths)) or any(not p.startswith(task_id+'/') for p in paths):
        raise ValueError('Task payload scope differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    deps = {}
    for name in task['dependencies']:
        parent = completed(spec, name, seen)
        if parent is None:
            raise ValueError('Committed child has uncommitted dependency')
        deps[name] = parent['content_hash']
    if row['dependencies'] != deps:
        raise ValueError('Teacher/dependency lineage differs')
    if task['kind'] == 'train':
        required = {task_id+'/'+s for s in ('training_report.json', 'selected.pt', 'validation_predictions.npz')}
        if not required <= set(paths):
            raise ValueError('Missing scientific fit output')
        report = load_json(root/task_id/'training_report.json')
        node = next(n for n in c.nodes() if n['node_id'] == task['node_id'])
        validate_training(spec, node, report)
        if row['result'] != dict(report_sha256=report['content_hash']):
            raise ValueError('Fit receipt/report binding differs')
    elif task['kind'] == 'reduce':
        report = load_json(root/('train_'+task['node_id'])/'training_report.json')
        bank = load_json(root/task_id/'train'/'manifest.json')
        kernel_validate(bank, 'PROBABILITY_BANK')
        expected = dict(foundation_sha256=spec['prepared_sha256'], teacher_report_sha256=report['content_hash'],
            teacher_node=task['node_id'], role='train', temperature=2., class_count=11, rows=spec['counts']['train'],
            dtype='float32', final_test_accessed=False)
        if any(bank.get(k) != v for k,v in expected.items()) or task_id+'/train/manifest.json' not in paths:
            raise ValueError('Scientific TRAIN bank lineage differs')
        for shard in bank['shards']:
            if dict(path=task_id+'/train/'+shard['path'], bytes=shard['bytes'], sha256=shard['sha256']) not in row['outputs']:
                raise ValueError('Unbound TRAIN probability shard')
        if row['result'] != dict(bank_sha256=bank['content_hash']):
            raise ValueError('Bank receipt differs')
    elif task['kind'] == 'aggregate':
        result = load_json(root/task_id/'results.json')
        validate(result, 'RESULTS', dict(spec=spec['content_hash']))
        if task_id+'/results.json' not in paths or result != row['result']:
            raise ValueError('Result receipt differs')
    elif row['result'] != dict(fits=9, reducers=5, final_test_accessed=False):
        raise ValueError('Completion semantics differ')
    seen[task_id] = row
    return row


def execution(spec, evidence):
    from hlt_classification.luka_fullsim.preflight_v2 import require_backend
    from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
    from hlt_classification.jetclass2_delphes.model import installed_environment
    require_backend()
    job, cpus, memory = allocation(spec['execution_site'])
    if ((cpus,memory) != (spec['cpus'],spec['memory_mb'])
            or gpu_identity() != evidence['gpu'] or installed_environment() != evidence['environment']):
        raise PermissionError('Science runtime differs from accepted Oscar environment')
    runtime = load_json(Path(spec['preflight']['path']).parent/'runtime.json')
    if os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset') != runtime['cudnn_v8_disabled']:
        raise PermissionError('cuDNN runtime differs from preflight')
    return job


def fit_or_reduce(spec, authenticated, task, directory, *, device='cuda'):
    """Testable core. Production run() authenticates all parents/allocation first."""
    import torch
    from hlt_classification.context_fusion.inputs import new_model
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
    from hlt_classification.jetclass2_delphes.banks import load_bank, publish_bank
    node = next(n for n in c.nodes() if n['node_id'] == task['node_id'])
    budget = int(spec['memory_mb']*1024**2*.35)
    train = cache.load(spec['cache_root'], authenticated, role='train', coordinate=node['coordinate'], max_ram_bytes=budget)
    model, batch = new_model(node), spec['training_recipe']['batch_size']
    root = Path(spec['root'])
    if task['kind'] == 'train':
        val = cache.load(spec['cache_root'], authenticated, role='validation', coordinate=node['coordinate'], max_ram_bytes=budget)
        kwargs = {}
        if node['teacher'] is not None:
            teacher = node['teacher']
            if completed(spec, 'reduce_'+teacher) is None:
                raise ValueError('Teacher bank is not committed')
            report = load_json(root/('train_'+teacher)/'training_report.json')
            probabilities = load_bank(root/('reduce_'+teacher)/'train', foundation_sha256=spec['prepared_sha256'],
                teacher_report_sha256=report['content_hash'], teacher_node=teacher, role='train', expected_identities=train.identities)
            kwargs = dict(teacher_probabilities=probabilities, teacher_identities=train.identities)
        report, state = train_kernel(model, train, val, node=node, device=device,
                                    training_recipe=spec['training_recipe'], **kwargs)
        validate_training(spec, node, report)
        save_state(directory/'selected.pt', state)
        restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
        if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
            raise ValueError('Selected state readback differs')
        model.load_state_dict(restored, strict=True)
        predictions = predict(model, val, device=device, batch_size=batch)
        if evaluate_probabilities(val.labels, predictions) != report['validation']:
            raise ValueError('Selected checkpoint validation replay differs')
        atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(dict(
            probabilities=predictions, labels=val.labels, identities=val.identities)))
        write_immutable_json(directory/'training_report.json', report)
        return dict(report_sha256=report['content_hash'])
    report = load_json(root/('train_'+node['node_id'])/'training_report.json')
    model.load_state_dict(torch.load(root/('train_'+node['node_id'])/'selected.pt', map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    probabilities = predict(model, train, device=device, temperature=2., batch_size=batch)
    args = dict(foundation_sha256=spec['prepared_sha256'], teacher_report_sha256=report['content_hash'],
                teacher_node=node['node_id'], role='train')
    bank = publish_bank(directory/'train', identities=train.identities, probabilities=probabilities, **args)
    if not np.array_equal(load_bank(directory/'train', expected_identities=train.identities, **args), probabilities):
        raise ValueError('Scientific TRAIN bank readback differs')
    return dict(bank_sha256=bank['content_hash'])


def result_rows(spec):
    rows, seen, metrics = [], {}, {}
    for node in c.nodes():
        name = node['node_id']
        done = completed(spec, 'train_'+name, seen)
        report = None if done is None else load_json(Path(spec['root'])/('train_'+name)/'training_report.json')
        metrics[name] = None if report is None else report['validation']
        rows.append(dict(node_id=name, deployable=node['deployable'], validation=metrics[name],
                         selected_epoch=None if report is None else report['selected_pass']))
    for row in rows:
        row['recovery'] = (None if row['validation'] is None or metrics.get('M0HLT') is None or metrics.get('OFFLINE') is None
                          else recovery(row['validation'], metrics['M0HLT'], metrics['OFFLINE']))
    return rows


def run(spec, task_id):
    gate, authenticated, evidence = c.validate_spec(spec)
    task = next((t for t in c.tasks() if t['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unknown science task; no final-test access')
    old = completed(spec, task_id)
    if old is not None:
        return old
    deps = {}
    for parent in task['dependencies']:
        row = completed(spec, parent)
        if row is None:
            raise ValueError('Scientific dependency is not committed')
        deps[parent] = row['content_hash']
    job = os.environ.get('SLURM_JOB_ID')
    if task['kind'] in ('train', 'reduce'):
        job = execution(spec, evidence)
    root, directory = Path(spec['root']), Path(spec['root'])/task_id
    directory.mkdir(exist_ok=False)
    if task['kind'] in ('train', 'reduce'):
        result = fit_or_reduce(spec, authenticated, task, directory)
        execution(spec, evidence)
    elif task['kind'] == 'aggregate':
        rows = result_rows(spec)
        if any(r['validation'] is None for r in rows):
            raise ValueError('All nine models required regardless of performance')
        result = artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
            counts=spec['counts'], final_test_accessed=False, event_independence_verified=False,
            old_trial_overlap_verified=False, single_seed=True, unequal_total_training_compute=True)
        write_immutable_json(directory/'results.json', result)
    else:
        result = dict(fits=9, reducers=5, final_test_accessed=False)
    c.validate_spec(spec)
    if any(completed(spec, p)['content_hash'] != digest for p,digest in deps.items()):
        raise ValueError('Dependencies changed during execution')
    receipt = artifact('TASK', parents=dict(spec=spec['content_hash']), task_id=task_id,
        source_commit=spec['source_commit'], dependencies=deps, result=result, job_id=job, final_test_accessed=False,
        outputs=[file_ref(p, root=root) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'receipt.json', receipt)
    return completed(spec, task_id)


def print_results(spec):
    c.validate_spec(spec)
    print('FullStat development validation: 3M train / 500k validation; 1.5M test remains sealed.')
    print('Single seed; event independence and old-trial overlap unverified. Recovery: M0HLT=0%, OFFLINE=100%.')
    print(f'{"node":20} {"pick/20":>10} {"accuracy":>10} {"AUC":>10} {"acc rec%":>10} {"AUC rec%":>10}')
    for row in result_rows(spec):
        if row['validation'] is None:
            print(f'{row["node_id"]:20} PENDING')
            continue
        m, r = row['validation'], row['recovery'] or {}
        def fmt(v, digits=6):
            return 'n/a' if v is None else f'{v:.{digits}f}'
        print(f'{row["node_id"]:20} {str(row["selected_epoch"])+"/20":>10} {m["accuracy"]:10.6f} '
              f'{m["macro_ovr_auc"]:10.6f} {fmt(r.get("accuracy"),1):>10} {fmt(r.get("macro_ovr_auc"),1):>10}')
        print('  QCD rejection @50%: '+', '.join(k+'='+('censored' if v['zero_fpr_censored'] else
              fmt(v['rejection_at_50pct'],1)) for k,v in m['per_class'].items()))
