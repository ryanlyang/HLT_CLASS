"""Receipt-bound CE/KD fits and TRAIN banks; no final-test capability."""
from pathlib import Path
import os

import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import safe, file_ref, validate_file_ref
from hlt_classification.context_fusion.inputs import new_model
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.reporting import recovery
from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
from . import science_campaign as c
from .science_cache import cache
from .preflight_v2 import require_backend, PhaseLog, memory_snapshot


def validate_training(spec, node, report):
    kernel_validate(report, 'KERNEL_TRAINING_REPORT')
    if (report['node'] != node or report['recipe_sha256'] != spec['training_recipe']['content_hash']
            or report['foundation_sha256'] != spec['prepared_sha256']
            or report['scientific_fit'] is not True or report['acceptance_only'] is not False
            or report['final_test_accessed'] is not False or report['selected_weights_restored'] is not True
            or not spec['training_recipe']['minimum_passes'] <= report['passes'] <= spec['training_recipe']['maximum_passes']
            or len(report['validation_history']) != report['passes']
            or not 1 <= report['selected_pass'] <= report['passes']):
        raise ValueError('Scientific fit report differs; probes cannot be reused')


def completed(spec, task_id, seen=None):
    seen = {} if seen is None else seen
    if task_id in seen:
        return seen[task_id]
    task = next((r for r in c.tasks() if r['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unregistered science task')
    root = Path(spec['root'])
    path = safe(root, task_id+'/receipt.json')
    if not path.is_file():
        return None
    row = load_json(path)
    c.validate(row, 'TASK', dict(spec=spec['content_hash'], preflight=spec['preflight_sha256']))
    if (row['task_id'] != task_id or row['source_commit'] != spec['source_commit']
            or row['prepared_sha256'] != spec['prepared_sha256']):
        raise ValueError('Task identity/source/population differs')
    paths = [ref['path'] for ref in row['outputs']]
    if len(paths) != len(set(paths)) or any(not name.startswith(task_id+'/') for name in paths):
        raise ValueError('Task output scope differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    deps = {}
    for name in task['dependencies']:
        parent = completed(spec, name, seen)
        if parent is None:
            raise ValueError('Published child lacks its committed parent')
        deps[name] = parent['content_hash']
    if row['dependencies'] != deps:
        raise ValueError('Task dependency hashes differ')
    if task['kind'] == 'train':
        result = row['result']
        expected_paths = dict(training_report=task_id+'/training_report.json', checkpoint=task_id+'/selected.pt')
        if any(result.get(k) != v or v not in paths for k, v in expected_paths.items()):
            raise ValueError('Missing scientific report/checkpoint binding')
        report = load_json(safe(root, result['training_report']))
        node = next(n for n in c.nodes() if n['node_id'] == task['node_id'])
        validate_training(spec, node, report)
        if report['content_hash'] != result['training_report_sha256']:
            raise ValueError('Scientific report hash differs')
    elif task['kind'] == 'reduce':
        result = row['result']
        fitted = seen['train_'+task['node_id']]['result']
        if (result['teacher_report_sha256'] != fitted['training_report_sha256']
                or result['train_bank'] != task_id+'/train'):
            raise ValueError('Reducer teacher/path differs')
        manifest_name = result['train_bank']+'/manifest.json'
        if manifest_name not in paths:
            raise ValueError('Missing probability bank binding')
        bank = load_json(safe(root, manifest_name))
        kernel_validate(bank, 'PROBABILITY_BANK')
        expected = dict(foundation_sha256=spec['prepared_sha256'], teacher_report_sha256=fitted['training_report_sha256'],
            teacher_node=task['node_id'], role='train', temperature=2., class_count=11, dtype='float32',
            rows=spec['counts']['train'], final_test_accessed=False)
        if any(bank.get(k) != v for k, v in expected.items()):
            raise ValueError('Probability manifest lineage differs')
        for shard in bank['shards']:
            ref = dict(path=result['train_bank']+'/'+shard['path'], bytes=shard['bytes'], sha256=shard['sha256'])
            if ref not in row['outputs']:
                raise ValueError('Missing bound probability shard')
    else:
        if task['kind'] == 'aggregate':
            if task_id+'/results.json' not in paths or load_json(safe(root, task_id+'/results.json')) != row['result']:
                raise ValueError('Aggregate results binding differs')
            c.validate(row['result'], 'RESULTS', dict(spec=spec['content_hash']))
            if row['result']['rows'] != result_rows(spec):
                raise ValueError('Aggregate rows differ from committed fits')
        elif row['result'] != dict(fits=12, reducers=7, scientific_performance_never_controls_completion=True):
            raise ValueError('Campaign completion counts differ')
    seen[task_id] = row
    return row


def teacher(spec, name, identities):
    receipt = completed(spec, 'reduce_'+name)
    if receipt is None:
        raise ValueError('Scientific teacher is not committed')
    result = receipt['result']
    return load_bank(safe(Path(spec['root']), result['train_bank']), foundation_sha256=spec['prepared_sha256'],
        teacher_report_sha256=result['teacher_report_sha256'], teacher_node=name, role='train', expected_identities=identities)


def execution(spec, evidence):
    require_backend()
    job, cpus, memory = allocation(spec['execution_site'])
    if ((cpus, memory) != (spec['cpus'], spec['memory_mb']) or gpu_identity() != evidence['gpu']
            or installed_environment() != evidence['environment']):
        raise PermissionError('Science allocation/GPU/environment differs from accepted preflight')
    runtime = load_json(Path(spec['admission']['preflight']['path']).parent/'runtime.json')
    if os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset') != runtime['cudnn_v8_disabled']:
        raise PermissionError('cuDNN execution setting differs from preflight')
    return job


def result_rows(spec):
    rows, seen, metrics = [], {}, {}
    for node in c.nodes():
        row = completed(spec, 'train_'+node['node_id'], seen)
        report = None if row is None else load_json(safe(Path(spec['root']), row['result']['training_report']))
        value = None if report is None else report['validation']
        metrics[node['node_id']] = value
        refs_ready = metrics.get('M0HLT') is not None and metrics.get('OFFLINE') is not None
        rows.append(dict(node_id=node['node_id'], deployable=node['deployable'], branch=node['branch'],
            state='UNFINISHED' if report is None else 'COMPLETE', validation=value,
            selected_pass=None if report is None else report['selected_pass'], passes=None if report is None else report['passes'],
            recovery_to_pure_offline=None if value is None or not refs_ready else recovery(value, metrics['M0HLT'], metrics['OFFLINE'])))
    # M0HLT precedes OFFLINE: recompute all recoveries after collecting controls.
    if metrics.get('M0HLT') is not None and metrics.get('OFFLINE') is not None:
        for row in rows:
            if row['validation'] is not None:
                row['recovery_to_pure_offline'] = recovery(row['validation'], metrics['M0HLT'], metrics['OFFLINE'])
    return rows


def run(spec, task_id):
    p, evidence = c.validate_spec(spec)
    task = next((r for r in c.tasks() if r['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unregistered task; no final-test operation')
    old = completed(spec, task_id)
    if old is not None:
        return old
    dependencies = {}
    for name in task['dependencies']:
        parent = completed(spec, name)
        if parent is None:
            raise ValueError('Uncommitted dependency: '+name)
        dependencies[name] = parent['content_hash']
    root, job = Path(spec['root']), os.environ.get('SLURM_JOB_ID')
    if task['kind'] in ('train', 'reduce'):
        job = execution(spec, evidence)
    directory = safe(root, task_id)
    directory.mkdir(exist_ok=False)  # Preserve failed/ambiguous work; no silent resume.
    if task['kind'] in ('train', 'reduce'):
        node = next(n for n in c.nodes() if n['node_id'] == task['node_id'])
        log = PhaseLog(directory, lambda: memory_snapshot(torch))
        with log.phase('prepare_train_views'):
            train = cache(spec, p, node, 'train')
        model = new_model(node)  # Always fresh; no technical model imports.
        if task['kind'] == 'train':
            with log.phase('prepare_validation_views'):
                val = cache(spec, p, node, 'validation')
            kwargs = {} if node['teacher'] is None else dict(teacher_probabilities=teacher(spec, node['teacher'], train.identities),
                                                          teacher_identities=train.identities)
            with log.phase('scientific_fit'):
                report, state = train_kernel(model, train, val, node=node, device='cuda', **kwargs)
                validate_training(spec, node, report)
            with log.phase('checkpoint_readback'):
                checkpoint = directory/'selected.pt'
                save_state(checkpoint, state)
                restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
                    raise ValueError('Scientific selected weights readback differs')
                write_immutable_json(directory/'training_report.json', report)
            result = dict(checkpoint=task_id+'/selected.pt', training_report=task_id+'/training_report.json',
                          training_report_sha256=report['content_hash'])
        else:
            fitted = completed(spec, 'train_'+node['node_id'])['result']
            with log.phase('selected_checkpoint_train_bank'):
                model.load_state_dict(torch.load(safe(root, fitted['checkpoint']), map_location='cpu', weights_only=True), strict=True)
                model.to('cuda').eval()
                probabilities = predict(model, train, device='cuda', temperature=2.)
                args = dict(foundation_sha256=spec['prepared_sha256'], teacher_report_sha256=fitted['training_report_sha256'],
                            teacher_node=node['node_id'], role='train')
                publish_bank(directory/'train', identities=train.identities, probabilities=probabilities, **args)
                replay = load_bank(directory/'train', expected_identities=train.identities, **args)
                if not np.array_equal(replay, probabilities):
                    raise ValueError('Science TRAIN bank readback differs')
            result = dict(train_bank=task_id+'/train', teacher_report_sha256=fitted['training_report_sha256'])
        execution(spec, evidence)
    elif task['kind'] == 'aggregate':
        rows = result_rows(spec)
        if any(row['state'] != 'COMPLETE' for row in rows):
            raise ValueError('Every registered row required regardless of metrics')
        result = c.make_artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
            recovery_reference=dict(zero='M0HLT', hundred='OFFLINE'),
            caveats=['single seed', 'validation reused for selection and reporting', 'unequal compute/capacity',
                     'provisional mm and error-only zero masking', 'cross-file event independence unverified'])
        write_immutable_json(directory/'results.json', result)
    else:
        result = dict(fits=12, reducers=7, scientific_performance_never_controls_completion=True)
    # Reauthenticate after long execution. Publication is completion-last.
    c.validate_spec(spec)
    for name, expected in dependencies.items():
        if completed(spec, name)['content_hash'] != expected:
            raise ValueError('Dependency changed during execution')
    receipt = c.make_artifact('TASK', parents=dict(spec=spec['content_hash'], preflight=spec['preflight_sha256']),
        task_id=task_id, source_commit=spec['source_commit'], prepared_sha256=spec['prepared_sha256'],
        dependencies=dependencies, job_id=job, result=result,
        outputs=[file_ref(path, root=root) for path in sorted(directory.rglob('*')) if path.is_file()])
    write_immutable_json(directory/'receipt.json', receipt)
    return completed(spec, task_id)


def print_results(spec):
    c.validate_spec(spec)
    rows = result_rows(spec)
    print('Luka genuine FullSim: 100,000 train / 50,000 validation; final test untouched.')
    print('Recovery: M0HLT=0%, pure OFFLINE=100%; single-seed development validation.')
    print(f"{'node':25} {'pick':>10} {'accuracy':>10} {'AUC':>10} {'acc rec.%':>10} {'AUC rec.%':>10}")
    def fmt(value, digits=6):
        return 'n/a' if value is None else f'{value:.{digits}f}'
    for row in rows:
        m, rec = row['validation'] or {}, row['recovery_to_pure_offline'] or {}
        pick = row['state'] if row['passes'] is None else f"{row['selected_pass']}/{row['passes']}"
        print(f"{row['node_id']:25} {pick:>10} {fmt(m.get('accuracy')):>10} {fmt(m.get('macro_ovr_auc')):>10} "
              f"{fmt(rec.get('accuracy'),1):>10} {fmt(rec.get('macro_ovr_auc'),1):>10}")
        if m:
            print('  QCD rejection @50%: '+', '.join(k+'='+('censored (0 passing)' if v['zero_fpr_censored'] else
                  fmt(v['rejection_at_50pct'], 1)) for k, v in m['per_class'].items()))
    print('Offline-containing fusion is an oracle; HLT/HLT fusion uses two encoders. Compare final single-HLT endpoints.')
