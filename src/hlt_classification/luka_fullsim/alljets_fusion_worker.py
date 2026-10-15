"""Cached paired inputs and receipt-bound fusion fits; no parent writes or raw fallback."""
from pathlib import Path
import os

import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import safe, file_ref, validate_file_ref
from hlt_classification.context_fusion.inputs import PairedCache, new_model
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.reporting import recovery, evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
from . import alljets_fusion_campaign as c, regular_cache, alljets_data
from .regular_worker import validate_training
from .preflight_v2 import require_backend, PhaseLog, memory_snapshot


def cache(spec, prepared, node, role, *, authenticated=None):
    if role not in ('train', 'validation'):
        raise PermissionError('No test cache capability')
    parent = load_json(spec['parent_import']['spec']['path'])
    root = parent['admission']['prepared_root']
    authenticated = regular_cache.metadata(root) if authenticated is None else authenticated
    if (authenticated[1]['content_hash'] != prepared['content_hash']
            or authenticated[0]['content_hash'] != spec['parent_import']['evidence']['cache']):
        raise ValueError('Imported durable cache lineage differs')
    def single(co):
        return regular_cache.load(root, role=role, coordinate=co,
            max_ram_bytes=int(spec['memory_mb']*1024**2*.15), authenticated=authenticated)
    primary = single(node['coordinate'])
    other = node['context_coordinate']
    if other is None:
        return primary
    context = primary if other == node['coordinate'] else single(other)
    return PairedCache(primary, context)


def execution(spec, measured=None):
    require_backend()
    if spec.get('schema_version') in (2, 3, 4):
        from .alljets_fusion_sync import PAIR_STORAGE
        if spec.get('pair_storage') != PAIR_STORAGE or os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
            raise PermissionError('Synchronous production policy/backend differs; no diagnostic launch blocking')
    job, cpus, memory = allocation(spec['execution_site'])
    if (cpus, memory) != (spec['cpus'], spec['memory_mb']):
        raise PermissionError('Wrong fusion allocation')
    if measured is not None and (gpu_identity() != measured['gpu'] or installed_environment() != measured['environment']
            or os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset') != measured['cudnn_v8_disabled']):
        raise PermissionError('Measured GPU/environment/backend changed')
    return job


def dependency_hashes(spec, task, seen):
    values = {}
    for dep in task['dependencies']:
        row = completed(spec, dep, seen)
        if row is None:
            raise ValueError('Missing committed teacher/dependency: '+dep)
        values[dep] = row['content_hash']
    if task['task_id'] == 'train_FUSION_U050' and spec.get('schema_version') != 4:
        values['external:reduce_U000'] = spec['parent_import']['evidence']['tasks']['reduce_U000']
    return values


def diagnostic(spec, task, report, arrays, *, authenticated=None):
    if authenticated is None:
        parent = load_json(spec['parent_import']['spec']['path'])
        authenticated = regular_cache.metadata(parent['admission']['prepared_root'])
    _, p, inv, splits = authenticated
    if p['content_hash'] != spec['prepared_sha256']:
        raise ValueError('Diagnostic population differs')
    if (set(arrays) != {'identities', 'labels', 'probabilities'}
            or arrays['identities'].dtype != np.uint8 or arrays['labels'].dtype != np.int64
            or arrays['probabilities'].dtype != np.float32
            or arrays['probabilities'].shape != (spec['counts']['validation'], 11)):
        raise ValueError('Invalid validation prediction schema')
    ix = alljets_data.comparison_indices(inv, splits, arrays['identities'], arrays['labels'])
    root = Path(spec['root'])
    return c.bound_artifact(spec, 'DIAGNOSTICS', parents=dict(spec=spec['content_hash'], report=report['content_hash']),
        task_id=task, selection_population='entire_validation', secondary_selection_used=False,
        checkpoint=file_ref(root/task/'selected.pt', root=root),
        predictions=file_ref(root/task/'validation_predictions.npz', root=root),
        validation=evaluate_probabilities(arrays['labels'], arrays['probabilities']),
        original_validation=evaluate_probabilities(arrays['labels'][ix], arrays['probabilities'][ix]))


def completed(spec, name, seen=None):
    seen = {} if seen is None else seen
    if name in seen:
        return seen[name]
    task = next((t for t in c.scoped_tasks(spec) if t['task_id'] == name), None)
    if task is None:
        raise ValueError('Unregistered fusion task')
    root = Path(spec['root'])
    if not (root/name/'receipt.json').is_file():
        return None
    row = load_json(root/name/'receipt.json')
    c.validate_bound(row, spec, 'TASK', dict(spec=spec['content_hash'], preflight=spec['preflight_sha256']))
    if (row['task_id'] != name or row['source_commit'] != spec['source_commit']
            or row['prepared_sha256'] != spec['prepared_sha256']
            or row['dependencies'] != dependency_hashes(spec, task, seen)):
        raise ValueError('Fusion task/source/dependency binding differs')
    paths = [ref['path'] for ref in row['outputs']]
    if len(paths) != len(set(paths)) or any(not x.startswith(name+'/') for x in paths):
        raise ValueError('Task output scope differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    result = row['result']
    if task['kind'] == 'train':
        expected = {'training_report': name+'/training_report.json', 'checkpoint': name+'/selected.pt'}
        if any(result.get(k) != v or v not in paths for k,v in expected.items()):
            raise ValueError('Missing scientific checkpoint/report')
        node = next(n for n in c.scoped_nodes(spec) if n['node_id'] == task['node_id'])
        report = load_json(root/expected['training_report'])
        validate_training(spec, node, report)
        if result['training_report_sha256'] != report['content_hash']:
            raise ValueError('Scientific report digest differs')
        required = {name+'/validation_predictions.npz', name+'/validation_diagnostics.json'}
        if not required <= set(paths):
            raise ValueError('Missing selected validation predictions')
        arrays = load_npz_arrays(root/name/'validation_predictions.npz')
        diag = load_json(root/name/'validation_diagnostics.json')
        if '_diagnostic_population' not in seen:
            parent = load_json(spec['parent_import']['spec']['path'])
            seen['_diagnostic_population'] = regular_cache.metadata(parent['admission']['prepared_root'])
        if (diag != diagnostic(spec, name, report, arrays, authenticated=seen['_diagnostic_population'])
                or result['diagnostics_sha256'] != diag['content_hash']):
            raise ValueError('Selected validation replay differs')
    elif task['kind'] == 'reduce':
        fit = seen['train_'+task['node_id']]['result']
        if result != dict(train_bank=name+'/train', teacher_report_sha256=fit['training_report_sha256']):
            raise ValueError('Reducer teacher/path differs')
        manifest_path = name+'/train/manifest.json'
        if manifest_path not in paths:
            raise ValueError('Missing bound TRAIN bank')
        bank = load_json(root/manifest_path)
        kernel_validate(bank, 'PROBABILITY_BANK')
        expected = dict(foundation_sha256=spec['prepared_sha256'], teacher_report_sha256=fit['training_report_sha256'],
            teacher_node=task['node_id'], role='train', temperature=2., class_count=11, dtype='float32',
            rows=spec['counts']['train'], final_test_accessed=False)
        if any(bank.get(k) != v for k,v in expected.items()):
            raise ValueError('TRAIN bank lineage differs')
        for shard in bank['shards']:
            if dict(path=name+'/train/'+shard['path'], bytes=shard['bytes'], sha256=shard['sha256']) not in row['outputs']:
                raise ValueError('Unbound probability shard')
    elif task['kind'] == 'aggregate':
        c.validate_bound(result, spec, 'RESULTS', dict(spec=spec['content_hash']))
        if (name+'/results.json' not in paths or load_json(root/name/'results.json') != result
                or result['rows'] != result_rows(spec, seen=seen)):
            raise ValueError('Aggregate row replay differs')
    elif result != c.completion(spec):
        raise ValueError('Wrong completion coverage')
    seen[name] = row
    return row


def teacher(spec, name, identities):
    if name == 'U000' and spec.get('schema_version') != 4:
        parent = load_json(spec['parent_import']['spec']['path'])
        root, result = Path(parent['root']), spec['parent_import']['evidence']['u000_bank']
    else:
        receipt = completed(spec, 'reduce_'+name)
        if receipt is None:
            raise ValueError('Teacher is not committed')
        root, result = Path(spec['root']), receipt['result']
    return load_bank(safe(root, result['train_bank']), foundation_sha256=spec['prepared_sha256'],
        teacher_report_sha256=result['teacher_report_sha256'], teacher_node=name, role='train', expected_identities=identities)


def input_label(node):
    if not node['deployable']:
        return 'OFFLINE ORACLE'
    return 'two HLT encoders' if node.get('context_coordinate') else 'single HLT'


def result_rows(spec, *, seen=None):
    rows = [dict(r, origin='parent', input='single HLT' if r['deployable'] else 'OFFLINE ORACLE')
            for r in ([] if spec.get('schema_version') == 4 else spec['parent_import']['evidence']['rows'])]
    seen = {} if seen is None else seen
    for node in c.scoped_nodes(spec):
        name = 'train_'+node['node_id']
        receipt = completed(spec, name, seen)
        report = None if receipt is None else load_json(Path(spec['root'])/name/'training_report.json')
        diag = None if receipt is None else load_json(Path(spec['root'])/name/'validation_diagnostics.json')
        rows.append(dict(node_id=node['node_id'], origin='new', input=input_label(node), deployable=node['deployable'],
            state='UNFINISHED' if report is None else 'COMPLETE',
            selected_pass=None if report is None else report['selected_pass'], passes=None if report is None else report['passes'],
            validation=None if report is None else report['validation'], original_validation=None if diag is None else diag['original_validation']))
    controls = {r['node_id']: r for r in rows}
    for row in rows:
        for field, dest in [('validation', 'recovery_to_pure_offline'), ('original_validation', 'original_validation_recovery')]:
            row[dest] = (None if row[field] is None or controls['M0HLT'][field] is None or controls['OFFLINE'][field] is None
                         else recovery(row[field], controls['M0HLT'][field], controls['OFFLINE'][field]))
    return rows


def run(spec, name):
    _, p, measured = c.validate_spec(spec)
    task = next((t for t in c.scoped_tasks(spec) if t['task_id'] == name), None)
    if task is None:
        raise ValueError('Unregistered task; no test or parent-write capability')
    prior = completed(spec, name)
    if prior is not None:
        return prior
    deps = dependency_hashes(spec, task, {})
    job = execution(spec, measured) if task['kind'] in ('train', 'reduce') else os.environ.get('SLURM_JOB_ID')
    root, directory = Path(spec['root']), Path(spec['root'])/name
    directory.mkdir(exist_ok=False)
    if task['kind'] in ('train', 'reduce'):
        node = next(n for n in c.scoped_nodes(spec) if n['node_id'] == task['node_id'])
        log = PhaseLog(directory, lambda: memory_snapshot(torch))
        with log.phase('load_authenticated_train_views'):
            train = cache(spec, p, node, 'train')
        if spec.get('schema_version') in (2, 3, 4):
            from .alljets_fusion_sync import new_model as model_factory
        else:
            model_factory = new_model
        model = model_factory(node)
        if task['kind'] == 'train':
            with log.phase('load_authenticated_validation_views'):
                val = cache(spec, p, node, 'validation')
            probabilities = None if node['teacher'] is None else teacher(spec, node['teacher'], train.identities)
            with log.phase('scientific_fit'):
                report, state = train_kernel(model, train, val, node=node, device='cuda',
                    teacher_probabilities=probabilities, teacher_identities=None if probabilities is None else train.identities,
                    **(dict(training_recipe=spec['training_recipe']) if spec.get('schema_version') == 4 else {}))
                validate_training(spec, node, report)
            with log.phase('checkpoint_and_validation_readback'):
                save_state(directory/'selected.pt', state)
                restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
                if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
                    raise ValueError('Selected checkpoint roundtrip differs')
                write_immutable_json(directory/'training_report.json', report)
                model.load_state_dict(restored, strict=True)
                arrays = dict(identities=val.identities, labels=val.labels,
                    probabilities=predict(model, val, device='cuda', temperature=1., batch_size=spec['training_recipe'].get('inference_batch_size', 256)))
                atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(arrays))
                diag = diagnostic(spec, name, report, arrays)
                write_immutable_json(directory/'validation_diagnostics.json', diag)
            result = dict(checkpoint=name+'/selected.pt', training_report=name+'/training_report.json',
                training_report_sha256=report['content_hash'], diagnostics_sha256=diag['content_hash'])
        else:
            fit = completed(spec, 'train_'+node['node_id'])['result']
            with log.phase('selected_teacher_train_bank'):
                model.load_state_dict(torch.load(root/fit['checkpoint'], map_location='cpu', weights_only=True), strict=True)
                model.to('cuda').eval()
                probabilities = predict(model, train, device='cuda', temperature=2., batch_size=spec['training_recipe'].get('inference_batch_size', 256))
                args = dict(foundation_sha256=p['content_hash'], teacher_report_sha256=fit['training_report_sha256'],
                    teacher_node=node['node_id'], role='train')
                publish_bank(directory/'train', identities=train.identities, probabilities=probabilities, **args)
                if not np.array_equal(probabilities, load_bank(directory/'train', expected_identities=train.identities, **args)):
                    raise ValueError('Teacher bank roundtrip differs')
            result = dict(train_bank=name+'/train', teacher_report_sha256=fit['training_report_sha256'])
        execution(spec, measured)
    elif task['kind'] == 'aggregate':
        rows = result_rows(spec)
        if any(r['state'] != 'COMPLETE' for r in rows):
            raise ValueError('Every registered model required, regardless of quality')
        result = c.bound_artifact(spec, 'RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
            caveats=['single seed', 'development validation', 'no independent test', 'unequal capacity/training compute',
                     'provisional mm', 'cross-file event independence unverified'])
        write_immutable_json(directory/'results.json', result)
    else:
        result = c.completion(spec)
    c.validate_spec(spec)
    if deps != dependency_hashes(spec, task, {}):
        raise ValueError('Dependency changed during execution')
    receipt = c.bound_artifact(spec, 'TASK', parents=dict(spec=spec['content_hash'], preflight=spec['preflight_sha256']),
        task_id=name, source_commit=spec['source_commit'], prepared_sha256=p['content_hash'], job_id=job,
        dependencies=deps, result=result,
        outputs=[file_ref(path, root=root) for path in sorted(directory.rglob('*')) if path.is_file()])
    write_immutable_json(directory/'receipt.json', receipt)
    return completed(spec, name)


def print_results(spec):
    c.validate_spec(spec)
    rows = result_rows(spec)
    print('Luka genuine FullSim fusion: 299,234 train / 74,895 validation; NO independent test.')
    print('M0HLT=0%, pure OFFLINE=100%; ' + ('all17 models refitted at true batch128, no accumulation or imported teachers.'
        if spec.get('schema_version') == 4 else 'nine parent fits reused read-only; eight fresh fusion-route fits.'))
    for field, recfield in [('validation', 'recovery_to_pure_offline'), ('original_validation', 'original_validation_recovery')]:
        print('\n'+('FULL validation (checkpoint selection)' if field == 'validation' else
            'ORIGINAL 50k validation subset; SAME selected checkpoints, no subset selection'))
        print(f"{'node':25} {'accuracy':>10} {'AUC':>10} {'acc rec%':>10} {'AUC rec%':>10} input")
        for row in rows:
            m, rec = row[field], row[recfield] or {}
            if m is None:
                print(f"{row['node_id']:25} UNFINISHED")
                continue
            def fmt(x, digits=6):
                return 'n/a' if x is None else f'{x:.{digits}f}'
            print(f"{row['node_id']:25} {fmt(m['accuracy']):>10} {fmt(m['macro_ovr_auc']):>10} "
                  f"{fmt(rec.get('accuracy'),1):>10} {fmt(rec.get('macro_ovr_auc'),1):>10} {row['input']}")
            if field == 'validation':
                print('  QCD rejection @50%: '+', '.join(k+'='+('censored (0 passing)' if v['zero_fpr_censored']
                      else fmt(v['rejection_at_50pct'],1)) for k,v in m['per_class'].items()))
    print('Compare single-HLT endpoints; offline fusion is oracle, D000/D000 has two encoders. Single seed, unequal compute.')
