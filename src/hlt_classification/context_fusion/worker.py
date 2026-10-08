"""Production paired-view training, measured Oscar acceptance, and read-only results."""
from io import BytesIO
from pathlib import Path
import gc
import math
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import atomic_publish_bytes
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.banks import load_bank, publish_bank
from hlt_classification.jetclass2_delphes.contracts import validate as validate_kernel
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment, distillation_loss
from hlt_classification.jetclass2_delphes.reporting import recovery
from hlt_classification.jetclass2_delphes.runner import train_kernel, predict, _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import native_mask_parity, native_offload_parity
from . import campaign as c
from .contracts import artifact, validate, file_ref, validate_file_ref, load_json, write_json, safe
from .inputs import cache, new_model, PairedCache


def save_state(path, state):
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def completed(spec, task_id, _seen=None):
    seen = {} if _seen is None else _seen
    if task_id in seen:
        return seen[task_id]
    task = next((r for r in c.tasks() if r['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unregistered fusion task')
    root = Path(spec['root'])
    path = safe(root, task_id+'/receipt.json')
    if not path.exists():
        return None
    row = load_json(path)
    measured = load_json(root/'preflight/result.json')
    validate(row, 'TASK', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash']))
    if (row['task_id'] != task_id or row['source_commit'] != spec['source_commit']
            or row['foundation_sha256'] != spec['parent_import']['foundation_sha256']):
        raise ValueError('Task identity differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    required = {}
    for name in task['dependencies']:
        dep = completed(spec, name, seen)
        if dep is None:
            raise ValueError('Published child lacks its parent')
        required[name] = dep['content_hash']
    if row['dependencies'] != required:
        raise ValueError('Task dependency artifacts differ')
    if task['kind'] == 'train':
        report = load_json(safe(root, row['result']['training_report']))
        validate_kernel(report, 'KERNEL_TRAINING_REPORT')
        node = next(n for n in c.nodes() if n['node_id'] == task['node_id'])
        if (report['node'] != node or report['content_hash'] != row['result']['training_report_sha256']
                or report['recipe_sha256'] != spec['training_recipe']['content_hash']
                or report['foundation_sha256'] != row['foundation_sha256']
                or report['scientific_fit'] is not True or report['acceptance_only'] is not False
                or report['final_test_accessed'] is not False or report['selected_weights_restored'] is not True
                or not spec['training_recipe']['minimum_passes'] <= report['passes'] <= spec['training_recipe']['maximum_passes']
                or not 1 <= report['selected_pass'] <= report['passes']):
            raise ValueError('Scientific training report differs')
        paths = {r['path'] for r in row['outputs']}
        if not {row['result']['training_report'], row['result']['checkpoint']} <= paths:
            raise ValueError('Missing declared report/checkpoint bytes')
    if task['kind'] == 'reduce':
        fitted = seen['train_'+task['node_id']]['result']
        if row['result']['teacher_report_sha256'] != fitted['training_report_sha256']:
            raise ValueError('Reducer teacher report differs')
        if row['result']['train_bank']+'/manifest.json' not in {r['path'] for r in row['outputs']}:
            raise ValueError('Missing declared probability manifest')
    seen[task_id] = row
    return row


def teacher(spec, parent, name, identities):
    if name == 'U000':
        result, root = spec['parent_import']['u000_bank'], Path(parent['campaign_root'])
    else:
        pointer = completed(spec, 'reduce_'+name)
        if pointer is None:
            raise ValueError('Teacher is not committed')
        result, root = pointer['result'], Path(spec['root'])
    return load_bank(safe(root, result['train_bank']), foundation_sha256=parent['foundation']['content_hash'],
        teacher_report_sha256=result['teacher_report_sha256'], teacher_node=name,
        role='train', expected_identities=identities)


def execution(spec, *, measured=None):
    _, cpus, memory = allocation(spec['execution_site'])
    if (cpus, memory) != (spec['cpus'], spec['memory_mb']):
        raise PermissionError('Wrong registered Oscar allocation')
    if measured is not None and (gpu_identity() != measured['gpu'] or installed_environment() != measured['environment']):
        raise PermissionError('Preflight/science GPU or installed environment differs')


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def largest_raw(cache_value, size=256):
    lengths = np.concatenate([np.diff(b.offsets) for b in cache_value.blocks])
    return cache_value.batch(np.repeat(np.argmax(lengths), size).astype(np.int64))


def stress(model, train):
    # Technical upper envelope: repeat the longest real TRAIN view 256 times,
    # independently on each side. These are not scored jets or KD targets.
    if isinstance(train, PairedCache):
        raws = [largest_raw(view) for view in (train.primary, train.context)]
        length = max(r['mask'].shape[-1] for r in raws)
        raw = dict(labels=raws[0]['labels'])
        for k in ('features', 'vectors', 'mask'):
            raw[k] = np.stack([np.pad(r[k], ((0, 0), (0, 0), (0, length-r[k].shape[-1]))) for r in raws], 1)
        model.reset_pair_offload_stats()
    else:
        raw = largest_raw(train)
    model.to('cuda').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, raw, 'cuda', bf16=True)
        loss = distillation_loss(logits, torch.from_numpy(raw['labels']).to('cuda'),
            teacher_probabilities=torch.full((256, 11), 1/11, device='cuda'))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite stress loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite stress gradient')
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return model.pair_offload_stats() if isinstance(train, PairedCache) else None


def measure(spec, parent, directory):
    from hlt_classification.cms2jc2_response.measurement import Measurement
    with Measurement() as measured:
        result = _measure(spec, parent, directory)
    result = artifact('PREFLIGHT', **{k: v for k, v in result.items()
        if k not in {'contract', 'schema_version', 'content_hash', 'final_test_accessed', 'peak_rss_bytes'}},
        peak_rss_bytes=max(result['peak_rss_bytes'], measured.report()['sampled_peak_tree_rss_bytes']))
    c.validate_preflight(result, spec, parent)
    return result


def _measure(spec, parent, directory):
    execution(spec)
    import resource
    torch.cuda.reset_peak_memory_stats()
    measurements, parities = [], {}
    cache_bytes = 0
    for index in (0, 6, 5):
        node = c.nodes()[index]
        print('CONTEXT-FUSION preflight='+node['node_id'], flush=True)
        started = time.monotonic()
        train = cache(parent, node, 'train', workers=spec['workers'], memory_mb=spec['memory_mb'])
        val = cache(parent, node, 'validation', workers=spec['workers'], memory_mb=spec['memory_mb'])
        cache_seconds = time.monotonic()-started
        if (len(train), len(val)) != (spec['counts']['train'], spec['counts']['validation']):
            raise ValueError('Preflight population differs')
        cache_bytes = max(cache_bytes, train.nbytes+val.nbytes)
        if index == 0:
            single_parity = installed_parity(train.primary, device='cuda')
            mask_parity = native_mask_parity(train.primary.batch(np.arange(4)), device='cuda')
        if isinstance(train, PairedCache):
            for bf16 in (False, True):
                key = ('adjacent' if index == 0 else 'endpoint')+('_bf16' if bf16 else '_fp32')
                parities[key] = native_offload_parity(train.primary.batch(np.arange(4)),
                    train.context.batch(np.arange(4)), device='cuda', bf16=bf16)
                clear()
        model = new_model(node)
        stats = stress(model, train)
        del model
        clear()
        # All acceptance KD uses the actual authenticated U000 TRAIN bank.
        # This is nonscientific resource measurement, not a new teacher route.
        q = teacher(spec, parent, 'U000', train.identities)
        model = new_model(node)
        training, state = train_kernel(model, train, val, node=node, device='cuda',
            teacher_probabilities=q, teacher_identities=train.identities, acceptance_passes=1)
        checkpoint = directory/(node['node_id']+'.pt')
        save_state(checkpoint, state)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
            raise ValueError('Selected checkpoint roundtrip differs')
        model.load_state_dict(restored, strict=True)
        started = time.monotonic()
        for role, view, temp in (('train', train, 2.), ('validation', val, 1.)):
            probabilities = predict(model, view, device='cuda', temperature=temp)
            bank = directory/(node['node_id']+'_'+role)
            publish_bank(bank, foundation_sha256=view.foundation_sha256, teacher_report_sha256=training['content_hash'],
                teacher_node=node['node_id'], role=role, identities=view.identities, probabilities=probabilities)
            replay = load_bank(bank, foundation_sha256=view.foundation_sha256, teacher_report_sha256=training['content_hash'],
                teacher_node=node['node_id'], role=role, expected_identities=view.identities)
            if not np.array_equal(probabilities, replay):
                raise ValueError('Acceptance bank roundtrip differs')
        inference_seconds = time.monotonic()-started
        measurements.append(dict(node=node, training=training, cache_seconds=cache_seconds,
            inference_seconds=inference_seconds, stress_steps=3, stress_batch=256, stress_offload=stats))
        del train, val, view, q, model, state, restored, probabilities, replay
        clear()
    train_minutes = max(60, math.ceil(max((r['cache_seconds']+100*r['training']['runtime_seconds'])*1.75 for r in measurements)/60))
    reduce_minutes = max(30, math.ceil(max((r['cache_seconds']+r['inference_seconds'])*2 for r in measurements)/60))
    result = artifact('PREFLIGHT', parents=dict(spec=spec['content_hash']), passed=True,
        counts=spec['counts'], site=spec['execution_site'], source_commit=spec['source_commit'],
        single_parity=single_parity, native_mask_parity=mask_parity, offload_parity=parities,
        checkpoint_bank_roundtrip=True, environment=installed_environment(), gpu=gpu_identity(),
        gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cache_bytes=cache_bytes, measurements=measurements, train_minutes=train_minutes, reduce_minutes=reduce_minutes)
    c.validate_preflight(result, spec, parent)
    return result


def result_rows(spec, *, parent=None):
    parent = c.validate_spec(spec) if parent is None else parent
    rows = [dict(r, origin='reused_parent') for r in spec['parent_import']['rows']]
    metrics = {r['node_id']: r['validation'] for r in rows}
    seen = {}
    for node in c.nodes():
        pointer = completed(spec, 'train_'+node['node_id'], seen)
        report = None if pointer is None else load_json(safe(Path(spec['root']), pointer['result']['training_report']))
        value = None if report is None else report['validation']
        rows.append(dict(node_id=node['node_id'], origin='new_fusion', deployable=node['deployable'],
            state='UNFINISHED' if report is None else 'COMPLETE', validation=value,
            selected_pass=None if report is None else report['selected_pass'], passes=None if report is None else report['passes'],
            recovery_to_pure_offline=None if value is None else recovery(value, metrics['M0HLT'], metrics['OFFLINE'])))
    return rows


def run(spec, task_id):
    parent = c.validate_spec(spec)
    root = Path(spec['root'])
    if task_id == 'preflight':
        path = root/'preflight'/'result.json'
        if path.exists():
            return c.preflight(spec, parent)
        directory = root/'preflight'
        directory.mkdir(exist_ok=False)
        result = measure(spec, parent, directory)
        write_json(path, result)
        return result
    task = next((r for r in c.tasks() if r['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unregistered task')
    evidence = c.preflight(spec, parent)
    old = completed(spec, task_id)
    if old is not None:
        return old
    dependencies = {}
    for name in task['dependencies']:
        pointer = completed(spec, name)
        if pointer is None:
            raise ValueError('Uncommitted dependency: '+name)
        dependencies[name] = pointer['content_hash']
    directory = safe(root, task_id)
    directory.mkdir(exist_ok=False)  # Failed attempts are not silently overwritten.
    outputs = []
    if task['kind'] in {'train', 'reduce'}:
        execution(spec, measured=evidence)
        node = next(n for n in c.nodes() if n['node_id'] == task['node_id'])
        train = cache(parent, node, 'train', workers=spec['workers'], memory_mb=spec['memory_mb'])
        model = new_model(node)
        if task['kind'] == 'train':
            val = cache(parent, node, 'validation', workers=spec['workers'], memory_mb=spec['memory_mb'])
            q = teacher(spec, parent, node['teacher'], train.identities)
            report, state = train_kernel(model, train, val, node=node, device='cuda',
                teacher_probabilities=q, teacher_identities=train.identities)
            checkpoint, training_path = directory/'selected.pt', directory/'training_report.json'
            save_state(checkpoint, state)
            write_json(training_path, report)
            outputs += [checkpoint, training_path]
            result = dict(checkpoint=checkpoint.relative_to(root).as_posix(),
                training_report=training_path.relative_to(root).as_posix(), training_report_sha256=report['content_hash'])
        else:
            fitted = completed(spec, 'train_'+node['node_id'])['result']
            model.load_state_dict(torch.load(safe(root, fitted['checkpoint']), map_location='cpu', weights_only=True), strict=True)
            model.to('cuda').eval()
            bank = directory/'train'
            probabilities = predict(model, train, device='cuda', temperature=2.)
            publish_bank(bank, foundation_sha256=train.foundation_sha256,
                teacher_report_sha256=fitted['training_report_sha256'], teacher_node=node['node_id'], role='train',
                identities=train.identities, probabilities=probabilities)
            load_bank(bank, foundation_sha256=train.foundation_sha256, teacher_report_sha256=fitted['training_report_sha256'],
                teacher_node=node['node_id'], role='train', expected_identities=train.identities)
            outputs += sorted(bank.iterdir())
            result = dict(train_bank=bank.relative_to(root).as_posix(), teacher_report_sha256=fitted['training_report_sha256'])
    elif task['kind'] == 'aggregate':
        rows = result_rows(spec, parent=parent)
        if any(r['state'] != 'COMPLETE' for r in rows):
            raise ValueError('All comparison rows are required, regardless of performance')
        result = artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
            recovery_reference=dict(zero='M0HLT', hundred='OFFLINE'),
            caveats=['reused development validation', 'single seed', 'unequal total compute and fusion capacity'])
        path = directory/'results.json'
        write_json(path, result)
        outputs.append(path)
    else:
        result = dict(new_fits=8, new_reducers=6, historical_comparisons=9,
            scientific_performance_never_controls_completion=True)
    receipt = artifact('TASK', parents=dict(spec=spec['content_hash'], preflight=evidence['content_hash']),
        source_commit=spec['source_commit'], task_id=task_id, dependencies=dependencies,
        foundation_sha256=parent['foundation']['content_hash'],
        result=result, outputs=[file_ref(p, root=root) for p in outputs])
    write_json(directory/'receipt.json', receipt)
    return receipt


def print_results(spec):
    rows = result_rows(spec)
    print('CONTEXT_V1: 100,000 train / 50,000 validation; final test untouched.')
    print('M0HLT=0%, pure OFFLINE=100% recovery. Parent controls/direct/coarse are reused.')
    print(f"{'node':44} {'pick':>9} {'accuracy':>10} {'AUC':>10} {'AUC rec.%':>10}")
    def fmt(x, digits=6):
        return 'n/a' if x is None else f'{x:.{digits}f}'
    for r in rows:
        m, rec = r['validation'] or {}, r['recovery_to_pure_offline'] or {}
        pick = r['state'] if r['passes'] is None else f"{r['selected_pass']}/{r['passes']}"
        print(f"{r['node_id']:44} {pick:>9} {fmt(m.get('accuracy')):>10} {fmt(m.get('macro_ovr_auc')):>10} {fmt(rec.get('macro_ovr_auc'),1):>10}")
        if m:
            print('  QCD rejection @50%: '+', '.join(f"{k}="+('censored (0 passing)' if v['zero_fpr_censored'] else fmt(v['rejection_at_50pct'], 1)) for k, v in m['per_class'].items()))
    print('Intermediate offline-containing fusion models are oracles. Compare final single-HLT endpoints.')
