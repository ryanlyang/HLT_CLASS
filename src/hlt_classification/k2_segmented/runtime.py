"""Execution receipts and source-authenticated K2 segmented training workers."""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.jetclass2_delphes import concat_k2_runtime as legacy
from hlt_classification.jetclass2_delphes.banks import load_bank, publish_bank
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment, model_contract
from hlt_classification.jetclass2_delphes.concat_k2_model import parity_backend
from hlt_classification.jetclass2_delphes.salience_learned_training import predict, train_kernel
from . import training
from .campaign import (artifact, validate, original, validate_spec, graph, NODES,
    authenticate_job, gate, submit, AUTHORIZE, require_authorization)


def completed(spec, name):
    root = Path(spec['campaign_root'])
    path = root / 'tasks' / (name + '.json')
    if not path.exists():
        return None
    value = load_json(path)
    validate(value, 'TASK_REPORT')
    row = next(r for r in graph() if r['task_id'] == name)
    if (value['campaign_sha256'] != spec['content_hash'] or value['task_id'] != name
            or value['executor_source'] != spec['source_commit']
            or value['original_scientific_source'] != spec['original_commit']
            or value['final_test_accessed'] or set(value['parents']) != set(row['dependencies'])):
        raise ValueError('Segmented task identity differs')
    for parent, digest in value['parents'].items():
        prior = load_json(root / 'tasks' / (parent + '.json'))
        validate(prior, 'TASK_REPORT')
        if prior['content_hash'] != digest or prior['campaign_sha256'] != spec['content_hash']:
            raise ValueError('Segmented task parent differs')
    if not value['outputs']:
        raise ValueError('Task has no authenticated outputs')
    for item in value['outputs']:
        if sha256_file(relative_file(root, item['path'])) != item['sha256']:
            raise ValueError('Segmented output bytes differ')
    return value


def node_for(source, name):
    return next(n for n in source['nodes'] if n['node_id'] == name)


def teacher(spec, source, node, identities):
    name = node['teacher_distribution']
    if name == 'CONCAT_K2_D050':
        return legacy.teacher(source, node, identities)
    receipt = completed(spec, 'reduce_' + name)
    if receipt is None:
        raise PermissionError('Teacher is not complete')
    result = receipt['result']
    q = load_bank(relative_file(Path(spec['campaign_root']), result['bank']),
        foundation_sha256=source['foundation']['content_hash'], teacher_report_sha256=result['training_report_sha256'],
        teacher_node=name, role='train', expected_identities=identities)
    return q, dict(task_sha256=receipt['content_hash'], teacher_node=name,
                   bank_manifest_sha256=result['bank_manifest_sha256'])


def binding_for(spec, node, lineage):
    return dict(campaign_sha256=spec['content_hash'], teacher_lineage=lineage,
                original_campaign_sha256=spec['source_spec']['content_hash'], training=spec['training'])


def full_binding(binding, node, train, validation, *, maximum=100, acceptance=False):
    return dict(**binding, node=node, identities=dict(train=training.population_hash(train),
        checkpoint=training.population_hash(validation)), maximum=maximum, acceptance=acceptance)


def can_start(state, deadline, *, reserve_seconds=1800, now=None):
    """Never start an epoch that is projected to run into shutdown/publication."""
    history = state['history'][-5:]
    estimate = max((r['train_seconds'] + r['validation_seconds'] for r in history), default=3600.) * 1.5
    return (time.monotonic() if now is None else now) + estimate + reserve_seconds < deadline


def _storage(spec, additional=0):
    root = Path(spec['campaign_root'])
    used = sum(p.stat().st_size for p in (root / 'checkpoints').rglob('*.pt'))
    if used + additional > spec['maximum_checkpoint_bytes']:
        raise OSError('Registered resumable-state storage budget exceeded')
    if shutil.disk_usage(root).free < spec['minimum_free_bytes'] + additional:
        raise OSError('Insufficient free space for an atomic full-state checkpoint')


def fit_segment(spec, source, row, deadline, device):
    root = Path(spec['campaign_root'])
    node = node_for(source, row['node_id'])
    if row['segment'] > 1:
        previous = completed(spec, row['dependencies'][0])
        if previous['result']['fit_complete']:
            # A prequeued spare segment authenticates the completed fit; no data
            # loading, optimizer creation or extra training is performed.
            return {**previous['result'], 'spare_segment': True}, [relative_file(root, p['path'])
                for p in previous['outputs'] if not p['path'].startswith('execution/')]
    values = legacy.caches(source, node)
    q, lineage = teacher(spec, source, node, values['train'].identities)
    binding = binding_for(spec, node, lineage)
    checkpoint_dir = root / 'checkpoints' / node['node_id']
    state, receipt = training.load_checkpoint(checkpoint_dir, full_binding(
        binding, node, values['train'], values['checkpoint']))
    if (row['segment'] == 1) != (state is None):
        raise ValueError('Segment/checkpoint start differs; no implicit retry')
    if row['segment'] > 1 and receipt['content_hash'] != previous['result']['resume_sha256']:
        raise ValueError('Resume checkpoint is not the preceding segment output')
    first_pass = 0 if state is None else state['pass_number']
    previous_hash = None if receipt is None else receipt['content_hash']
    estimate_bytes = 0 if receipt is None else receipt['payload_bytes']

    def save(current):
        nonlocal receipt, previous_hash, estimate_bytes
        # First science save is bounded by the measured native state size too.
        bound = max(estimate_bytes, gate(spec)['checkpoint_bytes_upper_bound'])
        _storage(spec, 2 * bound)
        receipt = training.write_checkpoint(checkpoint_dir, current, previous=previous_hash)
        previous_hash, estimate_bytes = receipt['content_hash'], receipt['payload_bytes']
        return False

    model = legacy.new_model(node)
    state = training.train_segment(model, values['train'], values['checkpoint'], node=node,
        device=device, teacher=q, binding=binding, resume=state, on_epoch=save,
        can_start_epoch=lambda current: can_start(current, deadline, reserve_seconds=spec['reserve_seconds']))
    if state['pass_number'] == first_pass:
        raise TimeoutError('Segment made no progress after rebuilding caches')
    paths = [checkpoint_dir / receipt['payload'], checkpoint_dir / f"epoch_{receipt['pass_number']:03d}.json"]
    result = dict(node_id=node['node_id'], fit_complete=state['complete'], passes=state['pass_number'],
        resume_sha256=receipt['content_hash'], resume_directory=checkpoint_dir.relative_to(root).as_posix(),
        spare_segment=False)
    if not state['complete']:
        if row['segment'] == 3:
            raise TimeoutError('Three-segment budget exhausted; state preserved, next rung remains blocked')
        return result, paths
    model.load_state_dict(state['best_state'], strict=True)
    probabilities = predict(model, values['report'], node=node, device=device, batch_size=128)
    directory = root / 'training' / node['node_id']
    checkpoint = directory / 'selected.pt'
    legacy.save_state(checkpoint, state['best_state'])
    report = artifact('TRAINING_REPORT', campaign_sha256=spec['content_hash'], node=node,
        training=source['training'], inference_batch_size=128, kernel_report=training.kernel_report(state),
        teacher_lineage=lineage, original_campaign_sha256=source['content_hash'],
        selected_checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_validation=state['best_metrics'],
        report_validation=legacy.evaluate_probabilities(values['report'].labels, probabilities),
        validation_partition_sha256=load_json(Path(source['campaign_root']) / 'validation_partition.json')['content_hash'],
        validation_report_not_final_test=True, matching_selection_used_validation=True, final_test_accessed=False)
    report_path = directory / 'training_report.json'
    write_immutable_json(report_path, report)
    paths += [checkpoint, report_path]
    if node['deployable']:
        deployment = directory / 'deployment.json'
        write_immutable_json(deployment, artifact('DEPLOYMENT', training_report_sha256=report['content_hash'],
            checkpoint=checkpoint.name, checkpoint_sha256=sha256_file(checkpoint), model=model_contract(),
            input_features=source['foundation']['inputs'],
            native_hlt_copies=1 if node['primary_coordinate'] == 'HLT_X1' else 3,
            preprocessing='concat_k2_views.deployment_inputs', offline_inputs=False, assignment_inputs=False,
            source_or_slot_embedding=False, final_test_accessed=False))
        paths.append(deployment)
    result.update(checkpoint=checkpoint.relative_to(root).as_posix(), training_report=report_path.relative_to(root).as_posix(),
                  training_report_sha256=report['content_hash'])
    return result, paths


def reduce(spec, source, row, device):
    root = Path(spec['campaign_root'])
    parent = completed(spec, row['dependencies'][0])
    result = parent['result']
    if not result['fit_complete']:
        raise PermissionError('Reducer cannot consume a partial fit')
    report = load_json(relative_file(root, result['training_report']))
    validate(report, 'TRAINING_REPORT')
    if not report['kernel_report']['scientific_fit'] or report['kernel_report']['acceptance_only']:
        raise PermissionError('Acceptance weights cannot teach science')
    node = node_for(source, row['node_id'])
    model = legacy.new_model(node)
    model.load_state_dict(torch.load(relative_file(root, result['checkpoint']), map_location='cpu', weights_only=True))
    model.to(device).eval()
    train = legacy.caches(source, node, train_only=True)['train']
    probabilities = predict(model, train, node=node, device=device, batch_size=128, temperature=2.)
    bank = root / 'banks' / node['node_id']
    manifest = publish_bank(bank, foundation_sha256=source['foundation']['content_hash'],
        teacher_report_sha256=report['content_hash'], teacher_node=node['node_id'], role='train',
        identities=train.identities, probabilities=probabilities)
    return dict(bank=bank.relative_to(root).as_posix(), bank_manifest_sha256=manifest['content_hash'],
        training_report_sha256=report['content_hash']), list(bank.glob('*'))


def assert_state_equal(left, right):
    """Exact comparison under the registered deterministic gate backend."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=False)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError('State keys differ')
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError('State lengths differ')
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    elif left != right:
        raise AssertionError(f'State value differs: {left!r} != {right!r}')


def scientific_history(history):
    return [{k: v for k, v in row.items() if k not in ('train_seconds', 'validation_seconds')} for row in history]


def resume_parity(train, validation, node, q, directory, device):
    """Real native reference -> continuous extension -> disk save/reconstruction."""
    model = legacy.new_model(node).to(device)
    initial = training.rng_state()
    ref_report, ref_best = train_kernel(model, lambda _: train, lambda _: validation,
        node=node, device=device, teacher_probabilities=q,
        teacher_identities=train.identities if q is not None else None,
        batch_size=128, inference_batch_size=128, acceptance_passes=3)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = legacy.new_model(node).to(device)
    training.restore_rng(initial)
    binding = dict(acceptance_subject='native_resume_check')
    continuous = training.train_segment(model, train, validation, node=node, device=device,
        teacher=q, binding=binding, maximum=3, acceptance=True)
    assert_state_equal(ref_best, continuous['best_state'])
    assert_state_equal(scientific_history(ref_report['validation_history']), scientific_history(continuous['history']))
    if ref_report['selected_pass'] != continuous['selected_pass']:
        raise AssertionError('Legacy checkpoint selection differs')
    del model
    gc.collect()
    model = legacy.new_model(node).to(device)
    training.restore_rng(initial)
    paused = training.train_segment(model, train, validation, node=node, device=device,
        teacher=q, binding=binding, maximum=3, acceptance=True, on_epoch=lambda _: True)
    commit = training.write_checkpoint(directory, paused)
    loaded, _ = training.load_checkpoint(directory, paused['binding'])
    del model, paused
    gc.collect()
    # Deliberately consume RNG and construct a new model and AdamW instance.
    torch.rand(17)
    np.random.rand(17)
    model = legacy.new_model(node).to(device)
    resumed = training.train_segment(model, train, validation, node=node, device=device,
        teacher=q, binding=binding, maximum=3, acceptance=True, resume=loaded)
    for key in ('model', 'optimizer', 'best_state', 'best_metrics', 'best_key', 'rng',
                'significant_auc', 'significant_pass', 'selected_pass', 'update', 'pass_number', 'complete'):
        assert_state_equal(continuous[key], resumed[key])
    assert_state_equal(scientific_history(continuous['history']), scientific_history(resumed['history']))
    return commit['payload_bytes']


def preflight(spec, source, device):
    if device != 'cuda' or not torch.cuda.is_available():
        raise PermissionError('Real installed-Weaver A100 resume acceptance required')
    root = Path(spec['campaign_root'])
    old_acceptance = legacy.science_gate(source)
    if old_acceptance['gpu'] != gpu_identity() or old_acceptance['environment'] != installed_environment():
        raise PermissionError('Hardware/software differs from original accepted K2 run')
    torch.cuda.reset_peak_memory_stats()
    node = node_for(source, NODES[0])
    started = time.monotonic()
    caches = legacy.caches(source, node)
    cache_seconds = time.monotonic() - started
    q, _ = legacy.teacher(source, node, caches['train'].identities)
    indices = legacy.representative(caches['train'], 256)
    val_indices = legacy.representative(caches['checkpoint'], 128)
    from hlt_classification.jetclass2_delphes.salience_learned_data import IndexedRamCache
    train = IndexedRamCache(caches['train'], indices, role='train')
    validation = IndexedRamCache(caches['checkpoint'], val_indices, role='validation')
    with parity_backend(device):
        size = resume_parity(train, validation, node, q[indices], root / 'preflight/checkpoint', device)
    checkpoint_bound = math.ceil(size * 1.5)  # Full history/metadata grows beyond the miniature.
    projected_storage = checkpoint_bound * 100 * len(NODES)
    if projected_storage > spec['maximum_checkpoint_bytes']:
        raise OSError('Full 100-pass checkpoint retention would exceed the registered budget')
    _storage(spec, projected_storage)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak = torch.cuda.max_memory_allocated()
    if rss > 320000 * 1024**2 * .8 or peak > gpu_identity()['total_memory_bytes'] * .9:
        raise MemoryError('Resume gate leaves insufficient production headroom')
    evidence = artifact('ACCEPTANCE', campaign_sha256=spec['content_hash'], passed=True,
        original_acceptance_sha256=old_acceptance['content_hash'], gpu=gpu_identity(),
        environment=installed_environment(), legacy_kernel_parity=True, uninterrupted_resumed_parity=True,
        ordinary_rows=dict(train=len(caches['train']), validation=source['role_counts']['validation']),
        cache_seconds=cache_seconds, peak_cpu_bytes=rss, peak_cuda_bytes=peak,
        checkpoint_bytes_upper_bound=checkpoint_bound, projected_checkpoint_storage_bytes=projected_storage,
        per_segment_minutes=1380, maximum_segments_per_fit=3, original_science_unchanged=True,
        final_test_accessed=False)
    path = root / 'preflight/acceptance.json'
    write_immutable_json(path, evidence)
    return dict(acceptance=path.relative_to(root).as_posix()), [path]


def result_rows(spec, source):
    rows = {row['node_id']: row for row in legacy.result_rows(source)}
    baseline = legacy.completed(source, 'train_HLT_X1_CE')
    offline = legacy.completed(source, 'train_OFFLINE_CE')
    def metric(receipt):
        return load_json(relative_file(Path(source['campaign_root']), receipt['result']['training_report']))['report_validation']
    for name in NODES:
        for segment in (1, 2, 3):
            receipt = completed(spec, f'train_{name}_part{segment}')
            if receipt and receipt['result']['fit_complete']:
                report = load_json(relative_file(Path(spec['campaign_root']), receipt['result']['training_report']))
                validate(report, 'TRAINING_REPORT')
                rows[name] = dict(node_id=name, deployable=report['node']['deployable'],
                    validation=report['report_validation'],
                    recovery=legacy.recovery(report['report_validation'], metric(baseline), metric(offline)),
                    selected_pass=report['kernel_report']['selected_pass'], passes=report['kernel_report']['passes'])
                break
    return list(rows.values())


def run(spec, name, *, device='cuda'):
    source = validate_spec(spec)
    deadline = authenticate_job(spec, name)
    if name == 'after_gate':
        require_authorization(spec)
        return submit(spec, stage='science', execute=True, authorization=AUTHORIZE, debug_policy_confirmed=True)
    root = Path(spec['campaign_root'])
    row = next(r for r in graph() if r['task_id'] == name)
    parents = {p: completed(spec, p) for p in row['dependencies']}
    if not all(parents.values()):
        raise PermissionError('Segmented task parents are incomplete')
    claim = root / 'claims' / (name + '.claim')
    claim.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)  # Persistent claim: failures need explicit reconciliation.
    if row['kind'] != 'preflight':
        accepted = gate(spec)
        if row['kind'] in ('segment', 'reduce') and (accepted['gpu'] != gpu_identity()
                or accepted['environment'] != installed_environment()):
            raise PermissionError('Segment GPU/software differs from resume acceptance')
    if row['kind'] == 'preflight':
        result, paths = preflight(spec, source, device)
    elif row['kind'] == 'segment':
        result, paths = fit_segment(spec, source, row, deadline, device)
    elif row['kind'] == 'reduce':
        result, paths = reduce(spec, source, row, device)
    else:
        if row['kind'] == 'aggregate':
            rows = result_rows(spec, source)
            if any(r['validation'] is None for r in rows):
                raise PermissionError('Aggregate lacks a complete scientific fit')
            result = dict(rows=rows, recovery_reference='HLT_X1_CE=0%, OFFLINE_CE=100%; report subset only')
        else:
            result = dict(complete=True, continued_fits=3, reused_completed_fits=7)
        path = root / (name + '.json')
        write_immutable_json(path, artifact('RESULT', campaign_sha256=spec['content_hash'], **result, final_test_accessed=False))
        paths = [path]
    execution = root / 'execution' / (name + '.json')
    receipt = artifact('TASK_REPORT', campaign_sha256=spec['content_hash'], task_id=name,
        parents={p: r['content_hash'] for p, r in parents.items()}, result=result,
        original_scientific_source=source['source_commit'], executor_source=spec['source_commit'],
        outputs=[dict(path=str(p.relative_to(root)).replace('\\', '/'), sha256=sha256_file(p)) for p in paths + [execution]],
        final_test_accessed=False)
    write_immutable_json(root / 'tasks' / (name + '.json'), receipt)
    return receipt
