"""Scientific rung workers with inherited acceptance and short integrated checks."""
from pathlib import Path
import gc
import os
import math
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullstat import cache
from hlt_classification.luka_fullstat_attention.worker import metadata, validation_arrays
from .execution import execution
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from . import campaign, banks, training, targets
from .contracts import artifact, validate, recipe, STAGES, teachers


def data(gate, role, coordinate, *, authenticated=None):
    if role not in ('train', 'validation') or coordinate != 'D000':
        raise PermissionError('Only registered ordinary-role rung views allowed')
    meta = metadata(gate) if authenticated is None else authenticated
    if meta[0]['content_hash'] != gate['anchor']['evidence']['prepared']:
        raise ValueError('Prepared cache binding differs')
    value = cache.load(gate['anchor']['evidence']['cache_root'], meta, role=role,
        coordinate=coordinate, max_ram_bytes=int(gate['memory_mb']*1024**2*.35))
    if len(value) != gate['counts'][role]:
        raise ValueError('Population differs')
    return value


def target_completed(spec, *, accounting=True):
    from .submission import accepted_jobs, successful
    path = Path(spec['root'])/'targets/task_receipt.json'
    if not path.is_file():
        return None
    r = load_json(path)
    validate(r, 'TARGET_TASK', dict(spec=spec['content_hash']))
    if (r['task_id'] != 'targets' or r['job_id'] != accepted_jobs(spec).get('targets')
            or r['source_commit'] != spec['source_commit']
            or r['manifest'] != file_ref(Path(spec['target_root'])/'manifest.json')):
        raise ValueError('Target task lineage or exact job binding differs')
    m, _ = banks.validate_bank(spec['target_root'], prepared=spec['anchor']['evidence']['prepared'],
        source_hash=spec['content_hash'], rows=spec['counts']['train'], tokens=spec['resources']['tokens'])
    if r['bank_hash'] != m['content_hash']:
        raise ValueError('Target task bank hash differs')
    if accounting:
        successful(spec, 'targets', r['job_id'])
    return r


def outputs(stage):
    return ['bundles/'+stage+'.json', 'train_'+stage+'_attempt/startup.json']+[
        'train_'+stage+'/'+f for f in ('selected.pt', 'training_report.json', 'validation_predictions.npz',
                                      'receipt.json', 'completed_native.pt', 'completed_training.json')]


def completed(spec, stage, *, accounting=True):
    from .submission import accepted_jobs, successful
    if stage not in STAGES:
        raise ValueError('Unknown neighborhood stage')
    root = Path(spec['root']); task = 'train_'+stage
    path = root/task/'task_receipt.json'
    if not path.is_file():
        return None
    prior = target_completed(spec, accounting=accounting)
    if prior is None:
        raise ValueError('Missing target task receipt')
    r = load_json(path)
    validate(r, 'TASK', dict(spec=spec['content_hash'], targets=prior['content_hash']))
    if (r['stage'] != stage or r['task_id'] != task or r['job_id'] != accepted_jobs(spec).get(task)
            or r['source_commit'] != spec['source_commit'] or [a['path'] for a in r['outputs']] != outputs(stage)):
        raise ValueError('Completed fit identity or outputs differ')
    for ref in r['outputs']:
        validate_file_ref(ref, root=root)
    bundle = banks.bundle_record(spec, stage)
    if r['bank_hash'] != bundle['content_hash'] or load_json(root/'bundles'/(stage+'.json')) != bundle:
        raise ValueError('Completed target bindings differ')
    validate_startup(spec, stage, load_json(root/(task+'_attempt')/'startup.json'), bundle['content_hash'])
    report = validate_fit(root/task, spec, stage, bundle['content_hash'])
    if r['report_hash'] != report['content_hash']:
        raise ValueError('Completed report differs')
    if accounting:
        successful(spec, task, r['job_id'])
    return r


def run(spec, task, *, device='cuda'):
    from .submission import accepted_jobs
    started = time.monotonic()
    if task not in {t['task_id'] for t in spec['tasks']}:
        raise PermissionError('Only one CPU target build and one fresh fit are authorized')
    campaign.check(spec)
    job = execution(spec, spec['reuse'], cpu=task == 'targets')
    if job != accepted_jobs(spec).get(task):
        raise PermissionError('Worker is not the exact submitted job')
    root = Path(spec['root'])
    if task == 'targets':
        (root/task).mkdir(exist_ok=False)
        campaign.storage(spec, before_targets=True)
        meta = metadata(spec)
        m = banks.build(spec['target_root'], targets.chunks(spec, meta),
            prepared=meta[0]['content_hash'], source_hash=spec['content_hash'],
            rows=spec['counts']['train'], tokens=spec['resources']['tokens'],
            available_bytes=spec['available_target_bytes'])
        campaign.check(spec)
        if execution(spec, spec['reuse'], cpu=True) != job:
            raise PermissionError('Target allocation changed')
        r = artifact('TARGET_TASK', parents=dict(spec=spec['content_hash']), task_id=task, job_id=job,
            source_commit=spec['source_commit'], bank_hash=m['content_hash'],
            manifest=file_ref(Path(spec['target_root'])/'manifest.json'), seconds=time.monotonic()-started)
        write_immutable_json(root/task/'task_receipt.json', r)
        return target_completed(spec, accounting=False)
    prior = target_completed(spec)
    if prior is None:
        raise ValueError('Target build must complete successfully first')
    stage = task.removeprefix('train_')
    attempt = root/(task+'_attempt'); attempt.mkdir(exist_ok=False)
    meta = metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Inherited support/resource metadata differs')
    log = PhaseLog(attempt, lambda: memory_snapshot(torch))
    with log.phase('load_native_HLT_and_frozen_TRAIN_targets'):
        tr = data(spec, 'train', 'D000', authenticated=meta)
        val = data(spec, 'validation', 'D000', authenticated=meta)
        bank = banks.publish(spec, stage, tr)
    with log.phase('integrated_technical_checks_no_preflight_job'):
        record = startup(spec, stage, tr, val, bank, device=device, preparation_seconds=time.monotonic()-started)
        validate_startup(spec, stage, record, bank.manifest['content_hash'])
        write_immutable_json(attempt/'startup.json', record)
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    with log.phase('scientific_twenty_epoch_fit'):
        fit(root/task, spec, stage, tr, val, bank, device=device)
    report = validate_fit(root/task, spec, stage, bank.manifest['content_hash'])
    limits(spec, memory_snapshot(torch))
    campaign.check(spec)
    if execution(spec, spec['reuse']) != job:
        raise PermissionError('Training allocation changed')
    r = artifact('TASK', parents=dict(spec=spec['content_hash'], targets=prior['content_hash']),
        task_id=task, stage=stage, job_id=job, source_commit=spec['source_commit'],
        report_hash=report['content_hash'], bank_hash=bank.manifest['content_hash'],
        outputs=[file_ref(root/p, root=root) for p in outputs(stage)])
    write_immutable_json(root/task/'task_receipt.json', r)
    return completed(spec, stage, accounting=False)


def results(spec):
    from hlt_classification.luka_fullstat_fine_tail.worker import rejection
    original = validation_arrays(spec)
    sources = [(name, refs, True) for name, refs in spec['references'].items()]
    pending, rows = [], []
    for stage in STAGES:
        if completed(spec, stage) is None:
            pending.append(stage)
        else:
            root = Path(spec['root'])/('train_'+stage)
            sources.append((stage, dict(report=file_ref(root/'training_report.json'),
                predictions=file_ref(root/'validation_predictions.npz')), False))
    for name, refs, historical in sources:
        report = load_json(validate_file_ref(refs['report']))
        a = load_npz_arrays(validate_file_ref(refs['predictions']))
        if (not np.array_equal(a['identities'], original['identities']) or not np.array_equal(a['labels'], original['labels'])
                or evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']):
            raise ValueError('Saved selected metrics or population differ')
        r50 = rejection(a['labels'], a['probabilities'], .5)
        if any(v != report['validation']['per_class'][c]['rejection_at_50pct'] for c, v in r50.items()):
            raise ValueError('R50 convention differs')
        rows.append(dict(node_id=name, historical=historical, selected_epoch=report['selected_pass'],
            validation=report['validation'], rejection_50=r50, rejection_30=rejection(a['labels'], a['probabilities'], .3)))
    by_name = {r['node_id']:r for r in rows}
    for row in rows:
        row['recovery'] = recovery(row['validation'], by_name['M0HLT']['validation'], by_name['OFFLINE']['validation'])
        row['rejection_recovery'] = {}
        for key in ('rejection_30', 'rejection_50'):
            out = row['rejection_recovery'][key] = {}
            for c, v in row[key].items():
                lo, hi = by_name['M0HLT'][key][c], by_name['OFFLINE'][key][c]
                out[c] = None if v is None or lo is None or hi is None or hi == lo else 100*(v-lo)/(hi-lo)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        single_seed=True, adaptive_validation=True, fresh_matched_control=False,
        historical_comparison='previous DE000 differs in K/E; not a neighborhood-only ablation',
        same_auc_selected_checkpoints_for_R30_R50=True)


def report_valid(report, gate, stage, bank_hash):
    training.validate_report(report, gate, stage, bank_hash)


def fit(directory, gate, stage, tr, val, bank, *, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    def snapshot(report, state):
        # Preserve the completed fit BEFORE validation/publication checks.
        save_state(directory/'completed_native.pt', state)
        write_immutable_json(directory/'completed_training.json', report)
    report, state = training.train(tr, val, node=gate['nodes'][stage], bank=bank, device=device, snapshot=snapshot)
    report_valid(report, gate, stage, bank.manifest['content_hash'])
    if any(k.startswith(('predictor.', 'predictors.', 'logit_heads.', 'neighborhood_head.')) for k in state):
        raise ValueError('Predictor leaked into deployable checkpoint')
    save_state(directory/'selected.pt', state)
    restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('Selected checkpoint roundtrip differs')
    native = DelphesParticleTransformer().to(device)
    native.load_state_dict(restored, strict=True)
    arrays = dict(identities=val.identities, labels=val.labels,
                  probabilities=predict(native, val, device=device, batch_size=128))
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Head-free native validation readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json', report)
    result = artifact('FIT', parents=dict(gate=gate['content_hash'], bank=bank.manifest['content_hash']),
        report_hash=report['content_hash'], technical=False,
        checkpoint_probability_roundtrip=True, head_free=True,
        outputs=[file_ref(directory/f, root=directory) for f in
                 ('selected.pt', 'training_report.json', 'validation_predictions.npz')])
    write_immutable_json(directory/'receipt.json', result)
    return result


def validate_fit(directory, gate, stage, bank_hash):
    directory = Path(directory)
    receipt = load_json(directory/'receipt.json')
    validate(receipt, 'FIT', dict(gate=gate['content_hash'], bank=bank_hash))
    if (receipt['technical'] is not False or receipt['head_free'] is not True
            or receipt['checkpoint_probability_roundtrip'] is not True
            or [ref['path'] for ref in receipt['outputs']] !=
               ['selected.pt', 'training_report.json', 'validation_predictions.npz']):
        raise ValueError('Fit publication differs')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=directory)
    report = load_json(directory/'training_report.json')
    report_valid(report, gate, stage, bank_hash)
    a = load_npz_arrays(directory/'validation_predictions.npz')
    original = validation_arrays(gate)
    if (receipt['report_hash'] != report['content_hash']
            or set(a) != {'identities', 'labels', 'probabilities'}
            or not np.array_equal(a['identities'], original['identities'])
            or not np.array_equal(a['labels'], original['labels'])
            or evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']):
        raise ValueError('Saved selected metrics/population differ')
    return report


def limits(spec, measured):
    values = [measured[k] for k in ('peak_rss_bytes', 'gpu_peak_bytes')]
    if (any(type(v) is not int or v <= 0 for v in values)
            or measured['peak_rss_bytes']+8*2**30 >= .85*spec['memory_mb']*1024**2
            or measured['gpu_peak_bytes'] >= .9*spec['reuse']['gpu']['total_memory_bytes']):
        raise MemoryError('Split-head execution exceeds reused Oscar memory envelope')


def longest(bundle):
    lengths = np.concatenate([np.diff(block.offsets) for block in bundle.blocks])
    return int(np.argmax(lengths)), int(lengths.max())


def startup(spec, stage, tr, val, bank, *, device, preparation_seconds=0.):
    """Three disposable max-support updates, native parity and CE-only head."""
    from .model import new_model
    from hlt_classification.jetclass2_delphes.runner import _forward
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    banks.check_join(bank, tr, stage)
    node = spec['nodes'][stage]
    torch.manual_seed(node['initialization_seed'])
    native = DelphesParticleTransformer().to(device).eval()
    model = new_model(node).to(device).eval()
    model.capture = True
    raw = tr.batch(np.arange(min(4, len(tr))))
    with parity_backend(device), torch.no_grad():
        for bf16 in (False, True):
            ordinary = _forward(native, raw, device, bf16=bf16)
            captured = _forward(model, raw, device, bf16=bf16)
            torch.testing.assert_close(ordinary, captured, rtol=0, atol=0)
    del native, ordinary, captured
    model.train()
    optimizer = training.optimizer(model, recipe(stage)['base'])
    index, maximum = longest(tr)
    ix = np.full(128, index, np.int64)
    raw = tr.batch(ix)
    gradient_norms = {}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, ce, *_ = training.batch_loss(model, raw, bank, ix, stage, device)
        head = tuple(model.mod.fc.parameters())
        expected = torch.autograd.grad(.5*ce, head, retain_graph=True)
        loss.backward()
        training.finite_gradients(model)
        for p, reference in zip(head, expected):
            torch.testing.assert_close(p.grad, reference, rtol=0, atol=0)
        if any(p.grad is None for p in model.parameters() if p.requires_grad):
            raise ValueError('Disconnected backbone or auxiliary head')
        gradient_norms = {prefix+t: float(torch.stack([p.grad.float().square().sum()
            for p in heads[t].parameters()]).sum().sqrt())
            for prefix, heads in [('embedding_', model.predictors), ('logit_', model.logit_heads)]
            for t in teachers(stage)}
        gradient_norms['neighborhood'] = float(torch.stack([
            p.grad.float().square().sum() for p in model.neighborhood_head.parameters()]).sum().sqrt())
        optimizer.step()
        model.predictions = model.auxiliary_logits = model.neighborhood_predictions = None
        del loss, ce, head, expected, p, reference
    model.eval(); model.capture = False
    vi, vmax = longest(val)
    with torch.inference_mode():
        logits = _forward(model, val.batch(np.full(128, vi, np.int64)), device,
                          bf16=str(device).startswith('cuda'))
    if logits.shape != (128, 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid validation support witness')
    del logits
    timing = benchmark(spec, stage, model, optimizer, tr, val, bank, device, preparation_seconds)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    measured = memory_snapshot(torch)
    limits(spec, measured)
    return artifact('STARTUP', parents=dict(spec=spec['content_hash'], bank=bank.manifest['content_hash']),
        stage=stage, coordinate=STAGES[stage], source_commit=spec['source_commit'],
        environment=spec['reuse']['environment'], gpu=spec['reuse']['gpu'], memory=measured,
        train_maximum=maximum, validation_maximum=vmax, batch=128, stress_steps=3,
        technical_epochs=0, benchmark_steps=32, full_validation_pass=False,
        relational=False, neighborhood_weight=10., timing=timing,
        native_tap_parity=True, native_fp32_bf16_parity=True, main_head_gradient_CE_only=True,
        finite_gradients=True, head_gradient_norms=gradient_norms,
        scientific_initialization='fresh_after_discarding_startup_model')


def validate_startup(spec, stage, record, bank_hash):
    validate(record, 'STARTUP', dict(spec=spec['content_hash'], bank=bank_hash))
    expected = dict(stage=stage, coordinate=STAGES[stage], source_commit=spec['source_commit'],
        environment=spec['reuse']['environment'], gpu=spec['reuse']['gpu'], batch=128,
        stress_steps=3, technical_epochs=0, benchmark_steps=32, full_validation_pass=False,
        relational=False, neighborhood_weight=10.,
        native_tap_parity=True, native_fp32_bf16_parity=True, main_head_gradient_CE_only=True, finite_gradients=True,
        scientific_initialization='fresh_after_discarding_startup_model',
        train_maximum=spec['reuse']['summaries']['train']['D000']['maximum'],
        validation_maximum=spec['reuse']['summaries']['validation']['D000']['maximum'])
    gradients = record['head_gradient_norms']
    if (any(record.get(k) != v for k,v in expected.items()) or set(gradients) != ({'neighborhood'} | {p+t for p in ('embedding_', 'logit_') for t in teachers(stage)})
            or any(not math.isfinite(v) or v < 0 for v in gradients.values())):
        raise ValueError('Integrated startup evidence differs')
    limits(spec, record['memory'])
    validate_timing(spec, stage, record['timing'])


def timing_record(spec, stage, train_seconds_per_update, validation_seconds_per_batch, preparation_seconds=0.):
    numbers = (train_seconds_per_update, validation_seconds_per_batch)
    if (any(not math.isfinite(x) or x <= 0 for x in numbers)
            or not math.isfinite(preparation_seconds) or preparation_seconds < 0):
        raise ValueError('Invalid integrated full-step timings')
    cfg = spec['startup']
    minutes = math.ceil(cfg['timing_margin']*20*(
        math.ceil(spec['counts']['train']/128)*train_seconds_per_update
        + math.ceil(spec['counts']['validation']/128)*validation_seconds_per_batch)/60
        + cfg['timing_allowance_minutes'] + preparation_seconds/60)
    if minutes > spec['task_minutes'][stage]:
        raise ValueError(f'Measured 20-epoch estimate {minutes}min exceeds allocation '
            f'{spec["task_minutes"][stage]}min (train={train_seconds_per_update:.6f}s/update, '
            f'validation={validation_seconds_per_batch:.6f}s/batch); preserve evidence, no truncated fit')
    return dict(train_seconds_per_update=train_seconds_per_update,
        validation_seconds_per_batch=validation_seconds_per_batch,
        warmup_steps=4, benchmark_steps=32, validation_steps=16,
        conservative_minutes=minutes, preparation_seconds=preparation_seconds, full_epoch_measured=False,
        complete_validation_measured=False, scientific_weights_discarded=True)


def validate_timing(spec, stage, value):
    expected = timing_record(spec, stage, value['train_seconds_per_update'],
                             value['validation_seconds_per_batch'], value['preparation_seconds'])
    if value != expected:
        raise ValueError('Integrated timing evidence differs')


def benchmark(spec, stage, model, optimizer, tr, val, bank, device, preparation_seconds):
    """Bounded full-objective timings; no reference-particle/teacher forwards."""
    from hlt_classification.jetclass2_delphes.runner import _forward
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    rng = np.random.default_rng(271828)
    model.train(); model.capture = True
    sync(); started = time.monotonic()
    for step in range(36):
        if step == 4:
            sync(); started = time.monotonic()
        ix = rng.integers(0, len(tr), size=128, dtype=np.int64)
        optimizer.zero_grad(set_to_none=True)
        loss, *_ = training.batch_loss(model, tr.batch(ix), bank, ix, stage, device)
        loss.backward(); training.finite_gradients(model); optimizer.step()
        model.predictions = model.auxiliary_logits = model.neighborhood_predictions = None
        del loss
    sync(); train_seconds = (time.monotonic()-started)/32
    model.eval(); model.capture = False
    sync(); started = time.monotonic()
    with torch.inference_mode():
        for step in range(16):
            ix = rng.integers(0, len(val), size=128, dtype=np.int64)
            logits = _forward(model, val.batch(ix), device, bf16=str(device).startswith('cuda'))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite integrated validation timing witness')
    sync(); validation_seconds = (time.monotonic()-started)/16
    value = timing_record(spec, stage, train_seconds, validation_seconds, preparation_seconds)
    print(f'FULLSTAT_NEIGHBOR integrated timing stage={stage} train_s_per_update={train_seconds:.5f} '
          f'validation_s_per_batch={validation_seconds:.5f} '
          f'conservative_minutes={value["conservative_minutes"]}; no full epoch measured', flush=True)
    return value
