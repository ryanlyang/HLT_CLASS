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
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullstat import cache
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_attention.worker import execution, metadata, validation_arrays
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from . import campaign, banks, kd_banks, training
from .contracts import (artifact, validate, recipe, STAGES, teachers, ancestors,
                        EMBEDDING_EXPORTS, KD_EXPORTS)


def data(gate, role, coordinate, *, authenticated=None):
    if (role not in ('train', 'validation') or coordinate not in (*STAGES.values(), 'OFFLINE')
            or (coordinate == 'OFFLINE' and role != 'train')):
        raise PermissionError('Only registered ordinary-role rung views allowed')
    meta = metadata(gate) if authenticated is None else authenticated
    if meta[0]['content_hash'] != gate['anchor']['evidence']['prepared']:
        raise ValueError('Prepared cache binding differs')
    value = cache.load(gate['anchor']['evidence']['cache_root'], meta, role=role,
        coordinate=coordinate, max_ram_bytes=int(gate['memory_mb']*1024**2*.35))
    if len(value) != gate['counts'][role]:
        raise ValueError('Population differs')
    return value


def report_valid(report, gate, stage, bank_hash):
    order = teachers(stage)
    weight = 5./len(order)
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe(stage)['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'], bank=bank_hash))
    history = report['validation_history']
    if (report['node'] != gate['nodes'][stage] or report['scientific_fit'] is not True
            or report['technical'] is not False or report['zero_control'] is not False
            or report['passes'] != 20 or len(history) != report['passes']
            or [r['pass_number'] for r in history] != list(range(1, len(history)+1))
            or report['selected_state'] != 'native_backbone_only'
            or any(r['validation']['rows'] != gate['counts']['validation'] for r in history)
            or any(r['learning_rate'] != learning_rate(r['pass_number']) for r in history)
            or any(r['update'] != r['pass_number']*((gate['counts']['train']+127)//128) for r in history)):
        raise ValueError('FullStat training report schedule/population/status differs')
    for r in history:
        auxiliary = r['auxiliary_losses']
        weighted = r['weighted_auxiliary_losses']
        values = [r[k] for k in ('ce_loss', 'kd_loss', 'weighted_ce_loss', 'weighted_kd_loss',
                                 'weighted_auxiliary_loss', 'total_loss', 'train_seconds', 'validation_seconds')]
        if (set(auxiliary) != set(order) or set(weighted) != set(order)
                or not np.isfinite(values + list(auxiliary.values()) + list(weighted.values())).all()
                or min(values + list(auxiliary.values()) + list(weighted.values())) < -1e-6
                or any(weighted[t] != weight*auxiliary[t] for t in order)
                or r['weighted_auxiliary_loss'] != sum(weighted[t] for t in order)
                or r['weighted_ce_loss'] != .25*r['ce_loss']
                or r['weighted_kd_loss'] != .75*r['kd_loss']
                or r['total_loss'] != r['weighted_ce_loss']+r['weighted_kd_loss']+r['weighted_auxiliary_loss']):
            raise ValueError('Hybrid loss/time evidence differs')
    if (report['initialization'] != 'from_scratch_coordinate_paired'
            or report['selected_weights_restored'] is not True):
        raise ValueError('Matched initialization or selected state differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Validation selection differs')


def fit(directory, gate, stage, tr, val, bank, *, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report, state = training.train(tr, val, node=gate['nodes'][stage], bank=bank, device=device)
    report_valid(report, gate, stage, bank.manifest['content_hash'])
    if any(k.startswith(('predictor.', 'predictors.')) for k in state):
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
        raise MemoryError('Hybrid execution exceeds reused Oscar memory envelope')


def longest(bundle):
    lengths = np.concatenate([np.diff(block.offsets) for block in bundle.blocks])
    return int(np.argmax(lengths)), int(lengths.max())


def startup(spec, stage, tr, val, bank, *, device):
    """Three disposable maximum-support steps, no benchmark or technical epoch."""
    from .model import new_model
    from hlt_classification.jetclass2_delphes.runner import _forward
    banks.check_join(bank, tr, stage)
    model = new_model(spec['nodes'][stage]).to(device).eval()
    raw = tr.batch(np.arange(min(4, len(tr))))
    with torch.inference_mode():
        ordinary = _forward(model, raw, device, bf16=False)
        model.capture = True
        captured = _forward(model, raw, device, bf16=False)
        torch.testing.assert_close(ordinary, captured, rtol=0, atol=0)
    model.train()
    config = recipe(stage)['base']
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for name,p in model.named_parameters()
        if p.requires_grad and (name in excluded) == exclude],
        weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])
    index, maximum = longest(tr)
    ix = np.full(128, index, np.int64)
    raw = tr.batch(ix)
    gradient_norms = {}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
        loss, *_ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
            model.predictions, {t: torch.from_numpy(bank.targets[t][ix]).to(device) for t in teachers(stage)},
            stage=stage, probabilities=torch.from_numpy(bank.probabilities[ix]).to(device))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
            raise ValueError('Missing/nonfinite backbone or predictor gradients')
        gradient_norms = {t: float(torch.stack([p.grad.float().square().sum()
            for p in model.predictors[t].parameters()]).sum().sqrt()) for t in teachers(stage)}
        optimizer.step()
        model.predictions = None
    model.eval(); model.capture = False; model.predictions = None
    vi, vmax = longest(val)
    with torch.inference_mode():
        logits = _forward(model, val.batch(np.full(128, vi, np.int64)), device,
                          bf16=str(device).startswith('cuda'))
    if logits.shape != (128, 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid validation support witness')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    measured = memory_snapshot(torch)
    limits(spec, measured)
    return artifact('STARTUP', parents=dict(spec=spec['content_hash'], bank=bank.manifest['content_hash']),
        stage=stage, coordinate=STAGES[stage], source_commit=spec['source_commit'],
        environment=spec['reuse']['environment'], gpu=spec['reuse']['gpu'], memory=measured,
        train_maximum=maximum, validation_maximum=vmax, batch=128, stress_steps=3,
        technical_epochs=0, benchmark_steps=0, full_validation_pass=False,
        native_tap_parity=True, finite_gradients=True, head_gradient_norms=gradient_norms,
        scientific_initialization='fresh_after_discarding_startup_model')


def validate_startup(spec, stage, record, bank_hash):
    validate(record, 'STARTUP', dict(spec=spec['content_hash'], bank=bank_hash))
    expected = dict(stage=stage, coordinate=STAGES[stage], source_commit=spec['source_commit'],
        environment=spec['reuse']['environment'], gpu=spec['reuse']['gpu'], batch=128,
        stress_steps=3, technical_epochs=0, benchmark_steps=0, full_validation_pass=False,
        native_tap_parity=True, finite_gradients=True,
        scientific_initialization='fresh_after_discarding_startup_model',
        train_maximum=spec['reuse']['summaries']['train'][STAGES[stage]]['maximum'],
        validation_maximum=spec['reuse']['summaries']['validation'][STAGES[stage]]['maximum'])
    gradients = record['head_gradient_norms']
    if (any(record.get(k) != v for k,v in expected.items()) or set(gradients) != set(teachers(stage))
            or any(not math.isfinite(v) or v < 0 for v in gradients.values())):
        raise ValueError('Integrated startup evidence differs')
    limits(spec, record['memory'])


def exports(stage):
    """Only banks needed by descendants; first fit also owns original-target exports."""
    result = []
    if stage == 'UE000':
        result += [('M0HLT', 'embedding'), ('OFFLINE', 'kd')]
    if stage in EMBEDDING_EXPORTS:
        result.append((stage, 'embedding'))
    if stage in KD_EXPORTS:
        result.append((stage, 'kd'))
    return result


def outputs(stage):
    result = ['bundles/'+stage+'.json', 'train_'+stage+'_attempt/startup.json']
    result += ['train_'+stage+'/'+name for name in
        ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'receipt.json')]
    for teacher, kind in exports(stage):
        directory, payload = ('targets', 'targets.npz') if kind == 'embedding' else ('logit_targets', 'probabilities.npz')
        result += [directory+'/'+teacher+'/'+name for name in ('manifest.json', payload)]
    return result


def completed(spec, stage, *, accounting=True, seen=None):
    from .submission import accepted_jobs, successful
    if stage not in STAGES:
        raise ValueError('Unknown scientific stage')
    seen = {} if seen is None else seen
    if stage in seen:
        return seen[stage]
    root = Path(spec['root']); directory = root/('train_'+stage)
    path = directory/'task_receipt.json'
    if not path.is_file():
        return None
    parents = dict(spec=spec['content_hash'])
    for t in ancestors(stage):
        previous = completed(spec, t, accounting=accounting, seen=seen)
        if previous is None:
            raise ValueError('Missing completed hybrid ancestor '+t)
        parents[t] = previous['content_hash']
    record = load_json(path)
    validate(record, 'TASK', parents)
    if (record['task_id'] != 'train_'+stage or record['stage'] != stage
            or record['source_commit'] != spec['source_commit']
            or record['job_id'] != accepted_jobs(spec).get('train_'+stage)
            or [r['path'] for r in record['outputs']] != outputs(stage)):
        raise ValueError('Completed stage identity or exact publication differs')
    for ref in record['outputs']:
        validate_file_ref(ref, root=root)
    bundle = banks.bundle_record(spec, stage)
    if load_json(root/'bundles'/(stage+'.json')) != bundle or record['bank_hash'] != bundle['content_hash']:
        raise ValueError('Completed stage target lineage changed')
    validate_startup(spec, stage, load_json(root/('train_'+stage+'_attempt')/'startup.json'), bundle['content_hash'])
    report = validate_fit(directory, spec, stage, bundle['content_hash'])
    if record['report_hash'] != report['content_hash']:
        raise ValueError('Selected report identity differs')
    for teacher, kind in exports(stage):
        validate_export(spec, teacher, kind)
    if accounting:
        successful(spec, 'train_'+stage, record['job_id'])
    seen[stage] = record
    return record


def rejection(labels, probabilities, efficiency):
    from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES
    if efficiency not in (.3, .5):
        raise ValueError('Only registered R30/R50 reporting')
    p = np.asarray(probabilities, np.float64)
    result = {}
    for c, name in enumerate(CLASS_NAMES[1:], 1):
        den = p[:, c]+p[:, 0]
        score = np.divide(p[:, c], den, out=np.zeros(len(p)), where=den>0)
        signal = np.sort(score[labels == c])[::-1]
        qcd = score[labels == 0]
        if not len(signal) or not len(qcd):
            result[name] = None
            continue
        cut = signal[math.ceil(efficiency*len(signal))-1]
        passing = np.count_nonzero(qcd >= cut)
        result[name] = float(len(qcd)/passing) if passing else None
    return result


def results(spec):
    original = validation_arrays(spec)
    rows, pending, seen = [], [], {}
    sources = [(name, spec['references'][name], True) for name in
               ('M0HLT', 'OFFLINE', 'DIRECT_D000', 'COARSE_D000', 'U100_EMBED_W5', 'MULTI4_EMBED_W5')]
    for stage in STAGES:
        if completed(spec, stage, seen=seen) is None:
            pending.append(stage)
        else:
            root = Path(spec['root'])/('train_'+stage)
            sources.append((stage, dict(report=file_ref(root/'training_report.json'),
                predictions=file_ref(root/'validation_predictions.npz')), False))
    for name, refs, historical in sources:
        report = load_json(validate_file_ref(refs['report']))
        a = load_npz_arrays(validate_file_ref(refs['predictions']))
        if (not np.array_equal(a['identities'], original['identities'])
                or not np.array_equal(a['labels'], original['labels'])
                or evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']):
            raise ValueError('Selected reference/results population or metrics differ')
        r50 = rejection(a['labels'], a['probabilities'], .5)
        if any(v != report['validation']['per_class'][c]['rejection_at_50pct'] for c,v in r50.items()):
            raise ValueError('R50 rank/tie convention differs')
        rows.append(dict(node_id=name, selected_epoch=report['selected_pass'], validation=report['validation'],
            rejection_30=rejection(a['labels'], a['probabilities'], .3), rejection_50=r50,
            historical=historical, input='richer_input_oracle' if name in ('OFFLINE','UE000','UE100','DE066','DE033') else 'native_HLT'))
    by_name = {r['node_id']: r for r in rows}
    for row in rows:
        row['recovery'] = recovery(row['validation'], by_name['M0HLT']['validation'], by_name['OFFLINE']['validation'])
        row['rejection_recovery'] = {}
        for key in ('rejection_30','rejection_50'):
            out = row['rejection_recovery'][key] = {}
            for c, v in row[key].items():
                lo, hi = by_name['M0HLT'][key][c], by_name['OFFLINE'][key][c]
                out[c] = None if v is None or lo is None or hi is None or hi == lo else 100*(v-lo)/(hi-lo)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        single_seed=True, adaptive_validation=True, fresh_matched_control=False,
        recovery_baseline='historical M0HLT', same_auc_selected_checkpoints_for_R30_R50=True)


def run(spec, task):
    from .submission import accepted_jobs
    if task not in {t['task_id'] for t in spec['tasks']}:
        raise PermissionError('Only five registered scientific fits; no preflight or test operation')
    campaign.check(spec)
    job = execution(spec, spec['reuse'])
    if job != accepted_jobs(spec).get(task):
        raise PermissionError('Worker is not the exact submitted job')
    stage = task.removeprefix('train_')
    seen, parents = {}, dict(spec=spec['content_hash'])
    for t in ancestors(stage):
        previous = completed(spec, t, seen=seen)
        if previous is None:
            raise ValueError('Required hybrid ancestor incomplete: '+t)
        parents[t] = previous['content_hash']
    root = Path(spec['root']); attempt = root/(task+'_attempt')
    attempt.mkdir(exist_ok=False)
    campaign.storage(spec)
    meta = metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Inherited coordinate support/resource metadata differs')
    log = PhaseLog(attempt, lambda: memory_snapshot(torch))
    if stage == 'UE000':
        # Load/export/release each original view sequentially, before rung caches.
        for teacher_id, coordinate, kind in [('M0HLT', 'D000', 'embedding'), ('OFFLINE', 'OFFLINE', 'kd')]:
            with log.phase('export_original_TRAIN_'+teacher_id+'_'+kind):
                original = data(spec, 'train', coordinate, authenticated=meta)
                export_target(spec, teacher_id, kind, original, device='cuda')
                del original
                gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_registered_rung_and_ancestral_TRAIN_targets'):
        tr = data(spec, 'train', STAGES[stage], authenticated=meta)
        val = data(spec, 'validation', STAGES[stage], authenticated=meta)
        bank = banks.publish(spec, stage, tr)
    with log.phase('three_integrated_startup_steps_no_preflight'):
        record = startup(spec, stage, tr, val, bank, device='cuda')
        validate_startup(spec, stage, record, bank.manifest['content_hash'])
        write_immutable_json(attempt/'startup.json', record)
    gc.collect(); torch.cuda.empty_cache()
    with log.phase('scientific_twenty_epoch_fit_'+stage):
        fit(root/task, spec, stage, tr, val, bank, device='cuda')
    report = validate_fit(root/task, spec, stage, bank.manifest['content_hash'])
    bank_hash = bank.manifest['content_hash']
    del bank, val
    gc.collect(); torch.cuda.empty_cache()
    for teacher_id, kind in exports(stage):
        if teacher_id == stage:
            with log.phase('export_selected_native_TRAIN_'+stage+'_'+kind):
                export_target(spec, stage, kind, tr, device='cuda')
    limits(spec, memory_snapshot(torch))
    campaign.check(spec)
    if execution(spec, spec['reuse']) != job:
        raise PermissionError('Allocation/environment changed')
    record = artifact('TASK', parents=parents, task_id=task, stage=stage, job_id=job,
        source_commit=spec['source_commit'], report_hash=report['content_hash'], bank_hash=bank_hash,
        outputs=[file_ref(root/p, root=root) for p in outputs(stage)])
    write_immutable_json(root/task/'task_receipt.json', record)
    return completed(spec, stage, accounting=False)


def validate_export(spec, teacher, kind):
    from .contracts import TEACHERS
    root = Path(spec['root'])
    refs = banks.refs_for(spec, teacher)
    directory, payload, contract, module = (('targets', 'targets.npz', 'BANK', banks)
        if kind == 'embedding' else ('logit_targets', 'probabilities.npz', 'KD_BANK', kd_banks))
    output = root/directory/teacher
    m = load_json(output/'manifest.json')
    validate(m, contract, module.parents(spec['anchor']['evidence']['prepared'], refs, teacher))
    expected = dict(role='train', teacher=teacher, coordinate=TEACHERS[teacher], rows=spec['counts']['train'],
        checkpoint=refs['checkpoint'], report=refs['report'])
    expected.update(dict(width=128) if kind == 'embedding' else dict(temperature=2., class_count=11))
    if (any(m.get(k) != v for k, v in expected.items())
            or m['payload'] != file_ref(output/payload, root=output)):
        raise ValueError('Outgoing selected-native bank differs')


def export_target(spec, teacher_id, kind, tr, *, device):
    refs = banks.refs_for(spec, teacher_id)
    native = DelphesParticleTransformer()
    native.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']), map_location='cpu', weights_only=True), strict=True)
    root = Path(spec['root'])
    if kind == 'embedding':
        banks.export(root/'targets'/teacher_id, native, tr, refs=refs, device=device, teacher_id=teacher_id)
        # Re-read exact joined arrays now; descendants also authenticate on loading.
        banks.arrays(banks.manifest_ref(spec, teacher_id), tr, refs=refs, teacher_id=teacher_id)
    elif kind == 'kd':
        kd_banks.export(root/'logit_targets'/teacher_id, native, tr, refs=refs, device=device, teacher_id=teacher_id)
    else:
        raise ValueError('Unregistered export kind')
    validate_export(spec, teacher_id, kind)
