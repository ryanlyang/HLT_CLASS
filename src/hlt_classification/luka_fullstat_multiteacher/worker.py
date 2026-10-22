"""Four frozen targets, measured setup admission and one HLT-only scientific fit."""
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
from . import campaign, banks, training
from .contracts import artifact, validate, recipe, TEACHERS, WEIGHT


def data(gate, role, coordinate, *, authenticated=None):
    if not ((role == 'train' and coordinate in ('D000', *TEACHERS.values())) or (role == 'validation' and coordinate == 'D000')):
        raise PermissionError('Only HLT validation and registered TRAIN views allowed')
    meta = metadata(gate) if authenticated is None else authenticated
    if meta[0]['content_hash'] != gate['anchor']['evidence']['prepared']:
        raise ValueError('Prepared cache binding differs')
    value = cache.load(gate['anchor']['evidence']['cache_root'], meta, role=role,
        coordinate=coordinate, max_ram_bytes=int(gate['memory_mb']*1024**2*.35))
    if len(value) != gate['counts'][role]:
        raise ValueError('Population differs')
    return value


def report_valid(report, gate, bank_hash, *, technical=False):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe()['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'], bank=bank_hash))
    history = report['validation_history']
    if (report['node'] != gate['node'] or report['scientific_fit'] is not (not technical)
            or report['technical'] is not technical or report['zero_control'] is not False
            or report['passes'] != (1 if technical else 20) or len(history) != report['passes']
            or [r['pass_number'] for r in history] != list(range(1, len(history)+1))
            or report['selected_state'] != 'native_backbone_only'
            or any(r['validation']['rows'] != gate['counts']['validation'] for r in history)
            or any(r['learning_rate'] != learning_rate(r['pass_number']) for r in history)
            or any(r['update'] != r['pass_number']*((gate['counts']['train']+127)//128) for r in history)):
        raise ValueError('FullStat training report schedule/population/status differs')
    for r in history:
        auxiliary = r['auxiliary_losses']
        weighted = r['weighted_auxiliary_losses']
        values = [r[k] for k in ('ce_loss', 'weighted_auxiliary_loss', 'total_loss', 'train_seconds', 'validation_seconds')]
        if (set(auxiliary) != set(TEACHERS) or set(weighted) != set(TEACHERS)
                or not np.isfinite(values + list(auxiliary.values()) + list(weighted.values())).all()
                or min(values + list(auxiliary.values()) + list(weighted.values())) < 0
                or any(weighted[t] != WEIGHT*auxiliary[t] for t in TEACHERS)
                or r['weighted_auxiliary_loss'] != sum(weighted[t] for t in TEACHERS)
                or r['total_loss'] != r['ce_loss']+r['weighted_auxiliary_loss']):
            raise ValueError('Four-teacher loss/time evidence differs')
    if (report['initialization'] != 'from_scratch_matched_M0HLT'
            or report['selected_weights_restored'] is not True):
        raise ValueError('Matched initialization or selected state differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Validation selection differs')


def fit(directory, gate, tr, val, bank, *, device, technical=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report, state = training.train(tr, val, node=gate['node'], bank=bank, device=device,
                                   technical=technical)
    report_valid(report, gate, bank.manifest['content_hash'], technical=technical)
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
        report_hash=report['content_hash'], technical=technical,
        checkpoint_probability_roundtrip=True, head_free=True,
        outputs=[file_ref(directory/f, root=directory) for f in
                 ('selected.pt', 'training_report.json', 'validation_predictions.npz')])
    write_immutable_json(directory/'receipt.json', result)
    return result


def validate_fit(directory, gate, bank_hash, *, technical=False):
    directory = Path(directory)
    receipt = load_json(directory/'receipt.json')
    validate(receipt, 'FIT', dict(gate=gate['content_hash'], bank=bank_hash))
    if (receipt['technical'] is not technical or receipt['head_free'] is not True
            or receipt['checkpoint_probability_roundtrip'] is not True
            or [ref['path'] for ref in receipt['outputs']] !=
               ['selected.pt', 'training_report.json', 'validation_predictions.npz']):
        raise ValueError('Fit publication differs')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=directory)
    report = load_json(directory/'training_report.json')
    report_valid(report, gate, bank_hash, technical=technical)
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
        raise MemoryError('Four-teacher setup exceeds Oscar memory envelope')


def longest(bundle):
    lengths = np.concatenate([np.diff(block.offsets) for block in bundle.blocks])
    return int(np.argmax(lengths)), int(lengths.max())


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def witnesses(spec, tr, val, bank, *, device):
    """Discarded probes: max-support stress plus sampled real training updates."""
    from .model import new_model
    from hlt_classification.jetclass2_delphes.runner import _forward
    banks.check_join(bank, tr)
    model = new_model(spec['node']).to(device)
    raw = tr.batch(np.arange(min(4, len(tr))))
    model.eval()
    with torch.inference_mode():
        native = _forward(model, raw, device, bf16=False)
        model.capture = True
        captured = _forward(model, raw, device, bf16=False)
        torch.testing.assert_close(native, captured, rtol=0, atol=0)
    model.train()
    config = recipe()['base']
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for name,p in model.named_parameters()
        if p.requires_grad and (name in excluded) == exclude],
        weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'],
        betas=tuple(config['betas']), eps=config['eps'])
    steps = math.ceil(len(tr)/128)

    def step(ix, update):
        raw = tr.batch(ix)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(update/steps)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
        loss, *_ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
            model.predictions, {t:torch.from_numpy(bank.targets[t][ix]).to(device) for t in TEACHERS})
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
            raise ValueError('Missing/nonfinite shared encoder or auxiliary-head gradients')
        optimizer.step()
        model.predictions = None

    index, maximum = longest(tr)
    for update in range(1, 4):
        step(np.full(128, index, np.int64), update)
    model.eval(); model.capture = False; model.predictions = None
    vi, vmax = longest(val)
    with torch.inference_mode():
        logits = _forward(model, val.batch(np.full(128, vi, np.int64)), device,
                          bf16=str(device).startswith('cuda'))
    if logits.shape != (128,11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid maximum-support validation output')
    model.train(); model.capture = True
    order = np.random.default_rng(np.random.SeedSequence([spec['node']['sampler_seed'],1])).permutation(len(tr))
    measured_steps = min(spec['acceptance']['benchmark_steps'], steps)
    synchronize(device)
    start = time.monotonic()
    for update in range(1, measured_steps+1):
        step(order[(update-1)*128:update*128], update)
        if update % 64 == 0:
            synchronize(device)
            print(f'MULTI4 benchmark {update}/{measured_steps}: {(time.monotonic()-start)/update:.5f} seconds/update', flush=True)
    synchronize(device)
    train_seconds = time.monotonic()-start
    model.capture = False; model.predictions = None
    synchronize(device)
    start = time.monotonic()
    probabilities = predict(model, val, device=device, batch_size=128)
    evaluate_probabilities(val.labels, probabilities)  # finite/shape checks, no quality gate
    synchronize(device)
    validation_seconds = time.monotonic()-start
    estimated = math.ceil(spec['acceptance']['margin']*20*(
        steps*train_seconds/measured_steps + validation_seconds)/60 +
        spec['acceptance']['fixed_allowance_minutes'])
    print(f'MULTI4 estimated epoch training={steps*train_seconds/measured_steps/60:.2f}min; '
          f'full validation={validation_seconds/60:.2f}min; request={estimated}min', flush=True)
    return dict(train_maximum=maximum, validation_maximum=vmax, batch=128, stress_steps=3,
        finite_gradients=True, native_tap_parity=True, benchmark_steps=measured_steps,
        benchmark_seconds=train_seconds, full_validation_seconds=validation_seconds,
        full_validation_rows=len(val), full_epoch_training_measured=False,
        estimated_train_minutes=estimated)


def setup(spec, *, device='cuda'):
    from hlt_classification.luka_embedding_aux.model import head_input
    from hlt_classification.jetclass2_delphes.runner import _forward
    meta = metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Prepared resource support differs from authenticated parent')
    root = Path(spec['root'])
    directory = root/'setup'
    directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    maxima = {}
    for name in ('COARSE_D066', 'COARSE_D033'):
        with log.phase('export_frozen_'+name+'_TRAIN_targets'):
            value = data(spec, 'train', TEACHERS[name], authenticated=meta)
            refs = spec['teachers'][name]['refs']
            teacher = DelphesParticleTransformer().to(device)
            teacher.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']),
                map_location='cpu', weights_only=True), strict=True)
            teacher.eval().requires_grad_(False)
            index, maxima[name] = longest(value)
            raw = value.batch(np.full(128, index, np.int64))
            tensors = {k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
            with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
                for _ in range(3):
                    logits, vector = head_input(teacher, **tensors)
                    if vector.shape != (128,128) or not torch.isfinite(vector).all() or not torch.isfinite(logits).all():
                        raise ValueError('Invalid maximum-support teacher export')
                torch.testing.assert_close(logits, _forward(teacher, raw, device, bf16=False), rtol=0, atol=0)
            limits(spec, memory_snapshot(torch))
            banks.export(root/'targets'/name, teacher, value, refs=refs, device=device, teacher_id=name)
            del value, teacher, raw, tensors, logits, vector
            clear()
    with log.phase('load_HLT_and_authenticate_four_TRAIN_banks'):
        tr = data(spec, 'train', 'D000', authenticated=meta)
        val = data(spec, 'validation', 'D000', authenticated=meta)
        original = validation_arrays(spec)
        if not np.array_equal(val.identities, original['identities']) or not np.array_equal(val.labels, original['labels']):
            raise ValueError('Validation population differs')
        bank = banks.publish(spec, tr)
    with log.phase('four_head_stress_benchmark_and_full_validation'):
        probe = witnesses(spec, tr, val, bank, device=device)
    memory = memory_snapshot(torch)
    limits(spec, memory)
    if probe['estimated_train_minutes'] > spec['train_minutes']:
        raise ValueError('Measured training estimate exceeds reviewed walltime; no science admitted')
    return artifact('SETUP', parents=dict(spec=spec['content_hash'], bank=bank.manifest['content_hash']),
        job_id=os.environ.get('SLURM_JOB_ID'), source_commit=spec['source_commit'],
        environment=spec['reuse']['environment'], gpu=spec['reuse']['gpu'], memory=memory,
        teacher_maxima=maxima, teacher_tap_parity=True, probe=probe, technical_checks_passed=True,
        bank_hash=bank.manifest['content_hash'],
        outputs=[file_ref(root/p,root=root) for p in ('targets/bundle.json',
            'targets/COARSE_D066/manifest.json','targets/COARSE_D066/targets.npz',
            'targets/COARSE_D033/manifest.json','targets/COARSE_D033/targets.npz')])


def validate_setup(spec, *, accounting=True):
    from .submission import accepted_jobs, successful_setup
    root = Path(spec['root'])
    r = load_json(root/'setup/receipt.json')
    m = load_json(root/'targets/bundle.json')
    if m != banks.bundle_record(spec):
        raise ValueError('Four-bank bundle changed')
    validate(r, 'SETUP', dict(spec=spec['content_hash'], bank=m['content_hash']))
    summaries, p = spec['reuse']['summaries'], r['probe']
    steps = math.ceil(spec['counts']['train']/128)
    if (r['technical_checks_passed'] is not True or r['teacher_tap_parity'] is not True
            or r['source_commit'] != spec['source_commit'] or r['environment'] != spec['reuse']['environment']
            or r['gpu'] != spec['reuse']['gpu'] or r['bank_hash'] != m['content_hash']
            or r['teacher_maxima'] != {t:summaries['train'][TEACHERS[t]]['maximum'] for t in ('COARSE_D066','COARSE_D033')}
            or p['train_maximum'] != summaries['train']['D000']['maximum']
            or p['validation_maximum'] != summaries['validation']['D000']['maximum']
            or p['batch'] != 128 or p['stress_steps'] != 3 or p['finite_gradients'] is not True
            or p['native_tap_parity'] is not True or p['full_epoch_training_measured'] is not False
            or p['benchmark_steps'] != min(spec['acceptance']['benchmark_steps'],steps)
            or p['full_validation_rows'] != spec['counts']['validation']
            or not np.isfinite([p['benchmark_seconds'],p['full_validation_seconds']]).all()
            or min(p['benchmark_seconds'],p['full_validation_seconds']) <= 0
            or [v['path'] for v in r['outputs']] != ['targets/bundle.json',
                'targets/COARSE_D066/manifest.json','targets/COARSE_D066/targets.npz',
                'targets/COARSE_D033/manifest.json','targets/COARSE_D033/targets.npz']):
        raise ValueError('Setup identity or measured acceptance differs')
    estimate = math.ceil(spec['acceptance']['margin']*20*(steps*p['benchmark_seconds']/p['benchmark_steps']
        + p['full_validation_seconds'])/60 + spec['acceptance']['fixed_allowance_minutes'])
    if p['estimated_train_minutes'] != estimate or estimate > spec['train_minutes']:
        raise ValueError('Registered walltime does not contain measured estimate')
    for ref in r['outputs']:
        validate_file_ref(ref, root=root)
    for ref in m['records'].values():
        path = validate_file_ref(ref)
        validate_file_ref(load_json(path)['payload'], root=Path(path).parent)
    limits(spec, r['memory'])
    if r['job_id'] != accepted_jobs(spec).get('setup'):
        raise ValueError('Setup receipt is not the exact submitted job')
    if accounting:
        successful_setup(spec, r['job_id'])
    return r


def results(spec, *, require_complete=True):
    rows = []
    original = validation_arrays(spec)
    for name in ('M0HLT','OFFLINE','U100_EMBED_W5'):
        refs = spec['reference'] if name == 'U100_EMBED_W5' else spec['anchor']['evidence']['models'][name]
        report = load_json(validate_file_ref(refs['report']))
        arrays = load_npz_arrays(validate_file_ref(refs['predictions']))
        if (not np.array_equal(arrays['identities'], original['identities'])
                or not np.array_equal(arrays['labels'], original['labels'])
                or evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']):
            raise ValueError('Historical reference prediction/validation identity differs')
        rows.append(dict(node_id=name, selected_epoch=report['selected_pass'],
            validation=report['validation'], origin='verified_completed_reference'))
    admitted = validate_setup(spec)
    root = Path(spec['root'])
    report = validate_fit(root/'train_multi4', spec, admitted['bank_hash'])
    rows.append(dict(node_id='MULTI4_EMBED_W5', selected_epoch=report['selected_pass'],
        validation=report['validation'], origin='new_fit'))
    for row in rows:
        row['recovery'] = recovery(row['validation'], rows[0]['validation'], rows[1]['validation'])
    summary = artifact('RESULTS', parents=dict(spec=spec['content_hash'], setup=admitted['content_hash']),
        rows=rows, single_seed=True, recovery_baseline='historical M0HLT; no new matched w0',
        adaptive_validation=True, fresh_matched_control=False,
        auc_delta_vs_U100_w5=rows[-1]['validation']['macro_ovr_auc']-rows[-2]['validation']['macro_ovr_auc'])
    if require_complete:
        from .submission import accepted_jobs
        record = load_json(root/'train_multi4/task_receipt.json')
        validate(record, 'TASK', dict(spec=spec['content_hash'], setup=admitted['content_hash']))
        expected = ['train_multi4/'+n for n in ('selected.pt','training_report.json',
            'validation_predictions.npz','receipt.json','results.json')]
        if (record['task_id'] != 'train_multi4' or record['source_commit'] != spec['source_commit']
                or record['job_id'] != accepted_jobs(spec).get('train_multi4')
                or record['report_hash'] != report['content_hash']
                or [ref['path'] for ref in record['outputs']] != expected):
            raise ValueError('Completed fit publication identity differs')
        for ref in record['outputs']:
            validate_file_ref(ref, root=root)
        if load_json(root/'train_multi4/results.json') != summary:
            raise ValueError('Saved comparison differs from authenticated predictions')
    return summary


def run(spec, task):
    from .submission import accepted_jobs
    campaign.check(spec)
    if task not in ('setup','train_multi4'):
        raise PermissionError('Only registered setup and HLT fit; no test operation')
    job = execution(spec, spec['reuse'])
    if job != accepted_jobs(spec).get(task):
        raise PermissionError('Worker is not the exact submitted job')
    root = Path(spec['root'])
    (root/(task+'_attempt')).mkdir(exist_ok=False)
    campaign.storage(spec)
    if task == 'setup':
        record = setup(spec)
        campaign.check(spec)
        if execution(spec, spec['reuse']) != job:
            raise PermissionError('Allocation changed')
        write_immutable_json(root/'setup/receipt.json', record)
        return validate_setup(spec, accounting=False)
    admitted = validate_setup(spec)
    meta = metadata(spec)
    log = PhaseLog(root/(task+'_attempt'), lambda: memory_snapshot(torch))
    with log.phase('load_HLT_and_four_frozen_TRAIN_banks'):
        tr = data(spec,'train','D000',authenticated=meta)
        val = data(spec,'validation','D000',authenticated=meta)
        bank = banks.load(spec,tr)
        if bank.manifest['content_hash'] != admitted['bank_hash']:
            raise ValueError('Admitted targets changed')
    with log.phase('scientific_twenty_epoch_fit'):
        fit(root/'train_multi4',spec,tr,val,bank,device='cuda')
    report = validate_fit(root/'train_multi4',spec,admitted['bank_hash'])
    summary = results(spec,require_complete=False)
    write_immutable_json(root/'train_multi4/results.json',summary)
    campaign.check(spec)
    if execution(spec,spec['reuse']) != job:
        raise PermissionError('Allocation changed')
    record = artifact('TASK',parents=dict(spec=spec['content_hash'],setup=admitted['content_hash']),
        task_id=task,job_id=job,source_commit=spec['source_commit'],report_hash=report['content_hash'],
        outputs=[file_ref(root/'train_multi4'/n,root=root) for n in (
            'selected.pt','training_report.json','validation_predictions.npz','receipt.json','results.json')])
    write_immutable_json(root/'train_multi4/task_receipt.json',record)
    return record
