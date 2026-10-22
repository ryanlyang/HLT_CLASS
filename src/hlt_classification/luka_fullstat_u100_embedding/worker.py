"""Frozen U100 export and standalone CE+embedding fit; no test operation."""
from pathlib import Path
import gc
import os
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
from .contracts import artifact, validate, recipe, REFERENCES


def data(gate, role, coordinate, *, authenticated=None):
    if (role, coordinate) not in (('train', 'D000'), ('validation', 'D000'), ('train', 'U100')):
        raise PermissionError('Only ordinary HLT views and U100 TRAIN allowed')
    meta = metadata(gate) if authenticated is None else authenticated
    if meta[0]['content_hash'] != gate['anchor']['evidence']['prepared']:
        raise ValueError('Prepared cache binding differs')
    value = cache.load(gate['anchor']['evidence']['cache_root'], meta, role=role,
        coordinate=coordinate, max_ram_bytes=int(gate['memory_mb']*1024**2*.35))
    if len(value) != gate['counts'][role]:
        raise ValueError('Population differs')
    return value


def report_valid(report, gate, bank_hash, *, technical=False):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe(gate['node']['auxiliary_weight'])['content_hash'],
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
    weight = gate['node']['auxiliary_weight']
    for r in history:
        values = [r[k] for k in ('ce_loss', 'auxiliary_loss', 'weighted_auxiliary_loss',
                                 'total_loss', 'train_seconds', 'validation_seconds')]
        if (not np.isfinite(values).all() or min(values) < 0
                or r['weighted_auxiliary_loss'] != weight*r['auxiliary_loss']
                or r['total_loss'] != r['ce_loss']+r['weighted_auxiliary_loss']):
            raise ValueError('Loss/time evidence differs')
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
    if any(k.startswith('predictor.') for k in state):
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
        raise MemoryError('New-target startup exceeds reused Oscar memory envelope')


def longest(bundle):
    lengths = np.concatenate([np.diff(block.offsets) for block in bundle.blocks])
    return int(np.argmax(lengths)), int(lengths.max())


def witnesses(spec, tr, val, bank, *, device):
    """Only three max-support updates; never a technical epoch or science seed."""
    from hlt_classification.luka_embedding_aux.model import new_model
    from hlt_classification.jetclass2_delphes.runner import _forward
    banks.check_join(bank, tr)
    model = new_model(spec['node']).to(device).train()
    model.capture = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    index, maximum = longest(tr)
    ix = np.full(128, index, np.int64)
    raw = tr.batch(ix)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
        loss, *_ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
            model.prediction, torch.from_numpy(bank.targets[ix]).to(device), coefficient=5.)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite U100 weight5 startup gradient')
        optimizer.step()
    model.eval(); model.capture = False; model.prediction = None
    vi, vmax = longest(val)
    with torch.inference_mode():
        logits = _forward(model, val.batch(np.full(128, vi, np.int64)), device,
                          bf16=str(device).startswith('cuda'))
    if logits.shape != (128,11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid HLT validation startup output')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return dict(train_maximum=maximum, validation_maximum=vmax, batch=128,
                steps=3, weight=5., finite_gradients=True, technical_epoch_run=False)


def export(spec, *, device='cuda'):
    from hlt_classification.luka_embedding_aux.model import head_input
    from hlt_classification.jetclass2_delphes.runner import _forward
    meta = metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Prepared resource support differs from authenticated reuse')
    directory = Path(spec['root'])/'export_U100'
    directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    with log.phase('export_frozen_U100_TRAIN_targets'):
        cache_ = data(spec, 'train', 'U100', authenticated=meta)
        teacher = DelphesParticleTransformer().to(device)
        teacher.load_state_dict(torch.load(validate_file_ref(spec['teacher_refs']['checkpoint']),
            map_location='cpu', weights_only=True), strict=True)
        teacher.eval().requires_grad_(False)
        index, maximum = longest(cache_)
        raw = cache_.batch(np.full(128, index, np.int64))
        tensors = {k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
        with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
            for _ in range(3):
                logits, vector = head_input(teacher, **tensors)
                if vector.shape != (128,128) or not torch.isfinite(vector).all() or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite U100 maximum-support export')
            torch.testing.assert_close(logits, _forward(teacher, raw, device, bf16=False), rtol=0, atol=0)
        limits(spec, memory_snapshot(torch))
        bank_record = banks.export(Path(spec['root'])/'targets', teacher, cache_,
            refs=spec['teacher_refs'], device=device)
        del cache_, teacher, raw, tensors, logits, vector
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    with log.phase('new_target_join_and_three_update_startup_check'):
        tr = data(spec, 'train', 'D000', authenticated=meta)
        val = data(spec, 'validation', 'D000', authenticated=meta)
        bank = banks.load(spec, tr)
        original = validation_arrays(spec)
        if not np.array_equal(val.identities, original['identities']) or not np.array_equal(val.labels, original['labels']):
            raise ValueError('Validation population differs')
        probe = witnesses(spec, tr, val, bank, device=device)
        del tr, val, bank, original
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    memory = memory_snapshot(torch)
    limits(spec, memory)
    return artifact('EXPORT', parents=dict(spec=spec['content_hash'], bank=bank_record['content_hash']),
        job_id=os.environ.get('SLURM_JOB_ID'), source_commit=spec['source_commit'],
        environment=spec['reuse']['environment'], gpu=spec['reuse']['gpu'],
        memory=memory, teacher_maximum=maximum, teacher_tap_parity=True, probe=probe,
        bank_hash=bank_record['content_hash'], new_full_preflight=False,
        outputs=[file_ref(Path(spec['root'])/p, root=Path(spec['root'])) for p in
                 ('targets/manifest.json','targets/targets.npz')])


def validate_export(spec, *, accounting=True):
    from .submission import accepted_jobs, successful_export
    root = Path(spec['root'])
    r = load_json(root/'export_U100/receipt.json')
    m = load_json(root/'targets/manifest.json')
    validate(m, 'BANK', banks.parents(spec['anchor']['evidence']['prepared'], spec['teacher_refs']))
    validate(r, 'EXPORT', dict(spec=spec['content_hash'], bank=m['content_hash']))
    summaries = spec['reuse']['summaries']
    expected = dict(train_maximum=summaries['train']['D000']['maximum'],
        validation_maximum=summaries['validation']['D000']['maximum'],
        batch=128, steps=3, weight=5., finite_gradients=True, technical_epoch_run=False)
    if (r['probe'] != expected or r['teacher_maximum'] != summaries['train']['U100']['maximum']
            or r['teacher_tap_parity'] is not True or r['new_full_preflight'] is not False
            or r['source_commit'] != spec['source_commit'] or r['environment'] != spec['reuse']['environment']
            or r['gpu'] != spec['reuse']['gpu'] or r['bank_hash'] != m['content_hash']
            or m['teacher'] != 'COARSE_U100' or m['coordinate'] != 'U100' or m['role'] != 'train'
            or m['width'] != 128 or m['rows'] != spec['counts']['train']
            or m['checkpoint'] != spec['teacher_refs']['checkpoint'] or m['report'] != spec['teacher_refs']['report']
            or m['payload']['path'] != 'targets.npz'
            or [p['path'] for p in r['outputs']] != ['targets/manifest.json','targets/targets.npz']):
        raise ValueError('Export admission identity/probes differ')
    for ref in r['outputs']:
        validate_file_ref(ref, root=root)
    validate_file_ref(m['payload'], root=root/'targets')
    limits(spec, r['memory'])
    if r['job_id'] != accepted_jobs(spec).get('export_U100'):
        raise ValueError('Export receipt is not the exact submitted job')
    if accounting:
        successful_export(spec, r['job_id'])
    return r


def results(spec, *, require_complete=True):
    rows = []
    for name in REFERENCES:
        refs = spec['teacher_refs'] if name == 'COARSE_U100' else spec['anchor']['evidence']['models'][name]
        report = load_json(validate_file_ref(refs['report']))
        arrays = load_npz_arrays(validate_file_ref(refs['predictions']))
        original = validation_arrays(spec)
        if (not np.array_equal(arrays['identities'], original['identities'])
                or not np.array_equal(arrays['labels'], original['labels'])
                or evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']):
            raise ValueError('Historical reference prediction/validation identity differs')
        rows.append(dict(node_id=name, selected_epoch=report['selected_pass'],
                         validation=report['validation'], origin='historical_reference'))
    exported = validate_export(spec)
    report = validate_fit(Path(spec['root'])/'train_w5', spec, exported['bank_hash'])
    rows.append(dict(node_id='U100_EMBED_W5', selected_epoch=report['selected_pass'],
        validation=report['validation'], origin='new_fit'))
    for row in rows:
        row['recovery'] = recovery(row['validation'], rows[0]['validation'], rows[1]['validation'])
    summary = artifact('RESULTS', parents=dict(spec=spec['content_hash'], export=exported['content_hash']),
        rows=rows, single_seed=True, recovery_baseline='historical M0HLT; no new matched w0',
        adaptive_validation=True, causal_teacher_comparison=False)
    if require_complete:
        from .submission import accepted_jobs
        root = Path(spec['root'])
        record = load_json(root/'train_w5/task_receipt.json')
        validate(record, 'TASK', dict(spec=spec['content_hash'], export=exported['content_hash']))
        expected = ['train_w5/'+name for name in ('selected.pt', 'training_report.json',
                    'validation_predictions.npz', 'receipt.json', 'results.json')]
        if (record['task_id'] != 'train_w5' or record['source_commit'] != spec['source_commit']
                or record['job_id'] != accepted_jobs(spec).get('train_w5')
                or record['report_hash'] != report['content_hash']
                or [ref['path'] for ref in record['outputs']] != expected):
            raise ValueError('Completed fit publication identity differs')
        for ref in record['outputs']:
            validate_file_ref(ref, root=root)
        if load_json(root/'train_w5/results.json') != summary:
            raise ValueError('Saved comparison differs from authenticated selected predictions')
    return summary


def run(spec, task):
    from .submission import accepted_jobs
    campaign.check(spec)
    if task not in ('export_U100', 'train_w5'):
        raise PermissionError('Only registered export/HLT fit; no test operation')
    job = execution(spec, spec['reuse'])
    if job != accepted_jobs(spec).get(task):
        raise PermissionError('Worker job does not match exact submission journal')
    root = Path(spec['root'])
    (root/(task+'_attempt')).mkdir(exist_ok=False)
    campaign.storage(spec)
    if task == 'export_U100':
        record = export(spec)
        campaign.check(spec)
        if execution(spec, spec['reuse']) != job:
            raise PermissionError('Allocation changed')
        write_immutable_json(root/'export_U100/receipt.json', record)
        return validate_export(spec, accounting=False)
    exported = validate_export(spec)
    meta = metadata(spec)
    log = PhaseLog(root/(task+'_attempt'), lambda: memory_snapshot(torch))
    with log.phase('load_HLT_and_frozen_U100_TRAIN_targets'):
        tr = data(spec, 'train', 'D000', authenticated=meta)
        val = data(spec, 'validation', 'D000', authenticated=meta)
        bank = banks.load(spec, tr)
        if bank.manifest['content_hash'] != exported['bank_hash']:
            raise ValueError('Exported bank changed')
    with log.phase('scientific_twenty_epoch_fit'):
        fit(root/'train_w5', spec, tr, val, bank, device='cuda')
    report = validate_fit(root/'train_w5', spec, exported['bank_hash'])
    summary = results(spec, require_complete=False)
    write_immutable_json(root/'train_w5/results.json', summary)
    campaign.check(spec)
    if execution(spec, spec['reuse']) != job:
        raise PermissionError('Allocation changed')
    record = artifact('TASK', parents=dict(spec=spec['content_hash'], export=exported['content_hash']),
        task_id=task, job_id=job, source_commit=spec['source_commit'], report_hash=report['content_hash'],
        outputs=[file_ref(root/'train_w5'/name, root=root) for name in (
            'selected.pt','training_report.json','validation_predictions.npz','receipt.json','results.json')])
    write_immutable_json(root/'train_w5/task_receipt.json', record)
    return record

