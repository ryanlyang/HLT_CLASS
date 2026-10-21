"""Standalone CE embedding sweep fit and native prediction replay, no test operation."""
from pathlib import Path
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
from .contracts import artifact, validate, recipe, REFERENCES, WEIGHTS, arm_gate


def data(gate, role, coordinate, *, authenticated=None):
    if (role, coordinate) not in (('train', 'D000'), ('validation', 'D000'), ('train', 'OFFLINE')):
        raise PermissionError('Only ordinary HLT views and pure OFFLINE TRAIN allowed')
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


def supervision_hash(gate):
    m = load_json(Path(gate['root'])/'targets/manifest.json')
    validate(m, 'BANK', banks.parents(gate['anchor']['evidence']['prepared'],
                                     gate['anchor']['evidence']['models']['OFFLINE']))
    return m['content_hash']


def completed(spec, gate, name):
    if name not in {*WEIGHTS, 'aggregate'}:
        raise ValueError('Unknown science task')
    root = Path(spec['root'])
    path = root/name/'task_receipt.json'
    if not path.is_file():
        return None
    row = load_json(path)
    validate(row, 'TASK', dict(spec=spec['content_hash']))
    names = [ref['path'] for ref in row['outputs']]
    if (row['task_id'] != name or row['source_commit'] != spec['source_commit']
            or len(names) != len(set(names)) or any(not p.startswith(name+'/') for p in names)):
        raise ValueError('Task/source/payload scope differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    if name in WEIGHTS:
        required = {name+'/'+f for f in ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'receipt.json')}
        report = validate_fit(root/name, arm_gate(gate, name), supervision_hash(gate))
        if not required <= set(names) or row['dependencies'] != {} or row['result'] != dict(report_hash=report['content_hash']):
            raise ValueError('Scientific fit task differs')
    else:
        deps = {arm: completed(spec, gate, arm) for arm in WEIGHTS}
        if (any(v is None for v in deps.values())
                or row['dependencies'] != {k:v['content_hash'] for k,v in deps.items()}
                or name+'/results.json' not in names or row['result'] != results(spec, gate)
                or load_json(root/name/'results.json') != row['result']):
            raise ValueError('Comparison/dependency differs')
    return row


def results(spec, gate):
    rows, pending = [], []
    for name in REFERENCES:
        refs = gate['anchor']['evidence']['models'][name]
        report = load_json(validate_file_ref(refs['report']))
        a = load_npz_arrays(validate_file_ref(refs['predictions']))
        if evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']:
            raise ValueError('Reference metric replay differs')
        rows.append(dict(node_id=name, origin='historical_reference', selected_epoch=report['selected_pass'],
                         validation=report['validation']))
    for arm, weight in WEIGHTS.items():
        if not completed(spec, gate, arm):
            pending.append(arm)
            continue
        report = load_json(Path(spec['root'])/arm/'training_report.json')
        rows.append(dict(node_id=arm, weight=weight, origin='new_fit', selected_epoch=report['selected_pass'],
                         validation=report['validation']))
    baseline = next((r['validation'] for r in rows if r['node_id'] == 'w0'), None)
    offline = rows[1]['validation']
    for row in rows:
        row['recovery'] = None if baseline is None else recovery(row['validation'], baseline, offline)
        row['delta_auc_vs_matched_CE'] = (None if baseline is None else
            row['validation']['macro_ovr_auc']-baseline['macro_ovr_auc'])
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        single_seed=True, recovery_baseline='new w0 (not historical M0HLT)',
        primary_contrast='each embedding weight minus matched w0', final_test_selected=False)


def run(spec, name):
    gate, measured = campaign.check_spec(spec)
    if name not in {*WEIGHTS, 'aggregate'}:
        raise PermissionError('Only registered sweep tasks; no test operation')
    previous = completed(spec, gate, name)
    if previous:
        return previous
    root = Path(spec['root'])
    directory = root/name
    if directory.exists():
        raise FileExistsError('Preserve partial output; no automatic retry')
    if name in WEIGHTS:
        campaign.storage(gate)
        job = execution(gate, measured)
        attempt = root/(name+'_attempt')
        attempt.mkdir(exist_ok=False)
        log = PhaseLog(attempt, lambda: memory_snapshot(torch))
        with log.phase('load_HLT_and_shared_TRAIN_embeddings'):
            meta = metadata(gate)
            tr = data(gate, 'train', 'D000', authenticated=meta)
            val = data(gate, 'validation', 'D000', authenticated=meta)
            bank = banks.load(gate, tr)
            if bank.manifest['content_hash'] != measured['bank_hash']:
                raise ValueError('Measured target bank changed')
        with log.phase('scientific_twenty_epoch_CE_embedding_fit'):
            scoped = arm_gate(gate, name)
            fit(directory, scoped, tr, val, bank, device='cuda')
        report = validate_fit(directory, scoped, bank.manifest['content_hash'])
        result, deps = dict(report_hash=report['content_hash']), {}
        execution(gate, measured)
    else:
        job = os.environ.get('SLURM_JOB_ID')
        done = {arm: completed(spec, gate, arm) for arm in WEIGHTS}
        if any(v is None for v in done.values()):
            raise ValueError('Missing completed fit')
        result, deps = results(spec, gate), {k:v['content_hash'] for k,v in done.items()}
        directory.mkdir(exist_ok=False)
        write_immutable_json(directory/'results.json', result)
    campaign.check_spec(spec)
    row = artifact('TASK', parents=dict(spec=spec['content_hash']), task_id=name, job_id=job,
        source_commit=spec['source_commit'], dependencies=deps, result=result,
        outputs=[file_ref(p, root=root) for p in sorted(directory.iterdir()) if p.is_file()])
    write_immutable_json(directory/'task_receipt.json', row)
    return completed(spec, gate, name)
