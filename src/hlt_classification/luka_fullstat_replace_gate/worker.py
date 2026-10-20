"""Authenticated cold fits, selected PRIV export, and all-model validation report."""
from pathlib import Path
import os
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_attention.worker import execution, metadata, validation_arrays, data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_gating_cold.training import new_student
from hlt_classification.luka_gating import banks as shared
from . import campaign, banks, training, checkpoints
from .contracts import artifact, validate, recipe, REFERENCES, NAMES


def report_valid(report, gate, node, bank_hash, *, technical=False):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe()['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'], bank=bank_hash))
    history = report['validation_history']
    if (node not in gate['nodes'] or report['node'] != node
            or report['scientific_fit'] is not (not technical) or report['technical'] is not technical
            or report['passes'] != (1 if technical else 20) or len(history) != report['passes']
            or [r['pass_number'] for r in history] != list(range(1, len(history)+1))
            or report['selected_state'] != 'native_and_gate'
            or report['initialization_policy'] != recipe()['initialization']
            or len(report['initialization_sha256']) != 64
            or any(r['backbone_frozen'] is not False for r in history)
            or any(r['validation']['rows'] != gate['counts']['validation'] for r in history)
            or any(r['learning_rate'] != learning_rate(r['pass_number']) for r in history)
            or any(r['update'] != r['pass_number']*((gate['counts']['train']+127)//128) for r in history)
            or (not node['gate_weight'] and any(r['losses']['gate'] != 0. for r in history))):
        raise ValueError('FullStat training policy/population differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Wrong validation selection')


def fit(directory, gate, node, tr, val, ctx, vctx, bank, *, device, technical=False, store=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report, state = training.train(tr, val, ctx, vctx, node=node, bank=bank, device=device,
                                   technical=technical, store=store)
    report_valid(report, gate, node, bank.manifest['content_hash'], technical=technical)
    if (not any(k.startswith('gate.') for k in state)
            or any(not k.startswith(('gate.', 'mod.')) for k in state)):
        raise ValueError('Selected checkpoint must retain native and gate')
    save_state(directory/'selected.pt', state)
    restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('Selected checkpoint roundtrip differs')
    model = new_student(node).to(device)
    model.load_state_dict(restored, strict=True)
    arrays = dict(identities=val.identities, labels=val.labels,
                  probabilities=shared.predict(model, val, vctx, device=device))
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Full-gated selected prediction readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json', report)
    result = artifact('FIT', parents=dict(gate=gate['content_hash'], bank=bank.manifest['content_hash']),
        node_id=node['node_id'], report_hash=report['content_hash'], technical=technical,
        checkpoint_probability_roundtrip=True, retained_gate=True,
        outputs=[file_ref(directory/f, root=directory) for f in
                 ('selected.pt', 'training_report.json', 'validation_predictions.npz')])
    write_immutable_json(directory/'receipt.json', result)
    return result


def validate_fit(directory, gate, node, bank_hash, *, technical=False):
    directory = Path(directory)
    receipt = load_json(directory/'receipt.json')
    validate(receipt, 'FIT', dict(gate=gate['content_hash'], bank=bank_hash))
    if (receipt['node_id'] != node['node_id'] or receipt['technical'] is not technical
            or receipt['retained_gate'] is not True
            or receipt['checkpoint_probability_roundtrip'] is not True
            or [ref['path'] for ref in receipt['outputs']] !=
               ['selected.pt', 'training_report.json', 'validation_predictions.npz']):
        raise ValueError('Fit publication differs')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=directory)
    report = load_json(directory/'training_report.json')
    report_valid(report, gate, node, bank_hash, technical=technical)
    a, original = load_npz_arrays(directory/'validation_predictions.npz'), validation_arrays(gate)
    if (receipt['report_hash'] != report['content_hash']
            or set(a) != {'identities', 'labels', 'probabilities'}
            or not np.array_equal(a['identities'], original['identities'])
            or not np.array_equal(a['labels'], original['labels'])
            or evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']):
        raise ValueError('Selected metrics/population differ')
    return report


def definition(gate, name):
    return next(t for t in gate['tasks'] if t['task_id'] == name)


def completed(spec, gate, name):
    task = definition(gate, name)
    root = Path(spec['root'])
    directory = root/name
    path = directory/'task_receipt.json'
    if not path.is_file():
        return None
    row = load_json(path)
    validate(row, 'TASK', dict(spec=spec['content_hash']))
    deps = {}
    for dep in task['dependencies']:
        previous = completed(spec, gate, dep)
        if previous is None:
            raise ValueError('Completed task has missing dependency')
        deps[dep] = previous['content_hash']
    names = [r['path'] for r in row['outputs']]
    if (row['task_id'] != name or row['source_commit'] != spec['source_commit']
            or row['dependencies'] != deps or len(names) != len(set(names))
            or any(not p.startswith(name+'/') for p in names)):
        raise ValueError('Task lineage/output scope differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    if task['kind'] == 'train':
        node = next(n for n in gate['nodes'] if name == 'train_'+n['node_id'])
        bank = load_json(directory/'supervision.json')
        if name == 'train_PRIV_D000':
            validate(bank, 'SUPERVISION', dict(logits=load_json(validate_file_ref(gate['anchor']['logits']))['content_hash']))
            if (bank['teacher'] != 'COARSE_D033' or bank['coordinate'] != 'D033'
                    or bank['technical'] is not False or bank['alignment']['role'] != 'train'
                    or bank['alignment']['prepared'] != gate['anchor']['evidence']['prepared']
                    or bank['alignment']['rows'] != gate['counts']['train']):
                raise ValueError('Original teacher supervision differs')
        elif bank != banks.check_manifest(root/'export_PRIV_D000', gate, root/'train_PRIV_D000'):
            raise ValueError('Replacement teacher bank differs')
        report = validate_fit(directory, gate, node, bank['content_hash'])
        required = {name+'/'+f for f in ('selected.pt', 'training_report.json',
                    'validation_predictions.npz', 'receipt.json', 'supervision.json')}
        if not required <= set(names) or row['result'] != dict(report_hash=report['content_hash']):
            raise ValueError('Training task differs')
    elif task['kind'] == 'export':
        m = banks.check_manifest(directory, gate, root/'train_PRIV_D000')
        payload = load_json(directory/'payload/manifest.json')
        required = {name+'/manifest.json', name+'/payload/manifest.json', name+'/payload/population.npz'} | {
            name+'/payload/'+s['payload']['path'] for s in payload['shards']}
        if not required <= set(names) or row['result'] != dict(bank_hash=m['content_hash']):
            raise ValueError('Export task differs')
    else:
        expected = results(spec, gate)
        if (name+'/results.json' not in names or row['result'] != expected
                or load_json(directory/'results.json') != expected):
            raise ValueError('Comparison differs')
    return row


def results(spec, gate):
    rows = []
    for name in REFERENCES:
        refs = gate['anchor']['evidence']['models'][name]
        report = load_json(validate_file_ref(refs['report']))
        a = load_npz_arrays(validate_file_ref(refs['predictions']))
        if evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']:
            raise ValueError('Historical metric replay differs')
        rows.append(dict(node_id=name, origin='historical_reference', selected_epoch=report['selected_pass'],
            validation=report['validation'], input='OFFLINE ORACLE' if name in ('OFFLINE', 'COARSE_D033') else 'single HLT'))
    for name in NAMES:
        if not completed(spec, gate, 'train_'+name):
            continue
        report = load_json(Path(spec['root'])/('train_'+name)/'training_report.json')
        rows.append(dict(node_id=name, origin='new_fit', selected_epoch=report['selected_pass'],
            validation=report['validation'], input='OFFLINE ORACLE' if name == 'PRIV_D000' else 'single HLT + self gate'))
    for row in rows:
        row['recovery'] = recovery(row['validation'], rows[0]['validation'], rows[1]['validation'])
    lookup = {r['node_id']: r for r in rows}
    contrasts = None
    if all(n in lookup for n in NAMES[1:]):
        contrasts = {k: lookup['REPLACE_GATE']['validation'][k]-lookup['REPLACE_LOGIT']['validation'][k]
                     for k in ('accuracy', 'macro_ovr_auc', 'cross_entropy')}
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, single_seed=True,
        primary_contrast='REPLACE_GATE minus REPLACE_LOGIT', contrasts=contrasts,
        pending=[n for n in NAMES if n not in lookup])


def run(spec, name):
    gate, measured = campaign.check_spec(spec)
    task = definition(gate, name)
    previous = completed(spec, gate, name)
    if previous:
        return previous
    root, directory = Path(spec['root']), Path(spec['root'])/name
    if directory.exists():
        raise FileExistsError('Preserve partial output; no automatic retry')
    deps = {}
    for dep in task['dependencies']:
        old = completed(spec, gate, dep)
        if old is None:
            raise ValueError('Missing completed dependency')
        deps[dep] = old['content_hash']
    if task['kind'] != 'aggregate':
        job = execution(gate, measured)
        # Remaining bytes, not the original free-space reservation after it is consumed.
        campaign.storage(gate, required=measured['additional_bytes'][name])
        attempt = root/(name+'_attempt')
        attempt.mkdir(exist_ok=False)
        log = PhaseLog(attempt, lambda: memory_snapshot(torch))
        with log.phase('load_authenticated_HLT_and_optional_D033'):
            meta = metadata(gate)
            tr = data(gate, 'train', 'D000', authenticated=meta)
            ctx = data(gate, 'train', 'D033', authenticated=meta) if 'PRIV' in name else None
            if ctx is not None:
                shared.paired(tr, ctx)
        if task['kind'] == 'export':
            with log.phase('export_selected_PRIV_TRAIN_factors'):
                m = banks.export(directory, gate, tr, ctx, root/'train_PRIV_D000',
                                 technical=False, device='cuda')
            result = dict(bank_hash=m['content_hash'])
        else:
            node = next(n for n in gate['nodes'] if name == 'train_'+n['node_id'])
            with log.phase('load_validation_and_compact_TRAIN_supervision'):
                val = data(gate, 'validation', 'D000', authenticated=meta)
                vctx = data(gate, 'validation', 'D033', authenticated=meta) if ctx is not None else None
                bank = (banks.original(gate, tr) if ctx is not None else
                        banks.Resident(root/'export_PRIV_D000', gate, tr, root/'train_PRIV_D000'))
            store = checkpoints.EpochStore(attempt/'epochs', binding=dict(
                gate=gate['content_hash'], spec=spec['content_hash'], node=node['node_id'],
                bank=bank.manifest['content_hash']))
            with log.phase('scientific_twenty_epoch_cold_fit'):
                fit(directory, gate, node, tr, val, ctx, vctx, bank, device='cuda', store=store)
            write_immutable_json(directory/'supervision.json', bank.manifest)
            report = validate_fit(directory, gate, node, bank.manifest['content_hash'])
            result = dict(report_hash=report['content_hash'])
        execution(gate, measured)
    else:
        job = os.environ.get('SLURM_JOB_ID')
        result = results(spec, gate)
        if result['pending']:
            raise ValueError('Cannot aggregate incomplete registered fits')
        directory.mkdir(exist_ok=False)
        write_immutable_json(directory/'results.json', result)
    campaign.check_spec(spec)
    row = artifact('TASK', parents=dict(spec=spec['content_hash']), task_id=name, job_id=job,
        source_commit=spec['source_commit'], dependencies=deps, result=result,
        outputs=[file_ref(p, root=root) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'task_receipt.json', row)
    return completed(spec, gate, name)
