"""One auxiliary fit and compact, re-verifiable CE comparison outputs."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullsim import regular_cache, alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_attention.worker import data, execution
from .contracts import artifact, validate, recipe
from . import training, campaign, source
from hlt_classification.luka_embedding_weight.worker import (
    storage, validate_predictions,
)
from hlt_classification.luka_embedding_aux.model import new_model


def report_valid(report, gate, bank_hash, *, technical):
    if gate['recipe'] != recipe():
        raise ValueError('Weight recipe differs')
    validate(report, 'TRAINING_REPORT', dict(recipe=gate['recipe']['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'], bank=bank_hash))
    epochs = 1 if technical else 30
    history = report['validation_history']
    steps = (gate['counts']['train']+127)//128
    if (report['node'] != gate['node'] or report['scientific_fit'] is not (not technical)
            or report['technical'] is not technical or report['zero_control'] is not False
            or report['passes'] != epochs or len(history) != epochs
            or report['selected_state'] != 'native_backbone_only'
            or report['initialization'] != 'from_scratch_matched_M0HLT'
            or report['selected_weights_restored'] is not True
            or [r['pass_number'] for r in history] != list(range(1, epochs+1))
            or any(r['update'] != r['pass_number']*steps or r['validation']['rows'] != gate['counts']['validation']
                   or r['learning_rate'] != learning_rate(r['pass_number'])
                   or r['weighted_auxiliary_loss'] != gate['recipe']['auxiliary_coefficient'] * r['auxiliary_loss']
                   or r['total_loss'] != r['ce_loss'] + r['weighted_auxiliary_loss']
                   or not np.isfinite([r['ce_loss'], r['auxiliary_loss'], r['train_seconds'], r['validation_seconds']]).all()
                   or min(r['ce_loss'], r['auxiliary_loss'], r['train_seconds'], r['validation_seconds']) < 0
                   for r in history)):
        raise ValueError('Fit recipe, population or technical/scientific status differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Selection differs from full validation AUC rule')


def fit(directory, gate, train, val, bank, *, device, technical):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model = new_model(gate['node'])
    report, state = training.train(model, train, val, bank=bank, node=gate['node'], device=device, technical=technical)
    report_valid(report, gate, bank.manifest['content_hash'], technical=technical)
    probabilities = predict(model, val, device=device, batch_size=128)
    save_state(directory/'selected.pt', state)
    restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('Selected native checkpoint roundtrip differs')
    native = DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(restored, strict=True)
    deployment = predict(native, val, device=device, batch_size=128)
    np.testing.assert_allclose(deployment, probabilities, rtol=1e-5, atol=1e-6)
    metrics = evaluate_probabilities(val.labels, deployment)
    if metrics != report['validation']:
        raise ValueError('Native deployment metrics differ from selected validation')
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=deployment)))
    write_immutable_json(directory/'training_report.json', report)
    result = artifact('FIT', parents=dict(gate=gate['content_hash'], bank=bank.manifest['content_hash']),
        technical=technical, checkpoint_probability_roundtrip=True,
        outputs=[file_ref(directory/f, root=directory) for f in
                 ('selected.pt', 'training_report.json', 'validation_predictions.npz')])
    write_immutable_json(directory/'receipt.json', result)
    return result


def validate_fit(directory, gate, bank_hash, *, technical):
    directory = Path(directory)
    fit_record = load_json(directory/'receipt.json')
    validate(fit_record, 'FIT', dict(gate=gate['content_hash'], bank=bank_hash))
    if (fit_record['technical'] is not technical or fit_record['checkpoint_probability_roundtrip'] is not True
            or [r['path'] for r in fit_record['outputs']] !=
               ['selected.pt', 'training_report.json', 'validation_predictions.npz']):
        raise ValueError('Selected fit output contract differs')
    for ref in fit_record['outputs']:
        validate_file_ref(ref, root=directory)
    report = load_json(directory/'training_report.json')
    report_valid(report, gate, bank_hash, technical=technical)
    a = load_npz_arrays(directory/'validation_predictions.npz')
    validate_predictions(a, gate['counts']['validation'])
    if evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']:
        raise ValueError('Saved selected metrics differ')
    return report, a


def snapshot_references(gate, val, meta, *, folder=None):
    folder = Path(gate['root'])/'preflight/references' if folder is None else Path(folder)
    folder.mkdir(exist_ok=False)
    ix = alljets_data.comparison_indices(meta[2], meta[3], val.identities, val.labels)
    atomic_publish_bytes(folder/'original50k.npz', deterministic_npz_bytes(dict(indices=ix)))
    refs = {n: gate['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT','OFFLINE')}
    refs.update(OFFLINE_EMBED_W5=gate['reference']['files']['predictions'],
                LOGIT_D066=gate['teacher']['files']['predictions'])
    for name, ref in refs.items():
        validate_file_ref(ref); a = load_npz_arrays(ref['path'])
        validate_predictions(a, len(val))
        if not np.array_equal(a['identities'], val.identities) or not np.array_equal(a['labels'], val.labels):
            raise ValueError('Reference population/order differs')
        atomic_publish_bytes(folder/(name+'.npz'), deterministic_npz_bytes(a))
        validate_file_ref(ref)


def comparison(gate, student, *, folder=None):
    from hlt_classification.jetclass2_delphes.reporting import recovery
    folder = Path(gate['root'])/'preflight/references' if folder is None else Path(folder)
    ix = load_npz_arrays(folder/'original50k.npz')['indices']
    if (ix.dtype != np.int64 or ix.shape != (gate['original50k_rows'],)
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(student['labels']))):
        raise ValueError('Original50k report-only indices differ')
    arrays = {n: load_npz_arrays(folder/(n+'.npz')) for n in
              ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','LOGIT_D066')}
    arrays['HLT_D066_EMBED_W5'] = student
    rows = []
    for name, a in arrays.items():
        validate_predictions(a, gate['counts']['validation'])
        if not np.array_equal(a['identities'], student['identities']) or not np.array_equal(a['labels'], student['labels']):
            raise ValueError('Comparison identities/labels differ')
        rows.append(dict(model=name, input='ORACLE' if name in ('OFFLINE','LOGIT_D066') else 'single HLT',
            validation=evaluate_probabilities(a['labels'], a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])))
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery'] = recovery(row[role], rows[0][role], rows[1][role])
            row[role+'_delta_vs_ce'] = row[role]['macro_ovr_auc']-rows[0][role]['macro_ovr_auc']
            row[role+'_delta_vs_offline_w5'] = row[role]['macro_ovr_auc']-rows[2][role]['macro_ovr_auc']
    return rows


def run(spec):
    from .targets import Bank
    gate, measured = campaign.check_spec(spec)
    job = execution(gate, measured); storage(gate)
    root = Path(spec['root'])
    if (root/'receipt.json').exists() or (root/'fit').exists():
        raise FileExistsError('Preserve existing/partial scientific fit')
    log = PhaseLog(root, lambda: memory_snapshot(torch))
    with log.phase('load_HLT_and_D066_TRAIN_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train, val = (data(gate, role, 'D000', meta) for role in ('train','validation'))
        bank = Bank(Path(gate['root'])/'targets', train, teacher_record=gate['teacher'])
        if bank.manifest['content_hash'] != measured['bank_hash']: raise ValueError('Measured bank differs')
    with log.phase('fresh_fixed30_HLT_CE_plus_D066_embedding'):
        record = fit(root/'fit', gate, train, val, bank, device='cuda', technical=False)
    with log.phase('selected_comparison'):
        _, student = validate_fit(root/'fit', gate, bank.manifest['content_hash'], technical=False)
        result = artifact('RESULTS', parents=dict(spec=spec['content_hash'], fit=record['content_hash']),
            counts=gate['counts'], selection_population='entire_validation', original50k_selection=False,
            single_training_seed=True, rows=comparison(gate, student))
        write_immutable_json(root/'results.json', result)
    storage(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during science')
    receipt = artifact('SCIENCE', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash']),
        job_id=job, outputs=[file_ref(root/p, root=root) for p in ('results.json', 'fit/receipt.json',
            'fit/selected.pt', 'fit/training_report.json', 'fit/validation_predictions.npz')])
    write_immutable_json(root/'receipt.json', receipt)
    return result


def results(spec):
    from .preflight import validate_report
    gate = load_json(spec['gate_path']); report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report); campaign.validate_spec_record(spec, gate, report)
    root = Path(spec['root']); receipt = load_json(root/'receipt.json')
    validate(receipt, 'SCIENCE', dict(spec=spec['content_hash'], preflight=report['content_hash']))
    if {r['path'] for r in receipt['outputs']} != {'results.json', 'fit/receipt.json',
            'fit/selected.pt', 'fit/training_report.json', 'fit/validation_predictions.npz'}:
        raise ValueError('Scientific output coverage differs')
    for ref in receipt['outputs']: validate_file_ref(ref, root=root)
    _, student = validate_fit(root/'fit', gate, report['bank_hash'], technical=False)
    result = load_json(root/'results.json'); record = load_json(root/'fit/receipt.json')
    validate(result, 'RESULTS', dict(spec=spec['content_hash'], fit=record['content_hash']))
    if (result['rows'] != comparison(gate, student) or result['counts'] != gate['counts']
            or result['selection_population'] != 'entire_validation'
            or result['original50k_selection'] is not False or result['single_training_seed'] is not True):
        raise ValueError('Saved comparison differs')
    return result


def print_results(result):
    print('HLT CE + weight5 frozen LOGIT_D066 embedding prediction; NO logit KD.')
    print('Exploratory development validation; NO independent test. Original50k is report-only.')
    for role in ('validation','original50k'):
        print('\n'+role)
        print(f'{"model":24} {"accuracy":>10} {"AUC":>10} {"delta CE":>11} {"delta OFF-w5":>13} {"AUC rec.%":>10}')
        for row in result['rows']:
            metrics=row[role]; rec=row[role+'_recovery']['macro_ovr_auc']
            formatted='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:24} {metrics["accuracy"]:10.6f} {metrics["macro_ovr_auc"]:10.6f} '
                  f'{row[role+"_delta_vs_ce"]:+11.6f} {row[role+"_delta_vs_offline_w5"]:+13.6f} {formatted:>10}')
