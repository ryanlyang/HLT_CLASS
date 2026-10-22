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
from hlt_classification.luka_fullsim import alljets_data
from hlt_classification.luka_attention.worker import data, execution
from .contracts import artifact, validate, recipe
from . import training
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
    folder = Path(gate['root'])/'references' if folder is None else Path(folder)
    folder.mkdir(exist_ok=False)
    ix = alljets_data.comparison_indices(meta[2], meta[3], val.identities, val.labels)
    atomic_publish_bytes(folder/'original50k.npz', deterministic_npz_bytes(dict(indices=ix)))
    refs = {n: gate['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT','OFFLINE')}
    refs.update(OFFLINE_EMBED_W5=gate['reference']['files']['predictions'],
                COARSE_U100=gate['teacher']['files']['predictions'])
    for name, ref in refs.items():
        validate_file_ref(ref); a = load_npz_arrays(ref['path'])
        validate_predictions(a, len(val))
        if not np.array_equal(a['identities'], val.identities) or not np.array_equal(a['labels'], val.labels):
            raise ValueError('Reference population/order differs')
        atomic_publish_bytes(folder/(name+'.npz'), deterministic_npz_bytes(a))
        validate_file_ref(ref)


def comparison(gate, student, *, folder=None):
    from hlt_classification.jetclass2_delphes.reporting import recovery
    folder = Path(gate['root'])/'references' if folder is None else Path(folder)
    ix = load_npz_arrays(folder/'original50k.npz')['indices']
    if (ix.dtype != np.int64 or ix.shape != (gate['original50k_rows'],)
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(student['labels']))):
        raise ValueError('Original50k report-only indices differ')
    arrays = {n: load_npz_arrays(folder/(n+'.npz')) for n in
              ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','COARSE_U100')}
    arrays['HLT_U100_EMBED_W5'] = student
    rows = []
    for name, a in arrays.items():
        validate_predictions(a, gate['counts']['validation'])
        if not np.array_equal(a['identities'], student['identities']) or not np.array_equal(a['labels'], student['labels']):
            raise ValueError('Comparison identities/labels differ')
        rows.append(dict(model=name, input='ORACLE' if name in ('OFFLINE','COARSE_U100') else 'single HLT',
            validation=evaluate_probabilities(a['labels'], a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])))
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery'] = recovery(row[role], rows[0][role], rows[1][role])
            row[role+'_delta_vs_ce'] = row[role]['macro_ovr_auc']-rows[0][role]['macro_ovr_auc']
            row[role+'_delta_vs_offline_w5'] = row[role]['macro_ovr_auc']-rows[2][role]['macro_ovr_auc']
    return rows


def print_results(result):
    print('HLT CE + weight5 frozen COARSE_U100 embedding prediction; NO logit KD.')
    print('Exploratory development validation; NO independent test. Original50k is report-only.')
    for role in ('validation','original50k'):
        print('\n'+role)
        print(f'{"model":24} {"accuracy":>10} {"AUC":>10} {"delta CE":>11} {"delta OFF-w5":>13} {"AUC rec.%":>10}')
        for row in result['rows']:
            metrics=row[role]; rec=row[role+'_recovery']['macro_ovr_auc']
            formatted='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:24} {metrics["accuracy"]:10.6f} {metrics["macro_ovr_auc"]:10.6f} '
                  f'{row[role+"_delta_vs_ce"]:+11.6f} {row[role+"_delta_vs_offline_w5"]:+13.6f} {formatted:>10}')


def teacher_readback(teacher, val, record, device):
    if val.role != 'validation' or val.coordinate_name != 'U100':
        raise PermissionError('Teacher readback requires its original U100 validation view')
    ref = record['files']['predictions']; validate_file_ref(ref)
    saved = load_npz_arrays(ref['path']); validate_predictions(saved, len(val))
    if not np.array_equal(saved['identities'], val.identities) or not np.array_equal(saved['labels'], val.labels):
        raise ValueError('Teacher validation identities differ')
    actual = predict(teacher, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, saved['probabilities'], rtol=1e-5, atol=1e-6)
    validate_file_ref(ref)
    return artifact('TEACHER_READBACK', parents=dict(teacher=record['content_hash']),
        rows=len(val), coordinate='U100', role='validation', normalization_used=False,
        native_probability_parity=True, rtol=1e-5, atol=1e-6,
        maximum_absolute_error=float(np.max(np.abs(actual-saved['probabilities']))))
