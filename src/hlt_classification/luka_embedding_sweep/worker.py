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
from .contracts import WEIGHTS, arm_gate
from hlt_classification.luka_embedding_weight.worker import (
    storage, validate_predictions, snapshot_references, comparison, load_bank,
)
from hlt_classification.luka_embedding_aux.model import new_model


def report_valid(report, gate, bank_hash, *, technical):
    if gate['recipe'] != recipe(gate['node']['auxiliary_weight']):
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


def run(spec, arm):
    gate, measured = campaign.check_spec(spec)
    scoped = arm_gate(gate, arm)
    job = execution(gate, measured)
    storage(gate)
    root = Path(spec['root'])/arm
    root.mkdir(exist_ok=False)
    log = PhaseLog(root, lambda: memory_snapshot(torch))
    with log.phase('load_authenticated_HLT_and_shared_TRAIN_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train = data(gate, 'train', 'D000', meta)
        val = data(gate, 'validation', 'D000', meta)
        bank = load_bank(gate, train)
        if bank.manifest['content_hash'] != measured['bank_hash']:
            raise ValueError('Measured bank changed')
    with log.phase('scientific_fixed30_fit'):
        record = fit(root/'fit', scoped, train, val, bank, device='cuda', technical=False)
    with log.phase('selected_comparison'):
        _, student = validate_fit(root/'fit', scoped, bank.manifest['content_hash'], technical=False)
        result = artifact('RESULTS', parents=dict(spec=spec['content_hash'], fit=record['content_hash']),
            arm=arm, weight=WEIGHTS[arm], counts=gate['counts'], single_training_seed=True,
            selection_population='entire_validation', original50k_selection=False,
            rows=comparison(scoped, student))
        write_immutable_json(root/'results.json', result)
    storage(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during science')
    outputs = [file_ref(root/f, root=root) for f in ('results.json', 'fit/receipt.json',
               'fit/selected.pt', 'fit/training_report.json', 'fit/validation_predictions.npz')]
    receipt = artifact('SCIENCE', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash']),
                       arm=arm, weight=WEIGHTS[arm], job_id=job, outputs=outputs)
    write_immutable_json(root/'receipt.json', receipt)
    return result


def results(spec):
    """Read partial or complete comparisons, without loading a model/old cache."""
    gate = load_json(spec['gate_path'])
    from .preflight import validate_report
    measured = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, measured)
    campaign.validate_spec_record(spec, gate, measured)
    rows, pending = [], []
    for arm in WEIGHTS:
        root = Path(spec['root'])/arm
        if not (root/'receipt.json').is_file():
            pending.append(arm)
            continue
        receipt = load_json(root/'receipt.json')
        validate(receipt, 'SCIENCE', dict(spec=spec['content_hash'], preflight=measured['content_hash']))
        required = {'results.json', 'fit/receipt.json', 'fit/selected.pt',
                    'fit/training_report.json', 'fit/validation_predictions.npz'}
        if (receipt['arm'] != arm or receipt['weight'] != WEIGHTS[arm]
                or len(receipt['outputs']) != len(required)
                or {r['path'] for r in receipt['outputs']} != required):
            raise ValueError('Science receipt differs')
        for ref in receipt['outputs']:
            validate_file_ref(ref, root=root)
        scoped = arm_gate(gate, arm)
        _, student = validate_fit(root/'fit', scoped, measured['bank_hash'], technical=False)
        result = load_json(root/'results.json')
        fit_record = load_json(root/'fit/receipt.json')
        validate(result, 'RESULTS', dict(spec=spec['content_hash'], fit=fit_record['content_hash']))
        if (result['rows'] != comparison(scoped, student) or result['counts'] != gate['counts']
                or result['arm'] != arm or result['weight'] != WEIGHTS[arm]
                or result['selection_population'] != 'entire_validation'
                or result['original50k_selection'] is not False or result['single_training_seed'] is not True):
            raise ValueError('Saved comparison replay differs')
        if not rows:
            rows.extend(result['rows'][:3])
        elif rows[:3] != result['rows'][:3]:
            raise ValueError('Comparison reference rows differ')
        rows.append(result['rows'][-1])
    return dict(rows=rows, pending=pending, independent_test_available=False)
