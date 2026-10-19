"""One auxiliary fit and compact, re-verifiable CE comparison outputs."""
from pathlib import Path
import shutil
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullsim import regular_cache, alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_attention.worker import data, execution
from .contracts import artifact, validate, recipe
from . import training, targets, campaign, source
from .model import new_model


def storage(gate):
    root = Path(gate['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    remaining = gate['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(root).free < remaining:
        raise OSError('Registered 2 GiB persistent allowance unavailable; no automatic deletion')
    return used


def report_valid(report, gate, bank_hash, *, technical):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe()['content_hash'],
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


def validate_predictions(a, count):
    if (set(a) != {'identities', 'labels', 'probabilities'} or a['identities'].dtype != np.uint8
            or a['identities'].shape != (count, 32) or len(np.unique(a['identities'], axis=0)) != count
            or a['labels'].dtype != np.int64 or a['labels'].shape != (count,)
            or a['probabilities'].dtype != np.float32 or a['probabilities'].shape != (count, 11)
            or not np.isfinite(a['probabilities']).all() or np.any(a['probabilities'] < 0)
            or not np.allclose(a['probabilities'].sum(-1), 1, rtol=0, atol=2e-6)):
        raise ValueError('Invalid prediction schema')


def snapshot_references(gate, val, meta):
    directory = Path(gate['root'])/'preflight/references'
    directory.mkdir(exist_ok=False)
    _, _, inventory, splits = meta
    indices = alljets_data.comparison_indices(inventory, splits, val.identities, val.labels)
    atomic_publish_bytes(directory/'original50k.npz', deterministic_npz_bytes(dict(indices=indices)))
    for name in ('M0HLT', 'OFFLINE'):
        refs = gate['anchor']['evidence']['models'][name]
        validate_file_ref(refs['predictions'])
        a = load_npz_arrays(refs['predictions']['path'])
        validate_predictions(a, len(val))
        if not np.array_equal(a['identities'], val.identities) or not np.array_equal(a['labels'], val.labels):
            raise ValueError('Baseline/teacher and student validation rows differ')
        atomic_publish_bytes(directory/(name+'.npz'), deterministic_npz_bytes(a))
        validate_file_ref(refs['predictions'])


def comparison(gate, student):
    directory = Path(gate['root'])/'preflight/references'
    ix = load_npz_arrays(directory/'original50k.npz')['indices']
    if (ix.dtype != np.int64 or ix.ndim != 1 or len(ix) != gate['original50k_rows'] or len(np.unique(ix)) != len(ix)
            or np.any(ix < 0) or np.any(ix >= gate['counts']['validation'])):
        raise ValueError('Original50k report-only indices differ')
    arrays = {n: load_npz_arrays(directory/(n+'.npz')) for n in ('M0HLT', 'OFFLINE')}
    arrays['HLT_EMBED_AUX'] = student
    rows = []
    for name, a in arrays.items():
        validate_predictions(a, gate['counts']['validation'])
        if not np.array_equal(a['identities'], student['identities']) or not np.array_equal(a['labels'], student['labels']):
            raise ValueError('Comparison identities differ')
        rows.append(dict(model=name, input='OFFLINE ORACLE' if name == 'OFFLINE' else 'single HLT',
            validation=evaluate_probabilities(a['labels'], a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])))
    for row in rows:
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(row[role], rows[0][role], rows[1][role])
        row['auc_delta_vs_ce'] = row['validation']['macro_ovr_auc']-rows[0]['validation']['macro_ovr_auc']
        row['accuracy_delta_vs_ce'] = row['validation']['accuracy']-rows[0]['validation']['accuracy']
    return rows


def run(spec):
    gate, measured = campaign.check_spec(spec)
    job = execution(gate, measured)
    storage(gate)
    root = Path(spec['root'])
    if (root/'receipt.json').exists() or (root/'fit').exists():
        raise FileExistsError('Preserve existing/partial science fit; no automatic restart')
    log = PhaseLog(root, lambda: memory_snapshot(torch))
    with log.phase('load_authenticated_HLT_and_compact_TRAIN_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train = data(gate, 'train', 'D000', meta)
        val = data(gate, 'validation', 'D000', meta)
        bank = targets.Bank(Path(gate['root'])/'targets', train,
                            teacher_ref=gate['anchor']['evidence']['models']['OFFLINE']['checkpoint'])
        if bank.manifest['content_hash'] != measured['bank_hash']:
            raise ValueError('Measured target bank changed')
    with log.phase('scientific_fixed30_fit'):
        fit_record = fit(root/'fit', gate, train, val, bank, device='cuda', technical=False)
    with log.phase('selected_comparison'):
        _, student = validate_fit(root/'fit', gate, bank.manifest['content_hash'], technical=False)
        results = artifact('RESULTS', parents=dict(spec=spec['content_hash'], fit=fit_record['content_hash']),
            counts=gate['counts'], selection_population='entire_validation', original50k_selection=False,
            single_training_seed=True, rows=comparison(gate, student))
        write_immutable_json(root/'results.json', results)
    storage(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during science')
    receipt = artifact('SCIENCE', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash']),
        job_id=job, outputs=[file_ref(p, root=root) for p in sorted(root.rglob('*')) if p.is_file()
                             and not p.name.endswith('.out') and 'submission' not in p.relative_to(root).parts
                             and p.name not in ('campaign_spec.json', 'receipt.json')]
        + [file_ref(root/'fit/receipt.json', root=root)])
    write_immutable_json(root/'receipt.json', receipt)
    return results


def results(spec):
    """Read compact completed outputs without requiring old caches or loading models."""
    gate = load_json(spec['gate_path'])
    from .preflight import validate_report
    measured = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, measured)
    campaign.validate_spec_record(spec, gate, measured)
    root = Path(spec['root'])
    receipt = load_json(root/'receipt.json')
    validate(receipt, 'SCIENCE', dict(spec=spec['content_hash'], preflight=measured['content_hash']))
    names = [r['path'] for r in receipt['outputs']]
    required = {'results.json', 'fit/receipt.json', 'fit/training_report.json',
                'fit/selected.pt', 'fit/validation_predictions.npz'}
    if len(names) != len(set(names)) or not required <= set(names):
        raise ValueError('Science receipt coverage differs')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=root)
    _, student = validate_fit(root/'fit', gate, measured['bank_hash'], technical=False)
    value = load_json(root/'results.json')
    fit_record = load_json(root/'fit/receipt.json')
    validate(value, 'RESULTS', dict(spec=spec['content_hash'], fit=fit_record['content_hash']))
    if value['rows'] != comparison(gate, student) or value['counts'] != gate['counts']:
        raise ValueError('Saved comparison replay differs')
    return value
