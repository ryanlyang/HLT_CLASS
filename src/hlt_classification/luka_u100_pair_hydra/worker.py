"""Exact-job production setup, four fresh fits and partial/final comparison."""
from pathlib import Path
import gc
import time
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from .site import allocation
from hlt_classification.jetclass2_delphes.reporting import recovery
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_contrastive.worker import extended_metrics
from .contracts import ARMS, artifact, validate
from hlt_classification.luka_u100_pair_correction.model import new_model
from hlt_classification.luka_u100_pair_correction import training, resources
from . import campaign, submission, inputs as targets
from .inputs import Bundle


def setup(spec, *, device='cuda'):
    if installed_environment() != spec['reference']['environment']:
        raise ValueError('Installed environment differs from completed matched control; no target work started')
    root = Path(spec['root'])/'setup'; root.mkdir(exist_ok=False)
    started = time.perf_counter()
    train = targets.load_view(spec, 'train', 'D000'); val = targets.load_view(spec, 'validation', 'D000')
    bank = targets.embedding_bank(spec, train)
    load_seconds = time.perf_counter()-started
    if spec.get('schema_version') == 2:
        from .portability import setup_record
        write_immutable_json(root/'portability.json', setup_record(spec, val))
    else:
        parity = control_parity(spec, val, device=device)
        write_immutable_json(root/'control_parity.json', parity)
        require_parity(parity, spec)
    particles = targets.export(spec, train, device=device)
    write_immutable_json(root/'bundle.json', artifact('BUNDLE', parents=dict(spec=spec['content_hash']),
        u100_manifest=file_ref(Bundle(spec['bundle_root']).path('targets/u100/manifest.json')), particle_manifest=file_ref(root/'particles/manifest.json'),
        reused_u100_payload=True, reference_retrained=False, new_teacher_fits=0))
    # Reopen published targets, checking exact payload and alignment before measurements.
    del particles; gc.collect(); particles = targets.particle_bank(spec, train)
    resources.measure(spec, train, val, bank, particles, load_seconds=load_seconds, device=device)


def receipt(spec, task):
    folder = Path(spec['root'])/task; value = load_json(folder/'receipt.json')
    validate(value, 'TASK', dict(spec=spec['content_hash']))
    if (value['task'] != task or value['job_id'] != submission.accepted_jobs(spec, 'setup' if task == 'setup' else 'science').get(task)
            or len({r['path'] for r in value['outputs']}) != len(value['outputs'])):
        raise ValueError('Exact accepted job/output ownership differs')
    for ref in value['outputs']: validate_file_ref(ref, root=folder)
    return value


def completed_setup(spec):
    value = receipt(spec, 'setup')
    required = {'portability.json' if spec.get('schema_version') == 2 else 'control_parity.json',
        'bundle.json', 'particles/manifest.json', 'particles/targets.npz', 'resource_diagnostic.json', 'acceptance.json'}
    required |= {'mini_'+a+'/receipt.json' for a in ARMS}
    if not required <= {r['path'] for r in value['outputs']}:
        raise ValueError('Incomplete shared setup evidence')
    bundle = load_json(Path(spec['root'])/'setup/bundle.json')
    validate(bundle, 'BUNDLE', dict(spec=spec['content_hash']))
    if (bundle['u100_manifest'] != file_ref(Bundle(spec['bundle_root']).path('targets/u100/manifest.json')) or not bundle['reused_u100_payload']
            or bundle['reference_retrained'] is not False or bundle['new_teacher_fits'] != 0
            or Path(bundle['particle_manifest']['path']) != Path(spec['root'])/'setup/particles/manifest.json'):
        raise ValueError('Target reuse/role differs')
    for key in ('u100_manifest', 'particle_manifest'): validate_file_ref(bundle[key])
    if spec.get('schema_version') == 2:
        from .portability import validate_setup
        validate_setup(spec, load_json(Path(spec['root'])/'setup/portability.json'))
    else:
        require_parity(load_json(Path(spec['root'])/'setup/control_parity.json'), spec)
    resources.accepted(spec)
    return value


def run(spec, task):
    if task not in ('setup', 'report', *ARMS): raise ValueError('Unregistered task')
    campaign.check(spec); require_backend()
    job = allocation(spec)
    stage = 'setup' if task == 'setup' else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec, stage).get(task) == job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the exact accepted task ID')
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    if task == 'setup': setup(spec)
    else:
        completed_setup(spec); accepted = resources.accepted(spec)
        if accepted['environment'] != installed_environment() or accepted['gpu'] != gpu_identity():
            raise PermissionError('Measured environment/GPU changed')
        folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
        if task == 'report':
            value = results(spec); write_immutable_json(folder/'results.json', value); print_results(value)
        else:
            train = targets.load_view(spec, 'train', 'D000'); val = targets.load_view(spec, 'validation', 'D000')
            bank, particles = targets.embedding_bank(spec, train), targets.particle_bank(spec, train)
            model = new_model(spec['imported'], task, particles.manifest['statistics'])
            training.fit(folder/'fit', spec, task, model, train, val, bank, particles, device='cuda')
            training.validate_fit(folder/'fit', spec, task)
    campaign.check(spec)
    folder = Path(spec['root'])/task
    outputs = [file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()]
    value = artifact('TASK', parents=dict(spec=spec['content_hash']), task=task, job_id=job, outputs=outputs)
    write_immutable_json(folder/'receipt.json', value)
    return value


def results(spec):
    campaign.check(spec, check_storage=False)
    bundle = Bundle(spec['bundle_root'])
    references = dict(M0HLT='M0HLT', OFFLINE='OFFLINE', U100_W5_MATCHED='matched', U100_W5_HISTORICAL='historical')
    arrays = {name: load_npz_arrays(bundle.path('reference/'+group+'/predictions.npz'))
              for name, group in references.items()}
    version = spec.get('schema_version', 1)
    if version == 2:
        arrays['U100_W5_HYDRA'] = load_npz_arrays(spec['portability_evidence']['refs']['predictions']['path'])
    comparison = 'U100_W5_HYDRA' if version == 2 else 'U100_W5_MATCHED'
    subset = load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],) or len(np.unique(subset)) != len(subset)
            or np.any(subset < 0) or np.any(subset >= spec['counts']['validation'])):
        raise ValueError('Invalid report-only subset')
    selected, pending = {}, []
    for arm in ARMS:
        folder = Path(spec['root'])/arm
        if not (folder/'receipt.json').exists():
            pending.append(arm); continue
        record = receipt(spec, arm)
        if not {'fit/receipt.json', 'fit/training_report.json', 'fit/selected.pt', 'fit/validation_predictions.npz', 'fit/deployment.json'} <= {r['path'] for r in record['outputs']}:
            raise ValueError('Incomplete fit evidence')
        report, array = training.validate_fit(folder/'fit', spec, arm)
        expected = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
            prepared=spec['anchor']['evidence']['prepared'], u100=bundle.json('targets/u100/manifest.json')['content_hash'],
            particles=load_json(Path(spec['root'])/'setup/particles/manifest.json')['content_hash'])
        if report['parents'] != expected: raise ValueError('Fit target parents differ')
        arrays[arm] = array; selected[arm] = report['selected_pass']
    metrics = {}
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], arrays['M0HLT']['identities']) or not np.array_equal(a['labels'], arrays['M0HLT']['labels']):
            raise ValueError('Comparison population/order differs')
        metrics[name] = {r: extended_metrics(a['labels'][ix], a['probabilities'][ix])
                         for r, ix in (('validation', slice(None)), ('original50k', subset))}
    rows = []
    for name, values in metrics.items():
        row = dict(model=name, selected_epoch=selected.get(name), oracle=name == 'OFFLINE', **values)
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(values[role], metrics['M0HLT'][role], metrics['OFFLINE'][role])
            row[role+'_delta_matched'] = values[role]['macro_ovr_auc']-metrics[comparison][role]['macro_ovr_auc']
        rows.append(row)
    return artifact('RESULTS', version=version, parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        pending_meaning='No completed receipt; inspect exact accounting for failure/cancellation',
        comparison=comparison if version == 2 else 'saved_completed_U100_w5_s0_job21993056',
        recovery_anchors='historical_sporc_M0HLT_OFFLINE', final_test_access=False)


def control_parity(spec, val, *, device):
    import torch
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    bundle = Bundle(spec['bundle_root'])
    saved = load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    validate_predictions(saved, spec['counts']['validation'])
    if not np.array_equal(saved['identities'], val.identities) or not np.array_equal(saved['labels'], val.labels):
        raise ValueError('Control parity population/order differs')
    model = DelphesParticleTransformer().to(device).eval()
    model.load_state_dict(torch.load(bundle.path('reference/matched/checkpoint.pt'),
                                    map_location=device, weights_only=True), strict=True)
    actual = predict(model, val, device=device, batch_size=128)
    delta = actual.astype(np.float64)-saved['probabilities'].astype(np.float64)
    result = artifact('CONTROL_PARITY', parents=dict(spec=spec['content_hash']),
        rows=len(val), batch_size=128, precision='original_BF16_prediction_path',
        max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.square(delta).mean())),
        auc_abs=abs(evaluate_probabilities(val.labels, actual)['macro_ovr_auc']-
                    evaluate_probabilities(val.labels, saved['probabilities'])['macro_ovr_auc']),
        checkpoint=file_ref(bundle.path('reference/matched/checkpoint.pt')),
        saved_predictions=file_ref(bundle.path('reference/matched/predictions.npz')),
        environment=installed_environment(), gpu=gpu_identity(), reference_retrained=False)
    del model
    return result


def require_parity(value, spec):
    validate(value, 'CONTROL_PARITY', dict(spec=spec['content_hash']))
    bundle = Bundle(spec['bundle_root'])
    if (value['rows'] != spec['counts']['validation'] or value['batch_size'] != 128
            or value['precision'] != 'original_BF16_prediction_path' or value['reference_retrained'] is not False
            or value['environment'] != spec['reference']['environment']
            or value['checkpoint'] != file_ref(bundle.path('reference/matched/checkpoint.pt'))
            or value['saved_predictions'] != file_ref(bundle.path('reference/matched/predictions.npz'))
            or any(not np.isfinite(value[k]) or not 0 <= value[k] <= v for k, v in spec['parity_tolerances'].items())):
        raise ValueError('Native control numerical portability failed; inspect control_parity.json')
    return value


def print_results(value):
    print('CE1 + U100 embedding5 + retained HLT particle/pair correction. Four fresh students; saved control reused.')
    print('Exploratory full validation; NO independent test. Original50k overlaps validation and is report-only.')
    if value.get('schema_version') == 2:
        print('Delta uses Hydra replay of U100-w5. Recovery anchors remain historical SPORC M0HLT/OFFLINE.')
    for role in ('validation', 'original50k'):
        print('\n'+role)
        print(f'{"model":24} {"epoch":>5} {"accuracy":>10} {"AUC":>12} {"recovery%":>10} {"delta U100-w5":>14}')
        for row in value['rows']:
            rec = row[role+'_recovery']['macro_ovr_auc']
            print(f'{row["model"]:24} {str(row["selected_epoch"] or "-"):>5} {row[role]["accuracy"]:10.6f} '
                  f'{row[role]["macro_ovr_auc"]:12.8f} {rec if rec is not None else float("nan"):10.1f} {row[role+"_delta_matched"]:+14.8f}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))
