"""Production setup/fit/report workers with immutable receipts and exact job ownership."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_u100_multiview.views import load as load_view
from hlt_classification.luka_u100_embedding import targets as u100_targets
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import artifact, validate, ARMS
from . import campaign, submission, targets, training, resources
from .model import new_model


def extended_metrics(labels, probabilities):
    """Extra report-only metrics do not change the historical selection rule."""
    from sklearn.metrics import roc_auc_score
    from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES
    value = evaluate_probabilities(labels, probabilities)
    p = np.asarray(probabilities, np.float64); p /= p.sum(axis=1, keepdims=True)
    value['per_class_auc'] = {name: float(roc_auc_score(labels == c, p[:, c]))
                              for c, name in enumerate(CLASS_NAMES)}
    qcd = labels == 0; background = int(qcd.sum()); extra = {}
    for c, name in enumerate(CLASS_NAMES[1:], 1):
        denominator = p[:, c]+p[:, 0]
        score = np.divide(p[:, c], denominator, out=np.zeros(len(p)), where=denominator > 0)
        signal = np.sort(score[labels == c])[::-1]
        threshold = float(signal[math.ceil(.3*len(signal))-1])
        count = int(np.count_nonzero(score[qcd] >= threshold))
        extra[name] = dict(rejection=background/count if count else None, background_count=background,
            background_passing=count, threshold=threshold, zero_fpr_censored=count == 0,
            one_event_resolution_rejection=background, achieved_signal_efficiency=float(np.mean(signal >= threshold)))
    value['rejection_at_30pct'] = extra
    return value


def readback(teacher, view, reference, device):
    validate_file_ref(reference)
    expected = load_npz_arrays(reference['path']); validate_predictions(expected, len(view))
    if view.role != 'validation' or not np.array_equal(view.identities, expected['identities']) or not np.array_equal(view.labels, expected['labels']):
        raise ValueError('Teacher readback population differs')
    actual = predict(teacher, view, device=device, batch_size=128)
    np.testing.assert_allclose(actual, expected['probabilities'], rtol=1e-5, atol=1e-6)
    return dict(native_probability_parity=True, rows=len(view), coordinate=view.coordinate_name,
                rtol=1e-5, atol=1e-6, maximum_absolute_error=float(np.max(np.abs(actual-expected['probabilities']))))


def setup(spec, *, device='cuda'):
    root = Path(spec['root'])/'setup'; root.mkdir(exist_ok=False)
    start = time.perf_counter()
    train = load_view(spec, 'train', 'D000'); val = load_view(spec, 'validation', 'D000')
    load_seconds = time.perf_counter()-start
    imported = spec['imported']; urecord = imported['teacher']
    old_manifest = Path(imported['files']['bank_manifest']['path'])
    old_payload = old_manifest.parent/'targets.npz'
    if old_payload.exists():
        # Validate before reuse, including original teacher and exact ordered TRAIN join.
        u100_targets.Bank(old_manifest.parent, train, teacher_record=urecord)
        manifest = old_manifest
        reused = True
    else:
        uview = load_view(spec, 'train', 'U100')
        teacher = DelphesParticleTransformer().to(device).eval()
        validate_file_ref(urecord['files']['checkpoint'])
        teacher.load_state_dict(torch.load(urecord['files']['checkpoint']['path'], map_location=device, weights_only=True), strict=True)
        uval = load_view(spec, 'validation', 'U100')
        proof = readback(teacher, uval, urecord['files']['predictions'], device)
        write_immutable_json(root/'u100_readback.json', artifact('TEACHER_READBACK',
            parents=dict(spec=spec['content_hash'], teacher=urecord['content_hash']), **proof))
        del uval
        u100_targets.export(root/'U100', teacher, uview, teacher_record=urecord, device=device)
        manifest = root/'U100/manifest.json'; reused = False
        del teacher, uview; gc.collect(); torch.cuda.empty_cache()
    print(f'U100 targets authenticated; existing payload reused={reused}', flush=True)
    offline_refs = spec['anchor']['evidence']['models']['OFFLINE']
    teacher = DelphesParticleTransformer().to(device).eval()
    validate_file_ref(offline_refs['checkpoint'])
    teacher.load_state_dict(torch.load(offline_refs['checkpoint']['path'], map_location=device, weights_only=True), strict=True)
    oval = load_view(spec, 'validation', 'OFFLINE')
    proof = readback(teacher, oval, offline_refs['predictions'], device)
    write_immutable_json(root/'offline_readback.json', artifact('TEACHER_READBACK',
        parents=dict(spec=spec['content_hash'], imported=imported['content_hash']), **proof))
    del oval; gc.collect()
    otrain = load_view(spec, 'train', 'OFFLINE')
    targets.export(root/'OFFLINE', teacher, otrain, train, offline_refs['checkpoint'],
        spec_hash=spec['content_hash'], imported_hash=imported['content_hash'], device=device)
    del teacher, otrain; gc.collect(); torch.cuda.empty_cache()
    bundle = artifact('BUNDLE', parents=dict(spec=spec['content_hash']),
        u100_manifest=file_ref(manifest), offline_manifest=file_ref(root/'OFFLINE/manifest.json'),
        reused_original_u100_payload=reused, teachers_retrained=False, setup_load_seconds=load_seconds)
    write_immutable_json(root/'bundle.json', bundle)
    u100, offline = targets.u100_bank(spec, train), targets.key_bank(spec, train)
    resources.measure(spec, train, val, u100, offline, load_seconds=load_seconds, device=device)


def _receipt(spec, task, *, metadata_only_keys=False):
    folder = Path(spec['root'])/task; value = load_json(folder/'receipt.json')
    validate(value, 'TASK', dict(spec=spec['content_hash']))
    if value['task'] != task or value['job_id'] != submission.accepted_jobs(spec, 'setup' if task == 'setup' else 'science').get(task):
        raise ValueError('Receipt task/exact accepted job binding differs')
    paths = [r['path'] for r in value['outputs']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate output inventory')
    for ref in value['outputs']:
        if metadata_only_keys and ref['path'] == 'OFFLINE/keys.npz':
            continue
        validate_file_ref(ref, root=folder)
    return value


def completed_setup(spec, *, metadata_only_keys=False):
    value = _receipt(spec, 'setup', metadata_only_keys=metadata_only_keys)
    names = {r['path'] for r in value['outputs']}
    if not {'bundle.json', 'acceptance.json', 'resource_diagnostic.json', 'capacity.json',
            'OFFLINE/keys.npz', 'OFFLINE/manifest.json', 'offline_readback.json', 'baseline_parity.json'} <= names:
        raise ValueError('Incomplete setup receipt')
    bundle = load_json(Path(spec['root'])/'setup/bundle.json')
    validate(bundle, 'BUNDLE', dict(spec=spec['content_hash']))
    for field in ('u100_manifest', 'offline_manifest'):
        validate_file_ref(bundle[field])
    resources.accepted(spec)
    return value


def run(spec, task):
    if task not in ('setup', 'report', *ARMS):
        raise ValueError('Unknown task')
    campaign.check(spec); require_backend()
    job, cpus, memory = allocation(spec['execution_site'])
    if (cpus, memory) != (spec['cpus'], spec['memory_mb']):
        raise PermissionError('Allocation shape differs')
    stage = 'setup' if task == 'setup' else 'science'
    # sbatch can start a worker before its accepted response is journalled.
    for _ in range(12):
        if submission.accepted_jobs(spec, stage).get(task) == job:
            break
        time.sleep(5)
    else:
        raise PermissionError('Allocation is not the exact accepted campaign task')
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    if task == 'setup':
        setup(spec)
    else:
        completed_setup(spec, metadata_only_keys=task in ARMS and ARMS[task]['coefficient'] == 0)
        accepted = resources.accepted(spec)
        if accepted['environment'] != installed_environment() or accepted['gpu'] != gpu_identity():
            raise PermissionError('Measured environment/GPU differs')
        folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
        if task == 'report':
            value = results(spec); write_immutable_json(folder/'results.json', value); print_results(value)
        else:
            train = load_view(spec, 'train', 'D000'); val = load_view(spec, 'validation', 'D000')
            bank = targets.u100_bank(spec, train)
            sampler, key_hash = None, None
            if ARMS[task]['coefficient'] != 0:
                offline = targets.key_bank(spec, train)
                sampler = resources.make_sampler(offline, task, 'cuda'); key_hash = offline.manifest['content_hash']
            model = new_model(spec['imported'], task)
            training.fit(folder/'fit', spec, task, model, train, val, bank, sampler,
                         device='cuda', key_hash=key_hash)
            training.validate_fit(folder/'fit', spec, task)
    campaign.check(spec)
    folder = Path(spec['root'])/task
    outputs = [file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()]
    value = artifact('TASK', parents=dict(spec=spec['content_hash']), task=task, job_id=job, outputs=outputs)
    write_immutable_json(folder/'receipt.json', value)
    return value


def results(spec):
    campaign.check(spec, check_storage=False)
    imported = spec['imported']
    references = {n: spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT', 'OFFLINE')}
    references.update(U100_W5_HISTORICAL=imported['files']['predictions'])
    arrays = {}
    for name, ref in references.items():
        validate_file_ref(ref); arrays[name] = load_npz_arrays(ref['path'])
    validate_file_ref(imported['files']['subset'])
    subset = load_npz_arrays(imported['files']['subset']['path'])['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],)
            or len(np.unique(subset)) != len(subset) or np.any(subset < 0) or np.any(subset >= spec['counts']['validation'])):
        raise ValueError('Invalid report-only original50k indices')
    pending, selected = [], {}
    for arm in ARMS:
        folder = Path(spec['root'])/arm
        if not (folder/'receipt.json').exists():
            pending.append(arm); continue
        record = _receipt(spec, arm)
        if not {'fit/receipt.json', 'fit/training_report.json', 'fit/selected.pt', 'fit/validation_predictions.npz'} <= {x['path'] for x in record['outputs']}:
            raise ValueError('Task is missing selected fit evidence')
        report, a = training.validate_fit(folder/'fit', spec, arm)
        # Validate bank hashes without opening training/model arrays.
        bundle = load_json(Path(spec['root'])/'setup/bundle.json')
        parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
            prepared=spec['anchor']['evidence']['prepared'], u100=load_json(bundle['u100_manifest']['path'])['content_hash'])
        if ARMS[arm]['coefficient']:
            parents['offline'] = load_json(bundle['offline_manifest']['path'])['content_hash']
        if report['parents'] != parents:
            raise ValueError('Report target lineage differs')
        arrays[arm] = a; selected[arm] = report['selected_pass']
    metrics = {}
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], arrays['M0HLT']['identities']) or not np.array_equal(a['labels'], arrays['M0HLT']['labels']):
            raise ValueError('Comparison population/order differs')
        metrics[name] = {role: extended_metrics(a['labels'][ix], a['probabilities'][ix])
                         for role, ix in (('validation', slice(None)), ('original50k', subset))}
    rows = []
    for name, values in metrics.items():
        row = dict(model=name, selected_epoch=selected.get(name), oracle=name == 'OFFLINE', **values)
        control = 'u100_w5_s'+str(ARMS[name]['replicate']) if name in ARMS else None
        row['matched_control'] = control
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(values[role], metrics['M0HLT'][role], metrics['OFFLINE'][role])
            row[role+'_delta_matched'] = (values[role]['macro_ovr_auc']-metrics[control][role]['macro_ovr_auc']
                                           if control in metrics else None)
        rows.append(row)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        pending_meaning='no completed receipt; consult exact accounting for failure/cancellation',
        comparison='matched_replicate_U100_w5', validation_is_exploratory=True)


def print_results(value):
    print('CE1 + U100 embedding5 + independent OFFLINE jet identification. Native HLT inference.')
    print('Exploratory validation; NO independent test. Original50k overlaps validation and is report-only.')
    for role in ('validation', 'original50k'):
        print('\n'+role)
        print(f'{"model":24} {"epoch":>5} {"accuracy":>10} {"AUC":>12} {"recovery%":>10} {"delta matched":>14}')
        for row in value['rows']:
            delta = row[role+'_delta_matched']; rec = row[role+'_recovery']['macro_ovr_auc']
            print(f'{row["model"]:24} {str(row["selected_epoch"] or "-"):>5} {row[role]["accuracy"]:10.6f} {row[role]["macro_ovr_auc"]:12.8f} '
                  f'{rec if rec is not None else float("nan"):10.1f} {"--" if delta is None else f"{delta:+.8f}":>14}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))
