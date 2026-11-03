"""Native HLT workers and combined, explicitly labelled historical/new results."""
from pathlib import Path
import time
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.reporting import recovery
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_u100_multiview.views import load as load_view
from hlt_classification.luka_u100_contrastive import targets, resources
from hlt_classification.luka_u100_contrastive.model import new_model
from hlt_classification.luka_u100_contrastive.worker import extended_metrics
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import ARMS, ORDER, artifact, validate
from . import campaign, submission, training


def receipt(spec, task):
    folder = Path(spec['root'])/task; value = load_json(folder/'receipt.json')
    validate(value, 'TASK', dict(spec=spec['content_hash']))
    if value['task'] != task or value['job_id'] != submission.accepted_jobs(spec).get(task):
        raise ValueError('Receipt/accepted job binding differs')
    paths = [r['path'] for r in value['outputs']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate output inventory')
    for ref in value['outputs']:
        validate_file_ref(ref, root=folder)
    return value


def run(spec, task):
    if task not in (*spec['remaining'], 'report'):
        raise ValueError('Only registered remaining tasks may run')
    old = campaign.check(spec); require_backend()
    job, cpus, memory = allocation(spec['execution_site'])
    if (cpus, memory) != (spec['cpus'], spec['memory_mb']):
        raise PermissionError('Allocation shape differs')
    for _ in range(12):
        if submission.accepted_jobs(spec).get(task) == job:
            break
        time.sleep(5)
    else:
        raise PermissionError('Not the exact accepted task allocation')
    accepted = spec['parent']['acceptance']
    if accepted['environment'] != installed_environment() or accepted['gpu'] != gpu_identity():
        raise PermissionError('Measured original environment/GPU differs')
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    if task == 'report':
        value = results(spec); write_immutable_json(folder/'results.json', value); print_results(value)
    else:
        train = load_view(spec, 'train', 'D000'); val = load_view(spec, 'validation', 'D000')
        bank = targets.u100_bank(old, train); sampler = None; key_hash = None
        if ARMS[task]['coefficient']:
            offline = targets.key_bank(old, train)
            sampler = resources.make_sampler(offline, task, 'cuda'); key_hash = offline.manifest['content_hash']
        model = new_model(spec['imported'], task)
        training.fit(folder/'fit', spec, task, model, train, val, bank, sampler, device='cuda', key_hash=key_hash)
        training.validate_fit(folder/'fit', spec, task)
    campaign.check(spec)
    value = artifact('TASK', parents=dict(spec=spec['content_hash']), task=task, job_id=job,
        outputs=[file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json', value)
    return value


def results(spec):
    old = campaign.check(spec, check_storage=False); imported = spec['imported']
    references = {n: spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT', 'OFFLINE')}
    references['U100_W5_HISTORICAL'] = imported['files']['predictions']
    arrays, regimes, selected, epochs, stop_reason = {}, {}, {}, {}, {}
    for name, ref in references.items():
        validate_file_ref(ref); arrays[name] = load_npz_arrays(ref['path']); regimes[name] = 'reference'
    for arm, record in spec['parent']['completed'].items():
        report = load_json(record['report']['path']); arrays[arm] = load_npz_arrays(record['predictions']['path'])
        regimes[arm] = 'original30'; selected[arm] = report['selected_pass']; epochs[arm] = report['epochs']
        stop_reason[arm] = 'historical_full_budget'
    pending = []
    bundle = load_json(Path(old['root'])/'setup/bundle.json')
    for arm in spec['remaining']:
        folder = Path(spec['root'])/arm
        if not (folder/'receipt.json').exists():
            pending.append(arm); continue
        record = receipt(spec, arm)
        if not {'fit/receipt.json','fit/training_report.json','fit/selected.pt','fit/validation_predictions.npz'} <= {x['path'] for x in record['outputs']}:
            raise ValueError('Incomplete fit artifact coverage')
        report, arrays[arm] = training.validate_fit(folder/'fit', spec, arm)
        parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
            prepared=spec['anchor']['evidence']['prepared'], u100=load_json(bundle['u100_manifest']['path'])['content_hash'])
        if ARMS[arm]['coefficient']:
            parents['offline'] = load_json(bundle['offline_manifest']['path'])['content_hash']
        if report['parents'] != parents:
            raise ValueError('Report bank lineage differs')
        regimes[arm] = 'patience5'; selected[arm] = report['selected_pass']; epochs[arm] = report['epochs']
        stop_reason[arm] = report['stopping']['reason']
    validate_file_ref(imported['files']['subset'])
    subset = load_npz_arrays(imported['files']['subset']['path'])['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],)
            or len(np.unique(subset)) != len(subset) or np.any(subset < 0) or np.any(subset >= spec['counts']['validation'])):
        raise ValueError('Invalid original50k indices')
    metrics = {}
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], arrays['M0HLT']['identities']) or not np.array_equal(a['labels'], arrays['M0HLT']['labels']):
            raise ValueError('Comparison population/order differs')
        metrics[name] = {role: extended_metrics(a['labels'][ix], a['probabilities'][ix])
                        for role, ix in (('validation', slice(None)), ('original50k', subset))}
    rows = []
    for name in [*references, *[a for a in ORDER if a in arrays]]:
        values = metrics[name]; control = 'u100_w5_s'+str(ARMS[name]['replicate']) if name in ARMS else None
        row = dict(model=name, regime=regimes[name], selected_epoch=selected.get(name),
            epochs_run=epochs.get(name), stop_reason=stop_reason.get(name), matched_control=control, **values)
        for role in ('validation','original50k'):
            row[role+'_recovery'] = recovery(values[role], metrics['M0HLT'][role], metrics['OFFLINE'][role])
            row[role+'_delta_matched'] = values[role]['macro_ovr_auc']-metrics[control][role]['macro_ovr_auc'] if control in metrics else None
        rows.append(row)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        comparison='matched_seed_controls_with_explicit_original30_or_patience5_regime',
        pending_meaning='missing completed receipt; consult exact accounting', independent_test=False)


def print_results(value):
    print('CE1 + U100 embedding5 + OFFLINE identification. Single debug lane; new fits min15/patience5/max30.')
    print('Original30 rows are reused historical runs. Exploratory validation; original50k overlaps; NO independent test.')
    for role in ('validation','original50k'):
        print('\n'+role)
        print(f'{"model":24} {"regime":11} {"epochs":>6} {"pick":>4} {"AUC":>12} {"rec%":>7} {"delta control":>14}')
        for row in value['rows']:
            delta = row[role+'_delta_matched']; rec = row[role+'_recovery']['macro_ovr_auc']
            print(f'{row["model"]:24} {row["regime"]:11} {str(row["epochs_run"] or "-"):>6} '
                  f'{str(row["selected_epoch"] or "-"):>4} {row[role]["macro_ovr_auc"]:12.8f} '
                  f'{rec if rec is not None else float("nan"):7.1f} {"--" if delta is None else f"{delta:+.8f}":>14}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))
