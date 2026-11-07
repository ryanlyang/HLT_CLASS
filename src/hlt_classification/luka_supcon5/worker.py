"""Real allocation workers and read-only completed/provisional recovery reports."""
from pathlib import Path
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullsim.preflight_v2 import require_backend, memory_snapshot
from hlt_classification.luka_embedding_aux import targets as offline_targets
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from hlt_classification.luka_u100_contrastive.worker import extended_metrics
from .contracts import TEACHERS, STUDENTS, FITS, artifact, validate
from . import campaign, submission, inputs, training, resources


def receipt(spec, task):
    folder = Path(spec['root'])/task; value = load_json(folder/'receipt.json')
    validate(value, 'TASK', dict(spec=spec['content_hash']))
    stage = 'setup' if task == 'setup' else 'science'
    if (value['task'] != task or value['job_id'] != submission.accepted_jobs(spec, stage).get(task)
            or len({r['path'] for r in value['outputs']}) != len(value['outputs'])
            or value['environment'] != spec['reference']['environment']):
        raise ValueError('Exact task/job/environment ownership differs')
    for ref in value['outputs']:
        validate_file_ref(ref, root=folder)
    names = {r['path'] for r in value['outputs']}
    required = {'acceptance.json', 'resource_diagnostic.json'} if task == 'setup' else {'results.json'} if task == 'report' else {
        'fit/selected.pt', 'fit/selected_full.pt', 'fit/training_report.json', 'fit/validation_predictions.npz', 'fit/receipt.json'}
    if task in TEACHERS:
        required |= {'bank/manifest.json', 'bank/targets.npz'}
    if task in STUDENTS:
        required |= {'startup/receipt.json', 'startup/training_report.json', 'startup_memory.json'}
    if task == 'setup':
        required |= {'mini_'+t+'/receipt.json' for t in FITS}
    if not required <= names:
        raise ValueError('Incomplete task receipt')
    return value


def completed_setup(spec):
    value = receipt(spec, 'setup'); resources.accepted(spec)
    for task in FITS:
        training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task), spec, task, technical=True)
    return value


def validate_teacher(spec, task):
    if task not in TEACHERS:
        raise ValueError('Unknown teacher')
    receipt(spec, task)
    folder = Path(spec['root'])/task
    report, arrays = training.validate_fit(folder/'fit', spec, task)
    m = load_json(folder/'bank/manifest.json'); ref = file_ref(folder/'fit/selected.pt')
    offline_targets.validate(m, 'BANK', dict(recipe=offline_targets.recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'], teacher=ref['sha256']))
    if (m['teacher'] != ref or m['role'] != 'train' or m['coordinate'] != 'OFFLINE'
            or m['width'] != 128 or m['rows'] != spec['counts']['train'] or m['payload']['path'] != 'targets.npz'):
        raise ValueError('Selected teacher target semantics differ')
    validate_file_ref(m['payload'], root=folder/'bank')
    return report, arrays


def fit_parents(spec, task):
    parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'])
    if task in STUDENTS:
        validate_teacher(spec, STUDENTS[task])
        ref = spec['reference']['u100_manifest']; validate_file_ref(ref)
        parents['u100'] = load_json(ref['path'])['content_hash']
        parents['offline'] = load_json(Path(spec['root'])/STUDENTS[task]/'bank/manifest.json')['content_hash']
    return parents


def run(spec, task):
    if task not in ('setup', 'report', *FITS):
        raise ValueError('Unregistered task')
    campaign.check(spec); require_backend()
    job, cpus, memory = allocation(spec['execution_site'])
    if (cpus, memory) != (spec['cpus'], spec['memory_mb']):
        raise PermissionError('Allocation shape differs')
    stage = 'setup' if task == 'setup' else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec, stage).get(task) == job:
            break
        time.sleep(5)
    else:
        raise PermissionError('Worker is not exact accepted job')
    if installed_environment() != spec['reference']['environment']:
        raise ValueError('Installed environment changed')
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    if task == 'setup':
        resources.measure(spec)
    else:
        completed_setup(spec); accepted = resources.accepted(spec)
        if accepted['gpu'] != gpu_identity():
            raise PermissionError('GPU differs from measured allocation')
        if task == 'report':
            value = results(spec); write_immutable_json(folder/'results.json', value); print_results(value)
        else:
            train, val = inputs.views(spec, task in TEACHERS); banks = inputs.banks(spec, task, train)
            if task in STUDENTS:
                torch.cuda.reset_peak_memory_stats()
                resources.miniature(folder/'startup', spec, task, train, val, banks, 'cuda')
                memory = memory_snapshot(torch)
                write_immutable_json(folder/'startup_memory.json', artifact('STARTUP', parents=fit_parents(spec, task),
                    memory=memory, real_selected_teacher_targets=True, technical_weights_discarded=True))
                if (memory['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
                        or memory['gpu_peak_bytes'] >= .88*accepted['gpu']['total_memory_bytes']):
                    raise MemoryError('Actual student targets exceed memory allowance')
                resources.free()
            native, report = training.fit(folder/'fit', spec, task, train, val, banks, device='cuda')
            if task in TEACHERS:
                with torch.autocast('cuda', enabled=False):
                    offline_targets.export(folder/'bank', native, train, teacher_ref=file_ref(folder/'fit/selected.pt'), device='cuda')
                # Authenticate payload, TRAIN normalization and ordered identity join before publication.
                offline_targets.Bank(folder/'bank', train, teacher_ref=file_ref(folder/'fit/selected.pt'))
            elif report['parents'] != fit_parents(spec, task):
                raise ValueError('Scientific target parents differ')
    campaign.check(spec)
    value = artifact('TASK', parents=dict(spec=spec['content_hash']), task=task, job_id=job,
        environment=installed_environment(), gpu=gpu_identity(),
        outputs=[file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json', value)
    return value


def results(spec):
    campaign.check(spec, check_storage=False)
    references = {k: spec['anchor']['evidence']['models'][k]['predictions'] for k in ('M0HLT', 'OFFLINE')}
    references['U100_W5'] = spec['reference']['files']['predictions']
    arrays = {}; selected = {}; pending = []
    for name, ref in references.items():
        validate_file_ref(ref); arrays[name] = load_npz_arrays(ref['path'])
    ref = spec['imported']['files']['subset']; validate_file_ref(ref)
    subset = load_npz_arrays(ref['path'])['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],) or len(np.unique(subset)) != len(subset)
            or np.any(subset < 0) or np.any(subset >= spec['counts']['validation'])):
        raise ValueError('Invalid report-only subset')
    for task in FITS:
        folder = Path(spec['root'])/task
        if not (folder/'receipt.json').exists():
            pending.append(task); continue
        receipt(spec, task)
        report, a = validate_teacher(spec, task) if task in TEACHERS else training.validate_fit(folder/'fit', spec, task)
        if report['parents'] != fit_parents(spec, task):
            raise ValueError('Report target lineage differs')
        arrays[task] = a; selected[task] = report['selected_epoch']
    metrics = {}
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], arrays['M0HLT']['identities']) or not np.array_equal(a['labels'], arrays['M0HLT']['labels']):
            raise ValueError('Comparison populations differ')
        metrics[name] = {role: extended_metrics(a['labels'][ix], a['probabilities'][ix]) for role, ix in (
            ('validation', slice(None)), ('original50k', subset))}
    rows = []
    for name, values in metrics.items():
        row = dict(model=name, epoch=selected.get(name), oracle=name == 'OFFLINE' or name in TEACHERS, **values)
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(values[role], metrics['M0HLT'][role], metrics['OFFLINE'][role])['macro_ovr_auc']
            row[role+'_delta_u100'] = values[role]['macro_ovr_auc'] - metrics['U100_W5'][role]['macro_ovr_auc']
        rows.append(row)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending)


def print_results(value):
    print('One OFFLINE CE+5*SupCon oracle and one fresh HLT CE+U100-w5+OFFLINE-w5 student.')
    print('Historical recovery anchors; original50k overlaps validation. No independent test.')
    for role in ('validation', 'original50k'):
        print('\n'+role)
        print(f'{"model":24} {"epoch":>5} {"AUC":>12} {"recovery%":>10} {"delta U100":>12}  role')
        for row in value['rows']:
            rec = row[role+'_recovery']
            print(f'{row["model"]:24} {str(row["epoch"] or "-"):>5} {row[role]["macro_ovr_auc"]:12.8f} '
                  f'{rec if rec is not None else float("nan"):10.1f} {row[role+"_delta_u100"]:+12.8f}  '
                  f'{"OFFLINE oracle" if row["oracle"] else "HLT"}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))


def epochs(spec):
    campaign.check(spec, check_storage=False)
    ref = spec['reference']['files']['predictions']; validate_file_ref(ref)
    a = load_npz_arrays(ref['path']); baseline = evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc']
    print('Provisional validation histories; teacher is an OFFLINE oracle. No inference or job changes.')
    for task in FITS:
        paths = sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json'))
        if not paths:
            print(task+': no published epoch'); continue
        parents = fit_parents(spec, task); best = None
        for e, path in enumerate(paths, 1):
            row = load_json(path); validate(row, 'EPOCH', parents)
            if row['task'] != task or row['epoch'] != e or row['technical'] is not False or row['validation']['rows'] != spec['counts']['validation']:
                raise ValueError('Epoch meaning differs')
            auc = row['validation']['macro_ovr_auc']; rec = recovery_percent(auc, spec['recovery_anchors'])
            print(f'{task} epoch={e:02} AUC={auc:.8f} recovery={rec:.1f}% delta_selected_U100={auc-baseline:+.8f}')
            if best is None or selection(row) > selection(best):
                best = row
        print(f'  BEST epoch={best["epoch"]} AUC={best["validation"]["macro_ovr_auc"]:.8f}')


def selection(row):
    from hlt_classification.jetclass2_delphes.runner import selection_key
    return selection_key(row['validation'], row['update'])
