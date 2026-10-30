"""Production workers, exact-job ownership and compact authenticated reporting."""
from pathlib import Path
import gc
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from .contracts import artifact, validate, ARMS, TASKS, node as registered_node
from . import campaign, submission, acceptance, views, teachers, targets, training
from .model import new_student


def teacher_ref(spec, name):
    if name == 'U100':
        return spec['imported']['teacher']['files']['checkpoint']
    folder = Path(spec['root'])/('teacher_'+name)
    completed(spec, 'teacher_'+name)
    return file_ref(folder/'fit/selected_full.pt')


def bank_folder(spec, name):
    if name == 'U100':
        return Path(spec['root'])/'accept/bank_U100'
    return Path(spec['root'])/('teacher_'+name)/'bank'


def completed(spec, task):
    if task == 'accept':
        return acceptance.accepted(spec)
    folder = Path(spec['root'])/task
    value = load_json(folder/'receipt.json'); validate(value, 'TASK', dict(spec=spec['content_hash']))
    if value['task'] != task or value['job_id'] != submission.accepted_jobs(spec, 'science').get(task):
        raise ValueError('Task receipt differs from exact submitted job')
    fit_files = {'fit/'+f for f in ('selected_full.pt', 'selected.pt', 'validation_predictions.npz',
                                   'training_report.json', 'fit_receipt.json')}
    expected = ({'results.json'} if task == 'report' else
                {'target_reuse.json'} if task == 'export_U100'
                else fit_files | ({'bank/targets.npz', 'bank/manifest.json'} if task.startswith('teacher_') else set()))
    if {r['path'] for r in value['outputs']} != expected or len(value['outputs']) != len(expected):
        raise ValueError('Task output inventory differs')
    from .retention import validate_output
    for ref in value['outputs']: validate_output(spec, ref, root=folder)
    if task == 'export_U100':
        reused = load_json(folder/'target_reuse.json')
        validate(reused, 'U100_REUSE', dict(spec=spec['content_hash'], acceptance=acceptance.accepted(spec)['content_hash']))
        if reused['manifest'] != file_ref(bank_folder(spec, 'U100')/'manifest.json'):
            raise ValueError('Original U100 target reuse binding differs')
    if task in ARMS or task.startswith('teacher_'):
        report, _ = training.validate_fit(folder/'fit', spec, task)
        expected_parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
                                prepared=spec['anchor']['evidence']['prepared'])
        if task in ARMS:
            for name in ARMS[task]['targets']:
                manifest = load_json(bank_folder(spec, name)/'manifest.json')
                validate(manifest, 'BANK')
                expected_parents[name] = manifest['content_hash']
        if report['parents'] != expected_parents:
            raise ValueError('Training report parent/bank binding differs')
        expected_node = registered_node(spec['imported'], task)
        if report['node'] != expected_node:
            raise ValueError('Fit initialization/sampler node differs')
    return value


def run(spec, task):
    if task not in TASKS:
        raise ValueError('Unregistered task')
    campaign.check(spec); require_backend()
    job, cpus, memory = allocation(spec['execution_site'])
    if (cpus, memory) != (spec['cpus'], spec['memory_mb']):
        raise PermissionError('Allocation differs from reviewed plan')
    stage = 'accept' if task == 'accept' else 'science'
    if submission.accepted_jobs(spec, stage).get(task) != job:
        raise PermissionError('Allocation is not the accepted exact task ID')
    claims = Path(spec['root'])/'claims'; claims.mkdir(exist_ok=True)
    (claims/task).mkdir(exist_ok=False)
    write_immutable_json(claims/task/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    if task == 'accept':
        return acceptance.run(spec, job)
    measured = acceptance.accepted(spec)
    if measured['environment'] != installed_environment() or measured['gpu'] != gpu_identity():
        raise PermissionError('Measured GPU/environment differs; new resource review required')
    folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    if task == 'report':
        value = results(spec)
        write_immutable_json(folder/'results.json', value)
    else:
        for dependency in TASKS[task]: completed(spec, dependency)
        if task == 'export_U100':
            # Acceptance exported the actual selected original U100, so reuse it
            # without a second forward pass or persistent copy. C/F technical
            # targets are never reused as scientific teacher supervision.
            tr = views.load(spec, 'train', 'D000')
            targets.Bank(bank_folder(spec, 'U100'), tr, 'U100', teacher_ref(spec, 'U100'),
                         source_hash=spec['source_snapshot']['content_hash'])
            reused = artifact('U100_REUSE', parents=dict(spec=spec['content_hash'], acceptance=measured['content_hash']),
                              manifest=file_ref(bank_folder(spec, 'U100')/'manifest.json'))
            write_immutable_json(folder/'target_reuse.json', reused)
        elif task.startswith('teacher_'):
            name = task.removeprefix('teacher_'); node = registered_node(spec['imported'], task)
            tr, val = (views.load(spec, role, name) for role in ('train', 'validation'))
            model = teachers.new_teacher(name, node)
            model, _ = training.fit(folder/'fit', spec, task, model, tr, val, node, {}, device='cuda')
            training.validate_fit(folder/'fit', spec, task)
            targets.export(folder/'bank', model, tr, name, file_ref(folder/'fit/selected_full.pt'), device='cuda',
                           source_hash=spec['source_snapshot']['content_hash'])
        else:
            tr, val = (views.load(spec, role, 'D000') for role in ('train', 'validation'))
            # Control reads no teacher banks/checkpoints; imported hashes are bookkeeping only.
            banks = {n: targets.Bank(bank_folder(spec, n), tr, n, teacher_ref(spec, n),
                                    source_hash=spec['source_snapshot']['content_hash']) for n in ARMS[task]['targets']}
            node = registered_node(spec['imported'], task)
            model = new_student(task, node)
            training.fit(folder/'fit', spec, task, model, tr, val, node, banks, device='cuda')
            training.validate_fit(folder/'fit', spec, task)
    campaign.check(spec)
    output = [file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()]
    receipt = artifact('TASK', parents=dict(spec=spec['content_hash']), task=task, job_id=job, outputs=output)
    write_immutable_json(folder/'receipt.json', receipt)
    return receipt


def results(spec):
    # No model or particle reads; complete rows are authenticated from retained evidence.
    campaign.check(spec)
    from hlt_classification.luka_embedding_weight.worker import validate_predictions
    imported = spec['imported']; references = {
        n: spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT', 'OFFLINE')}
    references.update(U100_EMBED_W5=imported['files']['predictions'],
                      COARSE_U100=imported['teacher']['files']['predictions'])
    arrays = {}
    for name, ref in references.items():
        validate_file_ref(ref); arrays[name] = load_npz_arrays(ref['path'])
    validate_file_ref(imported['files']['subset'])
    ix = load_npz_arrays(imported['files']['subset']['path'])['indices']
    if (ix.dtype != np.int64 or ix.shape != (spec['original50k_rows'],)
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= spec['counts']['validation'])):
        raise ValueError('Report-only subset differs')
    pending, selected = [], {}
    for task in ('teacher_CONCAT', 'teacher_FUSION', *ARMS):
        folder = Path(spec['root'])/task
        if not (folder/'receipt.json').exists():
            pending.append(task); continue
        completed(spec, task)
        report, a = training.validate_fit(folder/'fit', spec, task)
        arrays[task] = a; selected[task] = report['selected_epoch']
    rows = []
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        base = arrays['M0HLT']
        if not np.array_equal(a['identities'], base['identities']) or not np.array_equal(a['labels'], base['labels']):
            raise ValueError('Result population join differs')
        row = dict(model=name, selected_epoch=selected.get(name),
                   oracle=name in ('OFFLINE', 'COARSE_U100', 'teacher_CONCAT', 'teacher_FUSION'))
        for role, index in (('validation', slice(None)), ('original50k', ix)):
            row[role] = evaluate_probabilities(a['labels'][index], a['probabilities'][index])
            anchors = [evaluate_probabilities(arrays[n]['labels'][index], arrays[n]['probabilities'][index])
                       for n in ('M0HLT', 'OFFLINE')]
            row[role+'_recovery'] = recovery(row[role], *anchors)
        rows.append(row)
    by_name = {r['model']: r for r in rows}
    for row in rows:
        for role in ('validation', 'original50k'):
            for reference in ('u100_w5', 'hlt_capacity_control'):
                row[role+'_delta_'+reference] = (row[role]['macro_ovr_auc']-by_name[reference][role]['macro_ovr_auc']
                                                 if reference in by_name else None)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
                    pending_meaning='missing completed receipt; consult exact scheduler IDs for failure/cancellation')


def print_results(value):
    print('U100 multiview/readout screen. Historical recovery anchors; no independent test.')
    print('Original50k overlaps validation. New teachers are privileged ORACLE models.')
    for role in ('validation', 'original50k'):
        print('\n'+role)
        print(f'{"model":26} {"pick":>4} {"AUC":>12} {"recovery%":>10} {"input":>8}')
        for row in value['rows']:
            rec = row[role+'_recovery']['macro_ovr_auc']
            print(f'{row["model"]:26} {str(row["selected_epoch"] or "-"):>4} '
                  f'{row[role]["macro_ovr_auc"]:12.8f} {str(round(rec, 1)) if rec is not None else "n/a":>10} '
                  f'{"ORACLE" if row["oracle"] else "HLT":>8}')
    print('Incomplete: '+(', '.join(value['pending']) or 'none'))
