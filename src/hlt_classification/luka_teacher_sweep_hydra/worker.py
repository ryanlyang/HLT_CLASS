"""Real allocation workers and read-only completed/provisional recovery reports."""
from pathlib import Path
import time
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend, memory_snapshot
from hlt_classification.luka_embedding_aux import targets as offline_targets
from .contracts import TEACHERS, STUDENTS, FITS, artifact, validate
from . import campaign, submission, inputs, training, resources, site, targets


def receipt(spec, task):
    folder = Path(spec['root'])/task; value = load_json(folder/'receipt.json')
    validate(value, 'TASK', dict(spec=spec['content_hash']))
    stage = 'bootstrap' if task in ('setup', 'release') else 'science'
    if (value['task'] != task or value['job_id'] != submission.accepted_jobs(spec, stage).get(task)
            or len({r['path'] for r in value['outputs']}) != len(value['outputs'])
            or value['environment'] != spec['reference']['environment']):
        raise ValueError('Exact task/job/environment ownership differs')
    for ref in value['outputs']:
        validate_file_ref(ref, root=folder)
    names = {r['path'] for r in value['outputs']}
    required = {'acceptance.json', 'resource_diagnostic.json', 'statistics.json'} if task == 'setup' else {'results.json'} if task == 'report' else {
        'fit/selected.pt', 'fit/selected_full.pt', 'fit/training_report.json', 'fit/validation_predictions.npz', 'fit/receipt.json'}
    if task in TEACHERS:
        required |= {'bank/manifest.json', 'bank/targets.npz'}
    if task in STUDENTS:
        required |= {'startup/receipt.json', 'startup/training_report.json', 'startup_memory.json'}
    if task == 'setup':
        required |= {'mini_'+t+'/receipt.json' for t in resources.PROFILE_TASKS}
    if not required <= names:
        raise ValueError('Incomplete task receipt')
    return value


def completed_setup(spec):
    value = receipt(spec, 'setup'); resources.accepted(spec)
    for task in resources.PROFILE_TASKS:
        training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task), spec, task, technical=True)
    targets.load(spec)
    return resources.accepted(spec)


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
        parents['u100'] = inputs.Bundle(spec['bundle_root']).json('targets/u100/manifest.json')['content_hash']
        parents['offline'] = load_json(Path(spec['root'])/STUDENTS[task]/'bank/manifest.json')['content_hash']
    if task in ('T08', 'T09'): parents['statistics'] = targets.load(spec)['content_hash']
    return parents


def run(spec, task):
    if task not in ('setup', 'release', 'report', *FITS):
        raise ValueError('Unregistered task')
    campaign.check(spec)
    if task in ('release', 'report'): job = site.cpu_allocation(spec)
    else:
        require_backend(); job = site.allocation(spec)
    stage = 'bootstrap' if task in ('setup', 'release') else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec, stage).get(task) == job:
            break
        time.sleep(5)
    else:
        raise PermissionError('Worker is not exact accepted job')
    if installed_environment() != spec['reference']['environment']:
        raise ValueError('Installed environment changed')
    if task == 'release': return submission.finish(spec, job)
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    if task == 'setup':
        inputs.Bundle(spec['bundle_root']).verify_all(); inputs.delta(spec).verify_all()
        resources.measure(spec)
    else:
        completed_setup(spec); accepted = resources.accepted(spec)
        if task != 'report' and accepted['gpu'] != gpu_identity():
            raise PermissionError('GPU differs from measured allocation')
        if task == 'report':
            from .reporting import results
            value = results(spec); write_immutable_json(folder/'results.json', value)
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
        environment=installed_environment(), gpu=None if task == 'report' else gpu_identity(),
        outputs=[file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json', value)
    return value
