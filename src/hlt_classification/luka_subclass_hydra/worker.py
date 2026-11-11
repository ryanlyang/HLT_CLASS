"""Exact accepted allocation workers; setup success releases the independent fits."""
from pathlib import Path
import time
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend, memory_snapshot
from . import execution as site
from .contracts import FITS, artifact, validate
from . import campaign, submission, inputs, training, resources, targets


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
    if task == 'setup':
        required = {'acceptance.json', 'resource_diagnostic.json', 'replay.json', 'replay_predictions.npz',
            'subclasses/manifest.json', 'subclasses/targets.npz', 'subclasses/prototypes.npz', 'baseline_parity.json'}
        required |= {'mini_'+t+'/receipt.json' for t in FITS}
    elif task in ('report',):
        required = {'results.json'}
    elif task in FITS:
        required = {'fit/'+n for n in ('selected.pt', 'selected_full.pt', 'training_report.json',
            'validation_predictions.npz', 'deployment.json', 'receipt.json')}
        required |= {'startup/receipt.json', 'startup/training_report.json', 'startup_memory.json'}
    else:
        raise ValueError('Task has no published receipt')
    if not required <= names:
        raise ValueError('Incomplete task receipt')
    return value


def completed_setup(spec):
    receipt(spec, 'setup'); resources.check_replay(spec); resources.check_baseline(spec)
    targets.check_manifest(spec)
    for task in FITS:
        r, _ = training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task), spec, task, technical=True)
        if r['parents'] != inputs.fit_parents(spec, task):
            raise ValueError('Setup target lineage differs')
    return resources.accepted(spec)


def run(spec, task):
    if task not in ('setup', 'release', 'report', *FITS):
        raise ValueError('Unregistered task')
    campaign.check(spec)
    cpu = task in ('release', 'report')
    if cpu:
        job = site.cpu_allocation(spec)
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
    if task == 'release':
        return submission.finish(spec, job)
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    if task == 'setup':
        inputs.Bundle(spec['bundle_root']).verify_all()
        resources.measure(spec)
    else:
        acceptance = completed_setup(spec)
        if task == 'report':
            from .reporting import results
            write_immutable_json(folder/'results.json', results(spec))
        else:
            if acceptance['gpu'] != gpu_identity():
                raise PermissionError('GPU differs from measured allocation')
            train, val = inputs.views(spec, task); banks = inputs.banks(spec, task, train)
            torch.cuda.reset_peak_memory_stats()
            startup = resources.miniature(folder/'startup', spec, task, train, val, banks, 'cuda')
            memory = memory_snapshot(torch)
            for key in ('peak_rss_bytes', 'gpu_peak_bytes'):
                if startup['fit_memory'].get(key) is not None:
                    memory[key] = max(memory[key], startup['fit_memory'][key])
            write_immutable_json(folder/'startup_memory.json', artifact('STARTUP', parents=inputs.fit_parents(spec, task),
                memory=memory, real_targets=True, technical_weights_discarded=True))
            if (memory['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
                    or memory['gpu_peak_bytes'] >= .88*acceptance['gpu']['total_memory_bytes']):
                raise MemoryError('Actual fit exceeds memory allowance')
            resources.free()
            _, report = training.fit(folder/'fit', spec, task, train, val, banks, device='cuda')
            if report['parents'] != inputs.fit_parents(spec, task):
                raise ValueError('Scientific target lineage differs')
    campaign.check(spec)
    value = artifact('TASK', parents=dict(spec=spec['content_hash']), task=task, job_id=job,
        environment=installed_environment(), gpu=None if cpu else gpu_identity(),
        outputs=[file_ref(p, root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json', value)
    return value
