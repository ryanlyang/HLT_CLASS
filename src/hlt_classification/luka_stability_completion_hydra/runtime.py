"""Unchanged pinned SOFT-v2 workers with measured longer execution allocations."""
from datetime import timedelta
from pathlib import Path
import os
import subprocess
import sys
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_next_supervision_hydra import inputs, reconstruction_targets
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from . import core, submission


def allocation(spec, p, runtime, task):
    job = submission.require_job(spec, task)
    worker = runtime('worker'); previous = worker.require_job
    # The original site/GRES/environment validator is preserved. Only the exact
    # accepted-job lookup belongs to the new execution ledger, not the old IDs.
    def accepted(old, requested):
        if old != p or requested != task: raise PermissionError('Wrong allocation task')
        return job
    worker.require_job = accepted
    try: return worker.allocation(p, task)
    finally: worker.require_job = previous


def receipt(spec, task, p, runtime):
    folder = Path(spec['root']) / task; value = load_json(folder / 'receipt.json')
    core.validate(value, 'TASK', dict(spec=spec['content_hash'], scientific=p['content_hash']))
    if value['task'] != task or value['job_id'] != submission.accepted_jobs(spec).get(task):
        raise ValueError('Replacement receipt ownership differs')
    required = {'results.json'} if task == 'report_pascal' else {'startup/receipt.json', 'startup_memory.json', 'fit/receipt.json', 'fit/initialization.json'}
    names = [r['path'] for r in value['outputs']]
    if len(names) != len(set(names)) or not required <= set(names): raise ValueError('Replacement receipt inventory differs')
    for ref in value['outputs']: validate_file_ref(ref, root=folder)
    if task != 'report_pascal':
        runtime('training').validate_fit(folder / 'startup', p, task, technical=True)
        runtime('training').validate_fit(folder / 'fit', p, task)
    return value


def run(spec, task):
    _, _, p, old, _, _, runtime = core.check(spec, full=task != 'report_pascal')
    job, devices = allocation(spec, p, runtime, task)
    root = Path(spec['root']); claim = root / 'claims' / task
    claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim / 'intent.json', core.artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    folder = root / task; folder.mkdir(exist_ok=False)
    import shutil
    print(f'STORAGE advisory: available={shutil.disk_usage(root).free / 2**30:.2f} GiB', flush=True)
    if task == 'report_pascal':
        from .reporting import results
        write_immutable_json(folder / 'results.json', results(spec))
    else:
        require_backend(); reconstruction_targets.authenticate(old)
        subprocess.run([sys.executable, '-u', '-B', '-s', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                        '--max-restarts=0', '--nproc-per-node=4', spec['project_dir'] + '/scripts/luka_stability_completion_hydra.py',
                        'rank', '--spec', str(root / 'recovery_spec.json'), '--task', task], check=True)
        runtime('training').validate_fit(folder / 'fit', p, task)
    # Authenticate inputs/source again, but never check future output capacity.
    core.check(spec)
    value = core.artifact('TASK', parents=dict(spec=spec['content_hash'], scientific=p['content_hash']), task=task,
                          job_id=job, devices=devices, outputs=[file_ref(path, root=folder) for path in sorted(folder.rglob('*')) if path.is_file()])
    write_immutable_json(folder / 'receipt.json', value)
    return value


def rank(spec, task):
    if task not in core.TASKS: raise ValueError('Only the three unstarted SOFT replacements may train')
    _, _, p, old, _, _, runtime = core.check(spec); rank = int(os.environ.get('RANK', '-1'))
    if (rank not in range(4) or os.environ.get('LOCAL_RANK') != str(rank) or os.environ.get('WORLD_SIZE') != '4'
        or os.environ.get('LOCAL_WORLD_SIZE') != '4' or os.environ.get('GROUP_RANK') != '0'):
        raise PermissionError('Four local ranks on the registered node required')
    job, devices = allocation(spec, p, runtime, task)
    intent = core.artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job)
    if load_json(Path(spec['root']) / 'claims' / task / 'intent.json') != intent: raise PermissionError('Missing exact rank claim')
    torch.cuda.set_device(rank); runtime('kernel').configure(4, 12); require_backend()
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    ctx = d.Context(rank, 4, torch.device('cuda', rank)); targets = None
    try:
        train = inputs.hlt(old, 'train'); val = inputs.hlt(old, 'validation')
        bank = inputs.embedding_bank(old, train); targets = reconstruction_targets.Targets(old, train, 'OFFLINE')
        folder = Path(spec['root']) / task
        mini = runtime('resources').miniature(folder / 'startup', p, task, train, val, bank, targets, ctx)
        runtime('resources').check_memory(p, mini['fit_memory'], devices, 'pascal')
        ctx.main(lambda: write_immutable_json(folder / 'startup_memory.json', runtime('contracts').artifact('STARTUP',
                 parents=dict(spec=p['content_hash']), task=task, miniature=mini['content_hash'], memory=mini['fit_memory'], discarded=True)))
        runtime('kernel').configure(4, 12)
        runtime('training').fit(folder / 'fit', p, task, train, val, bank, targets, ctx)
    finally:
        if targets is not None: targets.close()
        dist.destroy_process_group()
