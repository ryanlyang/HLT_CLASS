"""Pinned Slurm launcher, disposable preparation, and two-rank production workers."""
from datetime import timedelta
from pathlib import Path
import os
import subprocess
import sys
import time
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from .contracts import FITS,artifact,validate
from . import campaign,submission,execution,inputs,targets,resources,training,distributed as d


def require_job(spec,task):
    job=os.environ.get('SLURM_JOB_ID','');stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if job and submission.accepted_jobs(spec,stage).get(task)==job:return job
        time.sleep(5)
    raise PermissionError('Worker is not the exact accepted task/job')


def receipt(spec,task):
    folder=Path(spec['root'])/task;r=load_json(folder/'receipt.json');validate(r,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task=='setup' else 'science'
    if (r['task']!=task or r['job_id']!=submission.accepted_jobs(spec,stage).get(task)
            or r['environment']!=spec['reference']['environment']):raise ValueError('Task ownership/environment differs')
    names=[v['path'] for v in r['outputs']]
    required={'results.json'} if task=='report' else ({'acceptance.json','resource_diagnostic.json','parity.json','replay.json',
        'replay_predictions.npz','preparation.json','subclasses/manifest.json','subclasses/targets.npz','subclasses/prototypes.npz'}|
        {'mini_'+t+'/receipt.json' for t in FITS} if task=='setup' else
        {'startup/receipt.json','startup/training_report.json','startup_memory.json'}|
        {'fit/'+n for n in ('receipt.json','training_report.json','selected.pt','selected_full.pt','validation_predictions.npz','deployment.json')})
    if len(names)!=len(set(names)) or not required<=set(names):raise ValueError('Task receipt inventory incomplete')
    for ref in r['outputs']:validate_file_ref(ref,root=folder)
    return r


def completed_setup(spec):
    receipt(spec,'setup');resources.check_replay(spec);targets.check_manifest(spec)
    for task in FITS:
        r,_=training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task),spec,task,technical=True)
        if r['parents']!=inputs.fit_parents(spec,task) or r['world_size']!=2:raise ValueError('Miniature lineage/execution differs')
    return resources.accepted(spec)


def prepare(spec):
    campaign.check(spec);require_job(spec,'setup');execution.allocation(spec)
    started=time.perf_counter();inputs.Bundle(spec['bundle_root']).verify_all()
    train,_=inputs.views(spec,'C0');offline=inputs.offline_bank(spec,train);targets.build(spec,train,offline)
    import resource
    value=artifact('PREPARATION',parents=dict(spec=spec['content_hash']),seconds=time.perf_counter()-started,
        host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        subclass=targets.check_manifest(spec)['content_hash'])
    write_immutable_json(Path(spec['root'])/'setup/preparation.json',value)
    return value


def run(spec,task):
    if task not in ('setup','release','report',*FITS):raise ValueError('Unregistered task')
    campaign.check(spec);cpu=task in ('release','report')
    devices=None if cpu else execution.allocation(spec)
    if devices is not None:
        print(f'SUBCLASS_DDP allocation: task={task}, GPUs=2 GTX1080Ti, CPUs={spec["cpus"]}, RAM={spec["memory_mb"]}MiB',flush=True)
    if cpu:execution.cpu_allocation(spec)
    job=require_job(spec,task)
    if task=='release':return submission.finish(spec,job)
    claim=Path(spec['root'])/'claims'/task;claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    folder=Path(spec['root'])/task;folder.mkdir(exist_ok=False)
    cli=str(Path(spec['project_dir'])/'scripts/luka_subclass_ddp_hydra.py');path=str(Path(spec['root'])/'campaign_spec.json')
    if task=='setup':subprocess.run([sys.executable,'-u','-B','-s',cli,'prepare','--spec',path],check=True)
    else:completed_setup(spec)
    if task=='report':
        from .reporting import results
        write_immutable_json(folder/'results.json',results(spec))
    else:
        subprocess.run([sys.executable,'-u','-B','-s','-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc-per-node=2',cli,'rank','--spec',path,'--task',task],check=True)
    campaign.check(spec)
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        environment=installed_environment(),devices=devices,
        outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',value)
    return value


def rank(spec,task):
    if task not in ('setup',*FITS):raise ValueError('Unregistered GPU task')
    rank=int(os.environ.get('RANK','-1'))
    if (rank not in (0,1) or os.environ.get('LOCAL_RANK')!=str(rank) or os.environ.get('WORLD_SIZE')!='2'
            or os.environ.get('LOCAL_WORLD_SIZE')!='2' or os.environ.get('GROUP_RANK')!='0'):
        raise PermissionError('One node and exactly two local ranks required')
    require_job(spec,task);devices=execution.allocation(spec)
    if campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:raise ValueError('Source changed')
    claim=load_json(Path(spec['root'])/'claims'/task/'intent.json')
    expected=artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=os.environ['SLURM_JOB_ID'])
    if claim!=expected:raise PermissionError('Missing exact launcher claim')
    torch.cuda.set_device(rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    ctx=d.Context(rank,2,torch.device('cuda',rank))
    try:
        started=time.perf_counter();train,val=inputs.views(spec,'C0' if task=='setup' else task)
        banks=inputs.banks(spec,'C0' if task=='setup' else task,train);inputs.join(train,banks)
        load_seconds=max(ctx.all(time.perf_counter()-started))
        if task=='setup':resources.measure(spec,train,val,banks,ctx,devices,load_seconds)
        else:
            acceptance=resources.accepted(spec)
            if devices!=acceptance['devices']:raise PermissionError('Devices differ from setup')
            folder=Path(spec['root'])/task;torch.cuda.reset_peak_memory_stats(ctx.device)
            startup=resources.miniature(folder/'startup',spec,task,train,val,banks,ctx)
            memory=startup['fit_memory'];resources.check_memory(spec,memory,devices)
            ctx.main(lambda:write_immutable_json(folder/'startup_memory.json',artifact('STARTUP',parents=inputs.fit_parents(spec,task),
                memory=memory,real_targets=True,technical_weights_discarded=True)))
            d.release_memory(ctx)
            training.fit(folder/'fit',spec,task,train,val,banks,ctx)
    finally:dist.destroy_process_group()
