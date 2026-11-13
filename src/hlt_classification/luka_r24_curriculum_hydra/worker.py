"""Exact accepted Slurm tasks and four local ranks; technical state is discarded."""
from datetime import timedelta
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_pair_hydra.site import _scheduler_errors
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_next_supervision_hydra import inputs,reconstruction_targets
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from .contracts import FITS,artifact,validate
from . import campaign,submission,resources,training,kernel


def require_job(spec,task):
    job=os.environ.get('SLURM_JOB_ID','');stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if re.fullmatch('[1-9][0-9]*',job) and submission.accepted_jobs(spec,stage).get(task)==job:return job
        time.sleep(5)
    raise PermissionError('Worker is not the exact accepted task/job')


def controller(args):
    for attempt in range(3):
        try:
            result=subprocess.run(['scontrol',*args],capture_output=True,text=True,timeout=30)
            if result.returncode==0:return result.stdout
            error=result.stderr.strip()
        except subprocess.TimeoutExpired:error='controller read timed out'
        if attempt<2:time.sleep(5*(attempt+1))
    raise RuntimeError('scontrol failed after three read attempts: '+error)


def allocation(spec,task):
    job=require_job(spec,task);config=controller(['show','config'])
    fields=dict(t.split('=',1) for t in controller(['show','job','-o',job]).split() if '=' in t)
    if task in ('release','report'):
        site=spec['cpu_site'];expected=dict(JobId=job,JobState='RUNNING',Partition=site['partition'],Account=site['account'],QOS=site['qos'],NumNodes='1')
        if any(fields.get(k)!=v for k,v in expected.items()):raise PermissionError('Wrong CPU task allocation')
        requested=dict(x.split('=',1) for x in fields.get('ReqTRES','').split(',') if '=' in x)
        allocated=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
        if requested.get('cpu')!='1' or allocated.get('cpu') not in ('1','2') or fields.get('NumCPUs')!=allocated.get('cpu'):
            raise PermissionError('CPU request/rounding differs')
        for tres in (requested,allocated):
            if tres.get('mem')!='4000M' or tres.get('node')!='1' or any(k.startswith('gres/gpu') for k in tres):raise PermissionError('CPU memory/node/GPU allocation differs')
        for key,env in (('NumTasks','SLURM_NTASKS'),('CPUs/Task','SLURM_CPUS_PER_TASK')):
            values=[v for v in (fields.get(key),os.environ.get(env)) if v not in (None,'','N/A','(null)')]
            if not values or any(v!='1' for v in values):raise PermissionError('CPU task/thread allocation differs')
        if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):raise PermissionError('Hydra CPU allocation required')
        return job,None
    cluster=re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$',config)
    errors=_scheduler_errors(spec,job,fields,os.environ,cluster[1] if cluster else None)
    tres=dict(t.split('=',1) for t in fields.get('AllocTRES','').split(',') if '=' in t)
    if tres.get('gres/gpu')!='4':errors.append('Exactly four GPUs required')
    if any(k.startswith('gres/gpu:') and (k!='gres/gpu:geforce_gtx_1080_ti' or v!='4') for k,v in tres.items()):errors.append('Wrong typed GPU allocation')
    if errors:raise PermissionError('Four-card allocation differs: '+'; '.join(errors))
    if os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise PermissionError('Use the isolated wrapper')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise PermissionError('Exactly four visible GPUs required')
    devices=[dict(name=torch.cuda.get_device_name(i),compute_capability=list(torch.cuda.get_device_capability(i)),
        total_memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(4)]
    resources.check_devices(devices);return job,devices


def receipt(spec,task):
    folder=Path(spec['root'])/task;r=load_json(folder/'receipt.json');validate(r,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task=='setup' else 'science'
    if r['task']!=task or r['job_id']!=submission.accepted_jobs(spec,stage).get(task) or r['environment']!=spec['environment']:
        raise ValueError('Task ownership/environment differs')
    names=[v['path'] for v in r['outputs']]
    required={'results.json'} if task=='report' else ({'acceptance.json','resource_diagnostic.json','parity.json','replay.json','replay_predictions.npz'}|
        {f'mini_{t}/receipt.json' for t in FITS} if task=='setup' else
        {'startup/receipt.json','startup_memory.json'}|{'fit/'+n for n in (*training.OUTPUTS,'receipt.json')})
    if len(names)!=len(set(names)) or not required<=set(names):raise ValueError('Task output inventory incomplete')
    for ref in r['outputs']:validate_file_ref(ref,root=folder)
    return r


def completed_setup(spec):
    receipt(spec,'setup');old=load_json(spec['original_spec']['path'])
    resources.check_parity(spec,load_json(Path(spec['root'])/'setup/parity.json'));resources.check_replay(spec,old)
    diagnostic=load_json(Path(spec['root'])/'setup/resource_diagnostic.json')
    for task in FITS:
        r,_=training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task),spec,task,technical=True)
        if r['world_size']!=4 or r['content_hash']!=diagnostic['profiles'][task]['miniature']:raise ValueError('Miniature lineage differs')
    return resources.accepted(spec)


def run(spec,task):
    if task not in ('setup','release','report',*FITS):raise ValueError('Unregistered task')
    old=campaign.check(spec,full=task not in ('release','report'));job,devices=allocation(spec,task)
    if task=='release':return submission.finish(spec,job)
    claim=Path(spec['root'])/'claims'/task;claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    folder=Path(spec['root'])/task;folder.mkdir(exist_ok=False)
    if task!='setup':completed_setup(spec)
    if task=='report':
        from .reporting import results
        write_immutable_json(folder/'results.json',results(spec))
    else:
        require_backend();reconstruction_targets.authenticate(old)
        print(f'R24_CURRICULUM allocation: task={task}, 4 GTX1080Ti, CPUs=12, RAM=64000MiB',flush=True)
        subprocess.run([sys.executable,'-u','-B','-s','-m','torch.distributed.run','--standalone','--nnodes=1',
            '--max-restarts=0','--nproc-per-node=4',str(Path(spec['project_dir'])/'scripts/luka_r24_curriculum_hydra.py'),
            'rank','--spec',str(Path(spec['root'])/'campaign_spec.json'),'--task',task],check=True)
    campaign.check(spec)
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,environment=spec['environment'],devices=devices,
        outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',value);return value


def rank(spec,task):
    if task not in ('setup',*FITS):raise ValueError('Unregistered GPU task')
    rank=int(os.environ.get('RANK','-1'))
    if (rank not in range(4) or os.environ.get('LOCAL_RANK')!=str(rank) or os.environ.get('WORLD_SIZE')!='4'
            or os.environ.get('LOCAL_WORLD_SIZE')!='4' or os.environ.get('GROUP_RANK')!='0'):
        raise PermissionError('Exactly four local ranks on one node required')
    old=campaign.check(spec);job,devices=allocation(spec,task)
    expected=artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job)
    if load_json(Path(spec['root'])/'claims'/task/'intent.json')!=expected:raise PermissionError('Missing exact launcher claim')
    torch.cuda.set_device(rank);kernel.configure(4,spec['cpus']);require_backend()
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1';dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    ctx=d.Context(rank,4,torch.device('cuda',rank));targets=None
    try:
        began=time.perf_counter();train=inputs.hlt(old,'train');val=inputs.hlt(old,'validation')
        bank=inputs.embedding_bank(old,train);targets=reconstruction_targets.Targets(old,train,'OFFLINE')
        load_seconds=max(ctx.all(time.perf_counter()-began))
        if task=='setup':resources.measure(spec,old,train,val,bank,targets,ctx,devices,load_seconds)
        else:
            # Same production path on the allocated hardware; reset all weights,
            # optimizer and RNG before the full scientific fit.
            folder=Path(spec['root'])/task
            mini=resources.miniature(folder/'startup',spec,task,train,val,bank,targets,ctx)
            resources.check_memory(spec,mini['fit_memory'],devices)
            ctx.main(lambda:write_immutable_json(folder/'startup_memory.json',artifact('STARTUP',parents=dict(spec=spec['content_hash']),
                task=task,miniature=mini['content_hash'],memory=mini['fit_memory'],discarded=True)))
            training.fit(folder/'fit',spec,task,train,val,bank,targets,ctx)
    finally:
        if targets is not None:targets.close()
        dist.destroy_process_group()
