"""Exact Hydra allocations; allocation-local shared inputs; four actual GPU ranks."""
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
from . import contracts as c,campaign,population,inputs,artifacts,resources,training,kernel,banks


def controller(args):
    for attempt in range(3):
        p=subprocess.run(['scontrol',*args],check=False,capture_output=True,text=True,timeout=30)
        if p.returncode==0:return p.stdout
        if attempt<2:time.sleep(5*(attempt+1))
    raise RuntimeError('Read-only Slurm query failed: '+p.stderr)


def allocation(spec,task):
    job=os.environ.get('SLURM_JOB_ID','')
    if task not in c.tasks() or not re.fullmatch('[1-9][0-9]*',job):raise PermissionError('Registered Slurm task required')
    for _ in range(12):
        matches=[r for r in campaign.accepted_jobs(spec,unambiguous=False) if r['job']==job and r['task']==task]
        if len(matches)==1:break
        time.sleep(1)
    else:raise PermissionError('No accepted task/job binding')
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',controller(['show','config'])):raise PermissionError('Hydra required')
    fields=dict(x.split('=',1) for x in controller(['show','job','-o',job]).split() if '=' in x)
    gpu=task=='qualify' or task in c.FITS;cpus=12 if gpu else 4 if task=='prepare' else 1
    memory=64000 if gpu or task=='prepare' else 4000
    expected=dict(JobId=job,JobState='RUNNING',Account='default',QOS='normal',Partition='gpus' if gpu else 'compute',
        JobName='lukabrec_'+task,Comment=spec['content_hash']+':'+task,WorkDir=spec['project'],NumNodes='1')
    if any(fields.get(k)!=v for k,v in expected.items()):raise PermissionError('Scheduler identity differs')
    if fields.get('NumCPUs') not in (str(cpus),str(2*cpus)) or fields.get('CPUs/Task')!=str(cpus) or fields.get('NumTasks')!='1':
        raise PermissionError('CPU allocation/request differs')
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    mem=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    if mem is None or Decimal(mem[1])*{'K':Decimal(1)/1024,'':1,'M':1,'G':1024,'T':1024**2}[mem[2]]!=memory:
        raise PermissionError('Allocation memory differs')
    minutes=360 if task=='qualify' else 2880 if gpu else 240 if task=='prepare' else 60
    if allocation_minutes(fields.get('TimeLimit',''))!=minutes:raise PermissionError('Allocation walltime differs')
    if gpu:
        if tres.get('gres/gpu')!='4' or any(k.startswith('gres/gpu:') and (k!='gres/gpu:geforce_gtx_1080_ti' or v!='4') for k,v in tres.items()):
            raise PermissionError('Four registered GTX1080Ti devices required')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise PermissionError('Exactly four visible CUDA devices required')
        for i in range(4):
            name=torch.cuda.get_device_name(i).removeprefix('NVIDIA ')
            if name!='GeForce GTX 1080 Ti' or torch.cuda.get_device_capability(i)!=(6,1):raise PermissionError('Wrong physical GPU family')
    elif any(k.startswith('gres/gpu') for k in tres):raise PermissionError('CPU task unexpectedly allocated GPUs')
    if os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep)[0]!=str(Path(spec['environment_dir'])/'lib'):
        raise ValueError('Use isolated worker environment')
    return job


def task_receipt(spec,task):
    r=load_json(Path(spec['root'])/'tasks'/(task+'.json'));c.validate(r,'TASK',dict(spec=spec['content_hash']))
    accepted=[x for x in campaign.accepted_jobs(spec) if x['job']==r['job'] and x['task']==task]
    if len(accepted)!=1 or r['task']!=task:raise ValueError('Completed task binding differs')
    for ref in r['files'].values():validate_file_ref(ref)
    return r


def scratch(job):
    root=Path('/dev/shm')/f'rlyang_lukabrec_{job}'
    if root.exists() or root.is_symlink() or not root.parent.is_dir():raise FileExistsError('Use a fresh allocation-local scratch root')
    root.mkdir(mode=0o700)
    return root


def cleanup(root,job):
    root=Path(root)
    if (root!=Path('/dev/shm')/f'rlyang_lukabrec_{job}' or root.is_symlink() or root.resolve()!=root
            or root.stat().st_uid!=os.getuid()):raise PermissionError('Unexpected allocation scratch cleanup target')
    shutil.rmtree(root)


def run(spec,task):
    kernel.configure(1,12);campaign.check(spec);job=allocation(spec,task)
    if task=='prepare':
        loaded=population.load(spec['dataset'],Path(spec['root'])/'population')
        if loaded[0]!=spec['population']:raise ValueError('Frozen subset differs')
        starts=artifacts.prepare_starts(spec)
        files=dict(starts=file_ref(Path(spec['root'])/'starts/manifest.json'))
    elif task=='release':
        task_receipt(spec,'qualify');return campaign.release(spec)
    elif task=='report':
        from .reporting import results
        value=results(spec);output=Path(spec['root'])/'reports'/(job+'.json')
        write_immutable_json(output,value)
        if not value['missing']:
            canonical=Path(spec['root'])/'results.json'
            if canonical.exists():
                if load_json(canonical)!=value:raise ValueError('Completed campaign results differ')
            else:write_immutable_json(canonical,value)
        files=dict(results=file_ref(output))
    else:
        task_receipt(spec,'prepare')
        if task=='qualify' and (Path(spec['root'])/'qualification.json').exists():
            # A replacement allocation can finish the completion-last TASK after
            # a technical export/launcher interruption, without repeating probes.
            resources.qualification(spec)
            files=dict(qualification=file_ref(Path(spec['root'])/'qualification.json'))
            receipt=c.artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job=job,files=files)
            write_immutable_json(Path(spec['root'])/'tasks/qualify.json',receipt);return receipt
        if task!='qualify':task_receipt(spec,'qualify');resources.qualification(spec)
        if task in c.FITS and (artifacts.fit_root(spec,task)/'result.json').exists():raise FileExistsError('Fit already published')
        node=scratch(job)
        try:
            loaded=population.load(spec['dataset'],Path(spec['root'])/'population')
            coords=('D000','OFFLINE','U100') if task=='qualify' else (c.cell(task)['coordinate'],)
            for coord in coords:
                needs=coord=='D000' and (task=='qualify' or c.cell(task)['reconstruction'])
                inputs.prepare_node(spec,node/coord,coord,targets=needs,loaded=loaded)
            command=[sys.executable,'-X','faulthandler','-u','-B','-s','-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc-per-node=4','--max-restarts=0',str(Path(spec['project'])/'scripts/luka_balanced_reconstruction_hydra.py'),
                'rank','--spec',str(Path(spec['root'])/'campaign_spec.json'),'--task',task,'--node',str(node)]
            env=dict(os.environ,LUKABREC_INPUT_LAUNCHER_PID=str(os.getpid()))
            subprocess.run(command,check=True,env=env)
        finally:cleanup(node,job)
        files=dict(qualification=file_ref(Path(spec['root'])/'qualification.json')) if task=='qualify' else dict(result=file_ref(artifacts.fit_root(spec,task)/'result.json'))
    campaign.check(spec)
    receipt=c.artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job=job,files=files)
    destination=Path(spec['root'])/'reports'/(job+'.task.json') if task=='report' else Path(spec['root'])/'tasks'/(task+'.json')
    write_immutable_json(destination,receipt);return receipt


def rank(spec,task,node):
    if task!='qualify' and task not in c.FITS:raise PermissionError('Not a GPU task')
    rank=int(os.environ.get('RANK','-1'))
    if rank not in range(4) or os.environ.get('LOCAL_RANK')!=str(rank) or os.environ.get('WORLD_SIZE')!='4':
        raise PermissionError('Four local ranks on one node required')
    torch.cuda.set_device(rank);kernel.configure(4,12);campaign.check(spec);job=allocation(spec,task)
    if Path(node)!=Path('/dev/shm')/f'rlyang_lukabrec_{job}':raise PermissionError('Scratch not bound to this allocation')
    # I/O was completed before NCCL initialization, avoiding a collective timeout
    # while the launcher reconstructs the large views.
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    ctx=Context(rank,4,torch.device('cuda',rank))
    try:
        coords=('D000','OFFLINE','U100') if task=='qualify' else (c.cell(task)['coordinate'],)
        views={coord:inputs.load_node(spec,Path(node)/coord) for coord in coords}
        arrays=load_npz_arrays(validate_file_ref(spec['population']['rows'],root=Path(spec['root'])/'population'))
        for caches,_,_ in views.values():
            for role,cache in caches.items():
                if not np.array_equal(cache.identities,arrays[role+'_identities']) or not np.array_equal(cache.labels,arrays[role+'_labels']):
                    raise ValueError('Node cache canonical join differs')
        folder=Path(spec['root'])/'runs'/f'{task}_{job}' if task=='qualify' else artifacts.fit_root(spec,task)/'runs'/job
        ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False))
        if task=='qualify':resources.qualify(spec,views,arrays['train_pt'],ctx,folder)
        else:
            caches,targets,node_manifest=views[c.cell(task)['coordinate']]
            target=targets if c.cell(task)['reconstruction'] else None
            if target is not None and target.normalization['content_hash']!=resources.qualification(spec)['target_normalization_sha256']:
                raise ValueError('TRAIN reconstruction normalization changed after qualification')
            bank=banks.Bank(spec,task,caches['train']) if c.cell(task)['teacher'] else None
            mini=resources.miniature(spec,task,caches,target,bank,arrays['train_pt'],ctx)
            startup=c.artifact('STARTUP',parents=dict(spec=spec['content_hash']),task=task,job=job,
                miniature=mini,discarded=True,qualification=resources.qualification(spec)['content_hash'])
            ctx.main(lambda:write_immutable_json(folder/'startup.json',startup))
            # fit owns its own fresh run directory so technical state cannot be resumed.
            training.fit(spec,task,caches,target,bank,arrays['train_pt'],ctx,folder/'fit',job,
                         validation_pt=arrays['validation_pt'])
    finally:dist.destroy_process_group()
