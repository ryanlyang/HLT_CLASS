"""Exact Hydra allocations; allocation-local shared inputs; four actual GPU ranks."""
from datetime import timedelta
from decimal import Decimal
import faulthandler
from pathlib import Path
import os
import re
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context as BaseContext
from hlt_classification.operations.hydra_reconstruction_gpu_diagnostic import context
from . import contracts as c,campaign,population,inputs,artifacts,resources,training,kernel,banks,launch,startup,transport,normalization as norms,diagnostics as log


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
    gpu=task=='qualify' or task in c.GPU_TASKS;cpus=12 if gpu else 4 if task=='prepare' else 1
    memory=96000 if gpu or task=='prepare' else 4000
    expected=dict(JobId=job,JobState='RUNNING',Account='default',QOS='normal',Partition='gpus' if gpu else 'compute',
        JobName='lukadfull_'+task,Comment=spec['content_hash']+':'+task,WorkDir=spec['project'],NumNodes='1')
    if any(fields.get(k)!=v for k,v in expected.items()):raise PermissionError('Scheduler identity differs')
    if fields.get('NumCPUs') not in (str(cpus),str(2*cpus)) or fields.get('CPUs/Task')!=str(cpus) or fields.get('NumTasks')!='1':
        raise PermissionError('CPU allocation/request differs')
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    mem=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    if mem is None or Decimal(mem[1])*{'K':Decimal(1)/1024,'':1,'M':1,'G':1024,'T':1024**2}[mem[2]]!=memory:
        raise PermissionError('Allocation memory differs')
    minutes=c.QUALIFICATION_MINUTES if task=='qualify' else c.RESOURCES['minutes'] if gpu else 240 if task=='prepare' else 60
    if allocation_minutes(fields.get('TimeLimit',''))!=minutes:raise PermissionError('Allocation walltime differs')
    if gpu:
        transport.check()
        if tres.get('gres/gpu')!='4' or any(k.startswith('gres/gpu:') and (k!='gres/gpu:geforce_gtx_1080_ti' or v!='4') for k,v in tres.items()):
            raise PermissionError('Four registered GTX1080Ti devices required')
        log.mark('BEGIN CUDA availability/device-count query')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise PermissionError('Exactly four visible CUDA devices required')
        log.mark('END CUDA availability/device-count query')
        for i in range(4):
            log.mark(f'BEGIN CUDA properties device={i}')
            name=torch.cuda.get_device_name(i).removeprefix('NVIDIA ')
            if name!='GeForce GTX 1080 Ti' or torch.cuda.get_device_capability(i)!=(6,1):raise PermissionError('Wrong physical GPU family')
            log.mark(f'END CUDA properties device={i} name={name}')
    elif any(k.startswith('gres/gpu') for k in tres):raise PermissionError('CPU task unexpectedly allocated GPUs')
    if os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep)[0]!=str(Path(spec['environment_dir'])/'lib'):
        raise ValueError('Use isolated worker environment')
    return job


def task_receipt(spec,task):
    paths=[Path(spec['root'])/'tasks'/(task+'.json'),*sorted((Path(spec['root'])/'tasks/recovery').glob(task+'-*.json'))]
    candidates=[load_json(path) for path in paths if path.exists()]
    if not candidates:raise FileNotFoundError('No completed task receipt')
    accepted=campaign.accepted_jobs(spec,unambiguous=False)
    order={row['job']:i for i,row in enumerate(accepted)}
    r=max(candidates,key=lambda row:order.get(row['job'],-1));c.validate(r,'TASK',dict(spec=spec['content_hash']))
    accepted=[x for x in campaign.accepted_jobs(spec,unambiguous=False) if x['job']==r['job'] and x['task']==task]
    if len(accepted)!=1 or r['task']!=task:raise ValueError('Completed task binding differs')
    for ref in r['files'].values():validate_file_ref(ref)
    if task in c.GPU_TASKS:
        expected=segment_files(spec,task)
        if r['files']!=expected:raise ValueError('Segment completion binds different epochs or final result')
    return r


def segment_files(spec,task):
    fit,part,first,end=c.segment(task);root=artifacts.fit_root(spec,fit)
    last=load_json(root/'epochs'/f'{end:02d}.json')
    rows,_=artifacts.history(spec,fit,last['parents'])
    if len(rows)<end:raise ValueError('Segment did not commit all registered epochs')
    files=dict(epoch=file_ref(root/'epochs'/f'{end:02d}.json'))
    if part==4:
        artifacts.completed(spec,fit);files['result']=file_ref(root/'result.json')
    return files


def publish_task(spec,task,job,files):
    campaign.check(spec)
    receipt=c.artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job=job,files=files)
    destination=Path(spec['root'])/'reports'/(job+'.task.json') if task=='report' else Path(spec['root'])/'tasks'/(task+'.json')
    if destination.exists():
        old=load_json(destination);c.validate(old,'TASK',dict(spec=spec['content_hash']))
        destination=Path(spec['root'])/'tasks'/'recovery'/f'{task}-{job}.json'
    write_immutable_json(destination,receipt);return receipt


def scratch(job):
    root=Path('/dev/shm')/f'rlyang_d20full_{job}'
    if root.exists() or root.is_symlink() or not root.parent.is_dir():raise FileExistsError('Use a fresh allocation-local scratch root')
    root.mkdir(mode=0o700)
    return root


def cleanup(root,job):
    root=Path(root)
    if (root!=Path('/dev/shm')/f'rlyang_d20full_{job}' or root.is_symlink() or root.resolve()!=root
            or root.stat().st_uid!=os.getuid()):raise PermissionError('Unexpected allocation scratch cleanup target')
    shutil.rmtree(root)


def run(spec,task):
    with log.stage('worker configuration and source/runtime verification'):
        kernel.configure(1,12);campaign.check(spec)
    with log.stage('Slurm and device verification'):
        job=allocation(spec,task)
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
        if task in c.GPU_TASKS:
            fit,part,_,end=c.segment(task)
            if part>1:task_receipt(spec,f'{fit}_p{part-1:02d}')
        if task=='qualify' and (Path(spec['root'])/'qualification.json').exists():
            # A replacement allocation can finish the completion-last TASK after
            # a technical export/launcher interruption, without repeating probes.
            resources.qualification(spec)
            files=dict(qualification=file_ref(Path(spec['root'])/'qualification.json'))
            return publish_task(spec,task,job,files)
        if task!='qualify':task_receipt(spec,'qualify');resources.qualification(spec)
        if task in c.GPU_TASKS and (artifacts.fit_root(spec,c.segment(task)[0])/'result.json').exists():
            return publish_task(spec,task,job,segment_files(spec,task))
        if os.name=='posix':
            import resource
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        node=scratch(job)
        try:
            coordinates=('D000','OFFLINE','U100') if task=='qualify' else (c.cell(c.segment(task)[0])['coordinate'],)
            plan=[(coordinate,coordinate=='D000' and (task=='qualify' or c.cell(c.segment(task)[0])['reconstruction'])) for coordinate in coordinates]
            needed=inputs.planned_peak(spec['population'],plan)
            if shutil.disk_usage(node).free<needed:
                raise OSError(f'Need {needed/2**30:.1f} GiB free in /dev/shm before particle preparation')
            command=launch.command(spec['project'],spec['root'],task,node)
            with log.stage('installed torchrun argument parser before input build'):
                launch.validate(command)
                launch.validate(launch.command(spec['project'],spec['root'],task,node,startup_only=True))
            env=dict(os.environ,LUKABREC_INPUT_LAUNCHER_PID=str(os.getpid()),**transport.ENVIRONMENT)
            startup.folder(spec,job).mkdir(parents=True,exist_ok=False)
            with log.stage('early full-support GPU acceptance before population or particle reads'):
                subprocess.run(launch.startup_command(spec['project'],spec['root'],task,node),check=True,env=env)
                startup.check(spec,'qualify' if task=='qualify' else c.segment(task)[0],job)
            with log.stage('population replay'):
                loaded=population.load(spec['dataset'],Path(spec['root'])/'population')
            coords=('D000','OFFLINE','U100') if task=='qualify' else (c.cell(c.segment(task)[0])['coordinate'],)
            for coord in coords:
                needs=coord=='D000' and (task=='qualify' or c.cell(c.segment(task)[0])['reconstruction'])
                with log.stage(f'input build coordinate={coord} targets={needs}'):
                    options={}
                    if needs:
                        path=Path(spec['root'])/'normalization/qualified.json'
                        fixed=norms.qualified(spec) if task!='qualify' else norms.validate(load_json(path),spec) if path.exists() else None
                        if fixed is not None:options=dict(frozen_normalization=fixed,
                            normalization_evidence=Path(spec['root'])/'normalization'/f'{task}-{job}.json')
                    inputs.prepare_node(spec,node/coord,coord,targets=needs,loaded=loaded,**options)
            # Child ranks own startup/probe tracing; waiting here is expected.
            log.disable_stall_traces()
            with log.stage('four-rank worker execution'):
                campaign.check(spec)
                subprocess.run(command,check=True,env=env)
        finally:
            with log.stage('allocation scratch cleanup'):
                cleanup(node,job)
        files=dict(qualification=file_ref(Path(spec['root'])/'qualification.json')) if task=='qualify' else segment_files(spec,task)
    return publish_task(spec,task,job,files)


def rank(spec,task,node,*,startup_only=False):
    if task!='qualify' and task not in c.GPU_TASKS:raise PermissionError('Not a GPU task')
    rank=int(os.environ.get('RANK','-1'))
    if rank not in range(4) or os.environ.get('LOCAL_RANK')!=str(rank) or os.environ.get('WORLD_SIZE')!='4':
        raise PermissionError('Four local ranks on one node required')
    with log.stage('rank device/configuration and source/runtime verification'):
        transport.check()
        torch.cuda.set_device(rank);kernel.configure(4,12);campaign.check(spec,live=False);job=allocation(spec,task)
    if Path(node)!=Path('/dev/shm')/f'rlyang_d20full_{job}':raise PermissionError('Scratch not bound to this allocation')
    # Control objects use a separate host group. Native SyncBN and tensor/gradient
    # reductions retain the default NCCL group and their scientific math.
    trace=None
    if startup_only:
        log.disable_stall_traces()
        trace=(startup.folder(spec,job)/f'rank_{rank}.stack.log').open('x')
        faulthandler.cancel_dump_traceback_later();faulthandler.enable(file=trace)
        faulthandler.dump_traceback_later(40,repeat=True,file=trace)
    try:
        with log.stage('NCCL tensor group initialization'):
            dist.init_process_group('nccl',timeout=timedelta(seconds=startup.COLLECTIVE_SECONDS) if startup_only else timedelta(seconds=120))
        with log.stage('Gloo CPU control group initialization'):
            control=dist.new_group(backend='gloo',timeout=timedelta(seconds=startup.COLLECTIVE_SECONDS if startup_only else 1200))
        ctx=context(BaseContext,control,dist)(rank,4,torch.device('cuda',rank))
        if startup_only:return startup.run(spec,'qualify' if task=='qualify' else c.segment(task)[0],ctx,job)
        coords=('D000','OFFLINE','U100') if task=='qualify' else (c.cell(c.segment(task)[0])['coordinate'],)
        views={}
        for coord in coords:
            with log.stage(f'rank mmap validation coordinate={coord}'):
                views[coord]=inputs.load_node(spec,Path(node)/coord)
        arrays=load_npz_arrays(validate_file_ref(spec['population']['rows'],root=Path(spec['root'])/'population'))
        for caches,_,_ in views.values():
            for role,cache in caches.items():
                if not np.array_equal(cache.identities,arrays[role+'_identities']) or not np.array_equal(cache.labels,arrays[role+'_labels']):
                    raise ValueError('Node cache canonical join differs')
        folder=Path(spec['root'])/'runs'/f'{task}_{job}' if task=='qualify' else artifacts.fit_root(spec,c.segment(task)[0])/'runs'/job
        ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False))
        log.mark('GPU working directory ready: '+str(folder))
        if task=='qualify':resources.qualify(spec,views,arrays['train_pt'],ctx,folder)
        else:
            caches,targets,node_manifest=views[c.cell(c.segment(task)[0])['coordinate']]
            target=targets if c.cell(c.segment(task)[0])['reconstruction'] else None
            if target is not None and target.normalization['content_hash']!=resources.qualification(spec)['target_normalization_sha256']:
                raise ValueError('TRAIN reconstruction normalization changed after qualification')
            fit_task,part,first_epoch,last_epoch=c.segment(task)
            committed=len(list((artifacts.fit_root(spec,fit_task)/'epochs').glob('*.json')))
            if not first_epoch-1<=committed<=last_epoch:raise ValueError('Segment predecessor/committed cursor differs')
            bank=banks.Bank(spec,fit_task,caches['train']) if c.cell(fit_task)['teacher'] else None
            with log.stage('science integrated miniature'):
                mini=resources.miniature(spec,fit_task,caches,target,bank,arrays['train_pt'],ctx)
            receipt=c.artifact('STARTUP',parents=dict(spec=spec['content_hash']),task=task,job=job,
                miniature=mini,discarded=True,qualification=resources.qualification(spec)['content_hash'])
            ctx.main(lambda:write_immutable_json(folder/'startup.json',receipt))
            # fit owns its own fresh run directory so technical state cannot be resumed.
            log.mark('BEGIN scientific fit '+task)
            log.disable_stall_traces()
            training.fit(spec,fit_task,caches,target,bank,arrays['train_pt'],ctx,folder/'fit',job,
                         validation_pt=arrays['validation_pt'],end_epoch=last_epoch)
    except Exception:
        if startup_only:
            error=c.artifact('EARLY_GPU_ERROR',parents=dict(spec=spec['content_hash'],policy=startup.policy()['content_hash']),
                task=task,job=job,rank=rank,traceback=traceback.format_exc(),synthetic_only=True)
            write_immutable_json(startup.folder(spec,job)/f'rank_{rank}.error.json',error)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()
        if trace is not None:
            faulthandler.cancel_dump_traceback_later();faulthandler.disable();trace.close()
