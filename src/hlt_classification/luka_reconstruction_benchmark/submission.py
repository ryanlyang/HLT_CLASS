"""One exact four-Pascal-GPU allocation; no scientific submission path."""
import os
from pathlib import Path
import re
import subprocess
import time
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_particle_teacher_hydra.submission import scheduler_review
from hlt_classification.luka_u100_pair_hydra.site import _scheduler_errors
from . import campaign as c

AUTHORIZATION='AUTHORIZE_R24_234_GPU_BENCHMARK_V1'


def plan(spec):
    p=spec['project_dir']; root=Path(spec['root'])
    command=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
        '--partition=gpus','--account=default','--qos=normal','--cpus-per-task=12','--mem=64000M',
        '--gres=gpu:geforce_gtx_1080_ti:4','--time=02:00:00','--job-name=lukarb_probe',
        '--comment=lukarb:'+spec['content_hash'],'--chdir='+p,'--output='+str(root/'probe-%j.out'),
        p+'/sbatch/run_luka_r24_gpu_benchmark.sh',p,str(root/'benchmark_spec.json'),'probe']
    return c.artifact('PLAN',parents=dict(spec=spec['content_hash']),authorization=AUTHORIZATION,
        commands=[dict(task_id='probe',dependencies=[],command=command)],science_release=False)


def jobs(spec):
    folder=Path(spec['root'])/'submission'; p=plan(spec)
    if not (folder/'command_plan.json').exists(): return {}
    if load_json(folder/'command_plan.json')!=p: raise ValueError('Submission plan differs')
    events,ids=load_exact_dag_journal(folder/'ledger_journal',identity=p['content_hash'],plan=p)
    if (folder/'ledger.json').exists() and load_json(folder/'ledger.json')!=assemble_submission_ledger(events,campaign_spec_sha256=p['content_hash']):
        raise ValueError('Benchmark ledger differs')
    return ids


def require_job(spec):
    job=os.environ.get('SLURM_JOB_ID','')
    for _ in range(12):
        if job and jobs(spec).get('probe')==job: return job
        time.sleep(5)
    raise PermissionError('Worker is not the exact accepted benchmark job')


def controller(args):
    for attempt in range(3):
        r=subprocess.run(['scontrol',*args],capture_output=True,text=True,timeout=30)
        if r.returncode==0: return r.stdout
        if attempt<2: time.sleep(5*(attempt+1))
    raise RuntimeError('scontrol failed after three read attempts: '+r.stderr.strip())


def allocation(spec):
    job=require_job(spec)
    config=controller(['show','config']); cluster=re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$',config)
    fields=dict(t.split('=',1) for t in controller(['show','job','-o',job]).split() if '=' in t)
    errors=_scheduler_errors(spec,job,fields,os.environ,cluster[1] if cluster else None)
    tres=dict(t.split('=',1) for t in fields.get('AllocTRES','').split(',') if '=' in t)
    if tres.get('gres/gpu')!='4': errors.append('Exactly four GPUs required')
    if any(k.startswith('gres/gpu:') and (k!='gres/gpu:geforce_gtx_1080_ti' or v!='4') for k,v in tres.items()):
        errors.append('Wrong typed GPU allocation')
    if errors: raise PermissionError('Benchmark allocation differs: '+'; '.join(errors))
    if os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise PermissionError('Use isolated benchmark wrapper')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4: raise PermissionError('Exactly four visible GPUs required')
    devices=[]
    for i in range(4):
        name=torch.cuda.get_device_name(i); cc=list(torch.cuda.get_device_capability(i))
        if name not in ('GeForce GTX 1080 Ti','NVIDIA GeForce GTX 1080 Ti') or cc!=[6,1]:
            raise PermissionError('Expected GTX1080Ti devices')
        devices.append(dict(name=name,compute_capability=cc,total_memory_bytes=torch.cuda.get_device_properties(i).total_memory))
    return job,devices


def submit(spec,*,execute=False,plan_hash=None,authorization=None):
    c.check(spec); p=plan(spec); folder=Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry.json'; live=folder/'ledger.json'
    if execute:
        if authorization!=AUTHORIZATION or plan_hash!=p['content_hash']: raise PermissionError('Exact benchmark authorization required')
        if not saved.exists() or load_json(saved)!=p or not dry.exists(): raise PermissionError('Full dry review required')
        if live.exists():
            if set(jobs(spec))!={'probe'}: raise ValueError('Incomplete benchmark ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'ledger_journal').exists(): raise RuntimeError('Ambiguous submission; preserve journals')
    write_immutable_json(saved,p)
    if not execute: scheduler_review(p)
    else: (folder/'live_claim').mkdir()
    return submit_exact_dag(identity=p['content_hash'],plan=p,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
