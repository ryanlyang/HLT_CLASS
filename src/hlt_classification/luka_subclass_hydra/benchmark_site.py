"""Exact two-Pascal-GPU allocation for a benchmark, never a science fallback."""
import os
import re
import subprocess
import sys
import platform
from pathlib import Path
import torch
from hlt_classification.luka_u100_pair_hydra.site import _scheduler_errors, environment


def allocation(spec):
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job): raise PermissionError('Real benchmark Slurm allocation required')
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    cluster=re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$',config)
    out=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(t.split('=',1) for t in out.split() if '=' in t)
    errors=_scheduler_errors(spec,job,fields,os.environ,cluster[1] if cluster else None)
    tres=dict(t.split('=',1) for t in fields.get('AllocTRES','').split(',') if '=' in t)
    if tres.get('gres/gpu')!='2': errors.append('Exactly two allocated GPUs required')
    if any(k.startswith('gres/gpu:') and (k!='gres/gpu:geforce_gtx_1080_ti' or v!='2') for k,v in tres.items()):
        errors.append('Wrong typed GPU allocation')
    if errors: raise PermissionError('Benchmark allocation differs: '+'; '.join(errors))
    if (platform.machine()!='x86_64' or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise PermissionError('Use the isolated pinned benchmark environment')
    environment(spec['reference']['environment'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise PermissionError('Exactly two visible CUDA GPUs required')
    result=[]
    for i in range(2):
        name=torch.cuda.get_device_name(i); cc=list(torch.cuda.get_device_capability(i))
        if name not in ('GeForce GTX 1080 Ti','NVIDIA GeForce GTX 1080 Ti') or cc!=[6,1]:
            raise PermissionError('Benchmark requires two GTX 1080 Ti cards on this node')
        result.append(dict(name=name,compute_capability=cc,total_memory_bytes=torch.cuda.get_device_properties(i).total_memory))
    return result
