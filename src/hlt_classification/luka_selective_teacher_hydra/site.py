"""Hydra A6000 allocation; schema-independent scheduler verification."""
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
import torch
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_pair_hydra import site as donor
from hlt_classification.luka_next_supervision_hydra.reconstruction_release import cpu_allocation


def allocation(spec):
    if spec['execution_site']!=A6000_SITE or spec['gpu_profile']!='a6000': raise ValueError('A6000 profile required')
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job): raise PermissionError('Real Slurm allocation required')
    controller=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(t.split('=',1) for t in controller.split() if '=' in t)
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    cluster=re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$',config)
    errors=donor._scheduler_errors(spec,job,fields,os.environ,cluster[1] if cluster else None)
    if errors: raise PermissionError('Teacher-sweep allocation differs:\n  '+'\n  '.join(errors))
    if (platform.machine()!=A6000_SITE['architecture'] or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0)!=A6000_SITE['gpu_name'] or torch.cuda.get_device_capability(0)!=(8,6)
            or not torch.cuda.is_bf16_supported()): raise PermissionError('Use pinned environment and one RTX A6000')
    donor.environment(spec['reference']['environment'])
    print(f'SELECTIVE_TEACHER A6000 allocation job={job}, CPUs=6, RAM=32000MiB',flush=True)
    return job
