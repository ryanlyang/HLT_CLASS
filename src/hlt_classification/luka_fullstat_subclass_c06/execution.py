"""Exact registered Oscar allocation and repaired environment fingerprint."""
import os
import platform
import re
import subprocess
import sys
import torch
from hlt_classification.jetclass2_delphes.execution import validate_site,gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_fullstat_readout.source import runtime,RUNTIME
from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
from .contracts import TASK


def execution(spec):
    require_backend();site=spec['execution_site'];validate_site(site)
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch(r'[1-9][0-9]*',job):raise PermissionError('Real Slurm allocation required')
    out=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(x.split('=',1) for x in out.split() if '=' in x)
    expected=dict(Account=site['account'],Partition=site['partition'],QOS=site['qos'],
                  NumNodes='1',NumTasks='1',NumCPUs=str(spec['cpus']))
    if (site['name']!='oscar_l40s' or any(fields.get(k)!=v for k,v in expected.items())
            or os.environ.get('SLURM_CLUSTER_NAME')!=site['cluster']
            or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=spec['cpus']
            or int(os.environ.get('SLURM_MEM_PER_NODE','0'))!=spec['memory_mb']
            or allocation_minutes(fields.get('TimeLimit',''))!=spec['minutes'][TASK]):
        raise PermissionError('Wrong registered allocation')
    if (platform.machine()!=site['architecture'] or sys.prefix!=RUNTIME or spec['runtime_prefix']!=RUNTIME
            or os.environ.get('PYTHONNOUSERSITE')!='1' or not torch.cuda.is_available()
            or torch.cuda.device_count()!=1 or site['gpu_family'] not in torch.cuda.get_device_name(0)
            or not torch.cuda.is_bf16_supported()):raise PermissionError('Wrong runtime/GPU')
    evidence=spec['reuse']['u100'];runtime(evidence['environment'])
    if (evidence['gpu']!=gpu_identity() or evidence['environment']!=installed_environment()
            or os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset')!=spec['anchor']['evidence']['cudnn_v8_disabled']):
        raise PermissionError('Original measured backend changed')
    return job
