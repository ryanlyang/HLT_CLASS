"""Verify the real Oscar allocation before particle preparation or training."""
import os
import platform
import re
import subprocess
import torch
from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from . import source,submission


def validate(spec,task):
    require_backend();site=spec['execution_site'];job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job) or task not in spec['tasks']:raise PermissionError('Registered Slurm worker required')
    if submission.accepted_jobs(spec,partial=True).get(task)!=job:raise ValueError('Worker job differs from submission journal')
    out=subprocess.run(['scontrol','show','job','-o',job],check=True,text=True,capture_output=True).stdout
    f=dict(x.split('=',1) for x in out.split() if '=' in x)
    expected=dict(Account=site['account'],Partition=site['partition'],QOS=site['qos'],NumNodes='1',NumTasks='1',
        NumCPUs=str(spec['cpus']),JobName='lukafsbal_'+task,Comment='lukafsbal:'+spec['content_hash']+':'+task)
    if (any(f.get(k)!=v for k,v in expected.items()) or os.environ.get('SLURM_CLUSTER_NAME')!=site['cluster']
            or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=spec['cpus']
            or int(os.environ.get('SLURM_MEM_PER_NODE','0'))!=spec['memory_mb']
            or allocation_minutes(f.get('TimeLimit',''))!=spec['minutes'][task]
            or platform.machine()!=site['architecture'] or os.environ.get('PYTHONNOUSERSITE')!='1'):
        raise PermissionError('Allocation differs from registered resources')
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()
            or site['gpu_family'] not in torch.cuda.get_device_name(0)):
        raise PermissionError('One registered BF16-capable L40S required')
    e=spec['inputs']['evidence'];source.runtime(e['environment'])
    if os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset')!=e['cudnn_v8_disabled']:
        raise PermissionError('Original backend setting differs')
    return job
