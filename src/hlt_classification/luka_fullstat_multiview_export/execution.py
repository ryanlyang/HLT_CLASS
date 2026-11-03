"""Exact Oscar allocation for export recovery, using the original runtime."""
import os
import platform
import re
import subprocess
import sys
import torch
from hlt_classification.jetclass2_delphes.execution import validate_site, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_fullstat_multiview import source
from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
from .contracts import TASKS


def execution(spec, task):
    if task not in TASKS or spec['contract'] != 'LUKA_FULLSTAT_MULTIVIEW_EXPORT_SPEC/v1':
        raise PermissionError('Unregistered export recovery task')
    require_backend(); site = spec['execution_site']; validate_site(site)
    job = os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch(r'[1-9][0-9]*',job):
        raise PermissionError('Real Slurm allocation required')
    fields = subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields = dict(t.split('=',1) for t in fields.split() if '=' in t)
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK','0'))
    memory = int(os.environ.get('SLURM_MEM_PER_NODE','0'))
    expected = dict(Account=site['account'],Partition=site['partition'],QOS=site['qos'],
        NumNodes='1',NumTasks='1',NumCPUs=str(spec['cpus']))
    if (site['name'] != 'oscar_l40s' or any(fields.get(k) != v for k,v in expected.items())
            or os.environ.get('SLURM_CLUSTER_NAME') != site['cluster']
            or (cpus,memory) != (spec['cpus'],spec['memory_mb'])
            or allocation_minutes(fields.get('TimeLimit','')) != spec['minutes'][task]):
        raise PermissionError('Wrong recovery allocation')
    if (platform.machine() != site['architecture'] or sys.prefix != source.RUNTIME
            or spec['runtime_prefix'] != source.RUNTIME or os.environ.get('CONDA_PREFIX') != source.RUNTIME
            or os.environ.get('PYTHONNOUSERSITE') != '1'):
        raise PermissionError('Wrong verified runtime')
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or site['gpu_family'] not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported()):
        raise PermissionError('One registered L40S GPU required')
    accepted = spec['anchor']['evidence']
    source.runtime(accepted['environment'])
    if os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset') != accepted['cudnn_v8_disabled']:
        raise PermissionError('Original backend changed')
    from .campaign import parent
    measured = parent(spec)['reuse']['u100']
    if measured['gpu'] != gpu_identity() or measured['environment'] != installed_environment():
        raise PermissionError('Original accepted GPU/runtime differs')
    return job
