"""Exact Oscar allocation with the explicitly registered replacement prefix.

Execution-only adaptation of jetclass2_delphes.execution.allocation and
luka_fullstat_attention.worker.execution at 420d22f8. No prefix spoofing and
no global change to the original execution-site profile.
"""
import os
import platform
import re
import subprocess
import sys
import torch
from hlt_classification.jetclass2_delphes.execution import validate_site, validate_resources, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from . import finalize_source as source

CONTRACTS = ('LUKA_FULLSTAT_SPLITHEAD_FINALIZE_SPEC/v1',
             'LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_SPEC/v1')


def execution(gate, measured=None):
    if gate.get('contract') not in CONTRACTS or gate.get('runtime_prefix') != source.RUNTIME:
        raise PermissionError('Replacement runtime is registered only for finalization specs')
    require_backend()
    site = gate['execution_site']
    validate_site(site)
    if site['name'] != 'oscar_l40s':
        raise PermissionError('Replacement runtime is Oscar L40S only')
    job = os.environ.get('SLURM_JOB_ID', '')
    if re.fullmatch(r'[1-9][0-9]*', job) is None:
        raise PermissionError('A real Slurm allocation is required')
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    memory = int(os.environ.get('SLURM_MEM_PER_NODE', '0'))
    validate_resources(site, cpus, memory, 1)
    details = subprocess.run(['scontrol', 'show', 'job', '-o', job],
        check=True, text=True, capture_output=True).stdout
    fields = dict(token.split('=', 1) for token in details.split() if '=' in token)
    if (fields.get('Account') != site['account'] or fields.get('Partition') != site['partition']
            or fields.get('QOS') != site['qos'] or os.environ.get('SLURM_CLUSTER_NAME') != site['cluster']
            or fields.get('NumNodes') != '1' or fields.get('NumTasks') != '1'
            or fields.get('NumCPUs') != str(cpus)):
        raise PermissionError('Not the registered single-task scheduler allocation')
    if (platform.machine() != site['architecture'] or sys.prefix != source.RUNTIME
            or os.environ.get('CONDA_PREFIX') != source.RUNTIME or os.environ.get('PYTHONNOUSERSITE') != '1'):
        raise PermissionError('Worker host/verified replacement environment differs')
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or site['gpu_family'] not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported()):
        raise PermissionError('Exactly one registered BF16-capable GPU is required')
    if (cpus, memory) != (gate['cpus'], gate['memory_mb']):
        raise PermissionError('Oscar allocation differs')
    evidence = gate['anchor']['evidence']
    source.runtime(evidence['environment'])
    if os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset') != evidence['cudnn_v8_disabled']:
        raise PermissionError('Backend setting differs from original FullStat')
    if measured and (measured['gpu'] != gpu_identity() or measured['environment'] != installed_environment()):
        raise PermissionError('Runtime differs from accepted new Oscar gate')
    return job
