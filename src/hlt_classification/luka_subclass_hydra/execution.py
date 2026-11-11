"""One explicitly selected, measured BF16 GPU type per Hydra execution."""
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
import torch
from hlt_classification.luka_u100_pair_hydra.contracts import SITE
from hlt_classification.luka_u100_pair_hydra import site as donor
from hlt_classification.luka_next_supervision_hydra.reconstruction_release import cpu_allocation


PROFILES = ('a6000', 'l40', 'l40s')


def configuration(name):
    hardware = {
        'a6000': ('nvidia_rtx_a6000', 'NVIDIA RTX A6000', [8, 6]),
        'l40': ('nvidia_l40', 'NVIDIA L40', [8, 9]),
        'l40s': ('nvidia_l40s', 'NVIDIA L40S', [8, 9]),
    }
    if name not in hardware:
        raise ValueError('Choose an explicit a6000, l40 or l40s GPU profile')
    gres, gpu_name, capability = hardware[name]
    return dict(execution_version=2, gpu_profile=name, gpu_compute_capability=capability,
        execution_site=dict(SITE, gres='gpu:'+gres+':1', gpu_name=gpu_name))


def check_profile(spec):
    expected = configuration(spec['gpu_profile'])
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered GPU profile differs; use a fresh execution')
    return expected


def check_gpu(spec, gpu):
    check_profile(spec)
    if (gpu.get('name') != spec['execution_site']['gpu_name']
            or gpu.get('compute_capability') != spec['gpu_compute_capability']
            or type(gpu.get('total_memory_bytes')) is not int or gpu['total_memory_bytes'] <= 0):
        raise ValueError('GPU differs from the selected profile')


def allocation(spec):
    check_profile(spec)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('Real Slurm allocation required')
    controller = subprocess.run(['scontrol', 'show', 'job', '-o', job],
        check=True, capture_output=True, text=True).stdout
    fields = dict(t.split('=', 1) for t in controller.split() if '=' in t)
    config = subprocess.run(['scontrol', 'show', 'config'], check=True, capture_output=True, text=True).stdout
    cluster = re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$', config)
    errors = donor._scheduler_errors(spec, job, fields, os.environ, cluster[1] if cluster else None)
    tres = dict(t.split('=', 1) for t in fields.get('AllocTRES', '').split(',') if '=' in t)
    if tres.get('gres/gpu') != '1':
        errors.append('AllocTRES must allocate exactly one GPU')
    expected_type = spec['execution_site']['gres'].split(':')[1]
    if any(k.startswith('gres/gpu:') and (k != 'gres/gpu:'+expected_type or v != '1')
            for k, v in tres.items()):
        errors.append('Typed GPU allocation differs from the selected profile')
    if errors:
        raise PermissionError('Subclass allocation differs:\n  '+'\n  '.join(errors))
    if (platform.machine() != spec['execution_site']['architecture']
            or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE') != '1'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != spec['execution_site']['gpu_name']
            or list(torch.cuda.get_device_capability(0)) != spec['gpu_compute_capability']
            or not torch.cuda.is_bf16_supported()):
        raise PermissionError('Use the pinned environment and one registered BF16 GPU')
    donor.environment(spec['reference']['environment'])
    print(f'SUBCLASS {spec["gpu_profile"]} allocation job={job}, '
          f'CPUs={spec["cpus"]}, RAM={spec["memory_mb"]}MiB', flush=True)
    return job
