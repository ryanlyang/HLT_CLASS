"""Hydra-specific scheduler and isolated-environment checks."""
import os
import platform
from pathlib import Path
import re
import subprocess
import sys

from hlt_classification.luka_fullsim.regular_submission import clean_environment
from .contracts import SITE


def check_site(plan):
    env = clean_environment()
    config = subprocess.run(['scontrol', 'show', 'config'], check=True, capture_output=True, text=True, env=env).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$', config):
        raise PermissionError('Submit this adapter on Brown CS Hydra')
    shapes = set()
    for row in plan['commands']:
        command = row['command']
        for key in ('partition', 'account', 'qos', 'gres'):
            if '--'+key+'='+SITE[key] not in command:
                raise ValueError('Plan does not request the registered Hydra site')
        shape = tuple(x for x in command if x.startswith(('--mem=', '--time=', '--cpus-per-task=', '--gres=')))
        if shape in shapes:
            continue
        shapes.add(shape)
        args = [x for x in command[:-4] if not x.startswith('--dependency=')]
        args.insert(1, '--test-only'); args.append('--wrap=true')
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        print((result.stdout+result.stderr).strip(), flush=True)
        if result.returncode:
            raise PermissionError('Hydra rejected the reviewed resource shape')


def environment(expected):
    from hlt_classification.jetclass2_delphes.model import installed_environment
    actual = installed_environment()
    if actual != expected:
        keys = [k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)]
        raise ValueError('Recorded SPORC software differs: '+', '.join(sorted(keys)))
    return actual


def allocation(spec):
    import torch
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real Slurm job is required')
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True)
    fields = dict(t.split('=', 1) for t in result.stdout.split() if '=' in t)
    expected = dict(Account=SITE['account'], Partition=SITE['partition'], QOS=SITE['qos'],
                    NumNodes='1', NumTasks='1', NumCPUs=str(spec['cpus']))
    if (any(fields.get(k) != v for k, v in expected.items())
            or os.environ.get('SLURM_CLUSTER_NAME') != SITE['cluster']
            or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != spec['cpus']
            or int(os.environ.get('SLURM_MEM_PER_NODE', '0')) != spec['memory_mb']):
        raise PermissionError('Not the registered Hydra allocation')
    if (platform.machine() != SITE['architecture'] or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE') != '1'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != SITE['gpu_name'] or not torch.cuda.is_bf16_supported()):
        raise PermissionError('Use the registered isolated environment and exactly one L40')
    environment(spec['reference']['environment'])
    return job
