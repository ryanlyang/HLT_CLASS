"""Hydra-specific scheduler and isolated-environment checks."""
from decimal import Decimal
import json
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


def _scheduler_errors(spec, job, fields, environ, cluster):
    """Cross-check available evidence; missing metadata is never a match."""
    errors = []

    def require(label, actual, expected, *, count=False):
        allowed = (str(expected), f'{expected}-{expected}') if count else (str(expected),)
        if actual not in allowed:
            errors.append(f'{label}: expected {expected!r}, observed {actual!r}')

    for key, value in dict(JobId=job, JobState='RUNNING', Account=SITE['account'],
                           Partition=SITE['partition'], QOS=SITE['qos']).items():
        require(key, fields.get(key), value)
    require('NumNodes', fields.get('NumNodes'), 1, count=True)
    require('NumCPUs', fields.get('NumCPUs'), spec['cpus'], count=True)
    require('ClusterName', cluster, SITE['cluster'])
    if environ.get('SLURM_CLUSTER_NAME') not in (None, '(null)'):
        require('SLURM_CLUSTER_NAME', environ['SLURM_CLUSTER_NAME'], SITE['cluster'])

    # Slurm can render an unspecified controller task field as N/A. Use the
    # explicitly requested batch environment in that case, while rejecting
    # any disagreement when both sources contain concrete values.
    for key, envkey, expected in (('NumTasks', 'SLURM_NTASKS', 1),
                                  ('CPUs/Task', 'SLURM_CPUS_PER_TASK', spec['cpus'])):
        reported = fields.get(key) not in (None, 'N/A', '(null)')
        if reported:
            require(key, fields[key], expected)
        if envkey in environ:
            require(envkey, environ[envkey], expected)
        elif not reported:
            errors.append(f'{key}/{envkey}: no task allocation evidence')

    # Prefer the actual controller allocation to a required environment export.
    # Still check an exported value: a conflicting environment must not pass.
    tres = dict(t.split('=', 1) for t in fields.get('AllocTRES', '').split(',') if '=' in t)
    memory = tres.get('mem')
    if memory is not None:
        match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGT]?)', memory)
        mib = (Decimal(match[1]) * {'K': Decimal(1)/1024, '': 1, 'M': 1, 'G': 1024,
                                   'T': 1024**2}[match[2]]) if match else None
        if mib != spec['memory_mb']:
            errors.append(f'AllocTRES.mem: expected {spec["memory_mb"]} MiB, observed {memory!r}')
    if 'SLURM_MEM_PER_NODE' in environ:
        require('SLURM_MEM_PER_NODE', environ['SLURM_MEM_PER_NODE'], spec['memory_mb'])
    elif memory is None:
        errors.append('AllocTRES.mem/SLURM_MEM_PER_NODE: no memory allocation evidence')
    return errors


def allocation(spec):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real Slurm job is required')
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True)
    fields = dict(t.split('=', 1) for t in result.stdout.split() if '=' in t)
    # Whitelist diagnostic fields; never dump unrelated environment variables.
    evidence = dict(job_id=job, controller={k: fields.get(k) for k in (
        'JobId', 'JobState', 'Account', 'Partition', 'QOS', 'NumNodes', 'NumTasks',
        'NumCPUs', 'CPUs/Task', 'AllocTRES')}, environment={k: os.environ.get(k) for k in (
        'SLURM_CLUSTER_NAME', 'SLURM_NTASKS', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE')})
    print('Hydra allocation evidence: '+json.dumps(evidence, sort_keys=True), flush=True)
    cluster = os.environ.get('SLURM_CLUSTER_NAME')
    # Hydra job 131385 exported the literal placeholder, not an actual name.
    # Resolve it from the controller; never substitute the expected site name.
    if cluster in (None, '(null)'):
        config = subprocess.run(['scontrol', 'show', 'config'], check=True,
                                capture_output=True, text=True).stdout
        match = re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$', config)
        cluster = match[1] if match else None
        print('Hydra controller ClusterName: '+repr(cluster), flush=True)
    errors = _scheduler_errors(spec, job, fields, os.environ, cluster)
    if errors:
        raise PermissionError('Not the registered Hydra allocation:\n  '+'\n  '.join(errors))
    import torch
    if (platform.machine() != SITE['architecture'] or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE') != '1'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != SITE['gpu_name'] or not torch.cuda.is_bf16_supported()):
        raise PermissionError('Use the registered isolated environment and exactly one L40')
    environment(spec['reference']['environment'])
    return job
