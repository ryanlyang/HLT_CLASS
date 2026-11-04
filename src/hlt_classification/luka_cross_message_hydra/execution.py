"""Separate A6000 execution qualification; original L40 evidence stays intact."""
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_u100_pair_hydra import site, portability, precision_campaign
from hlt_classification.luka_u100_pair_hydra.contracts import SITE
from .contracts import artifact, validate

A6000_SITE = dict(SITE, gres='gpu:nvidia_rtx_a6000:1', gpu_name='NVIDIA RTX A6000')


def allocation(spec):
    if spec.get('gpu_profile') != 'a6000':
        return site.allocation(spec)
    if spec['execution_site'] != A6000_SITE or spec['schema_version'] not in (2, 3):
        raise ValueError('A6000 requires its own v2/v3 execution registration')
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real Slurm allocation is required')
    controller = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True,
                                capture_output=True, text=True).stdout
    fields = dict(t.split('=', 1) for t in controller.split() if '=' in t)
    config = subprocess.run(['scontrol', 'show', 'config'], check=True, capture_output=True, text=True).stdout
    match = re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$', config)
    errors = site._scheduler_errors(spec, job, fields, os.environ, match[1] if match else None)
    if errors:
        raise PermissionError('A6000 allocation differs:\n  '+'\n  '.join(errors))
    import torch
    if (platform.machine() != A6000_SITE['architecture']
            or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE') != '1'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != A6000_SITE['gpu_name']
            or torch.cuda.get_device_capability(0) != (8, 6)
            or not torch.cuda.is_bf16_supported()):
        raise PermissionError('Use the pinned environment and exactly one RTX A6000')
    site.environment(spec['reference']['environment'])
    print(f'Cross-message A6000 allocation: job={job}, CPUs={spec["cpus"]}, RAM={spec["memory_mb"]} MiB', flush=True)
    return job


def check_site(plan):
    commands = [row['command'] for row in plan['commands']]
    if not commands:
        raise ValueError('Empty scheduler plan')
    if all('--gres='+SITE['gres'] in c for c in commands):
        return site.check_site(plan)
    env = clean_environment()
    config = subprocess.run(['scontrol', 'show', 'config'], check=True,
                            capture_output=True, text=True, env=env).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$', config):
        raise PermissionError('Submit on Brown CS Hydra')
    shapes = set()
    for cmd in commands:
        for key in ('partition', 'account', 'qos', 'gres'):
            if [x for x in cmd if x.startswith('--'+key+'=')] != ['--'+key+'='+A6000_SITE[key]]:
                raise ValueError('Expected one consistent A6000 execution profile')
        shape = tuple(x for x in cmd if x.startswith(('--mem=', '--time=', '--cpus-per-task=', '--gres=')))
        if shape in shapes:
            continue
        shapes.add(shape)
        args = [x for x in cmd[:-4] if not x.startswith('--dependency=')]
        args.insert(1, '--test-only'); args.append('--wrap=true')
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        print((result.stdout+result.stderr).strip(), flush=True)
        if result.returncode:
            raise PermissionError('Hydra rejected the A6000 resource request')


def control_predictions(spec):
    if spec.get('gpu_profile') == 'a6000':
        return Path(spec['root'])/'setup/control_precision/hydra_validation_predictions.npz'
    return spec['portability_evidence']['refs']['predictions']['path']


def setup_record(spec, val):
    if spec.get('gpu_profile') != 'a6000':
        return portability.setup_record(spec, val)
    import numpy as np
    from .inputs import Bundle
    saved = load_npz_arrays(Bundle(spec['bundle_root']).path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k], getattr(val, k)) for k in ('identities', 'labels')):
        raise ValueError('Control replay population/order differs')
    folder = Path(spec['root'])/'setup/control_precision'; folder.mkdir(exist_ok=False)
    # Reuse unchanged diagnostic kernels, not their old L40 admission result.
    # Runs full native-control validation plus BF16 repeat and CPU/GPU FP32 probes.
    report = precision_campaign.evaluate(spec, folder)
    value = artifact('GPU_PORTABILITY', parents=dict(spec=spec['content_hash'],
        historical=spec['portability_evidence']['content_hash']), profile='a6000',
        report=file_ref(folder/'report.json'), report_hash=report['content_hash'],
        reference_retrained=False, control_inference_reused=False,
        policy=portability.policy(), gpu=report['gpu'], environment=report['environment'])
    validate_setup(spec, value)
    return value


def validate_setup(spec, value):
    if spec.get('gpu_profile') != 'a6000':
        return portability.validate_setup(spec, value)
    from hlt_classification.luka_u100_pair_hydra.contracts import validate as old_validate
    from .inputs import Bundle
    validate(value, 'GPU_PORTABILITY', dict(spec=spec['content_hash'],
        historical=spec['portability_evidence']['content_hash']))
    historical, _ = portability.validate_evidence(spec['portability_evidence'], spec['bundle_root'],
                                                spec['reference'], spec['counts'])
    folder = Path(spec['root'])/'setup/control_precision'; bundle = Bundle(spec['bundle_root'])
    validate_file_ref(value['report'])
    report = load_json(folder/'report.json')
    old_validate(report, 'PRECISION_REPORT', dict(spec=spec['content_hash']))
    paths = ('hydra_validation_predictions.npz', 'full_validation.json', 'fixed_batch_predictions.npz')
    expected = artifact('GPU_PORTABILITY', parents=value['parents'], profile='a6000',
        report=file_ref(folder/'report.json'), report_hash=report['content_hash'],
        reference_retrained=False, control_inference_reused=False,
        policy=portability.policy(), gpu=report['gpu'], environment=report['environment'])
    if (value != expected or report['gpu']['name'] != A6000_SITE['gpu_name']
            or report['gpu']['compute_capability'] != [8, 6]
            or report['environment'] != spec['reference']['environment']
            or report['checkpoint'] != file_ref(bundle.path('reference/matched/checkpoint.pt'))
            or report['saved_predictions'] != file_ref(bundle.path('reference/matched/predictions.npz'))
            or report['outputs'] != [file_ref(folder/n, root=folder) for n in paths]
            or report['reference_retrained'] is not False
            or report['full_validation']['settings'] != historical['full_validation']['settings']):
        raise ValueError('A6000 control replay lineage/backend differs')
    full = load_json(folder/'full_validation.json')
    old_validate(full, 'PRECISION_FULL', dict(spec=spec['content_hash']))
    if any(full[k] != v for k, v in report['full_validation'].items()):
        raise ValueError('Full replay report differs')
    # Recompute the original thresholds; probability drift stays report-only.
    portability.qualify(report, load_npz_arrays(folder/'fixed_batch_predictions.npz'),
        load_npz_arrays(control_predictions(spec)),
        load_npz_arrays(bundle.path('reference/matched/predictions.npz')), spec['counts']['validation'])
    return value
