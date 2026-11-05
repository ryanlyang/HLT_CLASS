"""Exact-job execution, HLT-only miniature, then a fresh full CE fit."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_u100_contrastive.training import peaks
from hlt_classification.luka_cross_message_hydra import execution
from hlt_classification.luka_cross_message_hydra.resources import Subset, miniature_indices
from .contracts import TASKS, artifact, validate, resource_policy
from . import campaign, submission, training


def miniature_rows(cache):
    """Keep worst shapes/all classes and ensure a partial batch when possible."""
    indices = miniature_indices(cache)
    if len(indices) % 128 == 0:
        selected = set(indices.tolist())
        if len(cache) > len(indices):
            extra = next(i for i in range(len(cache)) if i not in selected)
            indices = np.append(indices,extra).astype(np.int64)
        elif len(indices) > 128:
            # For a small synthetic population whose entire size is a multiple
            # of128, retain all classes while leaving one partial batch.
            labels = cache.labels[indices]
            remove = next(i for i in reversed(range(len(indices))) if np.count_nonzero(labels==labels[i])>1)
            indices = np.delete(indices,remove)
    return indices


def resource_record(spec, task, report, memory, gpu, elapsed_seconds, *, warmup):
    policy = resource_policy()
    if spec.get('resource_policy') != policy:
        raise ValueError('CE-only warmed resource policy differs')
    # Preparation, cold kernels, allocator startup and warmup publication are
    # paid once in elapsed_seconds, not repeated for every projected batch.
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError('Invalid CE-only elapsed time')
    for value in (warmup, report):
        if (value['task'] != task or value['technical'] is not True or value['epochs'] != 1
                or value['parents']['spec'] != spec['content_hash']
                or len(value['validation_history']) != 1):
            raise ValueError('Expected fresh warmup and measured CE-only miniatures')
    if warmup['counts'] != report['counts']:
        raise ValueError('Warmup and measured miniature shapes differ')
    row = report['validation_history'][0]
    if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('train_seconds','validation_seconds')):
        raise ValueError('Invalid CE-only miniature timing')
    perstep = row['train_seconds']/math.ceil(report['counts']['train']/128)
    vstep = row['validation_seconds']/math.ceil(report['counts']['validation']/128)
    estimate = 30*(perstep*math.ceil(spec['counts']['train']/128)+vstep*math.ceil(spec['counts']['validation']/128))
    minutes = math.ceil((policy['safety_multiplier']*estimate+elapsed_seconds+policy['fixed_overhead_seconds'])/60)
    limits = dict(minutes=spec['minutes'][task], gpu_peak_bytes=int(gpu['total_memory_bytes']*policy['memory_fraction']),
                  peak_rss_bytes=int(spec['memory_mb']*1024**2*policy['memory_fraction']))
    measurements = dict(minutes=minutes,gpu_peak_bytes=memory['gpu_peak_bytes'],peak_rss_bytes=memory.get('peak_rss_bytes'))
    violations = []
    for key, value in measurements.items():
        if value is None and key == 'peak_rss_bytes': continue  # Local non-POSIX fixtures only.
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Nonfinite/invalid CE-only resource measurement')
        if value > limits[key]: violations.append(dict(resource=key,measured=value,limit=limits[key]))
    return artifact('RESOURCE_CHECK', parents=dict(spec=spec['content_hash'],
        acceptance=spec['qualification']['acceptance']['content_hash'],miniature=report['content_hash'],
        warmup=warmup['content_hash']),
        task=task,passed=not violations,violations=violations,limits=limits,memory=memory,gpu=gpu,
        environment=installed_environment(),projected_minutes=minutes,train_step_seconds=perstep,
        validation_step_seconds=vstep,elapsed_seconds=elapsed_seconds,safety_multiplier=2,
        resource_policy=policy,quality_gate=False)


def integrated_check(spec, task, train, val, started, *, device='cuda'):
    root = Path(spec['root'])/task
    tr = Subset(train,miniature_rows(train)); va = Subset(val,miniature_rows(val))
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    print('CE-only resource warmup: disposable production fit, excluded from per-batch projection.',flush=True)
    warmup = training.fit(root/'warmup',spec,task,tr,va,device=device,technical=True)
    # Do not empty the CUDA cache between the warmup and measured fit. Both
    # instantiate fresh matched weights/optimizer; neither carries into science.
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    print('CE-only resource measurement: fresh production fit after warmup.',flush=True)
    report = training.fit(root/'miniature',spec,task,tr,va,device=device,technical=True)
    memory = peaks(device)
    record = resource_record(spec,task,report,memory,gpu_identity(),time.perf_counter()-started,warmup=warmup)
    write_immutable_json(root/'resource_check.json',record)
    if not record['passed']:
        raise RuntimeError('CE-only resource limits exceeded: '+str(record['violations'])+
                           '; preserve resource_check.json')
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    print('CE-only production miniature passed; starting fresh matched initialization.',flush=True)
    return record


def run(spec, task):
    if task not in TASKS: raise ValueError('Unknown CE-only task')
    parent = campaign.check(spec); require_backend()
    # Parent and addon request the identical resource profile. Its allocation
    # validator checks that profile; no parent artifact is relabelled as v1.
    job = execution.allocation(parent)
    expected = spec['qualification']['acceptance']
    if gpu_identity() != expected['gpu'] or installed_environment() != expected['environment']:
        raise PermissionError('GPU/software differs from qualified parent')
    for _ in range(12):
        if submission.accepted_jobs(spec).get(task) == job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the accepted exact CE-only job')
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    root = Path(spec['root'])/task; root.mkdir(exist_ok=False); started = time.perf_counter()
    train = campaign.load_hlt(spec,'train'); val = campaign.load_hlt(spec,'validation')
    integrated_check(spec,task,train,val,started)
    training.fit(root/'fit',spec,task,train,val)
    campaign.check(spec)
    record = artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()])
    write_immutable_json(root/'receipt.json',record)
    return record


def receipt(spec, task):
    root = Path(spec['root'])/task; record = load_json(root/'receipt.json')
    validate(record,'TASK',dict(spec=spec['content_hash']))
    names = [r['path'] for r in record['outputs']]
    required = {'resource_check.json','warmup/receipt.json','miniature/receipt.json','fit/receipt.json',
                'fit/training_report.json','fit/selected.pt','fit/validation_predictions.npz'}
    if (record['task'] != task or record['job_id'] != submission.accepted_jobs(spec).get(task)
            or len(names) != len(set(names)) or not required <= set(names)):
        raise ValueError('CE-only exact-job/output ownership differs')
    for ref in record['outputs']: validate_file_ref(ref,root=root)
    warmup,_ = training.validate_fit(root/'warmup',spec,task,technical=True)
    mini,_ = training.validate_fit(root/'miniature',spec,task,technical=True)
    check = load_json(root/'resource_check.json')
    validate(check,'RESOURCE_CHECK',dict(spec=spec['content_hash'],
        acceptance=spec['qualification']['acceptance']['content_hash'],miniature=mini['content_hash'],
        warmup=warmup['content_hash']))
    recomputed = resource_record(spec,task,mini,check['memory'],check['gpu'],check['elapsed_seconds'],warmup=warmup)
    if (check != recomputed or check['passed'] is not True
            or check['gpu'] != spec['qualification']['acceptance']['gpu']):
        raise ValueError('CE-only production miniature qualification differs')
    return record
