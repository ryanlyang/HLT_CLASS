"""Real-Slurm acceptance, full CE baselines, and metric-independent completion."""
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
import torch

from hlt_classification.cms_proxy_ladder.production import _execution_gate
from hlt_classification.data.cache_contracts import atomic_publish_bytes
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import train_kernel
from . import campaign as c, cache
from .contracts import artifact, file_ref, write_json, load_json


def _caches(spec, parent, candidate):
    started = time.monotonic()
    budgets = cache.cache_budgets(parent['foundation'], spec['memory_mb'], spec['workers'])
    identity = c.candidate_identity(spec, parent, candidate)
    train, diagnostics = cache.prepare(parent, candidate, 'train', workers=spec['workers'],
        input_identity=identity, max_ram_bytes=budgets['train'])
    validation, val = cache.prepare(parent, candidate, 'validation', workers=spec['workers'],
        input_identity=identity, max_ram_bytes=budgets['validation'])
    return train, validation, diagnostics, dict(train=diagnostics['endpoint_sha256'],
        validation=val['endpoint_sha256']), time.monotonic()-started


def measure(spec, parent):
    _execution_gate(parent, 'cuda')
    replay = cache.replay(parent, spec['workers'])
    train, validation, _, _, seconds = _caches(spec, parent, 'S1')
    parity = installed_parity(train, device='cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    node = c.node(parent, 'S1')
    torch.manual_seed(node['initialization_seed'])
    model = DelphesParticleTransformer()
    acceptance, _ = train_kernel(model, train, validation, node=node, device='cuda', acceptance_passes=1)
    minutes = max(60, parent['runtime_profile']['train_minutes'],
        math.ceil((seconds+100*acceptance['runtime_seconds'])*1.75/60))
    result = artifact('PREFLIGHT', parents=dict(spec=spec['content_hash']), passed=True,
        counts=dict(train=len(train), validation=len(validation)), source_commit=spec['source_commit'],
        site=spec['execution_site'], installed_environment=installed_environment(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], installed_parity=parity, acceptance=acceptance,
        cache_seconds=seconds, cache_bytes=train.nbytes+validation.nbytes,
        gpu=gpu_identity(), gpu_peak_bytes=torch.cuda.max_memory_allocated(), train_minutes=minutes,
        replay=replay, replayed_candidates=list(c.CANDIDATES), exact_process_replay=True)
    return c.validate_preflight(result, spec, parent)


def fit(spec, parent, candidate):
    measured = c.preflight(spec)
    _execution_gate(parent, 'cuda')
    train, validation, diagnostics, digests, seconds = _caches(spec, parent, candidate)
    node = c.node(parent, candidate)
    torch.manual_seed(node['initialization_seed'])
    model = DelphesParticleTransformer()
    training, state = train_kernel(model, train, validation, node=node, device='cuda')
    root = Path(spec['root'])/('fit_'+candidate)
    root.mkdir(exist_ok=False)
    payload = BytesIO(); torch.save(state, payload)
    atomic_publish_bytes(root/'selected.pt', payload.getvalue())
    write_json(root/'training_report.json', training)
    identity = c.candidate_identity(spec, parent, candidate)
    diagnostic = artifact('DIAGNOSTICS', parents=dict(spec=spec['content_hash'], input=identity),
        candidate=candidate, role='train', development_only=True, **diagnostics)
    write_json(root/'diagnostics.json', diagnostic)
    result = artifact('FIT', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash']),
        candidate=candidate, input_identity=identity, counts=dict(train=len(train), validation=len(validation)),
        source_commit=spec['source_commit'], endpoint_sha256=digests, cache_seconds=seconds,
        training=file_ref(root/'training_report.json', root=root),
        checkpoint=file_ref(root/'selected.pt', root=root), diagnostics=file_ref(root/'diagnostics.json', root=root))
    write_json(root/'receipt.json', result)
    c.completed_fit(spec, parent, candidate)
    return result


def run(spec, task):
    parent = c.validate_spec(spec)
    root = Path(spec['root'])
    if task not in ('preflight', 'summary', *('fit_'+n for n in c.CANDIDATES)):
        raise ValueError('Unknown screen task')
    if task == 'preflight' and (root/'preflight.json').is_file():
        return c.preflight(spec)
    if task.startswith('fit_') and (root/task/'receipt.json').is_file():
        return c.completed_fit(spec, parent, task[4:])[0]
    if task == 'summary' and (root/'summary.json').is_file():
        expected = c.summarize(spec)
        if load_json(root/'summary.json') != expected:
            raise ValueError('Saved summary differs')
        return expected
    # A failed/interrupted task needs explicit inspection, never silent overwrite/retry.
    claim = root/(task+'.claim')
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(artifact('CLAIM', parents=dict(spec=spec['content_hash']), task=task), handle)
        handle.flush(); os.fsync(handle.fileno())
    print('GAP start '+task, flush=True)
    if task == 'preflight':
        value = measure(spec, parent)
        write_json(root/'preflight.json', value)
    elif task == 'summary':
        value = c.summarize(spec)
        write_json(root/'summary.json', value)
    else:
        value = fit(spec, parent, task[4:])
    print('GAP complete '+task, flush=True)
    return value
