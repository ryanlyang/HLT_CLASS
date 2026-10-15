"""Fresh matched batch128 campaign; import cached inputs, never old teachers."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from . import alljets_fusion_campaign as c, alljets_fusion_isolated_preflight as iso
from .foundation import _source
from .preflight_reuse import _files
from .regular_recovery import failed_accounting

FAILED_COMMIT = '5cab18304d991e46457aa0d9318f8731ae3b7338'
FAILED_JOB = '21898735'
ORDER = (*iso.ORDER, 'U000_probe')
POLICY = dict(iso.POLICY, version='isolated_matched_batch128_v1', order=list(ORDER),
              scientific_recipe_changed=True, stress_batch=128,
              technical_teacher='uniform_11_class_technical_only')
ADDITIONS = {'src/hlt_classification/luka_fullsim/alljets_batch128.py',
             'src/hlt_classification/jetclass2_delphes/batch128_recipe.py'}
CHANGES = {'src/hlt_classification/luka_fullsim/alljets_fusion_'+n+'.py'
           for n in ('campaign', 'worker', 'preflight', 'submission', 'isolated_preflight')}
CHANGES |= {'src/hlt_classification/jetclass2_delphes/runner.py'}
REQUIRED = sorted(ADDITIONS | CHANGES) + [
    'scripts/luka_fullsim_alljets_fusion.py', 'scripts/queue_luka_alljets_batch128.sh',
    'sbatch/run_luka_fullsim_alljets_fusion.sh', 'tests/test_luka_alljets_batch128.py',
    'docs/plans/LUKA_ALLJETS_BATCH128_PLAN.md', 'docs/contracts/LUKA_ALLJETS_BATCH128.md',
    'docs/LUKA_ALLJETS_BATCH128_RUNBOOK.md']
ORIGINAL_CHECK = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c
from hlt_classification.luka_fullsim.alljets_fusion_submission import accepted_jobs
g=load_json(sys.argv[1]); c.validate_gate(g)
print('LUKA_BATCH128_AUTH='+json.dumps(dict(gate=g['content_hash'],jobs=accepted_jobs(g))))
'''


def source(project, commit):
    value = _source(Path(project), commit)
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                             capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit all batch128 implementation/tests/contracts before queueing')
    if Path(__file__).resolve() != Path(project).resolve()/'src/hlt_classification/luka_fullsim/alljets_batch128.py':
        raise ValueError('Batch128 import escapes pinned source')
    return value


def compatible(old, active, project):
    before_root, after_root = Path(old['project_dir']), Path(project)
    validate_source_snapshot(old['source_snapshot'], repository=before_root, require_clean=True)
    validate_source_snapshot(active, repository=after_root, require_clean=True)
    before, after = _files(before_root), _files(after_root)
    if not before or not before <= after or after-before != ADDITIONS:
        raise ValueError('Unregistered batch128 source additions/deletions')
    records = {}
    for name in sorted(before):
        prior, current = sha256_file(before_root/name), sha256_file(after_root/name)
        if prior != current and name not in CHANGES:
            raise ValueError('Unregistered model/data/source change: '+name)
        records[name] = dict(original=prior, execution=current)
    return dict(modules=records, additions=sorted(ADDITIONS), allowed_changes=sorted(CHANGES))


def authenticate(path, active, project):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    c.validate(old, 'GATE')
    if (old['schema_version'] != 3 or old['source_commit'] != FAILED_COMMIT
            or old['source_snapshot']['git_commit'] != FAILED_COMMIT
            or path != Path(old['root'])/'gate_spec.json'
            or any(old.get(k) != v for k,v in c.registration(3).items())):
        raise ValueError('Expected original failed 5cab1830 isolated gate')
    reuse = compatible(old, active, project)
    root = Path(old['root'])
    if (root/'preflight/preflight.json').exists() or (root/'science').exists():
        raise PermissionError('Expected unadmitted failed gate; preserve prior work')
    env = dict(os.environ, PYTHONPATH=str(Path(old['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating original batch256 gate/source and reusable inputs; no model import...', flush=True)
    proc = subprocess.run([sys.executable, '-u', '-s', '-c', ORIGINAL_CHECK, str(path)],
        cwd=old['project_dir'], env=env, capture_output=True, text=True)
    if proc.returncode:
        raise ValueError('Original-source authentication failed:\n'+proc.stderr)
    rows = [json.loads(line.removeprefix('LUKA_BATCH128_AUTH=')) for line in proc.stdout.splitlines()
            if line.startswith('LUKA_BATCH128_AUTH=')]
    if rows != [dict(gate=old['content_hash'], jobs=dict(preflight=FAILED_JOB))]:
        raise ValueError('Original gate/journal differs')
    runtime_path = root/'preflight/runtime.json'
    if str(load_json(runtime_path)['job_id']) != FAILED_JOB:
        raise ValueError('Original runtime job differs')
    failure_path = root/'preflight/failure.json'
    failure = load_json(failure_path)
    if (failure.get('phase') != 'stress' or failure.get('node') != ORDER[0]
            or failure.get('exit_code') != 1 or failure.get('science_admissible') is not False):
        raise ValueError('Expected failed isolated D000 stress')
    return old, dict(failed_gate=file_ref(path), original_runtime=file_ref(runtime_path),
        failure=file_ref(failure_path), log=file_ref(root/f'preflight-{FAILED_JOB}.out'),
        source_reuse=reuse, accounting=dict(job_id=FAILED_JOB, state='FAILED', exit_code='1:0'))


def location(root, old, project):
    root = Path(root).resolve()
    parent = load_json(old['parent_import']['spec']['path'])
    c.protect(root, parent, Path(project).resolve())
    pending, seen = [old], set()
    while pending:
        prior = pending.pop()
        for name in (prior['root'], prior['project_dir']):
            path = Path(name).resolve()
            if root.is_relative_to(path) or path.is_relative_to(root):
                raise PermissionError('Batch128 root overlaps prior source/artifacts')
        for key in ('failed_gate', 'original_gate', 'diagnostic'):
            ref = prior.get('recovery', {}).get(key)
            if ref and ref['path'] not in seen:
                seen.add(ref['path'])
                pending.append(load_json(ref['path']))
    return parent


def create(*, failed_gate, root, project, commit):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source(project, commit)
    old, reuse = authenticate(failed_gate, active, project)
    location(root, old, project)
    if failed_accounting(FAILED_JOB) != reuse['accounting']:
        raise ValueError('Terminal failed accounting differs')
    gate = c.artifact('GATE', version=4, parents=dict(source=active['content_hash'], failed_gate=old['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        parent_import=old['parent_import'], prepared_sha256=old['prepared_sha256'], recovery=reuse,
        **c.registration(4))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def validate_gate(gate):
    c.validate(gate, 'GATE')
    if (gate['schema_version'] != 4 or any(gate.get(k) != v for k,v in c.registration(4).items())
            or source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Batch128 source/registration differs')
    for key in ('failed_gate', 'original_runtime', 'failure', 'log'):
        validate_file_ref(gate['recovery'][key])
    old, reuse = authenticate(gate['recovery']['failed_gate']['path'], gate['source_snapshot'], gate['project_dir'])
    parent = location(gate['root'], old, gate['project_dir'])
    if (gate['recovery'] != reuse or gate['parent_import'] != old['parent_import']
            or gate['prepared_sha256'] != old['prepared_sha256']
            or gate['parents'] != dict(source=gate['source_snapshot']['content_hash'], failed_gate=old['content_hash'])):
        raise ValueError('Batch128 lineage differs')
    return parent


def stress(model, train, *, device='cuda'):
    """Same longest-real-row stress as the donor, now true128/three updates."""
    from hlt_classification.context_fusion.inputs import PairedCache
    from hlt_classification.context_fusion.worker import largest_raw
    from hlt_classification.jetclass2_delphes.runner import _forward
    from hlt_classification.jetclass2_delphes.model import distillation_loss
    if train.role != 'train':
        raise PermissionError('Backward stress is TRAIN-only')
    if isinstance(train, PairedCache):
        raws = [largest_raw(view, size=128) for view in (train.primary, train.context)]
        length = max(r['mask'].shape[-1] for r in raws)
        raw = dict(labels=raws[0]['labels'])
        for k in ('features', 'vectors', 'mask'):
            raw[k] = np.stack([np.pad(r[k], ((0,0),(0,0),(0,length-r[k].shape[-1]))) for r in raws], 1)
        model.reset_pair_offload_stats()
    else:
        raw = largest_raw(train, size=128)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
        loss = distillation_loss(logits, torch.from_numpy(raw['labels']).to(device),
            teacher_probabilities=torch.full((128,11), 1/11, device=device))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite batch128 stress loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite batch128 stress gradient')
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return model.pair_offload_stats() if isinstance(train, PairedCache) else None
