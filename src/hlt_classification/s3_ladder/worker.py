"""Measured S3 KD execution, selected-state reducers and immutable receipts."""
from io import BytesIO
import json
import os
from pathlib import Path
import time
import torch

from hlt_classification.cms_proxy_ladder.production import _execution_gate
from hlt_classification.data.cache_contracts import atomic_publish_bytes
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
from . import campaign as c, cache
from .contracts import artifact, file_ref, validate_file_ref, write_json, load_json


def _caches(spec, parent, coordinate, *, roles=('train', 'validation')):
    started = time.monotonic()
    budgets = cache.cache_budgets(parent['foundation'], spec['memory_mb'], spec['workers'])
    values, digests = {}, {}
    for role in roles:
        values[role], diagnostics = cache.prepare(parent, coordinate, role, workers=spec['workers'],
            input_identity=spec['input_identity'], max_ram_bytes=budgets[role])
        digests[role] = diagnostics['endpoint_sha256']
        if coordinate == 'D000' and digests[role] != spec['frozen']['endpoint_sha256'][role]:
            raise ValueError('Full S3 cache does not match completed baseline')
    return values, digests, time.monotonic()-started


def measure(spec, parent):
    _execution_gate(parent, 'cuda')
    replay = cache.replay(parent, spec['workers'])
    endpoint, digests, endpoint_seconds = _caches(spec, parent, 'D000')
    endpoint_bytes = sum(v.nbytes for v in endpoint.values())
    del endpoint
    values, _, seconds = _caches(spec, parent, 'D066')
    train, validation = values['train'], values['validation']
    probabilities, _ = c.teacher_probabilities(spec, parent, 'OFFLINE', train.identities)
    parity = installed_parity(train, device='cuda')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    n = next(n for n in spec['graph']['nodes'] if n['coordinate'] == 'D066')
    torch.manual_seed(n['initialization_seed'])
    model = DelphesParticleTransformer()
    acceptance, _ = train_kernel(model, train, validation, node=n, device='cuda',
        teacher_probabilities=probabilities, teacher_identities=train.identities, acceptance_passes=1)
    started = time.monotonic()
    p = predict(model, train, device='cuda', temperature=2.)
    inference_seconds = time.monotonic()-started
    # Real production codec/identity join too; acceptance bank is not a science teacher.
    bank = Path(spec['root'])/'acceptance_bank'
    publish_bank(bank, foundation_sha256=spec['input_identity'], teacher_report_sha256=acceptance['content_hash'],
        teacher_node='ACCEPTANCE_ONLY', role='train', identities=train.identities, probabilities=p)
    load_bank(bank, foundation_sha256=spec['input_identity'], teacher_report_sha256=acceptance['content_hash'],
        teacher_node='ACCEPTANCE_ONLY', role='train', expected_identities=train.identities)
    seconds = max(seconds, endpoint_seconds)
    value = artifact('PREFLIGHT', parents=dict(spec=spec['content_hash']), passed=True,
        counts=dict(train=len(train), validation=len(validation)), source_commit=spec['source_commit'],
        site=spec['execution_site'], installed_environment=installed_environment(), gpu=gpu_identity(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], installed_parity=parity, acceptance=acceptance,
        cache_seconds=seconds, inference_seconds=inference_seconds,
        cache_bytes=max(endpoint_bytes, train.nbytes+validation.nbytes)+probabilities.nbytes+p.nbytes,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(), replay=replay, endpoint_sha256=digests,
        offline_bank=spec['frozen']['offline_bank'],
        **c.walltimes(parent, seconds, acceptance['runtime_seconds'], inference_seconds))
    return c.validate_preflight(value, spec, parent)


def fit(spec, parent, name):
    measured = c.preflight(spec, parent)
    _execution_gate(parent, 'cuda')
    n = c.node(spec, name)
    values, digests, seconds = _caches(spec, parent, n['coordinate'])
    train, validation = values['train'], values['validation']
    p, teacher = c.teacher_probabilities(spec, parent, n['teacher'], train.identities)
    torch.manual_seed(n['initialization_seed'])
    model = DelphesParticleTransformer()
    training, state = train_kernel(model, train, validation, node=n, device='cuda',
        teacher_probabilities=p, teacher_identities=train.identities)
    root = Path(spec['root'])/('fit_'+name)
    root.mkdir(exist_ok=False)
    payload = BytesIO(); torch.save(state, payload)
    atomic_publish_bytes(root/'selected.pt', payload.getvalue())
    write_json(root/'training_report.json', training)
    receipt = artifact('FIT', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash'], teacher=teacher),
        node=n, counts=dict(train=len(train), validation=len(validation)), source_commit=spec['source_commit'],
        endpoint_sha256=digests, cache_seconds=seconds, training=file_ref(root/'training_report.json', root=root),
        checkpoint=file_ref(root/'selected.pt', root=root))
    write_json(root/'receipt.json', receipt)
    c.completed_fit(spec, parent, name)
    return receipt


def reduce(spec, parent, name):
    c.preflight(spec, parent)
    _execution_gate(parent, 'cuda')
    fit, training = c.completed_fit(spec, parent, name)
    values, digests, _ = _caches(spec, parent, c.node(spec, name)['coordinate'], roles=('train',))
    if digests['train'] != fit['endpoint_sha256']['train']:
        raise ValueError('Reducer input differs from selected fit')
    train = values['train']
    model = DelphesParticleTransformer()
    checkpoint = validate_file_ref(fit['checkpoint'], root=Path(spec['root'])/('fit_'+name))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to('cuda').eval()
    probabilities = predict(model, train, device='cuda', temperature=2.)
    root = Path(spec['root'])/('reduce_'+name)
    root.mkdir(exist_ok=False)
    publish_bank(root/'train', foundation_sha256=spec['input_identity'], teacher_report_sha256=training['content_hash'],
        teacher_node=name, role='train', identities=train.identities, probabilities=probabilities)
    receipt = artifact('REDUCER', parents=dict(spec=spec['content_hash'], fit=fit['content_hash']),
        node_id=name, source_commit=spec['source_commit'], train_endpoint_sha256=digests['train'],
        bank=file_ref(root/'train/manifest.json', root=root))
    write_json(root/'receipt.json', receipt)
    return c.completed_reducer(spec, parent, name)


def run(spec, task):
    parent = c.validate_spec(spec)
    root = Path(spec['root'])
    registered = {r['task_id']: r for r in c.tasks(spec)}
    if task not in ('preflight', *registered):
        raise ValueError('Unknown S3 task')
    path = root/('preflight.json' if task == 'preflight' else 'summary.json' if task == 'summary' else task+'/receipt.json')
    if path.exists():
        if task == 'preflight':
            return c.preflight(spec, parent)
        if task == 'summary':
            expected = c.summarize(spec, parent)
            if load_json(path) != expected:
                raise ValueError('Saved S3 summary differs')
            return expected
        row = registered[task]
        return c.completed_fit(spec, parent, row['node_id'])[0] if row['kind'] == 'fit' else c.completed_reducer(spec, parent, row['node_id'])
    fd = os.open(root/(task+'.claim'), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(artifact('CLAIM', parents=dict(spec=spec['content_hash']), task=task), handle)
        handle.flush(); os.fsync(handle.fileno())
    print('S3 start '+task, flush=True)
    if task == 'preflight':
        value = measure(spec, parent); write_json(path, value)
    elif task == 'summary':
        c.preflight(spec, parent)
        value = c.summarize(spec, parent); write_json(path, value)
    else:
        row = registered[task]
        value = (fit if row['kind'] == 'fit' else reduce)(spec, parent, row['node_id'])
    print('S3 complete '+task, flush=True)
    return value
