"""Measured native acceptance, whole-ladder walltime and storage lifecycle."""
import math
from pathlib import Path
import time
import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment, distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_attention.preflight import parity
from hlt_classification.luka_attention.model import new_model, attention_kl
from hlt_classification.luka_attention import banks
from . import campaign, worker, retention
from . import data
from .contracts import artifact, validate

def stress(cache,node,device,*,backward):
    n=banks.counts(cache)
    witness=int(np.argmax(n))
    raw=cache.batch(np.repeat(witness,128))
    if int(raw['mask'].sum(-1).max())!=int(n.max()): raise ValueError('Wrong longest witness')
    model=new_model(node).to(device)
    model.train(backward); model.capture_attention=backward
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits=_forward(model,raw,device,bf16=str(device).startswith('cuda'))
            if backward:
                mask=torch.from_numpy(raw['mask']).to(device)
                a=model.take_attention()
                q=torch.ones_like(a)
                loss=distillation_loss(logits,torch.from_numpy(raw['labels']).to(device),
                                       teacher_probabilities=torch.full_like(logits.float(),1/11))
                loss=loss+node['attention_coefficient']*attention_kl(a,q,mask)
                if not torch.isfinite(loss): raise ValueError('Nonfinite stress loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite stress gradient')
                optimizer.step()
            elif not torch.isfinite(logits).all(): raise ValueError('Nonfinite validation stress')
    del model,optimizer
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return dict(role=cache.role,coordinate=cache.coordinate_name,maximum=int(n.max()),
                batch=128,steps=3,backward=backward,coefficient=node['attention_coefficient'])


def budgets(facts, authentication_seconds):
    rows = facts['measurements']
    values = [authentication_seconds, facts['anchor_seconds'], facts['construction_seconds']]
    values += [m[k] for m in rows for k in ('load_seconds','epoch_seconds','readback_seconds','export_seconds')]
    if len(rows) != 3 or any(type(v) not in (float,int) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Missing/invalid complete-chain measurement')
    seconds = authentication_seconds+facts['construction_seconds']+facts['anchor_seconds']+sum(
        m['load_seconds']+30*m['epoch_seconds']+m['readback_seconds']+m['export_seconds'] for m in rows)
    minutes = max(60, math.ceil(1.75*seconds/60))
    if minutes > 1440:
        raise ValueError('Measured whole ladder exceeds registered 24h ceiling; no shortening')
    return minutes


def memory_ok(gate, snapshot, gpu):
    for key in ('peak_rss_bytes','gpu_peak_bytes'):
        if type(snapshot[key]) is not int or snapshot[key] <= 0:
            raise ValueError('Missing memory high-water evidence')
    if snapshot['peak_rss_bytes'] >= .85*gate['memory_mb']*1024**2 or snapshot['gpu_peak_bytes'] >= .9*gpu['total_memory_bytes']:
        raise MemoryError('Measured host/GPU headroom failed')


def run(gate):
    started = time.monotonic()
    campaign.check_gate(gate)
    authentication_seconds = time.monotonic()-started
    job = worker.execution(gate)
    if installed_environment() != gate['anchor']['evidence']['environment']:
        raise PermissionError('Anchor environment differs')
    root = Path(gate['root'])/'preflight'
    root.mkdir(exist_ok=False)
    scratch_root = retention.scratch(gate, job)
    write_immutable_json(root/'scratch_location.json', artifact('SCRATCH',
        parents=dict(gate=gate['content_hash']), path=str(scratch_root), job_id=job))
    torch.cuda.reset_peak_memory_stats()
    from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog
    phase = PhaseLog(root, lambda: memory_snapshot(torch))
    with phase.phase('construct_and_replay_all_swap_inputs'):
        inputs = data.build(gate)
    tr = data.get(inputs, 'train', 'U100')
    storage = retention.budget(banks.counts(tr), root, scratch_root, gate['available_quota_gib'])
    raw = tr.batch(np.arange(4))
    parities = [parity(raw, 'cuda', bf16=b) for b in (False, True)]
    del tr
    facts = worker.run_chain(gate, root/'ladder', scratch_root, technical=True, stress_fn=stress, inputs=inputs)
    worker.result_rows(gate, root/'ladder', technical=True)
    started = time.monotonic()
    campaign.check_gate(gate)
    authentication_seconds += time.monotonic()-started
    snapshot, gpu = memory_snapshot(torch), gpu_identity()
    memory_ok(gate, snapshot, gpu)
    # Retained evidence only: large technical targets/checkpoints already retired.
    started = time.monotonic()
    outputs = [file_ref(p, root=root) for p in sorted(root.rglob('*')) if p.is_file()]
    authentication_seconds += 2*(time.monotonic()-started)
    measured = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash']), job_id=job,
        environment=installed_environment(), gpu=gpu, execution_site=gate['execution_site'],
        parities=parities, facts=facts, memory=snapshot, storage=storage,
        authentication_seconds=authentication_seconds, ladder_minutes=budgets(facts, authentication_seconds),
        outputs=outputs, scientific_fit=False, technical_checks_passed=True)
    write_immutable_json(root/'measurement.json', measured)
    validate_report(gate, measured)
    write_immutable_json(root/'receipt.json', measured)
    return measured


def validate_report(gate, report):
    campaign.validate_execution(gate)
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash']))
    facts = report['facts']
    if (report['environment'] != gate['anchor']['evidence']['environment']
            or report['execution_site'] != gate['execution_site']
            or report['technical_checks_passed'] is not True or report['scientific_fit'] is not False
            or report['ladder_minutes'] != budgets(facts, report['authentication_seconds'])
            or [m['node_id'] for m in facts['measurements']] != [n['node_id'] for n in gate['nodes']]):
        raise ValueError('Preflight recipe/environment/coverage differs')
    memory_ok(gate, report['memory'], report['gpu'])
    parities = report['parities']
    required = ('logits','input_gradients','parameter_gradients','buffers','optimizer_updates','native_attention_surface')
    if len(parities) != 2 or [p['bf16'] for p in parities] != [False,True] or any(p[k] is not True for p in parities for k in required):
        raise ValueError('Missing native parity evidence')
    expected_stress = [(n['coordinate'], role, role=='train') for n in gate['nodes'] for role in ('train','validation')]
    if [(s['coordinate'],s['role'],s['backward']) for s in facts['stress']] != expected_stress:
        raise ValueError('Missing full-support stress')
    for s in facts['stress']:
        if s['batch'] != 128 or s['steps'] != 3 or not 1 <= s['maximum'] <= 512 or s['coefficient'] != gate['coefficient']:
            raise ValueError('Stress does not exercise registered coefficient/support')
    root = Path(gate['root'])/'preflight'
    outputs = report['outputs']
    if len({r['path'] for r in outputs}) != len(outputs):
        raise ValueError('Duplicate evidence file')
    for ref in outputs: validate_file_ref(ref, root=root)
    actual = {p.relative_to(root).as_posix() for p in (root/'ladder').rglob('*') if p.is_file()}
    if not actual <= {r['path'] for r in outputs}:
        raise ValueError('Unbound technical evidence')
    worker.result_rows(gate, root/'ladder', technical=True)
    for role in ('train','validation'):
        a = facts[role+'_alignment']
        if a['role'] != role or a['rows'] != gate['counts'][role] or a['prepared'] != gate['input_identity']['content_hash']:
            raise ValueError('Technical population differs')
    validate(facts['input_audit'], 'INPUT_AUDIT', dict(inputs=gate['input_identity']['content_hash']))
    if (facts['input_audit'] != load_json(root/'ladder/input_audit.json')
            or facts['input_audit']['endpoint_fingerprints_replayed'] is not True
            or type(facts['input_audit']['resident_bytes']) is not int
            or facts['input_audit']['resident_bytes'] <= 0
            or facts['input_audit']['resident_bytes'] > .30*gate['memory_mb']*1024**2):
        raise ValueError('Swap endpoint/input RAM evidence differs')
    target_bytes = (facts['storage']['attention_bank_bytes'] if gate['coefficient'] else 0)+gate['counts']['train']*92+8
    if any(type(m['resident_target_bytes']) is not int or m['resident_target_bytes'] != target_bytes
           or m['resident_target_bytes'] > .15*gate['memory_mb']*1024**2 for m in facts['measurements']):
        raise ValueError('Resident target RAM evidence differs')
    for role in ('train','validation'):
        if set(facts['input_audit']['roles'][role]['views']) != set(data.COORDINATES):
            raise ValueError('Missing swap views')
        for summary in facts['input_audit']['roles'][role]['views'].values():
            if summary['rows'] != gate['counts'][role]:
                raise ValueError('Swap view population differs')
    for node in gate['nodes']:
        step = load_json(root/'ladder'/node['node_id']/'receipt.json')
        if (step['train_alignment'] != facts['train_alignment']
                or step['validation_alignment'] != facts['validation_alignment']):
            raise ValueError('Technical rung slot alignment differs')
    for storage in (report['storage'], facts['storage']):
        needed = storage['attention_bank_bytes']+gate['counts']['train']*128+3*2**30
        if (storage['attention_bank_bytes'] <= 0 or storage['scratch_required_bytes'] != needed
                or needed > storage['scratch_free_bytes'] or storage['persistent_required_bytes'] != 2**30
                or storage['declared_quota_gib'] != gate['available_quota_gib']
                or 2**30 > min(storage['persistent_free_bytes'], gate['available_quota_gib']*2**30)):
            raise ValueError('Storage admission differs')
