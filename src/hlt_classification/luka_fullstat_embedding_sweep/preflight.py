"""One full-population epoch plus native/auxiliary parity and measured admission."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import os
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import _forward, train_kernel
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_embedding_aux.model import new_model, head_input
from hlt_classification.luka_embedding_aux.preflight import parity
from hlt_classification.luka_fullstat_attention_fast.preflight import selected_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from . import campaign, worker, banks, training
from .contracts import artifact, validate, recipe, WEIGHTS, arm_gate


def lengths(cache):
    return np.concatenate([np.diff(block.offsets) for block in cache.blocks])


def stress(cache, node, device, *, backward, offline=False):
    ns = lengths(cache)
    raw = cache.batch(np.repeat(int(np.argmax(ns)), 128))
    model = (DelphesParticleTransformer() if offline else new_model(node)).to(device)
    model.train(backward)
    if not offline:
        model.capture = backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            if offline:
                tensors = {k: torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')}
                logits, z = head_input(model, **tensors)
                if not torch.isfinite(z).all():
                    raise ValueError('Nonfinite FP32 OFFLINE export stress')
            else:
                logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite stress logits')
            if backward:
                loss, *_ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
                    model.prediction, torch.ones_like(model.prediction), coefficient=node['auxiliary_weight'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite maximum-support loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite maximum-support gradient')
                optimizer.step()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return dict(role=cache.role, coordinate=cache.coordinate_name, maximum=int(ns.max()),
                batch=128, steps=3, backward=backward, fp32_export=offline,
                weight=None if offline else node['auxiliary_weight'])


def zero_parity(train, val, bank, node, device):
    ix = np.concatenate([np.flatnonzero(train.labels == k)[:2] for k in range(11)])
    vi = np.concatenate([np.flatnonzero(val.labels == k)[:2] for k in range(11)])
    tr, va = selected_cache(train, ix), selected_cache(val, vi)
    mini = SimpleNamespace(targets=bank.targets[ix], identities=bank.identities[ix], labels=bank.labels[ix],
        manifest=artifact('BANK', parents=bank.manifest['parents'], role='train',
                          coordinate='OFFLINE', teacher='OFFLINE', rows=len(tr)))
    with parity_backend(device):
        torch.manual_seed(node['initialization_seed'])
        report, expected = train_kernel(DelphesParticleTransformer(), tr, va, node=node,
            device=device, acceptance_passes=1, training_recipe=recipe(0.)['base'])
        actual, state = training.train(tr, va, bank=mini, node=node, device=device, technical=True)
    if actual['validation'] != report['validation'] or state.keys() != expected.keys():
        raise ValueError('Zero auxiliary native CE metrics/state differ')
    for key in state:
        torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
    return True


def admission(gate, measurement):
    times = measurement['timings']
    required = {'authentication', 'load_HLT_targets', 'train_epoch', 'validation_epoch', 'checkpoint_reporting'}
    if (set(times) != required or any(not math.isfinite(v) or v < 0 for v in times.values())
            or times['train_epoch'] <= 0 or times['validation_epoch'] <= 0
            or any(type(measurement[k]) is not int or measurement[k] <= 0
                   for k in ('peak_rss_bytes', 'gpu_peak_bytes', 'fit_bytes', 'bank_bytes'))):
        raise ValueError('Invalid full-epoch measurements')
    minutes = max(720, math.ceil(2*(2*times['authentication'] + times['load_HLT_targets']
        + 20*(times['train_epoch']+times['validation_epoch']) + times['checkpoint_reporting'] + 900)/60))
    if (minutes > gate['max_minutes']
            or measurement['peak_rss_bytes']+8*2**30 >= .85*gate['memory_mb']*1024**2
            or measurement['gpu_peak_bytes'] >= .9*measurement['gpu']['total_memory_bytes']
            or measurement['bank_bytes'] + 2*len(WEIGHTS)*measurement['fit_bytes'] + 2*2**30
               > gate['available_result_bytes']):
        raise ValueError('Measured time/memory/storage exceeds envelope; science is blocked')
    return minutes


def run(gate):
    begun = time.monotonic()
    campaign.check_gate(gate)
    authentication = time.monotonic()-begun
    job = worker.execution(gate)
    campaign.storage(gate)
    meta = worker.metadata(gate)
    root = Path(gate['root'])
    directory = root/'preflight'
    directory.mkdir(exist_ok=False)
    runtime = dict(job_id=job, environment=installed_environment(), gpu=gpu_identity(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset'))
    write_immutable_json(directory/'runtime.json', runtime)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    with log.phase('export_pure_OFFLINE_TRAIN_once'):
        off = worker.data(gate, 'train', 'OFFLINE', authenticated=meta)
        off_stress = stress(off, gate['nodes']['w0'], 'cuda', backward=False, offline=True)
        refs = gate['anchor']['evidence']['models']['OFFLINE']
        teacher = DelphesParticleTransformer()
        teacher.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']),
                                           map_location='cpu', weights_only=True), strict=True)
        bank_record = banks.export(root/'targets', teacher, off, refs=refs, device='cuda')
        del off, teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_exact_HLT_and_shared_targets') as loaded:
        tr = worker.data(gate, 'train', 'D000', authenticated=meta)
        val = worker.data(gate, 'validation', 'D000', authenticated=meta)
        bank = banks.load(gate, tr)
        banks.check_join(bank, tr)
        original = worker.validation_arrays(gate)
        if not np.array_equal(val.identities, original['identities']) or not np.array_equal(val.labels, original['labels']):
            raise ValueError('Original validation population differs')
        del original
    with log.phase('installed_native_predictor_and_zero_aux_CE_parity'):
        with parity_backend('cuda'):
            native = installed_parity(tr, device='cuda')
        modes = [parity(tr.batch(np.arange(min(4, len(tr)))), gate['nodes']['w0'], 'cuda', bf16=b)
                 for b in (False, True)]
        zero = zero_parity(tr, val, bank, gate['nodes']['w0'], 'cuda')
    measurements, stresses, requests = {}, {}, {}
    bank_bytes = sum(p.stat().st_size for p in (root/'targets').iterdir() if p.is_file())
    for arm in WEIGHTS:
        scoped = arm_gate(gate, arm)
        with log.phase(arm+'_full_population_support_stress'):
            stresses[arm] = [stress(tr, scoped['node'], 'cuda', backward=True),
                             stress(val, scoped['node'], 'cuda', backward=False)]
            for row in stresses[arm]+[off_stress]:
                if row['maximum'] != meta[0]['summaries'][row['role']][row['coordinate']]['maximum']:
                    raise ValueError('Stress does not cover full-population maximum')
            gc.collect(); torch.cuda.empty_cache()
        with log.phase(arm+'_full_technical_epoch_and_native_readback') as fitted:
            worker.fit(directory/arm, scoped, tr, val, bank, device='cuda', technical=True)
            report = worker.validate_fit(directory/arm, scoped, bank.manifest['content_hash'], technical=True)
        row = report['validation_history'][0]
        memory = memory_snapshot(torch)
        measured = dict(gpu=runtime['gpu'], environment=runtime['environment'],
            peak_rss_bytes=memory['peak_rss_bytes'], gpu_peak_bytes=memory['gpu_peak_bytes'],
            bank_bytes=bank_bytes, fit_bytes=sum(p.stat().st_size for p in (directory/arm).iterdir() if p.is_file()),
            timings=dict(authentication=authentication, load_HLT_targets=loaded['seconds'],
                train_epoch=row['train_seconds'], validation_epoch=row['validation_seconds'],
                checkpoint_reporting=max(0., fitted['seconds']-row['train_seconds']-row['validation_seconds'])))
        measurements[arm] = measured
        requests[arm] = admission(gate, measured)
        write_immutable_json(directory/(arm+'-measurement.json'), measured)
        print(f'FULLSTAT CE EMBED measured arm={arm} train_min={row["train_seconds"]/60:.2f} '
              f'val_min={row["validation_seconds"]/60:.2f} request_min={requests[arm]}', flush=True)
    campaign.check_gate(gate)
    if worker.execution(gate) != job or installed_environment() != runtime['environment'] or gpu_identity() != runtime['gpu']:
        raise ValueError('Environment/allocation changed during preflight')
    outputs = [file_ref(p, root=root) for base in (root/'targets', directory)
               for p in sorted(base.rglob('*')) if p.is_file()]
    receipt = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash'], bank=bank_record['content_hash']),
        bank_hash=bank_record['content_hash'], job_id=job, gpu=runtime['gpu'], environment=runtime['environment'],
        measurements=measurements, train_minutes=requests, native_parity=native,
        predictor_parities=modes, zero_aux_native_ce_parity=zero, stress=stresses, offline_stress=off_stress,
        full_population_training_measured=True, full_validation_measured=True,
        technical_checks_passed=True, checkpoint_probability_roundtrip=True, outputs=outputs)
    validate_report(gate, receipt)
    write_immutable_json(directory/'receipt.json', receipt)
    return receipt


def validate_report(gate, report):
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash'], bank=report['bank_hash']))
    root = Path(gate['root'])
    if (gate['recipes'] != {a:recipe(w) for a,w in WEIGHTS.items()}
            or report['environment'] != gate['anchor']['evidence']['environment']
            or 'L40S' not in report['gpu']['name']
            or not str(report['job_id']).isdigit() or int(report['job_id']) <= 0
            or any(report.get(k) is not True for k in ('technical_checks_passed',
                'checkpoint_probability_roundtrip', 'zero_aux_native_ce_parity',
                'full_population_training_measured', 'full_validation_measured'))
            or report['native_parity']['passed'] is not True
            or [p['bf16'] for p in report['predictor_parities']] != [False, True]
            or any(p.get(k) is not True for p in report['predictor_parities'] for k in
                   ('native_forward', 'zero_control_update', 'auxiliary_backbone_gradient', 'auxiliary_head_gradient'))
            or any(set(report[k]) != set(WEIGHTS) for k in ('measurements', 'stress', 'train_minutes'))):
        raise ValueError('Missing native/auxiliary/full-population Oscar evidence')
    required = {'targets/manifest.json', 'targets/targets.npz', 'preflight/runtime.json'}
    for arm in WEIGHTS:
        required.add('preflight/'+arm+'-measurement.json')
        required.update('preflight/'+arm+'/'+f for f in
                        ('receipt.json', 'selected.pt', 'training_report.json', 'validation_predictions.npz'))
    names = [r['path'] for r in report['outputs']]
    if (len(names) != len(set(names)) or not required <= set(names)
            or any(not p.startswith(('targets/', 'preflight/')) for p in names)):
        raise ValueError('Preflight evidence coverage differs')
    for ref in report['outputs']:
        validate_file_ref(ref, root=root)
    runtime = load_json(root/'preflight/runtime.json')
    if (any(runtime[k] != report[k] for k in ('job_id', 'environment', 'gpu'))
            or runtime['cublas_workspace_config'] != ':4096:8'
            or runtime['cudnn_v8_disabled'] != gate['anchor']['evidence']['cudnn_v8_disabled']):
        raise ValueError('Runtime backend evidence differs')
    m = load_json(root/'targets/manifest.json')
    refs = gate['anchor']['evidence']['models']['OFFLINE']
    validate(m, 'BANK', banks.parents(gate['anchor']['evidence']['prepared'], refs))
    if (m['content_hash'] != report['bank_hash'] or m['rows'] != gate['counts']['train']
            or m['checkpoint'] != refs['checkpoint'] or m['report'] != refs['report']
            or m['teacher'] != 'OFFLINE' or m['role'] != 'train' or m['coordinate'] != 'OFFLINE'
            or m['width'] != 128 or m['payload']['path'] != 'targets.npz'
            or dict(m['payload'], path='targets/'+m['payload']['path']) not in report['outputs']):
        raise ValueError('Embedding teacher bank binding differs')
    meta = worker.metadata(gate)[0]
    expected_off = dict(role='train', coordinate='OFFLINE', backward=False, fp32_export=True,
                        maximum=meta['summaries']['train']['OFFLINE']['maximum'], batch=128, steps=3, weight=None)
    if report['offline_stress'] != expected_off:
        raise ValueError('OFFLINE FP32 export support differs')
    for arm, weight in WEIGHTS.items():
        measurement = load_json(root/'preflight'/(arm+'-measurement.json'))
        saved = worker.validate_fit(root/'preflight'/arm, arm_gate(gate, arm), report['bank_hash'], technical=True)
        row = saved['validation_history'][0]
        expected_support = [dict(role=role, coordinate='D000', backward=backward, fp32_export=False,
            maximum=meta['summaries'][role]['D000']['maximum'], batch=128, steps=3, weight=weight)
            for role, backward in (('train', True), ('validation', False))]
        actual_bank_bytes = sum(p.stat().st_size for p in (root/'targets').iterdir() if p.is_file())
        actual_fit_bytes = sum(p.stat().st_size for p in (root/'preflight'/arm).iterdir() if p.is_file())
        if (report['measurements'][arm] != measurement
                or measurement['environment'] != report['environment'] or measurement['gpu'] != report['gpu']
                or admission(gate, measurement) != report['train_minutes'][arm]
                or report['stress'][arm] != expected_support
                or measurement['bank_bytes'] != actual_bank_bytes or measurement['fit_bytes'] != actual_fit_bytes
                or any(measurement['timings'][k] != row[v] for k, v in
                       (('train_epoch', 'train_seconds'), ('validation_epoch', 'validation_seconds')))):
            raise ValueError('Measured weight-specific resources or full-epoch evidence differ')
