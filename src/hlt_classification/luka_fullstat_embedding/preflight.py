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
from . import campaign, worker, banks, training, checkpoints
from .contracts import artifact, validate, recipe


def lengths(cache):
    return np.concatenate([np.diff(block.offsets) for block in cache.blocks])


def stress(cache, node, device, *, backward, offline=False):
    ns = lengths(cache)
    raw = cache.batch(np.repeat(int(np.argmax(ns)), 128))
    model = (DelphesParticleTransformer() if offline else new_model(node)).to(device)
    model.train(backward)
    if not offline:
        model.capture = backward
    optimizer = training.optimizer_for(model)
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
                q = torch.full_like(logits.float(), 1/11)
                loss, *_ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
                    q, model.prediction, torch.ones_like(model.prediction), .3)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite maximum-support gradient')
                optimizer.step()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return dict(role=cache.role, coordinate=cache.coordinate_name, maximum=int(ns.max()),
                batch=128, steps=3, backward=backward, fp32_export=offline)


def zero_parity(train, val, bank, node, device):
    ix = np.concatenate([np.flatnonzero(train.labels == k)[:2] for k in range(11)])
    vi = np.concatenate([np.flatnonzero(val.labels == k)[:2] for k in range(11)])
    tr, va = selected_cache(train, ix), selected_cache(val, vi)
    mini = SimpleNamespace(targets=bank.targets[ix], probabilities=bank.probabilities[ix],
        identities=bank.identities[ix], labels=bank.labels[ix],
        manifest=artifact('TECHNICAL_SUPERVISION', parents=dict(bank=bank.manifest['content_hash']),
                          prepared=tr.foundation_sha256, role='train', rows=len(tr)))
    with parity_backend(device):
        torch.manual_seed(node['initialization_seed'])
        report, expected = train_kernel(DelphesParticleTransformer(), tr, va, node=node,
            device=device, teacher_probabilities=mini.probabilities, teacher_identities=mini.identities,
            acceptance_passes=1, training_recipe=recipe()['base'])
        actual, state = training.train(tr, va, bank=mini, node=node, device=device,
                                      technical=True, zero_control=True)
    if actual['validation'] != report['validation'] or state.keys() != expected.keys():
        raise ValueError('Zero auxiliary native KD metrics/state differ')
    for key in state:
        torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
    return True


def admission(gate, measurement):
    times = measurement['timings']
    required = {'authentication', 'load_HLT_targets', 'train_epoch', 'validation_epoch', 'checkpoint_reporting'}
    if (set(times) != required or any(not math.isfinite(v) or v < 0 for v in times.values())
            or times['train_epoch'] <= 0 or times['validation_epoch'] <= 0
            or any(type(measurement[k]) is not int or measurement[k] <= 0
                   for k in ('peak_rss_bytes', 'gpu_peak_bytes', 'epoch_checkpoint_bytes'))):
        raise ValueError('Invalid full-epoch measurements')
    # OFFLINE export is already complete; science loads its compact bank once.
    minutes = max(720, math.ceil(2*(2*times['authentication'] + times['load_HLT_targets']
        + 20*(times['train_epoch']+times['validation_epoch']+times['checkpoint_reporting']) + 900)/60))
    if (minutes > gate['max_minutes']
            or measurement['peak_rss_bytes']+8*2**30 >= .85*gate['memory_mb']*1024**2
            or measurement['gpu_peak_bytes'] >= .9*measurement['gpu']['total_memory_bytes']
            or 22*measurement['epoch_checkpoint_bytes'] + 4*gate['counts']['train']*128 + 3*2**30
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
    with log.phase('authenticate_and_export_pure_OFFLINE_TRAIN'):
        off = worker.data(gate, 'train', 'OFFLINE', authenticated=meta)
        off_stress = stress(off, gate['node'], 'cuda', backward=False, offline=True)
        refs = gate['anchor']['evidence']['models']['OFFLINE']
        teacher = DelphesParticleTransformer()
        teacher.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']),
                                           map_location='cpu', weights_only=True), strict=True)
        bank_record = banks.export(root/'targets', teacher, off, refs=refs, device='cuda')
        del off, teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_exact_HLT_and_compact_targets') as load_phase:
        tr = worker.data(gate, 'train', 'D000', authenticated=meta)
        val = worker.data(gate, 'validation', 'D000', authenticated=meta)
        bank = banks.load(gate, tr)
        banks.check_join(bank, tr)
        original = worker.validation_arrays(gate)
        if not np.array_equal(val.identities, original['identities']) or not np.array_equal(val.labels, original['labels']):
            raise ValueError('Original validation population differs')
        del original
    with log.phase('installed_native_predictor_and_zero_aux_KD_parity'):
        with parity_backend('cuda'):
            native = installed_parity(tr, device='cuda')
        modes = [parity(tr.batch(np.arange(min(4, len(tr)))), gate['node'], 'cuda', bf16=b)
                 for b in (False, True)]
        zero = zero_parity(tr, val, bank, gate['node'], 'cuda')
    with log.phase('full_population_maximum_support_stress'):
        support = [stress(tr, gate['node'], 'cuda', backward=True),
                   stress(val, gate['node'], 'cuda', backward=False), off_stress]
        for row in support:
            if row['maximum'] != meta[0]['summaries'][row['role']][row['coordinate']]['maximum']:
                raise ValueError('Stress does not cover authenticated full-population maximum')
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('full_population_technical_epoch_and_native_readback') as fit_phase:
        store = checkpoints.EpochStore(directory/'checkpoints', binding=dict(
            gate=gate['content_hash'], bank=bank.manifest['content_hash'], technical=True))
        worker.fit(directory/'probe', gate, tr, val, bank, device='cuda', technical=True, store=store)
        report = worker.validate_fit(directory/'probe', gate, bank.manifest['content_hash'], technical=True)
        epoch = checkpoints.check(directory/'checkpoints/epoch-01.json', store.binding)
        restored = torch.load(epoch['payload']['path'], map_location='cpu', weights_only=True)
        if restored['history'] != report['validation_history'] or restored['selected'] != report['selected_pass']:
            raise ValueError('Epoch state readback differs')
        del restored
    row = report['validation_history'][0]
    memory = memory_snapshot(torch)
    measured = dict(gpu=runtime['gpu'], environment=runtime['environment'],
        peak_rss_bytes=memory['peak_rss_bytes'], gpu_peak_bytes=memory['gpu_peak_bytes'],
        epoch_checkpoint_bytes=epoch['payload']['bytes'],
        timings=dict(authentication=authentication, load_HLT_targets=load_phase['seconds'],
            train_epoch=row['train_seconds'], validation_epoch=row['validation_seconds'],
            checkpoint_reporting=max(0., fit_phase['seconds']-row['train_seconds']-row['validation_seconds'])))
    minutes = admission(gate, measured)
    write_immutable_json(directory/'measurement.json', measured)
    campaign.check_gate(gate)
    if worker.execution(gate) != job or installed_environment() != runtime['environment'] or gpu_identity() != runtime['gpu']:
        raise ValueError('Environment/allocation changed during preflight')
    outputs = [file_ref(p, root=root) for base in (root/'targets', directory)
               for p in sorted(base.rglob('*')) if p.is_file()]
    receipt = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash'], bank=bank_record['content_hash']),
        bank_hash=bank_record['content_hash'], supervision_hash=bank.manifest['content_hash'],
        job_id=job, **measured, train_minutes=minutes, native_parity=native,
        predictor_parities=modes, zero_aux_native_kd_parity=zero, stress=support,
        full_population_training_measured=True, full_validation_measured=True,
        technical_checks_passed=True, checkpoint_probability_roundtrip=True, outputs=outputs)
    validate_report(gate, receipt)
    write_immutable_json(directory/'receipt.json', receipt)
    print(f'FULLSTAT EMBED measured training={row["train_seconds"]/60:.2f}min/epoch '
          f'validation={row["validation_seconds"]/60:.2f}min; request={minutes}min', flush=True)
    return receipt


def validate_report(gate, report):
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash'], bank=report['bank_hash']))
    root = Path(gate['root'])
    if (gate['recipe'] != recipe() or report['environment'] != gate['anchor']['evidence']['environment']
            or 'L40S' not in report['gpu']['name']
            or not str(report['job_id']).isdigit() or int(report['job_id']) <= 0
            or any(report.get(k) is not True for k in ('technical_checks_passed',
                'checkpoint_probability_roundtrip', 'zero_aux_native_kd_parity',
                'full_population_training_measured', 'full_validation_measured'))
            or report['native_parity']['passed'] is not True
            or [p['bf16'] for p in report['predictor_parities']] != [False, True]
            or any(p.get(k) is not True for p in report['predictor_parities'] for k in
                   ('native_forward', 'zero_control_update', 'auxiliary_backbone_gradient', 'auxiliary_head_gradient'))):
        raise ValueError('Missing native/auxiliary/full-population Oscar evidence')
    required = {'targets/manifest.json', 'targets/targets.npz', 'preflight/runtime.json',
        'preflight/measurement.json', 'preflight/checkpoints/epoch-01.json', 'preflight/checkpoints/epoch-01.pt',
        'preflight/probe/receipt.json', 'preflight/probe/selected.pt',
        'preflight/probe/training_report.json', 'preflight/probe/validation_predictions.npz'}
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
    measurement = load_json(root/'preflight/measurement.json')
    if any(report[k] != v for k, v in measurement.items()) or admission(gate, measurement) != report['train_minutes']:
        raise ValueError('Measured scientific resource request differs')
    m = load_json(root/'targets/manifest.json')
    validate(m, 'BANK', banks.parents(gate['anchor']['evidence']['prepared'],
                                      gate['anchor']['evidence']['models']['OFFLINE']))
    if (m['content_hash'] != report['bank_hash'] or m['rows'] != gate['counts']['train']
            or m['teacher'] != 'OFFLINE' or m['role'] != 'train' or m['coordinate'] != 'OFFLINE'
            or m['width'] != 128 or m['payload']['path'] != 'targets.npz'
            or dict(m['payload'], path='targets/'+m['payload']['path']) not in report['outputs']
            or report['supervision_hash'] != worker.supervision_hash(gate)):
        raise ValueError('Embedding bank or logit teacher binding differs')
    meta = worker.metadata(gate)[0]
    expected = [('train', 'D000', True, False), ('validation', 'D000', False, False),
                ('train', 'OFFLINE', False, True)]
    if len(report['stress']) != len(expected):
        raise ValueError('Missing distinct support stress')
    for row, (role, co, backward, offline) in zip(report['stress'], expected):
        if row != dict(role=role, coordinate=co, backward=backward, fp32_export=offline,
                       maximum=meta['summaries'][role][co]['maximum'], batch=128, steps=3):
            raise ValueError('Full support stress differs')
    saved = worker.validate_fit(root/'preflight/probe', gate, report['supervision_hash'], technical=True)
    row = saved['validation_history'][0]
    if any(measurement['timings'][k] != row[v] for k, v in
           (('train_epoch', 'train_seconds'), ('validation_epoch', 'validation_seconds'))):
        raise ValueError('Epoch timings do not match technical report')
    epoch = checkpoints.check(root/'preflight/checkpoints/epoch-01.json', dict(
        gate=gate['content_hash'], bank=report['supervision_hash'], technical=True))
    if epoch['epoch'] != 1 or epoch['payload']['bytes'] != report['epoch_checkpoint_bytes']:
        raise ValueError('Epoch evidence differs')
