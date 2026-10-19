"""Real native-model checks and measured admission; never gate scientific quality."""
from copy import deepcopy
from pathlib import Path
import gc
import math
import re
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from . import campaign, targets, worker, source
from .contracts import artifact, validate, recipe
from .model import new_model, objective


def parity(raw, node, device, *, bf16=False):
    torch.manual_seed(node['initialization_seed'])
    native = DelphesParticleTransformer().to(device)
    augmented = new_model(node).to(device)
    for key, value in native.state_dict().items():
        torch.testing.assert_close(value, augmented.state_dict()[key], rtol=0, atol=0)
    augmented.capture = True
    snapshots = []
    atol, rtol = (5e-4, .01) if bf16 else (2e-6, 2e-5)
    with parity_backend(device):
        for model in (native, augmented):
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            model.train(); torch.manual_seed(917)
            tensors = {k: torch.from_numpy(raw[k]).to(device).clone() for k in ('features', 'vectors', 'mask')}
            tensors['features'].requires_grad_()
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(**tensors)
            labels = torch.from_numpy(raw['labels']).to(device)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            if model is augmented:
                loss, _, _ = objective(logits, labels, model.prediction, torch.zeros_like(model.prediction), coefficient=0.)
            loss.backward()
            gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
            optimizer.step()
            state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('predictor.')}
            snapshots.append((logits.detach(), tensors['features'].grad, gradients, state))
        def compare(a, b):
            if isinstance(a, dict):
                if a.keys() != b.keys():
                    raise ValueError('Zero-control gradient/state topology differs')
                for k in a:
                    compare(a[k], b[k])
            elif isinstance(a, tuple):
                for x, y in zip(a, b):
                    compare(x, y)
            else:
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError('Nonfinite parity evidence')
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        compare(snapshots[0], snapshots[1])
        augmented.zero_grad(set_to_none=True)
        logits = _forward(augmented, raw, device, bf16=bf16)
        # Auxiliary-only gradient witness: CE cannot masquerade as useful wiring.
        loss = torch.nn.functional.smooth_l1_loss(augmented.prediction, torch.ones_like(augmented.prediction))
        loss.backward()
        for prefix in ('mod.', 'predictor.'):
            grads = [p.grad for n, p in augmented.named_parameters() if n.startswith(prefix) and p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
                raise ValueError('Auxiliary-only gradient failed to reach both backbone and head')
    return dict(native_forward=True, zero_control_update=True, auxiliary_backbone_gradient=True,
                auxiliary_head_gradient=True, bf16=bf16, rtol=rtol, atol=atol)


def stress(cache, node, device, *, backward):
    lengths = np.concatenate([np.diff(b.offsets) for b in cache.blocks])
    witness = int(np.argmax(lengths))
    raw = cache.batch(np.repeat(witness, 128))
    model = new_model(node).to(device)
    model.train(backward); model.capture = backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if backward:
                loss, _, _ = objective(logits, torch.from_numpy(raw['labels']).to(device),
                    model.prediction, torch.ones_like(model.prediction))
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite stress gradient')
                optimizer.step()
            elif not torch.isfinite(logits).all():
                raise ValueError('Nonfinite validation stress')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return dict(role=cache.role, coordinate='D000', maximum=int(lengths.max()), batch=128,
                steps=3, backward=backward)


def admission(gate, measurement):
    times = measurement['timings']
    if (set(times) != {'authentication', 'load_HLT', 'train_epoch', 'validation_epoch', 'checkpoint_reporting'}
            or any(not math.isfinite(v) or v < 0 for v in times.values())
            or times['train_epoch'] <= 0 or times['validation_epoch'] <= 0
            or any(type(measurement[k]) is not int or measurement[k] <= 0
                   for k in ('peak_rss_bytes', 'gpu_peak_bytes'))):
        raise ValueError('Invalid measured resource evidence')
    # Setup occurs once in science; teacher export/rebuild is already complete.
    estimate = math.ceil(1.75*(times['authentication'] + times['load_HLT']
        + 30*(times['train_epoch']+times['validation_epoch']) + times['checkpoint_reporting'] + 300)/60)
    minutes = max(30, estimate)
    if (measurement['peak_rss_bytes'] >= .85*gate['memory_mb']*1024**2
            or measurement['gpu_peak_bytes'] >= .9*measurement['gpu']['total_memory_bytes']
            or minutes > gate['science_ceiling_minutes']):
        raise ValueError('Resource admission failed; do not weaken stress or scientific checks')
    return minutes


def run(gate):
    begun = time.monotonic()
    campaign.check_gate(gate)
    authentication = time.monotonic()-begun
    job = worker.execution(gate)
    worker.storage(gate)
    environment, gpu = installed_environment(), gpu_identity()
    if environment != gate['anchor']['evidence']['environment']:
        raise ValueError('Matched parent installed environment differs')
    root = Path(gate['root'])
    directory = root/'preflight'; directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    with log.phase('rebuild_only_OFFLINE_TRAIN_in_RAM_and_export_compact_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        offline = targets.offline_train(gate, authenticated=meta)
        teacher_ref = gate['anchor']['evidence']['models']['OFFLINE']['checkpoint']
        validate_file_ref(teacher_ref)
        teacher = DelphesParticleTransformer()
        teacher.load_state_dict(torch.load(teacher_ref['path'], map_location='cpu', weights_only=True), strict=True)
        bank_record = targets.export(root/'targets', teacher, offline, teacher_ref=teacher_ref, device='cuda')
        del offline, teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_HLT_and_join_targets') as load_phase:
        train = worker.data(gate, 'train', 'D000', meta)
        val = worker.data(gate, 'validation', 'D000', meta)
        bank = targets.Bank(root/'targets', train, teacher_ref=teacher_ref)
        worker.snapshot_references(gate, val, meta)
    with log.phase('installed_Weaver_and_predictor_parity'):
        with parity_backend('cuda'):
            native = installed_parity(train, device='cuda')
        raw = train.batch(np.arange(4))
        parities = [parity(raw, gate['node'], 'cuda', bf16=v) for v in (False, True)]
    with log.phase('full_support_stress'):
        stresses = [stress(train, gate['node'], 'cuda', backward=True),
                    stress(val, gate['node'], 'cuda', backward=False)]
        for row in stresses:
            if row['maximum'] != meta[1]['summaries'][row['role']]['D000']['maximum']:
                raise ValueError('Stress differs from prepared support maximum')
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('complete_population_technical_epoch_and_native_readback') as fit_phase:
        worker.fit(directory/'probe', gate, train, val, bank, device='cuda', technical=True)
        report, _ = worker.validate_fit(directory/'probe', gate, bank_record['content_hash'], technical=True)
    row = report['validation_history'][0]
    memory = memory_snapshot(torch)
    measurement = dict(gpu=gpu, environment=environment, peak_rss_bytes=memory['peak_rss_bytes'],
        gpu_peak_bytes=memory['gpu_peak_bytes'], timings=dict(authentication=authentication,
        load_HLT=load_phase['seconds'], train_epoch=row['train_seconds'], validation_epoch=row['validation_seconds'],
        checkpoint_reporting=max(0., fit_phase['seconds']-row['train_seconds']-row['validation_seconds'])))
    write_immutable_json(directory/'measurement.json', measurement)
    minutes = admission(gate, measurement)
    worker.storage(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during preflight')
    outputs = [file_ref(p, root=root) for base in (root/'targets', directory)
               for p in sorted(base.rglob('*')) if p.is_file()]
    receipt = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash'], bank=bank_record['content_hash']),
        bank_hash=bank_record['content_hash'], job_id=job, **measurement, train_minutes=minutes,
        native_parity=native, predictor_parities=parities, stress=stresses,
        technical_checks_passed=True, checkpoint_probability_roundtrip=True, outputs=outputs)
    write_immutable_json(directory/'receipt.json', receipt)
    validate_report(gate, receipt)
    return receipt


def validate_report(gate, report):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'], anchor=gate['anchor']['content_hash']))
    if gate['recipe'] != recipe():
        raise ValueError('Registered recipe differs')
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash'], bank=report['bank_hash']))
    root = Path(gate['root'])
    names = [r['path'] for r in report['outputs']]
    required = {'targets/manifest.json', 'targets/targets.npz', 'preflight/measurement.json',
        'preflight/references/M0HLT.npz', 'preflight/references/OFFLINE.npz',
        'preflight/references/original50k.npz', 'preflight/probe/receipt.json',
        'preflight/probe/selected.pt', 'preflight/probe/training_report.json',
        'preflight/probe/validation_predictions.npz'}
    if len(names) != len(set(names)) or not required <= set(names) or any(
            not p.startswith(('targets/', 'preflight/')) for p in names):
        raise ValueError('Preflight evidence coverage differs')
    for ref in report['outputs']:
        validate_file_ref(ref, root=root)
    bank = load_json(root/'targets/manifest.json')
    validate(bank, 'BANK', dict(recipe=recipe()['content_hash'], prepared=gate['anchor']['evidence']['prepared'],
        teacher=gate['anchor']['evidence']['models']['OFFLINE']['checkpoint']['sha256']))
    if bank['content_hash'] != report['bank_hash'] or report['environment'] != gate['anchor']['evidence']['environment']:
        raise ValueError('Measured environment/targets differ')
    if not re.fullmatch('[1-9][0-9]*', report['job_id']):
        raise ValueError('Missing real allocation job identity')
    measurement = load_json(root/'preflight/measurement.json')
    if any(report[k] != measurement[k] for k in measurement):
        raise ValueError('Measured resource evidence differs')
    if (report['train_minutes'] != admission(gate, measurement) or report['technical_checks_passed'] is not True
            or report['checkpoint_probability_roundtrip'] is not True or report['native_parity']['passed'] is not True
            or [p['bf16'] for p in report['predictor_parities']] != [False, True]
            or any(p.get(k) is not True for p in report['predictor_parities'] for k in
                   ('native_forward', 'zero_control_update', 'auxiliary_backbone_gradient', 'auxiliary_head_gradient'))
            or [(s['role'], s['coordinate'], s['batch'], s['steps'], s['backward']) for s in report['stress']] !=
               [('train', 'D000', 128, 3, True), ('validation', 'D000', 128, 3, False)]):
        raise ValueError('Technical acceptance evidence differs')
    worker.validate_fit(root/'preflight/probe', gate, report['bank_hash'], technical=True)
