"""One shared full-population gate, testing each new coefficient explicitly."""
from pathlib import Path
import gc
import re
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_embedding_aux.preflight import parity, admission
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux.contracts import validate as bank_validate, recipe as bank_recipe
from . import campaign, worker, source
from .training import objective
from .contracts import artifact, validate, WEIGHTS, arm_gate


def stress(cache, node, device, *, backward):
    lengths = np.concatenate([np.diff(b.offsets) for b in cache.blocks])
    raw = cache.batch(np.repeat(int(np.argmax(lengths)), 128))
    model = new_model(node).to(device)
    model.train(backward); model.capture = backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if backward:
                loss, _, _ = objective(logits, torch.from_numpy(raw['labels']).to(device),
                    model.prediction, torch.ones_like(model.prediction), coefficient=node['auxiliary_weight'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite stress loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite stress gradient')
                optimizer.step()
            elif not torch.isfinite(logits).all():
                raise ValueError('Nonfinite validation stress')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return dict(role=cache.role, coordinate='D000', maximum=int(lengths.max()), batch=128,
                steps=3, backward=backward, weight=node['auxiliary_weight'])


def run(gate):
    begun = time.monotonic()
    campaign.check_gate(gate)
    authentication = time.monotonic()-begun
    job = worker.execution(gate)
    worker.storage(gate)
    environment, gpu = installed_environment(), gpu_identity()
    if environment != gate['anchor']['evidence']['environment']:
        raise ValueError('Matched installed environment differs')
    root = Path(gate['root'])
    directory = root/'preflight'; directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    with log.phase('load_HLT_and_existing_TRAIN_bank_once') as loaded:
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train = worker.data(gate, 'train', 'D000', meta)
        val = worker.data(gate, 'validation', 'D000', meta)
        bank = worker.load_bank(gate, train)
        write_immutable_json(directory/'target_manifest.json', bank.manifest)
        support = artifact('INPUTS', parents=dict(prepared=gate['anchor']['evidence']['prepared']),
            counts=gate['counts'], coordinate='D000',
            maxima={role: meta[1]['summaries'][role]['D000']['maximum'] for role in ('train', 'validation')})
        write_immutable_json(directory/'support.json', support)
        worker.snapshot_references(gate, val, meta)
    with log.phase('installed_Weaver_and_zero_weight_parity'):
        with parity_backend('cuda'):
            native = installed_parity(train, device='cuda')
        parities = [parity(train.batch(np.arange(4)), gate['nodes']['w3'], 'cuda', bf16=v)
                    for v in (False, True)]
    measurements, stresses, requests = {}, {}, {}
    for arm in WEIGHTS:
        scoped = arm_gate(gate, arm)
        with log.phase(arm+'_longest_support_stress'):
            stresses[arm] = [stress(train, scoped['node'], 'cuda', backward=True),
                             stress(val, scoped['node'], 'cuda', backward=False)]
            for row in stresses[arm]:
                if row['maximum'] != meta[1]['summaries'][row['role']]['D000']['maximum']:
                    raise ValueError('Prepared support maximum differs')
            gc.collect(); torch.cuda.empty_cache()
        with log.phase(arm+'_full_technical_epoch_native_readback') as fitted:
            worker.fit(directory/arm, scoped, train, val, bank, device='cuda', technical=True)
            report, _ = worker.validate_fit(directory/arm, scoped, bank.manifest['content_hash'], technical=True)
        row = report['validation_history'][0]
        memory = memory_snapshot(torch)
        measured = dict(gpu=gpu, environment=environment, peak_rss_bytes=memory['peak_rss_bytes'],
            gpu_peak_bytes=memory['gpu_peak_bytes'], timings=dict(authentication=authentication,
            load_HLT=loaded['seconds'], train_epoch=row['train_seconds'], validation_epoch=row['validation_seconds'],
            checkpoint_reporting=max(0., fitted['seconds']-row['train_seconds']-row['validation_seconds'])))
        measurements[arm] = measured
        write_immutable_json(directory/(arm+'-measurement.json'), measured)
        requests[arm] = admission(gate, measured)
    worker.storage(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during preflight')
    receipt = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash'], bank=bank.manifest['content_hash']),
        bank_hash=bank.manifest['content_hash'], job_id=job, gpu=gpu, environment=environment,
        train_minutes=requests, measurements=measurements, stress=stresses, native_parity=native,
        predictor_parities=parities, technical_checks_passed=True, checkpoint_probability_roundtrip=True,
        outputs=[file_ref(p, root=root) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'receipt.json', receipt)
    validate_report(gate, receipt)
    return receipt


def validate_report(gate, report):
    campaign.validate_gate_record(gate)
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash'], bank=report['bank_hash']))
    root = Path(gate['root'])
    names = [r['path'] for r in report['outputs']]
    required = {'preflight/target_manifest.json', 'preflight/support.json'} | {
        'preflight/references/'+n for n in ('M0HLT.npz', 'OFFLINE.npz', 'EMBED_0.3.npz', 'original50k.npz')}
    for arm in WEIGHTS:
        required.add('preflight/'+arm+'-measurement.json')
        required.update('preflight/'+arm+'/'+n for n in
                        ('receipt.json', 'selected.pt', 'training_report.json', 'validation_predictions.npz'))
    if (len(names) != len(set(names)) or not required <= set(names)
            or any(not n.startswith('preflight/') for n in names)):
        raise ValueError('Preflight evidence coverage differs')
    for ref in report['outputs']:
        validate_file_ref(ref, root=root)
    support = load_json(root/'preflight/support.json')
    validate(support, 'INPUTS', dict(prepared=gate['anchor']['evidence']['prepared']))
    if support['counts'] != gate['counts'] or support['coordinate'] != 'D000':
        raise ValueError('Prepared support population differs')
    bank = load_json(root/'preflight/target_manifest.json')
    bank_validate(bank, 'BANK', dict(recipe=bank_recipe()['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'],
        teacher=gate['anchor']['evidence']['models']['OFFLINE']['checkpoint']['sha256']))
    if (bank['content_hash'] != report['bank_hash']
            or bank['teacher'] != gate['anchor']['evidence']['models']['OFFLINE']['checkpoint']
            or bank['rows'] != gate['counts']['train'] or bank['role'] != 'train'
            or bank['coordinate'] != 'OFFLINE' or bank['width'] != 128
            or bank['payload']['sha256'] != gate['embedding']['payload']['sha256']
            or bank['payload']['path'] != 'targets.npz'
            or report['environment'] != gate['anchor']['evidence']['environment']):
        raise ValueError('Measured target/environment differs')
    if (not re.fullmatch('[1-9][0-9]*', report['job_id'])
            or report['technical_checks_passed'] is not True
            or report['checkpoint_probability_roundtrip'] is not True
            or report['native_parity']['passed'] is not True
            or [p['bf16'] for p in report['predictor_parities']] != [False, True]
            or any(p.get(k) is not True for p in report['predictor_parities'] for k in
                   ('native_forward', 'zero_control_update', 'auxiliary_backbone_gradient', 'auxiliary_head_gradient'))
            or any(set(report[k]) != set(WEIGHTS) for k in ('measurements', 'stress', 'train_minutes'))):
        raise ValueError('Technical acceptance differs')
    for arm, weight in WEIGHTS.items():
        measured = load_json(root/('preflight/'+arm+'-measurement.json'))
        if (report['measurements'][arm] != measured or measured['gpu'] != report['gpu']
                or measured['environment'] != report['environment']
                or report['train_minutes'][arm] != admission(gate, measured)
                or [(r['role'], r['coordinate'], r['batch'], r['steps'], r['backward'], r['weight'])
                    for r in report['stress'][arm]] !=
                    [('train', 'D000', 128, 3, True, weight), ('validation', 'D000', 128, 3, False, weight)]
                or any(type(r['maximum']) is not int or r['maximum'] < 1
                       or r['maximum'] != support['maxima'][r['role']] for r in report['stress'][arm])):
            raise ValueError('Weight-specific resource/stress evidence differs')
        worker.validate_fit(root/'preflight'/arm, arm_gate(gate, arm), report['bank_hash'], technical=True)
