"""Real target export, teacher readback, full-population epoch and admission."""
from pathlib import Path
import gc
import re
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment, DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_embedding_aux.preflight import parity, admission
from hlt_classification.luka_embedding_sweep.preflight import stress
from .contracts import artifact, validate, recipe
from . import campaign, worker, source, targets


def teacher_readback(teacher, val, record, device):
    if val.role != 'validation' or val.coordinate_name != 'D066':
        raise PermissionError('Teacher readback requires its original D066 validation view')
    ref = record['files']['predictions']; validate_file_ref(ref)
    saved = load_npz_arrays(ref['path']); worker.validate_predictions(saved, len(val))
    if not np.array_equal(saved['identities'], val.identities) or not np.array_equal(saved['labels'], val.labels):
        raise ValueError('Teacher validation identities differ')
    actual = predict(teacher, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, saved['probabilities'], rtol=1e-5, atol=1e-6)
    validate_file_ref(ref)
    return artifact('TEACHER_READBACK', parents=dict(teacher=record['content_hash']),
        rows=len(val), coordinate='D066', role='validation', normalization_used=False,
        native_probability_parity=True, rtol=1e-5, atol=1e-6,
        maximum_absolute_error=float(np.max(np.abs(actual-saved['probabilities']))))


def run(gate):
    started = time.monotonic(); campaign.check_gate(gate)
    authentication = time.monotonic()-started
    job = worker.execution(gate); worker.storage(gate)
    environment, gpu = installed_environment(), gpu_identity()
    if environment != gate['anchor']['evidence']['environment']:
        raise ValueError('Matched installed environment differs')
    root = Path(gate['root']); folder = root/'preflight'; folder.mkdir(exist_ok=False)
    log = PhaseLog(folder, lambda: memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
    with log.phase('native_LOGIT_D066_teacher_checkpoint_readback'):
        ref = gate['teacher']['files']['checkpoint']; validate_file_ref(ref)
        teacher = DelphesParticleTransformer().to('cuda').eval()
        teacher.load_state_dict(torch.load(ref['path'], map_location='cpu', weights_only=True), strict=True)
        val_teacher = worker.data(gate, 'validation', 'D066', meta)
        proof = teacher_readback(teacher, val_teacher, gate['teacher'], 'cuda')
        write_immutable_json(folder/'teacher_readback.json', proof)
        del val_teacher
    with log.phase('export_frozen_D066_TRAIN_targets_once'):
        train_teacher = worker.data(gate, 'train', 'D066', meta)
        bank_record = targets.export(root/'targets', teacher, train_teacher,
                                     teacher_record=gate['teacher'], device='cuda')
        del train_teacher, teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_HLT_and_D066_bank') as loaded:
        train, val = (worker.data(gate, role, 'D000', meta) for role in ('train','validation'))
        bank = targets.Bank(root/'targets', train, teacher_record=gate['teacher'])
        worker.snapshot_references(gate, val, meta)
        support = {role: meta[1]['summaries'][role]['D000']['maximum'] for role in ('train','validation')}
        write_immutable_json(folder/'support.json', artifact('SUPPORT', counts=gate['counts'], maxima=support))
    with log.phase('installed_native_and_zero_auxiliary_parity'):
        with parity_backend('cuda'): native = installed_parity(train, device='cuda')
        parities = [parity(train.batch(np.arange(4)), gate['node'], 'cuda', bf16=b) for b in (False,True)]
    with log.phase('maximum_support_weight5_stress'):
        stresses = [stress(train, gate['node'], 'cuda', backward=True),
                    stress(val, gate['node'], 'cuda', backward=False)]
        if any(s['maximum'] != support[s['role']] for s in stresses):
            raise ValueError('Stress support differs from prepared data')
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('full_population_technical_epoch_and_native_readback') as fitted:
        worker.fit(folder/'probe', gate, train, val, bank, device='cuda', technical=True)
        report, _ = worker.validate_fit(folder/'probe', gate, bank_record['content_hash'], technical=True)
    row = report['validation_history'][0]; memory = memory_snapshot(torch)
    measurement = dict(environment=environment, gpu=gpu, peak_rss_bytes=memory['peak_rss_bytes'],
        gpu_peak_bytes=memory['gpu_peak_bytes'], timings=dict(authentication=authentication,
        load_HLT=loaded['seconds'], train_epoch=row['train_seconds'], validation_epoch=row['validation_seconds'],
        checkpoint_reporting=max(0., fitted['seconds']-row['train_seconds']-row['validation_seconds'])))
    write_immutable_json(folder/'measurement.json', measurement)
    minutes = admission(gate, measurement); worker.storage(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during preflight')
    receipt = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash'], bank=bank_record['content_hash']),
        bank_hash=bank_record['content_hash'], job_id=job, **measurement, train_minutes=minutes,
        native_parity=native, predictor_parities=parities, stress=stresses,
        technical_checks_passed=True, checkpoint_probability_roundtrip=True,
        outputs=[file_ref(p, root=root) for base in (root/'targets', folder)
                 for p in sorted(base.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json', receipt); validate_report(gate, receipt)
    return receipt


def validate_report(gate, report):
    campaign.validate_gate_record(gate)
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash'], bank=report['bank_hash']))
    root = Path(gate['root']); names = [r['path'] for r in report['outputs']]
    required = {'targets/manifest.json', 'targets/targets.npz', 'preflight/measurement.json',
                'preflight/teacher_readback.json', 'preflight/support.json'}
    required |= {'preflight/references/'+p+'.npz' for p in
                 ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','LOGIT_D066','original50k')}
    required |= {'preflight/probe/'+p for p in
                 ('receipt.json','selected.pt','training_report.json','validation_predictions.npz')}
    if len(names) != len(set(names)) or not required <= set(names) or any(
            not p.startswith(('targets/','preflight/')) for p in names):
        raise ValueError('Preflight evidence coverage differs')
    for ref in report['outputs']: validate_file_ref(ref, root=root)
    bank = load_json(root/'targets/manifest.json')
    validate(bank, 'BANK', dict(recipe=recipe()['content_hash'], prepared=gate['anchor']['evidence']['prepared'],
                              teacher=gate['teacher']['content_hash']))
    if (bank['content_hash'] != report['bank_hash'] or bank['coordinate'] != 'D066'
            or bank['role'] != 'train' or bank['rows'] != gate['counts']['train'] or bank['width'] != 128
            or bank['teacher'] != gate['teacher']['files']['checkpoint'] or bank['payload']['path'] != 'targets.npz'):
        raise ValueError('D066 target manifest differs')
    proof = load_json(root/'preflight/teacher_readback.json')
    validate(proof, 'TEACHER_READBACK', dict(teacher=gate['teacher']['content_hash']))
    if (proof['rows'] != gate['counts']['validation'] or proof['coordinate'] != 'D066'
            or proof['role'] != 'validation' or proof['normalization_used'] is not False
            or proof['native_probability_parity'] is not True
            or (proof['rtol'],proof['atol']) != (1e-5,1e-6)
            or not np.isfinite(proof['maximum_absolute_error']) or proof['maximum_absolute_error'] < 0
            or proof['maximum_absolute_error'] > 1.1e-5):
        raise ValueError('Teacher native checkpoint readback differs')
    support = load_json(root/'preflight/support.json'); validate(support, 'SUPPORT')
    if support['counts'] != gate['counts'] or set(support['maxima']) != {'train','validation'}:
        raise ValueError('Support population differs')
    measured = load_json(root/'preflight/measurement.json')
    if (any(report[k] != measured[k] for k in measured)
            or report['environment'] != gate['anchor']['evidence']['environment']
            or report['train_minutes'] != admission(gate, measured)
            or not re.fullmatch('[1-9][0-9]*', report['job_id'])
            or report['technical_checks_passed'] is not True or report['checkpoint_probability_roundtrip'] is not True
            or report['native_parity']['passed'] is not True
            or [p['bf16'] for p in report['predictor_parities']] != [False,True]
            or any(p.get(k) is not True for p in report['predictor_parities'] for k in
                   ('native_forward','zero_control_update','auxiliary_backbone_gradient','auxiliary_head_gradient'))
            or [(s['role'],s['coordinate'],s['batch'],s['steps'],s['backward'],s['weight']) for s in report['stress']] !=
               [('train','D000',128,3,True,5.),('validation','D000',128,3,False,5.)]
            or any(type(s['maximum']) is not int or s['maximum'] < 1 or
                   s['maximum'] != support['maxima'][s['role']] for s in report['stress'])):
        raise ValueError('Technical acceptance/resource evidence differs')
    worker.validate_fit(root/'preflight/probe', gate, report['bank_hash'], technical=True)
