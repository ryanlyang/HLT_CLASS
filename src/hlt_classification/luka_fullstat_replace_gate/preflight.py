"""Real full-population three-path rehearsal; no scientific-score admission."""
from pathlib import Path
import gc
import math
import os
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_gating.preflight import parity, stress
from hlt_classification.luka_gating import banks as shared
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from . import campaign, worker, banks, checkpoints
from .contracts import artifact, validate, recipe, NAMES


def budgets(gate, measurement):
    fits = measurement['fits']
    if set(fits) != set(NAMES):
        raise ValueError('Missing full-population technical fits')
    for row in fits.values():
        for key in ('train_seconds', 'validation_seconds', 'other_seconds'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Invalid measured fit timing')
        if row['train_seconds'] <= 0 or row['validation_seconds'] <= 0 or row['checkpoint_bytes'] <= 0:
            raise ValueError('Incomplete full epoch measurement')
    for key in ('authentication_seconds', 'load_seconds', 'export_seconds'):
        if not math.isfinite(measurement[key]) or measurement[key] < 0:
            raise ValueError('Invalid overhead/export timing')
    minutes = {}
    overhead = (3*measurement['authentication_seconds'] + measurement['load_seconds']
                + measurement['export_seconds'] + 900)
    for name, row in fits.items():
        minutes['train_'+name] = max(120, math.ceil(2*(overhead +
            20*(row['train_seconds']+row['validation_seconds']+row['other_seconds']))/60))
    minutes['export_PRIV_D000'] = max(60, math.ceil(2*(overhead+measurement['export_seconds'])/60))
    minutes['aggregate'] = max(60, math.ceil(4*overhead/60))
    additional = {'train_'+name: int(22*row['checkpoint_bytes']+2*2**30) for name, row in fits.items()}
    additional['export_PRIV_D000'] = int(measurement['bank_bytes']+2**30)
    additional['aggregate'] = 2**30
    total = sum(additional.values()) + measurement['preflight_bytes']
    if (any(type(v) is not int or not 1 <= v <= gate['max_minutes'] for v in minutes.values())
            or type(measurement['peak_rss_bytes']) is not int or measurement['peak_rss_bytes'] <= 0
            or type(measurement['gpu_peak_bytes']) is not int or measurement['gpu_peak_bytes'] <= 0
            or measurement['peak_rss_bytes']+8*2**30 >= .85*gate['memory_mb']*1024**2
            or measurement['gpu_peak_bytes'] >= .9*measurement['gpu']['total_memory_bytes']
            or measurement['bank_bytes'] <= 0 or measurement['preflight_bytes'] < measurement['bank_bytes']
            or total > gate['available_result_bytes']):
        raise ValueError('Measured time/RAM/GPU/storage exceeds envelope; science blocked')
    return minutes, additional


def run(gate):
    begun = time.monotonic()
    campaign.check_gate(gate)
    authentication = time.monotonic()-begun
    job = worker.execution(gate)
    campaign.storage(gate)
    directory = Path(gate['root'])/'preflight'
    directory.mkdir(exist_ok=False)
    runtime = dict(job_id=job, environment=installed_environment(), gpu=gpu_identity(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset'))
    write_immutable_json(directory/'runtime.json', runtime)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    with log.phase('load_full_D000_D033_and_original_TRAIN_logits') as load:
        meta = worker.metadata(gate)
        tr, val, ctx, vctx = [worker.data(gate, role, co, authenticated=meta) for role, co in
            [('train', 'D000'), ('validation', 'D000'), ('train', 'D033'), ('validation', 'D033')]]
        shared.paired(tr, ctx)
        shared.paired(val, vctx)
        bank = banks.original(gate, tr)
        original = worker.validation_arrays(gate)
        if not np.array_equal(val.identities, original['identities']) or not np.array_equal(val.labels, original['labels']):
            raise ValueError('Original validation population differs')
    with log.phase('native_and_gate_FP32_BF16_parity'):
        with parity_backend('cuda'):
            native = installed_parity(tr, device='cuda')
        modes = [parity(tr.batch(np.arange(min(4, len(tr)))), 'cuda', bf16=b) for b in (False, True)]
    support, fits, ids = [], {}, []
    for node in gate['nodes']:
        privileged = node['node_id'] == 'PRIV_D000'
        c, vc = (ctx, vctx) if privileged else (None, None)
        with log.phase(node['node_id']+'_max_support_TRAIN_and_validation'):
            support += [stress(tr, c, node, 'cuda', backward=True),
                        stress(val, vc, node, 'cuda', backward=False)]
            gc.collect()
            torch.cuda.empty_cache()
        store = checkpoints.EpochStore(directory/(node['node_id']+'_epochs'), binding=dict(
            gate=gate['content_hash'], bank=bank.manifest['content_hash'], node=node['node_id'], technical=True))
        with log.phase(node['node_id']+'_full_population_epoch_and_readback') as phase:
            worker.fit(directory/node['node_id'], gate, node, tr, val, c, vc, bank,
                       device='cuda', technical=True, store=store)
            report = worker.validate_fit(directory/node['node_id'], gate, node,
                                         bank.manifest['content_hash'], technical=True)
            write_immutable_json(directory/node['node_id']/'supervision.json', bank.manifest)
            epoch = checkpoints.check(store.root/'epoch-01.json', store.binding)
            state = torch.load(epoch['payload']['path'], map_location='cpu', weights_only=True)
            if state['history'] != report['validation_history'] or state['selected'] != report['selected_pass']:
                raise ValueError('Epoch state readback differs')
            del state
        row = report['validation_history'][0]
        fits[node['node_id']] = dict(train_seconds=row['train_seconds'], validation_seconds=row['validation_seconds'],
            other_seconds=max(0., phase['seconds']-row['train_seconds']-row['validation_seconds']),
            checkpoint_bytes=epoch['payload']['bytes'])
        ids.append(report['initialization_sha256'])
        if privileged:
            with log.phase('technical_PRIV_TRAIN_export_and_resident_readback') as export:
                banks.export(directory/'targets', gate, tr, ctx, directory/'PRIV_D000',
                             technical=True, device='cuda')
                bank = banks.Resident(directory/'targets', gate, tr, directory/'PRIV_D000', technical=True)
            # No richer view is retained or supplied to replacement execution.
            del ctx, vctx, c, vc
            ctx = vctx = None
        gc.collect()
        torch.cuda.empty_cache()
    if len(set(ids)) != 1:
        raise ValueError('Fresh paired initial states differ')
    memory = memory_snapshot(torch)
    measurement = dict(gpu=runtime['gpu'], environment=runtime['environment'],
        peak_rss_bytes=memory['peak_rss_bytes'], gpu_peak_bytes=memory['gpu_peak_bytes'],
        authentication_seconds=authentication, load_seconds=load['seconds'],
        export_seconds=export['seconds'], fits=fits,
        bank_bytes=sum(p.stat().st_size for p in (directory/'targets').rglob('*') if p.is_file()),
        preflight_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()))
    minutes, additional = budgets(gate, measurement)
    write_immutable_json(directory/'measurement.json', measurement)
    campaign.check_gate(gate)
    if worker.execution(gate) != job or gpu_identity() != runtime['gpu'] or installed_environment() != runtime['environment']:
        raise ValueError('Preflight execution changed')
    receipt = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash']), job_id=job, **measurement,
        minutes=minutes, additional_bytes=additional, native_parity=native, gate_parities=modes, stress=support,
        fresh_initializations_equal=True, technical_checks_passed=True,
        checkpoint_probability_roundtrip=True, full_population_training_measured=True, full_validation_measured=True,
        outputs=[file_ref(p, root=directory) for p in sorted(directory.rglob('*')) if p.is_file()])
    validate_report(gate, receipt)
    write_immutable_json(directory/'receipt.json', receipt)
    print('FULLSTAT GATE measured requests (minutes):', minutes, flush=True)
    return receipt


def validate_report(gate, report):
    validate(report, 'PREFLIGHT', dict(gate=gate['content_hash']))
    root = Path(gate['root'])/'preflight'
    checks = ('fresh_initializations_equal', 'technical_checks_passed', 'checkpoint_probability_roundtrip',
              'full_population_training_measured', 'full_validation_measured')
    parity_checks = ('identity_logits', 'identity_gradients', 'identity_updates', 'gate_gradient',
        'context_sensitivity', 'endpoint_context_rejected', 'native_attention_surface',
        'multiplicative_equivalence', 'auxiliary_vector_gradient_topology')
    if (gate['recipe'] != recipe() or report['environment'] != gate['anchor']['evidence']['environment']
            or 'L40S' not in report['gpu']['name'] or not str(report['job_id']).isdigit()
            or int(report['job_id']) <= 0 or any(report.get(k) is not True for k in checks)
            or report['native_parity']['passed'] is not True
            or [p['bf16'] for p in report['gate_parities']] != [False, True]
            or any(p.get(k) is not True for p in report['gate_parities'] for k in parity_checks)):
        raise ValueError('Missing real native/gate/full-population evidence')
    names = [r['path'] for r in report['outputs']]
    required = {'runtime.json', 'measurement.json', 'targets/manifest.json',
                'targets/payload/manifest.json', 'targets/payload/population.npz'}
    for name in NAMES:
        required |= {name+'/'+f for f in ('selected.pt', 'receipt.json', 'training_report.json',
                     'validation_predictions.npz', 'supervision.json')}
        required |= {name+'_epochs/epoch-01'+suffix for suffix in ('.pt', '.json')}
    if len(names) != len(set(names)) or not required <= set(names):
        raise ValueError('Missing preflight payload evidence')
    for ref in report['outputs']:
        validate_file_ref(ref, root=root)
    runtime, measured = load_json(root/'runtime.json'), load_json(root/'measurement.json')
    if (any(runtime[k] != report[k] for k in ('job_id', 'environment', 'gpu'))
            or runtime['cublas_workspace_config'] != ':4096:8'
            or runtime['cudnn_v8_disabled'] != gate['anchor']['evidence']['cudnn_v8_disabled']
            or any(report[k] != v for k, v in measured.items())
            or budgets(gate, measured) != (report['minutes'], report['additional_bytes'])):
        raise ValueError('Runtime/resource evidence differs')
    m = banks.check_manifest(root/'targets', gate, root/'PRIV_D000', technical=True)
    payload = load_json(root/'targets/payload/manifest.json')
    if not {'targets/payload/'+s['payload']['path'] for s in payload['shards']} <= set(names):
        raise ValueError('Missing factor output digest')
    ids = []
    meta = worker.metadata(gate)[0]
    expected = [(n['node_id'], role, backward) for n in gate['nodes']
                for role, backward in [('train', True), ('validation', False)]]
    if len(report['stress']) != len(expected):
        raise ValueError('Incomplete role-specific stress coverage')
    for stress_row, (name, role, backward) in zip(report['stress'], expected):
        for k, v in dict(node_id=name, role=role, coordinate='D000', batch=128, steps=3,
                backward=backward, mode='joint' if backward else 'inference',
                frozen_backbone_unchanged=None, maximum=meta['summaries'][role]['D000']['maximum']).items():
            if stress_row.get(k) != v:
                raise ValueError('Full population support differs')
    for node in gate['nodes']:
        name = node['node_id']
        supervision = load_json(root/name/'supervision.json')
        if name == 'PRIV_D000':
            validate(supervision, 'SUPERVISION', dict(logits=load_json(validate_file_ref(gate['anchor']['logits']))['content_hash']))
        elif supervision != m:
            raise ValueError('Wrong technical selected teacher')
        saved = worker.validate_fit(root/name, gate, node, supervision['content_hash'], technical=True)
        ids.append(saved['initialization_sha256'])
        row = saved['validation_history'][0]
        if any(measured['fits'][name][k] != row[k] for k in ('train_seconds', 'validation_seconds')):
            raise ValueError('Epoch timings differ')
        epoch = checkpoints.check(root/(name+'_epochs/epoch-01.json'), dict(
            gate=gate['content_hash'], bank=supervision['content_hash'], node=name, technical=True))
        if epoch['epoch'] != 1 or epoch['payload']['bytes'] != measured['fits'][name]['checkpoint_bytes']:
            raise ValueError('Epoch measurement differs')
    if len(set(ids)) != 1:
        raise ValueError('Paired fresh initialization evidence differs')
