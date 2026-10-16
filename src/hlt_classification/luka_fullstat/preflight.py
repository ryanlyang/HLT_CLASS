"""Real Oscar technical miniature; no scientific fit or test capability."""
from pathlib import Path
import math
import os
import time

import numpy as np

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from .contracts import COORDINATES, ORDINARY, artifact, validate
from . import cache, campaign as c


def projected(measurements, counts, batch):
    train, reduce = [], []
    for m in measurements:
        hist = m['training']['validation_history'][0]
        ratios = {r: math.ceil(counts[r]/batch)/math.ceil(m['counts'][r]/batch) for r in ORDINARY}
        io = m['full_cache_seconds_estimate']
        train.append(math.ceil(2*(io+20*(hist['train_seconds']*ratios['train']+
                                       hist['validation_seconds']*ratios['validation'])+
                                  m['bank_seconds']*ratios['validation'])/60))
        reduce.append(math.ceil(2*(io+m['bank_seconds']*ratios['train'])/60))
    return max(30, max(train)), max(15, max(reduce))


def stress(model, bundle, *, batch, training, device='cuda'):
    import torch
    from hlt_classification.jetclass2_delphes.runner import _forward
    if bundle.role != ('train' if training else 'validation'):
        raise PermissionError('Stress role differs')
    lengths = np.concatenate([np.diff(b.offsets) for b in bundle.blocks])
    raw = bundle.batch(np.full(batch, int(np.argmax(lengths)), np.int64))
    raw = {k: raw[k] for k in ('features', 'vectors', 'mask')}
    model.to(device).train(training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) if training else None
    for _ in range(3):
        with torch.set_grad_enabled(training):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if logits.shape != (batch, 11) or not torch.isfinite(logits).all():
                raise ValueError('Invalid stress output')
            if training:
                optimizer.zero_grad(set_to_none=True)
                # Shape-only training target, no validation labels used.
                torch.nn.functional.cross_entropy(logits.float(), torch.zeros(batch, dtype=torch.long, device=device)).backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError('Nonfinite stress gradient')
                optimizer.step()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return int(lengths.max())


def validate_evidence(value, gate, manifest):
    validate(value, 'PREFLIGHT', dict(gate=gate['content_hash'], cache=manifest['content_hash']))
    if (value['recipe'] != gate['training_recipe'] or value['source_snapshot'] != gate['source_snapshot']
            or value['site'] != gate['execution_site'] or value['counts'] != gate['counts']
            or value['cpus'] != gate['cpus'] or value['memory_mb'] != gate['memory_mb']
            or value['final_test_accessed'] is not False or value['scientific_fit'] is not False
            or value['technical_checks_passed'] is not True or value['checkpoint_probability_roundtrip'] is not True
            or value['resource_envelope_ok'] is not True or value['fits_registered_walltime'] is not True
            or value['full_population_training_measured'] is not False
            or not str(value['job_id']).isdigit() or int(value['job_id']) <= 0
            or 'L40S' not in value['gpu']['name']):
        raise ValueError('Oscar preflight registration/evidence differs')
    from hlt_classification.jetclass2_delphes.model import model_contract
    parity = value['parity']
    kernel_validate(parity, 'WEAVER_PARITY')
    if (parity['passed'] is not True or parity['device'] != 'cuda'
            or parity['forward_and_feature_and_parameter_gradients'] is not True
            or parity['model'] != model_contract()):
        raise ValueError('Installed-Weaver parity differs')
    if (not 0 < value['gpu_peak_bytes'] < .90*value['gpu']['total_memory_bytes']
            or not 0 < value['reserved_host_peak_bytes'] < .85*gate['memory_mb']*1024**2):
        raise ValueError('Measured resource headroom differs')
    expected = ('U000', 'D000', 'OFFLINE', 'U050', 'U100', 'D066', 'D033')
    if tuple(m['coordinate'] for m in value['measurements']) != expected:
        raise ValueError('Seven-coordinate miniature coverage differs')
    for m in value['measurements']:
        if m['counts'] != value['measurements'][0]['counts']:
            raise ValueError('Miniature population must be shared across coordinates')
        for r in ORDINARY:
            if not 0 < m['counts'][r] <= gate['counts'][r] or m['maxima'][r] != manifest['summaries'][r][m['coordinate']]['maximum']:
                raise ValueError('Miniature rows/full-support witness differs')
        report = m['training']
        kernel_validate(report, 'KERNEL_TRAINING_REPORT')
        if (report['passes'] != 1 or report['acceptance_only'] is not True or report['scientific_fit'] is not False
                or report['recipe_sha256'] != gate['training_recipe']['content_hash']
                or report['foundation_sha256'] != manifest['content_hash']
                or report['node']['coordinate'] != m['coordinate'] or report['final_test_accessed'] is not False
                or report['node']['node_id'] != m['coordinate']+'_technical'
                or report['selected_weights_restored'] is not True or len(report['validation_history']) != 1
                or report['validation']['rows'] != m['counts']['validation']
                or m['stress_batch'] != gate['training_recipe']['batch_size'] or m['stress_steps'] != 3):
            raise ValueError('Technical report/stress semantics differ')
        for n in ('bank_seconds', 'full_cache_seconds_estimate'):
            if not math.isfinite(m[n]) or m[n] <= 0:
                raise ValueError('Invalid measured timing')
        for n in ('train_seconds', 'validation_seconds'):
            if not math.isfinite(report['validation_history'][0][n]) or report['validation_history'][0][n] <= 0:
                raise ValueError('Invalid miniature pass timing')
    times = projected(value['measurements'], gate['counts'], gate['training_recipe']['batch_size'])
    if times != (value['train_minutes'], value['reduce_minutes']) or max(times) > gate['max_minutes']:
        raise ValueError('Measured twenty-epoch request exceeds registered ceiling')
    output = Path(gate['root'])/'preflight'
    paths = [ref['path'] for ref in value['outputs']]
    required = {'runtime.json'} | {co+'.pt' for co in expected} | {co+'_bank/manifest.json' for co in expected}
    if len(paths) != len(set(paths)) or not required <= set(paths):
        raise ValueError('Missing preflight payload evidence')
    for ref in value['outputs']:
        validate_file_ref(ref, root=output)
    runtime = load_json(output/'runtime.json')
    if (any(runtime[k] != value[k] for k in ('job_id','gpu','environment'))
            or runtime['cublas_workspace_config'] != ':4096:8'):
        raise ValueError('Preflight runtime evidence differs')
    by_path = {ref['path']: ref for ref in value['outputs']}
    for m in value['measurements']:
        co = m['coordinate']
        bank = load_json(output/(co+'_bank')/'manifest.json')
        kernel_validate(bank, 'PROBABILITY_BANK')
        expected_bank = dict(foundation_sha256=manifest['content_hash'], teacher_report_sha256=m['training']['content_hash'],
            teacher_node=co+'_technical', role='train', temperature=2., rows=m['counts']['train'],
            class_count=11, dtype='float32', final_test_accessed=False)
        if any(bank.get(k) != v for k,v in expected_bank.items()):
            raise ValueError('Technical TRAIN probability bank differs')
        for shard in bank['shards']:
            expected_ref = dict(path=co+'_bank/'+shard['path'], bytes=shard['bytes'], sha256=shard['sha256'])
            if by_path.get(expected_ref['path']) != expected_ref:
                raise ValueError('Unbound technical probability shard')


def run(gate):
    import torch
    from hlt_classification.context_fusion.inputs import new_model
    from hlt_classification.context_fusion.worker import clear, save_state
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
    from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
    from hlt_classification.jetclass2_delphes.model import installed_environment
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
    from hlt_classification.luka_fullsim.preflight_v2 import require_backend, memory_snapshot, PhaseLog
    require_backend()
    c.validate_gate(gate)
    job, cpus, memory = allocation(gate['execution_site'])
    if (cpus, memory) != (gate['cpus'], gate['memory_mb']):
        raise PermissionError('Oscar GPU allocation differs')
    authenticated = c.authenticated_cache(gate)
    manifest = authenticated[0]
    output = Path(gate['root'])/'preflight'
    output.mkdir(exist_ok=False)
    environment, gpu = installed_environment(), gpu_identity()
    write_immutable_json(output/'runtime.json', dict(job_id=job, environment=environment, gpu=gpu,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset')))
    # No CUDA calls in heartbeat/failure cleanup: retain the originating error.
    def host_sample():
        import resource
        return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    log = PhaseLog(output, host_sample)
    with log.phase('authenticate_all_prepared_payloads'):
        for chunk in manifest['chunks']:
            validate_file_ref(chunk['assignment'], root=Path(gate['cache_root']))
            for ref in chunk['views'].values():
                validate_file_ref(ref, root=Path(gate['cache_root']))
    torch.cuda.reset_peak_memory_stats()
    measurements, teacher, parity = [], None, None
    batch = gate['training_recipe']['batch_size']
    future_resident = max(sum(manifest['summaries'][r][co]['resident_bytes'] for r in ORDINARY) for co in COORDINATES)
    if 2*future_resident+2**30 >= .75*memory*1024**2:
        raise MemoryError('Full-population caches exceed reserved host envelope')
    for co in ('U000', 'D000', 'OFFLINE', 'U050', 'U100', 'D066', 'D033'):
        started = time.monotonic()
        with log.phase('load_miniature_'+co):
            train, val = [cache.load(gate['cache_root'], authenticated, role=r, coordinate=co,
                max_ram_bytes=int(memory*1024**2*.25), miniature=True) for r in ORDINARY]
        io = max(time.monotonic()-started, .001)
        full_bytes = sum(manifest['summaries'][r][co]['resident_bytes'] for r in ORDINARY)
        io *= full_bytes/(train.nbytes+val.nbytes)
        node = dict(next(n for n in c.nodes() if n['coordinate'] == co))
        node.update(node_id=co+'_technical', teacher=None if co in ('U000','OFFLINE') else 'U000_technical')
        if parity is None:
            with log.phase('installed_Weaver_FP32_parity'):
                parity = installed_parity(train, device='cuda')
            clear()
        maxima = {}
        for role, bundle in (('train',train), ('validation',val)):
            with log.phase(co+'_'+role+'_full_support_stress'):
                model = new_model(node)
                maxima[role] = stress(model, bundle, batch=batch, training=role=='train')
                del model
                clear()
                snap = memory_snapshot(torch)
                if (snap['gpu_peak_bytes'] >= .9*gpu['total_memory_bytes']
                        or snap['peak_rss_bytes']+2*future_resident >= .85*memory*1024**2):
                    raise MemoryError('Stress plus full cache reservation exceeds headroom')
        kwargs = {}
        if node['teacher'] is not None:
            args = dict(foundation_sha256=manifest['content_hash'], teacher_report_sha256=teacher,
                        teacher_node='U000_technical', role='train')
            kwargs = dict(teacher_probabilities=load_bank(output/'U000_bank', expected_identities=train.identities, **args),
                          teacher_identities=train.identities)
        with log.phase(co+'_technical_one_pass'):
            model = new_model(node)
            report, state = train_kernel(model, train, val, node=node, device='cuda', acceptance_passes=1,
                                        training_recipe=gate['training_recipe'], **kwargs)
        with log.phase(co+'_checkpoint_TRAIN_bank_readback') as phase:
            save_state(output/(co+'.pt'), state)
            saved = torch.load(output/(co+'.pt'), weights_only=True, map_location='cpu')
            if saved.keys() != state.keys() or any(not torch.equal(state[k], saved[k]) for k in state):
                raise ValueError('Checkpoint bytes/readback differ')
            before = predict(model, train, device='cuda', temperature=2., batch_size=batch)
            model.load_state_dict(saved, strict=True)
            probs = predict(model, train, device='cuda', temperature=2., batch_size=batch)
            if not np.array_equal(before, probs):
                raise ValueError('Checkpoint probability replay differs')
            args = dict(foundation_sha256=manifest['content_hash'], teacher_report_sha256=report['content_hash'],
                        teacher_node=node['node_id'], role='train')
            publish_bank(output/(co+'_bank'), identities=train.identities, probabilities=probs, **args)
            if not np.array_equal(probs, load_bank(output/(co+'_bank'), expected_identities=train.identities, **args)):
                raise ValueError('TRAIN bank replay differs')
        if co == 'U000':
            teacher = report['content_hash']
        measurements.append(dict(coordinate=co, training=report, counts=dict(train=len(train), validation=len(val)),
            maxima=maxima, stress_steps=3, stress_batch=batch, bank_seconds=phase['seconds'],
            full_cache_seconds_estimate=io))
        del model, train, val, bundle, before, probs, state, saved, kwargs
        clear()
    c.validate_gate(gate)
    train_minutes, reduce_minutes = projected(measurements, gate['counts'], batch)
    snap = memory_snapshot(torch)
    reserved = snap['peak_rss_bytes']+2*future_resident
    result = artifact('PREFLIGHT', parents=dict(gate=gate['content_hash'], cache=manifest['content_hash']),
        counts=gate['counts'], source_snapshot=gate['source_snapshot'], job_id=job, cpus=cpus, memory_mb=memory,
        site=gate['execution_site'], environment=environment, gpu=gpu, parity=parity, recipe=gate['training_recipe'],
        measurements=measurements, train_minutes=train_minutes, reduce_minutes=reduce_minutes,
        reserved_host_peak_bytes=reserved, gpu_peak_bytes=snap['gpu_peak_bytes'],
        resource_envelope_ok=reserved < .85*memory*1024**2 and snap['gpu_peak_bytes'] < .9*gpu['total_memory_bytes'],
        fits_registered_walltime=max(train_minutes,reduce_minutes) <= gate['max_minutes'],
        technical_checks_passed=True, checkpoint_probability_roundtrip=True, scientific_fit=False,
        final_test_accessed=False, full_population_training_measured=False,
        outputs=[file_ref(p, root=output) for p in sorted(output.rglob('*')) if p.is_file()])
    write_immutable_json(output/'measurement.json', result)
    validate_evidence(result, gate, manifest)
    write_immutable_json(output/'preflight.json', result)
    return result
