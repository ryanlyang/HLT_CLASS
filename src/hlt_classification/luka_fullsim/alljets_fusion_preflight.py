"""New measured paired-model admission on unchanged full-population disk views."""
from pathlib import Path
import math
import os
import re

import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.context_fusion.inputs import PairedCache, new_model
from hlt_classification.context_fusion.worker import clear, save_state, stress, largest_raw
from hlt_classification.jetclass2_delphes.runner import train_kernel, predict, _forward
from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
from hlt_classification.jetclass2_delphes.model import installed_environment, model_contract
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import (
    native_mask_parity, native_offload_parity, validate_offload_stats,
    PARITY_CHECKS, PARITY_BACKEND, PAIR_OFFLOAD_POLICY, PARITY_TOLERANCES)
from hlt_classification.cms2jc2_response.measurement import Measurement
from .preflight_v2 import PhaseLog, memory_snapshot, check_headroom
from . import alljets_fusion_campaign as c, alljets_fusion_worker as w, regular_cache


def probe_nodes():
    chosen = {'FUSION_U050', 'FUSION_D000_D000', 'FINAL_DIRECT_D000'}
    return [{**n, 'node_id':n['node_id']+'_probe', 'teacher':'U000'} for n in c.nodes() if n['node_id'] in chosen]


def support(prepared):
    maxima = {r:{co:v['maximum'] for co,v in rows.items()} for r,rows in prepared['summaries'].items()}
    if set(maxima) != {'train', 'validation'}:
        raise ValueError('Wrong support roles')
    for role, rows in maxima.items():
        if set(rows) != set(regular_cache.s.COORDINATES) or any(type(n) is not int or not 1 <= n <= 512 for n in rows.values()):
            raise ValueError('Invalid full-cardinality support')
        if any(v > rows['U000'] for v in rows.values()) or any(rows[co] > rows['U050'] for co in ('U100','D066','D033','D000')):
            raise ValueError('Science support exceeds measured fusion families')
        if any(v['rows'] != c.COUNTS[role] for v in prepared['summaries'][role].values()):
            raise ValueError('Support population differs')
    return maxima


def validation_stress(model, value, *, device='cuda'):
    if value.role != 'validation':
        raise PermissionError('Validation inference-only stress requires validation role')
    views = (value.primary, value.context) if isinstance(value, PairedCache) else (value,)
    raws = [largest_raw(view) for view in views]
    width = max(raw['mask'].shape[-1] for raw in raws)
    raw = {key: np.stack([np.pad(x[key], ((0,0),(0,0),(0,width-x[key].shape[-1]))) for x in raws], axis=1)
           for key in ('features','vectors','mask')} if len(raws) == 2 else {
               key: raws[0][key] for key in ('features','vectors','mask')}
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if logits.shape != (256,11) or not torch.isfinite(logits).all():
                raise ValueError('Invalid validation inference stress logits')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def projections(measurements):
    train = max(30, math.ceil(max(1.75*(m['cache_seconds']+100*m['training']['runtime_seconds']) for m in measurements)/60))
    reduce = max(15, math.ceil(max(2*(m['cache_seconds']+m['inference_seconds']) for m in measurements)/60))
    return train, reduce


def validate_evidence(r, gate, p):
    c.validate_bound(r, gate, 'PREFLIGHT', dict(gate=gate['content_hash'], prepared=p['content_hash']))
    storage = PAIR_OFFLOAD_POLICY
    if gate.get('schema_version') in (2, 3):
        from .alljets_fusion_sync import PAIR_STORAGE
        storage = PAIR_STORAGE
        if r.get('pair_storage') != storage or gate.get('pair_storage') != storage:
            raise ValueError('Synchronous preflight storage policy differs')
    expected = dict(counts=c.COUNTS, cpus=6, memory_mb=c.MEMORY_MB,
        source_snapshot=gate['source_snapshot'], preparation_source_snapshot=p['source_snapshot'],
        site=gate['execution_site'], recipe=gate['training_recipe'], scientific_fit=False,
        scientific_metrics_for_selection=False, checkpoint_probability_roundtrip=True,
        technical_checks_passed=True, resource_envelope_ok=True, fits_debug_24h=True,
        cublas_workspace_config=':4096:8', support_maxima=support(p),
        technical_teacher='authenticated_parent_U000_train_bank')
    if any(r.get(k) != v for k,v in expected.items()):
        raise ValueError('Fusion preflight facts differ or resource admission failed')
    if re.fullmatch('[1-9][0-9]*', str(r['job_id'])) is None:
        raise ValueError('Genuine Slurm GPU preflight required')
    kernel_validate(r['environment'], 'INSTALLED_ENVIRONMENT', version=2)
    if r['environment']['architecture'] != 'x86_64' or 'A100' not in r['gpu']['name']:
        raise ValueError('Expected SPORC A100 installed environment')
    par = r['parities']
    required_parities = {'installed', 'native_mask'} | {n['node_id']+'_'+precision for n in probe_nodes()
        if n['context_coordinate'] is not None for precision in ('fp32','bf16')}
    if set(par) != required_parities:
        raise ValueError('Missing fusion parity coverage')
    kernel_validate(par['installed'], 'WEAVER_PARITY')
    if (par['native_mask'] is not True or par['installed']['passed'] is not True
            or par['installed']['device'] != 'cuda' or par['installed']['model'] != model_contract()
            or par['installed']['forward_and_feature_and_parameter_gradients'] is not True):
        raise ValueError('Installed/native-mask parity differs')
    for key in required_parities-{'installed','native_mask'}:
        row, precision = par[key], key.rsplit('_',1)[1]
        if (row['passed'] is not True or row['precision'] != precision or row['device_type'] != 'cuda'
                or row['steps'] != 3 or row['checks'] != PARITY_CHECKS or row['parity_backend'] != PARITY_BACKEND
                or row['saved_tensor_storage'] != storage or row['tolerance'] != PARITY_TOLERANCES[precision]):
            raise ValueError('Lossless fusion offload parity differs')
        validate_offload_stats(row['offload_stats'], calls=3)
    if [m['node'] for m in r['measurements']] != probe_nodes():
        raise ValueError('Missing measured model family')
    for m in r['measurements']:
        t = m['training']
        kernel_validate(t, 'KERNEL_TRAINING_REPORT')
        if (t['node'] != m['node'] or t['foundation_sha256'] != p['content_hash']
                or t['recipe_sha256'] != gate['training_recipe']['content_hash'] or t['passes'] != 1
                or t['scientific_fit'] is not False or t['acceptance_only'] is not True
                or t['selected_weights_restored'] is not True or t['final_test_accessed'] is not False
                or t['validation']['rows'] != c.COUNTS['validation'] or m['counts'] != c.COUNTS
                or m['stress_batch'] != 256 or m['stress_steps'] != 3
                or m['validation_stress_batch'] != 256 or m['validation_stress_steps'] != 3
                or m['validation_stress_inference_only'] is not True):
            raise ValueError('Full-population technical-only evidence differs')
        if m['node']['context_coordinate'] is not None:
            validate_offload_stats(m['stress_offload'], calls=3)
        if any(not math.isfinite(x) or x <= 0 for x in (m['cache_seconds'], m['inference_seconds'], t['runtime_seconds'])):
            raise ValueError('Invalid timing measurements')
    times = projections(r['measurements'])
    if times != (r['projected_train_minutes'], r['projected_reduce_minutes']) or max(times) > 1440:
        raise ValueError('Measured full schedule exceeds debug24h; do not shorten training')
    if (not 0 < r['peak_rss_bytes'] < .85*c.MEMORY_MB*1024**2
            or not 0 < r['gpu_peak_bytes'] < .90*r['gpu']['total_memory_bytes']):
        raise ValueError('Measured fusion memory headroom insufficient')
    required_outputs = {'runtime.json'} | {n['node_id']+'.pt' for n in probe_nodes()} | {
        n['node_id']+'_bank/manifest.json' for n in probe_nodes()}
    if gate.get('schema_version') == 3:
        from .alljets_fusion_isolated_preflight import validate_execution
        validate_execution(r, gate)
        required_outputs = {'runtime.json'} | {
            'children/measure_'+n['node_id']+'/'+suffix for n in probe_nodes()
            for suffix in (n['node_id']+'.pt', n['node_id']+'_bank/manifest.json')}
    paths = [ref['path'] for ref in r['outputs']]
    if len(paths) != len(set(paths)) or not required_outputs <= set(paths):
        raise ValueError('Missing bound technical payloads')
    return r


def run(gate):
    if gate.get('schema_version') == 3:
        from .alljets_fusion_isolated_preflight import run as isolated_run
        return isolated_run(gate)
    # CUDA workspace check precedes even expensive source authentication.
    from .preflight_v2 import require_backend
    require_backend()
    parent = c.validate_gate(gate)
    job = w.execution(gate)
    factory, parity = new_model, native_offload_parity
    extra = {}
    if gate.get('schema_version') == 2:
        from .alljets_fusion_sync import new_model as factory, native_offload_parity as parity, PAIR_STORAGE
        extra['pair_storage'] = dict(PAIR_STORAGE)
    authenticated = regular_cache.metadata(parent['admission']['prepared_root'])
    _, p, _, _ = authenticated
    maxima = support(p)
    gpu, environment = gpu_identity(), installed_environment()
    if 'A100' not in gpu['name']:
        raise PermissionError('SPORC A100 required')
    if gate.get('schema_version') == 2:
        original = load_json(gate['recovery']['original_runtime']['path'])
        if (environment != original['environment'] or gpu != original['gpu']
                or os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset') != original['cudnn_v8_disabled']):
            raise PermissionError('Recovery must use the diagnosed software/GPU/backend')
    bound = max(sum(2*p['summaries'][r][co]['resident_bytes'] for r in ('train','validation')
        for co in {n['coordinate'], n['context_coordinate']}-{None}) for n in c.nodes())+256*2**20
    if bound > .30*c.MEMORY_MB*1024**2:
        raise MemoryError('Paired cached inputs exceed reserved RAM envelope')
    output = Path(gate['root'])/'preflight'
    output.mkdir(exist_ok=False)
    runtime = dict(job_id=job, environment=environment, gpu=gpu,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset'))
    write_immutable_json(output/'runtime.json', runtime)
    log = PhaseLog(output, lambda: memory_snapshot(torch))
    parities, measurements = {}, []
    with Measurement() as measured:
        torch.cuda.reset_peak_memory_stats()
        for node in probe_nodes():
            name = node['node_id']
            with log.phase(name+'_load_cached_inputs') as load_phase:
                train = w.cache(gate, p, node, 'train', authenticated=authenticated)
                val = w.cache(gate, p, node, 'validation', authenticated=authenticated)
            native = train.primary if isinstance(train, PairedCache) else train
            if not parities:
                with log.phase('installed_and_native_mask_parity'):
                    parities['installed'] = installed_parity(native, device='cuda')
                    parities['native_mask'] = native_mask_parity(native.batch(np.arange(4)), device='cuda')
                    clear()
            if isinstance(train, PairedCache):
                for bf16 in (False, True):
                    key = name+('_bf16' if bf16 else '_fp32')
                    with log.phase(key+'_offload_parity'):
                        parities[key] = parity(train.primary.batch(np.arange(4)),
                            train.context.batch(np.arange(4)), device='cuda', bf16=bf16)
                        clear()
            with log.phase(name+'_train_stress256_steps3'):
                model = factory(node)
                stats = stress(model, train)
                if isinstance(train, PairedCache):
                    validate_offload_stats(stats, calls=3)
                del model
                clear()
                check_headroom(memory_snapshot(torch), memory_mb=c.MEMORY_MB, future_cache_bytes=0, gpu_bytes=gpu['total_memory_bytes'])
            with log.phase(name+'_validation_inference_stress256_steps3'):
                model = factory(node)
                validation_stress(model, val)
                del model
                clear()
                check_headroom(memory_snapshot(torch), memory_mb=c.MEMORY_MB, future_cache_bytes=0, gpu_bytes=gpu['total_memory_bytes'])
            targets = w.teacher(gate, 'U000', train.identities)
            with log.phase(name+'_full_population_technical_pass'):
                model = factory(node)
                training, state = train_kernel(model, train, val, node=node, device='cuda', acceptance_passes=1,
                    teacher_probabilities=targets, teacher_identities=train.identities)
            with log.phase(name+'_checkpoint_train_bank_readback') as bank_phase:
                save_state(output/(name+'.pt'), state)
                restored = torch.load(output/(name+'.pt'), map_location='cpu', weights_only=True)
                if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
                    raise ValueError('Technical selected checkpoint roundtrip differs')
                model.load_state_dict(restored, strict=True)
                probabilities = predict(model, train, device='cuda', temperature=2.)
                args = dict(foundation_sha256=p['content_hash'], teacher_report_sha256=training['content_hash'], teacher_node=name, role='train')
                publish_bank(output/(name+'_bank'), identities=train.identities, probabilities=probabilities, **args)
                if not np.array_equal(probabilities, load_bank(output/(name+'_bank'), expected_identities=train.identities, **args)):
                    raise ValueError('Technical TRAIN bank roundtrip differs')
            measurements.append(dict(node=node, counts=dict(train=len(train), validation=len(val)), training=training,
                cache_seconds=load_phase['seconds'], inference_seconds=bank_phase['seconds'], stress_offload=stats,
                stress_batch=256, stress_steps=3, validation_stress_batch=256, validation_stress_steps=3,
                validation_stress_inference_only=True))
            del model, state, restored, probabilities, targets, native, train, val
            clear()
    c.validate_gate(gate)
    if regular_cache.metadata(parent['admission']['prepared_root'])[:2] != authenticated[:2] or installed_environment() != environment:
        raise ValueError('Prepared cache/environment changed')
    snap = memory_snapshot(torch)
    peak = max(snap['peak_rss_bytes'], measured.report()['sampled_peak_tree_rss_bytes'])
    train_minutes, reduce_minutes = projections(measurements)
    result = c.bound_artifact(gate, 'PREFLIGHT', parents=dict(gate=gate['content_hash'], prepared=p['content_hash']),
        **runtime, cpus=6, memory_mb=c.MEMORY_MB, counts=c.COUNTS, site=gate['execution_site'],
        source_snapshot=gate['source_snapshot'], preparation_source_snapshot=p['source_snapshot'],
        recipe=gate['training_recipe'], parities=parities, support_maxima=maxima, measurements=measurements,
        technical_teacher='authenticated_parent_U000_train_bank', peak_rss_bytes=peak, gpu_peak_bytes=snap['gpu_peak_bytes'],
        projected_train_minutes=train_minutes, projected_reduce_minutes=reduce_minutes,
        resource_envelope_ok=bool(peak < .85*c.MEMORY_MB*1024**2 and snap['gpu_peak_bytes'] < .9*gpu['total_memory_bytes']),
        fits_debug_24h=max(train_minutes,reduce_minutes) <= 1440, checkpoint_probability_roundtrip=True,
        scientific_fit=False, scientific_metrics_for_selection=False, technical_checks_passed=True,
        outputs=[file_ref(path, root=output) for path in sorted(output.rglob('*')) if path.is_file()], **extra)
    write_immutable_json(output/'measurement.json', result)
    print(f'LUKA fusion measured requests: train={train_minutes}min reduce={reduce_minutes}min; debug ceiling=1440min', flush=True)
    validate_evidence(result, gate, p)
    write_immutable_json(output/'preflight.json', result)
    return result
