"""Fresh real-SPORC acceptance, not scientific model selection or training reuse."""
from pathlib import Path
import math
import os
import re

import numpy as np

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from . import regular_campaign as c, regular_cache as cache
from .preflight_v2 import PhaseLog, memory_snapshot, require_backend


def stress_policy():
    return dict(batch=256, steps=3, train='longest_train_backward',
                validation='longest_validation_inference_only', validation_labels_used=False)


def validation_stress(model, validation, *, device='cuda'):
    """Shape/memory test only: never a loss, gradient, metric or fitted checkpoint."""
    import torch
    from hlt_classification.jetclass2_delphes.runner import _forward
    if validation.role != 'validation':
        raise PermissionError('Validation stress accepts validation only')
    lengths = np.concatenate([np.diff(b.offsets) for b in validation.blocks])
    indices = np.repeat(np.argmax(lengths), 256).astype(np.int64)
    raw = validation.batch(indices)
    # The forward path consumes no labels, identities or matching information.
    raw = {key: raw[key] for key in ('features', 'vectors', 'mask')}
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if logits.shape != (256, 11) or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite/invalid validation stress output')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def probe_nodes():
    # Anchor first, then full coordinate coverage. All KD probes use a technical
    # U000 TRAIN bank, never a scientific teacher or held-out label target.
    names = ['U000', 'OFFLINE', 'COARSE_U050', 'COARSE_U100', 'COARSE_D066', 'COARSE_D033', 'DIRECT_D000']
    indexed = {n['node_id']: n for n in c.nodes()}
    return [{**indexed[name], 'node_id': name+'_probe', 'teacher': None if name in ('U000', 'OFFLINE') else 'U000_probe'}
            for name in names]


def projections(measurements):
    return (max(30, math.ceil(max(1.75*(x['cache_seconds']+100*x['training']['runtime_seconds']) for x in measurements)/60)),
            max(15, math.ceil(max(2*(x['cache_seconds']+x['inference_seconds']) for x in measurements)/60)))


def validate_evidence(r, gate, p, manifest):
    c.validate(r, 'PREFLIGHT', dict(gate=gate['content_hash'], prepared=p['content_hash'], cache=manifest['content_hash']), version=2)
    expected = dict(counts=c.COUNTS, cpus=6, memory_mb=90000, site=gate['execution_site'],
        recipe=c.recipe(), source_snapshot=gate['source_snapshot'], scientific_fit=False,
        scientific_metrics_for_selection=False, checkpoint_probability_roundtrip=True,
        technical_checks_passed=True, resource_envelope_ok=True, fits_debug_24h=True,
        preparation_source_snapshot=p['source_snapshot'], stress_policy=stress_policy())
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Fresh regular preflight evidence/policy differs')
    parity = r['parities']['installed']
    kernel_validate(parity, 'WEAVER_PARITY')
    from hlt_classification.jetclass2_delphes.model import model_contract
    if (not parity['passed'] or parity['device'] != 'cuda' or not parity['forward_and_feature_and_parameter_gradients']
            or parity['model'] != model_contract()
            or r['peak_rss_bytes'] >= .85*90000*1024**2
            or r['gpu_peak_bytes'] >= .9*r['gpu']['total_memory_bytes']
            or 'A100' not in r['gpu']['name']
            or any(not math.isfinite(r[k]) or r[k] <= 0 for k in ('peak_rss_bytes', 'gpu_peak_bytes'))):
        raise ValueError('Installed parity/headroom differs')
    if re.fullmatch(r'[1-9][0-9]*', str(r['job_id'])) is None:
        raise ValueError('Genuine preflight Slurm job identity required')
    if [m['node'] for m in r['measurements']] != probe_nodes():
        raise ValueError('All seven technical coordinate probes required')
    for m in r['measurements']:
        report = m['training']
        kernel_validate(report, 'KERNEL_TRAINING_REPORT')
        if (report['node'] != m['node'] or report['foundation_sha256'] != p['content_hash']
                or report['recipe_sha256'] != c.recipe()['content_hash'] or report['passes'] != 1
                or report['scientific_fit'] is not False or report['acceptance_only'] is not True
                or report['selected_weights_restored'] is not True or report['final_test_accessed'] is not False
                or report['validation']['rows'] != c.COUNTS['validation']
                or m['counts'] != c.COUNTS
                or m['stress_batch'] != 256 or m['stress_steps'] != 3
                or m['validation_stress_batch'] != 256 or m['validation_stress_steps'] != 3
                or m['validation_stress_inference_only'] is not True
                or any(not math.isfinite(m[k]) or m[k] < 0 for k in ('cache_seconds', 'inference_seconds'))
                or not math.isfinite(report['runtime_seconds']) or report['runtime_seconds'] <= 0):
            raise ValueError('Probe coverage/technical-only training differs')
    train, reduce = projections(r['measurements'])
    if (r['projected_train_minutes'], r['projected_reduce_minutes']) != (train, reduce) or max(train, reduce) > 1440:
        raise ValueError('Measured regular resources exceed debug or differ')
    expected_maxima = {role: {co: p['summaries'][role][co]['maximum'] for co in cache.s.COORDINATES}
                      for role in ('train', 'validation')}
    if r['stress_maxima'] != expected_maxima or any(
            type(n) is not int or not 1 <= n <= 512 for rows in expected_maxima.values() for n in rows.values()):
        raise ValueError('Role-specific full-support stress witnesses differ')
    required = {'runtime.json'} | {n['node_id']+'.pt' for n in probe_nodes()} | {n['node_id']+'_bank/manifest.json' for n in probe_nodes()}
    paths = [v['path'] for v in r['outputs']]
    if len(paths) != len(set(paths)) or not required <= set(paths):
        raise ValueError('Missing preflight payload bindings')
    return r


def run(gate):
    require_backend()
    import torch
    from hlt_classification.context_fusion.inputs import new_model
    from hlt_classification.context_fusion.worker import clear, save_state, stress
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
    from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
    from hlt_classification.jetclass2_delphes.model import installed_environment
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
    from hlt_classification.cms2jc2_response.measurement import Measurement

    c.validate_gate(gate)
    root = Path(gate['root'])
    prepared_root = c.prepared_root(gate)
    original_gate = c.preparation_gate(gate)
    authenticated = cache.metadata(prepared_root)
    manifest, p, inventory, splits = authenticated
    c.check_population(inventory, splits)
    if (p['source_snapshot'] != original_gate['source_snapshot'] or manifest['parents']['gate'] != original_gate['content_hash']
            or p['foundation_root'] != str(Path(original_gate['root'])/'foundation')
            or p['conventions'] != c.conventions(p['audit']['content_hash'])):
        raise ValueError('Cache/source gate lineage differs')
    job, cpus, memory = allocation(gate['execution_site'])
    if (cpus, memory) != (6, 90000):
        raise PermissionError('Regular preflight requires 6 CPUs/90000 MiB')
    output = root/'preflight'
    output.mkdir(exist_ok=False)
    environment, gpu = installed_environment(), gpu_identity()
    write_immutable_json(output/'runtime.json', dict(job_id=job, environment=environment, gpu=gpu,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset')))
    bound = sum(2*m['resident_bytes'] for role in p['summaries'].values() for m in role.values())+128*2**20
    if bound > .30*memory*1024**2:
        raise MemoryError('All-view acceptance cache bound exceeds reserved 30% host RAM')
    log = PhaseLog(output, lambda: memory_snapshot(torch))
    bundles, timings, measurements, maxima = {}, {}, [], {'train':{}, 'validation':{}}
    teacher_arguments = None
    with Measurement() as measured:
        torch.cuda.reset_peak_memory_stats()
        for role in ('train', 'validation'):
            bundles[role] = {}
            for co in cache.s.COORDINATES:
                with log.phase(f'load_{role}_{co}') as phase:
                    bundles[role][co] = cache.load(prepared_root, role=role, coordinate=co,
                        max_ram_bytes=int(memory*1024**2*.15), authenticated=authenticated)
                timings[role, co] = phase['seconds']
        with log.phase('installed_single_encoder_parity'):
            parities = dict(installed=installed_parity(bundles['train']['U000'], device='cuda'))
            clear()
        for node in probe_nodes():
            co, name = node['coordinate'], node['node_id']
            train, val = bundles['train'][co], bundles['validation'][co]
            for role, value in (('train', train), ('validation', val)):
                maxima[role][co] = max(int(np.diff(b.offsets).max()) for b in value.blocks)
            print(f'LUKA support coordinate={co} train_max={maxima["train"][co]} '
                  f'validation_max={maxima["validation"][co]} validation_mode=inference_only', flush=True)
            with log.phase(name+'_stress256_steps3'):
                model = new_model(node)
                stress(model, train)
                del model
                clear()
                snap = memory_snapshot(torch)
                if snap['peak_rss_bytes'] >= .85*memory*1024**2 or snap['gpu_peak_bytes'] >= .9*gpu['total_memory_bytes']:
                    raise MemoryError('Regular full-support stress exceeds headroom')
            with log.phase(name+'_validation_inference_stress256_steps3'):
                model = new_model(node)
                validation_stress(model, val)
                del model
                clear()
                snap = memory_snapshot(torch)
                if snap['peak_rss_bytes'] >= .85*memory*1024**2 or snap['gpu_peak_bytes'] >= .9*gpu['total_memory_bytes']:
                    raise MemoryError('Regular validation stress exceeds headroom')
            kwargs = {} if node['teacher'] is None else dict(teacher_identities=train.identities,
                teacher_probabilities=load_bank(output/'U000_probe_bank', expected_identities=train.identities, **teacher_arguments))
            with log.phase(name+'_full_population_pass'):
                model = new_model(node)
                training, state = train_kernel(model, train, val, node=node, device='cuda', acceptance_passes=1, **kwargs)
            with log.phase(name+'_checkpoint_train_bank') as phase:
                save_state(output/(name+'.pt'), state)
                restored = torch.load(output/(name+'.pt'), weights_only=True, map_location='cpu')
                if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
                    raise ValueError('Technical checkpoint readback differs')
                model.load_state_dict(restored, strict=True)
                probabilities = predict(model, train, device='cuda', temperature=2.)
                arguments = dict(foundation_sha256=p['content_hash'], teacher_report_sha256=training['content_hash'], teacher_node=name, role='train')
                publish_bank(output/(name+'_bank'), identities=train.identities, probabilities=probabilities, **arguments)
                replay = load_bank(output/(name+'_bank'), expected_identities=train.identities, **arguments)
                if not np.array_equal(replay, probabilities):
                    raise ValueError('Technical TRAIN probability roundtrip differs')
                if name == 'U000_probe':
                    teacher_arguments = arguments
            measurements.append(dict(node=node, training=training, counts=dict(train=len(train), validation=len(val)),
                cache_seconds=timings['train', co]+timings['validation', co],
                inference_seconds=phase['seconds'], stress_steps=3, stress_batch=256,
                validation_stress_batch=256, validation_stress_steps=3, validation_stress_inference_only=True))
            del model, state, restored, probabilities, replay, kwargs
            clear()
    c.validate_gate(gate)
    if cache.metadata(prepared_root)[:2] != authenticated[:2] or installed_environment() != environment:
        raise ValueError('Preflight source/cache/environment changed')
    train_minutes, reduce_minutes = projections(measurements)
    snap = memory_snapshot(torch)
    peak = max(snap['peak_rss_bytes'], measured.report()['sampled_peak_tree_rss_bytes'])
    result = c.make_artifact('PREFLIGHT', version=2, parents=dict(gate=gate['content_hash'], prepared=p['content_hash'], cache=manifest['content_hash']),
        counts=c.COUNTS, source_snapshot=gate['source_snapshot'], job_id=job, cpus=cpus, memory_mb=memory,
        site=gate['execution_site'], environment=environment, gpu=gpu, parities=parities,
        preparation_source_snapshot=p['source_snapshot'], stress_policy=stress_policy(),
        stress_maxima=maxima, measurements=measurements, recipe=c.recipe(), peak_rss_bytes=peak, gpu_peak_bytes=snap['gpu_peak_bytes'],
        projected_train_minutes=train_minutes, projected_reduce_minutes=reduce_minutes,
        resource_envelope_ok=bool(peak < .85*memory*1024**2 and snap['gpu_peak_bytes'] < .9*gpu['total_memory_bytes']),
        fits_debug_24h=max(train_minutes, reduce_minutes) <= 1440, checkpoint_probability_roundtrip=True,
        technical_checks_passed=True, scientific_fit=False, scientific_metrics_for_selection=False,
        outputs=[file_ref(path, root=output) for path in sorted(output.rglob('*')) if path.is_file()])
    # Keep diagnostics even when resource admission fails; no false PASS report.
    write_immutable_json(output/'measurement.json', result)
    validate_evidence(result, gate, p, manifest)
    write_immutable_json(output/'preflight.json', result)
    return result
