"""Read-only replay, executed with the FAILED job's original Python package.

This file is launched by absolute path from the new source. All package imports
below must resolve to the old pinned worktree. It never creates a CUDA context,
loads particle arrays/checkpoints, or writes an original artifact.
"""
from pathlib import Path
import json
import re
import sys

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c
from hlt_classification.luka_fullsim import alljets_fusion_preflight as pre
from hlt_classification.luka_fullsim import alljets_fusion_isolated_preflight as iso
from hlt_classification.luka_fullsim import alljets_fusion_submission as sub, regular_cache

FAILED_COMMIT = '922bd972ef9ec52079e9e750e6e7896e857aba8b'
FAILED_JOB = '21898881'
FAILURE_END = 'ProcessLookupError: [Errno 3] No such process'
MEMORY_METHOD = 'slurm_single_task_batch_maxrss_plus_largest_sequential_child_tree_peak_v1'


def validate_child_memory(row, kind, name):
    """Check high-water evidence after check_child authenticates the phase/node.

    In the pinned v4 producer, these two single-encoder parity phases load CPU
    inputs but perform neither installed/native nor paired offload CUDA parity.
    Their zero CUDA allocation is a measurement, not missing evidence.
    """
    no_cuda_parity = kind == 'parity' and name in (
        'FINAL_DIRECT_D000_probe', 'U000_probe')
    if no_cuda_parity and row.get('parities') != {}:
        raise ValueError(f'{kind}/{name}: no-op parity must record empty parities')
    for field, minimum in (('peak_rss_bytes', 1), ('gpu_peak_bytes', 0 if no_cuda_parity else 1)):
        value = row.get(field)
        if type(value) is not int or value < minimum:
            raise ValueError(f'{kind}/{name}: {field} must be an integer >= {minimum}; got {value!r}')


def logged_results(text, gate):
    """All printed child receipts precede the coordinator-only terminal error."""
    start = text.rfind('Traceback (most recent call last):')
    tail = text[start:]
    if (start < 0 or not tail.rstrip().endswith(FAILURE_END)
            or any(token not in tail for token in (
                'alljets_fusion_isolated_preflight.py', 'with Measurement() as measured:',
                'measurement.py', 'in __exit__', 'raise self.error', 'in _loop', 'self.sample()'))):
        raise ValueError('Not the deferred coordinator process-exit race')
    markers = re.findall(r'^LUKA isolated (parity|stress|measure): (\w+); fresh process, no automatic retry$',
                         text[:start], re.M)
    if markers != iso.sequence(gate):
        raise ValueError('Incomplete or reordered original child execution')
    decoder, rows = json.JSONDecoder(), []
    for match in re.finditer(r'^\{\s*$', text[:start], re.M):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('contract') == 'LUKA_ALLJETS_FUSION_PROBE/v4':
            rows.append(value)
    if len(rows) != 12:
        raise ValueError('Expected exactly twelve logged successful probe receipts')
    return rows


def replay(gate, accounting):
    if (gate.get('schema_version') != 4 or gate.get('source_commit') != FAILED_COMMIT
            or Path(c.__file__).resolve() != Path(gate['project_dir']).resolve()/
                'src/hlt_classification/luka_fullsim/alljets_fusion_campaign.py'):
        raise ValueError('Replay requires the original 922bd972 package')
    parent = c.validate_gate(gate)
    if sub.accepted_jobs(gate) != {'preflight': FAILED_JOB}:
        raise ValueError('Original accepted job journal differs')
    if accounting['job_id'] != FAILED_JOB or accounting['state'] != 'FAILED' or accounting['exit_code'] != '1:0':
        raise ValueError('Original failed accounting differs')
    root, output = Path(gate['root']), Path(gate['root'])/'preflight'
    if any((output/name).exists() for name in ('preflight.json', 'measurement.json', 'failure.json')) or (root/'science').exists():
        raise ValueError('Expected incomplete report only; preserve existing admission or failures')
    if list(output.rglob('failure.json')):
        raise ValueError('A probe failed; report-only recovery is forbidden')
    log_path = root/f'preflight-{FAILED_JOB}.out'
    log_ref = file_ref(log_path)
    logged = logged_results(log_path.read_text(), gate)
    expected_results = {f'{kind}_{name}/result.json' for kind, name in iso.sequence(gate)}
    actual_results = {p.relative_to(output/'children').as_posix() for p in (output/'children').glob('*/result.json')}
    if actual_results != expected_results:
        raise ValueError('Probe result population differs')
    _, prepared, _, _ = regular_cache.metadata(parent['admission']['prepared_root'])
    maxima = pre.support(prepared)
    runtime = load_json(output/'runtime.json')
    if str(runtime['job_id']) != FAILED_JOB:
        raise ValueError('Probe runtime job differs')
    rows, children, pids = {}, [], set()
    for index, (kind, name) in enumerate(iso.sequence(gate)):
        prefix = 'children/'+kind+'_'+name
        row = iso.check_child(load_json(output/prefix/'result.json'), gate, kind, name, runtime)
        if row != logged[index] or row['process_id'] in pids or row['support_maxima'] != maxima:
            raise ValueError('Logged probe/PID/support differs')
        validate_child_memory(row, kind, name)
        pids.add(row['process_id'])
        rows[kind, name] = row
        children.append(dict(phase=kind, node=name, pid=row['process_id'], exit_code=0,
            result=file_ref(output/prefix/'result.json', root=output),
            log=file_ref(output/(prefix+'.out'), root=output)))
    parities, measurements = {}, []
    for node in pre.probe_nodes(gate):
        name = node['node_id']
        stress, timing = rows['stress', name], rows['measure', name]
        parities.update(rows['parity', name]['parities'])
        measurements.append(dict(node=node, counts=c.COUNTS, training=timing['training'],
            cache_seconds=timing['cache_seconds'], inference_seconds=timing['inference_seconds'],
            **{k:stress[k] for k in ('stress_offload', 'stress_batch', 'stress_steps',
                'validation_stress_batch', 'validation_stress_steps', 'validation_stress_inference_only')}))
    rss = accounting['batch_maxrss_bytes']
    if type(rss) is not int or rss <= 0:
        raise ValueError('Missing Slurm batch memory evidence')
    child_peak = max(row['peak_rss_bytes'] for row in rows.values())
    peak = rss + child_peak  # Deliberate double counting, NOT max(rss, child_peak).
    gpu_peak = max(row['gpu_peak_bytes'] for row in rows.values())
    train_minutes, reduce_minutes = pre.projections(measurements)
    report = c.bound_artifact(gate, 'PREFLIGHT', parents=dict(gate=gate['content_hash'], prepared=prepared['content_hash']),
        **runtime, cpus=6, memory_mb=c.MEMORY_MB, counts=c.COUNTS, site=gate['execution_site'],
        source_snapshot=gate['source_snapshot'], preparation_source_snapshot=prepared['source_snapshot'],
        recipe=gate['training_recipe'], parities=parities, support_maxima=maxima, measurements=measurements,
        technical_teacher=pre.technical_teacher(gate), peak_rss_bytes=peak, gpu_peak_bytes=gpu_peak,
        projected_train_minutes=train_minutes, projected_reduce_minutes=reduce_minutes,
        resource_envelope_ok=bool(peak < .85*c.MEMORY_MB*1024**2 and gpu_peak < .9*runtime['gpu']['total_memory_bytes']),
        fits_debug_24h=max(train_minutes,reduce_minutes) <= 1440, checkpoint_probability_roundtrip=True,
        scientific_fit=False, scientific_metrics_for_selection=False, technical_checks_passed=True,
        pair_storage=gate['pair_storage'], preflight_execution=iso.policy(gate), isolated_children=children,
        outputs=[file_ref(path, root=output) for path in sorted(output.rglob('*')) if path.is_file()])
    # A recovered candidate is returned in memory only, never written into the old run.
    # The new versioned recovery receipt is the ONLY admission authority.
    for ref in report['outputs']:
        validate_file_ref(ref, root=output)
    for m in measurements:
        name = m['node']['node_id']
        prefix = 'children/measure_'+name+'/'+name+'_bank'
        bank = load_json(output/prefix/'manifest.json')
        kernel_validate(bank, 'PROBABILITY_BANK')
        expected = dict(foundation_sha256=prepared['content_hash'], teacher_report_sha256=m['training']['content_hash'],
            teacher_node=name, role='train', rows=c.COUNTS['train'], temperature=2., class_count=11,
            dtype='float32', final_test_accessed=False)
        if any(bank.get(k) != v for k, v in expected.items()):
            raise ValueError('Technical bank lineage differs')
        for shard in bank['shards']:
            if dict(path=prefix+'/'+shard['path'], bytes=shard['bytes'], sha256=shard['sha256']) not in report['outputs']:
                raise ValueError('Missing authenticated bank shard')
    issue = None
    try:
        pre.validate_evidence(report, gate, prepared)
    except (ValueError, PermissionError) as exc:
        issue = str(exc)  # No admission on ANY validator failure; preserve diagnostics.
    validate_file_ref(log_ref)
    return dict(gate_sha256=gate['content_hash'], prepared=prepared, parent=parent,
        candidate=report, admission_error=issue, original_job_status='FAILED',
        exit_evidence='original_coordinator_finished_all_children_then_deferred_sampler_error',
        memory_evidence=dict(method=MEMORY_METHOD, slurm_batch_peak_bytes=rss,
            largest_child_tree_peak_bytes=child_peak, conservative_sum_bytes=peak,
            coordinator_sampler_incomplete=True, sampled_not_exact=True),
        inputs=[file_ref(root/'gate_spec.json'), log_ref] +
            [file_ref(output/ref['path']) for ref in report['outputs']])


if __name__ == '__main__':
    gate = load_json(sys.argv[1])
    accounting = json.load(sys.stdin)
    print('LUKA_REPORT_REPLAY='+json.dumps(replay(gate, accounting), sort_keys=True, allow_nan=False))
