"""Explicit v2 qualification from the completed, original-source diagnostic."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.provenance import validate_source_snapshot
from .contracts import artifact, validate, SITE, PARITY, MANIFEST_HASH
from .inputs import Bundle
from . import precision

DIAGNOSTIC_COMMIT = 'ad55d3cd32d7826d6e16a475228fece3c6099da4'
DIAGNOSTIC_JOB = '131484'
PROGRAM = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_pair_hydra.precision_campaign import results,accepted_job
spec=load_json(sys.argv[1]); report=results(spec)
print('HYDRA_PRECISION_REUSE='+json.dumps(dict(report_hash=report['content_hash'],job=accepted_job(spec)),sort_keys=True,allow_nan=False))
'''


def policy():
    return artifact('PORTABILITY_POLICY', version=2,
        diagnostic_commit=DIAGNOSTIC_COMMIT, diagnostic_job=DIAGNOSTIC_JOB,
        cpu_gpu_fp32_limits=dict(max_abs=1e-5, rms=1e-6), bf16_same_batch_repeat='exact',
        saved_metric_replay_tolerance=dict(absolute=1e-12, relative=1e-12),
        cross_site_auc_limit=PARITY['auc_abs'], cross_site_probability_limits='report_only_original_v1',
        baseline='authenticated_hydra_bf16_replay_of_saved_control',
        recovery_anchors='historical_sporc_M0HLT_OFFLINE', reference_retrained=False,
        all_four_real_miniatures_required=True)


def same_metrics(actual, saved):
    """Allow scalar reduction roundoff, never different identities or structure."""
    if isinstance(actual, dict) and isinstance(saved, dict):
        return actual.keys() == saved.keys() and all(same_metrics(actual[k], saved[k]) for k in actual)
    if isinstance(actual, list) and isinstance(saved, list):
        return len(actual) == len(saved) and all(same_metrics(a, b) for a, b in zip(actual, saved))
    if type(actual) is float and type(saved) is float:
        return math.isfinite(actual) and math.isfinite(saved) and math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12)
    return type(actual) is type(saved) and actual == saved


def qualify(report, arrays, hydra, saved, rows):
    """Recompute numerical evidence from immutable predictions, never GPU inference."""
    for a in (hydra, saved):
        if len(a['labels']) != rows or a['probabilities'].shape != (rows, 11):
            raise ValueError('Diagnostic full population differs')
        precision.validate_probabilities(a['probabilities'])
    if any(not np.array_equal(hydra[k], saved[k]) for k in ('identities', 'labels')):
        raise ValueError('Diagnostic full identity/label order differs')
    ix = np.concatenate(precision.fixed_batches(rows))
    if (arrays['indices'].dtype != np.int64 or not np.array_equal(arrays['indices'], ix)
            or any(not np.array_equal(arrays[k], saved[k][ix]) for k in ('identities', 'labels'))
            or not np.array_equal(arrays['saved_sporc_bf16'], saved['probabilities'][ix])
            or not np.array_equal(arrays['full_gpu_bf16'], hydra['probabilities'][ix])):
        raise ValueError('Fixed original diagnostic batches differ')
    probe = report['probe']; batches = precision.fixed_batches(rows)
    if (probe['rows'] != len(ix) or probe['batch_starts'] != [int(x[0]) for x in batches]
            or probe['batch_lengths'] != [len(x) for x in batches]
            or probe['identities_sha256'] != hashlib.sha256(arrays['identities'].tobytes()).hexdigest()
            or set(probe['settings']) != set(precision.MODES)):
        raise ValueError('Probe selection/identity/settings evidence differs')
    for mode in precision.MODES:
        if precision.validate_probabilities(arrays[mode]).shape != (len(ix), 11):
            raise ValueError('Probe prediction shape differs')
        settings = probe['settings'][mode]
        if settings['torch_cpu_threads'] != 1:
            raise ValueError('Numerical thread configuration differs')
        if mode.endswith('_math') and (not settings['math_sdp'] or any(settings[k] for k in (
                'matmul_tf32', 'cudnn_tf32', 'bf16_reduced_precision_reduction',
                'flash_sdp', 'memory_efficient_sdp', 'cudnn_sdp'))):
            raise ValueError('FP32/math probe settings differ')
    for name, expected in probe['comparisons'].items():
        a, b = name.split('_vs_')
        if not same_metrics(precision.differences(arrays[a], arrays[b]), expected):
            raise ValueError('Saved probe metrics differ from probability arrays')
    full = precision.full_comparison(saved['labels'], hydra['probabilities'], saved['probabilities'])
    if any(not same_metrics(v, report['full_validation'][k]) for k, v in full.items()):
        raise ValueError('Saved full metrics differ from probability arrays')
    if report['full_validation']['settings'] != probe['settings']['gpu_bf16_default']:
        raise ValueError('Default backend settings differ between full/probe inference')
    fp32 = precision.differences(arrays['gpu_fp32_math'], arrays['cpu_fp32_math'])
    if (any(fp32[k] > limit for k, limit in policy()['cpu_gpu_fp32_limits'].items())
            or not np.array_equal(arrays['gpu_bf16_default'], arrays['full_gpu_bf16'])
            or full['differences']['auc_abs'] > policy()['cross_site_auc_limit']):
        raise ValueError('Hydra v2 numerical qualification failed')
    # Keep authenticated original scalar values in reusable hashes after checking
    # the recomputation, so different CPU reduction paths cannot change a receipt.
    return dict(cpu_gpu_fp32=probe['comparisons']['gpu_fp32_math_vs_cpu_fp32_math'],
        full_validation={k:report['full_validation'][k] for k in full},
        interpretation='Local FP32 consistency and reproducible BF16; no A100 FP32 equivalence claim')


def authenticate(path, bundle_root, reference, counts):
    path = Path(path).resolve(strict=True); original = load_json(path)
    validate(original, 'PRECISION_SPEC')
    if (original['source_commit'] != DIAGNOSTIC_COMMIT
            or original['source_snapshot']['git_commit'] != DIAGNOSTIC_COMMIT
            or path != Path(original['root'])/'campaign_spec.json'):
        raise ValueError('Expected original completed diagnostic source')
    # Validate foreign source bytes before importing them. The module-ownership
    # guard belongs to that checkout: PROGRAM's results(spec) runs its original
    # precision_campaign.check -> campaign.current -> foundation._source.
    validate_source_snapshot(original['source_snapshot'], repository=original['project_dir'], require_clean=True)
    print('Authenticating completed Hydra precision job 131484 in its original source; no inference...', flush=True)
    spec_ref, proof = replay(path, original['project_dir'], PROGRAM, 'HYDRA_PRECISION_REUSE=')
    if proof['job'] != DIAGNOSTIC_JOB:
        raise ValueError('Not the reviewed Hydra precision job')
    root = path.parent; folder = root/'precision'
    report = load_json(folder/'report.json')
    if report['content_hash'] != proof['report_hash']:
        raise ValueError('Diagnostic changed during original-source replay')
    refs = dict(spec=spec_ref, report=file_ref(folder/'report.json'), receipt=file_ref(folder/'receipt.json'),
        full=file_ref(folder/'full_validation.json'), probe=file_ref(folder/'fixed_batch_predictions.npz'),
        predictions=file_ref(folder/'hydra_validation_predictions.npz'))
    journals = [file_ref(p) for p in sorted((root/'submission').rglob('*')) if p.is_file()]
    value = artifact('PORTABILITY_EVIDENCE', version=2,
        parents=dict(diagnostic=original['content_hash'], report=report['content_hash']),
        policy=policy(), job_id=DIAGNOSTIC_JOB, refs=refs, submission_files=journals)
    validate_evidence(value, bundle_root, reference, counts)
    return value


def validate_evidence(value, bundle_root, reference, counts):
    refs = value['refs']
    for r in [*refs.values(), *value['submission_files']]: validate_file_ref(r)
    old = load_json(refs['spec']['path']); report = load_json(refs['report']['path'])
    receipt = load_json(refs['receipt']['path'])
    validate(old, 'PRECISION_SPEC')
    validate(report, 'PRECISION_REPORT', dict(spec=old['content_hash']))
    validate(receipt, 'PRECISION_RECEIPT', dict(spec=old['content_hash'], report=report['content_hash']))
    validate(value, 'PORTABILITY_EVIDENCE', dict(diagnostic=old['content_hash'], report=report['content_hash']), version=2)
    folder = Path(old['root'])/'precision'; bundle = Bundle(bundle_root)
    expected_refs = dict(spec=Path(old['root'])/'campaign_spec.json', report=folder/'report.json',
        receipt=folder/'receipt.json', full=folder/'full_validation.json',
        probe=folder/'fixed_batch_predictions.npz', predictions=folder/'hydra_validation_predictions.npz')
    if (value['policy'] != policy() or value['job_id'] != DIAGNOSTIC_JOB or receipt['job_id'] != DIAGNOSTIC_JOB
            or old['source_commit'] != DIAGNOSTIC_COMMIT or old['parents']['transfer'] != MANIFEST_HASH
            or old['reference'] != reference or old['counts'] != counts
            or Path(old['bundle_root']).resolve() != Path(bundle_root).resolve()
            or old['execution_site'] != SITE or report['environment'] != reference['environment']
            or report['gpu']['name'] != SITE['gpu_name'] or report['reference_retrained'] is not False
            or report['admission'] is not False or report['scientific_fits_released'] is not False
            or refs != {k:file_ref(p) for k,p in expected_refs.items()}
            or receipt['report'] != file_ref(folder/'report.json', root=folder)
            or report['checkpoint'] != file_ref(bundle.path('reference/matched/checkpoint.pt'))
            or report['saved_predictions'] != file_ref(bundle.path('reference/matched/predictions.npz'))
            or report['outputs'] != [file_ref(folder/n, root=folder) for n in (
                'hydra_validation_predictions.npz', 'full_validation.json', 'fixed_batch_predictions.npz')]
            or not value['submission_files']):
        raise ValueError('Hydra portability lineage differs')
    full = load_json(refs['full']['path'])
    if full != artifact('PRECISION_FULL', parents=dict(spec=old['content_hash']), **report['full_validation']):
        raise ValueError('Full diagnostic report evidence differs')
    summary = qualify(report, load_npz_arrays(refs['probe']['path']), load_npz_arrays(refs['predictions']['path']),
                      load_npz_arrays(bundle.path('reference/matched/predictions.npz')), counts['validation'])
    return report, summary


def setup_record(spec, val):
    from hlt_classification.jetclass2_delphes.execution import gpu_identity
    report, summary = validate_evidence(spec['portability_evidence'], spec['bundle_root'], spec['reference'], spec['counts'])
    baseline = load_npz_arrays(spec['portability_evidence']['refs']['predictions']['path'])
    if any(not np.array_equal(baseline[k], getattr(val, k)) for k in ('identities', 'labels')):
        raise ValueError('Setup population/order differs from qualified Hydra diagnostic')
    gpu = gpu_identity(); settings = precision.backend_settings()
    if gpu != report['gpu'] or settings != report['full_validation']['settings']:
        differences = {k:dict(expected=report['full_validation']['settings'].get(k), actual=v)
                       for k,v in settings.items() if report['full_validation']['settings'].get(k) != v}
        raise ValueError('Setup GPU/backend differs from qualified diagnostic: '+json.dumps(dict(
            expected_gpu=report['gpu'], actual_gpu=gpu, settings=differences), sort_keys=True))
    return artifact('PORTABILITY', version=2, parents=dict(spec=spec['content_hash'],
        evidence=spec['portability_evidence']['content_hash']), policy=policy(),
        gpu=report['gpu'], environment=report['environment'], settings=report['full_validation']['settings'],
        qualified=True, control_inference_reused=True, resource_acceptance=False, summary=summary)


def validate_setup(spec, value):
    report, summary = validate_evidence(spec['portability_evidence'], spec['bundle_root'], spec['reference'], spec['counts'])
    expected = artifact('PORTABILITY', version=2, parents=dict(spec=spec['content_hash'],
        evidence=spec['portability_evidence']['content_hash']), policy=policy(),
        gpu=report['gpu'], environment=report['environment'], settings=report['full_validation']['settings'],
        qualified=True, control_inference_reused=True, resource_acceptance=False, summary=summary)
    if value != expected:
        raise ValueError('Saved v2 portability record differs')
    return value
