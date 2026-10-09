"""Authenticate the accepted technical evidence without importing probe weights."""
from pathlib import Path
import math

from hlt_classification.data.cache_contracts import load_json, sha256_file, validate_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
from hlt_classification.jetclass2_delphes.execution import execution_site
from . import stage2 as s, preflight_reuse
from .contracts import artifact
from .preflight_v2 import CASES, MEMORY_MB, _node

PREFLIGHT_HASH = 'ffaa43d6d83b4587a9e8a6fca0995ad30238ccc7936898022642e6856b53274c'
PREFLIGHT_COMMIT = 'bcccd498ee66fdee6fd3731b823d09ac2bb5df8b'
COUNTS = dict(train=100000, validation=50000)
ADDITIONS = frozenset('src/hlt_classification/luka_fullsim/'+name+'.py' for name in
    ('science_admission', 'science_campaign', 'science_cache', 'science_worker', 'science_submission'))
REQUIRED = (*sorted(ADDITIONS), 'scripts/luka_fullsim_science.py',
    'scripts/queue_luka_fullsim_science.sh', 'sbatch/run_luka_fullsim_science.sh',
    'docs/plans/LUKA_FULLSIM_SCIENCE_PLAN.md', 'docs/contracts/LUKA_FULLSIM_SCIENCE.md',
    'docs/LUKA_FULLSIM_SCIENCE_RUNBOOK.md', 'tests/test_luka_fullsim_science.py')


def source(project, commit):
    import subprocess
    value = s.source(project, commit)
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
        capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit all Luka science code/contracts/tests before execution')
    if Path(__file__).resolve() != Path(project).resolve() / 'src/hlt_classification/luka_fullsim/science_admission.py':
        raise ValueError('Science import escapes pinned project')
    return value


def compatible_source(measured, active, *, preflight_project, project):
    """A new source parent; every old scientific module still byte-identical."""
    old, new = Path(preflight_project).resolve(), Path(project).resolve()
    validate_source_snapshot(measured, repository=old, require_clean=True)
    validate_source_snapshot(active, repository=new, require_clean=True)
    before, after = preflight_reuse._files(old), preflight_reuse._files(new)
    if not before or after-before != ADDITIONS or not before <= after:
        raise ValueError('Science source additions/deletions differ from exact allowlist')
    records = {}
    for name in sorted(before):
        digest = sha256_file(old/name)
        if sha256_file(new/name) != digest:
            raise ValueError('Accepted scientific source changed: '+name)
        records[name] = digest
    return artifact('SCIENCE_SOURCE_REUSE', parents=dict(preflight_source=measured['content_hash'],
        science_source=active['content_hash']), identical_source_files=records,
        added_modules=sorted(ADDITIONS), final_test_accessed=False)


def validate_facts(r, p):
    """Semantic checks in addition to the exact accepted content hash."""
    from hlt_classification.jetclass2_delphes.model import model_contract
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import (
        PARITY_CHECKS, PARITY_BACKEND, PAIR_OFFLOAD_POLICY, PARITY_TOLERANCES, validate_offload_stats)
    validate_content_hash(r, expected_contract='LUKA_FULLSIM_GPU_PREFLIGHT/v2', expected_schema_version=2)
    if (r['counts'] != COUNTS or p['counts'] != COUNTS or r['recipe'] != recipe()
            or r['site'] != execution_site('sporc_a100_debug') or (r['cpus'], r['memory_mb']) != (6, MEMORY_MB)
            or any(r.get(k) is not True for k in ('technical_checks_passed', 'checkpoint_probability_roundtrip',
                                                'resource_envelope_ok', 'fits_debug_24h'))
            or any(r.get(k) is not False for k in ('scientific_fit', 'scientific_metrics_for_selection',
                'final_test_accessed', 'production_submission_authorized', 'automatic_followup'))
            or r['preparation_source_snapshot'] != p['source_snapshot']):
        raise ValueError('Accepted full-population technical evidence differs')
    kernel_validate(r['environment'], 'INSTALLED_ENVIRONMENT', version=2)
    if r['environment']['architecture'] != 'x86_64' or 'A100' not in r['gpu']['name']:
        raise ValueError('Expected measured SPORC A100 environment')
    parity = r['parities']
    if set(parity) != {'installed', 'native_mask', 'FUSION_probe_False', 'FUSION_probe_True',
                      'HLT_PAIR_probe_False', 'HLT_PAIR_probe_True'}:
        raise ValueError('Missing technical parity cases')
    single = parity['installed']
    kernel_validate(single, 'WEAVER_PARITY')
    if (single['passed'] is not True or single['device'] != 'cuda' or single['model'] != model_contract()
            or single['forward_and_feature_and_parameter_gradients'] is not True or parity['native_mask'] is not True):
        raise ValueError('Installed/native-mask parity differs')
    for key in ('FUSION_probe_False', 'FUSION_probe_True', 'HLT_PAIR_probe_False', 'HLT_PAIR_probe_True'):
        row, precision = parity[key], 'bf16' if key.endswith('_True') else 'fp32'
        if (row['passed'] is not True or row['precision'] != precision or row['device_type'] != 'cuda'
                or row['steps'] != 3 or row['checks'] != PARITY_CHECKS or row['parity_backend'] != PARITY_BACKEND
                or row['saved_tensor_storage'] != PAIR_OFFLOAD_POLICY or row['tolerance'] != PARITY_TOLERANCES[precision]):
            raise ValueError('Fusion saved-tensor parity differs')
        validate_offload_stats(row['offload_stats'], calls=3)
    if len(r['measurements']) != len(CASES):
        raise ValueError('Missing measured model family')
    for i, row in enumerate(r['measurements']):
        report = row['training']
        kernel_validate(report, 'KERNEL_TRAINING_REPORT')
        if (row['node'] != _node(i) or row['name'] != CASES[i][0] or report['node'] != row['node']
                or report['foundation_sha256'] != p['content_hash'] or report['recipe_sha256'] != recipe()['content_hash']
                or report['scientific_fit'] is not False or report['acceptance_only'] is not True
                or report['passes'] != 1 or report['selected_weights_restored'] is not True
                or report['final_test_accessed'] is not False or row['stress_batch'] != 256 or row['stress_steps'] != 3):
            raise ValueError('Technical full-pass/stress lineage differs')
        if row['node']['context_coordinate'] is not None:
            validate_offload_stats(row['stress_offload'], calls=3)
        for seconds in (row['cache_seconds'], report['runtime_seconds'], row['inference_seconds']):
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Invalid measured timing')
    train = math.ceil(max(1.75*(v['cache_seconds']+100*v['training']['runtime_seconds']) for v in r['measurements'])/60)
    reduce = math.ceil(max(2*(v['cache_seconds']+v['inference_seconds']) for v in r['measurements'])/60)
    if ((train, reduce) != (r['projected_train_minutes'], r['projected_reduce_minutes'])
            or not 0 < train <= 1440 or not 0 < reduce <= 1440
            or not 0 < r['peak_rss_bytes'] < .85*MEMORY_MB*1024**2
            or not 0 < r['gpu_peak_bytes'] < .9*r['gpu']['total_memory_bytes']):
        raise ValueError('Measured resource envelope differs')
    # U000 contains the HLT shell and unused offline tail. All other views
    # are subsets in cardinality, including OFFLINE; features still replay exactly.
    for role in ('train', 'validation'):
        top = p['summaries'][role]['U000']
        for value in p['summaries'][role].values():
            if value['rows'] != COUNTS[role] or value['maximum'] > top['maximum']:
                raise ValueError('Science support exceeds accepted U000 envelope')
        largest_two = sorted((v['resident_bytes'] for v in p['summaries'][role].values()), reverse=True)[:2]
        accepted = sum(p['summaries'][role][c]['resident_bytes'] for c in ('U000', 'U050', 'D000'))
        if sum(largest_two) > accepted:
            raise MemoryError('Science cache reservation exceeds measured envelope')
    return r


def authenticate(*, prepared_root, preflight_path, prepared_project, preflight_project, project, active):
    p, f, _, _ = s.load_prepared(prepared_root)
    path = Path(preflight_path).resolve(strict=True)
    r = load_json(path)
    validate_facts(r, p)
    if r['content_hash'] != PREFLIGHT_HASH or r['source_snapshot']['git_commit'] != PREFLIGHT_COMMIT:
        raise ValueError('This campaign requires the reviewed 21835630 preflight report')
    old_reuse = preflight_reuse.authenticate(p, r['source_snapshot'],
        project=preflight_project, prepared_project=prepared_project)
    if (r['parents'] != dict(prepared=p['content_hash'], source=r['source_snapshot']['content_hash'], reuse=old_reuse['content_hash'])
            or load_json(path.parent/'prepared_reuse.json') != old_reuse):
        raise ValueError('Historical preflight/preparation reuse differs')
    if not {'prepared_reuse.json', 'runtime.json'} <= set(r['outputs']):
        raise ValueError('Incomplete technical output binding')
    for name, ref in r['outputs'].items():
        validate_file_ref(dict(path=name, **ref), root=path.parent)
    reuse = compatible_source(r['source_snapshot'], active, preflight_project=preflight_project, project=project)
    admission = artifact('SCIENCE_ADMISSION', parents=dict(prepared=p['content_hash'], preflight=r['content_hash'],
        source=active['content_hash'], reuse=reuse['content_hash']), preflight=file_ref(path),
        source_reuse=reuse, prepared_root=str(Path(prepared_root).resolve()),
        prepared_project=str(Path(prepared_project).resolve()), preflight_project=str(Path(preflight_project).resolve()),
        final_test_accessed=False, imported_models=[], probe_models_used_for_science=False)
    return admission, p, f, r
