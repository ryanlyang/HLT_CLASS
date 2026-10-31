"""Immutable debug registration and explicit storage/admission envelope."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, recipe, sampler_recipe, ARMS, COUNTS
from . import source


def registration():
    return dict(recipe=recipe(), sampler=sampler_recipe(), arms=ARMS, counts=COUNTS,
        original50k_rows=50000, execution_site=execution_site('sporc_a100_debug'),
        cpus=6, memory_mb=120000, storage_budget_bytes=4*2**30,
        setup_minutes=720, fit_max_minutes=1440, report_minutes=120, max_concurrent_fits=2,
        admission='shared_setup_real_miniatures_then_automatic_measured_submission',
        standalone_preflight=False, new_teacher_fits=0, scientific_fits=20, automatic_retry=False,
        exclusion_metadata='historical_RamCache_unique_rows_no_authenticated_event_or_replica_groups')


def storage(spec):
    root = Path(spec['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    base = root if root.exists() else root.parent
    if remaining <= 0 or shutil.disk_usage(base).free < remaining+2**30:
        raise OSError('Four GiB campaign allowance plus one GiB free headroom unavailable; no automatic deletion')
    return used


def create(root, parent, reference, project, commit, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    if root.resolve() != root or root.is_symlink():
        raise PermissionError('No redirected output root')
    if type(quota_gib) is not int or quota_gib < 4:
        raise ValueError('Reserve at least four personally available GiB; the budget stays four GiB')
    snapshot = source.current(project, commit)
    imported = source.import_parent(parent, reference)
    anchor = imported['anchor']
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    recovery_anchors = {}
    for name in ('M0HLT', 'OFFLINE'):
        a = load_npz_arrays(anchor['evidence']['models'][name]['predictions']['path'])
        recovery_anchors[name] = evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc']
    protected = [project, Path(parent).resolve().parent, Path(reference).resolve().parent,
                 Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Campaign overlaps protected source/data/results')
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], imported=imported['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        imported=imported, anchor=anchor, recovery_anchors=recovery_anchors, available_quota_gib=quota_gib, **registration())
    storage(spec)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], imported=spec['imported']['content_hash']))
    if (any(spec.get(k) != v for k, v in registration().items())
            or spec['anchor'] != spec['imported']['anchor'] or spec['available_quota_gib'] < 4
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable campaign registration changed')
    source.validate_import(spec['imported'])
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Pinned source changed')
    if check_storage:
        storage(spec)
    return spec
