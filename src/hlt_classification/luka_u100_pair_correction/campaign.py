"""Immutable four-fit debug registration with bounded resource admission."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import ARMS, COUNTS, artifact, validate, recipe
from . import source


def registration():
    return dict(recipe=recipe(), arms=ARMS, counts=COUNTS, original50k_rows=50000,
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=120000,
        storage_budget_bytes=8*2**30, setup_minutes=720, fit_max_minutes=1440, report_minutes=120,
        max_concurrent_fits=2, scientific_fits=4, new_teacher_fits=0, reference_fits=0,
        standalone_preflight=False, admission='shared_setup_then_automatic_measured_four_fits', automatic_retry=False)


def storage(spec):
    root = Path(spec['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(root if root.exists() else root.parent).free < remaining+2**30:
        raise OSError('Eight GiB campaign allowance plus one GiB headroom unavailable; no automatic deletion')
    return used


def create(root, reference, project, commit, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    if root.resolve() != root or root.is_symlink():
        raise PermissionError('Redirected output root forbidden')
    if type(quota_gib) is not int or quota_gib < 8:
        raise ValueError('Reserve at least eight personally available GiB; campaign budget remains eight')
    snapshot = source.current(project, commit)
    control = source.import_control(reference); imported = control['imported']; anchor = imported['anchor']
    protected = [project, Path(reference).resolve().parent, Path(anchor['evidence']['prepared_root']),
                 Path(anchor['evidence']['raw_container']), Path(control['u100_manifest']['path']).parent]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Output overlaps protected source/data/reference')
    anchors = {}
    for name in ('M0HLT', 'OFFLINE'):
        a = load_npz_arrays(anchor['evidence']['models'][name]['predictions']['path'])
        anchors[name] = evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc']
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], reference=control['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        reference=control, imported=imported, anchor=anchor, recovery_anchors=anchors, available_quota_gib=quota_gib, **registration())
    storage(spec); root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reference=spec['reference']['content_hash']))
    if (any(spec.get(k) != v for k, v in registration().items()) or spec['available_quota_gib'] < 8
            or spec['imported'] != spec['reference']['imported'] or spec['anchor'] != spec['imported']['anchor']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable campaign registration differs')
    source.validate_reference(spec['reference'])
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Pinned source changed')
    if check_storage: storage(spec)
    return spec
