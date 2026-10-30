"""Two-stage admission: measured debug acceptance before any science submission."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, recipe, COUNTS, TASKS
from . import source


def storage(spec):
    root = Path(spec['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(root if root.exists() else root.parent).free < remaining + 2**30:
        raise OSError('Four GiB campaign allowance plus one GiB filesystem headroom unavailable')
    return used


def registration():
    return dict(recipe=recipe(), counts=COUNTS, original50k_rows=50000, tasks=TASKS,
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=120000,
        storage_budget_bytes=4*2**30, acceptance_minutes=720,
        admission='measured_debug_acceptance_before_science_submission', automatic_retry=False)


def create(root, parent_path, reference_path, project, commit, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    if root.resolve() != root or root.is_symlink():
        raise PermissionError('No redirected campaign root')
    snapshot = source.current(project, commit)
    imported = source.import_parent(parent_path, reference_path)
    anchor = imported['anchor']
    protected = [project, Path(parent_path).resolve().parent, Path(reference_path).resolve().parent,
                 Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Campaign root overlaps protected data/source/results')
    if type(quota_gib) is not int or quota_gib < 4:
        raise ValueError('Reserve at least four personally available GiB')
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], imported=imported['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        imported=imported, anchor=anchor, quota_gib=quota_gib, **registration())
    storage(spec)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], imported=spec['imported']['content_hash']))
    if (any(spec.get(k) != v for k, v in registration().items()) or spec['anchor'] != spec['imported']['anchor']
            or type(spec['quota_gib']) is not int or spec['quota_gib'] < 4
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable campaign registration differs')
    source.validate_import(spec['imported'])
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source drift')
    storage(spec)
    return spec
