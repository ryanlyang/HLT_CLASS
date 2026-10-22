"""One new immutable debug DAG; existing campaigns remain read-only."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, recipe, node, COUNTS, MINUTES
from . import source


def registration(parent):
    return dict(counts=COUNTS, original50k_rows=50000, node=node(parent), recipe=recipe(),
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=90000,
        minutes=MINUTES, storage_budget_bytes=2*2**30, automatic_retry=False,
        comparison='historical_not_compute_or_teacher_matched')


def create(root, parent_path, reference_path, project, commit, quota_gib):
    project, root = Path(project).resolve(), Path(root).resolve()
    snapshot = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    teacher = source.teacher(anchor)
    reference = source.reference(reference_path, anchor, node(parent))
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(reference_path).resolve().parent.parent, Path(anchor['evidence']['prepared_root']),
        Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New output root overlaps source/shared artifacts')
    if type(quota_gib) is not int or quota_gib < 2:
        raise ValueError('Reserve at least 2 GiB personally available space')
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash'],
        teacher=teacher['content_hash'], reference=reference['content_hash']), root=str(root),
        project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        anchor=anchor, teacher=teacher, reference=reference, available_quota_gib=quota_gib, **registration(parent))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec, *, payloads=True):
    validate(spec, 'SPEC', {k: spec['source_snapshot' if k == 'source' else k]['content_hash']
                          for k in ('source', 'anchor', 'teacher', 'reference')})
    original = dict(spec['node'], node_id='M0HLT', branch='BASELINE')
    original.pop('auxiliary_weight')
    if (any(spec.get(k) != v for k, v in registration(dict(nodes=[original])).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 2):
        raise ValueError('Registered science/resources differ')
    source.validate_imports(spec, payloads=payloads)


def check(spec):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']):
        raise ValueError('Pinned spec/source changed')
    anchor, parent = source.authenticate(spec['anchor']['spec']['path'])
    if (anchor != spec['anchor'] or source.teacher(anchor) != spec['teacher']
            or source.reference(spec['reference']['spec']['path'], anchor, node(parent)) != spec['reference']):
        raise ValueError('Original-source imported artifacts changed')
    return spec
