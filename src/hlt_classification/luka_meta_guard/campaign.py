"""New immutable root; a fixed four-GiB budget, not all declared free space."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_meta_embedding.campaign import storage
from .contracts import artifact, validate, recipe, COUNTS, MINUTES
from . import source


def registration(parent):
    return dict(counts=COUNTS,original50k_rows=50000,node=parent['node'],recipe=recipe(),
        execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000,
        minutes=MINUTES,storage_budget_bytes=4*2**30,automatic_retry=False,
        comparison='completed_FIXED_low_rate_guarded_fixed_vs_meta')


def create(root, parent_path, project, commit, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    if root.resolve()!=root or root.is_symlink(): raise PermissionError('No redirected campaign root')
    snapshot = source.current(project,commit); parent = source.authenticate(parent_path)
    if parent['counts'] != COUNTS: raise ValueError('Original fixed30 populations required')
    if type(quota_gib) is not int or quota_gib < 4: raise ValueError('Reserve at least four personally available GiB')
    old = load_json(parent['spec']['path'])
    protected = [project,Path(old['root']),Path(old['project_dir']),
        Path(parent['anchor']['evidence']['prepared_root']),Path(parent['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New campaign overlaps protected inputs/source')
    spec = artifact('SPEC',parents=dict(source=snapshot['content_hash'],parent=parent['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        parent=parent,anchor=parent['anchor'],available_quota_gib=quota_gib,**registration(parent))
    storage(spec); root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def validate_spec(spec):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],parent=spec['parent']['content_hash']))
    source.validate_parent(spec['parent'])
    if (spec['anchor'] != spec['parent']['anchor']
            or any(spec.get(k)!=v for k,v in registration(spec['parent']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<4):
        raise ValueError('Registered guarded recipe/parent/resources differ')


def check(spec):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or source.authenticate(spec['parent']['spec']['path'])!=spec['parent']):
        raise ValueError('Pinned source/spec/original evidence changed')
    return spec
