"""One immutable debug DAG; original and concurrent experiments remain untouched."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_embedding_ladder.contracts import validate as validate_anchor
from hlt_classification.luka_fullsim.regular_cache import metadata
from .contracts import artifact, validate, recipe, nodes, stage_node, COUNTS, COORDINATES, VIEWS, MINUTES
from . import source


def registration(parent):
    return dict(counts=COUNTS, original50k_rows=50000, nodes=nodes(parent), recipe=recipe(),
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=90000,
        minutes=MINUTES, storage_budget_bytes=2*2**30, automatic_retry=False)


def support(anchor):
    p = metadata(anchor['evidence']['prepared_root'])[1]
    if p['content_hash'] != anchor['evidence']['prepared']: raise ValueError('Support population differs')
    return {r: {co: p['summaries'][r][co]['maximum'] for co in VIEWS} for r in ('train', 'validation')}


def create(root, parent_path, project, commit, quota_gib):
    project, root = Path(project).resolve(), Path(root).resolve()
    snapshot = source.current(project, commit); anchor, parent = source.authenticate(parent_path)
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
                 Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New output overlaps protected shared artifacts')
    if type(quota_gib) is not int or quota_gib < 2: raise ValueError('Reserve at least 2 GiB personally available space')
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot, anchor=anchor,
        available_quota_gib=quota_gib, support=support(anchor), **registration(parent))
    root.mkdir(parents=True, exist_ok=False); write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec, *, payloads=True):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], anchor=spec['anchor']['content_hash']))
    validate_anchor(spec['anchor'], 'ANCHOR')
    parent = dict(nodes=[dict(n, node_id='COARSE_'+co) for co, n in zip(COORDINATES, spec['nodes'])])
    if (len(spec['nodes']) != 4 or any(spec.get(k) != v for k, v in registration(parent).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 2
            or set(spec['anchor']['evidence']['models']) != {'M0HLT','OFFLINE','U000'}
            or set(spec['support']) != {'train','validation'}):
        raise ValueError('Registered science/resources differ')
    for role in spec['support']:
        if set(spec['support'][role]) != set(VIEWS) or any(type(x) is not int or not 0 < x <= 512 for x in spec['support'][role].values()):
            raise ValueError('Invalid support declaration')
    for co in COORDINATES: stage_node(spec, co)
    if payloads:
        validate_file_ref(spec['anchor']['spec'])
        for row in spec['anchor']['evidence']['models'].values():
            for ref in row.values(): validate_file_ref(ref)


def check(spec):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']):
        raise ValueError('Pinned source/spec changed')
    anchor, parent = source.authenticate(spec['anchor']['spec']['path'])
    if (anchor != spec['anchor'] or support(anchor) != spec['support']
            or any(spec.get(k) != v for k, v in registration(parent).items())):
        raise ValueError('Original-source inputs/seeds changed')
    return spec
