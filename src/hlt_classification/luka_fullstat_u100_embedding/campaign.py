"""Queue two jobs once, with authenticated measured-resource reuse."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullstat_embedding_sweep.campaign import separate
from . import source
from .contracts import artifact, validate, recipe, node, tasks, COUNTS


def policy(reuse):
    validate(reuse, 'REUSE')
    if (reuse['counts'] != COUNTS or reuse['cpus'] != 6 or reuse['memory_mb'] != 180000
            or reuse['export_minutes'] != 480 or 'L40S' not in reuse['gpu']['name']
            or set(reuse['train_minutes']) != {'w0','w2','w10','w50','w100','w500','w1000'}
            or any(type(v) is not int or not 720 <= v <= 2880 for v in reuse['train_minutes'].values())):
        raise ValueError('Accepted resource reuse is inapplicable')
    if reuse['teacher_node']['node_id'] != 'COARSE_U100' or reuse['teacher_node']['coordinate'] != 'U100':
        raise ValueError('Wrong embedding teacher')
    return dict(anchor=reuse['anchor'], teacher_refs=reuse['teacher_refs'],
        counts=dict(COUNTS), node=node(reuse['base_node']), recipe=recipe(), tasks=tasks(),
        execution_site=reuse['execution_site'], cpus=6, memory_mb=180000,
        export_minutes=480, train_minutes=max(reuse['train_minutes'].values()),
        resource_reuse=True, new_full_preflight=False, automatic_retry=False)


def protected(root, reuse, project):
    separate(root, reuse['anchor'], project)
    for other in (reuse['old_root'], reuse['old_project']):
        a, b = Path(root).resolve(), Path(other).resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise PermissionError('Output overlaps old sweep/source')


def storage(spec):
    if spec['available_result_bytes'] < 6*2**30 or shutil.disk_usage(spec['root']).free < 6*2**30:
        raise OSError('Need 6 GiB personally available durable space; no deletion')


def create(root, previous_gate, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source.current(project, commit)
    reuse = source.authenticate(previous_gate, project)
    protected(root, reuse, project)
    if type(quota_gib) is not int or quota_gib < 6:
        raise ValueError('Declare at least 6 GiB personal durable headroom')
    spec = artifact('SPEC', parents=dict(source=active['content_hash'], reuse=reuse['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        reuse=reuse, available_result_bytes=quota_gib*2**30, **policy(reuse))
    root.mkdir(parents=True, exist_ok=False)
    storage(spec)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reuse=spec['reuse']['content_hash']))
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Execution source changed')
    reuse = source.authenticate(spec['reuse']['gate']['path'], spec['project_dir'])
    if (reuse != spec['reuse'] or any(spec.get(k) != v for k,v in policy(reuse).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < 6*2**30):
        raise ValueError('Reuse/source/recipe registration changed')
    protected(spec['root'], reuse, spec['project_dir'])
    return reuse
