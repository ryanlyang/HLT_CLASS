"""One new fit, automatically dependent on authenticated target setup/admission."""
from pathlib import Path
import math
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullstat_embedding_sweep.campaign import separate
from . import source
from .contracts import artifact, validate, recipe, node, tasks, TEACHERS, COUNTS


def policy(reuse):
    validate(reuse, 'REUSE')
    if (reuse['counts'] != COUNTS or 'L40S' not in reuse['gpu']['name']
            or set(reuse['teachers']) != set(TEACHERS)
            or set(reuse['reuse_banks']) != {'OFFLINE', 'COARSE_U100'}
            or type(reuse['parent_train_minutes']) is not int
            or not 720 <= reuse['parent_train_minutes'] <= 2880):
        raise ValueError('Inapplicable original FullStat evidence')
    for teacher, coordinate in TEACHERS.items():
        n = reuse['teachers'][teacher]['node']
        if n['node_id'] != teacher or n['coordinate'] != coordinate or n.get('context_coordinate') is not None:
            raise ValueError('Wrong teacher coordinate or fused teacher')
    return dict(anchor=reuse['anchor'], teachers=reuse['teachers'], reuse_banks=reuse['reuse_banks'],
        reference=reuse['reference'], counts=dict(COUNTS), node=node(reuse['base_node']),
        recipe=recipe(), tasks=tasks(), execution_site=reuse['execution_site'],
        cpus=6, memory_mb=180000, setup_minutes=720,
        train_minutes=min(2880, math.ceil(1.5*reuse['parent_train_minutes'])),
        acceptance=dict(stress_steps=3, benchmark_steps=512, full_epoch_training_measured=False,
            full_validation_measured=True, margin=1.5, fixed_allowance_minutes=90),
        new_scientific_fits=1, automatic_dependency=True, automatic_retry=False)


def protected(root, reuse, project):
    separate(root, reuse['anchor'], project)
    for other in reuse['protected_roots']:
        a, b = Path(root).resolve(), Path(other).resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise PermissionError('Output overlaps an existing campaign or worktree')


def storage(spec):
    if spec['available_result_bytes'] < 12*2**30 or shutil.disk_usage(spec['root']).free < 12*2**30:
        raise OSError('Need 12 GiB personally available durable headroom; no deletion')


def create(root, previous_spec, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source.current(project, commit)
    reuse = source.authenticate(previous_spec, project)
    protected(root, reuse, project)
    if type(quota_gib) is not int or quota_gib < 12:
        raise ValueError('Declare at least 12 GiB personal durable headroom')
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
    reuse = source.authenticate(spec['reuse']['parent_spec']['path'], spec['project_dir'])
    if (reuse != spec['reuse'] or any(spec.get(k) != v for k, v in policy(reuse).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < 12*2**30):
        raise ValueError('Source, bank, recipe or reference registration changed')
    protected(spec['root'], reuse, spec['project_dir'])
    return reuse
