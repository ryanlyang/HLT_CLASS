"""Immutable two-job direct registration; no new technical training epoch."""
from pathlib import Path
import math
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_particle_prediction.campaign import storage
from .contracts import artifact, validate, recipe, node, tasks, ARMS, COUNTS
from . import source


def policy(parent, acceptance):
    pre = acceptance['preflight']
    minutes = {a:max(240,math.ceil(1.25*pre['projections'][a])+30) for a in ARMS}
    if max(minutes.values())>1440: raise ValueError('Inherited allowance exceeds debug24h; do not silently cap')
    site = execution_site('sporc_a100_debug')
    if acceptance['resources'] != dict(execution_site=site,cpus=6,memory_mb=90000):
        raise ValueError('Expected existing debug/A100/6CPU/90GB acceptance')
    return dict(counts=COUNTS,original50k_rows=50000,recipe=recipe(),nodes={a:node(parent,a) for a in ARMS},
        tasks=tasks(),execution_site=site,cpus=6,memory_mb=90000,train_minutes=minutes,
        preflight_jobs=0,technical_training_epochs=0,resource_policy='prior_token_projection_times1.25_plus30min_min4h',
        combined_objective_remote_measured=False,storage_budget_bytes=2*2**30,
        scratch_base='/tmp',scratch_policy='verified_node_local_disk_unique_owned_per_job_no_home_fallback',automatic_retry=False)


def parents(spec):
    return dict(source=spec['source_snapshot']['content_hash'],acceptance=spec['acceptance']['content_hash'])


def create(root, token_study, project, commit, *, available_quota_gib):
    root,project = Path(root).resolve(),Path(project).resolve()
    snapshot = source.current(project,commit); accepted,parent = source.import_acceptance(token_study)
    protected = [project,Path(parent['root']).parent,Path(parent['project_dir']),Path(token_study).resolve().parent,
        Path(accepted['reference']['spec']['path']).parent,Path(accepted['anchor']['evidence']['prepared_root']),
        Path(accepted['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh root overlaps shared artifacts')
    if type(available_quota_gib) is not int or available_quota_gib<2:
        raise ValueError('Reserve at least2GiB personally available persistent space')
    fields = dict(root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        acceptance=accepted,reference=accepted['reference'],anchor=accepted['anchor'],teacher=accepted['teacher'],
        available_quota_gib=available_quota_gib,**policy(parent,accepted))
    spec = artifact('SPEC',parents=parents(fields),**fields)
    root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    storage(spec); return spec


def validate_spec(spec):
    validate(spec,'SPEC',parents(spec)); source.validate_acceptance(spec['acceptance'])
    parent = load_json(spec['anchor']['spec']['path'])
    if (any(spec.get(k)!=v for k,v in policy(parent,spec['acceptance']).items())
            or any(spec[k]!=spec['acceptance'][k] for k in ('anchor','teacher','reference'))
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<2
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Combined direct debug registration differs')


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:
        raise ValueError('Source changed')
    accepted,_=source.import_acceptance(spec['acceptance']['spec']['path'])
    if accepted!=spec['acceptance']: raise ValueError('Prior evidence changed')
