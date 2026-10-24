"""One immutable 22-job direct plan, with independently learned branch teachers."""
from pathlib import Path
import math
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_embedding_weight.worker import storage
from .contracts import artifact,validate,recipe,nodes,tasks,COUNTS,ARMS,STAGES
from . import source


def policy(parent,accepted):
    site=execution_site('sporc_a100_debug')
    if accepted['resources']!=dict(execution_site=site,cpus=6,memory_mb=90000):
        raise ValueError('Expected prior debug A100/6CPU/90GB acceptance')
    projection=accepted['preflight']['projections']['aligned']
    if type(projection) is not int or projection<1: raise ValueError('Missing prior measured projection')
    minutes=max(480,math.ceil(1.5*projection)+120)
    if minutes>1440: raise ValueError('Allowance exceeds debug24h; no silent cap')
    return dict(counts=COUNTS,original50k_rows=50000,nodes=nodes(parent),tasks=tasks(),
        recipes={a:{s:recipe(a,s) for s in STAGES} for a in ARMS},execution_site=site,cpus=6,memory_mb=90000,
        task_minutes={r['task_id']:480 if r['task_id']=='targets' else 120 if r['task_id']=='report' else minutes for r in tasks()},
        report_cpus=1,report_memory_mb=16384,preflight_jobs=0,technical_training_epochs=0,
        startup_training_steps=0,new_scientific_fits=20,resource_policy='inherited_token_projection_x1.5_plus120_min8h',
        new_objective_remote_measured=False,storage_budget_bytes=8*2**30,automatic_retry=False,
        target_retirement='own_payloads_only_after_all20_fits_and_saved_report',retain_native_models=True)


def support(anchor):
    m,p,_,_=regular_cache.metadata(anchor['evidence']['prepared_root'])
    if m['content_hash']!=anchor['evidence']['cache'] or p['content_hash']!=anchor['evidence']['prepared']:
        raise ValueError('Prepared lineage differs')
    return {r:{co:p['summaries'][r][co] for co in ('OFFLINE',*STAGES.values())} for r in ('train','validation')}


def check_support(summaries):
    for role in ('train','validation'):
        if set(summaries[role])!=set(('OFFLINE',*STAGES.values())): raise ValueError('Support inventory differs')
        for s in summaries[role].values():
            if type(s['maximum']) is not int or not 1<=s['maximum']<=512: raise ValueError('Invalid support')
            if not 0<s['resident_bytes'] or 2*s['resident_bytes']+2*2**30 > .3*90000*1024**2:
                raise MemoryError('Single-view reconstruction/target footprint exceeds host budget')


def create(root,token_study,project,commit,*,available_quota_gib):
    root,project=Path(root).resolve(),Path(project).resolve()
    snapshot=source.current(project,commit); accepted,parent=source.import_acceptance(token_study)
    anchor=accepted['anchor']
    protected=[project,Path(parent['project_dir']),Path(parent['root']).parent,Path(token_study).resolve().parent,
        Path(anchor['evidence']['prepared_root']),Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh root overlaps protected source/data/results')
    if type(available_quota_gib) is not int or available_quota_gib<8: raise ValueError('Reserve8GiB personally available space')
    summaries=support(anchor); check_support(summaries)
    spec=artifact('SPEC',parents=dict(source=snapshot['content_hash'],acceptance=accepted['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,acceptance=accepted,
        anchor=anchor,support=summaries,available_quota_gib=available_quota_gib,**policy(parent,accepted))
    root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    storage(spec); return spec


def validate_spec(spec,*,imports=True):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],acceptance=spec['acceptance']['content_hash']))
    if imports: source.validate_acceptance(spec['acceptance'])
    # Reconstruct only the common native base, then regenerate every registered node.
    from hlt_classification.luka_fullsim.regular_campaign import nodes as original_nodes
    parent=dict(nodes=original_nodes())
    if (any(spec.get(k)!=v for k,v in policy(parent,spec['acceptance']).items())
            or spec['anchor']!=spec['acceptance']['anchor']
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<8
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Registered science, resources or immutable spec differs')
    check_support(spec['support'])


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:
        raise ValueError('Pinned source changed')
    accepted,parent=source.import_acceptance(spec['acceptance']['spec']['path'])
    if (accepted!=spec['acceptance'] or support(spec['anchor'])!=spec['support']
            or any(spec[k]!=v for k,v in policy(parent,accepted).items())):
        raise ValueError('Original-source evidence changed')
