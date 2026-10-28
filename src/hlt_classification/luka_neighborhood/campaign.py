"""Immutable direct-debug campaign; no standalone preflight."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_embedding_weight.worker import storage
from .contracts import artifact,validate,ARMS,COUNTS,recipe,tasks
from . import source


def policy(parent,quota):
    site=execution_site('sporc_a100_debug')
    if parent['counts']!=COUNTS or parent['execution_site']!=site or (parent['cpus'],parent['memory_mb'])!=(6,90000):
        raise ValueError('Require matched debug depth parent')
    base=parent['nodes']['baseline']
    if base['coordinate']!='D000' or not base['deployable'] or base['context_coordinate'] is not None: raise ValueError('Native HLT baseline required')
    if type(quota)!=int or quota<4: raise ValueError('Reserve at least four GiB; actual target count admission follows')
    return dict(counts=COUNTS,original50k_rows=50000,nodes={a:dict(base,node_id='NEIGHBOR_'+a,arm=a) for a in ARMS},
        recipes={a:recipe(a) for a in ARMS},tasks=tasks(),execution_site=site,cpus=6,memory_mb=90000,
        report_cpus=1,report_memory_mb=16384,task_minutes={r['task_id']:120 if r['task_id']=='report' else 720 for r in tasks()},
        storage_budget_bytes=quota*2**30,available_quota_gib=quota,preflight_jobs=0,scientific_fits=14,teacher_fits=0,
        automatic_retry=False,technical_epochs=0,shared_targets_borrowed=True,automatic_retirement=False,
        inherited_resources_new_objective_unmeasured=True)


def create(root,parent_path,project,commit,*,available_quota_gib):
    root,project=Path(root).resolve(),Path(project).resolve(); snap=source.current(project,commit)
    imported=source.import_parent(parent_path,project); parent=source.validate_import(imported)
    protected=[project,Path(parent['root']),Path(parent['project_dir']),Path(parent['anchor']['evidence']['prepared_root']),
               Path(parent['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected): raise PermissionError('Root overlaps borrowed artifacts')
    spec=artifact('SPEC',parents=dict(source=snap['content_hash'],imported=imported['content_hash']),root=str(root),project_dir=str(project),
        source_commit=commit,source_snapshot=snap,imported=imported,reuse=imported['reuse'],anchor=imported['anchor'],**policy(parent,available_quota_gib))
    root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec); storage(spec)
    return spec


def validate_spec(spec,*,imports=True):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],imported=spec['imported']['content_hash']))
    parent=source.validate_import(spec['imported'],live=imports)
    if (any(spec.get(k)!=v for k,v in policy(parent,spec['available_quota_gib']).items())
            or spec['reuse']!=spec['imported']['reuse'] or spec['anchor']!=spec['imported']['anchor']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec): raise ValueError('Registered policy differs')
    return parent


def check(spec):
    parent=validate_spec(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Source drift')
    return parent
