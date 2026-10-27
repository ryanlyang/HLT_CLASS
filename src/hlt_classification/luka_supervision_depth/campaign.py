"""Fresh immutable direct-debug specification, isolated from the finished grid."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_embedding_weight.worker import storage
from .contracts import artifact,validate,COUNTS,ARMS,TEACHER_TASKS,recipe,teacher_recipe,nodes,tasks
from . import source


def policy(parent):
    site=execution_site('sporc_a100_debug')
    if parent['counts']!=COUNTS or parent['execution_site']!=site or (parent['cpus'],parent['memory_mb'])!=(6,90000):
        raise ValueError('Matched debug parent required')
    inherited=parent['task_minutes']['c50_DE000']
    if type(inherited)!=int or not 1<=inherited<=1440: raise ValueError('Invalid inherited time allowance')
    return dict(counts=COUNTS,original50k_rows=50000,nodes=nodes(parent),tasks=tasks(),
        recipes={a:recipe(a) for a in ARMS},teacher_recipes={t:teacher_recipe(t) for t in TEACHER_TASKS},
        execution_site=site,cpus=6,memory_mb=90000,report_cpus=1,report_memory_mb=16384,
        task_minutes={r['task_id']:120 if r['task_id'] in ('report','cleanup') else max(720,inherited) for r in tasks()},
        storage_budget_bytes=8*2**30,scratch_base='/tmp',preflight_jobs=0,scientific_fits=20,teacher_fits=2,
        technical_epochs=0,startup_steps=1,startup_batch=128,automatic_retry=False,
        inherited_resources_new_objective_unmeasured=True,native_models_retained=True)


def create(root,parent_path,project,commit,*,available_quota_gib):
    root,project=Path(root).resolve(),Path(project).resolve(); snap=source.current(project,commit)
    imported=source.import_parent(parent_path,project); parent=source.validate_import(imported)
    protected=[project,Path(parent['project_dir']),Path(parent['root']),Path(imported['spec']['path']).parent,
        Path(parent['anchor']['evidence']['prepared_root']),Path(parent['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected): raise PermissionError('New root overlaps protected artifacts')
    if type(available_quota_gib)!=int or available_quota_gib<8: raise ValueError('Reserve eight personally available GiB')
    spec=artifact('SPEC',parents=dict(source=snap['content_hash'],imported=imported['content_hash']),root=str(root),
        project_dir=str(project),source_commit=commit,source_snapshot=snap,imported=imported,
        reuse=imported['reuse'],anchor=imported['anchor'],available_quota_gib=available_quota_gib,**policy(parent))
    root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec); storage(spec)
    return spec


def validate_spec(spec,*,imports=True):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],imported=spec['imported']['content_hash']))
    parent=source.validate_import(spec['imported'],live=imports)
    if (any(spec.get(k)!=v for k,v in policy(parent).items()) or spec['reuse']!=spec['imported']['reuse']
            or spec['anchor']!=spec['imported']['anchor'] or type(spec['available_quota_gib'])!=int or spec['available_quota_gib']<8
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec): raise ValueError('Registered policy differs')


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Pinned source changed')
