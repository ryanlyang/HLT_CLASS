"""Direct tier3 registration; no new preflight or technical training."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, COUNTS, ARMS, node, recipe, tasks
from . import source


def policy(parent, resources):
    minutes=resources['measured_w5_minutes']
    if type(minutes) is not int or minutes<1 or max(360,2*minutes+60)>1440:
        raise ValueError('Inherited request cannot support registered <=24h estimate')
    return dict(counts=COUNTS,original50k_rows=50000,nodes={a:node(parent,a) for a in ARMS},
        recipes={a:recipe(a) for a in ARMS},tasks=tasks(),execution_site=execution_site('sporc_a100'),
        cpus=6,memory_mb=90000,export_minutes=360,train_minutes=max(360,2*minutes+60),
        preflight_jobs=0,technical_training_epochs=0,scientific_fits=5,automatic_retry=False,
        resource_estimate='twice_inherited_single_head_request_plus60min_min6h_not_new_measurement',
        storage_budget_bytes=4*2**30)


def parents(spec):
    return {k:spec[k]['content_hash'] for k in ('source_snapshot','anchor','resources','reference')} | {
        'teachers':artifact('TEACHER_SET',teachers=spec['teachers'])['content_hash'],
        'u100_reference':None if spec['u100_reference'] is None else spec['u100_reference']['content_hash']}


def create(root,parent_path,attention_path,reference_path,project,commit,*,available_quota_gib,u100_reference=None):
    root,project=Path(root).resolve(),Path(project).resolve()
    snapshot=source.current(project,commit)
    anchor,parent=source.authenticate(parent_path)
    teachers=source.teachers(anchor,attention_path)
    ref=source.reference(reference_path,anchor,source.donor_node(parent))
    resources=source.resources(ref,anchor,source.donor_node(parent))
    uref=source.optional_u100(u100_reference,anchor,parent)
    if type(available_quota_gib) is not int or available_quota_gib<4:
        raise ValueError('Reserve at least4GiB personally available persistent storage')
    protected=[project,Path(parent['project_dir']),Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']),Path(anchor['evidence']['raw_container'])]
    protected += [Path(p).resolve().parent.parent for p in (attention_path,reference_path)]
    if u100_reference: protected.append(Path(u100_reference).resolve().parent)
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New root overlaps shared source/data/campaign')
    fields=dict(root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        anchor=anchor,teachers=teachers,reference=ref,resources=resources,u100_reference=uref,
        available_quota_gib=available_quota_gib,**policy(parent,resources))
    spec=artifact('SPEC',parents=parents(fields),**fields)
    source.imports(spec)
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    storage(spec)
    return spec


def validate_spec(spec):
    validate(spec,'SPEC',parents(spec))
    parent=load_json(spec['anchor']['spec']['path'])
    if (any(spec.get(k)!=v for k,v in policy(parent,spec['resources']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<4
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Five-arm direct registration differs')


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:
        raise ValueError('Pinned source changed')
    source.imports(spec)


def storage(spec):
    root=Path(spec['root'])
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(root).free<remaining:
        raise OSError('Registered4GiB allowance unavailable; no automatic deletion')
    return used
