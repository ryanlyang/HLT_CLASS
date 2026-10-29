"""Fresh root, exact debug recipe, no changes to historical campaigns."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact,validate,recipe,node,COUNTS,MINUTES
from . import source


def registration(parent):
    return dict(counts=COUNTS,original50k_rows=50000,node=node(parent),recipe=recipe(),
        execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000,
        minutes=MINUTES,storage_budget_bytes=4*2**30,automatic_retry=False,
        comparison='selected_OFFLINE_w5_fixed_vs_student_benefit_continuation')


def storage(spec):
    root=Path(spec['root']); existing=root if root.exists() else root.parent
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=spec['storage_budget_bytes']-used
    if remaining<=0 or shutil.disk_usage(existing).free<remaining:
        raise OSError('Registered 4 GiB allowance unavailable; nothing automatically deleted')
    return used


def create(root,parent_path,reference_path,project,commit,quota_gib):
    root,project=Path(root).absolute(),Path(project).resolve()
    if root.resolve()!=root or root.is_symlink(): raise PermissionError('No redirected campaign root')
    snapshot=source.current(project,commit); anchor,parent=source.authenticate(parent_path)
    ref=source.reference(reference_path,anchor)
    protected=[project,Path(parent['project_dir']),Path(parent['root']).parent,
        Path(reference_path).resolve().parent.parent,Path(anchor['evidence']['prepared_root']),
        Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New root overlaps shared artifacts')
    if type(quota_gib) is not int or quota_gib<4: raise ValueError('Reserve at least4 personally available GiB')
    spec=artifact('SPEC',parents=dict(source=snapshot['content_hash'],anchor=anchor['content_hash'],reference=ref['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        anchor=anchor,reference=ref,initial_checkpoint=ref['files']['checkpoint'],available_quota_gib=quota_gib,
        **registration(parent))
    # Check BEFORE claiming root; user's free-space declaration is not the budget.
    storage(spec); root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def validate_spec(spec):
    validate(spec,'SPEC',{k:spec['source_snapshot' if k=='source' else k]['content_hash'] for k in ('source','anchor','reference')})
    n=dict(spec['node'],node_id='M0HLT',branch='BASELINE'); n.pop('auxiliary_weight')
    if any(spec.get(k)!=v for k,v in registration(dict(nodes=[n])).items()):
        raise ValueError('Registered science/resource recipe differs')
    if type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<4:
        raise ValueError('Invalid personal storage declaration')
    source.validate_imports(spec)


def check(spec):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']):
        raise ValueError('Pinned spec/source differs')
    anchor,_=source.authenticate(spec['anchor']['spec']['path'])
    if anchor!=spec['anchor'] or source.reference(spec['reference']['spec']['path'],anchor)!=spec['reference']:
        raise ValueError('Original replay evidence changed')
    return spec
