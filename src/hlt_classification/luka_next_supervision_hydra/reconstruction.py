"""Twelve additive direct reconstruction fits, using the authenticated A inputs."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from .reconstruction_contracts import TASKS,SCOPE,recipe,BASE,artifact,validate
from .group_b import original,COPIED
from . import campaign,inputs,reuse,reconstruction_targets as targets


def registration():
    result=campaign.registration(); result.pop('storage_budget_bytes')
    result.update(recipe=recipe(),tasks=[*TASKS,'C0','C1'],global_order=list(TASKS),scope=SCOPE,
        execution_version='R3',reconstruction_kernel='batched_exact_chunk_v1',assignment_workers=6,
        deferred_groups=['A','B','C'],sampler_index='direct_complete_measured_set_v1',candidate_workers=0,
        release_policy='slurm_afterok_measured_independent_v1',release_minutes=60,
        cpu_site=dict(partition='compute',account='default',qos='normal',cpus=1,memory_mb=4000))
    return result


def create(root,group_a_spec,project,commit,quota_gib):
    old=original(group_a_spec); project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<1: raise ValueError('State personally available output GiB after existing inputs/environment')
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate reconstruction output root')
    snapshot=campaign.current(project,commit)
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',BASE,commit],check=True,capture_output=True)
    reused={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
    if set(reused)!={'C0','C1'}: raise ValueError('Authenticated C0/C1 references required; do not retrain')
    spec=dict(registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        environment_dir=str(Path(sys.prefix).resolve()),group_a_spec=file_ref(Path(group_a_spec).resolve()),
        available_quota_gib=quota_gib,reused=reused,remaining=list(TASKS),**{k:old[k] for k in COPIED})
    for proof in reused.values(): reuse.check(proof,imported=spec['imported'])
    print('DIRECT_RECONSTRUCTION authenticating full raw TRAIN storage/count inventory',flush=True)
    spec['target_inventory']=targets.inventory(spec); spec['storage_budget_bytes']=spec['target_inventory']['storage_budget_bytes']
    if quota_gib*2**30<spec['storage_budget_bytes']+2**30:
        raise OSError(f"Reconstruction needs {spec['storage_budget_bytes']/2**30:.2f} GiB plus one GiB headroom; no root or jobs created")
    value=artifact('SPEC',parents=dict(source=snapshot['content_hash'],group_a=old['content_hash'],raw=old['raw_hash'],
        inventory=spec['target_inventory']['content_hash'],recipe=recipe()['content_hash']),**spec)
    campaign.storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    if spec.get('campaign_kind')=='R8_FOLLOWUP_V2':
        from .reconstruction_followup import check as followup
        return followup(spec,check_storage=check_storage)
    validate_file_ref(spec['group_a_spec']); old=original(spec['group_a_spec']['path'])
    inv=spec['target_inventory']; validate(inv,'INVENTORY',dict(raw=spec['raw_hash']))
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],group_a=old['content_hash'],raw=old['raw_hash'],
        inventory=inv['content_hash'],recipe=recipe()['content_hash']))
    if (set(inv['sources'])!={'HLT','OFFLINE'} or inv['maximum']!=max(1,*(s['maximum_count'] for s in inv['sources'].values()))
            or any(s['rows']!=spec['counts']['train'] or s['tokens']<s['rows'] or s['maximum_count']<1 for s in inv['sources'].values())
            or inv['storage_budget_bytes']!=targets.storage_bytes(inv['sources'],spec['counts']['train'])
            or spec['storage_budget_bytes']!=inv['storage_budget_bytes'] or inv['quality_gate'] is not False
            or any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=old[k] for k in COPIED)
            or spec['reused']!={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
            or spec['remaining']!=list(TASKS) or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or spec['available_quota_gib']*2**30<spec['storage_budget_bytes']+2**30
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']):
        raise ValueError('Immutable direct reconstruction registration differs')
    for proof in spec['reused'].values(): reuse.check(proof,imported=spec['imported'])
    inputs.raw(spec)
    if check_storage: campaign.storage(spec)
    return load_json(spec['parent_spec']['path'])
