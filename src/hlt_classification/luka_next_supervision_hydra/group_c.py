"""Seven additive C fits, authenticated against the original pinned A input spec."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from .contracts import artifact,validate,ORDER
from .probabilistic_contracts import TASKS,recipe,BASE
from .group_b import original,COPIED
from . import campaign,inputs,reuse


def registration():
    result=campaign.registration(); result.update(recipe=recipe(),tasks=[t for t in ORDER if t in TASKS or t in ('C0','C1')],
        scope='group_C_only',execution_version='C1',deferred_groups=['A','B'],sampler_index='full_set_bank_and_joint_density_v1',
        storage_budget_bytes=8*2**30)
    return result


def create(root,group_a_spec,project,commit,quota_gib):
    old=original(group_a_spec); project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<8: raise ValueError('Reserve eight GiB for C full-set targets and outputs after existing data/environment')
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate Group C output root')
    snapshot=campaign.current(project,commit)
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',BASE,commit],check=True,capture_output=True)
    reused={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
    if set(reused)!={'C0','C1'}: raise ValueError('Compatible C0/C1 references required; do not retrain')
    spec=dict(registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        environment_dir=str(Path(sys.prefix).resolve()),group_a_spec=file_ref(Path(group_a_spec).resolve()),
        available_quota_gib=quota_gib,reused=reused,remaining=list(TASKS),**{k:old[k] for k in COPIED})
    for proof in reused.values(): reuse.check(proof,imported=spec['imported'])
    inputs.raw(spec)
    value=artifact('PROBABILISTIC_SPEC',parents=dict(source=snapshot['content_hash'],group_a=old['content_hash'],
        raw=old['raw_hash'],recipe=recipe()['content_hash']),**spec)
    campaign.storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    validate_file_ref(spec['group_a_spec']); old=original(spec['group_a_spec']['path'])
    validate(spec,'PROBABILISTIC_SPEC',dict(source=spec['source_snapshot']['content_hash'],group_a=old['content_hash'],raw=old['raw_hash'],recipe=recipe()['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=old[k] for k in COPIED)
            or spec['reused']!={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
            or spec['remaining']!=list(TASKS) or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or spec['available_quota_gib']<8 or load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']):
        raise ValueError('Immutable Group C registration differs')
    for proof in spec['reused'].values(): reuse.check(proof,imported=spec['imported'])
    inputs.raw(spec)
    if check_storage: campaign.storage(spec)
    return load_json(spec['parent_spec']['path'])
