"""Additive B registration on the existing Group A data and execution foundation."""
from pathlib import Path
import json
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from .contracts import artifact,validate,ORDER
from .structured_contracts import TASKS,recipe,BASE
from . import campaign,inputs,reuse


def original(path):
    path=Path(path).resolve(strict=True); old=load_json(path)
    if old.get('scope')!='group_A_only': raise ValueError('Use the existing Group A spec as the input parent')
    code='''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_next_supervision_hydra.campaign import check
s=load_json(sys.argv[1]); check(s,check_storage=False)
print('GROUP_B_PARENT='+s['content_hash'])
'''
    env=clean_environment(); env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(old['project_dir'])/'src'))
    run=subprocess.run([sys.executable,'-B','-s','-c',code,str(path),old['project_dir']],cwd=old['project_dir'],env=env,capture_output=True,text=True)
    if run.returncode or run.stdout.splitlines().count('GROUP_B_PARENT='+old['content_hash'])!=1:
        raise ValueError('Original Group A input/source authentication failed:\n'+run.stdout+run.stderr)
    return old


def registration():
    result=campaign.registration(); result.update(recipe=recipe(),tasks=[t for t in ORDER if t in TASKS or t in ('C0','C1')],
        scope='group_B_only',execution_version='B1',deferred_groups=['A','C'],sampler_index='structured_Q_oracle_V_bank_v1')
    return result


COPIED=campaign.COPIED+('parent_spec','qualification','raw_root','raw_hash','delta_root','delta_hash','embedding_hash','key_hash')


def create(root,group_a_spec,project,commit,quota_gib):
    old=original(group_a_spec); project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<4: raise ValueError('Reserve four GiB after existing transfers/environment')
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate Group B output root')
    snapshot=campaign.current(project,commit)
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',BASE,commit],check=True,capture_output=True)
    reused={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
    if set(reused)!={'C0','C1'}: raise ValueError('Compatible C0/C1 references are required; do not retrain them')
    spec=dict(registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        environment_dir=str(Path(sys.prefix).resolve()),group_a_spec=file_ref(Path(group_a_spec).resolve()),
        available_quota_gib=quota_gib,reused=reused,remaining=list(TASKS),**{k:old[k] for k in COPIED})
    for proof in reused.values(): reuse.check(proof,imported=spec['imported'])
    inputs.raw(spec)
    value=artifact('STRUCTURED_SPEC',parents=dict(source=snapshot['content_hash'],group_a=old['content_hash'],
        raw=old['raw_hash'],recipe=recipe()['content_hash']),**spec)
    campaign.storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    validate_file_ref(spec['group_a_spec']); old=original(spec['group_a_spec']['path'])
    validate(spec,'STRUCTURED_SPEC',dict(source=spec['source_snapshot']['content_hash'],group_a=old['content_hash'],raw=old['raw_hash'],recipe=recipe()['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=old[k] for k in COPIED)
            or spec['reused']!={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
            or spec['remaining']!=list(TASKS) or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or spec['available_quota_gib']<4
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']):
        raise ValueError('Immutable Group B registration differs')
    for proof in spec['reused'].values(): reuse.check(proof,imported=spec['imported'])
    inputs.raw(spec)
    if check_storage: campaign.storage(spec)
    return load_json(spec['parent_spec']['path'])
