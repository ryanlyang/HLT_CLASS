"""Explicit post-completion retirement of three owned descriptor payloads only."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from .contracts import artifact,validate,FAMILIES
from . import campaign,submission

AUTHORIZATION='DELETE_ONLY_COMPLETED_NEIGHBORHOOD_PAYLOADS_V1'


def names(): return ['targets/neighborhoods/'+f+'.npy' for f in FAMILIES]


def retired_evidence(spec):
    root=Path(spec['root']); folder=root/'retirement'
    if not folder.exists(): return None
    if not (folder/'complete.json').is_file(): raise ValueError('Partial retirement; preserve intent, no automatic retry')
    plan=load_json(folder/'intent.json'); done=load_json(folder/'complete.json')
    validate(plan,'RETIREMENT_PLAN',dict(spec=spec['content_hash']))
    validate(done,'RETIRED',dict(spec=spec['content_hash'],plan=plan['content_hash']))
    if [r['path'] for r in plan['payloads']]!=names() or done['results']!=plan['results']: raise ValueError('Retirement scope differs')
    validate_file_ref(plan['results'],root=root)
    for ref in plan['payloads']:
        p=root/ref['path']
        if p.exists() or p.is_symlink(): raise ValueError('Retired payload unexpectedly present')
    return plan


def plan(spec):
    from . import worker
    campaign.check(spec); root=Path(spec['root'])
    if root!=root.resolve(strict=True) or root.is_symlink(): raise PermissionError('Noncanonical campaign root')
    if (root/'retirement').exists(): raise FileExistsError('Retirement already started')
    value=worker.results(spec)
    if value['pending'] or load_json(root/'report/results.json')!=value: raise ValueError('All rows and final report required')
    row=worker.validate_task(spec,'report'); submission.successful(spec,'report',row['job_id'])
    jobs=submission.accepted_jobs(spec)
    active=subprocess.run(['squeue','-h','-j',','.join(jobs.values()),'-o','%i'],check=True,capture_output=True,text=True).stdout.strip()
    if active: raise ValueError('Campaign jobs still active')
    refs=[]
    for name in names():
        p=root/name
        if p.is_symlink() or p!=p.resolve(strict=True) or not p.is_relative_to(root) or p.stat().st_nlink!=1:
            raise PermissionError('Payload is not exclusively owned')
        refs.append(file_ref(p,root=root))
    return artifact('RETIREMENT_PLAN',parents=dict(spec=spec['content_hash']),payloads=refs,
        bytes=sum(r['bytes'] for r in refs),results=file_ref(root/'report/results.json',root=root),authorization=AUTHORIZATION)


def retire(spec,*,plan_hash,authorization):
    value=plan(spec)
    if plan_hash!=value['content_hash'] or authorization!=AUTHORIZATION: raise PermissionError('Exact retirement review required')
    root=Path(spec['root']); folder=root/'retirement'; folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'intent.json',value)
    for ref in value['payloads']:
        p=root/ref['path']
        if p.is_symlink() or p!=p.resolve(strict=True) or p.stat().st_nlink!=1: raise PermissionError('Ownership changed')
        validate_file_ref(ref,root=root); p.unlink()
    done=artifact('RETIRED',parents=dict(spec=spec['content_hash'],plan=value['content_hash']),results=value['results'])
    write_immutable_json(folder/'complete.json',done)
    from .worker import results
    if results(spec)!=load_json(root/'report/results.json'): raise ValueError('Results changed after retirement')
    print('Removed exactly three owned neighborhood payloads permanently; no backup. All models/results and borrowed data retained.')
    return done
