"""Detached bounded authorization, full dry review, exact DAG submission."""
from pathlib import Path
import json
from hlt_classification.operations import fullstat_pt_metadata as meta
from . import contracts as c,campaign,submission,recovery


def run(root,parent,project,commit,quota_gib,authorization,*,recover=False):
    if authorization!=c.LAUNCH_AUTHORIZATION:raise PermissionError('Bounded launch authorization required')
    root=Path(root).resolve();folder=Path(str(root)+'.launch')
    if not folder.is_dir() or root.exists():raise FileExistsError('Fresh detached launch directory required')
    (folder/'process_claim').mkdir(exist_ok=False)
    try:
        intent=c.artifact('LAUNCH_INTENT',parent_spec=meta.file_ref(parent),root=str(root),source_commit=commit,
            quota_gib=quota_gib,authorization=authorization,mode='recovery' if recover else 'six_fresh_fits',preflight_jobs=0)
        meta.write_immutable_json(folder/'intent.json',intent)
        fn=recovery.create if recover else campaign.create
        spec=fn(root,parent,project,commit,quota_gib)
        plan=submission.submit(spec)
        if plan!=submission.render(spec) or meta.file_ref(parent)!=intent['parent_spec']:raise ValueError('Reviewed plan/input changed')
        print(json.dumps(plan,indent=2),flush=True)
        approval=c.artifact('LAUNCH_APPROVAL',parents=dict(intent=intent['content_hash'],spec=spec['content_hash'],plan=plan['content_hash']),
            authorization=authorization)
        meta.write_immutable_json(folder/'approval.json',approval)
        print('DRY_REVIEW_PASSED: submitting exact balanced campaign DAG.',flush=True)
        submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
        value=c.artifact('LAUNCH_COMPLETE',parents=dict(approval=approval['content_hash']),jobs=submission.accepted_jobs(spec))
        meta.write_immutable_json(folder/'complete.json',value)
        print('LAUNCH_COMPLETE: '+json.dumps(value['jobs'],sort_keys=True),flush=True)
        return value
    except Exception as exc:
        meta.write_immutable_json(folder/'failure.json',c.artifact('FAILURE',error_type=type(exc).__name__,message=str(exc)))
        print('STOPPED: preserve launch records and submission journal; do not relaunch or delete claims.',flush=True)
        raise
