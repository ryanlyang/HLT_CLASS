"""Bounded detached launch with a saved dry review, no interactive wait."""
from pathlib import Path
import json
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from .contracts import artifact,TASK,AUTHORIZATION,LAUNCH_AUTHORIZATION
from . import campaign,submission


def run(root,path,project,commit,*,quota_gib,authorization,resume_spec=None):
    if authorization!=LAUNCH_AUTHORIZATION:raise PermissionError('Bounded launch authorization required')
    root=Path(root).resolve();folder=Path(str(root)+'.launch')
    if not folder.is_dir() or root.exists():raise FileExistsError('Fresh detached launch directory required')
    (folder/'process_claim').mkdir(exist_ok=False)
    try:
        intent=artifact('LAUNCH_INTENT',parent_spec=file_ref(Path(path)),source_commit=commit,
            root=str(root),quota_gib=quota_gib,authorization=authorization,jobs=1,teacher_fits=0,preflight_jobs=0,
            resume_spec=None if resume_spec is None else file_ref(Path(resume_spec)))
        write_immutable_json(folder/'intent.json',intent)
        spec=campaign.create(root,path,project,commit,quota_gib=quota_gib,resume_spec=resume_spec)
        plan=submission.submit(spec)
        if plan!=submission.render(spec) or file_ref(Path(path))!=intent['parent_spec']:
            raise ValueError('Reviewed plan/parent changed')
        print(json.dumps(plan,indent=2),flush=True)
        approval=artifact('LAUNCH_APPROVAL',parents=dict(intent=intent['content_hash'],spec=spec['content_hash'],plan=plan['content_hash']),
            authorization=authorization)
        write_immutable_json(folder/'approval.json',approval)
        print('DRY_REVIEW_PASSED: submitting one Multi3 + OFFLINE matching fit; no teacher retraining.',flush=True)
        submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=AUTHORIZATION)
        jobs=submission.accepted_jobs(spec)
        value=artifact('LAUNCH_COMPLETE',parents=dict(approval=approval['content_hash']),jobs=jobs)
        write_immutable_json(folder/'complete.json',value)
        print('LAUNCH_COMPLETE: '+json.dumps(jobs,sort_keys=True),flush=True)
        return value
    except Exception as exc:
        write_immutable_json(folder/'failure.json',artifact('FAILURE',error_type=type(exc).__name__,message=str(exc)))
        print('STOPPED: inspect launch failure and submission journals; do not repeat or delete claims.',flush=True)
        raise
