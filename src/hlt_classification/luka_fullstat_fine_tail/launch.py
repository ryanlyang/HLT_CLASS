"""One bounded authorization: authenticate, save dry plan, submit exactly three fits.

The shell launcher owns detachment and an exclusive adjacent launch directory.
This process never retries, polls for another campaign, or modifies old jobs.
"""
from pathlib import Path
import json
import os
import socket
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from . import campaign, submission, source
from .contracts import artifact, AUTHORIZATION, LAUNCH_AUTHORIZATION, CONFIGURATION, STAGES, recipe


def run(root, previous_spec, project, commit, *, quota_gib, authorization):
    if authorization != LAUNCH_AUTHORIZATION:
        raise PermissionError('Explicit authorization for automatic three-fit submission required')
    root, project, previous_spec = map(lambda p: Path(p).resolve(), (root, project, previous_spec))
    directory = Path(str(root)+'.launch')
    if not directory.is_dir() or root.exists():
        raise FileExistsError('Fresh exclusive launch directory and absent campaign root required')
    # Claim first, so even failure during authentication is not silently retried.
    (directory/'process_claim').mkdir(exist_ok=False)
    try:
        intent = artifact('LAUNCH_INTENT', project_dir=str(project), source_commit=commit,
            root=str(root), parent_spec=file_ref(previous_spec), quota_gib=quota_gib,
            runtime_prefix=source.RUNTIME, authorization=authorization,
            recipes={s:recipe(s) for s in STAGES}, jobs=3, preflight_jobs=0,
            host=socket.gethostname(), pid=os.getpid(), automatic_retry=False)
        write_immutable_json(directory/'intent.json', intent)
        print('STARTED: '+CONFIGURATION+'; authenticate completed parent, save full dry plan, then submit exactly three fits.', flush=True)
        spec = campaign.create(root, previous_spec, project, commit, quota_gib=quota_gib)
        print('Authenticated parent and exact pushed source. Running complete dry review...', flush=True)
        plan = submission.submit(spec)
        if (plan != submission.render(spec) or plan['jobs'] != 3 or plan['scientific_fits'] != 3
                or plan['preflight_jobs'] != 0 or plan['setup_jobs'] != 0
                or spec['recipes'] != intent['recipes']
                or [r['task_id'] for r in plan['commands']] != ['train_'+s for s in STAGES]
                or file_ref(previous_spec) != intent['parent_spec']):
            raise ValueError('Dry plan exceeds the explicitly authorized three-fit intent')
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), flush=True)
        approval = artifact('LAUNCH_APPROVAL', parents=dict(intent=intent['content_hash'],
            spec=spec['content_hash'], plan=plan['content_hash']), authorization=authorization,
            full_dry_review_saved=True, bounded_automatic_authorization=True)
        write_immutable_json(directory/'approval.json', approval)
        print('DRY_REVIEW_PASSED: plan '+plan['content_hash']+'; submitting the authorized three fits.', flush=True)
        submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=AUTHORIZATION)
        jobs = submission.accepted_jobs(spec)
        if set(jobs) != {'train_'+s for s in STAGES} or len(set(jobs.values())) != 3:
            raise ValueError('Incomplete/ambiguous accepted job ledger; inspect, do not retry')
        result = artifact('LAUNCH_COMPLETE', parents=dict(approval=approval['content_hash']), jobs=jobs,
            submission_ledger=file_ref(root/'submission/submission_ledger.json'))
        write_immutable_json(directory/'complete.json', result)
        print('SUBMITTED: '+json.dumps(jobs, sort_keys=True), flush=True)
        print('LAUNCH_COMPLETE: Slurm owns the dependency chain. No login controller remains.', flush=True)
        return result
    except Exception as exc:
        write_immutable_json(directory/'failure.json', artifact('LAUNCH_FAILURE',
            error_type=type(exc).__name__, message=str(exc), automatic_retry=False))
        print('STOPPED: inspect launch failure.json and submission journals; do not rerun or delete claims.', flush=True)
        raise
