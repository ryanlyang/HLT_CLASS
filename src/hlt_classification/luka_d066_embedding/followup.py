"""One authenticated gate then one fresh fit; optional owned pending-job holds."""
from contextlib import contextmanager
from pathlib import Path
import getpass
import json
import os
import re
import socket
import subprocess
import time
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from . import campaign, submission, priority
from .contracts import artifact, AUTHORIZATION as SUBMIT_AUTHORIZATION

AUTHORIZATION = 'AUTHORIZE_LUKA_D066_EMBEDDING_FOLLOWUP_V1'
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD'}
WAIT_SECONDS = 14*86400
POLL_SECONDS = 60


@contextmanager
def controller_lock(gate):
    # Host crash releases this advisory lock; durable submit/hold intents remain.
    # Recovery cannot race a still-live controller, including on another host.
    import fcntl
    path=Path(gate['root'])/'followup/controller.lock'
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def ceiling_spec(gate):
    return dict(content_hash='${MEASURED_SCIENCE_SPEC_HASH}',
        root=str(Path(gate['root'])/'science'), project_dir=gate['project_dir'],
        gate_path=str(Path(gate['root'])/'gate_spec.json'), train_minutes=1440)


def plan(gate, *, order_pending):
    return artifact('FOLLOWUP_PLAN', parents=dict(gate=gate['content_hash']),
        gate_plan=submission.render(gate), science_ceiling_plan=submission.render(ceiling_spec(gate)),
        priority_plan=priority.plan(gate) if order_pending else None,
        source_commit=gate['source_commit'], science_jobs=1, max_total_jobs=2,
        max_train_minutes=1440, poll_seconds=POLL_SECONDS, max_wait_seconds=WAIT_SECONDS,
        automatic_retry=False, quality_threshold=False, authorization=AUTHORIZATION)


def no_science(gate):
    path=Path(gate['root'])/'science'
    if path.exists() or path.is_symlink():
        raise FileExistsError('Science exists; preserve and inspect, never arm a duplicate')


def bound_gate_job(gate):
    folder=Path(gate['root'])/'submission'
    if (folder/'submission_ledger.json').exists(): return submission.accepted_jobs(gate)['preflight']
    if (folder/'live_claim').exists() or (folder/'submission_ledger_journal').exists():
        raise PermissionError('Ambiguous gate submission; inspect exact journal, do not retry')
    return None


def accounting(raw, *, job, task, user, account):
    lines=[s.strip() for s in raw.splitlines() if s.strip()]
    if not lines: return 'ACCOUNTING_PENDING'
    if len(lines)!=1: raise ValueError('Duplicate scheduler accounting rows')
    fields=lines[0].split('|')
    if len(fields)==7 and fields[-1]=='': fields.pop()
    if len(fields)!=6: raise ValueError('Malformed scheduler accounting')
    actual,name,owner,acct,state,code=fields
    if (actual,name,owner,acct)!=(job,'lukad6e_'+task,user,account):
        raise PermissionError('Exact scheduler identity differs')
    if not state: raise ValueError('Missing scheduler state')
    state=state.split()[0].rstrip('+')
    if state=='COMPLETED' and code=='0:0': return state
    if state not in ACTIVE: raise RuntimeError(f'{task} {job}: {state}, exit={code}')
    return state


def wait_job(gate, job, task, *, sleep=time.sleep, monotonic=time.monotonic):
    if not re.fullmatch('[1-9][0-9]*', job): raise ValueError('Exact numeric job ID required')
    deadline=monotonic()+WAIT_SECONDS
    while monotonic()<deadline:
        raw=subprocess.run(['sacct','-X','-n','-P','-j',job,'-o',
            'JobIDRaw,JobName%100,User%100,Account%100,State%40,ExitCode'],
            check=True,capture_output=True,text=True,timeout=60,env=submission.clean_environment()).stdout
        state=accounting(raw,job=job,task=task,user=getpass.getuser(),account=gate['execution_site']['account'])
        print(f'{task} {job}: {state}',flush=True)
        if state=='COMPLETED': return
        sleep(POLL_SECONDS)
    raise TimeoutError('14-day bound reached; no automatic retry. Submitted jobs are not cancelled.')


def review_science(gate, spec, reviewed):
    checked,_=campaign.check_spec(spec)
    if checked!=gate or reviewed!=plan(gate,order_pending=reviewed['priority_plan'] is not None):
        raise ValueError('Reviewed gate/policy changed')
    if type(spec['train_minutes']) is not int or not 1<=spec['train_minutes']<=1440:
        raise ValueError('Science exceeds authorized resource ceiling')
    normalized=dict(spec, content_hash='${MEASURED_SCIENCE_SPEC_HASH}',train_minutes=1440)
    if submission.render(normalized)!=reviewed['science_ceiling_plan']:
        raise ValueError('Science differs from exact authorized graph')
    return submission.render(spec)


def science_job(spec, exact_plan, ledger):
    folder=Path(spec['root'])/'submission'
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=spec['content_hash'],plan=exact_plan)
    expected=assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'],events=events)
    if set(jobs)!={'fit'} or ledger!=expected or load_json(folder/'submission_ledger.json')!=expected:
        raise ValueError('Science journal is incomplete or changed')
    return jobs['fit']


def run(gate, *, order_pending=False, execute=False, plan_hash=None, authorization=None,
        priority_authorization=None):
    campaign.check_gate(gate); reviewed=plan(gate,order_pending=order_pending)
    folder=Path(gate['root'])/'followup'; folder.mkdir(exist_ok=True)
    saved=folder/'plan.json'; no_science(gate); existing=bound_gate_job(gate)
    if not execute:
        submission.submit(gate)
        if order_pending: priority.inspect(reviewed['priority_plan'])
        write_immutable_json(saved,reviewed)
        return reviewed
    if plan_hash!=reviewed['content_hash'] or authorization!=AUTHORIZATION:
        raise PermissionError('Exact reviewed followup hash and authorization required')
    if order_pending and priority_authorization!=priority.AUTHORIZATION:
        raise PermissionError('Separate pending-hold authorization required')
    if not saved.is_file() or load_json(saved)!=reviewed: raise ValueError('Unchanged dry review required')
    submission.check_site(reviewed['gate_plan'])
    with controller_lock(gate):
        (folder/'live_claim').mkdir(exist_ok=False)
        intent=artifact('FOLLOWUP_AUTHORIZATION',parents=dict(plan=reviewed['content_hash']),
            host=socket.gethostname(),pid=os.getpid(),source_commit=gate['source_commit'])
        write_immutable_json(folder/'authorization.json',intent)
        try:
            if order_pending: priority.hold(gate,reviewed['priority_plan'],priority_authorization)
            if existing is None:
                submission.submit(gate,execute=True,plan_hash=reviewed['gate_plan']['content_hash'],
                                  authorization=SUBMIT_AUTHORIZATION)
            job=submission.accepted_jobs(gate)['preflight']
            if existing is not None and job!=existing: raise ValueError('Gate job binding changed')
            write_immutable_json(folder/'armed.json',artifact('FOLLOWUP_ARMED',
                parents=dict(authorization=intent['content_hash']),preflight_job=job))
            print(f'ARMED: debug preflight {job}; one fresh D066-target weight-5 fit will submit '
                  'after verified success. Do not submit science manually.',flush=True)
            wait_job(gate,job,'preflight')
            no_science(gate)
            if submission.accepted_jobs(gate)['preflight']!=job: raise ValueError('Gate journal changed')
            spec=campaign.science(gate); exact=review_science(gate,spec,reviewed)
            if submission.submit(spec)!=exact: raise ValueError('Measured exact dry review differs')
            write_immutable_json(folder/'science_review.json',artifact('FOLLOWUP_REVIEW',
                parents=dict(authorization=intent['content_hash'],spec=spec['content_hash'],plan=exact['content_hash'])))
            ledger=submission.submit(spec,execute=True,plan_hash=exact['content_hash'],authorization=SUBMIT_AUTHORIZATION)
            fit=science_job(spec,exact,ledger)
            write_immutable_json(folder/'science_queued.json',artifact('SCIENCE_QUEUED',
                parents=dict(authorization=intent['content_hash'],ledger=ledger['content_hash']),job_id=fit))
            print('SCIENCE QUEUED: '+fit+'; keeping owned holds until this fit finishes.',flush=True)
            wait_job(gate,fit,'fit')
            complete=artifact('FOLLOWUP_COMPLETE',parents=dict(authorization=intent['content_hash']),
                              preflight_job=job,fit_job=fit)
            write_immutable_json(folder/'complete.json',complete)
            return complete
        except BaseException as exc:
            write_immutable_json(folder/'failure.json',artifact('FOLLOWUP_FAILURE',
                parents=dict(authorization=intent['content_hash']),error_type=type(exc).__name__,error=str(exc),
                automatic_retry=False,instruction='Inspect logs and journals; no blind resubmission.'))
            raise
        finally:
            # Errors do not strand known holds. SIGKILL/host loss requires the
            # separate lock-protected recovery command. Never cancel any job.
            priority.release_owned(gate)


def recover_holds(gate):
    # No gate rerun or science submission, and no need to reopen deleted banks.
    campaign.validate_gate_record(gate)
    with controller_lock(gate):
        return priority.release_owned(gate)
