"""Optional owned holds on exactly the five user-selected pending debug fits.

This is queue ordering, not elevated priority or preemption. Never scancel.
Slurm has no atomic pending-only hold: recheck immediately before/after each
request and stop/release our own hold if a job starts in that narrow race.
"""
from pathlib import Path
import getpass
import re
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_embedding_sweep import debug
from hlt_classification.luka_embedding_sweep.debug_retirement import bound_rows, SWEEP_HASH, TERMINAL
from .contracts import artifact, validate
from .submission import clean_environment

AUTHORIZATION = 'HOLD_ONLY_REVIEWED_PENDING_SWEEP_JOBS_V1'
SELECTED = dict(w50='21940433', w100='21940434', w200='21940435',
                w500='21940436', w1000='21940437')


def plan(gate):
    ref = gate['reference']['spec']; validate_file_ref(ref)
    spec = load_json(ref['path'])
    debug.validate(spec, 'SPEC')
    if spec['parents']['original'] != SWEEP_HASH:
        raise PermissionError('Not the explicitly selected transferred sweep')
    command_plan = debug.render(spec)
    folder = Path(spec['root'])/'submission'
    events, jobs = load_exact_dag_journal(folder/'submission_ledger_journal',
                                         identity=spec['content_hash'], plan=command_plan)
    if (load_json(folder/'command_plan.json') != command_plan or
            load_json(folder/'submission_ledger.json') != assemble_submission_ledger(
                campaign_spec_sha256=spec['content_hash'], events=events)):
        raise ValueError('Sweep journal/ledger/plan differs')
    rows = [r for r in bound_rows(spec, command_plan, jobs, sweep=True) if r['task_id'] in SELECTED]
    if {r['task_id']:r['job_id'] for r in rows} != SELECTED or any(r['partition'] != 'debug' for r in rows):
        raise PermissionError('Only the five reviewed exact debug job IDs may be held')
    return artifact('PRIORITY_PLAN', parents=dict(gate=gate['content_hash'], sweep=spec['content_hash']),
        user=getpass.getuser(), rows=rows, authorization=AUTHORIZATION,
        reference=ref, ledger=file_ref(folder/'submission_ledger.json'),
        pending_only=True, cancel=False, preempt=False, release_after='new_fit_terminal_or_controller_failure')


def live(row, user):
    out = subprocess.run(['scontrol','show','job','-o',row['job_id']], check=True,
        capture_output=True, text=True, timeout=60, env=clean_environment()).stdout
    record = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(\S*)', out))
    expected = dict(JobId=row['job_id'], JobName=row['job_name'], Account=row['account'], QOS=row['qos'],
        Partition=row['partition'], Comment=row['comment'], WorkDir=row['workdir'],
        Command=row['command'], StdOut=row['stdout'], Requeue='0', Restarts='0')
    if (any(record.get(k) != v for k,v in expected.items()) or
            not re.fullmatch(re.escape(user)+r'\(\d+\)', record.get('UserId',''))):
        raise PermissionError('Scheduler job identity differs: '+row['job_id'])
    return record


def pending(record):
    if record.get('JobState') != 'PENDING':
        return False  # Never stop a running fit, including a review/start race.
    if (record.get('RunTime') != '00:00:00' or not record.get('Priority','').isdigit()
            or int(record['Priority']) == 0 or 'Held' in record.get('Reason','')):
        raise PermissionError('Already held/started/unknown job; do not take ownership')
    return True


def inspect(reviewed):
    for row in reviewed['rows']:
        state = live(row, reviewed['user'])
        action = 'HOLD if still pending at execution' if pending(state) else 'LEAVE UNCHANGED'
        print(row['job_id'], row['job_name'], state['JobState'], action, flush=True)


def hold(gate, reviewed, authorization):
    if authorization != AUTHORIZATION or plan(gate) != reviewed:
        raise PermissionError('Separate exact priority authorization required')
    folder = Path(gate['root'])/'followup/priority'
    folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'plan.json', reviewed)
    inspect(reviewed)  # Check every target before the first mutation.
    for row in reviewed['rows']:
        before = live(row, reviewed['user'])
        if not pending(before):
            continue
        job = row['job_id']
        write_immutable_json(folder/(job+'-hold-intent.json'), artifact('HOLD_INTENT',
            parents=dict(plan=reviewed['content_hash']), job_id=job, before=before))
        # A lost reply is ambiguous. Never automatically retry an intent.
        subprocess.run(['scontrol','hold',job], check=True, capture_output=True,
                       text=True, timeout=60, env=clean_environment())
        write_immutable_json(folder/(job+'-hold-accepted.json'), artifact('HOLD_ACCEPTED',
            parents=dict(plan=reviewed['content_hash']), job_id=job))
        after = live(row, reviewed['user'])
        if after.get('JobState') != 'PENDING' or after.get('Reason') != 'JobHeldUser' or after.get('Priority') != '0':
            raise RuntimeError('Job changed during hold; release our acknowledged hold and inspect: '+job)
        print('HELD exact pending job '+job, flush=True)


def release_owned(gate):
    """Release acknowledged holds once, preserving any ambiguous request for review."""
    folder = Path(gate['root'])/'followup/priority'
    if not folder.exists(): return []
    reviewed = load_json(folder/'plan.json'); validate(reviewed, 'PRIORITY_PLAN')
    if reviewed != plan(gate): raise PermissionError('Owned-hold plan changed')
    if reviewed['user'] != getpass.getuser(): raise PermissionError('Wrong hold owner')
    released, errors = [], []
    for row in reviewed['rows']:
        job = row['job_id']; accepted = folder/(job+'-hold-accepted.json')
        done = folder/(job+'-released.json'); intent = folder/(job+'-release-intent.json')
        if done.exists():
            value=load_json(done); validate(value,'HOLD_RELEASED',dict(plan=reviewed['content_hash']))
            if value['job_id']!=job: raise ValueError('Release job differs')
            released.append(job); continue
        if not accepted.exists():
            if (folder/(job+'-hold-intent.json')).exists(): errors.append(job+': ambiguous hold; inspect manually')
            continue
        try:
            request=load_json(folder/(job+'-hold-intent.json'))
            validate(request,'HOLD_INTENT',dict(plan=reviewed['content_hash']))
            if request['job_id']!=job or not pending(request['before']):
                raise ValueError('Missing original pending-state ownership evidence')
            value=load_json(accepted); validate(value,'HOLD_ACCEPTED',dict(plan=reviewed['content_hash']))
            if value['job_id']!=job: raise ValueError('Accepted hold job differs')
            if intent.exists(): raise RuntimeError('Ambiguous release; inspect, do not retry')
            state = live(row, reviewed['user'])
            if state.get('JobState') not in TERMINAL and state.get('Priority') == '0':
                if state.get('JobState') == 'PENDING' and state.get('Reason') != 'JobHeldUser':
                    raise PermissionError('Not our user-held state; preserve administrator hold')
                write_immutable_json(intent, artifact('RELEASE_INTENT',
                    parents=dict(plan=reviewed['content_hash']), job_id=job))
                subprocess.run(['scontrol','release',job], check=True, capture_output=True,
                               text=True, timeout=60, env=clean_environment())
            write_immutable_json(done, artifact('HOLD_RELEASED', parents=dict(plan=reviewed['content_hash']),job_id=job))
            released.append(job); print('RELEASED owned hold '+job, flush=True)
        except Exception as exc:
            errors.append(job+': '+str(exc))
    if errors: raise RuntimeError('Hold recovery needs inspection: '+'; '.join(errors))
    return released
