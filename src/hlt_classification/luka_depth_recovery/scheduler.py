"""Read-only accounting and explicitly approved, pending-only cancellation."""
from pathlib import Path
import getpass
import re
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact, validate
from . import source

ACTIVE = {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'RESIZING'}


def accounting(spec, jobs):
    if not jobs or any(not re.fullmatch('[1-9][0-9]*', j) for j in jobs.values()):
        raise ValueError('Exact numeric accepted IDs required')
    p = subprocess.run(['sacct', '-j', ','.join(jobs.values()), '-X', '-n', '-P',
        '-o', 'JobIDRaw,JobName%80,User,Account,State%40,ExitCode,ElapsedRaw,Start'],
        check=True, capture_output=True, text=True)
    rows = [s.strip().split('|')[:8] for s in p.stdout.splitlines() if s.strip()]
    out = {}
    for task, job in jobs.items():
        match = [r for r in rows if r[0] == job]
        if len(match) != 1 or len(match[0]) != 8:
            raise ValueError('Missing/ambiguous accounting for ' + job)
        r = match[0]
        if r[1:4] != ['lukasd_' + task, getpass.getuser(), spec['execution_site']['account']]:
            raise PermissionError('Original job owner/account/name differs: ' + job)
        out[task] = dict(job_id=job, state=r[4].split()[0], exit_code=r[5], elapsed=int(r[6]), start=r[7])
    return out


def live_job(spec, task, job):
    p = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True)
    f = dict(x.split('=', 1) for x in p.stdout.split() if '=' in x)
    if (f.get('JobId') != job or f.get('JobName') != 'lukasd_' + task
            or f.get('Comment') != 'lukasd:' + spec['content_hash'] + ':' + task
            or f.get('WorkDir') != spec['project_dir'] or f.get('Account') != spec['execution_site']['account']
            or f.get('Partition') != 'debug' or f.get('UserId', '').split('(')[0] != getpass.getuser()):
        raise PermissionError('Live job is not the exact original campaign allocation')
    return f


def stop_plan(parent, project, output):
    ref, spec, jobs = source.authenticate(parent, project)
    states = accounting(spec, jobs)
    value = artifact('STOP_PLAN', parents=dict(original=spec['content_hash']), original=ref,
        jobs=jobs, snapshot=states, cancel_candidates=[t['task_id'] for t in reversed(spec['tasks'])
            if states[t['task_id']]['state'] == 'PENDING'], running_jobs_preserved=True)
    write_immutable_json(output, value)
    return value


def stop(path, project, plan_hash):
    value = load_json(path); validate(value, 'STOP_PLAN')
    if value['content_hash'] != plan_hash:
        raise PermissionError('Review the exact cancellation plan hash')
    ref, spec, jobs = source.authenticate(value['original']['path'], project)
    if ref != value['original'] or jobs != value['jobs']:
        raise ValueError('Original submission changed')
    expected = [t['task_id'] for t in reversed(spec['tasks']) if value['snapshot'][t['task_id']]['state'] == 'PENDING']
    if value['cancel_candidates'] != expected:
        raise ValueError('Cancellation scope differs')
    folder = Path(path).parent/'stop_execution'; folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'intent.json', value)
    cancelled = []; preserved = []
    for task in value['cancel_candidates']:
        job = jobs[task]; state = accounting(spec, {task: job})[task]['state']
        if state != 'PENDING':
            preserved.append(task); continue
        f = live_job(spec, task, job)
        if f['JobState'] != 'PENDING':
            preserved.append(task); continue
        # Freeze before cancelling. If dispatch wins the race, never kill it.
        held = subprocess.run(['scontrol', 'hold', job], capture_output=True, text=True)
        f = live_job(spec, task, job)
        if f['JobState'] != 'PENDING':
            preserved.append(task); continue
        if held.returncode or f.get('Priority') != '0' or f.get('Reason') != 'JobHeldUser':
            raise RuntimeError('Could not safely hold ' + job + '; inspect before retrying')
        write_immutable_json(folder/(task+'-held.json'), dict(task=task, job_id=job, fields=f))
        subprocess.run(['scancel', job], check=True)
        cancelled.append(task)
    result = artifact('STOP', parents=dict(plan=value['content_hash']), cancelled=cancelled,
        preserved=preserved, original=value['original'], jobs=jobs)
    write_immutable_json(folder/'receipt.json', result)
    return result
