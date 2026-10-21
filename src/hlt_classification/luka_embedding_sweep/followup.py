"""Explicitly authorized, bounded preflight-to-science continuation on submit host.

The detached controller waits without a compute allocation. It reviews the
exact eight-command graph and resource ceilings first, then authenticates and
saves the measured exact plan before submitting science. No gate is skipped.
"""
import getpass
import json
from pathlib import Path
import re
import subprocess
import time

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from . import campaign, submission
from .contracts import artifact, AUTHORIZATION as SUBMIT_AUTHORIZATION, WEIGHTS

AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_SWEEP_FOLLOWUP_V1'
MAX_MINUTES = 1440
POLL_SECONDS = 60
WAIT_SECONDS = 14 * 86400
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
          'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD'}


def ceiling_spec(gate):
    # Deliberately NOT a scientific specification or executable plan.
    return dict(content_hash='${MEASURED_SCIENCE_SPEC_HASH}',
                root=str(Path(gate['root'])/'science'), project_dir=gate['project_dir'],
                train_minutes={arm: MAX_MINUTES for arm in WEIGHTS})


def plan(gate):
    return artifact('FOLLOWUP_PLAN', parents=dict(gate=gate['content_hash']),
        gate_plan=submission.render(gate),
        science_ceiling_plan=submission.render(ceiling_spec(gate), gate),
        source_commit=gate['source_commit'], weights=WEIGHTS, fits=len(WEIGHTS),
        science_jobs=len(WEIGHTS), max_total_jobs=1+len(WEIGHTS),
        max_train_minutes=MAX_MINUTES, poll_seconds=POLL_SECONDS, max_wait_seconds=WAIT_SECONDS,
        requires=['exact gate journal', 'COMPLETED/0:0', 'authenticated technical receipt',
                  'unchanged source and graph', 'measured resource admission',
                  'saved measured exact-plan dry review'],
        automatic_retry=False, quality_threshold=False, authorization=AUTHORIZATION)


def no_science(gate):
    # Never take over a manual submitter or interrupted science creation.
    root = Path(gate['root'])/'science'
    if root.exists() or root.is_symlink():
        raise FileExistsError('Science already exists; preserve and inspect, do not arm another submitter')


def bound_gate_job(gate):
    root = Path(gate['root'])/'submission'
    if (root/'submission_ledger.json').exists():
        return submission.accepted_jobs(gate)['preflight']
    if (root/'live_claim').exists() or (root/'submission_ledger_journal').exists():
        raise PermissionError('Incomplete/ambiguous gate submission; inspect exact journals, do not retry')
    return None


def accounting(raw, *, job, user, account):
    """Missing accounting means wait; unknown/terminal states are never success."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return 'ACCOUNTING_PENDING'
    if len(lines) != 1:
        raise ValueError('Unexpected/duplicate preflight accounting rows')
    fields = lines[0].split('|')
    if len(fields) != 6:
        raise ValueError('Malformed preflight accounting row')
    actual_job, name, owner, actual_account, state, code = fields
    if actual_job != job or name != 'lukaes_preflight' or owner != user or actual_account != account:
        raise PermissionError('Preflight scheduler identity differs from authorized gate')
    if not state.strip():
        raise ValueError('Missing preflight state')
    state = state.split()[0].rstrip('+')
    if state == 'COMPLETED' and code == '0:0':
        return state
    if state not in ACTIVE:
        raise RuntimeError(f'Preflight {job}: {state}, exit={code}; no science submission')
    return state


def wait_gate(gate, job, *, sleep=time.sleep, monotonic=time.monotonic):
    if not re.fullmatch(r'[1-9][0-9]*', job):
        raise ValueError('Expected exact numeric preflight ID')
    deadline = monotonic()+WAIT_SECONDS
    user = getpass.getuser()
    while monotonic() < deadline:
        raw = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
            '-o', 'JobIDRaw,JobName%100,User%100,Account%100,State%40,ExitCode'],
            check=True, capture_output=True, text=True, timeout=60,
            env=submission.clean_environment()).stdout
        state = accounting(raw, job=job, user=user, account=gate['execution_site']['account'])
        print(f'Preflight {job}: {state}; science not submitted yet.', flush=True)
        if state == 'COMPLETED':
            return
        sleep(POLL_SECONDS)
    raise TimeoutError('14-day wait bound reached; no science submitted; existing jobs unchanged')


def review_science(gate, spec, reviewed):
    """Only measured times and the measured spec hash may differ from the ceiling."""
    checked, _ = campaign.check_spec(spec)
    if checked != gate or reviewed != plan(gate):
        raise ValueError('Follow-up gate/policy changed')
    if set(spec['train_minutes']) != set(WEIGHTS):
        raise ValueError('Measured arms differ from authorized eight weights')
    for value in spec['train_minutes'].values():
        if type(value) is not int or not 1 <= value <= MAX_MINUTES:
            raise ValueError('Measured request exceeds authorized tier3 bounds')
    normalized = dict(spec, content_hash='${MEASURED_SCIENCE_SPEC_HASH}',
                      train_minutes={arm: MAX_MINUTES for arm in WEIGHTS})
    if submission.render(normalized, gate) != reviewed['science_ceiling_plan']:
        raise ValueError('Measured science graph/commands exceed reviewed scope')
    return submission.render(spec, gate)


def science_jobs(spec, exact_plan, ledger):
    directory = Path(spec['root'])/'submission'
    events, jobs = load_exact_dag_journal(directory/'submission_ledger_journal',
                                         identity=spec['content_hash'], plan=exact_plan)
    expected = assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'], events=events)
    if (set(jobs) != set(WEIGHTS) or len(set(jobs.values())) != exact_plan['jobs']
            or ledger != expected or load_json(directory/'submission_ledger.json') != expected):
        raise ValueError('Science live ledger/journal incomplete or different')
    return jobs


def run(gate, *, execute=False, plan_hash=None, authorization=None):
    campaign.check_gate(gate)
    reviewed = plan(gate)
    directory = Path(gate['root'])/'followup'
    saved = directory/'plan.json'
    no_science(gate)
    existing = bound_gate_job(gate)
    if not execute:
        submission.submit(gate)  # Canonical gate dry run and real site's test-only check.
        directory.mkdir(exist_ok=True)
        write_immutable_json(saved, reviewed)
        return reviewed
    if plan_hash != reviewed['content_hash'] or authorization != AUTHORIZATION:
        raise PermissionError('Exact reviewed FOLLOWUP plan hash and explicit authorization required')
    if not saved.is_file() or load_json(saved) != reviewed:
        raise ValueError('Saved unchanged follow-up dry review required')
    submission.check_site(reviewed['gate_plan'])
    # Permanent atomic claim. A crash, partial submission, or duplicate launch
    # requires inspection; never silently resume/requeue after ambiguous sbatch.
    (directory/'live_claim').mkdir(exist_ok=False)
    intent = artifact('FOLLOWUP_AUTHORIZATION', parents=dict(plan=reviewed['content_hash']),
                      source_commit=gate['source_commit'], gate=gate['content_hash'])
    write_immutable_json(directory/'authorization.json', intent)
    try:
        if existing is None:
            submission.submit(gate, execute=True, plan_hash=reviewed['gate_plan']['content_hash'],
                              authorization=SUBMIT_AUTHORIZATION)
        job = submission.accepted_jobs(gate)['preflight']
        if existing is not None and job != existing:
            raise ValueError('Preflight binding changed')
        ledger = load_json(Path(gate['root'])/'submission/submission_ledger.json')
        write_immutable_json(directory/'armed.json', artifact('FOLLOWUP_ARMED', parents=dict(
            authorization=intent['content_hash'], gate_ledger=ledger['content_hash']), preflight_job=job))
        print(f'ARMED: preflight {job}; after verified success, all {reviewed["science_jobs"]} '
              'tier3 science jobs will submit automatically. Do not also submit science manually.', flush=True)
        wait_gate(gate, job)
        no_science(gate)
        if submission.accepted_jobs(gate)['preflight'] != job:
            raise ValueError('Gate journal changed while waiting')
        # Recheck source, technical payloads, resource admission and accounting,
        # not merely Slurm's exit code. No validation-quality threshold.
        spec = campaign.science(gate)
        exact_plan = review_science(gate, spec, reviewed)
        dry = submission.submit(spec)
        if dry != exact_plan:
            raise ValueError('Measured exact dry plan changed')
        write_immutable_json(directory/'review.json', artifact('FOLLOWUP_REVIEW', parents=dict(
            authorization=intent['content_hash'], spec=spec['content_hash'], plan=exact_plan['content_hash'])))
        result = submission.submit(spec, execute=True, plan_hash=exact_plan['content_hash'],
                                   authorization=SUBMIT_AUTHORIZATION)
        jobs = science_jobs(spec, exact_plan, result)
        complete = artifact('FOLLOWUP_COMPLETE', parents=dict(authorization=intent['content_hash'],
            plan=exact_plan['content_hash'], ledger=result['content_hash']), jobs=jobs, training_complete=False)
        write_immutable_json(directory/'complete.json', complete)
        print('ALL SCIENCE JOBS QUEUED: '+json.dumps(jobs, sort_keys=True), flush=True)
        return complete
    except BaseException as exc:
        write_immutable_json(directory/'failure.json', artifact('FOLLOWUP_FAILURE',
            parents=dict(authorization=intent['content_hash']), error_type=type(exc).__name__, error=str(exc),
            automatic_retry=False, instruction='Preserve logs and journals; do not rerun or submit manually.'))
        print('FOLLOWUP STOPPED: inspect failure.json and submission journals; no automatic retry.', flush=True)
        raise
